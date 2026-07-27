"""Manifest-driven SAM3.1 native Multiplex video Public API."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
from pathlib import Path
import secrets
from typing import Any, Protocol, Sequence

import numpy as np
from PIL import Image

from sam3.runtime.base_video import (
    ObjectStateError,
    PreviewHandle,
    PreviewHandleError,
    VideoFramePrediction,
    VideoHandle,
    VideoPrediction,
    VideoPreview,
    VideoStateError,
    _normalize_video_frame,
)
from sam3.runtime.image_pcs import SessionClosedError, _hash_parts
from sam3.runtime.interactive_image import (
    InteractivePredictOptions,
    InteractivePrompt,
    _preprocess_interactive_image,
    _prompt_arrays,
    _resize_and_threshold,
    _validate_options,
)
from sam3.runtime.manifest import (
    MULTIPLEX_VIDEO_PLAN_IDS,
    SUPPORTED_PLAN_IDS,
    ManifestError,
    PlanNotFoundError,
    ResolvedPlan,
    resolve_plan,
)
from sam3.runtime.multiplex_state import (
    MULTIPLEX_STATE_ABI,
    MultiplexCapacityError,
    MultiplexStateV1,
    MultiplexVariantParameters,
    SlotAssignment,
)

LOGGER = logging.getLogger(__name__)
MULTIPLEX_FRAME_CONDITION_ID = "sam3-1-tri-interactive-propagation-v1"


@dataclass(frozen=True)
class _MultiplexBackendPreview:
    low_res_logits: np.ndarray
    scores: np.ndarray
    commit_mask: object
    object_pointer: object
    object_score: object


@dataclass(frozen=True)
class _MultiplexBackendFrame:
    low_res_logits: np.ndarray
    scores: np.ndarray


class _MultiplexVideoBackendAdapter(Protocol):
    counters: dict[str, int]

    def reset_state(self) -> None: ...

    def add_slot(self, assignment: SlotAssignment) -> None: ...

    def remove_slot(self, assignment: SlotAssignment) -> None: ...

    def encode_frame(self, values: np.ndarray) -> object: ...

    def preview(
        self,
        frame_cache: object,
        assignment: SlotAssignment,
        prompt_inputs: dict[str, np.ndarray],
        *,
        multimask: bool,
    ) -> _MultiplexBackendPreview: ...

    def commit(
        self,
        frame_cache: object,
        assignment: SlotAssignment,
        preview: _MultiplexBackendPreview,
        *,
        frame_index: int,
    ) -> None: ...

    def propagate(
        self,
        frame_cache: object,
        assignments: np.ndarray,
        *,
        frame_index: int,
        reverse: bool,
    ) -> _MultiplexBackendFrame: ...

    def close(self) -> None: ...


@dataclass
class _PreviewRecord:
    object_id: int
    frame_index: int
    assignment_revision: int
    mutation_revision: int
    frame_cache: object
    backend_preview: _MultiplexBackendPreview
    options: InteractivePredictOptions
    used: bool = False


def _validate_prompt_range(
    prompt: InteractivePrompt, frame_size: tuple[int, int]
) -> None:
    height, width = frame_size
    if prompt.points_xy is not None:
        points = np.asarray(prompt.points_xy)
        if points.ndim == 2 and points.shape[1] == 2 and np.all(np.isfinite(points)):
            if np.any(points[:, 0] < 0) or np.any(points[:, 0] >= width):
                raise ValueError("point x coordinate is outside the video frame")
            if np.any(points[:, 1] < 0) or np.any(points[:, 1] >= height):
                raise ValueError("point y coordinate is outside the video frame")
    if prompt.box_xyxy is not None:
        box = np.asarray(prompt.box_xyxy)
        if box.shape == (4,) and np.all(np.isfinite(box)):
            if (
                box[0] < 0
                or box[1] < 0
                or box[2] > width
                or box[3] > height
            ):
                raise ValueError("box_xyxy is outside the video frame")


class MultiplexVideoSession:
    """Public object-ID session over private device-resident bucket state."""

    def __init__(
        self, plan: ResolvedPlan, adapter: _MultiplexVideoBackendAdapter
    ) -> None:
        self._plan = plan
        self._adapter = adapter
        self._variant = MultiplexVariantParameters.from_manifest(plan.manifest)
        self._state = MultiplexStateV1()
        self._session_token = secrets.token_hex(16)
        self._video_handle: VideoHandle | None = None
        self._video_token: str | None = None
        self._frames: tuple[np.ndarray, ...] = ()
        self._frame_cache: dict[int, object] = {}
        self._frame_cache_keys: dict[int, str] = {}
        self._previews: dict[str, _PreviewRecord] = {}
        self._used_preview_tokens: set[str] = set()
        self._conditioned_objects: set[int] = set()
        self._mutation_revision = 0
        self._closed = False

    @property
    def plan_id(self) -> str:
        return self._plan.plan_id

    def _ensure_open(self) -> None:
        if self._closed:
            raise SessionClosedError("multiplex video session is closed")

    def _require_video(self) -> VideoHandle:
        if self._video_handle is None:
            raise VideoStateError("set_video must be called before this operation")
        return self._video_handle

    def set_video(self, frames: Sequence[Image.Image | np.ndarray]) -> VideoHandle:
        self._ensure_open()
        if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
            raise TypeError("frames must be a non-empty sequence")
        if len(frames) == 0:
            raise ValueError("frames must be non-empty")
        normalized = tuple(_normalize_video_frame(frame) for frame in frames)
        shape = normalized[0].shape
        if any(frame.shape != shape for frame in normalized):
            raise ValueError("all video frames must have identical RGB dimensions")
        digest = hashlib.sha256()
        for frame in normalized:
            digest.update(frame.tobytes())
        token = digest.hexdigest()
        handle = VideoHandle(
            self._session_token,
            token,
            frame_count=len(normalized),
            frame_size=(shape[0], shape[1]),
        )
        self._frames = normalized
        self._video_token = token
        self._video_handle = handle
        self._frame_cache.clear()
        self._frame_cache_keys.clear()
        self._previews.clear()
        self._used_preview_tokens.clear()
        self._conditioned_objects.clear()
        self._mutation_revision = 0
        self._state = MultiplexStateV1()
        self._adapter.reset_state()
        return handle

    def add_object(self, object_id: int) -> None:
        self._ensure_open()
        self._require_video()
        try:
            assignment = self._state.add_object(object_id)
        except (TypeError, ValueError, MultiplexCapacityError) as exc:
            raise ObjectStateError(str(exc)) from exc
        self._adapter.add_slot(assignment)

    def remove_object(self, object_id: int) -> None:
        self._ensure_open()
        self._require_video()
        try:
            assignment = self._state.remove_object(object_id)
        except KeyError as exc:
            raise ObjectStateError(str(exc)) from exc
        self._conditioned_objects.discard(object_id)
        self._adapter.remove_slot(assignment)

    def _require_object(self, object_id: int) -> SlotAssignment:
        try:
            return self._state.require_object(object_id)
        except KeyError as exc:
            raise ObjectStateError(str(exc)) from exc

    def _check_frame_index(self, frame_index: int) -> None:
        handle = self._require_video()
        if not isinstance(frame_index, int) or isinstance(frame_index, bool):
            raise TypeError("frame_index must be an integer")
        if not 0 <= frame_index < handle.frame_count:
            raise IndexError(
                f"frame_index={frame_index} is outside [0,{handle.frame_count})"
            )

    def _get_frame_cache(self, frame_index: int) -> object:
        cached = self._frame_cache.get(frame_index)
        if cached is not None:
            return cached
        values, original_size = _preprocess_interactive_image(
            self._frames[frame_index]
        )
        checkpoint = self._plan.manifest["model"]["checkpoint"]["digest"]["value"]
        cache = next(
            item
            for item in self._plan.manifest["caches"]
            if item["id"] == "multiplex-frame-cache"
        )
        cache_key = _hash_parts(
            values.tobytes(),
            str(self._video_token).encode(),
            str(frame_index).encode(),
            np.asarray(original_size, dtype=np.int64).tobytes(),
            str(checkpoint).encode(),
            self._plan.profile_id.encode(),
            MULTIPLEX_FRAME_CONDITION_ID.encode(),
            str(cache["key_version"]).encode(),
        )
        encoded = self._adapter.encode_frame(values)
        self._frame_cache[frame_index] = encoded
        self._frame_cache_keys[frame_index] = cache_key
        return encoded

    def preview(
        self,
        object_id: int,
        frame_index: int,
        prompt: InteractivePrompt = InteractivePrompt(),
        options: InteractivePredictOptions = InteractivePredictOptions(),
    ) -> VideoPreview:
        self._ensure_open()
        assignment = self._require_object(object_id)
        self._check_frame_index(frame_index)
        handle = self._require_video()
        output_size = _validate_options(options, handle.frame_size)
        _validate_prompt_range(prompt, handle.frame_size)
        prompt_inputs, prompt_facts = _prompt_arrays(prompt, handle.frame_size)
        frame_cache = self._get_frame_cache(frame_index)
        backend = self._adapter.preview(
            frame_cache,
            assignment,
            prompt_inputs,
            multimask=options.multimask_output,
        )
        expected = 3 if options.multimask_output else 1
        if backend.low_res_logits.shape != (expected, 288, 288):
            raise RuntimeError(
                "multiplex preview logits shape mismatch: "
                f"{backend.low_res_logits.shape}"
            )
        if backend.scores.shape != (expected,):
            raise RuntimeError(
                f"multiplex preview score shape mismatch: {backend.scores.shape}"
            )
        preview_handle: PreviewHandle | None = None
        if not options.multimask_output:
            token = secrets.token_hex(16)
            preview_handle = PreviewHandle(self._session_token, token)
            self._previews[token] = _PreviewRecord(
                object_id=object_id,
                frame_index=frame_index,
                assignment_revision=self._state.revision,
                mutation_revision=self._mutation_revision,
                frame_cache=frame_cache,
                backend_preview=backend,
                options=options,
            )
        metadata: dict[str, object] = {
            "plan_id": self._plan.plan_id,
            "contract_version": self._plan.contract_version,
            "profile_id": self._plan.profile_id,
            "state_abi": MULTIPLEX_STATE_ABI,
            "assignment_revision": self._state.revision,
            "object_id": object_id,
            "frame_index": frame_index,
            "frame_cache_key": self._frame_cache_keys[frame_index],
            "multimask_artifact": (
                "multimask3" if options.multimask_output else "single1"
            ),
            **prompt_facts,
        }
        logits = backend.low_res_logits.astype(np.float32, copy=False)
        return VideoPreview(
            masks=_resize_and_threshold(logits, output_size, options.mask_threshold),
            scores=backend.scores.astype(np.float32, copy=False),
            low_res_logits=logits,
            preview_handle=preview_handle,
            metadata=metadata,
        )

    def _resolve_preview(self, preview_handle: PreviewHandle) -> _PreviewRecord:
        if not isinstance(preview_handle, PreviewHandle):
            raise PreviewHandleError("commit requires a PreviewHandle")
        if preview_handle._session_token != self._session_token:
            raise PreviewHandleError("preview handle belongs to another session")
        if preview_handle._token in self._used_preview_tokens:
            raise PreviewHandleError("preview handle was already committed")
        record = self._previews.get(preview_handle._token)
        if record is None:
            raise PreviewHandleError("unknown or multimask preview handle")
        if record.assignment_revision != self._state.revision:
            raise PreviewHandleError("preview handle is stale after assignment update")
        if record.mutation_revision != self._mutation_revision:
            raise PreviewHandleError("preview handle is stale after state mutation")
        return record

    def commit(self, preview_handle: PreviewHandle) -> VideoPrediction:
        self._ensure_open()
        record = self._resolve_preview(preview_handle)
        assignment = self._require_object(record.object_id)
        self._adapter.commit(
            record.frame_cache,
            assignment,
            record.backend_preview,
            frame_index=record.frame_index,
        )
        record.used = True
        self._used_preview_tokens.add(preview_handle._token)
        self._previews.pop(preview_handle._token, None)
        self._conditioned_objects.add(record.object_id)
        self._mutation_revision += 1
        logits = record.backend_preview.low_res_logits[0].astype(
            np.float32, copy=False
        )
        score = float(record.backend_preview.scores[0])
        handle = self._require_video()
        mask = _resize_and_threshold(
            logits[None], handle.frame_size, record.options.mask_threshold
        )[0]
        return VideoPrediction(
            object_id=record.object_id,
            frame_index=record.frame_index,
            mask=mask,
            score=score,
            low_res_logits=logits,
            metadata={
                "plan_id": self._plan.plan_id,
                "profile_id": self._plan.profile_id,
                "state_abi": MULTIPLEX_STATE_ABI,
                "assignment_revision": self._state.revision,
                "conditioning": True,
            },
        )

    def propagate(
        self,
        *,
        start_frame: int,
        end_frame: int,
        reverse: bool = False,
    ) -> list[VideoFramePrediction]:
        self._ensure_open()
        handle = self._require_video()
        self._check_frame_index(start_frame)
        self._check_frame_index(end_frame)
        if not isinstance(reverse, bool):
            raise TypeError("reverse must be bool")
        if (not reverse and start_frame > end_frame) or (
            reverse and start_frame < end_frame
        ):
            raise ValueError("frame range does not match propagation direction")
        object_ids = self._state.object_ids
        if not object_ids:
            raise ObjectStateError("propagate requires at least one object")
        missing = sorted(set(object_ids) - self._conditioned_objects)
        if missing:
            raise ObjectStateError(
                f"objects require a committed conditioning frame: {missing}"
            )
        assignments = self._state.assignment_array(object_ids)
        step = -1 if reverse else 1
        results: list[VideoFramePrediction] = []
        for frame_index in range(start_frame, end_frame + step, step):
            backend = self._adapter.propagate(
                self._get_frame_cache(frame_index),
                assignments,
                frame_index=frame_index,
                reverse=reverse,
            )
            if backend.low_res_logits.shape != (len(object_ids), 288, 288):
                raise RuntimeError(
                    "multiplex propagation logits shape mismatch: "
                    f"{backend.low_res_logits.shape}"
                )
            if backend.scores.shape != (len(object_ids),):
                raise RuntimeError(
                    "multiplex propagation score shape mismatch: "
                    f"{backend.scores.shape}"
                )
            masks = _resize_and_threshold(
                backend.low_res_logits, handle.frame_size, 0.0
            )
            self._mutation_revision += 1
            results.append(
                VideoFramePrediction(
                    frame_index=frame_index,
                    object_ids=np.asarray(object_ids, dtype=np.int64),
                    masks=masks,
                    scores=backend.scores.astype(np.float32, copy=False),
                    metadata={
                        "plan_id": self._plan.plan_id,
                        "profile_id": self._plan.profile_id,
                        "state_abi": MULTIPLEX_STATE_ABI,
                        "assignment_revision": self._state.revision,
                        "reverse": reverse,
                    },
                )
            )
        return results

    def close(self) -> None:
        self._ensure_open()
        self._adapter.close()
        self._frames = ()
        self._frame_cache.clear()
        self._frame_cache_keys.clear()
        self._previews.clear()
        self._used_preview_tokens.clear()
        self._conditioned_objects.clear()
        self._closed = True


_multiplex_video_adapter_factory: Any = None


def create_multiplex_video_session(
    plan_id: str, *, bundle_dir: str | Path
) -> MultiplexVideoSession:
    """Validate and create the shipped SAM3.1 Multiplex video session."""

    if plan_id not in MULTIPLEX_VIDEO_PLAN_IDS:
        if plan_id in SUPPORTED_PLAN_IDS:
            raise ManifestError(
                f"plan scope mismatch: {plan_id} is not a multiplex video plan"
            )
        raise PlanNotFoundError(f"unknown multiplex video plan: {plan_id}")
    resolved = resolve_plan(bundle_dir, plan_id)
    if resolved.manifest["scope"]["use_case"] != "multiplex-video-tracking":
        raise ManifestError(
            f"plan scope mismatch: {plan_id} is not multiplex-video-tracking"
        )
    factory = _multiplex_video_adapter_factory
    if factory is None:
        from sam3.runtime.multiplex_video_ort import OrtCudaMultiplexVideoAdapter

        factory = OrtCudaMultiplexVideoAdapter
    adapter = factory(resolved)
    LOGGER.info(
        "created multiplex video session plan_id=%s manifest_sha256=%s "
        "contract=%s profile=%s scope=%s roles=%s",
        resolved.plan_id,
        resolved.manifest_digest,
        resolved.contract_version,
        resolved.profile_id,
        resolved.manifest["scope"]["scope_label"],
        sorted(resolved.artifacts_by_role),
    )
    return MultiplexVideoSession(resolved, adapter)


__all__ = [
    "MultiplexVideoSession",
    "_MultiplexBackendFrame",
    "_MultiplexBackendPreview",
    "create_multiplex_video_session",
]
