"""Manifest-driven M4 Public API for SAM3 base video tracking."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
from pathlib import Path
import secrets
from typing import Any, Protocol, Sequence

import numpy as np
from PIL import Image

from sam3.runtime.base_video_state import (
    BaseVideoStateV1,
    BaseVideoVariantParameters,
    PackedObjectState,
    StateCapacityError,
    VideoStateEntry,
)
from sam3.runtime.image_pcs import SessionClosedError, SessionStateError, _hash_parts
from sam3.runtime.interactive_image import (
    InteractivePredictOptions,
    InteractivePrompt,
    _preprocess_interactive_image,
    _prompt_arrays,
    _resize_and_threshold,
    _validate_options,
)
from sam3.runtime.manifest import (
    BASE_VIDEO_PLAN_IDS,
    SUPPORTED_PLAN_IDS,
    ManifestError,
    PlanNotFoundError,
    ResolvedPlan,
    resolve_plan,
)

LOGGER = logging.getLogger(__name__)
MEMORY_AWARE_CONDITION_ID = "memory-aware-frame-view-v1"


class VideoStateError(SessionStateError):
    """The video/session lifecycle does not permit the requested operation."""


class ObjectStateError(SessionStateError):
    """An object ID is duplicate, unknown, or not ready for propagation."""


class PreviewHandleError(SessionStateError):
    """A preview handle is foreign, stale, non-committable, or already used."""


class VideoHandle:
    """Session-bound opaque handle for one validated in-memory video."""

    __slots__ = ("_session_token", "_token", "frame_count", "frame_size")

    def __init__(
        self,
        session_token: str,
        token: str,
        *,
        frame_count: int,
        frame_size: tuple[int, int],
    ) -> None:
        self._session_token = session_token
        self._token = token
        self.frame_count = frame_count
        self.frame_size = frame_size

    def __repr__(self) -> str:
        return (
            f"VideoHandle(frame_count={self.frame_count}, frame_size={self.frame_size})"
        )


class PreviewHandle:
    """Session-bound opaque handle for one single-mask preview revision."""

    __slots__ = ("_session_token", "_token")

    def __init__(self, session_token: str, token: str) -> None:
        self._session_token = session_token
        self._token = token

    def __repr__(self) -> str:
        return "PreviewHandle()"


@dataclass(frozen=True)
class VideoPreview:
    masks: np.ndarray
    scores: np.ndarray
    low_res_logits: np.ndarray
    preview_handle: PreviewHandle | None
    metadata: dict[str, object]


@dataclass(frozen=True)
class VideoPrediction:
    object_id: int
    frame_index: int
    mask: np.ndarray
    score: float
    low_res_logits: np.ndarray
    metadata: dict[str, object]


@dataclass(frozen=True)
class VideoFramePrediction:
    frame_index: int
    object_ids: np.ndarray
    masks: np.ndarray
    scores: np.ndarray
    metadata: dict[str, object]


@dataclass(frozen=True)
class _BackendPreviewBatch:
    low_res_logits: tuple[np.ndarray | None, ...]
    scores: tuple[np.ndarray | None, ...]
    commit_masks: object
    object_pointers: object
    object_scores: object


@dataclass(frozen=True)
class _BackendCommit:
    memory_features: object
    memory_position: object
    object_pointer: object
    high_res_logits: np.ndarray
    object_score: float


class _BaseVideoBackendAdapter(Protocol):
    counters: dict[str, int]
    batch_capacity: int
    fused_default: bool

    def encode_frame(self, values: np.ndarray) -> object: ...

    def preview(
        self,
        frame_cache: object,
        packed: PackedObjectState,
        prompt_inputs: dict[str, np.ndarray],
        *,
        multimask: bool,
    ) -> _BackendPreviewBatch: ...

    def commit(
        self,
        frame_cache: object,
        packed: PackedObjectState,
        preview: _BackendPreviewBatch,
        *,
        is_mask_from_points: bool,
    ) -> tuple[_BackendCommit | None, ...]: ...

    def step_and_commit(
        self,
        frame_cache: object,
        packed: PackedObjectState,
        prompt_inputs: dict[str, np.ndarray],
    ) -> tuple[_BackendPreviewBatch, tuple[_BackendCommit | None, ...]]: ...

    def close(self) -> None: ...


@dataclass
class _PreviewRecord:
    object_id: int
    frame_index: int
    revision: int
    frame_cache: object
    packed: PackedObjectState
    backend_preview: _BackendPreviewBatch
    options: InteractivePredictOptions
    used: bool = False


def _normalize_video_frame(frame: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(frame, Image.Image):
        return np.asarray(frame.convert("RGB"), dtype=np.uint8)
    if not isinstance(frame, np.ndarray):
        raise TypeError("video frames must be PIL images or uint8 NumPy arrays")
    if frame.dtype != np.uint8:
        raise ValueError("NumPy video frames must use uint8 pixels")
    if frame.ndim != 3 or frame.shape[2] not in (3, 4):
        raise ValueError("NumPy video frames must have shape [H,W,3] or [H,W,4]")
    if frame.shape[2] == 4:
        return np.asarray(Image.fromarray(frame).convert("RGB"), dtype=np.uint8)
    return np.ascontiguousarray(frame)


class BaseVideoSession:
    """M4 per-object base-video session; backend state remains private."""

    def __init__(self, plan: ResolvedPlan, adapter: _BaseVideoBackendAdapter) -> None:
        self._plan = plan
        self._adapter = adapter
        self._variant = BaseVideoVariantParameters.from_manifest(plan.manifest)
        self._state = BaseVideoStateV1(batch_capacity=adapter.batch_capacity)
        self._session_token = secrets.token_hex(16)
        self._video_handle: VideoHandle | None = None
        self._video_token: str | None = None
        self._frames: tuple[np.ndarray, ...] = ()
        self._frame_cache: dict[int, object] = {}
        self._frame_cache_keys: dict[int, str] = {}
        self._previews: dict[str, _PreviewRecord] = {}
        self._closed = False

    @property
    def plan_id(self) -> str:
        return self._plan.plan_id

    def _ensure_open(self) -> None:
        if self._closed:
            raise SessionClosedError("base video session is closed")

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
        self._state = BaseVideoStateV1(batch_capacity=self._adapter.batch_capacity)
        return handle

    def add_object(self, object_id: int) -> None:
        self._ensure_open()
        self._require_video()
        try:
            self._state.add_object(object_id)
        except (TypeError, ValueError) as exc:
            raise ObjectStateError(str(exc)) from exc

    def _require_object(self, object_id: int) -> None:
        try:
            self._state.require_object(object_id)
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
        frame = self._frames[frame_index]
        values, original_size = _preprocess_interactive_image(frame)
        checkpoint = self._plan.manifest["model"]["checkpoint"]["digest"]["value"]
        cache = next(
            item
            for item in self._plan.manifest["caches"]
            if item["id"] == "frame-cache"
        )
        cache_key = _hash_parts(
            values.tobytes(),
            str(self._video_token).encode(),
            str(frame_index).encode(),
            np.asarray(original_size, dtype=np.int64).tobytes(),
            str(checkpoint).encode(),
            self._plan.profile_id.encode(),
            MEMORY_AWARE_CONDITION_ID.encode(),
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
        self._require_object(object_id)
        self._check_frame_index(frame_index)
        handle = self._require_video()
        output_size = _validate_options(options, handle.frame_size)
        prompt_inputs, prompt_facts = _prompt_arrays(prompt, handle.frame_size)
        packed = self._state.pack(
            [object_id],
            frame_index=frame_index,
            reverse=False,
            video_frame_count=handle.frame_count,
        )[0]
        revision = self._state.revision
        frame_cache = self._get_frame_cache(frame_index)
        backend = self._adapter.preview(
            frame_cache,
            packed,
            prompt_inputs,
            multimask=options.multimask_output,
        )
        logits = backend.low_res_logits[0]
        scores = backend.scores[0]
        if logits is None or scores is None:
            raise RuntimeError("backend omitted the active preview output")
        expected = 3 if options.multimask_output else 1
        if logits.shape != (expected, 288, 288) or scores.shape != (expected,):
            raise RuntimeError(
                f"base video preview shape mismatch: {logits.shape}, {scores.shape}"
            )
        preview_handle: PreviewHandle | None = None
        if not options.multimask_output:
            token = secrets.token_hex(16)
            preview_handle = PreviewHandle(self._session_token, token)
            self._previews[token] = _PreviewRecord(
                object_id=object_id,
                frame_index=frame_index,
                revision=revision,
                frame_cache=frame_cache,
                packed=packed,
                backend_preview=backend,
                options=options,
            )
        metadata: dict[str, object] = {
            "plan_id": self._plan.plan_id,
            "contract_version": self._plan.contract_version,
            "profile_id": self._plan.profile_id,
            "state_abi": "BaseVideoStateV1",
            "state_revision": revision,
            "object_id": object_id,
            "frame_index": frame_index,
            "frame_cache_key": self._frame_cache_keys[frame_index],
            "multimask_artifact": (
                "multimask3" if options.multimask_output else "single1"
            ),
            **prompt_facts,
        }
        return VideoPreview(
            masks=_resize_and_threshold(logits, output_size, options.mask_threshold),
            scores=scores.astype(np.float32, copy=False),
            low_res_logits=logits.astype(np.float32, copy=False),
            preview_handle=preview_handle,
            metadata=metadata,
        )

    def _resolve_preview(self, preview_handle: PreviewHandle) -> _PreviewRecord:
        if not isinstance(preview_handle, PreviewHandle):
            raise PreviewHandleError("commit requires a PreviewHandle")
        if preview_handle._session_token != self._session_token:
            raise PreviewHandleError("preview handle belongs to another session")
        record = self._previews.get(preview_handle._token)
        if record is None:
            raise PreviewHandleError("unknown or multimask preview handle")
        if record.used:
            raise PreviewHandleError("preview handle was already committed")
        if record.revision != self._state.revision:
            raise PreviewHandleError("preview handle is stale for this state revision")
        return record

    def _capacity_check(self, object_id: int, frame_index: int) -> None:
        state = self._state.require_object(object_id)
        if frame_index not in state.conditioning and len(state.conditioning) >= 4:
            raise StateCapacityError(f"object {object_id} conditioning capacity is 4")

    def _entry_from_commit(
        self,
        object_id: int,
        frame_index: int,
        backend_preview: _BackendPreviewBatch,
        committed: _BackendCommit,
        *,
        conditioning: bool,
    ) -> VideoStateEntry:
        logits = backend_preview.low_res_logits[0]
        scores = backend_preview.scores[0]
        if logits is None or scores is None:
            raise RuntimeError("backend omitted committed preview outputs")
        best = int(np.argmax(scores))
        return VideoStateEntry(
            frame_index=frame_index,
            conditioning=conditioning,
            memory_features=committed.memory_features,
            memory_position=committed.memory_position,
            object_pointer=committed.object_pointer,
            mask=(committed.high_res_logits[0] > 0),
            score=float(scores[best]),
            low_res_logits=logits[best].astype(np.float32, copy=False),
            object_score=committed.object_score,
        )

    def _public_prediction(
        self, object_id: int, entry: VideoStateEntry
    ) -> VideoPrediction:
        return VideoPrediction(
            object_id=object_id,
            frame_index=entry.frame_index,
            mask=entry.mask.copy(),
            score=entry.score,
            low_res_logits=entry.low_res_logits.copy(),
            metadata={
                "plan_id": self._plan.plan_id,
                "profile_id": self._plan.profile_id,
                "state_abi": "BaseVideoStateV1",
                "state_revision": self._state.revision,
                "conditioning": entry.conditioning,
            },
        )

    def commit(self, preview_handle: PreviewHandle) -> VideoPrediction:
        self._ensure_open()
        record = self._resolve_preview(preview_handle)
        self._capacity_check(record.object_id, record.frame_index)
        committed = self._adapter.commit(
            record.frame_cache,
            record.packed,
            record.backend_preview,
            is_mask_from_points=True,
        )[0]
        if committed is None:
            raise RuntimeError("backend omitted the active memory commit")
        entry = self._entry_from_commit(
            record.object_id,
            record.frame_index,
            record.backend_preview,
            committed,
            conditioning=True,
        )
        self._state.commit(record.object_id, entry)
        record.used = True
        return self._public_prediction(record.object_id, entry)

    def _empty_prompt(self) -> dict[str, np.ndarray]:
        handle = self._require_video()
        values, _ = _prompt_arrays(InteractivePrompt(), handle.frame_size)
        return values

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
        object_ids = list(self._state.objects)
        if not object_ids:
            raise ObjectStateError("propagate requires at least one object")
        missing = [
            value
            for value in object_ids
            if not self._state.require_object(value).conditioning
        ]
        if missing:
            raise ObjectStateError(
                f"objects require a committed conditioning frame: {missing}"
            )
        step = -1 if reverse else 1
        results: list[VideoFramePrediction] = []
        for frame_index in range(start_frame, end_frame + step, step):
            frame_cache = self._get_frame_cache(frame_index)
            predictions: dict[int, VideoPrediction] = {}
            pending: list[int] = []
            for object_id in object_ids:
                object_state = self._state.require_object(object_id)
                existing = object_state.conditioning.get(frame_index)
                if existing is None:
                    existing = object_state.non_conditioning.get(frame_index)
                if existing is None:
                    pending.append(object_id)
                else:
                    predictions[object_id] = self._public_prediction(
                        object_id, existing
                    )
            for packed in self._state.pack(
                pending,
                frame_index=frame_index,
                reverse=reverse,
                video_frame_count=handle.frame_count,
            ):
                if self._adapter.fused_default:
                    backend, committed = self._adapter.step_and_commit(
                        frame_cache, packed, self._empty_prompt()
                    )
                else:
                    backend = self._adapter.preview(
                        frame_cache,
                        packed,
                        self._empty_prompt(),
                        multimask=False,
                    )
                    committed = self._adapter.commit(
                        frame_cache,
                        packed,
                        backend,
                        is_mask_from_points=False,
                    )
                for row, object_id in enumerate(packed.object_ids):
                    if object_id is None:
                        continue
                    device_entry = committed[row]
                    if device_entry is None:
                        raise RuntimeError("backend omitted an active propagation row")
                    row_backend = _BackendPreviewBatch(
                        low_res_logits=(backend.low_res_logits[row],),
                        scores=(backend.scores[row],),
                        commit_masks=backend.commit_masks,
                        object_pointers=backend.object_pointers,
                        object_scores=backend.object_scores,
                    )
                    entry = self._entry_from_commit(
                        object_id,
                        frame_index,
                        row_backend,
                        device_entry,
                        conditioning=False,
                    )
                    self._state.commit(object_id, entry)
                    predictions[object_id] = self._public_prediction(object_id, entry)
            ordered = [predictions[value] for value in object_ids]
            results.append(
                VideoFramePrediction(
                    frame_index=frame_index,
                    object_ids=np.asarray(object_ids, dtype=np.int64),
                    masks=np.stack([value.mask for value in ordered]),
                    scores=np.asarray(
                        [value.score for value in ordered], dtype=np.float32
                    ),
                    metadata={
                        "plan_id": self._plan.plan_id,
                        "profile_id": self._plan.profile_id,
                        "state_abi": "BaseVideoStateV1",
                        "state_revision": self._state.revision,
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
        self._closed = True


_base_video_adapter_factory: Any = None


def create_video_session(plan_id: str, *, bundle_dir: str | Path) -> BaseVideoSession:
    """Validate and create the shipped M4 SAM3 base video session."""

    if plan_id not in BASE_VIDEO_PLAN_IDS:
        if plan_id in SUPPORTED_PLAN_IDS:
            raise ManifestError(
                f"plan scope mismatch: {plan_id} is not a base video plan"
            )
        raise PlanNotFoundError(f"unknown base video plan: {plan_id}")
    resolved = resolve_plan(bundle_dir, plan_id)
    if resolved.manifest["scope"]["use_case"] != "base-video-tracking":
        raise ManifestError(
            f"plan scope mismatch: {plan_id} is not base-video-tracking"
        )
    factory = _base_video_adapter_factory
    if factory is None:
        from sam3.runtime.base_video_ort import OrtCudaBaseVideoAdapter

        factory = OrtCudaBaseVideoAdapter
    adapter = factory(resolved)
    LOGGER.info(
        "created base video session plan_id=%s manifest_sha256=%s "
        "contract=%s profile=%s scope=%s roles=%s",
        resolved.plan_id,
        resolved.manifest_digest,
        resolved.contract_version,
        resolved.profile_id,
        resolved.manifest["scope"]["scope_label"],
        sorted(resolved.artifacts_by_role),
    )
    return BaseVideoSession(resolved, adapter)


__all__ = [
    "BaseVideoSession",
    "ObjectStateError",
    "PreviewHandle",
    "PreviewHandleError",
    "StateCapacityError",
    "VideoFramePrediction",
    "VideoHandle",
    "VideoPrediction",
    "VideoPreview",
    "VideoStateError",
    "create_video_session",
]
