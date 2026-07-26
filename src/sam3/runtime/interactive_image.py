"""Manifest-driven M3 Public API for SAM3 base interactive image PVS."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from sam3.runtime.image_pcs import (
    ImageHandle,
    SessionClosedError,
    SessionStateError,
    _hash_parts,
)
from sam3.runtime.manifest import (
    INTERACTIVE_PLAN_IDS,
    SUPPORTED_PLAN_IDS,
    CapabilityError,
    ManifestError,
    PlanNotFoundError,
    ResolvedPlan,
    resolve_plan,
)

LOGGER = logging.getLogger(__name__)
POINT_CAPACITY = 16
MODEL_SIZE = 1008
MASK_SIZE = 288
INITIAL_CONDITION_ID = "initial-no-memory"


@dataclass(frozen=True)
class InteractivePrompt:
    """Point, box, and prior low-resolution mask logits for one click."""

    points_xy: np.ndarray | None = None
    point_labels: np.ndarray | None = None
    box_xyxy: tuple[float, float, float, float] | None = None
    mask_logits: np.ndarray | None = None


@dataclass(frozen=True)
class InteractivePredictOptions:
    """Host-owned mask selection and output policy."""

    multimask_output: bool = True
    mask_threshold: float = 0.0
    output_size: tuple[int, int] | None = None


@dataclass(frozen=True)
class InteractivePrediction:
    """Interactive masks, scores, and reusable low-resolution logits."""

    masks: np.ndarray
    scores: np.ndarray
    low_res_logits: np.ndarray
    metadata: dict[str, object]


class _InteractiveBackendAdapter(Protocol):
    counters: dict[str, int]

    def encode_image(self, values: np.ndarray) -> object: ...

    def predict(
        self, image_cache: object, prompt_inputs: dict[str, np.ndarray], multimask: bool
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def close(self) -> None: ...


class _OrtCudaInteractiveAdapter:
    """Known-role ORT adapter; all image features stay as CUDA OrtValues."""

    _ENCODE_ROLE = "interactive-image-encode-initial"
    _MULTIMASK_ROLE = "interactive-predict-multimask3"
    _SINGLE_ROLE = "interactive-predict-single1"

    def __init__(self, plan: ResolvedPlan) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise CapabilityError(
                "ONNX Runtime CUDA is not installed; install sam3[ort-cuda]"
            ) from exc
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise CapabilityError("CUDAExecutionProvider is unavailable")

        self._ort = ort
        self._artifacts = plan.artifacts_by_role
        required_roles = {self._ENCODE_ROLE, self._MULTIMASK_ROLE, self._SINGLE_ROLE}
        missing_roles = sorted(required_roles - set(self._artifacts))
        if missing_roles:
            raise CapabilityError(f"interactive artifacts are missing: {missing_roles}")
        files = {
            record["id"]: plan.bundle_dir / record["path"]
            for record in plan.manifest["files"]
        }
        self._sessions: dict[str, Any] = {}
        for role in required_roles:
            artifact = self._artifacts[role]
            session = ort.InferenceSession(
                str(files[artifact["entry_file_ref"]]),
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            if "CUDAExecutionProvider" not in session.get_providers():
                raise CapabilityError(f"CUDAExecutionProvider did not load for {role}")
            declared_inputs = {item["backend_name"] for item in artifact["inputs"]}
            declared_outputs = {item["backend_name"] for item in artifact["outputs"]}
            actual_inputs = {item.name for item in session.get_inputs()}
            actual_outputs = {item.name for item in session.get_outputs()}
            if declared_inputs != actual_inputs or declared_outputs != actual_outputs:
                raise CapabilityError(
                    f"ORT binding mismatch for {role}: "
                    f"inputs={sorted(actual_inputs)} outputs={sorted(actual_outputs)}"
                )
            self._sessions[role] = session
        self.counters = {
            "image_encodes": 0,
            "predict_launches": 0,
            "session_launches": 0,
            "d2h_bytes": 0,
            "h2d_bytes": 0,
            "memory_encodes": 0,
            "memory_commits": 0,
        }

    def _upload(self, values: np.ndarray) -> Any:
        contiguous = np.ascontiguousarray(values)
        self.counters["h2d_bytes"] += contiguous.nbytes
        return self._ort.OrtValue.ortvalue_from_numpy(contiguous, "cuda", 0)

    def _run(self, role: str, inputs: dict[str, Any]) -> dict[str, Any]:
        artifact = self._artifacts[role]
        backend_names = {
            item["tensor_ref"]: item["backend_name"] for item in artifact["inputs"]
        }
        binding = self._sessions[role].io_binding()
        for tensor_ref, value in inputs.items():
            binding.bind_ortvalue_input(backend_names[tensor_ref], value)
        for output in self._sessions[role].get_outputs():
            binding.bind_output(output.name, "cuda", 0)
        self._sessions[role].run_with_iobinding(binding)
        self.counters["session_launches"] += 1
        values = binding.get_outputs()
        if any(value.device_name() != "cuda" for value in values):
            raise CapabilityError(f"IOBinding output left CUDA for {role}")
        refs = [item["tensor_ref"] for item in artifact["outputs"]]
        return dict(zip(refs, values))

    def _to_numpy(self, value: Any) -> np.ndarray:
        result = value.numpy()
        self.counters["d2h_bytes"] += result.nbytes
        return result

    def encode_image(self, values: np.ndarray) -> object:
        self.counters["image_encodes"] += 1
        return self._run(self._ENCODE_ROLE, {"pixel-values": self._upload(values)})

    def predict(
        self, image_cache: object, prompt_inputs: dict[str, np.ndarray], multimask: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(image_cache, dict):
            raise TypeError("invalid interactive image cache state")
        role = self._MULTIMASK_ROLE if multimask else self._SINGLE_ROLE
        inputs = {
            "image-embedding": image_cache["image-embedding"],
            "high-res-0": image_cache["high-res-0"],
            "high-res-1": image_cache["high-res-1"],
        }
        inputs.update(
            {key: self._upload(value) for key, value in prompt_inputs.items()}
        )
        outputs = self._run(role, inputs)
        self.counters["predict_launches"] += 1
        policy = "multimask3" if multimask else "single1"
        logits = self._to_numpy(outputs[f"low-res-logits-{policy}"])[0].astype(
            np.float32, copy=False
        )
        scores = self._to_numpy(outputs[f"scores-{policy}"])[0].astype(
            np.float32, copy=False
        )
        return logits, scores

    def close(self) -> None:
        self._sessions.clear()


_interactive_adapter_factory: Any = _OrtCudaInteractiveAdapter


def _preprocess_interactive_image(
    image: Image.Image | np.ndarray,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Match the official SAM2Transforms image path for the M3 profile."""

    from torchvision.transforms import v2

    if isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[2] not in (3, 4):
            raise ValueError("NumPy image must have shape [H,W,3] or [H,W,4]")
        if image.dtype != np.uint8:
            raise ValueError("NumPy image must use uint8 pixels")
        pil = Image.fromarray(image).convert("RGB")
    elif isinstance(image, Image.Image):
        pil = image.convert("RGB")
    else:
        raise TypeError("image must be a PIL image or uint8 NumPy array")
    original_size = (pil.height, pil.width)
    transform = v2.Compose(
        [
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(size=(MODEL_SIZE, MODEL_SIZE)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    values = transform(v2.functional.to_image(pil)).unsqueeze(0)
    return (
        np.ascontiguousarray(values.numpy().astype(np.float16)),
        original_size,
    )


def _prompt_arrays(
    prompt: InteractivePrompt, original_size: tuple[int, int]
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    if not isinstance(prompt, InteractivePrompt):
        raise TypeError("prompt must be InteractivePrompt")
    height, width = original_size
    coords = np.zeros((1, POINT_CAPACITY, 2), dtype=np.float32)
    labels = np.full((1, POINT_CAPACITY), -1, dtype=np.int64)
    valid = np.zeros((1, POINT_CAPACITY), dtype=np.bool_)

    if prompt.points_xy is None:
        if prompt.point_labels is not None:
            raise ValueError("point_labels requires points_xy")
        count = 0
    else:
        raw_coords = np.asarray(prompt.points_xy)
        if raw_coords.ndim != 2 or raw_coords.shape[1] != 2:
            raise ValueError("points_xy must have shape [N,2]")
        count = int(raw_coords.shape[0])
        if count > POINT_CAPACITY:
            raise ValueError("interactive point capacity is 16")
        if prompt.point_labels is None:
            raise ValueError("point_labels is required with points_xy")
        raw_labels = np.asarray(prompt.point_labels)
        if raw_labels.shape != (count,):
            raise ValueError("point_labels shape must match points_xy")
        if not np.all(np.isin(raw_labels, (0, 1))):
            raise ValueError("point_labels must contain only 0 or 1")
        raw_coords = raw_coords.astype(np.float32)
        if not np.all(np.isfinite(raw_coords)):
            raise ValueError("points_xy must be finite")
        coords[0, :count, 0] = raw_coords[:, 0] / float(width) * MODEL_SIZE
        coords[0, :count, 1] = raw_coords[:, 1] / float(height) * MODEL_SIZE
        labels[0, :count] = raw_labels.astype(np.int64)
        valid[0, :count] = True

    box = np.zeros((1, 4), dtype=np.float32)
    has_box = np.asarray([prompt.box_xyxy is not None], dtype=np.bool_)
    if prompt.box_xyxy is not None:
        raw_box = np.asarray(prompt.box_xyxy, dtype=np.float32)
        if raw_box.shape != (4,) or not np.all(np.isfinite(raw_box)):
            raise ValueError("box_xyxy must contain four finite values")
        if raw_box[0] >= raw_box[2] or raw_box[1] >= raw_box[3]:
            raise ValueError("box_xyxy must satisfy x0 < x1 and y0 < y1")
        box[0, 0::2] = raw_box[0::2] / float(width) * MODEL_SIZE
        box[0, 1::2] = raw_box[1::2] / float(height) * MODEL_SIZE

    mask = np.zeros((1, 1, MASK_SIZE, MASK_SIZE), dtype=np.float32)
    has_mask = np.asarray([prompt.mask_logits is not None], dtype=np.bool_)
    if prompt.mask_logits is not None:
        raw_mask = np.asarray(prompt.mask_logits)
        if raw_mask.shape == (MASK_SIZE, MASK_SIZE):
            raw_mask = raw_mask[None]
        if raw_mask.shape != (1, MASK_SIZE, MASK_SIZE):
            raise ValueError("mask_logits must have shape [288,288] or [1,288,288]")
        if raw_mask.dtype == np.bool_ or not np.issubdtype(raw_mask.dtype, np.number):
            raise ValueError("mask_logits must contain numeric logits")
        raw_mask = raw_mask.astype(np.float32)
        if not np.all(np.isfinite(raw_mask)):
            raise ValueError("mask_logits must be finite")
        mask[0] = raw_mask

    arrays = {
        "point-coords": coords,
        "point-labels": labels,
        "point-valid": valid,
        "box-xyxy": box,
        "has-box": has_box,
        "mask-input": mask,
        "has-mask": has_mask,
    }
    facts: dict[str, object] = {
        "point_count": count,
        "box_count": int(has_box[0]),
        "has_mask": bool(has_mask[0]),
    }
    return arrays, facts


def _validate_options(
    options: InteractivePredictOptions, original_size: tuple[int, int]
) -> tuple[int, int]:
    if not isinstance(options, InteractivePredictOptions):
        raise TypeError("options must be InteractivePredictOptions")
    if not isinstance(options.multimask_output, bool):
        raise TypeError("multimask_output must be bool")
    if not np.isfinite(options.mask_threshold):
        raise ValueError("mask_threshold must be finite")
    output_size = options.output_size or original_size
    if len(output_size) != 2 or output_size[0] <= 0 or output_size[1] <= 0:
        raise ValueError("output_size must contain positive (height, width)")
    return int(output_size[0]), int(output_size[1])


def _resize_and_threshold(
    logits: np.ndarray, output_size: tuple[int, int], threshold: float
) -> np.ndarray:
    values = torch.from_numpy(logits).float().unsqueeze(1)
    resized = F.interpolate(
        values, size=output_size, mode="bilinear", align_corners=False
    )[:, 0]
    return (resized > threshold).numpy()


class InteractiveImageSession:
    """Image-cached interactive session with no video or object state."""

    def __init__(self, plan: ResolvedPlan, adapter: _InteractiveBackendAdapter) -> None:
        self._plan = plan
        self._adapter = adapter
        self._image_handle: ImageHandle | None = None
        self._image_cache: object | None = None
        self._closed = False

    @property
    def plan_id(self) -> str:
        return self._plan.plan_id

    def _ensure_open(self) -> None:
        if self._closed:
            raise SessionClosedError("interactive image session is closed")

    def set_image(self, image: Image.Image | np.ndarray) -> ImageHandle:
        self._ensure_open()
        values, original_size = _preprocess_interactive_image(image)
        checkpoint = self._plan.manifest["model"]["checkpoint"]["digest"]["value"]
        cache = next(
            item
            for item in self._plan.manifest["caches"]
            if item["id"] == "image-cache"
        )
        condition_id = next(
            item["value"]
            for item in self._plan.manifest["policies"]
            if item["name"] == "image-condition"
        )
        cache_key = _hash_parts(
            values.tobytes(),
            np.asarray(original_size, dtype=np.int64).tobytes(),
            str(checkpoint).encode(),
            self._plan.profile_id.encode(),
            str(condition_id).encode(),
            str(cache["key_version"]).encode(),
        )
        if self._image_handle is not None and self._image_handle.cache_key == cache_key:
            return self._image_handle
        new_cache = self._adapter.encode_image(values)
        new_handle = ImageHandle(cache_key=cache_key, original_size=original_size)
        self._image_cache = new_cache
        self._image_handle = new_handle
        return new_handle

    def predict(
        self,
        prompt: InteractivePrompt = InteractivePrompt(),
        options: InteractivePredictOptions = InteractivePredictOptions(),
    ) -> InteractivePrediction:
        self._ensure_open()
        if self._image_cache is None or self._image_handle is None:
            raise SessionStateError("set_image must be called before predict")
        output_size = _validate_options(options, self._image_handle.original_size)
        prompt_inputs, prompt_facts = _prompt_arrays(
            prompt, self._image_handle.original_size
        )
        logits, scores = self._adapter.predict(
            self._image_cache, prompt_inputs, options.multimask_output
        )
        expected_masks = 3 if options.multimask_output else 1
        if logits.shape != (expected_masks, MASK_SIZE, MASK_SIZE):
            raise CapabilityError(f"interactive logits shape mismatch: {logits.shape}")
        if scores.shape != (expected_masks,):
            raise CapabilityError(f"interactive scores shape mismatch: {scores.shape}")
        artifact = "multimask3" if options.multimask_output else "single1"
        metadata: dict[str, object] = {
            "plan_id": self._plan.plan_id,
            "contract_version": self._plan.contract_version,
            "profile_id": self._plan.profile_id,
            "image_cache_key": self._image_handle.cache_key,
            **prompt_facts,
            "multimask_artifact": artifact,
            "original_size": self._image_handle.original_size,
            "output_size": output_size,
            "output_policy": "bilinear low-resolution logits; strict > threshold",
        }
        return InteractivePrediction(
            masks=_resize_and_threshold(logits, output_size, options.mask_threshold),
            scores=scores.astype(np.float32, copy=False),
            low_res_logits=logits.astype(np.float32, copy=False),
            metadata=metadata,
        )

    def close(self) -> None:
        self._ensure_open()
        self._adapter.close()
        self._image_cache = None
        self._image_handle = None
        self._closed = True


def create_interactive_session(
    plan_id: str, *, bundle_dir: str | Path
) -> InteractiveImageSession:
    """Validate and create the shipped M3 interactive image session."""

    if plan_id not in INTERACTIVE_PLAN_IDS:
        if plan_id in SUPPORTED_PLAN_IDS:
            raise ManifestError(
                f"plan scope mismatch: {plan_id} is not an interactive image plan"
            )
        raise PlanNotFoundError(f"unknown interactive plan: {plan_id}")
    resolved = resolve_plan(bundle_dir, plan_id)
    if resolved.manifest["scope"]["use_case"] != "interactive-image-pvs":
        raise ManifestError(
            f"plan scope mismatch: {plan_id} is not interactive-image-pvs"
        )
    adapter = _interactive_adapter_factory(resolved)
    LOGGER.info(
        "created interactive image session plan_id=%s manifest_sha256=%s "
        "contract=%s profile=%s scope=%s roles=%s",
        resolved.plan_id,
        resolved.manifest_digest,
        resolved.contract_version,
        resolved.profile_id,
        resolved.manifest["scope"]["scope_label"],
        sorted(resolved.artifacts_by_role),
    )
    return InteractiveImageSession(resolved, adapter)


__all__ = [
    "InteractiveImageSession",
    "InteractivePrediction",
    "InteractivePredictOptions",
    "InteractivePrompt",
    "create_interactive_session",
]
