"""Manifest-driven M2 Public API for SAM3 base text-only image PCS."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from sam3.grounding.tokenizer_ve import SimpleTokenizer
from sam3.runtime.manifest import (
    AOTINDUCTOR_PLAN_ID,
    DEFAULT_PLAN_ID,
    IMAGE_PCS_PLAN_IDS,
    SELECTED_K32_PLAN_ID,
    SPLIT_PLAN_ID,
    SUPPORTED_PLAN_IDS,
    CapabilityError,
    ManifestError,
    PlanNotFoundError,
    ResolvedPlan,
    resolve_plan,
    sha256_file,
)
from sam3.runtime.nms import nms_masks

LOGGER = logging.getLogger(__name__)
IMAGE_SIZE = 1008
TEXT_LENGTH = 32
SELECTED_K = 32


class SessionStateError(RuntimeError):
    """A session operation was called before its required state was set."""


class SessionClosedError(RuntimeError):
    """A closed image PCS session was used or closed again."""


@dataclass(frozen=True)
class PredictOptions:
    """Host-owned result policy for one text prediction."""

    score_threshold: float = 0.5
    nms_iou_threshold: float | None = None
    max_results: int | None = None
    output_size: tuple[int, int] | None = None


@dataclass(frozen=True)
class ImageHandle:
    """Public identity for the active preprocessed image cache."""

    cache_key: str
    original_size: tuple[int, int]


@dataclass(frozen=True)
class PromptHandle:
    """Public identity for the active token/text cache."""

    cache_key: str
    valid_tokens: int


@dataclass(frozen=True)
class Prediction:
    """Image-space boxes and probability masks returned by the Public API."""

    boxes_xyxy: np.ndarray
    scores: np.ndarray
    masks: np.ndarray
    metadata: dict[str, object]


@dataclass(frozen=True)
class _BackendPrediction:
    boxes_cxcywh: np.ndarray
    scores: np.ndarray
    mask_logits: np.ndarray
    query_indices: np.ndarray


class _BackendAdapter(Protocol):
    counters: dict[str, int]

    def encode_image(self, values: np.ndarray) -> object: ...

    def encode_text(
        self, token_ids: np.ndarray, attention_mask: np.ndarray
    ) -> object: ...

    def predict(
        self, image_cache: object, text_cache: object, score_threshold: float
    ) -> _BackendPrediction: ...

    def close(self) -> None: ...


class _OrtCudaAdapter:
    """Known-role ORT CUDA adapter; backend names never cross this class."""

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
        self._plan = plan
        self.counters = {
            "image_encodes": 0,
            "text_encodes": 0,
            "session_launches": 0,
            "d2h_bytes": 0,
            "h2d_bytes": 0,
            "mask_skips": 0,
        }
        self._artifacts = plan.artifacts_by_role
        self._files = {
            record["id"]: plan.bundle_dir / record["path"]
            for record in plan.manifest["files"]
        }
        self._sessions: dict[str, Any] = {}
        for role, artifact in self._artifacts.items():
            session = ort.InferenceSession(
                str(self._files[artifact["entry_file_ref"]]),
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            if "CUDAExecutionProvider" not in session.get_providers():
                raise CapabilityError(f"CUDAExecutionProvider did not load for {role}")
            declared_inputs = {item["backend_name"] for item in artifact["inputs"]}
            actual_inputs = {item.name for item in session.get_inputs()}
            declared_outputs = {item["backend_name"] for item in artifact["outputs"]}
            actual_outputs = {item.name for item in session.get_outputs()}
            if declared_inputs != actual_inputs or declared_outputs != actual_outputs:
                raise CapabilityError(
                    f"ORT binding mismatch for {role}: "
                    f"inputs={sorted(actual_inputs)} outputs={sorted(actual_outputs)}"
                )
            self._sessions[role] = session
        self._constant_image_mask = self._upload(np.zeros((1, 72, 72), dtype=np.bool_))

    def _upload(self, values: np.ndarray) -> Any:
        contiguous = np.ascontiguousarray(values)
        self.counters["h2d_bytes"] += contiguous.nbytes
        return self._ort.OrtValue.ortvalue_from_numpy(contiguous, "cuda", 0)

    def _run(self, role: str, inputs: dict[str, Any]) -> dict[str, Any]:
        artifact = self._artifacts[role]
        bindings_by_ref = {
            item["tensor_ref"]: item["backend_name"] for item in artifact["inputs"]
        }
        binding = self._sessions[role].io_binding()
        for tensor_ref, value in inputs.items():
            binding.bind_ortvalue_input(bindings_by_ref[tensor_ref], value)
        for output in self._sessions[role].get_outputs():
            binding.bind_output(output.name, "cuda", 0)
        self._sessions[role].run_with_iobinding(binding)
        self.counters["session_launches"] += 1
        values = binding.get_outputs()
        non_cuda = [
            value.device_name() for value in values if value.device_name() != "cuda"
        ]
        if non_cuda:
            raise CapabilityError(f"IOBinding output left CUDA for {role}: {non_cuda}")
        refs = [item["tensor_ref"] for item in artifact["outputs"]]
        return dict(zip(refs, values))

    def _to_numpy(self, value: Any) -> np.ndarray:
        result = value.numpy()
        self.counters["d2h_bytes"] += result.nbytes
        return result

    def encode_image(self, values: np.ndarray) -> object:
        self.counters["image_encodes"] += 1
        return self._run(
            "detector-image-encode", {"pixel-values": self._upload(values)}
        )

    def encode_text(self, token_ids: np.ndarray, attention_mask: np.ndarray) -> object:
        self.counters["text_encodes"] += 1
        return self._run(
            "text-encode",
            {
                "input-ids": self._upload(token_ids),
                "attention-mask": self._upload(attention_mask),
            },
        )

    def _full_inputs(
        self, image: dict[str, Any], text: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "image-feature-0": image["image-feature-0"],
            "image-feature-1": image["image-feature-1"],
            "image-feature-2": image["image-feature-2"],
            "image-pos-2": image["image-pos-2"],
            "image-mask-2": self._constant_image_mask,
            "text-memory": text["text-memory"],
            "text-padding-mask": text["text-padding-mask"],
        }

    def _encode_grounding(
        self, image: dict[str, Any], text: dict[str, Any]
    ) -> dict[str, Any]:
        return self._run(
            "grounding-encode",
            {
                "image-feature-2": image["image-feature-2"],
                "image-pos-2": image["image-pos-2"],
                "image-mask-2": self._constant_image_mask,
                "text-memory": text["text-memory"],
                "text-padding-mask": text["text-padding-mask"],
            },
        )

    @staticmethod
    def _scores(logits: np.ndarray, presence: np.ndarray) -> np.ndarray:
        logits_probability = 1.0 / (1.0 + np.exp(-logits.astype(np.float32)))
        presence_probability = 1.0 / (1.0 + np.exp(-presence.astype(np.float32)))
        return (logits_probability.squeeze(-1) * presence_probability.reshape(1, 1))[0]

    def _raw_prediction(self, outputs: dict[str, Any]) -> _BackendPrediction:
        logits = self._to_numpy(outputs["logits"])
        boxes = self._to_numpy(outputs["boxes-cxcywh"])[0]
        masks = self._to_numpy(outputs["mask-logits"])[0]
        presence = self._to_numpy(outputs["presence-logits"])
        scores = self._scores(logits, presence)
        return _BackendPrediction(
            boxes_cxcywh=boxes.astype(np.float32),
            scores=scores.astype(np.float32),
            mask_logits=masks.astype(np.float32),
            query_indices=np.arange(scores.shape[0], dtype=np.int64),
        )

    def _predict_default(
        self, image: dict[str, Any], text: dict[str, Any]
    ) -> _BackendPrediction:
        return self._raw_prediction(
            self._run("grounding-full", self._full_inputs(image, text))
        )

    def _predict_split(
        self, image: dict[str, Any], text: dict[str, Any]
    ) -> _BackendPrediction:
        encoded = self._encode_grounding(image, text)
        outputs = self._run(
            "grounding-decode",
            {
                "image-feature-0": image["image-feature-0"],
                "image-feature-1": image["image-feature-1"],
                "image-feature-2": image["image-feature-2"],
                **encoded,
            },
        )
        return self._raw_prediction(outputs)

    def _predict_selected(
        self,
        image: dict[str, Any],
        text: dict[str, Any],
        score_threshold: float,
    ) -> _BackendPrediction:
        encoded = self._encode_grounding(image, text)
        query = self._run(
            "grounding-query-core",
            {
                "image-feature-2": image["image-feature-2"],
                "memory": encoded["memory"],
                "pos-embed": encoded["pos-embed"],
                "memory-padding-mask": encoded["memory-padding-mask"],
                "level-start-index": encoded["level-start-index"],
                "spatial-shapes": encoded["spatial-shapes"],
                "valid-ratios": encoded["valid-ratios"],
                "prompt-memory": encoded["prompt-memory"],
                "prompt-padding-mask": encoded["prompt-padding-mask"],
            },
        )
        logits = self._to_numpy(query["logits"])
        boxes = self._to_numpy(query["boxes-cxcywh"])[0].astype(np.float32)
        presence = self._to_numpy(query["presence-logits"])
        scores = self._scores(logits, presence).astype(np.float32)
        admitted = np.flatnonzero(scores > score_threshold)
        order = np.lexsort((admitted, -scores[admitted]))
        selected = admitted[order][:SELECTED_K]
        selected_fixed = np.zeros((1, SELECTED_K), dtype=np.int64)
        valid = np.zeros((1, SELECTED_K), dtype=np.bool_)
        selected_fixed[0, : selected.size] = selected
        valid[0, : selected.size] = True
        if selected.size == 0:
            self.counters["mask_skips"] += 1
            masks = np.empty((0, 288, 288), dtype=np.float32)
        else:
            mask_outputs = self._run(
                "grounding-mask-selected-k32",
                {
                    "image-feature-0": image["image-feature-0"],
                    "image-feature-1": image["image-feature-1"],
                    "image-feature-2": image["image-feature-2"],
                    "memory": encoded["memory"],
                    "prompt-memory": encoded["prompt-memory"],
                    "prompt-padding-mask": encoded["prompt-padding-mask"],
                    "query-embeddings": query["query-embeddings"],
                    "selected-indices": self._upload(selected_fixed),
                    "valid-mask": self._upload(valid),
                },
            )
            masks = self._to_numpy(mask_outputs["selected-mask-logits"])[
                0, : selected.size
            ].astype(np.float32)
        return _BackendPrediction(
            boxes_cxcywh=boxes[selected],
            scores=scores[selected],
            mask_logits=masks,
            query_indices=selected.astype(np.int64),
        )

    def predict(
        self, image_cache: object, text_cache: object, score_threshold: float
    ) -> _BackendPrediction:
        image = image_cache
        text = text_cache
        if not isinstance(image, dict) or not isinstance(text, dict):
            raise TypeError("invalid ORT cache state")
        if self._plan.plan_id in {DEFAULT_PLAN_ID, AOTINDUCTOR_PLAN_ID}:
            return self._predict_default(image, text)
        if self._plan.plan_id == SPLIT_PLAN_ID:
            return self._predict_split(image, text)
        if self._plan.plan_id == SELECTED_K32_PLAN_ID:
            return self._predict_selected(image, text, score_threshold)
        raise CapabilityError(f"adapter does not implement plan {self._plan.plan_id}")

    def close(self) -> None:
        self._sessions.clear()
        self._constant_image_mask = None


class _AOTInductorCudaAdapter(_OrtCudaAdapter):
    """AOTInductor CUDA adapter with the same semantic image PCS ABI."""

    def __init__(self, plan: ResolvedPlan) -> None:
        if not torch.cuda.is_available():
            raise CapabilityError("CUDA is unavailable for AOTInductor")
        self._plan = plan
        self.counters = {
            "image_encodes": 0,
            "text_encodes": 0,
            "session_launches": 0,
            "d2h_bytes": 0,
            "h2d_bytes": 0,
            "mask_skips": 0,
        }
        self._artifacts = plan.artifacts_by_role
        self._files = {
            record["id"]: plan.bundle_dir / record["path"]
            for record in plan.manifest["files"]
        }
        try:
            self._sessions = {
                role: torch._inductor.aoti_load_package(  # type: ignore[attr-defined]
                    self._files[artifact["entry_file_ref"]]
                )
                for role, artifact in self._artifacts.items()
            }
        except (OSError, RuntimeError) as exc:
            raise CapabilityError("AOTInductor package could not be loaded") from exc
        self._constant_image_mask = self._upload(np.zeros((1, 72, 72), dtype=np.bool_))

    def _upload(self, values: np.ndarray) -> torch.Tensor:
        contiguous = np.ascontiguousarray(values)
        self.counters["h2d_bytes"] += contiguous.nbytes
        return torch.from_numpy(contiguous).to("cuda")

    def _run(
        self, role: str, inputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        artifact = self._artifacts[role]
        expected = [item["tensor_ref"] for item in artifact["inputs"]]
        missing = sorted(set(expected) - set(inputs))
        extra = sorted(set(inputs) - set(expected))
        if missing or extra:
            raise CapabilityError(
                f"AOTInductor binding mismatch for {role}: "
                f"missing={missing} extra={extra}"
            )
        raw = self._sessions[role](*[inputs[name] for name in expected])
        self.counters["session_launches"] += 1
        values = list(raw) if isinstance(raw, (tuple, list)) else [raw]
        if any(not isinstance(value, torch.Tensor) for value in values):
            raise CapabilityError(f"AOTInductor returned non-tensor output for {role}")
        if any(value.device.type != "cuda" for value in values):
            raise CapabilityError(f"AOTInductor output left CUDA for {role}")
        refs = [item["tensor_ref"] for item in artifact["outputs"]]
        if len(refs) != len(values):
            raise CapabilityError(
                f"AOTInductor output count mismatch for {role}: "
                f"{len(values)} != {len(refs)}"
            )
        return dict(zip(refs, values))

    def _to_numpy(self, value: torch.Tensor) -> np.ndarray:
        result = value.detach().cpu().numpy()
        self.counters["d2h_bytes"] += result.nbytes
        return result

    def close(self) -> None:
        self._sessions.clear()
        self._constant_image_mask = None


_adapter_factory: Any = None


def _hash_parts(*parts: bytes) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(len(part).to_bytes(8, "little"))
        digest.update(part)
    return digest.hexdigest()


def _preprocess_image(
    image: Image.Image | np.ndarray,
) -> tuple[np.ndarray, tuple[int, int]]:
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
    resized = pil.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    values = np.asarray(resized, dtype=np.float32)
    values = ((values / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)[None]
    return np.ascontiguousarray(values.astype(np.float16)), original_size


def _validate_options(
    options: PredictOptions, selected_k: bool
) -> tuple[int, int] | None:
    if not np.isfinite(options.score_threshold):
        raise ValueError("score_threshold must be finite")
    if options.nms_iou_threshold is not None and not (
        0.0 <= options.nms_iou_threshold <= 1.0
    ):
        raise ValueError("nms_iou_threshold must be in [0, 1]")
    if options.max_results is not None and options.max_results < 0:
        raise ValueError("max_results must be non-negative")
    if (
        selected_k
        and options.max_results is not None
        and options.max_results > SELECTED_K
    ):
        raise ValueError("selected-K32 plan cannot return more than 32 results")
    if options.output_size is not None:
        height, width = options.output_size
        if height <= 0 or width <= 0:
            raise ValueError("output_size must contain positive (height, width)")
    return options.output_size


def _select_prediction(
    prediction: _BackendPrediction, options: PredictOptions
) -> np.ndarray:
    admitted = np.flatnonzero(prediction.scores > options.score_threshold)
    order = np.lexsort(
        (prediction.query_indices[admitted], -prediction.scores[admitted])
    )
    selected = admitted[order]
    if options.nms_iou_threshold is not None and selected.size:
        keep = nms_masks(
            torch.from_numpy(prediction.scores[selected]),
            torch.from_numpy(prediction.mask_logits[selected]),
            float("-inf"),
            options.nms_iou_threshold,
        ).numpy()
        selected = selected[keep]
    if options.max_results is not None:
        selected = selected[: options.max_results]
    return selected


def _boxes_to_xyxy(boxes: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
    height, width = output_size
    cx, cy, box_width, box_height = boxes.T
    result = np.stack(
        (
            (cx - box_width / 2.0) * width,
            (cy - box_height / 2.0) * height,
            (cx + box_width / 2.0) * width,
            (cy + box_height / 2.0) * height,
        ),
        axis=1,
    ).astype(np.float32)
    result[:, 0::2] = np.clip(result[:, 0::2], 0.0, float(width))
    result[:, 1::2] = np.clip(result[:, 1::2], 0.0, float(height))
    return result


def _resize_masks(mask_logits: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
    if mask_logits.shape[0] == 0:
        return np.empty((0, *output_size), dtype=np.float32)
    probabilities = torch.sigmoid(torch.from_numpy(mask_logits).float()).unsqueeze(1)
    resized = F.interpolate(
        probabilities,
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )
    return resized[:, 0].numpy().astype(np.float32, copy=False)


class ImagePCSSession:
    """Stateful public session with independent image and prompt caches."""

    def __init__(self, plan: ResolvedPlan, adapter: _BackendAdapter) -> None:
        self._plan = plan
        self._adapter = adapter
        tokenizer_path = plan.bundle_dir / "tokenizer" / "bpe_simple_vocab_16e6.txt.gz"
        self._tokenizer = SimpleTokenizer(tokenizer_path, context_length=TEXT_LENGTH)
        self._tokenizer_digest = sha256_file(tokenizer_path)
        self._image_handle: ImageHandle | None = None
        self._prompt_handle: PromptHandle | None = None
        self._image_cache: object | None = None
        self._prompt_cache: object | None = None
        self._last_query_indices = np.empty((0,), dtype=np.int64)
        self._closed = False

    @property
    def plan_id(self) -> str:
        return self._plan.plan_id

    def _ensure_open(self) -> None:
        if self._closed:
            raise SessionClosedError("image PCS session is closed")

    def set_image(self, image: Image.Image | np.ndarray) -> ImageHandle:
        self._ensure_open()
        values, original_size = _preprocess_image(image)
        checkpoint = self._plan.manifest["model"]["checkpoint"]["digest"]["value"]
        cache = next(
            item
            for item in self._plan.manifest["caches"]
            if item["id"] == "image-cache"
        )
        cache_key = _hash_parts(
            values.tobytes(),
            np.asarray(original_size, dtype=np.int64).tobytes(),
            str(checkpoint).encode(),
            self._plan.profile_id.encode(),
            str(cache["key_version"]).encode(),
        )
        if self._image_handle is not None and self._image_handle.cache_key == cache_key:
            return self._image_handle
        new_cache = self._adapter.encode_image(values)
        self._image_cache = new_cache
        self._image_handle = ImageHandle(
            cache_key=cache_key, original_size=original_size
        )
        return self._image_handle

    def set_text(self, text: str) -> PromptHandle:
        self._ensure_open()
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        token_ids_torch = self._tokenizer([text], context_length=TEXT_LENGTH)
        attention_torch = token_ids_torch.ne(0)
        token_ids = token_ids_torch.numpy().astype(np.int64, copy=False)
        attention = attention_torch.numpy().astype(np.bool_, copy=False)
        checkpoint = self._plan.manifest["model"]["checkpoint"]["digest"]["value"]
        model_revision = self._plan.manifest["model"]["model_revision"]
        cache = next(
            item
            for item in self._plan.manifest["caches"]
            if item["id"] == "prompt-cache"
        )
        cache_key = _hash_parts(
            token_ids.tobytes(),
            self._tokenizer_digest.encode(),
            str(checkpoint).encode(),
            str(model_revision).encode(),
            self._plan.profile_id.encode(),
            str(cache["key_version"]).encode(),
        )
        if (
            self._prompt_handle is not None
            and self._prompt_handle.cache_key == cache_key
        ):
            return self._prompt_handle
        new_cache = self._adapter.encode_text(token_ids, attention)
        self._prompt_cache = new_cache
        self._prompt_handle = PromptHandle(
            cache_key=cache_key, valid_tokens=int(np.count_nonzero(attention))
        )
        return self._prompt_handle

    def predict_text(self, options: PredictOptions = PredictOptions()) -> Prediction:
        self._ensure_open()
        _validate_options(options, self._plan.plan_id == SELECTED_K32_PLAN_ID)
        if self._image_cache is None or self._image_handle is None:
            raise SessionStateError("set_image must be called before predict_text")
        if self._prompt_cache is None or self._prompt_handle is None:
            raise SessionStateError("set_text must be called before predict_text")
        raw = self._adapter.predict(
            self._image_cache, self._prompt_cache, options.score_threshold
        )
        selected = _select_prediction(raw, options)
        self._last_query_indices = raw.query_indices[selected].copy()
        output_size = options.output_size or self._image_handle.original_size
        boxes = _boxes_to_xyxy(raw.boxes_cxcywh[selected], output_size)
        masks = _resize_masks(raw.mask_logits[selected], output_size)
        scores = raw.scores[selected].astype(np.float32, copy=False)
        output_policy = next(
            item["value"]
            for item in self._plan.manifest["policies"]
            if item["name"] == "output-policy"
        )
        metadata: dict[str, object] = {
            "plan_id": self._plan.plan_id,
            "contract_version": self._plan.contract_version,
            "profile_id": self._plan.profile_id,
            "image_cache_key": self._image_handle.cache_key,
            "prompt_cache_key": self._prompt_handle.cache_key,
            "output_policy": output_policy,
            "original_size": self._image_handle.original_size,
            "output_size": output_size,
        }
        return Prediction(
            boxes_xyxy=boxes, scores=scores, masks=masks, metadata=metadata
        )

    def close(self) -> None:
        self._ensure_open()
        self._adapter.close()
        self._image_cache = None
        self._prompt_cache = None
        self._image_handle = None
        self._prompt_handle = None
        self._closed = True


def create_image_session(plan_id: str, *, bundle_dir: str | Path) -> ImagePCSSession:
    """Validate and create one manifest-driven M2 image PCS session."""

    if plan_id not in IMAGE_PCS_PLAN_IDS:
        if plan_id in SUPPORTED_PLAN_IDS:
            raise ManifestError(
                f"plan scope mismatch: {plan_id} is not an image PCS plan"
            )
        raise PlanNotFoundError(f"unknown image PCS plan: {plan_id}")
    resolved = resolve_plan(bundle_dir, plan_id)
    if resolved.manifest["scope"]["use_case"] != "image-pcs":
        raise ManifestError(f"plan scope mismatch: {plan_id} is not image-pcs")
    factory = _adapter_factory
    if factory is None:
        factory = (
            _AOTInductorCudaAdapter
            if resolved.manifest["backend"]["kind"] == "aotinductor"
            else _OrtCudaAdapter
        )
    adapter = factory(resolved)
    LOGGER.info(
        "created image PCS session plan_id=%s manifest_sha256=%s "
        "contract=%s profile=%s dispatch_role=%s",
        resolved.plan_id,
        resolved.manifest_digest,
        resolved.contract_version,
        resolved.profile_id,
        resolved.dispatch_role,
    )
    return ImagePCSSession(resolved, adapter)


__all__ = [
    "ImageHandle",
    "ImagePCSSession",
    "PredictOptions",
    "Prediction",
    "PromptHandle",
    "SessionClosedError",
    "SessionStateError",
    "create_image_session",
]
