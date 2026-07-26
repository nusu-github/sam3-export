"""Replay M1 E1-E3 export, parity, and ORT CUDA IOBinding measurements.

The ``all`` command runs each GPU-heavy phase in a fresh process so exporter
allocations cannot contaminate ORT peak-memory measurements.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable
from functools import partial
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image
import torch
from torch import Tensor, nn
from torch.export import export
from torchvision.transforms import v2

from sam3.export import (
    GroundingDecode,
    GroundingEncode,
    GroundingEncodeTextOnly,
    GroundingFull,
    GroundingFullFeatureOnly,
    GroundingMaskSelectedK,
    GroundingQueryCore,
    TextOnlyPromptEncode,
    TextTower,
    VisionTowerFlat,
    VisionTowerProfiled,
)
from sam3.grounding.tokenizer_ve import SimpleTokenizer
from sam3.weights.load_sam3 import (
    build_production_text_detector,
    resolve_sam3_checkpoint,
)

IMAGE_SIZE = 1008
TEXT_LENGTH = 32
OPSET_VERSION = 18
K_PROFILES = (16, 32, 64)
OUTPUT_NAMES = ("logits", "boxes_cxcywh", "mask_logits", "presence_logits")


class VisionTowerScalpedFlat(nn.Module):
    """Apply the production VL backbone scalp before selecting an E2 profile."""

    def __init__(self, tower: nn.Module, scalp: int) -> None:
        super().__init__()
        self.tower = tower
        self.scalp = int(scalp)

    def forward(self, pixel_values: Tensor) -> tuple[Tensor, ...]:
        values = self.tower(pixel_values)
        kept = 4 - self.scalp
        return values[:kept] + values[4 : 4 + kept]


class GroundingEncodeFlat(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(
        self,
        image_feature_2: Tensor,
        image_pos_2: Tensor,
        image_mask_2: Tensor,
        text_memory: Tensor,
        text_padding_mask: Tensor,
    ) -> tuple[Tensor, ...]:
        return self.module(
            (image_feature_2,),
            (image_pos_2,),
            (image_mask_2,),
            text_memory,
            text_padding_mask,
        )


class GroundingDecodeFlat(nn.Module):
    def __init__(self, module: GroundingDecode) -> None:
        super().__init__()
        self.module = module

    def forward(
        self,
        image_feature_0: Tensor,
        image_feature_1: Tensor,
        image_feature_2: Tensor,
        memory: Tensor,
        pos_embed: Tensor,
        memory_padding_mask: Tensor,
        level_start_index: Tensor,
        spatial_shapes: Tensor,
        valid_ratios: Tensor,
        encoded_text_memory: Tensor,
        text_padding_mask: Tensor,
    ) -> tuple[Tensor, ...]:
        return self.module(
            (image_feature_0, image_feature_1, image_feature_2),
            memory,
            pos_embed,
            memory_padding_mask,
            level_start_index,
            spatial_shapes,
            valid_ratios,
            encoded_text_memory,
            text_padding_mask,
        )


class GroundingFullFlat(nn.Module):
    def __init__(self, module: GroundingFull) -> None:
        super().__init__()
        self.module = module

    def forward(
        self,
        image_feature_0: Tensor,
        image_feature_1: Tensor,
        image_feature_2: Tensor,
        image_pos_2: Tensor,
        image_mask_2: Tensor,
        text_memory: Tensor,
        text_padding_mask: Tensor,
    ) -> tuple[Tensor, ...]:
        return self.module(
            (image_feature_0, image_feature_1, image_feature_2),
            (image_pos_2,),
            (image_mask_2,),
            text_memory,
            text_padding_mask,
        )


class GroundingFullFeatureOnlyFlat(nn.Module):
    def __init__(self, module: GroundingFullFeatureOnly) -> None:
        super().__init__()
        self.module = module

    def forward(
        self,
        image_feature_0: Tensor,
        image_feature_1: Tensor,
        image_feature_2: Tensor,
        image_mask_2: Tensor,
        text_memory: Tensor,
        text_padding_mask: Tensor,
    ) -> tuple[Tensor, ...]:
        return self.module(
            (image_feature_0, image_feature_1, image_feature_2),
            (image_mask_2,),
            text_memory,
            text_padding_mask,
        )


class GroundingQueryCoreFlat(nn.Module):
    def __init__(self, module: GroundingQueryCore) -> None:
        super().__init__()
        self.module = module

    def forward(
        self,
        image_feature_low: Tensor,
        memory: Tensor,
        pos_embed: Tensor,
        memory_padding_mask: Tensor,
        level_start_index: Tensor,
        spatial_shapes: Tensor,
        valid_ratios: Tensor,
        encoded_text_memory: Tensor,
        text_padding_mask: Tensor,
    ) -> tuple[Tensor, ...]:
        return self.module(
            image_feature_low,
            memory,
            pos_embed,
            memory_padding_mask,
            level_start_index,
            spatial_shapes,
            valid_ratios,
            encoded_text_memory,
            text_padding_mask,
        )


class GroundingMaskSelectedKFlat(nn.Module):
    def __init__(self, module: GroundingMaskSelectedK) -> None:
        super().__init__()
        self.module = module

    def forward(
        self,
        image_feature_0: Tensor,
        image_feature_1: Tensor,
        image_feature_2: Tensor,
        memory: Tensor,
        encoded_text_memory: Tensor,
        text_padding_mask: Tensor,
        query_embeddings: Tensor,
        selected_indices: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor]:
        return self.module(
            (image_feature_0, image_feature_1, image_feature_2),
            memory,
            encoded_text_memory,
            text_padding_mask,
            query_embeddings,
            selected_indices,
            valid_mask,
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _fixtures(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value["fixture_version"] != "m1-image-pcs-v1":
        raise RuntimeError("unexpected M1 fixture version")
    return value


def _image_path(workspace: Path, fixtures: dict[str, Any], image_id: str) -> Path:
    record = next(item for item in fixtures["images"] if item["id"] == image_id)
    path = workspace / record["workspace_path"]
    if _sha256(path) != record["sha256"]:
        raise RuntimeError(f"fixture image hash mismatch: {path}")
    return path


def _preprocess_image(path: Path) -> Tensor:
    transform = v2.Compose(
        [
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(size=(IMAGE_SIZE, IMAGE_SIZE)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    image = Image.open(path).convert("RGB")
    value = transform(v2.functional.to_image(image).to("cuda")).unsqueeze(0)
    return value.to(dtype=torch.float16)


def _as_tuple(value: Tensor | tuple[Tensor, ...]) -> tuple[Tensor, ...]:
    return value if isinstance(value, tuple) else (value,)


def _parity(expected: Iterable[Tensor], actual: Iterable[Tensor]) -> dict[str, float]:
    max_abs = 0.0
    max_rel = 0.0
    for expected_value, actual_value in zip(expected, actual):
        delta = (expected_value.float() - actual_value.float()).abs()
        max_abs = max(max_abs, float(delta.max()))
        denominator = expected_value.float().abs().clamp_min(1e-6)
        max_rel = max(max_rel, float((delta / denominator).max()))
    return {"max_abs": max_abs, "max_rel": max_rel}


def _capture_parity(module: nn.Module, args: tuple[Tensor, ...]) -> dict[str, float]:
    module.eval()
    args = tuple(value.detach().clone() for value in args)
    with torch.no_grad():
        eager = _as_tuple(module(*args))
        exported = export(module, args, strict=False).module()
        ep_outputs = _as_tuple(exported(*args))
    return _parity(eager, ep_outputs)


def _artifact_files(path: Path) -> list[Path]:
    return sorted(
        item
        for item in path.parent.iterdir()
        if item.name == path.name or item.name.startswith(path.name + ".data")
    )


def _artifact_record(path: Path) -> dict[str, Any]:
    files = _artifact_files(path)
    return {
        "files": [
            {
                "name": item.name,
                "size_bytes": item.stat().st_size,
                "sha256": _sha256(item),
            }
            for item in files
        ],
        "size_bytes": sum(item.stat().st_size for item in files),
    }


def _export_one(
    module: nn.Module,
    args: tuple[Tensor, ...],
    path: Path,
    input_names: list[str],
    output_names: list[str],
) -> dict[str, Any]:
    module.eval()
    args = tuple(value.detach().clone() for value in args)
    with torch.no_grad():
        eager = _as_tuple(module(*args))
        exported_program = export(module, args, strict=False)
        ep_outputs = _as_tuple(exported_program.module()(*args))
    ep_parity = _parity(eager, ep_outputs)
    torch.onnx.export(
        exported_program,
        (),
        path,
        input_names=input_names,
        output_names=output_names,
        opset_version=OPSET_VERSION,
        dynamo=True,
        external_data=True,
        optimize=False,
    )
    onnx.checker.check_model(path)
    return {
        "capture_mode": "torch.export strict=False; static profile",
        "eager_to_exported_program": ep_parity,
        **_artifact_record(path),
    }


def _mask_iou(expected: np.ndarray, actual: np.ndarray) -> float:
    expected_binary = expected > 0
    actual_binary = actual > 0
    intersection = np.logical_and(expected_binary, actual_binary).sum(axis=(-2, -1))
    union = np.logical_or(expected_binary, actual_binary).sum(axis=(-2, -1))
    return float(np.mean(np.where(union == 0, 1.0, intersection / union)))


def _reference_parity(
    fixtures: dict[str, Any],
    official_reference: Path,
    local_arrays: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    official = np.load(official_reference)
    records: list[dict[str, Any]] = []
    for case in fixtures["cases"]:
        prefix = case["id"]
        official_indices = official[f"{prefix}__indices"]
        official_scores = official[f"{prefix}__scores"]
        local_scores = local_arrays[f"{prefix}__scores"]
        local_indices = local_arrays[f"{prefix}__indices"]
        records.append(
            {
                "id": prefix,
                "score_max_abs": float(np.max(np.abs(official_scores - local_scores))),
                "admitted_indices_exact_at_0_5": bool(
                    np.array_equal(
                        np.flatnonzero(official_scores > 0.5),
                        np.flatnonzero(local_scores > 0.5),
                    )
                ),
                "top16_exact_indices": bool(
                    np.array_equal(official_indices, local_indices)
                ),
                "top16_overlap": int(len(set(official_indices) & set(local_indices))),
                "box_max_abs_at_official_indices": float(
                    np.max(
                        np.abs(
                            official[f"{prefix}__boxes"]
                            - local_arrays[f"{prefix}__boxes_at_official"]
                        )
                    )
                ),
                "mask_iou_at_official_indices": _mask_iou(
                    official[f"{prefix}__masks"],
                    local_arrays[f"{prefix}__masks_at_official"],
                ),
            }
        )
    return records


def _local_reference_phase(args: argparse.Namespace) -> None:
    work_dir = args.work_dir.resolve()
    fixtures = _fixtures(args.fixtures)
    workspace = args.fixtures.resolve().parents[4]
    checkpoint = resolve_sam3_checkpoint(
        str(args.checkpoint) if args.checkpoint else None
    )
    official = np.load(args.official_reference)
    model = build_production_text_detector(
        add_sam2_neck=False,
        checkpoint_path=str(checkpoint),
        device="cuda",
        dtype="fp16",
        load_weights=True,
    ).eval()
    vision = VisionTowerProfiled(
        VisionTowerScalpedFlat(
            VisionTowerFlat(model.backbone.vision_backbone), model.backbone.scalp
        ),
        "full",
    ).eval()
    text = TextTower(model.backbone.language_backbone).eval()
    full = GroundingFullFlat(
        GroundingFull(
            GroundingEncodeTextOnly(
                GroundingEncode(model.transformer.encoder, model.num_feature_levels),
                TextOnlyPromptEncode(model.geometry_encoder),
            ),
            GroundingDecode(
                model.transformer.decoder,
                model.dot_prod_scoring,
                model.segmentation_head,
            ),
        )
    ).eval()
    tokenizer = SimpleTokenizer()
    image_mask = torch.zeros((1, 72, 72), device="cuda", dtype=torch.bool)
    cached_vision: dict[str, tuple[Tensor, ...]] = {}
    local_arrays: dict[str, np.ndarray] = {}
    with torch.inference_mode():
        for case in fixtures["cases"]:
            image_id = case["image"]
            if image_id not in cached_vision:
                cached_vision[image_id] = vision(
                    _preprocess_image(_image_path(workspace, fixtures, image_id))
                )
            case_vision = cached_vision[image_id]
            case_ids = tokenizer([case["text"]], context_length=TEXT_LENGTH).to("cuda")
            case_text = text(case_ids, case_ids.ne(0))
            logits, boxes, masks, presence = full(
                *case_vision[:3],
                case_vision[5],
                image_mask,
                *case_text,
            )
            scores = (
                logits.sigmoid().squeeze(-1)
                * presence.sigmoid().reshape(presence.shape[0], 1)
            )[0]
            local_indices = torch.argsort(scores, descending=True, stable=True)[:16]
            official_indices_np = official[f"{case['id']}__indices"]
            official_indices = torch.from_numpy(official_indices_np).to("cuda")
            prefix = case["id"]
            local_arrays[f"{prefix}__scores"] = scores.float().cpu().numpy()
            local_arrays[f"{prefix}__indices"] = local_indices.cpu().numpy()
            local_arrays[f"{prefix}__boxes"] = (
                boxes[0, local_indices].float().cpu().numpy()
            )
            local_arrays[f"{prefix}__masks"] = (
                masks[0, local_indices].float().cpu().numpy()
            )
            local_arrays[f"{prefix}__boxes_at_official"] = (
                boxes[0, official_indices].float().cpu().numpy()
            )
            local_arrays[f"{prefix}__masks_at_official"] = (
                masks[0, official_indices].float().cpu().numpy()
            )
    local_reference = work_dir / "local_reference.npz"
    np.savez_compressed(local_reference, **local_arrays)
    metadata = {
        "format": "sam3-m1-local-reference-v1",
        "fixture_version": fixtures["fixture_version"],
        "checkpoint_sha256": _sha256(checkpoint),
        "official_eager_to_local_eager": _reference_parity(
            fixtures, args.official_reference, local_arrays
        ),
        "npz_sha256": _sha256(local_reference),
    }
    (work_dir / "local_reference.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def _export_phase(args: argparse.Namespace) -> None:
    work_dir = args.work_dir.resolve()
    artifact_dir = work_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    fixtures = _fixtures(args.fixtures)
    workspace = args.fixtures.resolve().parents[4]
    checkpoint = resolve_sam3_checkpoint(
        str(args.checkpoint) if args.checkpoint else None
    )
    torch.manual_seed(int(fixtures["measurement"]["seed"]))

    model = build_production_text_detector(
        add_sam2_neck=False,
        checkpoint_path=str(checkpoint),
        device="cuda",
        dtype="fp16",
        load_weights=True,
    ).eval()
    vision_unscalped = VisionTowerFlat(model.backbone.vision_backbone)
    vision_scalped = VisionTowerScalpedFlat(
        vision_unscalped, model.backbone.scalp
    ).eval()
    vision_full = VisionTowerProfiled(vision_scalped, "full").eval()
    vision_required = VisionTowerProfiled(
        vision_scalped, "required-position-only"
    ).eval()
    vision_feature = VisionTowerProfiled(vision_scalped, "feature-only").eval()
    text = TextTower(model.backbone.language_backbone).eval()
    legacy_encoder = GroundingEncode(
        model.transformer.encoder, model.num_feature_levels
    ).eval()
    encoder = GroundingEncodeTextOnly(
        GroundingEncode(model.transformer.encoder, model.num_feature_levels),
        TextOnlyPromptEncode(model.geometry_encoder),
    ).eval()
    decoder = GroundingDecode(
        model.transformer.decoder,
        model.dot_prod_scoring,
        model.segmentation_head,
    ).eval()
    legacy_encoder_flat = GroundingEncodeFlat(legacy_encoder).eval()
    encoder_flat = GroundingEncodeFlat(encoder).eval()
    decoder_flat = GroundingDecodeFlat(decoder).eval()
    full_flat = GroundingFullFlat(GroundingFull(encoder, decoder)).eval()
    full_feature_flat = GroundingFullFeatureOnlyFlat(
        GroundingFullFeatureOnly(
            GroundingFull(encoder, decoder),
            model.backbone.vision_backbone.position_encoding,
        )
    ).eval()
    query_flat = GroundingQueryCoreFlat(
        GroundingQueryCore(model.transformer.decoder, model.dot_prod_scoring)
    ).eval()
    mask_flat = GroundingMaskSelectedKFlat(
        GroundingMaskSelectedK(model.segmentation_head)
    ).eval()

    tokenizer = SimpleTokenizer()
    first_case = fixtures["cases"][0]
    pixels = _preprocess_image(_image_path(workspace, fixtures, first_case["image"]))
    token_ids = tokenizer([first_case["text"]], context_length=TEXT_LENGTH).to("cuda")
    attention_mask = token_ids.ne(0)
    image_mask = torch.zeros((1, 72, 72), device="cuda", dtype=torch.bool)
    with torch.inference_mode():
        vision_outputs = vision_full(pixels)
        text_outputs = text(token_ids, attention_mask)
        legacy_encoder_outputs = legacy_encoder_flat(
            vision_outputs[2],
            vision_outputs[5],
            image_mask,
            *text_outputs,
        )
        encoder_outputs = encoder_flat(
            vision_outputs[2],
            vision_outputs[5],
            image_mask,
            *text_outputs,
        )
        full_args = (
            *vision_outputs[:3],
            vision_outputs[5],
            image_mask,
            *text_outputs,
        )
        _ = full_flat(*full_args)
        feature_full_args = (
            *vision_outputs[:3],
            image_mask,
            *text_outputs,
        )
        query_args = (
            vision_outputs[2],
            *encoder_outputs[:7],
            encoder_outputs[7],
        )
        query_outputs = query_flat(*query_args)
        decoder_args = (
            *vision_outputs[:3],
            *encoder_outputs[:7],
            encoder_outputs[7],
        )

    common_names = [
        "image_feature_0",
        "image_feature_1",
        "image_feature_2",
        "image_pos_2",
        "image_mask_2",
        "text_memory",
        "text_padding_mask",
    ]
    report: dict[str, Any] = {
        "format": "sam3-m1-export-report-v1",
        "fixture_version": fixtures["fixture_version"],
        "checkpoint_sha256": _sha256(checkpoint),
        "sam3_export_commit": _git_revision(Path(__file__).resolve().parents[1]),
        "official_commit": _git_revision(workspace / "sam3"),
        "torch_version": torch.__version__,
        "onnx_version": onnx.__version__,
        "candidates": {},
    }
    report["baseline_exported_program"] = {
        "legacy_grounding_encode": _capture_parity(
            legacy_encoder_flat,
            (
                vision_outputs[2],
                vision_outputs[5],
                image_mask,
                *text_outputs,
            ),
        ),
        "m1_grounding_encode_text_only": _capture_parity(
            encoder_flat,
            (
                vision_outputs[2],
                vision_outputs[5],
                image_mask,
                *text_outputs,
            ),
        ),
        "legacy_grounding_decode": _capture_parity(
            decoder_flat,
            (*vision_outputs[:3], *legacy_encoder_outputs, text_outputs[1]),
        ),
        "m1_grounding_decode_text_only": _capture_parity(
            decoder_flat,
            decoder_args,
        ),
        "vision_full": _capture_parity(vision_full, (pixels,)),
    }
    encoder_output_names = [
        "memory",
        "pos_embed",
        "memory_padding_mask",
        "level_start_index",
        "spatial_shapes",
        "valid_ratios",
        "prompt_memory",
        "prompt_padding_mask",
    ]
    report["candidates"]["grounding_encoder_text_only"] = _export_one(
        encoder_flat,
        (
            vision_outputs[2],
            vision_outputs[5],
            image_mask,
            *text_outputs,
        ),
        artifact_dir / "grounding_encoder_text_only.onnx",
        [
            "image_feature_2",
            "image_pos_2",
            "image_mask_2",
            "text_memory",
            "text_padding_mask",
        ],
        encoder_output_names,
    )
    report["candidates"]["grounding_decoder_text_only"] = _export_one(
        decoder_flat,
        decoder_args,
        artifact_dir / "grounding_decoder_text_only.onnx",
        [
            "image_feature_0",
            "image_feature_1",
            "image_feature_2",
            *encoder_output_names[:7],
            "prompt_padding_mask",
        ],
        list(OUTPUT_NAMES),
    )
    report["candidates"]["grounding_full"] = _export_one(
        full_flat,
        full_args,
        artifact_dir / "grounding_full.onnx",
        common_names,
        list(OUTPUT_NAMES),
    )
    report["candidates"]["vision_required_position_only"] = _export_one(
        vision_required,
        (pixels,),
        artifact_dir / "vision_required_position_only.onnx",
        ["pixel_values"],
        [
            "image_feature_0",
            "image_feature_1",
            "image_feature_2",
            "image_pos_2",
        ],
    )
    report["candidates"]["vision_feature_only"] = _export_one(
        vision_feature,
        (pixels,),
        artifact_dir / "vision_feature_only.onnx",
        ["pixel_values"],
        ["image_feature_0", "image_feature_1", "image_feature_2"],
    )
    report["candidates"]["grounding_full_feature_only"] = _export_one(
        full_feature_flat,
        feature_full_args,
        artifact_dir / "grounding_full_feature_only.onnx",
        [name for name in common_names if name != "image_pos_2"],
        list(OUTPUT_NAMES),
    )
    report["candidates"]["grounding_query_core"] = _export_one(
        query_flat,
        query_args,
        artifact_dir / "grounding_query_core.onnx",
        [
            "image_feature_2",
            "memory",
            "pos_embed",
            "memory_padding_mask",
            "level_start_index",
            "spatial_shapes",
            "valid_ratios",
            "prompt_memory",
            "prompt_padding_mask",
        ],
        ["logits", "boxes_cxcywh", "presence_logits", "query_embeddings"],
    )
    for k in K_PROFILES:
        indices = torch.arange(k, device="cuda", dtype=torch.int64).unsqueeze(0)
        valid = torch.ones((1, k), device="cuda", dtype=torch.bool)
        mask_args = (
            *vision_outputs[:3],
            encoder_outputs[0],
            encoder_outputs[6],
            encoder_outputs[7],
            query_outputs[3],
            indices,
            valid,
        )
        report["candidates"][f"grounding_mask_selected_k{k}"] = _export_one(
            mask_flat,
            mask_args,
            artifact_dir / f"grounding_mask_selected_k{k}.onnx",
            [
                "image_feature_0",
                "image_feature_1",
                "image_feature_2",
                "memory",
                "prompt_memory",
                "prompt_padding_mask",
                "query_embeddings",
                "selected_indices",
                "valid_mask",
            ],
            ["selected_mask_logits"],
        )

    local_reference = work_dir / "local_reference.npz"
    local_metadata = json.loads(
        (work_dir / "local_reference.json").read_text(encoding="utf-8")
    )
    report["official_eager_to_local_eager"] = local_metadata[
        "official_eager_to_local_eager"
    ]
    report["local_reference_sha256"] = _sha256(local_reference)
    (work_dir / "export_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


def _session(path: Path, *, profiling: bool = False) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.enable_profiling = profiling
    session = ort.InferenceSession(
        str(path),
        sess_options=options,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    if "CUDAExecutionProvider" not in session.get_providers():
        raise RuntimeError(f"CUDAExecutionProvider did not load for {path}")
    return session


def _cuda_ortvalue(value: np.ndarray | Tensor) -> ort.OrtValue:
    if isinstance(value, Tensor):
        value = value.detach().cpu().numpy()
    return ort.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(value), "cuda", 0)


def _run_iobound(
    session: ort.InferenceSession, inputs: dict[str, ort.OrtValue]
) -> list[ort.OrtValue]:
    binding = session.io_binding()
    for name, value in inputs.items():
        binding.bind_ortvalue_input(name, value)
    for output in session.get_outputs():
        binding.bind_output(output.name, "cuda", 0)
    session.run_with_iobinding(binding)
    outputs = binding.get_outputs()
    if any(value.device_name() != "cuda" for value in outputs):
        raise RuntimeError("IOBinding produced a non-CUDA output")
    return outputs


def _gpu_memory_mib() -> float:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    pid = str(os.getpid())
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[0] == pid:
            return float(fields[1])
    return 0.0


def _benchmark(
    function: Callable[[], Any], warmup: int, repeats: int
) -> tuple[dict[str, float], Any]:
    last: Any = None
    for _ in range(warmup):
        last = function()
    persistent = _gpu_memory_mib()
    samples: list[float] = []
    stop = threading.Event()

    def sample_memory() -> None:
        while not stop.is_set():
            samples.append(_gpu_memory_mib())
            stop.wait(0.05)

    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    durations: list[float] = []
    try:
        for _ in range(repeats):
            start = time.perf_counter()
            last = function()
            durations.append((time.perf_counter() - start) * 1000.0)
    finally:
        stop.set()
        sampler.join()
    durations.sort()
    p95_index = min(len(durations) - 1, max(0, int(np.ceil(0.95 * len(durations))) - 1))
    return (
        {
            "median_ms": statistics.median(durations),
            "p95_ms": durations[p95_index],
            "persistent_vram_mib": persistent,
            "peak_vram_mib": max(samples, default=persistent),
        },
        last,
    )


def _release(*values: object) -> None:
    del values
    gc.collect()
    time.sleep(0.2)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value.astype(np.float32)))


def _select_fixed_k(
    logits: np.ndarray, presence: np.ndarray, k: int, threshold: float = 0.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores = _sigmoid(logits[..., 0]) * _sigmoid(presence).reshape(logits.shape[0], 1)
    selected = np.zeros((logits.shape[0], k), dtype=np.int64)
    valid = np.zeros((logits.shape[0], k), dtype=np.bool_)
    for batch_index, row in enumerate(scores):
        order = np.lexsort((np.arange(row.size), -row))
        admitted = order[row[order] > threshold][:k]
        selected[batch_index, : admitted.size] = admitted
        valid[batch_index, : admitted.size] = True
    return scores, selected, valid


def _array_parity(
    expected: list[np.ndarray], actual: list[np.ndarray]
) -> dict[str, float]:
    max_abs = 0.0
    for expected_value, actual_value in zip(expected, actual):
        max_abs = max(
            max_abs,
            float(
                np.max(
                    np.abs(
                        expected_value.astype(np.float32)
                        - actual_value.astype(np.float32)
                    )
                )
            ),
        )
    return {"max_abs": max_abs}


def _ort_local_parity(
    fixtures: dict[str, Any],
    local_reference: Path,
    outputs: dict[str, list[np.ndarray]],
) -> list[dict[str, Any]]:
    local = np.load(local_reference)
    records: list[dict[str, Any]] = []
    for case in fixtures["cases"]:
        prefix = case["id"]
        logits, boxes, masks, presence = outputs[prefix]
        scores = (_sigmoid(logits[..., 0]) * _sigmoid(presence).reshape(1, 1))[0]
        indices = local[f"{prefix}__indices"]
        records.append(
            {
                "id": prefix,
                "score_max_abs": float(
                    np.max(np.abs(scores - local[f"{prefix}__scores"]))
                ),
                "box_max_abs_top16": float(
                    np.max(
                        np.abs(
                            boxes[0, indices].astype(np.float32)
                            - local[f"{prefix}__boxes"]
                        )
                    )
                ),
                "mask_iou_top16": _mask_iou(
                    local[f"{prefix}__masks"], masks[0, indices].astype(np.float32)
                ),
            }
        )
    return records


def _environment() -> dict[str, Any]:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total,pstate,power.limit,clocks.current.sm,clocks.current.memory",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "gpu": query,
        "onnxruntime": ort.__version__,
        "providers": ort.get_available_providers(),
        "torch": torch.__version__,
        "python": sys.version,
    }


def _boundary_bytes(outputs: list[ort.OrtValue]) -> int:
    return sum(value.numpy().nbytes for value in outputs)


def _profile_mask_graph(
    path: Path, inputs: dict[str, ort.OrtValue]
) -> dict[str, float]:
    session = _session(path, profiling=True)
    outputs = _run_iobound(session, inputs)
    _ = outputs[0].numpy()
    profile_path = Path(session.end_profiling())
    events = json.loads(profile_path.read_text(encoding="utf-8"))
    totals = {"pixel_decoder_ms": 0.0, "mask_query_projection_and_einsum_ms": 0.0}
    for event in events:
        if event.get("cat") != "Node":
            continue
        name = str(event.get("name", "")).lower()
        duration_ms = float(event.get("dur", 0.0)) / 1000.0
        if "pixel_decoder" in name:
            totals["pixel_decoder_ms"] += duration_ms
        elif any(token in name for token in ("mask_predictor", "mask_embed", "einsum")):
            totals["mask_query_projection_and_einsum_ms"] += duration_ms
    profile_path.unlink(missing_ok=True)
    return totals


def _measure_phase(args: argparse.Namespace) -> None:
    work_dir = args.work_dir.resolve()
    artifact_dir = work_dir / "artifacts"
    legacy_dir = Path(__file__).resolve().parents[1] / "artifacts" / "sam3-onnx"
    fixtures = _fixtures(args.fixtures)
    workspace = args.fixtures.resolve().parents[4]
    warmup = int(fixtures["measurement"]["warmup"])
    repeats = int(fixtures["measurement"]["repeats"])
    tokenizer = SimpleTokenizer()

    pixels: dict[str, ort.OrtValue] = {}
    for image in fixtures["images"]:
        value = _preprocess_image(_image_path(workspace, fixtures, image["id"]))
        pixels[image["id"]] = _cuda_ortvalue(value)
    token_inputs: dict[str, tuple[ort.OrtValue, ort.OrtValue]] = {}
    for case in fixtures["cases"]:
        ids = tokenizer([case["text"]], context_length=TEXT_LENGTH)
        token_inputs[case["id"]] = (_cuda_ortvalue(ids), _cuda_ortvalue(ids.ne(0)))

    text_session = _session(legacy_dir / "text_encoder.onnx")
    text_outputs: dict[str, list[ort.OrtValue]] = {}
    for case in fixtures["cases"]:
        ids, mask = token_inputs[case["id"]]
        text_outputs[case["id"]] = _run_iobound(
            text_session, {"input_ids": ids, "attention_mask": mask}
        )
    del text_session
    gc.collect()

    vision_paths = {
        "full": legacy_dir / "vision_encoder.onnx",
        "required-position-only": artifact_dir / "vision_required_position_only.onnx",
        "feature-only": artifact_dir / "vision_feature_only.onnx",
    }
    vision_outputs: dict[str, dict[str, list[ort.OrtValue]]] = {}
    vision_metrics: dict[str, dict[str, float]] = {}
    representative_image = fixtures["cases"][0]["image"]
    for profile, path in vision_paths.items():
        session = _session(path)
        vision_metrics[profile], _ = _benchmark(
            partial(
                _run_iobound, session, {"pixel_values": pixels[representative_image]}
            ),
            warmup,
            repeats,
        )
        vision_outputs[profile] = {
            image["id"]: _run_iobound(session, {"pixel_values": pixels[image["id"]]})
            for image in fixtures["images"]
        }
        del session
        gc.collect()

    image_mask = _cuda_ortvalue(np.zeros((1, 72, 72), dtype=np.bool_))

    def encoder_inputs(
        case: dict[str, Any], profile: str = "full"
    ) -> dict[str, ort.OrtValue]:
        vision = vision_outputs[profile][case["image"]]
        text = text_outputs[case["id"]]
        pos_index = 5 if profile == "full" else 3
        return {
            "image_feature_2": vision[2],
            "image_pos_2": vision[pos_index],
            "image_mask_2": image_mask,
            "text_memory": text[0],
            "text_padding_mask": text[1],
        }

    def decoder_inputs(
        case: dict[str, Any], encoded: list[ort.OrtValue], profile: str = "full"
    ) -> dict[str, ort.OrtValue]:
        vision = vision_outputs[profile][case["image"]]
        return {
            "image_feature_0": vision[0],
            "image_feature_1": vision[1],
            "image_feature_2": vision[2],
            "memory": encoded[0],
            "pos_embed": encoded[1],
            "memory_padding_mask": encoded[2],
            "level_start_index": encoded[3],
            "spatial_shapes": encoded[4],
            "valid_ratios": encoded[5],
            "prompt_memory": encoded[6],
            "prompt_padding_mask": encoded[7],
        }

    representative = fixtures["cases"][0]
    baseline_encoder = _session(artifact_dir / "grounding_encoder_text_only.onnx")
    baseline_decoder = _session(artifact_dir / "grounding_decoder_text_only.onnx")

    def run_split_case(
        case: dict[str, Any],
        active_encoder: ort.InferenceSession,
        active_decoder: ort.InferenceSession,
    ) -> list[np.ndarray]:
        encoded = _run_iobound(active_encoder, encoder_inputs(case))
        outputs = _run_iobound(active_decoder, decoder_inputs(case, encoded))
        return [output.numpy() for output in outputs]

    split_metric, _ = _benchmark(
        partial(run_split_case, representative, baseline_encoder, baseline_decoder),
        warmup,
        repeats,
    )
    split_outputs = {
        case["id"]: run_split_case(case, baseline_encoder, baseline_decoder)
        for case in fixtures["cases"]
    }
    del baseline_encoder, baseline_decoder
    gc.collect()

    full_session = _session(artifact_dir / "grounding_full.onnx")

    def full_inputs(
        case: dict[str, Any], profile: str = "full"
    ) -> dict[str, ort.OrtValue]:
        vision = vision_outputs[profile][case["image"]]
        text = text_outputs[case["id"]]
        pos_index = 5 if profile == "full" else 3
        return {
            "image_feature_0": vision[0],
            "image_feature_1": vision[1],
            "image_feature_2": vision[2],
            "image_pos_2": vision[pos_index],
            "image_mask_2": image_mask,
            "text_memory": text[0],
            "text_padding_mask": text[1],
        }

    def run_full_case(
        case: dict[str, Any],
        active_session: ort.InferenceSession,
        profile: str = "full",
    ) -> list[np.ndarray]:
        outputs = _run_iobound(active_session, full_inputs(case, profile))
        return [output.numpy() for output in outputs]

    full_metric, _ = _benchmark(
        partial(run_full_case, representative, full_session), warmup, repeats
    )
    full_outputs = {
        case["id"]: run_full_case(case, full_session) for case in fixtures["cases"]
    }
    del full_session
    gc.collect()

    e1_case_parity = {
        case["id"]: _array_parity(split_outputs[case["id"]], full_outputs[case["id"]])
        for case in fixtures["cases"]
    }
    e1 = {
        "fixed_conditions": fixtures["profile"],
        "split": {
            **split_metric,
            "session_launches": 2,
            "copy_bytes": sum(
                value.nbytes for value in split_outputs[representative["id"]]
            ),
            "artifact_size_bytes": _artifact_record(
                artifact_dir / "grounding_encoder_text_only.onnx"
            )["size_bytes"]
            + _artifact_record(artifact_dir / "grounding_decoder_text_only.onnx")[
                "size_bytes"
            ],
        },
        "fused": {
            **full_metric,
            "session_launches": 1,
            "copy_bytes": sum(
                value.nbytes for value in full_outputs[representative["id"]]
            ),
            "artifact_size_bytes": _artifact_record(
                artifact_dir / "grounding_full.onnx"
            )["size_bytes"],
        },
        "parity": e1_case_parity,
        "local_eager_to_ort": _ort_local_parity(
            fixtures, work_dir / "local_reference.npz", full_outputs
        ),
    }

    e2: dict[str, Any] = {}
    e2_outputs: dict[str, dict[str, list[np.ndarray]]] = {"full": full_outputs}
    e2_grounding_metrics: dict[str, dict[str, float]] = {"full": full_metric}
    full_session = _session(artifact_dir / "grounding_full.onnx")
    required_metric, _ = _benchmark(
        partial(run_full_case, representative, full_session, "required-position-only"),
        warmup,
        repeats,
    )
    required_outputs = {
        case["id"]: run_full_case(case, full_session, "required-position-only")
        for case in fixtures["cases"]
    }
    del full_session
    gc.collect()
    e2_outputs["required-position-only"] = required_outputs
    e2_grounding_metrics["required-position-only"] = required_metric

    feature_session = _session(artifact_dir / "grounding_full_feature_only.onnx")

    def feature_inputs(case: dict[str, Any]) -> dict[str, ort.OrtValue]:
        vision = vision_outputs["feature-only"][case["image"]]
        text = text_outputs[case["id"]]
        return {
            "image_feature_0": vision[0],
            "image_feature_1": vision[1],
            "image_feature_2": vision[2],
            "image_mask_2": image_mask,
            "text_memory": text[0],
            "text_padding_mask": text[1],
        }

    def run_feature_case(
        case: dict[str, Any], active_session: ort.InferenceSession
    ) -> list[np.ndarray]:
        outputs = _run_iobound(active_session, feature_inputs(case))
        return [output.numpy() for output in outputs]

    feature_metric, _ = _benchmark(
        partial(run_feature_case, representative, feature_session), warmup, repeats
    )
    feature_outputs = {
        case["id"]: run_feature_case(case, feature_session)
        for case in fixtures["cases"]
    }
    del feature_session
    gc.collect()
    e2_outputs["feature-only"] = feature_outputs
    e2_grounding_metrics["feature-only"] = feature_metric

    for profile in vision_paths:
        representative_outputs = vision_outputs[profile][representative_image]
        boundary_bytes = _boundary_bytes(representative_outputs)
        output_parity = {
            case["id"]: _array_parity(
                full_outputs[case["id"]], e2_outputs[profile][case["id"]]
            )
            for case in fixtures["cases"]
        }
        e2[profile] = {
            "vision": vision_metrics[profile],
            "grounding": e2_grounding_metrics[profile],
            "combined_median_ms": vision_metrics[profile]["median_ms"]
            + e2_grounding_metrics[profile]["median_ms"],
            "combined_p95_ms": vision_metrics[profile]["p95_ms"]
            + e2_grounding_metrics[profile]["p95_ms"],
            "boundary_bytes": boundary_bytes,
            "parity_to_full": output_parity,
            "vision_artifact_size_bytes": _artifact_record(vision_paths[profile])[
                "size_bytes"
            ],
        }

    e3: dict[str, Any] = {
        "all_200": e1["split"],
        "profiles": {},
    }
    for k in K_PROFILES:
        encoder_session = _session(artifact_dir / "grounding_encoder_text_only.onnx")
        query_session = _session(artifact_dir / "grounding_query_core.onnx")
        mask_path = artifact_dir / f"grounding_mask_selected_k{k}.onnx"
        mask_session = _session(mask_path)

        def run_selected_case(
            case: dict[str, Any],
            active_encoder: ort.InferenceSession,
            active_query: ort.InferenceSession,
            active_mask: ort.InferenceSession,
            profile_k: int,
            *,
            retain_device: bool = False,
        ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, list[ort.OrtValue]]:
            encoded = _run_iobound(active_encoder, encoder_inputs(case))
            query = _run_iobound(
                active_query,
                {
                    "image_feature_2": vision_outputs["full"][case["image"]][2],
                    "memory": encoded[0],
                    "pos_embed": encoded[1],
                    "memory_padding_mask": encoded[2],
                    "level_start_index": encoded[3],
                    "spatial_shapes": encoded[4],
                    "valid_ratios": encoded[5],
                    "prompt_memory": encoded[6],
                    "prompt_padding_mask": encoded[7],
                },
            )
            compact = [query[index].numpy() for index in range(3)]
            _scores, selected, valid = _select_fixed_k(
                compact[0], compact[2], profile_k
            )
            if valid.any():
                vision = vision_outputs["full"][case["image"]]
                mask_outputs = _run_iobound(
                    active_mask,
                    {
                        "image_feature_0": vision[0],
                        "image_feature_1": vision[1],
                        "image_feature_2": vision[2],
                        "memory": encoded[0],
                        "prompt_memory": encoded[6],
                        "prompt_padding_mask": encoded[7],
                        "query_embeddings": query[3],
                        "selected_indices": _cuda_ortvalue(selected),
                        "valid_mask": _cuda_ortvalue(valid),
                    },
                )
                masks = mask_outputs[0].numpy()
            else:
                masks = np.zeros((1, profile_k, 288, 288), dtype=np.float16)
            device_values = encoded + query if retain_device else []
            return compact + [masks], selected, valid, device_values

        selected_metric, _ = _benchmark(
            partial(
                run_selected_case,
                representative,
                encoder_session,
                query_session,
                mask_session,
                k,
            ),
            warmup,
            repeats,
        )
        selected_results = {
            case["id"]: run_selected_case(
                case, encoder_session, query_session, mask_session, k
            )
            for case in fixtures["cases"]
        }
        parity_records: list[dict[str, Any]] = []
        skipped_cases = 0
        for case in fixtures["cases"]:
            compact, selected, valid, _device = selected_results[case["id"]]
            full = split_outputs[case["id"]]
            valid_count = int(valid.sum())
            if valid_count == 0:
                skipped_cases += 1
                mask_iou = 1.0
                mask_max_abs = 0.0
            else:
                indices = selected[0, :valid_count]
                expected_masks = full[2][0, indices]
                actual_masks = compact[3][0, :valid_count]
                mask_iou = _mask_iou(expected_masks, actual_masks)
                mask_max_abs = float(
                    np.max(
                        np.abs(
                            expected_masks.astype(np.float32)
                            - actual_masks.astype(np.float32)
                        )
                    )
                )
            parity_records.append(
                {
                    "id": case["id"],
                    "query_max_abs": _array_parity(
                        [full[0], full[1], full[3]], compact[:3]
                    )["max_abs"],
                    "valid_count": valid_count,
                    "selected_mask_max_abs": mask_max_abs,
                    "selected_mask_iou": mask_iou,
                }
            )

        compact, selected, valid, device_values = run_selected_case(
            representative,
            encoder_session,
            query_session,
            mask_session,
            k,
            retain_device=True,
        )
        encoded = device_values[:8]
        query = device_values[8:]
        vision = vision_outputs["full"][representative["image"]]
        mask_inputs = {
            "image_feature_0": vision[0],
            "image_feature_1": vision[1],
            "image_feature_2": vision[2],
            "memory": encoded[0],
            "prompt_memory": encoded[6],
            "prompt_padding_mask": encoded[7],
            "query_embeddings": query[3],
            "selected_indices": _cuda_ortvalue(selected),
            "valid_mask": _cuda_ortvalue(valid),
        }
        operator_profile = _profile_mask_graph(mask_path, mask_inputs)
        mask_value = _run_iobound(mask_session, mask_inputs)[0]
        d2h_metric, _ = _benchmark(lambda: mask_value.numpy(), warmup, repeats)
        d2h_bytes = sum(value.nbytes for value in compact[:3]) + compact[3].nbytes
        h2d_bytes = selected.nbytes + valid.nbytes
        e3["profiles"][f"K={k}"] = {
            **selected_metric,
            "session_launches_with_proposals": 3,
            "session_launches_zero_proposals": 2,
            "copy_bytes": {"d2h": d2h_bytes, "h2d": h2d_bytes},
            "artifact_size_bytes": _artifact_record(
                artifact_dir / "grounding_encoder_text_only.onnx"
            )["size_bytes"]
            + _artifact_record(artifact_dir / "grounding_query_core.onnx")["size_bytes"]
            + _artifact_record(mask_path)["size_bytes"],
            "parity": parity_records,
            "zero_proposal_cases": skipped_cases,
            "mask_operator_profile": operator_profile,
            "mask_d2h": d2h_metric,
        }
        del encoder_session, query_session, mask_session
        gc.collect()

    report = {
        "format": "sam3-m1-measurement-report-v1",
        "fixture_version": fixtures["fixture_version"],
        "measurement": fixtures["measurement"],
        "environment": _environment(),
        "commit": _git_revision(Path(__file__).resolve().parents[1]),
        "checkpoint_sha256": fixtures["checkpoint_sha256"],
        "E1": e1,
        "E2": e2,
        "E3": e3,
    }
    (work_dir / "measurement_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


def _run_phase(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _all_phase(args: argparse.Namespace) -> None:
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = resolve_sam3_checkpoint(
        str(args.checkpoint) if args.checkpoint else None
    )
    official_reference = work_dir / "official_reference.npz"
    helper = Path(__file__).with_name("m1_official_reference.py")
    common = [
        "--work-dir",
        str(work_dir),
        "--fixtures",
        str(args.fixtures.resolve()),
        "--official-root",
        str(args.official_root.resolve()),
        "--checkpoint",
        str(checkpoint),
        "--official-reference",
        str(official_reference),
    ]
    _run_phase(
        [
            sys.executable,
            str(helper),
            "--official-root",
            str(args.official_root.resolve()),
            "--checkpoint",
            str(checkpoint),
            "--fixtures",
            str(args.fixtures.resolve()),
            "--output",
            str(official_reference),
        ]
    )
    _run_phase(
        [sys.executable, str(Path(__file__).resolve()), "local-reference", *common]
    )
    _run_phase([sys.executable, str(Path(__file__).resolve()), "export", *common])
    _run_phase([sys.executable, str(Path(__file__).resolve()), "measure", *common])
    print(work_dir / "export_report.json")
    print(work_dir / "measurement_report.json")


def _parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    workspace = repo_root.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=("all", "local-reference", "export", "measure")
    )
    parser.add_argument("--work-dir", type=Path, default=repo_root / ".m1-work")
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=repo_root / "tests" / "fixtures" / "m1_image_pcs" / "cases.json",
    )
    parser.add_argument("--official-root", type=Path, default=workspace / "sam3")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--official-reference", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    if args.phase == "all":
        _all_phase(args)
    elif args.phase == "local-reference":
        if args.official_reference is None:
            raise SystemExit("local-reference requires --official-reference")
        _local_reference_phase(args)
    elif args.phase == "export":
        if args.official_reference is None:
            raise SystemExit("export requires --official-reference")
        _export_phase(args)
    else:
        _measure_phase(args)


if __name__ == "__main__":
    main()
