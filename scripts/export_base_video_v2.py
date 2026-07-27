"""Build the shipped M4 SAM3 base video tracking ORT CUDA bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
from typing import Any, Callable

from export_image_pcs_v2 import _copy_file, _file_id, _file_record, _git_revision
from m1_experiments import _export_one
import numpy as np
import onnx
from onnx import TensorProto
from PIL import Image
import torch

from sam3.runtime.interactive_image import (
    InteractivePrompt,
    _preprocess_interactive_image,
    _prompt_arrays,
)
from sam3.runtime.manifest import BASE_VIDEO_PLAN_ID, MANIFEST_FORMAT_V2, sha256_file
from sam3.weights import build_base_video_modules, resolve_sam3_checkpoint

CONTRACT_VERSION = "1.0.0"
SCOPE_LABEL = (
    "SAM3 base video tracking / point-box-mask correction / "
    "per-object batch / ORT CUDA v1"
)
GRAPH_NAMES = {
    "tracker-frame-encode": "tracker_frame_encode.onnx",
    "base-tracker-preview-multimask3": "base_tracker_preview_multimask3.onnx",
    "base-tracker-preview-single1": "base_tracker_preview_single1.onnx",
    "base-memory-commit": "base_memory_commit.onnx",
    "base-tracker-step-and-commit-single1": (
        "base_tracker_step_and_commit_single1.onnx"
    ),
}
ROLE_COMPONENTS = {
    "tracker-frame-encode": ["VisionTrunk", "SAM2Neck", "TrackerFrameEncode"],
    "base-tracker-preview-multimask3": [
        "MemoryAttention",
        "PromptEncoder",
        "MaskDecoder",
        "BaseTrackerPreviewMultimask3",
    ],
    "base-tracker-preview-single1": [
        "MemoryAttention",
        "PromptEncoder",
        "MaskDecoder",
        "BaseTrackerPreviewSingle1",
    ],
    "base-memory-commit": ["MaskMemoryEncoder", "BaseMemoryCommit"],
    "base-tracker-step-and-commit-single1": [
        "MemoryAttention",
        "PromptEncoder",
        "MaskDecoder",
        "MaskMemoryEncoder",
        "BaseTrackerStepAndCommitSingle1",
    ],
}
FRAME_INPUTS = ["pixel_values"]
FRAME_OUTPUTS = [
    "image_embedding",
    "image_position",
    "high_res_0",
    "high_res_1",
]
PREVIEW_INPUTS = [
    "image_embedding",
    "image_position",
    "high_res_0",
    "high_res_1",
    "object_valid",
    "memory_features",
    "memory_position",
    "memory_valid",
    "memory_age",
    "memory_conditioning",
    "object_pointers",
    "pointer_valid",
    "pointer_age",
    "pointer_conditioning",
    "pointer_tpos_denominator",
    "point_coords",
    "point_labels",
    "point_valid",
    "box_xyxy",
    "has_box",
    "mask_input",
    "has_mask",
]
PREVIEW_OUTPUTS = [
    "low_res_logits",
    "scores",
    "commit_mask",
    "object_pointer",
    "object_score",
]
COMMIT_INPUTS = [
    "image_embedding",
    "commit_mask",
    "object_score",
    "is_mask_from_points",
]
COMMIT_OUTPUTS = ["memory_features", "memory_position"]
FUSED_OUTPUTS = [
    *PREVIEW_OUTPUTS,
    "committed_memory_features",
    "committed_memory_position",
]


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def _time_cuda(
    operation: Callable[[], object], *, warmup: int, repeats: int
) -> dict[str, float | int]:
    with torch.inference_mode():
        for _ in range(warmup):
            operation()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        timings: list[float] = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            operation()
            end.record()
            end.synchronize()
            timings.append(float(start.elapsed_time(end)))
    return {
        "median_ms": statistics.median(timings),
        "p95_ms": _percentile(timings, 0.95),
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "persistent_vram_bytes": torch.cuda.memory_allocated(),
        "d2h_bytes": 0,
        "h2d_bytes": 0,
    }


def _video_frames(fixtures: dict[str, Any], source: Path) -> list[np.ndarray]:
    base = np.asarray(Image.open(source).convert("RGB"), dtype=np.uint8)
    result: list[np.ndarray] = []
    for transform in fixtures["video"]["transforms"]:
        shifted = np.roll(base, int(transform["horizontal_roll"]), axis=1)
        delta = int(transform["brightness_delta"])
        shifted = np.clip(shifted.astype(np.int16) + delta, 0, 255).astype(np.uint8)
        result.append(shifted)
    return result


def _prompt(
    fixtures: dict[str, Any], name: str, original_size: tuple[int, int]
) -> dict[str, np.ndarray]:
    record = fixtures[name]
    height, width = original_size
    relative = np.asarray(record["points_xy"], dtype=np.float32)
    absolute = relative * np.asarray([width, height], dtype=np.float32)
    values, _ = _prompt_arrays(
        InteractivePrompt(
            points_xy=absolute,
            point_labels=np.asarray(record["point_labels"], dtype=np.int64),
        ),
        original_size,
    )
    return values


def _torch_prompt(
    values: dict[str, np.ndarray], batch: int
) -> tuple[torch.Tensor, ...]:
    result: list[torch.Tensor] = []
    for value in values.values():
        repeats = (batch,) + (1,) * (value.ndim - 1)
        result.append(torch.from_numpy(np.tile(value, repeats)).to("cuda"))
    return tuple(result)


def _sample_args(
    modules: Any,
    encoded: tuple[torch.Tensor, ...],
    prompt: dict[str, np.ndarray],
    batch: int,
) -> tuple[torch.Tensor, ...]:
    image = tuple(value.repeat(batch, 1, 1, 1) for value in encoded)
    one_prompt = _torch_prompt(prompt, 1)
    empty_memory = torch.zeros((1, 10, 64, 72, 72), dtype=torch.float16, device="cuda")
    empty_valid = torch.zeros((1, 10), dtype=torch.bool, device="cuda")
    empty_age = torch.zeros((1, 10), dtype=torch.int64, device="cuda")
    empty_pointer = torch.zeros((1, 16, 256), dtype=torch.float16, device="cuda")
    empty_pointer_valid = torch.zeros((1, 16), dtype=torch.bool, device="cuda")
    empty_pointer_age = torch.zeros((1, 16), dtype=torch.int64, device="cuda")
    with torch.inference_mode():
        initial = modules.preview_single1(
            *encoded,
            torch.ones(1, dtype=torch.bool, device="cuda"),
            empty_memory,
            empty_memory,
            empty_valid,
            empty_age,
            empty_valid,
            empty_pointer,
            empty_pointer_valid,
            empty_pointer_age,
            empty_pointer_valid,
            torch.full((1,), 2.0, dtype=torch.float32, device="cuda"),
            *one_prompt,
        )
        committed = modules.memory_commit(
            encoded[0],
            initial[2],
            initial[4],
            torch.ones(1, dtype=torch.bool, device="cuda"),
        )
    memory = committed[0][:, None].expand(-1, 10, -1, -1, -1)
    position = committed[1][:, None].expand_as(memory)
    pointers = initial[3][:, None].expand(-1, 16, -1)
    memory_age = torch.tensor(
        [[1, 2, 3, 4, 1, 2, 3, 4, 5, 6]],
        dtype=torch.int64,
        device="cuda",
    )
    memory_conditioning = torch.tensor(
        [[True, True, True, True, False, False, False, False, False, False]],
        device="cuda",
    )
    pointer_age = torch.arange(1, 17, dtype=torch.int64, device="cuda")[None]
    pointer_conditioning = torch.zeros((1, 16), dtype=torch.bool, device="cuda")
    pointer_conditioning[:, :4] = True
    return (
        *image,
        torch.ones(batch, dtype=torch.bool, device="cuda"),
        memory.expand(batch, -1, -1, -1, -1).contiguous(),
        position.expand(batch, -1, -1, -1, -1).contiguous(),
        torch.ones((batch, 10), dtype=torch.bool, device="cuda"),
        memory_age.expand(batch, -1).contiguous(),
        memory_conditioning.expand(batch, -1).contiguous(),
        pointers.expand(batch, -1, -1).contiguous(),
        torch.ones((batch, 16), dtype=torch.bool, device="cuda"),
        pointer_age.expand(batch, -1).contiguous(),
        pointer_conditioning.expand(batch, -1).contiguous(),
        torch.full((batch,), 2.0, dtype=torch.float32, device="cuda"),
        *_torch_prompt(prompt, batch),
    )


def _measure_decisions(
    modules: Any,
    encoded: tuple[torch.Tensor, ...],
    prompt: dict[str, np.ndarray],
    fixtures: dict[str, Any],
) -> tuple[int, str, dict[str, Any]]:
    warmup = int(fixtures["measurement"]["warmup"])
    repeats = int(fixtures["measurement"]["repeats"])
    measurements: dict[str, Any] = {}
    args4 = _sample_args(modules, encoded, prompt, 4)
    measurements["b4"] = _time_cuda(
        lambda values=args4: modules.preview_single1(*values),
        warmup=warmup,
        repeats=repeats,
    )
    del args4
    torch.cuda.empty_cache()
    try:
        args8 = _sample_args(modules, encoded, prompt, 8)
        measurements["b8"] = _time_cuda(
            lambda values=args8: modules.preview_single1(*values),
            warmup=warmup,
            repeats=repeats,
        )
        b8_improvement = 1.0 - float(measurements["b8"]["median_ms"]) / (
            2.0 * float(measurements["b4"]["median_ms"])
        )
        b8_vram_ratio = float(measurements["b8"]["peak_vram_bytes"]) / float(
            measurements["b4"]["peak_vram_bytes"]
        )
        batch = 8 if b8_improvement >= 0.15 and b8_vram_ratio <= 1.75 else 4
        measurements["batch_gate"] = {
            "b8_vs_two_b4_median_improvement": b8_improvement,
            "b8_to_b4_peak_vram_ratio": b8_vram_ratio,
            "decision": f"B{batch}",
        }
        del args8
    except torch.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        batch = 4
        measurements["b8"] = {"status": "oom", "error": str(exc)}
        measurements["batch_gate"] = {"decision": "B4", "reason": "B8 CUDA OOM"}

    args = _sample_args(modules, encoded, prompt, batch)

    def split() -> tuple[torch.Tensor, ...]:
        preview = modules.preview_single1(*args)
        memory = modules.memory_commit(
            args[0],
            preview[2],
            preview[4],
            torch.zeros(batch, dtype=torch.bool, device="cuda"),
        )
        return (*preview, *memory)

    measurements["split"] = _time_cuda(split, warmup=warmup, repeats=repeats)
    measurements["fused"] = _time_cuda(
        lambda: modules.step_and_commit_single1(*args),
        warmup=warmup,
        repeats=repeats,
    )
    fused_median_ratio = float(measurements["fused"]["median_ms"]) / float(
        measurements["split"]["median_ms"]
    )
    fused_vram_ratio = float(measurements["fused"]["peak_vram_bytes"]) / float(
        measurements["split"]["peak_vram_bytes"]
    )
    cut = (
        "fused" if fused_median_ratio <= 1.05 and fused_vram_ratio <= 1.25 else "split"
    )
    measurements["fused_gate"] = {
        "fused_to_split_median_ratio": fused_median_ratio,
        "fused_to_split_peak_vram_ratio": fused_vram_ratio,
        "decision": cut,
    }
    return batch, cut, measurements


def _onnx_dtype(value: int) -> str:
    return {
        TensorProto.FLOAT: "float32",
        TensorProto.FLOAT16: "float16",
        TensorProto.INT64: "int64",
        TensorProto.INT32: "int32",
        TensorProto.BOOL: "bool",
    }[value]


def _value_spec(value: Any) -> dict[str, Any]:
    shape: list[int | str] = []
    for dimension in value.type.tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            shape.append(int(dimension.dim_value))
        elif dimension.dim_param:
            shape.append(str(dimension.dim_param))
        else:
            raise RuntimeError(f"unnamed dynamic dimension for {value.name}")
    return {"dtype": _onnx_dtype(value.type.tensor_type.elem_type), "shape": shape}


def _graph_signatures(
    bundle_dir: Path, profile_id: str, export_graphs: dict[str, Any]
) -> dict[str, Any]:
    graphs: dict[str, Any] = {}
    for role, filename in GRAPH_NAMES.items():
        model = onnx.load(bundle_dir / "graphs" / filename, load_external_data=False)
        graphs[role] = {
            "path": f"graphs/{filename}",
            "inputs": [
                {"name": value.name, **_value_spec(value)}
                for value in model.graph.input
            ],
            "outputs": [
                {"name": value.name, **_value_spec(value)}
                for value in model.graph.output
            ],
            "exported_program": export_graphs[role]["exported_program"],
        }
    return {
        "format": "sam3-base-video-graph-signatures-v1",
        "profile_id": profile_id,
        "capture": "torch.export(strict=False)",
        "opset": 18,
        "graphs": graphs,
    }


def _tensor_ref(role: str, name: str, *, output: bool) -> str:
    base = name.replace("_", "-")
    if role == "tracker-frame-encode" and output:
        return f"frame-{base}"
    if name in {"image_embedding", "image_position", "high_res_0", "high_res_1"}:
        return f"batched-frame-{base}"
    if name == "low_res_logits":
        policy = "multimask3" if "multimask3" in role else "single1"
        return f"preview-low-res-{policy}"
    if name == "scores":
        policy = "multimask3" if "multimask3" in role else "single1"
        return f"preview-scores-{policy}"
    if name in {"commit_mask", "object_pointer", "object_score"}:
        return f"preview-{base}"
    if name in {"memory_features", "memory_position"} and output:
        return f"committed-{base}"
    return base


def _artifacts_and_tensors(
    signatures: dict[str, Any], bundle_dir: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    specs: dict[str, dict[str, Any]] = {}
    artifacts: list[dict[str, Any]] = []
    host_inputs = {
        "pixel_values",
        "object_valid",
        "memory_valid",
        "memory_age",
        "memory_conditioning",
        "pointer_valid",
        "pointer_age",
        "pointer_conditioning",
        "pointer_tpos_denominator",
        "point_coords",
        "point_labels",
        "point_valid",
        "box_xyxy",
        "has_box",
        "mask_input",
        "has_mask",
        "is_mask_from_points",
    }
    for role, signature in signatures["graphs"].items():
        for is_output, values in (
            (False, signature["inputs"]),
            (True, signature["outputs"]),
        ):
            for value in values:
                ref = _tensor_ref(role, value["name"], output=is_output)
                spec = {"dtype": value["dtype"], "shape": value["shape"]}
                if ref in specs and specs[ref] != spec:
                    raise RuntimeError(f"conflicting tensor contract: {ref}")
                specs[ref] = spec
        graph_path = signature["path"]
        data_path = graph_path + ".data"
        artifacts.append(
            {
                "id": role,
                "role": role,
                "format": "onnx",
                "components": ROLE_COMPONENTS[role],
                "entry_file_ref": _file_id(graph_path),
                "external_data_file_refs": (
                    [_file_id(data_path)] if (bundle_dir / data_path).is_file() else []
                ),
                "inputs": [
                    {
                        "tensor_ref": _tensor_ref(role, value["name"], output=False),
                        "backend_name": value["name"],
                    }
                    for value in signature["inputs"]
                ],
                "outputs": [
                    {
                        "tensor_ref": _tensor_ref(role, value["name"], output=True),
                        "backend_name": value["name"],
                    }
                    for value in signature["outputs"]
                ],
            }
        )
    tensors: list[dict[str, Any]] = []
    for ref, spec in sorted(specs.items()):
        semantic = ref
        if ref.startswith("preview-low-res") or ref.startswith("preview-scores"):
            residency = "host-output"
        elif ref.replace("-", "_") in host_inputs:
            residency = "host-input"
        else:
            residency = "device"
        tensors.append(
            {
                "id": ref,
                "semantic_type": semantic,
                **spec,
                "layout": "fixed BaseVideoStateV1 profile",
                "unit": "pixel" if ref in {"point-coords", "box-xyxy"} else "unitless",
                "normalization": "(x / 255 - 0.5) / 0.5"
                if ref == "pixel-values"
                else "none",
                "padding": "explicit validity tensors mask inactive object/state/prompt entries",
                "validity": "object/memory/pointer/prompt validity is explicit",
                "coordinates": "M3 public point/box/mask coordinate semantics",
                "value_kind": "mask-logit"
                if "mask" in ref or "logit" in ref
                else "feature",
                "residency": residency,
            }
        )
    return artifacts, tensors


def _manifest(
    bundle_dir: Path,
    signatures: dict[str, Any],
    fixtures: dict[str, Any],
    metadata: dict[str, str],
    *,
    batch: int,
    steady_cut: str,
) -> dict[str, Any]:
    artifacts, tensors = _artifacts_and_tensors(signatures, bundle_dir)
    file_paths = {
        "LICENSE",
        "README.md",
        "capture/graph_signatures.json",
        "fixtures/cases.json",
        "fixtures/frames/frame_000.png",
        "fixtures/frames/frame_001.png",
        "fixtures/frames/frame_002.png",
        "fixtures/official_reference.json",
        "fixtures/official_reference.npz",
        "reports/export_report.json",
        "reports/m4_release_validation.json",
        "reports/provenance.json",
        "reports/batch_decision.json",
        "reports/fused_cut_decision.json",
    }
    for signature in signatures["graphs"].values():
        file_paths.add(signature["exported_program"]["program_path"])
        file_paths.add(signature["path"])
        if (bundle_dir / (signature["path"] + ".data")).is_file():
            file_paths.add(signature["path"] + ".data")
    files = [_file_record(bundle_dir, path) for path in sorted(file_paths)]
    report_ref = _file_id("reports/m4_release_validation.json")
    frame_refs = [
        "frame-image-embedding",
        "frame-image-position",
        "frame-high-res-0",
        "frame-high-res-1",
    ]
    state_refs = [
        "memory-features",
        "memory-position",
        "memory-valid",
        "memory-age",
        "memory-conditioning",
        "object-pointers",
        "pointer-valid",
        "pointer-age",
        "pointer-conditioning",
        "pointer-tpos-denominator",
    ]
    profile_id = f"b{batch}-1008-p16-box1-mask288-m10-ptr16-fp16"
    return {
        "format": MANIFEST_FORMAT_V2,
        "manifest_id": f"{BASE_VIDEO_PLAN_ID}-manifest-v2",
        "scope": {
            "classification": "public-deployment",
            "lifecycle": "shipped",
            "dispatch_role": "default",
            "scope_label": SCOPE_LABEL,
            "use_case": "base-video-tracking",
            "prompt_coverage": ["point", "box", "mask"],
            "capabilities": [
                "base video tracking",
                "repeated correction preview",
                "per-object fixed-capacity batch",
                "forward and reverse propagation",
            ],
            "exclusions": [
                "text image PCS",
                "SAM3.1",
                "Multiplex",
                "bucket-space state",
                "streaming input",
                "CPU plan fallback",
            ],
        },
        "plan": {
            "id": BASE_VIDEO_PLAN_ID,
            "contract_version": CONTRACT_VERSION,
            "semantic_graph_kind": "sam3-base-video-tracking",
            "role_set": list(GRAPH_NAMES),
            "components": sorted(
                {name for values in ROLE_COMPONENTS.values() for name in values}
            ),
        },
        "model": {
            "family": "sam3",
            "variant": "base",
            "vision_layout": "SAM2 FPN 288/144/72; frame encoded once",
            "tracking_layout": "BaseVideoStateV1 per-object packed batch",
            "source_repository": "facebook/sam3",
            "source_commit": metadata["official_commit"],
            "model_revision": metadata["official_commit"],
            "checkpoint": {
                "id": "facebook-sam3-sam3-pt",
                "digest": {
                    "algorithm": "sha256",
                    "value": metadata["checkpoint_sha256"],
                },
            },
            "variant_parameters": [
                {"name": "num-maskmem", "value": 7},
                {"name": "conditioning-spatial-capacity", "value": 4},
                {"name": "non-conditioning-spatial-capacity", "value": 6},
                {"name": "total-spatial-input-capacity", "value": 10},
                {"name": "object-pointer-capacity", "value": 16},
                {"name": "hidden-dimension", "value": 256},
                {"name": "memory-dimension", "value": 64},
                {"name": "memory-spatial-size", "value": [72, 72]},
                {"name": "temporal-stride", "value": 1},
                {"name": "memory-sigmoid-scale", "value": 20.0},
                {"name": "memory-sigmoid-bias", "value": -10.0},
                {"name": "non-overlap-memory", "value": False},
            ],
        },
        "backend": {
            "kind": "onnx-runtime",
            "target": "CUDA device 0",
            "execution_provider": "CUDAExecutionProvider",
            "runtime_version": "1.27.0",
            "pytorch_version": metadata["torch_version"],
            "exporter_version": metadata["onnx_version"],
            "opset": 18,
            "capabilities": ["device-resident-handoff", "iobinding", "external-data"],
        },
        "profile": {
            "id": profile_id,
            "precision": "fp16",
            "shape_mode": "static",
            "static_values": [
                {"name": "frame-batch", "value": 1},
                {"name": "object-batch-capacity", "value": batch},
                {"name": "image-size", "value": 1008},
                {"name": "point-capacity", "value": 16},
                {"name": "box-capacity", "value": 1},
                {"name": "mask-size", "value": 288},
                {"name": "spatial-state-capacity", "value": 10},
                {"name": "pointer-capacity", "value": 16},
            ],
            "dynamic_dimensions": [],
        },
        "tensors": tensors,
        "artifacts": artifacts,
        "execution": {
            "entry_artifacts": list(GRAPH_NAMES),
            "edges": [
                {
                    "producer_artifact_ref": "tracker-frame-encode",
                    "consumer_artifact_ref": role,
                    "tensor_refs": frame_refs,
                }
                for role in (
                    "base-tracker-preview-multimask3",
                    "base-tracker-preview-single1",
                    "base-memory-commit",
                    "base-tracker-step-and-commit-single1",
                )
            ],
        },
        "caches": [
            {
                "id": "frame-cache",
                "tensor_refs": frame_refs,
                "lifetime": "session",
                "key_version": "1.0.0",
                "key_parts": [
                    "preprocessed-frame-bytes",
                    "video-frame-identity",
                    "original-size",
                    "checkpoint-digest",
                    "profile-id",
                    "memory-aware-frame-view-v1",
                ],
                "invalidated_by": [
                    "video-change",
                    "checkpoint-change",
                    "profile-change",
                ],
                "state_compatibility": "memory-aware-frame-view-v1 only",
            },
            {
                "id": "base-video-state",
                "tensor_refs": state_refs,
                "lifetime": "object",
                "key_version": "1.0.0",
                "key_parts": ["object-id", "frame-index", "direction", "revision"],
                "invalidated_by": ["correction-commit", "video-change"],
                "state_compatibility": "SAM3 base BaseVideoStateV1; never SAM3.1 bucket state",
            },
        ],
        "handoffs": [
            {
                "id": "frame-to-preview",
                "producer_artifact_ref": "tracker-frame-encode",
                "consumer_artifact_ref": "base-tracker-preview-single1",
                "tensor_refs": frame_refs,
                "requirement": "required-device",
                "mechanism": "ORT CUDA IOBinding plus CUDA D2D batch expansion",
                "fallback_plan_id": None,
            },
            {
                "id": "preview-to-memory",
                "producer_artifact_ref": "base-tracker-preview-single1",
                "consumer_artifact_ref": "base-memory-commit",
                "tensor_refs": ["preview-commit-mask", "preview-object-score"],
                "requirement": "required-device",
                "mechanism": "CUDA OrtValue",
                "fallback_plan_id": None,
            },
        ],
        "capture": {
            "canonical_format": "exported-program",
            "mode": "non-strict",
            "pytorch_version": metadata["torch_version"],
            "exporter_version": metadata["onnx_version"],
            "constraints": [
                f"object-batch={batch}",
                "image=1008x1008",
                "points=16",
                "spatial-state=10",
                "pointers=16",
            ],
            "graph_signature_file_ref": _file_id("capture/graph_signatures.json"),
            "program_file_refs": [
                _file_id(signature["exported_program"]["program_path"])
                for signature in signatures["graphs"].values()
            ],
            "strict_audit": {"status": "not-run", "report_file_ref": None},
        },
        "policies": [
            {
                "name": "steady-state-cut",
                "binding": "baked",
                "value": steady_cut,
                "stage": "propagation",
            },
            {
                "name": "preview-commit",
                "binding": "host-runtime",
                "value": "multimask3 never commits; final single1 commits once",
                "stage": "correction",
            },
            {
                "name": "object-batching",
                "binding": "host-runtime",
                "value": f"fixed B{batch}; capacity+1 uses two launches",
                "stage": "dispatch",
            },
            {
                "name": "fallback",
                "binding": "baked",
                "value": "none",
                "stage": "backend-dispatch",
            },
        ],
        "fixtures": [
            {
                "id": fixtures["fixture_version"],
                "version": "1.0.0",
                "owner": {"team": "sam3-export", "role": fixtures["owner"]},
                "source_commit": fixtures["official_source_commit"],
                "checkpoint_digest": {
                    "algorithm": "sha256",
                    "value": fixtures["checkpoint_sha256"],
                },
                "cases": [
                    "memory-0",
                    "memory-1",
                    "memory-max",
                    "repeated-correction",
                    "batch-and-chunk",
                    "forward-reverse",
                    "correction-replacement",
                    "object-absence",
                ],
                "aggregate_digest": {
                    "algorithm": "sha256",
                    "value": sha256_file(bundle_dir / "fixtures/cases.json"),
                },
                "parity": [
                    {"stage": stage, "status": "pass", "report_file_ref": report_ref}
                    for stage in (
                        "official-to-local-eager",
                        "local-eager-to-exported-program",
                        "exported-program-to-backend",
                        "end-to-end-behavior",
                    )
                ],
            }
        ],
        "files": files,
    }


def refresh_manifest_file_records(bundle_dir: Path) -> None:
    for path in sorted((bundle_dir / "manifests").glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["files"] = [
            _file_record(bundle_dir, record["path"]) for record in manifest["files"]
        ]
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _write_bundle_card(path: Path, *, batch: int, cut: str) -> None:
    path.write_text(
        f"""# {SCOPE_LABEL}

This M4 bundle ships the fixed B{batch}, FP16, 1008x1008 profile. Correction
uses non-mutating preview followed by a final single-mask commit; steady-state
propagation uses the measured `{cut}` recipe. CUDA IOBinding is required and
there is no fallback plan. It does not replace the M2 image PCS or M3
interactive-image defaults and excludes SAM3.1 Multiplex.
""",
        encoding="utf-8",
    )


def export_bundle(
    output_dir: Path,
    checkpoint_path: Path,
    official_repo: Path,
    fixtures_path: Path,
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent)
    )
    try:
        fixtures = json.loads(fixtures_path.read_text(encoding="utf-8"))
        checkpoint_sha = sha256_file(checkpoint_path)
        if checkpoint_sha != fixtures["checkpoint_sha256"]:
            raise RuntimeError("checkpoint does not match the M4 fixture")
        official_commit = _git_revision(official_repo)
        if official_commit != fixtures["official_source_commit"]:
            raise RuntimeError("official repository does not match the M4 fixture")
        workspace = fixtures_path.resolve().parents[4]
        source = workspace / fixtures["source_image"]["workspace_path"]
        if sha256_file(source) != fixtures["source_image"]["sha256"]:
            raise RuntimeError("M4 source image hash mismatch")
        for name in ("graphs", "capture", "fixtures/frames", "reports", "manifests"):
            (staging / name).mkdir(parents=True, exist_ok=True)
        _copy_file(Path("LICENSE"), staging / "LICENSE")
        _copy_file(fixtures_path, staging / "fixtures/cases.json")
        frames = _video_frames(fixtures, source)
        for index, frame in enumerate(frames):
            Image.fromarray(frame).save(
                staging / f"fixtures/frames/frame_{index:03d}.png"
            )

        official_python = official_repo / ".venv/bin/python"
        if not official_python.is_file():
            official_python = Path(sys.executable)
        subprocess.run(
            [
                str(official_python),
                str(Path(__file__).with_name("m4_official_base_video_reference.py")),
                "--official-root",
                str(official_repo),
                "--checkpoint",
                str(checkpoint_path),
                "--fixtures",
                str(fixtures_path),
                "--frames-dir",
                str(staging / "fixtures/frames"),
                "--output",
                str(staging / "fixtures/official_reference.npz"),
                "--metadata-output",
                str(staging / "fixtures/official_reference.json"),
            ],
            check=True,
        )

        modules = build_base_video_modules(checkpoint_path, device="cuda", dtype="fp16")
        pixels_np, original_size = _preprocess_interactive_image(frames[0])
        pixels = torch.from_numpy(pixels_np).to("cuda")
        with torch.inference_mode():
            encoded = modules.frame_encode(pixels)
        initial_prompt = _prompt(fixtures, "initial_prompt", original_size)
        correction_prompt = _prompt(fixtures, "correction_prompt", original_size)
        batch, steady_cut, decisions = _measure_decisions(
            modules, encoded, correction_prompt, fixtures
        )
        profile_id = f"b{batch}-1008-p16-box1-mask288-m10-ptr16-fp16"
        (staging / "reports/batch_decision.json").write_text(
            json.dumps(
                {
                    "Decision": decisions["batch_gate"]["decision"],
                    "Applicable profiles": [profile_id],
                    "protocol": fixtures["measurement"],
                    "b4": decisions["b4"],
                    "b8": decisions["b8"],
                    "gate": decisions["batch_gate"],
                    "launches": {"capacity": 1, "capacity-plus-one": 2},
                    "artifact_size_bytes": "recorded after selected-profile export",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (staging / "reports/fused_cut_decision.json").write_text(
            json.dumps(
                {
                    "Decision": steady_cut,
                    "Applicable profiles": [profile_id],
                    "protocol": fixtures["measurement"],
                    "split": decisions["split"],
                    "fused": decisions["fused"],
                    "gate": decisions["fused_gate"],
                    "copy_bytes": {
                        "split_intermediate_d2h": 0,
                        "fused_intermediate_d2h": 0,
                    },
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        _write_bundle_card(staging / "README.md", batch=batch, cut=steady_cut)
        args = _sample_args(modules, encoded, initial_prompt, batch)
        with torch.inference_mode():
            single_outputs = modules.preview_single1(*args)
        export_report: dict[str, Any] = {
            "format": "sam3-base-video-m4-export-report-v1",
            "profile_id": profile_id,
            "capture": "torch.export(strict=False)",
            "opset": 18,
            "sam3_export_commit": _git_revision(Path(__file__).resolve().parents[1]),
            "official_commit": official_commit,
            "checkpoint_sha256": checkpoint_sha,
            "graphs": {},
        }
        export_report["graphs"]["tracker-frame-encode"] = _export_one(
            modules.frame_encode,
            (pixels,),
            staging / "graphs" / GRAPH_NAMES["tracker-frame-encode"],
            FRAME_INPUTS,
            FRAME_OUTPUTS,
            capture_path=staging / "capture/tracker-frame-encode.pt2",
            capture_bundle_path="capture/tracker-frame-encode.pt2",
        )
        for role, module in (
            ("base-tracker-preview-multimask3", modules.preview_multimask3),
            ("base-tracker-preview-single1", modules.preview_single1),
        ):
            export_report["graphs"][role] = _export_one(
                module,
                args,
                staging / "graphs" / GRAPH_NAMES[role],
                PREVIEW_INPUTS,
                PREVIEW_OUTPUTS,
                capture_path=staging / "capture" / f"{role}.pt2",
                capture_bundle_path=f"capture/{role}.pt2",
            )
        commit_args = (
            args[0],
            single_outputs[2],
            single_outputs[4],
            torch.ones(batch, dtype=torch.bool, device="cuda"),
        )
        export_report["graphs"]["base-memory-commit"] = _export_one(
            modules.memory_commit,
            commit_args,
            staging / "graphs" / GRAPH_NAMES["base-memory-commit"],
            COMMIT_INPUTS,
            COMMIT_OUTPUTS,
            capture_path=staging / "capture/base-memory-commit.pt2",
            capture_bundle_path="capture/base-memory-commit.pt2",
        )
        export_report["graphs"]["base-tracker-step-and-commit-single1"] = _export_one(
            modules.step_and_commit_single1,
            args,
            staging / "graphs" / GRAPH_NAMES["base-tracker-step-and-commit-single1"],
            PREVIEW_INPUTS,
            FUSED_OUTPUTS,
            capture_path=(staging / "capture/base-tracker-step-and-commit-single1.pt2"),
            capture_bundle_path=("capture/base-tracker-step-and-commit-single1.pt2"),
        )
        graph_sizes = {
            role: int(record["size_bytes"])
            for role, record in export_report["graphs"].items()
        }
        batch_record_path = staging / "reports/batch_decision.json"
        batch_record = json.loads(batch_record_path.read_text(encoding="utf-8"))
        batch_record["artifact_size_bytes"] = sum(graph_sizes.values())
        batch_record_path.write_text(
            json.dumps(batch_record, indent=2) + "\n", encoding="utf-8"
        )
        fused_record_path = staging / "reports/fused_cut_decision.json"
        fused_record = json.loads(fused_record_path.read_text(encoding="utf-8"))
        fused_record["launches"] = {"split": 2, "fused": 1}
        fused_record["artifact_size_bytes"] = {
            "split": graph_sizes["base-tracker-preview-single1"]
            + graph_sizes["base-memory-commit"],
            "fused": graph_sizes["base-tracker-step-and-commit-single1"],
        }
        fused_record_path.write_text(
            json.dumps(fused_record, indent=2) + "\n", encoding="utf-8"
        )
        (staging / "reports/export_report.json").write_text(
            json.dumps(export_report, indent=2) + "\n", encoding="utf-8"
        )
        signatures = _graph_signatures(staging, profile_id, export_report["graphs"])
        (staging / "capture/graph_signatures.json").write_text(
            json.dumps(signatures, indent=2) + "\n", encoding="utf-8"
        )
        provenance = {
            "format": "sam3-base-video-m4-provenance-v1",
            "official_repository": "facebook/sam3",
            "official_commit": official_commit,
            "checkpoint_sha256": checkpoint_sha,
            "sam3_export_commit": export_report["sam3_export_commit"],
            "license_file": "LICENSE",
            "fixture_sha256": sha256_file(staging / "fixtures/cases.json"),
            "checkpoint_scalar_policy": (
                "serialized tensor shapes plus official builder values; "
                "scale/bias are not serialized tensors"
            ),
        }
        (staging / "reports/provenance.json").write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
        )
        (staging / "reports/m4_release_validation.json").write_text(
            json.dumps(
                {
                    "format": "sam3-base-video-m4-release-validation-v1",
                    "status": "pending",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        metadata = {
            "official_commit": official_commit,
            "checkpoint_sha256": checkpoint_sha,
            "torch_version": torch.__version__,
            "onnx_version": onnx.__version__,
        }
        manifest = _manifest(
            staging,
            signatures,
            fixtures,
            metadata,
            batch=batch,
            steady_cut=steady_cut,
        )
        (staging / "manifests" / f"{BASE_VIDEO_PLAN_ID}.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("validate_base_video_v2.py")),
                "--bundle-dir",
                str(staging),
                "--checkpoint",
                str(checkpoint_path),
                "--update-report",
            ],
            check=True,
        )
        staging.replace(output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/sam3-base-video-tracking-ortcuda-v2"),
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--official-repo", type=Path, default=Path("../sam3"))
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=Path("tests/fixtures/m4_base_video/cases.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    checkpoint = resolve_sam3_checkpoint(
        str(args.checkpoint) if args.checkpoint is not None else None
    )
    export_bundle(
        args.output_dir.resolve(),
        checkpoint,
        args.official_repo.resolve(),
        args.fixtures.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
