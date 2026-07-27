"""Build the shipped M3 SAM3 base interactive image PVS ORT CUDA bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
from typing import Any

from export_image_pcs_v2 import (
    _copy_file,
    _file_id,
    _file_record,
    _git_revision,
)
from m1_experiments import _export_one
import numpy as np
import onnx
from onnx import TensorProto
from PIL import Image
import torch

from sam3.export import (
    InitialNoMemoryCondition,
    InteractiveFeatureProject,
    InteractiveImageEncodeInitial,
    InteractivePredictMultimask3,
    InteractivePredictSingle1,
)
from sam3.runtime.interactive_image import (
    InteractivePrompt,
    _preprocess_interactive_image,
    _prompt_arrays,
)
from sam3.runtime.manifest import INTERACTIVE_PLAN_ID, MANIFEST_FORMAT_V2, sha256_file
from sam3.weights.load_sam3 import (
    build_production_interactive,
    resolve_sam3_checkpoint,
)

IMAGE_SIZE = 1008
POINT_CAPACITY = 16
MASK_SIZE = 288
PROFILE_ID = "b1-1008-p16-box1-mask288-fp16"
CONTRACT_VERSION = "1.0.0"
SCOPE_LABEL = "SAM3 base interactive image PVS / point-box-mask / ORT CUDA v1"
GRAPH_NAMES = {
    "interactive-image-encode-initial": "interactive_image_encode_initial.onnx",
    "interactive-predict-multimask3": "interactive_predict_multimask3.onnx",
    "interactive-predict-single1": "interactive_predict_single1.onnx",
}
ROLE_COMPONENTS = {
    "interactive-image-encode-initial": [
        "VisionTrunk",
        "SAM2Neck",
        "InteractiveFeatureProject",
        "InitialNoMemoryCondition",
    ],
    "interactive-predict-multimask3": [
        "PromptEncoder",
        "MaskDecoder",
        "InteractivePredictMultimask3",
    ],
    "interactive-predict-single1": [
        "PromptEncoder",
        "MaskDecoder",
        "InteractivePredictSingle1",
    ],
}
PROMPT_INPUT_NAMES = [
    "point_coords",
    "point_labels",
    "point_valid",
    "box_xyxy",
    "has_box",
    "mask_input",
    "has_mask",
]


def _preprocess_image(path: Path) -> tuple[torch.Tensor, tuple[int, int]]:
    image = Image.open(path).convert("RGB")
    values, original_size = _preprocess_interactive_image(image)
    return torch.from_numpy(values).to("cuda"), original_size


def _radial_logits() -> np.ndarray:
    axis = np.linspace(-1.0, 1.0, MASK_SIZE, dtype=np.float32)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    return ((0.58**2 - (xx * xx + yy * yy)) * 12.0).astype(np.float32)


def _case_prompt(
    case: dict[str, Any], mask: np.ndarray | None = None
) -> InteractivePrompt:
    points = np.asarray(case["points_xy"], dtype=np.float32).reshape(-1, 2)
    labels = np.asarray(case["point_labels"], dtype=np.int64)
    box = None if case.get("box_xyxy") is None else tuple(case["box_xyxy"])
    if mask is None and case.get("mask_source") is not None:
        mask = _radial_logits()
    return InteractivePrompt(
        points_xy=points,
        point_labels=labels,
        box_xyxy=box,
        mask_logits=mask,
    )


def _torch_prompt(
    prompt: InteractivePrompt, original_size: tuple[int, int]
) -> tuple[torch.Tensor, ...]:
    arrays, _facts = _prompt_arrays(prompt, original_size)
    return tuple(torch.from_numpy(value).to("cuda") for value in arrays.values())


def _mask_iou(expected: np.ndarray, actual: np.ndarray) -> float:
    expected_binary = expected > 0.0
    actual_binary = actual > 0.0
    intersection = np.logical_and(expected_binary, actual_binary).sum(axis=(-2, -1))
    union = np.logical_or(expected_binary, actual_binary).sum(axis=(-2, -1))
    return float(np.mean(np.where(union == 0, 1.0, intersection / union)))


def _compare_case(
    expected_scores: np.ndarray,
    expected_logits: np.ndarray,
    actual_scores: np.ndarray,
    actual_logits: np.ndarray,
) -> dict[str, Any]:
    score_max_abs = float(np.max(np.abs(expected_scores - actual_scores)))
    logit_mean_abs = float(np.mean(np.abs(expected_logits - actual_logits)))
    task_iou = _mask_iou(expected_logits, actual_logits)
    top_index_match = int(np.argmax(expected_scores)) == int(np.argmax(actual_scores))
    return {
        "mask_count_match": expected_logits.shape == actual_logits.shape,
        "top_score_index_match": top_index_match,
        "score_max_abs": score_max_abs,
        "low_res_logit_mean_abs": logit_mean_abs,
        "task_mask_iou": task_iou,
        "pass": bool(
            expected_logits.shape == actual_logits.shape
            and top_index_match
            and score_max_abs <= 0.02
            and logit_mean_abs <= 0.05
            and task_iou >= 0.98
        ),
    }


def _local_references(
    fixtures: dict[str, Any],
    original_size: tuple[int, int],
    encoded: tuple[torch.Tensor, ...],
    multimask: InteractivePredictMultimask3,
    single: InteractivePredictSingle1,
    output: Path,
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "image_embedding": encoded[0].float().cpu().numpy(),
        "high_res_0": encoded[1].float().cpu().numpy(),
        "high_res_1": encoded[2].float().cpu().numpy(),
    }
    with torch.inference_mode():
        for case in fixtures["cases"]:
            module = multimask if case["multimask_output"] else single
            low_res, scores = module(
                *encoded, *_torch_prompt(_case_prompt(case), original_size)
            )
            prefix = case["id"]
            arrays[f"{prefix}__scores"] = scores[0].cpu().numpy()
            arrays[f"{prefix}__low_res"] = low_res[0].cpu().numpy()

        repeated = fixtures["repeated_click"]
        first_case = {
            "points_xy": repeated["first_points_xy"],
            "point_labels": repeated["first_point_labels"],
            "box_xyxy": None,
            "mask_source": None,
        }
        first_low, first_scores = multimask(
            *encoded, *_torch_prompt(_case_prompt(first_case), original_size)
        )
        selected = int(torch.argmax(first_scores[0]))
        second_case = {
            "points_xy": repeated["second_points_xy"],
            "point_labels": repeated["second_point_labels"],
            "box_xyxy": None,
            "mask_source": None,
        }
        second_low, second_scores = single(
            *encoded,
            *_torch_prompt(
                _case_prompt(second_case, first_low[0, selected].float().cpu().numpy()),
                original_size,
            ),
        )
        prefix = repeated["id"]
        arrays[f"{prefix}__first_scores"] = first_scores[0].cpu().numpy()
        arrays[f"{prefix}__first_low_res"] = first_low[0].cpu().numpy()
        arrays[f"{prefix}__selected_index"] = np.asarray(selected, dtype=np.int64)
        arrays[f"{prefix}__second_scores"] = second_scores[0].cpu().numpy()
        arrays[f"{prefix}__second_low_res"] = second_low[0].cpu().numpy()
    np.savez_compressed(output, **arrays)
    return arrays


def _official_local_report(
    fixtures: dict[str, Any], official_path: Path, local: dict[str, np.ndarray]
) -> dict[str, Any]:
    official = np.load(official_path)
    feature_diffs = {
        name: {
            "max_abs": float(np.max(np.abs(official[name] - local[name]))),
            "mean_abs": float(np.mean(np.abs(official[name] - local[name]))),
        }
        for name in ("image_embedding", "high_res_0", "high_res_1")
    }
    cases: list[dict[str, Any]] = []
    for case in fixtures["cases"]:
        prefix = case["id"]
        result = _compare_case(
            official[f"{prefix}__scores"],
            official[f"{prefix}__low_res"],
            local[f"{prefix}__scores"],
            local[f"{prefix}__low_res"],
        )
        cases.append({"id": prefix, **result})
    prefix = fixtures["repeated_click"]["id"]
    repeated_first = _compare_case(
        official[f"{prefix}__first_scores"],
        official[f"{prefix}__first_low_res"],
        local[f"{prefix}__first_scores"],
        local[f"{prefix}__first_low_res"],
    )
    repeated_second = _compare_case(
        official[f"{prefix}__second_scores"],
        official[f"{prefix}__second_low_res"],
        local[f"{prefix}__second_scores"],
        local[f"{prefix}__second_low_res"],
    )
    selected_match = bool(
        official[f"{prefix}__selected_index"] == local[f"{prefix}__selected_index"]
    )
    passed = all(case["pass"] for case in cases)
    passed = passed and repeated_first["pass"] and repeated_second["pass"]
    passed = passed and selected_match
    if not passed:
        raise RuntimeError("official-to-local interactive parity gate failed")
    return {
        "status": "pass",
        "feature_differences": feature_diffs,
        "cases": cases,
        "repeated_click": {
            "first": repeated_first,
            "selected_index_match": selected_match,
            "second": repeated_second,
            "image_encode_count": 1,
            "predict_launch_count": 2,
            "memory_encode_count": 0,
            "memory_commit_count": 0,
        },
    }


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def _measure_cut(
    module: InteractiveImageEncodeInitial,
    pixels: torch.Tensor,
    fixtures: dict[str, Any],
) -> dict[str, Any]:
    warmup = int(fixtures["measurement"]["warmup"])
    repeats = int(fixtures["measurement"]["repeats"])

    def fused() -> tuple[torch.Tensor, ...]:
        return module(pixels)

    def logical_split() -> tuple[torch.Tensor, ...]:
        with torch.autocast("cuda", dtype=torch.float16):
            _sam3, _pos, sam2, _pos2 = module.backbone(pixels)
            if sam2 is None:
                raise RuntimeError("missing SAM2 FPN")
            levels = list(sam2)[:-1][-3:]
            base, high0, high1 = module.feature_project(*levels)
            return module.initial_condition(base), high0, high1

    measurements: dict[str, Any] = {}
    with torch.inference_mode():
        expected = fused()
        actual = logical_split()
        for left, right in zip(expected, actual):
            torch.testing.assert_close(left, right)
        for name, operation in (("fused", fused), ("logical_split", logical_split)):
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
            measurements[name] = {
                "median_ms": statistics.median(timings),
                "p95_ms": _percentile(timings, 0.95),
                "peak_vram_bytes": torch.cuda.max_memory_allocated(),
                "intermediate_d2h_bytes": 0,
            }
    return {
        "fixture": fixtures["fixture_version"],
        "dtype": "float16",
        "warmup": warmup,
        "repeats": repeats,
        "outputs_exact": True,
        "measurements": measurements,
        "decision": "ship fused image-only recipe; retain logical component contracts",
        "applicable_profiles": [PROFILE_ID],
    }


def _onnx_dtype(value: int) -> str:
    names = {
        TensorProto.FLOAT: "float32",
        TensorProto.FLOAT16: "float16",
        TensorProto.INT64: "int64",
        TensorProto.INT32: "int32",
        TensorProto.BOOL: "bool",
    }
    return names[value]


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
    bundle_dir: Path, export_graphs: dict[str, Any]
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
        "format": "sam3-interactive-image-graph-signatures-v1",
        "profile_id": PROFILE_ID,
        "capture": "torch.export(strict=False)",
        "opset": 18,
        "graphs": graphs,
    }


def _artifact_and_tensors(
    signatures: dict[str, Any], bundle_dir: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    specs: dict[str, dict[str, Any]] = {}
    artifacts: list[dict[str, Any]] = []

    def tensor_id(role: str, name: str, *, output: bool) -> str:
        base = name.replace("_", "-")
        if output and name in {"low_res_logits", "scores"}:
            return f"{base}-{role.removeprefix('interactive-predict-')}"
        return base

    for role, signature in signatures["graphs"].items():
        for is_output, values in (
            (False, signature["inputs"]),
            (True, signature["outputs"]),
        ):
            for value in values:
                ref = tensor_id(role, value["name"], output=is_output)
                spec = {
                    "semantic_name": value["name"],
                    "dtype": value["dtype"],
                    "shape": value["shape"],
                }
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
                        "tensor_ref": tensor_id(role, value["name"], output=False),
                        "backend_name": value["name"],
                    }
                    for value in signature["inputs"]
                ],
                "outputs": [
                    {
                        "tensor_ref": tensor_id(role, value["name"], output=True),
                        "backend_name": value["name"],
                    }
                    for value in signature["outputs"]
                ],
            }
        )
    host_inputs = {
        "pixel_values",
        "point_coords",
        "point_labels",
        "point_valid",
        "box_xyxy",
        "has_box",
        "mask_input",
        "has_mask",
    }
    tensors: list[dict[str, Any]] = []
    for tensor_ref, full_spec in sorted(specs.items()):
        spec = dict(full_spec)
        name = spec.pop("semantic_name")
        if name in {"low_res_logits", "scores"}:
            residency = "host-output"
        elif name in host_inputs:
            residency = "host-input"
        else:
            residency = "device"
        coordinates = "not-applicable"
        if name in {"point_coords", "box_xyxy"}:
            coordinates = "absolute XY or XYXY in the 1008 model frame"
        elif name in {"mask_input", "low_res_logits"}:
            coordinates = "288x288 low-resolution logit grid"
        elif name == "pixel_values":
            coordinates = "RGB image resized bilinearly to 1008x1008"
        tensors.append(
            {
                "id": tensor_ref,
                "semantic_type": name,
                **spec,
                "layout": "static profile; see shape",
                "unit": "pixel" if name in {"point_coords", "box_xyxy"} else "unitless",
                "normalization": "(x / 255 - 0.5) / 0.5"
                if name == "pixel_values"
                else "none",
                "padding": "invalid point slots use label -1 and point_valid=false",
                "validity": "explicit point_valid/has_box/has_mask",
                "coordinates": coordinates,
                "value_kind": "mask-logit" if "logit" in name else "feature",
                "residency": residency,
            }
        )
    return artifacts, tensors


def _manifest(
    bundle_dir: Path,
    signatures: dict[str, Any],
    fixtures: dict[str, Any],
    metadata: dict[str, str],
) -> dict[str, Any]:
    artifacts, tensors = _artifact_and_tensors(signatures, bundle_dir)
    file_paths = {
        "LICENSE",
        "README.md",
        "capture/graph_signatures.json",
        "fixtures/cases.json",
        "fixtures/images/truck.jpg",
        "fixtures/official_reference.npz",
        "fixtures/official_reference.json",
        "fixtures/local_reference.npz",
        "reports/export_report.json",
        "reports/m3_release_validation.json",
        "reports/provenance.json",
        "reports/cut_measurement.json",
    }
    for signature in signatures["graphs"].values():
        file_paths.add(signature["exported_program"]["program_path"])
        file_paths.add(signature["path"])
        if (bundle_dir / (signature["path"] + ".data")).is_file():
            file_paths.add(signature["path"] + ".data")
    files = [_file_record(bundle_dir, path) for path in sorted(file_paths)]
    encoded_refs = ["image-embedding", "high-res-0", "high-res-1"]
    edges = [
        {
            "producer_artifact_ref": "interactive-image-encode-initial",
            "consumer_artifact_ref": role,
            "tensor_refs": encoded_refs,
        }
        for role in ("interactive-predict-multimask3", "interactive-predict-single1")
    ]
    report_ref = _file_id("reports/m3_release_validation.json")
    return {
        "format": MANIFEST_FORMAT_V2,
        "manifest_id": f"{INTERACTIVE_PLAN_ID}-manifest-v2",
        "scope": {
            "classification": "public-deployment",
            "lifecycle": "shipped",
            "dispatch_role": "default",
            "scope_label": SCOPE_LABEL,
            "use_case": "interactive-image-pvs",
            "prompt_coverage": ["point", "box", "mask"],
            "capabilities": [
                "interactive image PVS",
                "repeated click",
                "multimask3",
                "single1",
            ],
            "exclusions": [
                "text image PCS",
                "video tracking",
                "memory state",
                "object batching",
                "SAM3.1",
            ],
        },
        "plan": {
            "id": INTERACTIVE_PLAN_ID,
            "contract_version": CONTRACT_VERSION,
            "semantic_graph_kind": "sam3-base-interactive-image-pvs",
            "role_set": list(GRAPH_NAMES),
            "components": [
                "InteractiveFeatureProject",
                "InitialNoMemoryCondition",
                "InteractiveImageEncodeInitial",
                "InteractivePredictMultimask3",
                "InteractivePredictSingle1",
            ],
        },
        "model": {
            "family": "sam3",
            "variant": "base",
            "vision_layout": "SAM2 FPN 288/144/72 with projected high-resolution views",
            "tracking_layout": "not-applicable; initial/no-memory condition only",
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
                {"name": "image-condition", "value": "initial-no-memory"}
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
            "id": PROFILE_ID,
            "precision": "fp16",
            "shape_mode": "static",
            "static_values": [
                {"name": "batch", "value": 1},
                {"name": "image-size", "value": 1008},
                {"name": "point-capacity", "value": 16},
                {"name": "box-capacity", "value": 1},
                {"name": "mask-size", "value": 288},
            ],
            "dynamic_dimensions": [],
        },
        "tensors": tensors,
        "artifacts": artifacts,
        "execution": {"entry_artifacts": list(GRAPH_NAMES), "edges": edges},
        "caches": [
            {
                "id": "image-cache",
                "tensor_refs": encoded_refs,
                "lifetime": "image-session",
                "key_version": "1.0.0",
                "key_parts": [
                    "preprocessed-bytes",
                    "original-size",
                    "checkpoint-digest",
                    "profile-id",
                    "initial-no-memory-condition-id",
                ],
                "invalidated_by": [
                    "image-change",
                    "checkpoint-change",
                    "profile-change",
                    "condition-change",
                ],
                "state_compatibility": "initial/no-memory image view for the exact checkpoint and profile only",
            }
        ],
        "handoffs": [
            {
                "id": f"handoff-image-to-{index}",
                "producer_artifact_ref": edge["producer_artifact_ref"],
                "consumer_artifact_ref": edge["consumer_artifact_ref"],
                "tensor_refs": edge["tensor_refs"],
                "requirement": "required-device",
                "mechanism": "ORT CUDA IOBinding / CUDA OrtValue",
                "fallback_plan_id": None,
            }
            for index, edge in enumerate(edges, start=1)
        ],
        "capture": {
            "canonical_format": "exported-program",
            "mode": "non-strict",
            "pytorch_version": metadata["torch_version"],
            "exporter_version": metadata["onnx_version"],
            "constraints": [
                "batch=1",
                "image=1008x1008",
                "points=16",
                "boxes=1",
                "mask=288x288",
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
                "name": "image-condition",
                "binding": "baked",
                "value": "initial-no-memory",
                "stage": "image-cache",
            },
            {
                "name": "prompt-abi",
                "binding": "host-runtime",
                "value": "fixed P16 + box1 + mask288 with explicit validity",
                "stage": "predict-input",
            },
            {
                "name": "multimask-dispatch",
                "binding": "host-runtime",
                "value": "multimask3 default; single1 explicit",
                "stage": "artifact-selection",
            },
            {
                "name": "output-policy",
                "binding": "host-runtime",
                "value": "bilinear logits resize; strict > mask threshold",
                "stage": "public-result",
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
                "cases": [case["id"] for case in fixtures["cases"]]
                + [fixtures["repeated_click"]["id"]],
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


def _write_bundle_card(path: Path) -> None:
    path.write_text(
        """# SAM3 base interactive image PVS / point-box-mask / ORT CUDA v1

This M3 bundle ships the fixed `b1-1008-p16-box1-mask288-fp16` profile with
one fused initial image encoder and static multimask3/single1 prediction
artifacts. The image cache is valid only for the baked `initial-no-memory`
condition. CUDA IOBinding is required and there is no fallback.

This default applies only to interactive image PVS. It does not replace the
SAM3 base text-only image PCS default and excludes video/memory state,
object batching, and SAM3.1.
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
            raise RuntimeError("checkpoint does not match the M3 fixture")
        official_commit = _git_revision(official_repo)
        if official_commit != fixtures["official_source_commit"]:
            raise RuntimeError("official repository does not match the M3 fixture")
        workspace = fixtures_path.resolve().parents[4]
        image_path = workspace / fixtures["image"]["workspace_path"]
        if sha256_file(image_path) != fixtures["image"]["sha256"]:
            raise RuntimeError("fixture image hash mismatch")

        _copy_file(Path("LICENSE"), staging / "LICENSE")
        _copy_file(fixtures_path, staging / "fixtures/cases.json")
        _copy_file(image_path, staging / "fixtures/images/truck.jpg")
        _write_bundle_card(staging / "README.md")
        (staging / "reports").mkdir()
        (staging / "graphs").mkdir()
        (staging / "capture").mkdir()

        official_python = official_repo / ".venv/bin/python"
        if not official_python.is_file():
            official_python = Path(sys.executable)
        subprocess.run(
            [
                str(official_python),
                str(Path(__file__).with_name("m3_official_interactive_reference.py")),
                "--official-root",
                str(official_repo),
                "--checkpoint",
                str(checkpoint_path),
                "--fixtures",
                str(fixtures_path),
                "--image",
                str(image_path),
                "--output",
                str(staging / "fixtures/official_reference.npz"),
            ],
            check=True,
        )

        pixels, original_size = _preprocess_image(image_path)
        predictor = build_production_interactive(
            checkpoint_path=str(checkpoint_path),
            device="cuda",
            dtype="fp16",
            load_weights=True,
        ).eval()
        feature_project = InteractiveFeatureProject(predictor.head.mask_decoder).eval()
        initial_condition = InitialNoMemoryCondition(predictor.no_mem_embed).eval()
        encode = InteractiveImageEncodeInitial(
            predictor.backbone,
            feature_project,
            initial_condition,
            use_cuda_autocast=True,
        ).eval()
        multimask = InteractivePredictMultimask3(
            predictor.head, use_cuda_autocast=True
        ).eval()
        single = InteractivePredictSingle1(
            predictor.head, use_cuda_autocast=True
        ).eval()

        with torch.inference_mode():
            encoded = encode(pixels)
        local = _local_references(
            fixtures,
            original_size,
            encoded,
            multimask,
            single,
            staging / "fixtures/local_reference.npz",
        )
        official_local = _official_local_report(
            fixtures, staging / "fixtures/official_reference.npz", local
        )
        cut_measurement = _measure_cut(encode, pixels, fixtures)
        (staging / "reports/cut_measurement.json").write_text(
            json.dumps(cut_measurement, indent=2) + "\n", encoding="utf-8"
        )

        sample_prompt = _torch_prompt(_case_prompt(fixtures["cases"][1]), original_size)
        export_report: dict[str, Any] = {
            "format": "sam3-interactive-image-m3-export-report-v1",
            "profile_id": PROFILE_ID,
            "capture": "torch.export(strict=False)",
            "opset": 18,
            "sam3_export_commit": _git_revision(Path(__file__).resolve().parents[1]),
            "official_commit": official_commit,
            "checkpoint_sha256": checkpoint_sha,
            "official_to_local": official_local,
            "graphs": {},
        }
        export_report["graphs"]["interactive-image-encode-initial"] = _export_one(
            encode,
            (pixels,),
            staging / "graphs" / GRAPH_NAMES["interactive-image-encode-initial"],
            ["pixel_values"],
            ["image_embedding", "high_res_0", "high_res_1"],
            capture_path=staging / "capture/interactive-image-encode-initial.pt2",
            capture_bundle_path="capture/interactive-image-encode-initial.pt2",
        )
        predict_input_names = [
            "image_embedding",
            "high_res_0",
            "high_res_1",
            *PROMPT_INPUT_NAMES,
        ]
        for role, module in (
            ("interactive-predict-multimask3", multimask),
            ("interactive-predict-single1", single),
        ):
            export_report["graphs"][role] = _export_one(
                module,
                (*encoded, *sample_prompt),
                staging / "graphs" / GRAPH_NAMES[role],
                predict_input_names,
                ["low_res_logits", "scores"],
                capture_path=staging / "capture" / f"{role}.pt2",
                capture_bundle_path=f"capture/{role}.pt2",
            )
        (staging / "reports/export_report.json").write_text(
            json.dumps(export_report, indent=2) + "\n", encoding="utf-8"
        )
        signatures = _graph_signatures(staging, export_report["graphs"])
        (staging / "capture/graph_signatures.json").write_text(
            json.dumps(signatures, indent=2) + "\n", encoding="utf-8"
        )
        provenance = {
            "format": "sam3-interactive-image-m3-provenance-v1",
            "official_repository": "facebook/sam3",
            "official_commit": official_commit,
            "checkpoint_sha256": checkpoint_sha,
            "sam3_export_commit": export_report["sam3_export_commit"],
            "license_file": "LICENSE",
            "fixture_sha256": sha256_file(staging / "fixtures/cases.json"),
        }
        (staging / "reports/provenance.json").write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
        )
        (staging / "reports/m3_release_validation.json").write_text(
            json.dumps(
                {
                    "format": "sam3-interactive-image-m3-release-validation-v1",
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
        (staging / "manifests").mkdir()
        manifest = _manifest(staging, signatures, fixtures, metadata)
        (staging / "manifests" / f"{INTERACTIVE_PLAN_ID}.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("validate_interactive_image_v2.py")),
                "--bundle-dir",
                str(staging),
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
        default=Path("artifacts/sam3-interactive-image-pvs-ortcuda-v2"),
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--official-repo", type=Path, default=Path("../sam3"))
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=Path("tests/fixtures/m3_interactive/cases.json"),
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
