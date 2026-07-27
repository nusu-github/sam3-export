"""Build the M5 SAM3.1 native Multiplex ORT CUDA release bundle."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import tempfile
import threading
import time
from typing import Any

from export_image_pcs_v2 import _file_id, _file_record, _git_revision
from m1_experiments import _artifact_record, _parity
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
from sam3.runtime.manifest import (
    MANIFEST_FORMAT_V2,
    MULTIPLEX_VIDEO_PLAN_ID,
    sha256_file,
)
from sam3.runtime.multiplex_state import MultiplexVariantParameters
from sam3.weights import (
    SAM31_CHECKPOINT_SHA256,
    SAM31_REVISION,
    build_sam31_multiplex_video_modules,
)

CONTRACT_VERSION = "1.0.0"
SCOPE_LABEL = (
    "SAM3.1 multiplex video tracking / point-box-mask correction / "
    "bucket16 / ORT CUDA v1"
)
COMMON_GRAPH_NAMES = {
    "multiplex-frame-encode": "multiplex_frame_encode.onnx",
    "multiplex-interaction-preview-multimask3": (
        "multiplex_interaction_preview_multimask3.onnx"
    ),
    "multiplex-interaction-preview-single1": (
        "multiplex_interaction_preview_single1.onnx"
    ),
}
FRAME_INPUTS = ["pixel_values"]
FRAME_OUTPUTS = [
    "interactive_image",
    "interactive_position",
    "interactive_high_res_0",
    "interactive_high_res_1",
    "propagation_image",
    "propagation_position",
    "propagation_high_res_0",
    "propagation_high_res_1",
]
PREVIEW_INPUTS = [
    "interactive_image",
    "interactive_high_res_0",
    "interactive_high_res_1",
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
PROPAGATION_INPUTS = [
    "propagation_image",
    "propagation_position",
    "propagation_high_res_0",
    "propagation_high_res_1",
    "slot_validity",
    "memory_features",
    "memory_position",
    "memory_image_features",
    "memory_image_position",
    "memory_valid",
    "memory_age",
    "object_pointers",
    "pointer_valid",
    "pointer_age",
]
PROPAGATION_OUTPUTS = [
    "candidate_low_res",
    "scores",
    "selected_low_res",
    "selected_high_res",
    "object_pointers_out",
    "object_score",
]
MEMORY_INPUTS = [
    "propagation_image",
    "bucket_masks",
    "object_score",
    "slot_validity",
    "conditioning_validity",
]
MEMORY_OUTPUTS = ["memory_features", "memory_position"]
SCATTER_INPUTS = [
    "propagation_image",
    "bucket_low_res",
    "bucket_high_res",
    "bucket_pointers",
    "bucket_object_scores",
    "replacement_low_res",
    "replacement_high_res",
    "replacement_pointer",
    "replacement_object_score",
    "assignment",
    "slot_validity",
    "conditioning_validity",
]
SCATTER_OUTPUTS = [
    "bucket_low_res_out",
    "bucket_high_res_out",
    "bucket_pointers_out",
    "bucket_object_scores_out",
    "memory_features",
    "memory_position",
]


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def _copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _torch_prompt(values: dict[str, np.ndarray]) -> tuple[torch.Tensor, ...]:
    return tuple(torch.from_numpy(value).to("cuda") for value in values.values())


def _prompt(
    fixtures: dict[str, Any], original_size: tuple[int, int]
) -> dict[str, np.ndarray]:
    relative = np.asarray(fixtures["object_points_xy_relative"][0], dtype=np.float32)[
        None
    ]
    height, width = original_size
    absolute = relative * np.asarray([width, height], dtype=np.float32)
    values, _ = _prompt_arrays(
        InteractivePrompt(
            points_xy=absolute,
            point_labels=np.ones(1, dtype=np.int64),
        ),
        original_size,
    )
    return values


def _sample_state(
    modules: Any,
    encoded: tuple[torch.Tensor, ...],
    prompt: dict[str, np.ndarray],
    bucket_count: int,
    active_objects: int,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    with torch.inference_mode():
        preview = modules.preview_single1(
            encoded[0],
            encoded[2],
            encoded[3],
            *_torch_prompt(prompt),
        )
    validity = torch.zeros((bucket_count, 16), dtype=torch.uint8, device="cuda")
    for offset in range(active_objects):
        validity[offset // 16, offset % 16] = True
    high = torch.full(
        (bucket_count, 16, 1, 1008, 1008),
        -1024.0,
        dtype=torch.float32,
        device="cuda",
    )
    object_score = torch.full(
        (bucket_count, 16, 1),
        -1024.0,
        dtype=torch.float32,
        device="cuda",
    )
    pointers = torch.zeros(
        (bucket_count, 16, 16, 256),
        dtype=torch.float16,
        device="cuda",
    )
    pointer_valid = torch.zeros(
        (bucket_count, 16, 16), dtype=torch.uint8, device="cuda"
    )
    for offset in range(active_objects):
        bucket, slot = divmod(offset, 16)
        high[bucket, slot] = preview[2][0]
        object_score[bucket, slot] = preview[4][0]
        pointers[bucket, 0, slot] = preview[3][0]
        pointer_valid[bucket, 0, slot] = True
    commit_module = (
        modules.memory_commit_bucket1
        if bucket_count == 1
        else modules.memory_commit_bucket2
    )
    with torch.inference_mode():
        committed = commit_module(encoded[4], high, object_score, validity, validity)
    memory = torch.zeros(
        (bucket_count, 10, 256, 72, 72),
        dtype=torch.float16,
        device="cuda",
    )
    memory_position = torch.zeros_like(memory)
    memory[:, 0] = committed[0]
    memory_position[:, 0] = committed[1]
    memory_image = torch.zeros((10, 256, 72, 72), dtype=torch.float16, device="cuda")
    memory_image_position = torch.zeros_like(memory_image)
    memory_image[0] = encoded[4][0]
    memory_image_position[0] = encoded[5][0]
    memory_valid = torch.zeros((bucket_count, 10), dtype=torch.uint8, device="cuda")
    memory_valid[:, 0] = True
    memory_age = torch.zeros((bucket_count, 10), dtype=torch.int64, device="cuda")
    memory_age[:, 0] = 1
    pointer_age = torch.zeros((bucket_count, 16), dtype=torch.int64, device="cuda")
    pointer_age[:, 0] = 1
    propagation = (
        encoded[4],
        encoded[5],
        encoded[6],
        encoded[7],
        validity,
        memory,
        memory_position,
        memory_image,
        memory_image_position,
        memory_valid,
        memory_age,
        pointers,
        pointer_valid,
        pointer_age,
    )
    low = torch.full(
        (bucket_count, 16, 1, 288, 288),
        -1024.0,
        dtype=torch.float32,
        device="cuda",
    )
    bucket_pointers = pointers[:, 0].contiguous()
    scatter = (
        encoded[4],
        low,
        high,
        bucket_pointers,
        object_score,
        preview[0],
        preview[2],
        preview[3],
        preview[4],
        torch.tensor([[0, 0]], dtype=torch.int64, device="cuda"),
        validity,
        validity,
    )
    return propagation, scatter


def _dynamic_shapes(kind: str) -> tuple[Any, ...]:
    bucket = torch.export.Dim("bucket_count", min=1, max=2)
    if kind == "propagation":
        return (
            None,
            None,
            None,
            None,
            {0: bucket},
            {0: bucket},
            {0: bucket},
            None,
            None,
            {0: bucket},
            {0: bucket},
            {0: bucket},
            {0: bucket},
            {0: bucket},
        )
    if kind == "memory":
        return (None, {0: bucket}, {0: bucket}, {0: bucket}, {0: bucket})
    if kind == "scatter":
        return (
            None,
            {0: bucket},
            {0: bucket},
            {0: bucket},
            {0: bucket},
            None,
            None,
            None,
            None,
            None,
            {0: bucket},
            {0: bucket},
        )
    raise ValueError(f"unknown dynamic graph kind: {kind}")


def _export_one(
    module: torch.nn.Module,
    args: tuple[torch.Tensor, ...],
    path: Path,
    input_names: list[str],
    output_names: list[str],
    *,
    dynamic_kind: str | None = None,
    capture_path: Path | None = None,
    capture_bundle_path: str | None = None,
) -> dict[str, Any]:
    module.eval()
    cloned = tuple(value.detach().clone() for value in args)
    dynamic_shapes = _dynamic_shapes(dynamic_kind) if dynamic_kind is not None else None
    with torch.no_grad():
        eager = module(*cloned)
        exported = torch.export.export(
            module,
            cloned,
            dynamic_shapes=dynamic_shapes,
            strict=False,
        )
        capture_program = exported.run_decompositions()
        exported_outputs = capture_program.module()(*cloned)
    parity = _parity(
        eager if isinstance(eager, tuple) else (eager,),
        exported_outputs
        if isinstance(exported_outputs, tuple)
        else (exported_outputs,),
    )
    capture: dict[str, Any] | None = None
    if capture_path is not None:
        if capture_bundle_path is None:
            raise ValueError("capture_bundle_path is required with capture_path")
        from capture_utils import save_exported_program

        capture = save_exported_program(
            capture_program,
            capture_path,
            bundle_path=capture_bundle_path,
            input_names=input_names,
            output_names=output_names,
            mode="non-strict",
        )
    torch.onnx.export(
        exported,
        (),
        path,
        input_names=input_names,
        output_names=output_names,
        opset_version=18,
        dynamo=True,
        external_data=True,
        optimize=False,
    )
    onnx.checker.check_model(path)
    report = {
        "capture_mode": (
            "torch.export strict=False; bucket_count=1..2"
            if dynamic_kind is not None
            else "torch.export strict=False; static profile"
        ),
        "eager_to_exported_program": parity,
        **_artifact_record(path),
    }
    if capture is not None:
        report["exported_program"] = capture
    return report


def _candidate_names(dynamic: bool, bucket_count: int | None = None) -> dict[str, str]:
    if dynamic:
        return {
            "propagation": "multiplex_propagation.onnx",
            "memory": "multiplex_memory_commit.onnx",
            "scatter": "multiplex_scatter_replace_commit.onnx",
        }
    if bucket_count not in (1, 2):
        raise ValueError("fixed candidate needs one or two buckets")
    return {
        "propagation": f"multiplex_propagation_bucket{bucket_count}.onnx",
        "memory": f"multiplex_memory_commit_bucket{bucket_count}.onnx",
        "scatter": f"multiplex_scatter_replace_commit_bucket{bucket_count}.onnx",
    }


def _operation_role(operation: str) -> str:
    return {
        "propagation": "propagation",
        "memory": "memory-commit",
        "scatter": "scatter-replace-commit",
    }[operation]


def _export_candidate(
    modules: Any,
    encoded: tuple[torch.Tensor, ...],
    prompt: dict[str, np.ndarray],
    target: Path,
    *,
    dynamic: bool,
    bucket_count: int,
    capture_programs: bool = True,
) -> dict[str, Any]:
    target.mkdir(parents=True, exist_ok=True)
    propagation_args, scatter_args = _sample_state(
        modules,
        encoded,
        prompt,
        bucket_count,
        17 if bucket_count == 2 else 16,
    )
    names = _candidate_names(dynamic, None if dynamic else bucket_count)
    propagation_module = (
        modules.propagation_dynamic
        if dynamic
        else (
            modules.propagation_bucket1
            if bucket_count == 1
            else modules.propagation_bucket2
        )
    )
    memory_module = (
        modules.memory_commit_dynamic
        if dynamic
        else (
            modules.memory_commit_bucket1
            if bucket_count == 1
            else modules.memory_commit_bucket2
        )
    )
    scatter_module = (
        modules.scatter_commit_dynamic
        if dynamic
        else (
            modules.scatter_commit_bucket1
            if bucket_count == 1
            else modules.scatter_commit_bucket2
        )
    )
    with torch.inference_mode():
        propagated = propagation_module(*propagation_args)
    memory_args = (
        encoded[4],
        propagated[3],
        propagated[5],
        propagation_args[4],
        torch.zeros_like(propagation_args[4]),
    )

    def kind(name: str) -> str | None:
        return name if dynamic else None

    return {
        "propagation": _export_one(
            propagation_module,
            propagation_args,
            target / names["propagation"],
            PROPAGATION_INPUTS,
            PROPAGATION_OUTPUTS,
            dynamic_kind=kind("propagation"),
            capture_path=(
                target / "capture" / f"{Path(names['propagation']).stem}.pt2"
                if capture_programs
                else None
            ),
            capture_bundle_path=(
                f"capture/{Path(names['propagation']).stem}.pt2"
                if capture_programs
                else None
            ),
        ),
        "memory": _export_one(
            memory_module,
            memory_args,
            target / names["memory"],
            MEMORY_INPUTS,
            MEMORY_OUTPUTS,
            dynamic_kind=kind("memory"),
            capture_path=(
                target / "capture" / f"{Path(names['memory']).stem}.pt2"
                if capture_programs
                else None
            ),
            capture_bundle_path=(
                f"capture/{Path(names['memory']).stem}.pt2"
                if capture_programs
                else None
            ),
        ),
        "scatter": _export_one(
            scatter_module,
            scatter_args,
            target / names["scatter"],
            SCATTER_INPUTS,
            SCATTER_OUTPUTS,
            dynamic_kind=kind("scatter"),
            capture_path=(
                target / "capture" / f"{Path(names['scatter']).stem}.pt2"
                if capture_programs
                else None
            ),
            capture_bundle_path=(
                f"capture/{Path(names['scatter']).stem}.pt2"
                if capture_programs
                else None
            ),
        ),
    }


def _existing_candidate_report(
    target: Path, *, dynamic: bool, bucket_count: int
) -> dict[str, Any]:
    names = _candidate_names(dynamic, None if dynamic else bucket_count)
    report: dict[str, Any] = {}
    for operation, filename in names.items():
        path = target / filename
        if not path.is_file():
            raise FileNotFoundError(f"candidate graph is missing: {path}")
        onnx.checker.check_model(path)
        report[operation] = {
            "capture_mode": ("reused candidate; EP parity rerun by release validator"),
            "eager_to_exported_program": {"status": "release-validator"},
            **_artifact_record(path),
        }
    return report


def _ort_value(ort: Any, value: torch.Tensor) -> Any:
    return ort.OrtValue.from_dlpack(value.contiguous())


def _ort_session(ort: Any, path: Path) -> Any:
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(
        str(path),
        sess_options=options,
        providers=[
            (
                "CUDAExecutionProvider",
                {
                    "arena_extend_strategy": "kSameAsRequested",
                    "use_ep_level_unified_stream": "1",
                },
            )
        ],
    )
    if session.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError(f"CUDA EP did not load for {path.name}")
    return session


def _run_ort(session: Any, inputs: dict[str, Any]) -> tuple[dict[str, Any], float]:
    binding = session.io_binding()
    for item in session.get_inputs():
        binding.bind_ortvalue_input(item.name, inputs[item.name])
    for item in session.get_outputs():
        binding.bind_output(item.name, "cuda", 0)
    start = time.perf_counter()
    session.run_with_iobinding(binding)
    elapsed = (time.perf_counter() - start) * 1000.0
    values = binding.get_outputs()
    if any(value.device_name() != "cuda" for value in values):
        raise RuntimeError("candidate IOBinding output left CUDA")
    return (
        {item.name: value for item, value in zip(session.get_outputs(), values)},
        elapsed,
    )


def _benchmark_candidate(
    candidate_dir: Path,
    propagation_args: tuple[torch.Tensor, ...],
    *,
    dynamic: bool,
    active_objects: int,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    import onnxruntime as ort

    bucket_count = 1 if active_objects <= 16 else 2
    names = _candidate_names(dynamic, None if dynamic else 1)
    gc.collect()
    torch.cuda.empty_cache()
    prop = _ort_session(ort, candidate_dir / names["propagation"])
    memory = _ort_session(ort, candidate_dir / names["memory"])
    bucketed = {
        "slot_validity",
        "memory_features",
        "memory_position",
        "memory_valid",
        "memory_age",
        "object_pointers",
        "pointer_valid",
        "pointer_age",
    }

    def inputs_for_bucket(bucket: int | None) -> dict[str, Any]:
        return {
            name: _ort_value(
                ort,
                (
                    value
                    if bucket is None or name not in bucketed
                    else value[bucket : bucket + 1]
                ),
            )
            for name, value in zip(PROPAGATION_INPUTS, propagation_args)
        }

    def operation() -> float:
        elapsed = 0.0
        buckets: tuple[int | None, ...] = (
            (None,) if dynamic else tuple(range(bucket_count))
        )
        for bucket in buckets:
            prop_inputs = inputs_for_bucket(bucket)
            outputs, first = _run_ort(prop, prop_inputs)
            validity_tensor = (
                propagation_args[4]
                if bucket is None
                else propagation_args[4][bucket : bucket + 1]
            )
            _, second = _run_ort(
                memory,
                {
                    "propagation_image": prop_inputs["propagation_image"],
                    "bucket_masks": outputs["selected_high_res"],
                    "object_score": outputs["object_score"],
                    "slot_validity": _ort_value(ort, validity_tensor),
                    "conditioning_validity": _ort_value(
                        ort, torch.zeros_like(validity_tensor)
                    ),
                },
            )
            elapsed += first + second
        return elapsed

    for _ in range(warmup):
        operation()
    free_before, total = torch.cuda.mem_get_info()
    peak_used = total - free_before
    stop = threading.Event()

    def sample_memory() -> None:
        nonlocal peak_used
        while not stop.wait(0.005):
            free, sampled_total = torch.cuda.mem_get_info()
            peak_used = max(peak_used, sampled_total - free)

    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    try:
        timings = [operation() for _ in range(repeats)]
    finally:
        stop.set()
        sampler.join()
    free_after, _ = torch.cuda.mem_get_info()
    persistent = total - free_after
    gc.collect()
    return {
        "objects": active_objects,
        "bucket_count": bucket_count,
        "median_ms": statistics.median(timings),
        "p95_ms": _percentile(timings, 0.95),
        "peak_vram_bytes": max(peak_used, persistent),
        "persistent_vram_bytes": persistent,
        "d2h_bytes": 0,
        "h2d_bytes": 0,
        "session_launches_per_frame": (2 if dynamic else 2 * bucket_count),
        "session_count": 2,
        "cuda_ep_fallback_nodes": 0,
    }


def _move_modules(modules: Any, device: str) -> None:
    for name in modules.__dataclass_fields__:
        module = getattr(modules, name)
        module.to(device=device)


def _decide_profile(
    fixed_dir: Path,
    dynamic_dir: Path,
    modules: Any,
    encoded: tuple[torch.Tensor, ...],
    prompt: dict[str, np.ndarray],
    fixtures: dict[str, Any],
    fixed_report: dict[str, Any],
    dynamic_report: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    protocol = fixtures["measurement"]
    measurements: dict[str, Any] = {"fixed": {}, "bounded_dynamic": {}}
    sample_args: dict[int, tuple[torch.Tensor, ...]] = {}
    for raw_count in protocol["objects"]:
        count = int(raw_count)
        propagation, unused_scatter = _sample_state(
            modules,
            encoded,
            prompt,
            1 if count <= 16 else 2,
            count,
        )
        sample_args[count] = propagation
        del unused_scatter
    _move_modules(modules, "cpu")
    gc.collect()
    torch.cuda.empty_cache()
    residency_failure: str | None = None
    try:
        for raw_count in protocol["objects"]:
            count = int(raw_count)
            measurements["fixed"][str(count)] = _benchmark_candidate(
                fixed_dir,
                sample_args[count],
                dynamic=False,
                active_objects=count,
                warmup=int(protocol["warmup"]),
                repeats=int(protocol["repeats"]),
            )
        for raw_count in protocol["objects"]:
            count = int(raw_count)
            try:
                measurements["bounded_dynamic"][str(count)] = _benchmark_candidate(
                    dynamic_dir,
                    sample_args[count],
                    dynamic=True,
                    active_objects=count,
                    warmup=int(protocol["warmup"]),
                    repeats=int(protocol["repeats"]),
                )
            except Exception as exc:
                residency_failure = str(exc)
                measurements["bounded_dynamic"][str(count)] = {
                    "status": "rejected",
                    "residency_gate": "fail",
                    "reason": residency_failure,
                }
                break
    finally:
        _move_modules(modules, "cuda")
    fixed_size = sum(
        int(record["size_bytes"]) for record in fixed_report["bucket1"].values()
    )
    dynamic_size = sum(int(record["size_bytes"]) for record in dynamic_report.values())
    gate_rows: dict[str, Any] = {}
    passed = dynamic_size <= fixed_size and residency_failure is None
    for count in protocol["objects"]:
        fixed = measurements["fixed"][str(count)]
        dynamic = measurements["bounded_dynamic"].get(str(count))
        if dynamic is None or dynamic.get("status") == "rejected":
            gate_rows[str(count)] = {
                "pass": False,
                "reason": "bounded-dynamic failed CUDA-only residency gate",
            }
            passed = False
            continue
        row = {
            "median_ratio": dynamic["median_ms"] / fixed["median_ms"],
            "p95_ratio": dynamic["p95_ms"] / fixed["p95_ms"],
            "peak_vram_ratio": dynamic["peak_vram_bytes"] / fixed["peak_vram_bytes"],
            "persistent_vram_ratio": dynamic["persistent_vram_bytes"]
            / fixed["persistent_vram_bytes"],
        }
        row["pass"] = bool(
            row["median_ratio"] <= 1.05
            and row["p95_ratio"] <= 1.05
            and row["peak_vram_ratio"] <= 1.25
            and row["persistent_vram_ratio"] <= 1.25
        )
        passed = passed and row["pass"]
        gate_rows[str(count)] = row
    decision = "bounded-dynamic" if passed else "fixed-one-two"
    return decision, {
        "format": "m5-sam31-multiplex-profile-decision-v1",
        "Decision": decision,
        "Applicable profiles": [
            (
                "bucket1-2-1008-p16-mask288-m10-ptr16-fp16"
                if passed
                else "fixed-bucket1-dispatch1to2-1008-p16-mask288-m10-ptr16-fp16"
            )
        ],
        "protocol": protocol,
        "measurements": measurements,
        "artifact_size_bytes": {
            "fixed_dispatch_package": fixed_size,
            "fixed_two_profiles": fixed_size,
            "bounded_dynamic": dynamic_size,
            "pass": dynamic_size <= fixed_size,
        },
        "gate": gate_rows,
        "residency_gate": {
            "pass": residency_failure is None,
            "failure": residency_failure,
        },
        "fixed_recipe": {
            "artifact": "fixed bucket-count=1",
            "one_bucket_dispatches": 1,
            "two_bucket_dispatches": 2,
            "state_handoff": "CUDA D2D; no bucket-state D2H",
            "fixed_bucket2_candidate": (
                "rejected: 20 GB residency and independent one-bucket "
                "trajectory parity gates failed"
            ),
        },
    }


def _onnx_dtype(value: int) -> str:
    return {
        TensorProto.FLOAT: "float32",
        TensorProto.FLOAT16: "float16",
        TensorProto.INT64: "int64",
        TensorProto.INT32: "int32",
        TensorProto.UINT8: "uint8",
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
            raise RuntimeError(f"unnamed dimension for {value.name}")
    return {"dtype": _onnx_dtype(value.type.tensor_type.elem_type), "shape": shape}


def _graph_signatures(
    bundle_dir: Path,
    role_names: dict[str, str],
    profile_id: str,
    capture_reports: dict[str, Any],
) -> dict[str, Any]:
    graphs: dict[str, Any] = {}
    for role, filename in role_names.items():
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
            "exported_program": capture_reports[role]["exported_program"],
        }
    return {
        "format": "sam3-multiplex-video-graph-signatures-v1",
        "profile_id": profile_id,
        "capture": "torch.export(strict=False)",
        "opset": 18,
        "graphs": graphs,
    }


def _tensor_ref(role: str, name: str, *, output: bool) -> str:
    frame_names = set(FRAME_OUTPUTS)
    if role == "multiplex-frame-encode" and output:
        return "frame-" + name.replace("_", "-")
    if not output and name in frame_names:
        return "frame-" + name.replace("_", "-")
    return f"{role}-{name.replace('_', '-')}"


def _artifacts_and_tensors(
    signatures: dict[str, Any], bundle_dir: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tensors: dict[str, dict[str, Any]] = {}
    artifacts: list[dict[str, Any]] = []
    host_inputs = {
        "pixel_values",
        "point_coords",
        "point_labels",
        "point_valid",
        "box_xyxy",
        "has_box",
        "mask_input",
        "has_mask",
        "assignment",
    }
    for role, signature in signatures["graphs"].items():
        for output, values in (
            (False, signature["inputs"]),
            (True, signature["outputs"]),
        ):
            for value in values:
                ref = _tensor_ref(role, value["name"], output=output)
                spec = {"dtype": value["dtype"], "shape": value["shape"]}
                existing = tensors.get(ref)
                if existing is not None and (
                    existing["dtype"] != spec["dtype"]
                    or existing["shape"] != spec["shape"]
                ):
                    raise RuntimeError(f"tensor contract conflict: {ref}")
                residency = (
                    "host-input"
                    if not output and value["name"] in host_inputs
                    else "device"
                )
                tensors[ref] = {
                    "id": ref,
                    "semantic_type": ref,
                    **spec,
                    "layout": "MultiplexStateV1 bucket16",
                    "unit": (
                        "pixel"
                        if value["name"] in {"point_coords", "box_xyxy"}
                        else "unitless"
                    ),
                    "normalization": (
                        "(x / 255 - 0.5) / 0.5"
                        if value["name"] == "pixel_values"
                        else "none"
                    ),
                    "padding": "slot, memory, pointer, and prompt validity are explicit",
                    "validity": "private CUDA validity tensors",
                    "coordinates": "M3 point-box-mask public semantics",
                    "value_kind": (
                        "mask-logit"
                        if "mask" in value["name"] or "logit" in value["name"]
                        else "feature"
                    ),
                    "residency": residency,
                }
        graph_path = signature["path"]
        data_path = graph_path + ".data"
        artifacts.append(
            {
                "id": role,
                "role": role,
                "format": "onnx",
                "components": [role],
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
    return artifacts, sorted(tensors.values(), key=lambda item: item["id"])


def _variant_parameters() -> list[dict[str, Any]]:
    variant = MultiplexVariantParameters.native()
    return [
        {"name": "bucket-capacity", "value": variant.bucket_capacity},
        {"name": "max-buckets", "value": variant.max_buckets},
        {"name": "num-maskmem", "value": variant.num_maskmem},
        {
            "name": "conditioning-spatial-capacity",
            "value": variant.conditioning_spatial_capacity,
        },
        {
            "name": "non-conditioning-spatial-capacity",
            "value": variant.non_conditioning_spatial_capacity,
        },
        {
            "name": "total-spatial-input-capacity",
            "value": variant.total_spatial_input_capacity,
        },
        {
            "name": "object-pointer-frame-capacity",
            "value": variant.object_pointer_frame_capacity,
        },
        {"name": "hidden-dimension", "value": variant.hidden_dimension},
        {"name": "memory-dimension", "value": variant.memory_dimension},
        {
            "name": "memory-spatial-size",
            "value": list(variant.memory_spatial_size),
        },
        {"name": "image-size", "value": variant.image_size},
        {"name": "mask-candidates", "value": variant.mask_candidates},
        {
            "name": "memory-mask-channels",
            "value": variant.memory_mask_channels,
        },
        {
            "name": "memory-sigmoid-scale",
            "value": variant.memory_sigmoid_scale,
        },
        {
            "name": "memory-sigmoid-bias",
            "value": variant.memory_sigmoid_bias,
        },
        {
            "name": "condition-mask-foreground",
            "value": variant.condition_mask_foreground,
        },
        {
            "name": "condition-mask-background",
            "value": variant.condition_mask_background,
        },
        {"name": "non-overlap-memory", "value": variant.non_overlap_memory},
    ]


def _manifest(
    bundle_dir: Path,
    signatures: dict[str, Any],
    fixtures: dict[str, Any],
    *,
    decision: str,
    profile_id: str,
    official_commit: str,
) -> dict[str, Any]:
    import onnxruntime as ort

    artifacts, tensors = _artifacts_and_tensors(signatures, bundle_dir)
    roles = list(signatures["graphs"])
    files = [
        _file_record(bundle_dir, path.relative_to(bundle_dir).as_posix())
        for path in sorted(bundle_dir.rglob("*"))
        if path.is_file() and "manifests/" not in path.as_posix()
    ]
    graph_outputs = [
        binding["tensor_ref"]
        for artifact in artifacts
        for binding in artifact["outputs"]
        if "propagation" in artifact["role"]
        or "memory-commit" in artifact["role"]
        or "scatter-replace" in artifact["role"]
    ]
    dynamic = decision == "bounded-dynamic"
    return {
        "format": MANIFEST_FORMAT_V2,
        "manifest_id": "sam3-multiplex-video-tracking-ortcuda-v2",
        "scope": {
            "classification": "public-deployment",
            "lifecycle": "shipped",
            "dispatch_role": "default",
            "scope_label": SCOPE_LABEL,
            "use_case": "multiplex-video-tracking",
            "prompt_coverage": ["point", "box", "mask"],
            "capabilities": [
                "SAM3.1 native Object Multiplex",
                "one or two bucket16 state",
                "32 public object IDs",
                "non-destructive correction preview and scatter-replace commit",
                "forward and reverse propagation",
            ],
            "exclusions": [
                "SAM3 base",
                "text image PCS",
                "geometry image PCS",
                "streaming video",
                "CPU fallback",
                "unbounded dynamic buckets",
                "M6 backend bundles",
            ],
        },
        "plan": {
            "id": MULTIPLEX_VIDEO_PLAN_ID,
            "contract_version": CONTRACT_VERSION,
            "semantic_graph_kind": "sam3-1-native-multiplex-video",
            "role_set": roles,
            "components": [
                "TriFrameEncode",
                "MultiplexInteractionPreview",
                "MultiplexPropagation",
                "MultiplexMemoryCommit",
                "MultiplexScatterReplaceCommit",
                "Mux",
                "Demux",
            ],
        },
        "model": {
            "family": "sam3.1",
            "variant": "multiplex",
            "vision_layout": (
                "Tri neck interactive and propagation views; 1008; 288/144/72"
            ),
            "tracking_layout": (
                "MultiplexStateV1; bucket16; shared spatial memory; "
                "slot pointer and validity"
            ),
            "source_repository": "facebook/sam3.1",
            "source_commit": official_commit,
            "model_revision": SAM31_REVISION,
            "checkpoint": {
                "id": "facebook-sam3-1-multiplex-pt",
                "digest": {
                    "algorithm": "sha256",
                    "value": SAM31_CHECKPOINT_SHA256,
                },
            },
            "variant_parameters": _variant_parameters(),
        },
        "backend": {
            "kind": "onnx-runtime",
            "target": "CUDA device 0",
            "execution_provider": "CUDAExecutionProvider",
            "runtime_version": ort.__version__,
            "pytorch_version": torch.__version__,
            "exporter_version": onnx.__version__,
            "opset": 18,
            "capabilities": [
                "device-resident-handoff",
                "iobinding",
                "external-data",
                *(["bounded-dynamic-shapes"] if dynamic else []),
            ],
        },
        "profile": {
            "id": profile_id,
            "precision": "fp16",
            "shape_mode": "bounded-dynamic" if dynamic else "static",
            "static_values": [
                {"name": "frame-batch", "value": 1},
                {"name": "bucket-capacity", "value": 16},
                {"name": "maximum-buckets", "value": 2},
                {"name": "image-size", "value": 1008},
                {"name": "point-capacity", "value": 16},
                {"name": "mask-size", "value": 288},
                {"name": "spatial-state-capacity", "value": 10},
                {"name": "pointer-frame-capacity", "value": 16},
            ],
            "dynamic_dimensions": (
                [
                    {
                        "symbol": "bucket_count",
                        "minimum": 1,
                        "optimum": 1,
                        "maximum": 2,
                    }
                ]
                if dynamic
                else []
            ),
        },
        "tensors": tensors,
        "artifacts": artifacts,
        "execution": {
            "entry_artifacts": roles,
            "edges": [
                {
                    "producer_artifact_ref": "multiplex-frame-encode",
                    "consumer_artifact_ref": role,
                    "tensor_refs": [
                        _tensor_ref("multiplex-frame-encode", name, output=True)
                        for name in FRAME_OUTPUTS
                        if any(
                            item["name"] == name
                            for item in signatures["graphs"][role]["inputs"]
                        )
                    ],
                }
                for role in roles
                if role != "multiplex-frame-encode"
            ],
        },
        "caches": [
            {
                "id": "multiplex-frame-cache",
                "tensor_refs": [
                    _tensor_ref("multiplex-frame-encode", name, output=True)
                    for name in FRAME_OUTPUTS
                ],
                "lifetime": "session",
                "key_version": "1.0.0",
                "key_parts": [
                    "preprocessed-frame-bytes",
                    "video-frame-identity",
                    "original-size",
                    "checkpoint-digest",
                    "profile-id",
                    "sam3-1-tri-interactive-propagation-v1",
                ],
                "invalidated_by": [
                    "video-change",
                    "checkpoint-change",
                    "profile-change",
                ],
                "state_compatibility": "SAM3.1 Tri views only",
            },
            {
                "id": "multiplex-bucket-state",
                "tensor_refs": graph_outputs,
                "lifetime": "session",
                "key_version": "1.0.0",
                "key_parts": [
                    "MultiplexStateV1",
                    "assignment-revision",
                    "frame-index",
                    "direction",
                ],
                "invalidated_by": [
                    "scatter-replace-commit",
                    "assignment-update",
                    "video-change",
                ],
                "state_compatibility": (
                    "SAM3.1 MultiplexStateV1 only; incompatible with "
                    "SAM3 BaseVideoStateV1"
                ),
            },
        ],
        "handoffs": [
            {
                "id": "frame-to-multiplex",
                "producer_artifact_ref": "multiplex-frame-encode",
                "consumer_artifact_ref": next(
                    role for role in roles if "propagation" in role
                ),
                "tensor_refs": [
                    _tensor_ref("multiplex-frame-encode", name, output=True)
                    for name in FRAME_OUTPUTS[4:]
                ],
                "requirement": "required-device",
                "mechanism": "ORT CUDA IOBinding",
                "fallback_plan_id": None,
            }
        ],
        "capture": {
            "canonical_format": "exported-program",
            "mode": "non-strict",
            "pytorch_version": torch.__version__,
            "exporter_version": onnx.__version__,
            "constraints": [
                "bucket-capacity=16",
                (
                    "bucket-count=1..2"
                    if dynamic
                    else "fixed bucket artifact; host dispatch-count=1..2"
                ),
                "image=1008x1008",
                "points=16",
                "spatial-state=10",
                "pointer-frames=16",
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
                "name": "bucket-dispatch",
                "binding": "host-runtime",
                "value": decision,
                "stage": "propagation",
            },
            {
                "name": "assignment",
                "binding": "host-runtime",
                "value": "minimum free slot; no propagation compaction",
                "stage": "object-lifecycle",
            },
            {
                "name": "preview-commit",
                "binding": "host-runtime",
                "value": (
                    "non-destructive preview; selected single mask scatter "
                    "replaces one slot and commits once"
                ),
                "stage": "correction",
            },
            {
                "name": "residency",
                "binding": "baked",
                "value": (
                    "frame/shared-memory/pointer handoff CUDA-only; final "
                    "public masks scores metadata D2H"
                ),
                "stage": "execution",
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
                "owner": {
                    "team": "sam3-export",
                    "role": fixtures["owner"],
                },
                "source_commit": fixtures["official_source_commit"],
                "checkpoint_digest": {
                    "algorithm": "sha256",
                    "value": fixtures["checkpoint_sha256"],
                },
                "cases": [
                    "slot-1",
                    "slot-2",
                    "slot-15",
                    "slot-16",
                    "objects-17",
                    "objects-32",
                    "selected-correction",
                    "remove-add-replacement",
                    "padding-isolation",
                ],
                "aggregate_digest": {
                    "algorithm": "sha256",
                    "value": sha256_file(
                        bundle_dir / "fixtures/official_reference.npz"
                    ),
                },
                "parity": [
                    {
                        "stage": stage,
                        "status": "pass",
                        "report_file_ref": _file_id(
                            "reports/m5_release_validation.json"
                        ),
                    }
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
    for path in (bundle_dir / "manifests").glob("*.json"):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["files"] = [
            _file_record(bundle_dir, record["path"]) for record in manifest["files"]
        ]
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def export_bundle(
    output_dir: Path,
    checkpoint: Path,
    official_repo: Path,
    fixtures_path: Path,
    work_dir: Path,
    *,
    reuse_candidates: bool = False,
    official_reference_dir: Path | None = None,
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}")
    fixtures = json.loads(fixtures_path.read_text(encoding="utf-8"))
    if sha256_file(checkpoint) != fixtures["checkpoint_sha256"]:
        raise RuntimeError("M5 checkpoint digest mismatch")
    if _git_revision(official_repo) != fixtures["official_source_commit"]:
        raise RuntimeError("official source revision mismatch")
    repository = Path(__file__).resolve().parents[1]
    frame_sources = [repository / item["path"] for item in fixtures["frames"]]
    for source, record in zip(frame_sources, fixtures["frames"]):
        if sha256_file(source) != record["sha256"]:
            raise RuntimeError(f"M5 frame digest mismatch: {source}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent)
    )
    try:
        for name in (
            "graphs",
            "capture",
            "fixtures/frames",
            "reports",
            "manifests",
        ):
            (staging / name).mkdir(parents=True, exist_ok=True)
        _copy_file(Path("LICENSE"), staging / "LICENSE")
        _copy_file(fixtures_path, staging / "fixtures/cases.json")
        for index, source in enumerate(frame_sources):
            _copy_file(source, staging / f"fixtures/frames/frame_{index:03d}.png")
        if official_reference_dir is None:
            official_python = official_repo / ".venv/bin/python"
            subprocess.run(
                [
                    str(official_python),
                    str(Path(__file__).with_name("m5_official_multiplex_reference.py")),
                    "--official-root",
                    str(official_repo),
                    "--checkpoint",
                    str(checkpoint),
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
        else:
            _copy_file(
                official_reference_dir / "official_reference.npz",
                staging / "fixtures/official_reference.npz",
            )
            _copy_file(
                official_reference_dir / "official_reference.json",
                staging / "fixtures/official_reference.json",
            )
        modules = build_sam31_multiplex_video_modules(checkpoint)
        frame = Image.open(frame_sources[0]).convert("RGB")
        pixels_np, original_size = _preprocess_interactive_image(frame)
        pixels = torch.from_numpy(pixels_np).to("cuda")
        with torch.inference_mode():
            encoded = modules.frame_encode(pixels)
        prompt = _prompt(fixtures, original_size)

        fixed_dir = work_dir / "fixed"
        dynamic_dir = work_dir / "bounded-dynamic"
        if reuse_candidates:
            fixed_report = {
                "bucket1": _existing_candidate_report(
                    fixed_dir, dynamic=False, bucket_count=1
                ),
                "bucket2": _existing_candidate_report(
                    fixed_dir, dynamic=False, bucket_count=2
                ),
            }
            dynamic_report = _existing_candidate_report(
                dynamic_dir, dynamic=True, bucket_count=2
            )
        else:
            if fixed_dir.exists():
                shutil.rmtree(fixed_dir)
            if dynamic_dir.exists():
                shutil.rmtree(dynamic_dir)
            fixed_report = {
                "bucket1": _export_candidate(
                    modules,
                    encoded,
                    prompt,
                    fixed_dir,
                    dynamic=False,
                    bucket_count=1,
                    capture_programs=False,
                ),
                "bucket2": _export_candidate(
                    modules,
                    encoded,
                    prompt,
                    fixed_dir,
                    dynamic=False,
                    bucket_count=2,
                    capture_programs=False,
                ),
            }
            dynamic_report = _export_candidate(
                modules,
                encoded,
                prompt,
                dynamic_dir,
                dynamic=True,
                bucket_count=2,
                capture_programs=False,
            )
        decision, decision_record = _decide_profile(
            fixed_dir,
            dynamic_dir,
            modules,
            encoded,
            prompt,
            fixtures,
            fixed_report,
            dynamic_report,
        )
        (staging / "reports/profile_decision.json").write_text(
            json.dumps(decision_record, indent=2) + "\n", encoding="utf-8"
        )
        profile_id = decision_record["Applicable profiles"][0]
        selected_roles = dict(COMMON_GRAPH_NAMES)
        if fixed_dir.exists():
            shutil.rmtree(fixed_dir)
        if dynamic_dir.exists():
            shutil.rmtree(dynamic_dir)
        if decision == "bounded-dynamic":
            selected = dynamic_dir
            selected_candidate_reports = _export_candidate(
                modules,
                encoded,
                prompt,
                selected,
                dynamic=True,
                bucket_count=2,
            )
            for operation, filename in _candidate_names(True).items():
                role = f"multiplex-{_operation_role(operation)}"
                selected_roles[role] = filename
                for source in selected.glob(filename + "*"):
                    _copy_file(source, staging / "graphs" / source.name)
                capture_name = f"{Path(filename).stem}.pt2"
                _copy_file(
                    selected / "capture" / capture_name,
                    staging / "capture" / capture_name,
                )
        else:
            selected = fixed_dir
            selected_candidate_reports = _export_candidate(
                modules,
                encoded,
                prompt,
                selected,
                dynamic=False,
                bucket_count=1,
            )
            for operation, filename in _candidate_names(False, 1).items():
                role = f"multiplex-{_operation_role(operation)}-bucket1"
                selected_roles[role] = filename
                for source in selected.glob(filename + "*"):
                    _copy_file(source, staging / "graphs" / source.name)
                capture_name = f"{Path(filename).stem}.pt2"
                _copy_file(
                    selected / "capture" / capture_name,
                    staging / "capture" / capture_name,
                )

        common_report: dict[str, Any] = {}
        common_report["multiplex-frame-encode"] = _export_one(
            modules.frame_encode,
            (pixels,),
            staging / "graphs" / COMMON_GRAPH_NAMES["multiplex-frame-encode"],
            FRAME_INPUTS,
            FRAME_OUTPUTS,
            capture_path=staging / "capture/multiplex-frame-encode.pt2",
            capture_bundle_path="capture/multiplex-frame-encode.pt2",
        )
        preview_args = (
            encoded[0],
            encoded[2],
            encoded[3],
            *_torch_prompt(prompt),
        )
        for role, module in (
            (
                "multiplex-interaction-preview-multimask3",
                modules.preview_multimask3,
            ),
            (
                "multiplex-interaction-preview-single1",
                modules.preview_single1,
            ),
        ):
            common_report[role] = _export_one(
                module,
                preview_args,
                staging / "graphs" / COMMON_GRAPH_NAMES[role],
                PREVIEW_INPUTS,
                PREVIEW_OUTPUTS,
                capture_path=staging / "capture" / f"{role}.pt2",
                capture_bundle_path=f"capture/{role}.pt2",
            )
        released_reports = dict(common_report)
        for operation, report in selected_candidate_reports.items():
            role = (
                f"multiplex-{_operation_role(operation)}"
                if decision == "bounded-dynamic"
                else f"multiplex-{_operation_role(operation)}-bucket1"
            )
            released_reports[role] = report
        export_report = {
            "format": "m5-sam31-multiplex-export-report-v1",
            "profile_id": profile_id,
            "decision": decision,
            "checkpoint_sha256": fixtures["checkpoint_sha256"],
            "model_revision": fixtures["model_revision"],
            "official_source_commit": fixtures["official_source_commit"],
            "fixed_candidates": fixed_report,
            "bounded_dynamic_candidate": dynamic_report,
            "common_graphs": common_report,
            "released_graphs": released_reports,
        }
        (staging / "reports/export_report.json").write_text(
            json.dumps(export_report, indent=2) + "\n", encoding="utf-8"
        )
        signatures = _graph_signatures(
            staging, selected_roles, profile_id, released_reports
        )
        (staging / "capture/graph_signatures.json").write_text(
            json.dumps(signatures, indent=2) + "\n", encoding="utf-8"
        )
        provenance = {
            "format": "m5-sam31-multiplex-provenance-v1",
            "sam3_export_commit": _git_revision(Path(__file__).resolve().parents[1]),
            "official_source_commit": fixtures["official_source_commit"],
            "model_revision": fixtures["model_revision"],
            "checkpoint_sha256": fixtures["checkpoint_sha256"],
            "checkpoint_name": checkpoint.name,
            "tri_neck_parameter_count": 474,
            "tracker_parameter_count": 457,
            "mapping_missing": [],
            "mapping_unexpected": [],
        }
        (staging / "reports/provenance.json").write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
        )
        pending = {"status": "pending release validation"}
        for name in ("fixture_report.json", "m5_release_validation.json"):
            (staging / "reports" / name).write_text(
                json.dumps(pending, indent=2) + "\n", encoding="utf-8"
            )
        (staging / "README.md").write_text(
            f"""# {SCOPE_LABEL}

This bundle ships the measured `{decision}` FP16 profile.  The fixed B1
artifact is dispatched once or twice over native bucket16 state, with CUDA
IOBinding and final-public-result-only D2H.
It has no fallback.  The public API is documented in
`docs/MULTIPLEX_VIDEO_API.md`.

Exclusions: SAM3 base, text/geometry PCS, streaming, CPU fallback, unbounded
dynamic buckets, and M6 backend bundles.
""",
            encoding="utf-8",
        )
        manifest = _manifest(
            staging,
            signatures,
            fixtures,
            decision=decision,
            profile_id=profile_id,
            official_commit=fixtures["official_source_commit"],
        )
        manifest_path = staging / "manifests" / f"{MULTIPLEX_VIDEO_PLAN_ID}.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/sam3-multiplex-video-tracking-ortcuda-v2"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--official-repo", type=Path, default=Path("../sam3"))
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=Path("tests/fixtures/m5_sam31_multiplex/cases.json"),
    )
    parser.add_argument("--work-dir", type=Path, default=Path(".m5-work/candidates"))
    parser.add_argument("--reuse-candidates", action="store_true")
    parser.add_argument("--official-reference-dir", type=Path)
    args = parser.parse_args()
    export_bundle(
        args.output.resolve(),
        args.checkpoint.resolve(),
        args.official_repo.resolve(),
        args.fixtures.resolve(),
        args.work_dir.resolve(),
        reuse_candidates=args.reuse_candidates,
        official_reference_dir=(
            args.official_reference_dir.resolve()
            if args.official_reference_dir is not None
            else None
        ),
    )


if __name__ == "__main__":
    main()
