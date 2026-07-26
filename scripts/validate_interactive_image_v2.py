"""Run the M3 CUDA/Public API release gate for an interactive image bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import statistics
import subprocess
import tempfile
import time
from typing import Any

from export_interactive_image_v2 import (
    _case_prompt,
    _compare_case,
    refresh_manifest_file_records,
)
import numpy as np
from PIL import Image

from sam3.runtime import (
    InteractivePredictOptions,
    InteractivePrompt,
    create_interactive_session,
)
from sam3.runtime.manifest import INTERACTIVE_PLAN_ID, validate_manifest_package


def _gpu_used_bytes() -> int:
    output = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()[0]
    return int(output.strip()) * 1024 * 1024


def _environment() -> dict[str, Any]:
    import onnxruntime as ort

    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "python": platform.python_version(),
        "onnxruntime": ort.__version__,
        "providers": ort.get_available_providers(),
        "gpu": gpu,
    }


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def _validate_no_internal_d2h(bundle_dir: Path) -> dict[str, list[str]]:
    import onnx
    import onnxruntime as ort

    result: dict[str, list[str]] = {}
    with tempfile.TemporaryDirectory(prefix="sam3-m3-ort-") as temp_dir:
        for graph_path in sorted((bundle_dir / "graphs").glob("*.onnx")):
            optimized = Path(temp_dir) / graph_path.name
            options = ort.SessionOptions()
            options.optimized_model_filepath = str(optimized)
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            ort.InferenceSession(
                str(graph_path),
                sess_options=options,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            graph = onnx.load(optimized, load_external_data=False)
            copies = [
                node.name for node in graph.graph.node if node.op_type == "MemcpyToHost"
            ]
            if copies:
                raise RuntimeError(
                    f"internal D2H is forbidden for {graph_path.name}: {copies}"
                )
            result[graph_path.name] = copies
    return result


def validate_bundle(bundle_dir: Path) -> dict[str, Any]:
    resolved = validate_manifest_package(
        bundle_dir / "manifests" / f"{INTERACTIVE_PLAN_ID}.json",
        expected_plan_id=INTERACTIVE_PLAN_ID,
    )
    fixtures = json.loads(
        (bundle_dir / "fixtures/cases.json").read_text(encoding="utf-8")
    )
    reference = np.load(bundle_dir / "fixtures/local_reference.npz")
    export_report = json.loads(
        (bundle_dir / "reports/export_report.json").read_text(encoding="utf-8")
    )
    internal_d2h = _validate_no_internal_d2h(bundle_dir)
    image = Image.open(bundle_dir / "fixtures/images/truck.jpg").convert("RGB")
    session = create_interactive_session(INTERACTIVE_PLAN_ID, bundle_dir=bundle_dir)
    cases: list[dict[str, Any]] = []
    before_vram = _gpu_used_bytes()
    vram_samples = [{"stage": "before-session-work", "bytes": before_vram}]
    try:
        handle = session.set_image(image)
        vram_samples.append({"stage": "after-image-encode", "bytes": _gpu_used_bytes()})
        if session.set_image(image) != handle:
            raise RuntimeError("same-image cache key changed")
        cache = session._image_cache
        if not isinstance(cache, dict) or any(
            value.device_name() != "cuda" for value in cache.values()
        ):
            raise RuntimeError("image feature cache is not fully CUDA resident")

        for case in fixtures["cases"]:
            prediction = session.predict(
                _case_prompt(case),
                InteractivePredictOptions(
                    multimask_output=bool(case["multimask_output"])
                ),
            )
            prefix = case["id"]
            result = _compare_case(
                reference[f"{prefix}__scores"],
                reference[f"{prefix}__low_res"],
                prediction.scores,
                prediction.low_res_logits,
            )
            if not result["pass"]:
                raise RuntimeError(f"ORT/Public parity failed for {prefix}: {result}")
            if prediction.masks.dtype != np.bool_:
                raise RuntimeError(f"Public masks are not bool for {prefix}")
            cases.append({"id": prefix, **result})
        vram_samples.append(
            {"stage": "after-fixture-cases", "bytes": _gpu_used_bytes()}
        )

        repeated = fixtures["repeated_click"]
        first_prompt = InteractivePrompt(
            points_xy=np.asarray(repeated["first_points_xy"], dtype=np.float32),
            point_labels=np.asarray(repeated["first_point_labels"], dtype=np.int64),
        )
        first = session.predict(first_prompt)
        selected_index = int(np.argmax(first.scores))
        second = session.predict(
            InteractivePrompt(
                points_xy=np.asarray(repeated["second_points_xy"], dtype=np.float32),
                point_labels=np.asarray(
                    repeated["second_point_labels"], dtype=np.int64
                ),
                mask_logits=first.low_res_logits[selected_index],
            ),
            InteractivePredictOptions(multimask_output=False),
        )
        prefix = repeated["id"]
        first_result = _compare_case(
            reference[f"{prefix}__first_scores"],
            reference[f"{prefix}__first_low_res"],
            first.scores,
            first.low_res_logits,
        )
        second_result = _compare_case(
            reference[f"{prefix}__second_scores"],
            reference[f"{prefix}__second_low_res"],
            second.scores,
            second.low_res_logits,
        )
        selected_match = selected_index == int(reference[f"{prefix}__selected_index"])
        if not first_result["pass"] or not second_result["pass"] or not selected_match:
            raise RuntimeError("repeated-click parity gate failed")

        fixture_counters = dict(session._adapter.counters)
        expected_predicts = len(fixtures["cases"]) + 2
        if fixture_counters["image_encodes"] != 1:
            raise RuntimeError(f"image cache reuse failed: {fixture_counters}")
        if fixture_counters["predict_launches"] != expected_predicts:
            raise RuntimeError(f"predict launch count failed: {fixture_counters}")
        if (
            fixture_counters["memory_encodes"] != 0
            or fixture_counters["memory_commits"] != 0
        ):
            raise RuntimeError(f"M3 launched memory work: {fixture_counters}")
        expected_d2h = (
            sum(
                ((3 if case["multimask_output"] else 1) * (288 * 288 + 1) * 4)
                for case in fixtures["cases"]
            )
            + (3 * (288 * 288 + 1) * 4)
            + ((288 * 288 + 1) * 4)
        )
        if fixture_counters["d2h_bytes"] != expected_d2h:
            raise RuntimeError(
                "D2H must contain only final scores and low-resolution logits: "
                f"{fixture_counters['d2h_bytes']} != {expected_d2h}"
            )

        measure_prompt = first_prompt
        warmup = int(fixtures["measurement"]["warmup"])
        repeats = int(fixtures["measurement"]["repeats"])
        for _ in range(warmup):
            session.predict(measure_prompt)
        vram_samples.append({"stage": "after-warmup", "bytes": _gpu_used_bytes()})
        timings: list[float] = []
        for _ in range(repeats):
            start = time.perf_counter()
            session.predict(measure_prompt)
            timings.append((time.perf_counter() - start) * 1000.0)
        after_vram = _gpu_used_bytes()
        vram_samples.append({"stage": "after-measurement", "bytes": after_vram})
        performance = {
            "warmup": warmup,
            "repeats": repeats,
            "median_ms": statistics.median(timings),
            "p95_ms": _percentile(timings, 0.95),
            "persistent_vram_bytes_before": before_vram,
            "persistent_vram_bytes_after": after_vram,
            "observed_peak_vram_bytes": max(
                int(sample["bytes"]) for sample in vram_samples
            ),
            "vram_samples": vram_samples,
            "vram_measurement": "device-used bytes sampled at release-stage boundaries",
        }
    finally:
        session.close()

    artifact_hashes = {
        record["path"]: record["digest"]["value"]
        for record in resolved.manifest["files"]
        if record["role"] in {"graph", "external-data"}
    }
    return {
        "format": "sam3-interactive-image-m3-release-validation-v1",
        "status": "pass",
        "profile_id": resolved.profile_id,
        "plan_id": resolved.plan_id,
        "manifest_sha256": resolved.manifest_digest,
        "sam3_export_commit": export_report["sam3_export_commit"],
        "official_commit": export_report["official_commit"],
        "checkpoint_sha256": export_report["checkpoint_sha256"],
        "environment": _environment(),
        "artifact_hashes": artifact_hashes,
        "stages": {
            "official_to_local_eager": export_report["official_to_local"],
            "local_eager_to_exported_program": {
                role: graph["eager_to_exported_program"]
                for role, graph in export_report["graphs"].items()
            },
            "exported_program_to_ort_cuda": cases,
            "public_api": {
                "masks": "bool [M,H,W], bilinear resize, strict > threshold",
                "low_res_logits": "float32 [M,288,288] reusable as mask prompt",
            },
        },
        "repeated_click": {
            "first": first_result,
            "selected_index_match": selected_match,
            "second": second_result,
            "image_encode_count": fixture_counters["image_encodes"],
            "predict_launch_count": 2,
            "memory_encode_count": fixture_counters["memory_encodes"],
            "memory_commit_count": fixture_counters["memory_commits"],
        },
        "cache_residency_and_copies": {
            "image_features": "three CUDA OrtValues retained through every prediction",
            "fallback": None,
            "internal_memcpy_to_host_nodes": internal_d2h,
            "fixture_counters": fixture_counters,
            "d2h_policy": "scores and final low-resolution logits only",
            "h2d_policy": "preprocessed image once; fixed prompt tensors per prediction",
        },
        "performance": performance,
        "gates": {
            "task_mask_iou_minimum": 0.98,
            "score_max_abs_maximum": 0.02,
            "low_res_logit_mean_abs_maximum": 0.05,
            "top_score_index": "exact",
            "point_capacities": [0, 1, 16],
            "memory_launches": 0,
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--update-report", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    bundle_dir = args.bundle_dir.resolve()
    report = validate_bundle(bundle_dir)
    if args.update_report:
        report_path = bundle_dir / "reports/m3_release_validation.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        refresh_manifest_file_records(bundle_dir)
        validate_manifest_package(
            bundle_dir / "manifests" / f"{INTERACTIVE_PLAN_ID}.json",
            expected_plan_id=INTERACTIVE_PLAN_ID,
        )
    else:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
