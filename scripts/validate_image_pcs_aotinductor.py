"""Validate the M6 AOTInductor image PCS candidate and Public API parity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time
from typing import Any

from export_image_pcs_v2 import _file_id, refresh_manifest_file_records
import numpy as np
from PIL import Image
import torch
from validate_image_pcs_v2 import (
    _expected_boxes,
    _expected_masks,
    _mask_iou,
)

from sam3.runtime import PredictOptions, create_image_session
from sam3.runtime.manifest import (
    AOTINDUCTOR_PLAN_ID,
    EXPORTED_PROGRAM_PLAN_ID,
    validate_manifest_package,
)

WARMUP = 2
REPEATS = 5


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(np.ceil(fraction * len(ordered))) - 1))
    return ordered[index]


def _environment() -> dict[str, Any]:
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
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": gpu,
    }


def _benchmark(session: Any) -> dict[str, Any]:
    for _ in range(WARMUP):
        session.predict_text(PredictOptions(score_threshold=0.5))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    persistent = torch.cuda.memory_allocated()
    timings: list[float] = []
    for _ in range(REPEATS):
        started = time.perf_counter()
        session.predict_text(PredictOptions(score_threshold=0.5))
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - started) * 1000.0)
    return {
        "warmup": WARMUP,
        "repeats": REPEATS,
        "median_ms": statistics.median(timings),
        "p95_ms": _percentile(timings, 0.95),
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "persistent_vram_bytes": persistent,
    }


def validate_bundle(
    bundle_dir: Path, plan_id: str, *, allow_parity_failure: bool = False
) -> dict[str, Any]:
    resolved = validate_manifest_package(
        bundle_dir / "manifests" / f"{plan_id}.json",
        expected_plan_id=plan_id,
    )
    fixtures = json.loads(
        (bundle_dir / "fixtures/cases.json").read_text(encoding="utf-8")
    )
    reference = np.load(bundle_dir / "fixtures/official_reference.npz")
    image_records = {item["id"]: item for item in fixtures["images"]}
    session = create_image_session(plan_id, bundle_dir=bundle_dir)
    cases: list[dict[str, Any]] = []
    failures: list[str] = []
    benchmark: dict[str, Any] | None = None
    try:
        for case_index, case in enumerate(fixtures["cases"]):
            image_name = Path(image_records[case["image"]]["workspace_path"]).name
            image = Image.open(bundle_dir / "fixtures/images" / image_name).convert(
                "RGB"
            )
            image_handle = session.set_image(image)
            assert session.set_image(image) == image_handle
            prompt_handle = session.set_text(case["text"])
            assert session.set_text(case["text"]) == prompt_handle
            prediction = session.predict_text(PredictOptions(score_threshold=0.5))
            if case_index == 0:
                benchmark = _benchmark(session)
            prefix = case["id"]
            official_scores = reference[f"{prefix}__scores"]
            expected_indices = np.flatnonzero(official_scores > 0.5)
            expected_indices = expected_indices[
                np.lexsort((expected_indices, -official_scores[expected_indices]))
            ]
            actual_indices = session._last_query_indices
            official_top = reference[f"{prefix}__indices"]
            top_positions = {
                int(index): offset for offset, index in enumerate(official_top)
            }
            positions = np.asarray(
                [top_positions[int(index)] for index in expected_indices],
                dtype=np.int64,
            )
            expected_scores = official_scores[expected_indices].astype(np.float32)
            expected_boxes = _expected_boxes(
                reference[f"{prefix}__boxes"][positions],
                image_handle.original_size,
            )
            expected_masks = _expected_masks(
                reference[f"{prefix}__masks"][positions],
                image_handle.original_size,
            )
            indices_match = bool(np.array_equal(expected_indices, actual_indices))
            mask_iou = (
                _mask_iou(expected_masks, prediction.masks)
                if indices_match and expected_masks.size
                else (1.0 if indices_match else 0.0)
            )
            if not indices_match:
                message = (
                    f"{prefix}: admitted query indices do not match: "
                    f"expected={expected_indices.tolist()} "
                    f"actual={actual_indices.tolist()} "
                    f"actual_scores={prediction.scores.tolist()}"
                )
                if not allow_parity_failure:
                    raise RuntimeError(message)
                failures.append(message)
            if mask_iou < 0.98:
                message = f"{prefix}: task mask IoU {mask_iou:.6f} < 0.98"
                if not allow_parity_failure:
                    raise RuntimeError(message)
                failures.append(message)
            cases.append(
                {
                    "id": prefix,
                    "admitted_indices_exact": indices_match,
                    "expected_admitted_indices": expected_indices.tolist(),
                    "actual_admitted_indices": actual_indices.tolist(),
                    "score_max_abs": (
                        float(np.max(np.abs(expected_scores - prediction.scores)))
                        if indices_match and expected_scores.size
                        else (0.0 if indices_match else None)
                    ),
                    "box_max_abs_pixels": (
                        float(np.max(np.abs(expected_boxes - prediction.boxes_xyxy)))
                        if indices_match and expected_boxes.size
                        else (0.0 if indices_match else None)
                    ),
                    "task_mask_iou": mask_iou,
                }
            )
        counters = dict(session._adapter.counters)
        if counters["image_encodes"] != len(fixtures["images"]):
            message = f"image cache reuse gate failed: {counters}"
            if not allow_parity_failure:
                raise RuntimeError(message)
            failures.append(message)
        if counters["text_encodes"] != len(fixtures["cases"]):
            message = f"text cache reuse gate failed: {counters}"
            if not allow_parity_failure:
                raise RuntimeError(message)
            failures.append(message)
    finally:
        session.close()

    artifact_size = sum(
        int(record["size_bytes"])
        for record in resolved.manifest["files"]
        if record["path"].startswith(("packages/", "capture/"))
    )
    return {
        "format": "sam3-image-pcs-m6-aotinductor-validation-v1",
        "status": "fail" if failures else "pass",
        "plan_id": plan_id,
        "backend_kind": resolved.manifest["backend"]["kind"],
        "profile_id": resolved.profile_id,
        "environment": _environment(),
        "cases": cases,
        "benchmark": benchmark,
        "artifact_size_bytes": artifact_size,
        "cache_and_transfer_counters": counters,
        "failures": failures,
        "gates": {
            "admitted_indices": "exact for every fixture at strict > 0.5",
            "task_mask_iou_minimum": 0.98,
            "cache": "same image/text keys do not rerun encoders",
            "residency": "all role outputs asserted CUDA before handoff",
            "d2h": "final public logits/boxes/masks/presence only",
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--update-report", action="store_true")
    parser.add_argument(
        "--plan-id",
        choices=[AOTINDUCTOR_PLAN_ID, EXPORTED_PROGRAM_PLAN_ID],
        default=AOTINDUCTOR_PLAN_ID,
    )
    parser.add_argument("--record-failure", action="store_true")
    return parser.parse_args()


def main() -> int:
    arguments = _parse_args()
    bundle_dir = arguments.bundle_dir.resolve()
    try:
        report = validate_bundle(
            bundle_dir,
            arguments.plan_id,
            allow_parity_failure=arguments.record_failure,
        )
    except RuntimeError as exc:
        if not arguments.record_failure:
            raise
        report = {
            "format": "sam3-image-pcs-m6-backend-validation-v1",
            "status": "fail",
            "plan_id": arguments.plan_id,
            "failure": str(exc),
            "environment": _environment(),
        }
    if not arguments.update_report:
        print(json.dumps(report, indent=2))
        return 0

    report_name = (
        "m6_aotinductor_validation.json"
        if arguments.plan_id == AOTINDUCTOR_PLAN_ID
        else "m6_exported_program_validation.json"
    )
    report_path = bundle_dir / "reports" / report_name
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    manifest_path = bundle_dir / "manifests" / f"{arguments.plan_id}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report_ref = _file_id(f"reports/{report_name}")
    for parity in manifest["fixtures"][0]["parity"]:
        if parity["stage"] in {
            "exported-program-to-backend",
            "end-to-end-behavior",
        }:
            parity["status"] = report["status"]
            parity["report_file_ref"] = report_ref
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    refresh_manifest_file_records(bundle_dir)
    validate_manifest_package(manifest_path, expected_plan_id=arguments.plan_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
