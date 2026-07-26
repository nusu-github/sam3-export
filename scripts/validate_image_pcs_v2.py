"""Run the M2 CUDA/Public API release gate for an image PCS v2 bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import subprocess
from typing import Any

from export_image_pcs_v2 import refresh_manifest_file_records
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from sam3.runtime import PredictOptions, create_image_session
from sam3.runtime.manifest import (
    DEFAULT_PLAN_ID,
    SELECTED_K32_PLAN_ID,
    SPLIT_PLAN_ID,
    validate_manifest_package,
)

PLAN_IDS = (DEFAULT_PLAN_ID, SELECTED_K32_PLAN_ID, SPLIT_PLAN_ID)


def _mask_iou(expected: np.ndarray, actual: np.ndarray) -> float:
    expected_binary = expected > 0.5
    actual_binary = actual > 0.5
    intersection = np.logical_and(expected_binary, actual_binary).sum(axis=(-2, -1))
    union = np.logical_or(expected_binary, actual_binary).sum(axis=(-2, -1))
    return float(np.mean(np.where(union == 0, 1.0, intersection / union)))


def _expected_boxes(boxes: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    height, width = size
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
    result[:, 0::2] = np.clip(result[:, 0::2], 0, width)
    result[:, 1::2] = np.clip(result[:, 1::2], 0, height)
    return result


def _expected_masks(logits: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    values = torch.sigmoid(torch.from_numpy(logits).float()).unsqueeze(1)
    return F.interpolate(values, size=size, mode="bilinear", align_corners=False)[
        :, 0
    ].numpy()


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
        "torch": torch.__version__,
        "onnxruntime": ort.__version__,
        "providers": ort.get_available_providers(),
        "gpu": gpu,
    }


def validate_bundle(bundle_dir: Path) -> dict[str, Any]:
    manifests = {
        plan_id: validate_manifest_package(
            bundle_dir / "manifests" / f"{plan_id}.json",
            expected_plan_id=plan_id,
        )
        for plan_id in PLAN_IDS
    }
    fixtures = json.loads(
        (bundle_dir / "fixtures" / "cases.json").read_text(encoding="utf-8")
    )
    reference = np.load(bundle_dir / "fixtures" / "official_reference.npz")
    export_report = json.loads(
        (bundle_dir / "reports" / "export_report.json").read_text(encoding="utf-8")
    )
    image_records = {item["id"]: item for item in fixtures["images"]}
    plan_reports: dict[str, Any] = {}
    for plan_id in PLAN_IDS:
        session = create_image_session(plan_id, bundle_dir=bundle_dir)
        cases: list[dict[str, Any]] = []
        try:
            for case in fixtures["cases"]:
                image_name = Path(image_records[case["image"]]["workspace_path"]).name
                image = Image.open(
                    bundle_dir / "fixtures" / "images" / image_name
                ).convert("RGB")
                image_handle = session.set_image(image)
                # The second call must reuse the same device cache.
                assert session.set_image(image) == image_handle
                prompt_handle = session.set_text(case["text"])
                assert session.set_text(case["text"]) == prompt_handle
                prediction = session.predict_text(PredictOptions(score_threshold=0.5))
                prefix = case["id"]
                official_scores = reference[f"{prefix}__scores"]
                expected_indices = np.flatnonzero(official_scores > 0.5)
                expected_indices = expected_indices[
                    np.lexsort((expected_indices, -official_scores[expected_indices]))
                ]
                actual_indices = session._last_query_indices  # release-only diagnostic
                official_top = reference[f"{prefix}__indices"]
                top_positions = {
                    int(index): offset for offset, index in enumerate(official_top)
                }
                if any(int(index) not in top_positions for index in expected_indices):
                    raise RuntimeError(
                        "official reference does not retain an admitted mask"
                    )
                positions = np.asarray(
                    [top_positions[int(index)] for index in expected_indices],
                    dtype=np.int64,
                )
                expected_scores = official_scores[expected_indices].astype(np.float32)
                expected_boxes = _expected_boxes(
                    reference[f"{prefix}__boxes"][positions], image_handle.original_size
                )
                expected_masks = _expected_masks(
                    reference[f"{prefix}__masks"][positions], image_handle.original_size
                )
                score_max_abs = (
                    float(np.max(np.abs(expected_scores - prediction.scores)))
                    if expected_scores.size
                    else 0.0
                )
                box_max_abs = (
                    float(np.max(np.abs(expected_boxes - prediction.boxes_xyxy)))
                    if expected_boxes.size
                    else 0.0
                )
                mask_iou = (
                    _mask_iou(expected_masks, prediction.masks)
                    if expected_masks.size
                    else 1.0
                )
                indices_match = bool(np.array_equal(expected_indices, actual_indices))
                if not indices_match:
                    raise RuntimeError(
                        f"{plan_id}/{prefix}: admitted query indices do not match"
                    )
                if mask_iou < 0.98:
                    raise RuntimeError(
                        f"{plan_id}/{prefix}: task mask IoU {mask_iou:.6f} < 0.98"
                    )
                cases.append(
                    {
                        "id": prefix,
                        "admitted_indices_at_0_5": actual_indices.tolist(),
                        "admitted_indices_exact": indices_match,
                        "score_max_abs": score_max_abs,
                        "box_max_abs_pixels": box_max_abs,
                        "task_mask_iou": mask_iou,
                        "empty_output": bool(actual_indices.size == 0),
                    }
                )
            counters = dict(session._adapter.counters)
            if counters["image_encodes"] != len(fixtures["images"]):
                raise RuntimeError(
                    f"{plan_id}: image cache reuse gate failed: {counters}"
                )
            if counters["text_encodes"] != len(fixtures["cases"]):
                raise RuntimeError(
                    f"{plan_id}: text cache reuse gate failed: {counters}"
                )
            if plan_id == SELECTED_K32_PLAN_ID and counters["mask_skips"] != 2:
                raise RuntimeError(f"{plan_id}: zero-proposal mask skip gate failed")
            plan_reports[plan_id] = {
                "manifest_sha256": manifests[plan_id].manifest_digest,
                "dispatch_role": manifests[plan_id].dispatch_role,
                "cases": cases,
                "cache_and_transfer_counters": counters,
                "cuda_residency": "all IOBinding outputs asserted CUDA before handoff",
            }
        finally:
            session.close()

    return {
        "format": "sam3-image-pcs-m2-release-validation-v1",
        "status": "pass",
        "profile_id": "b1-1008-l32-q200-fp16",
        "sam3_export_commit": export_report["sam3_export_commit"],
        "official_commit": export_report["official_commit"],
        "checkpoint_sha256": export_report["checkpoint_sha256"],
        "environment": _environment(),
        "gates": {
            "admitted_indices": "exact for every fixture/plan at strict > 0.5",
            "task_mask_iou_minimum": 0.98,
            "selected_k_valid_gather": "covered by M1 report; exact for every valid mask",
            "zero_proposal": "empty Public API output and selected-K mask graph skipped",
            "cache": "same image/text keys do not rerun encoders",
            "residency": "CUDA OrtValue IOBinding asserted for every graph output",
        },
        "plans": plan_reports,
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
        report_path = bundle_dir / "reports" / "m2_release_validation.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        refresh_manifest_file_records(bundle_dir)
        for plan_id in PLAN_IDS:
            validate_manifest_package(
                bundle_dir / "manifests" / f"{plan_id}.json",
                expected_plan_id=plan_id,
            )
    else:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
