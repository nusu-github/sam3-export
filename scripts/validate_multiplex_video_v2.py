"""Validate the shipped M5 SAM3.1 Multiplex bundle and release gates."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

from export_multiplex_video_v2 import refresh_manifest_file_records
import numpy as np
from PIL import Image
import torch

from sam3.runtime import (
    MULTIPLEX_VIDEO_PLAN_ID,
    InteractivePredictOptions,
    InteractivePrompt,
    ObjectStateError,
    PreviewHandle,
    PreviewHandleError,
    SessionClosedError,
    create_multiplex_video_session,
)
from sam3.runtime.interactive_image import (
    _preprocess_interactive_image,
    _prompt_arrays,
    _resize_and_threshold,
)
from sam3.runtime.manifest import sha256_file, validate_manifest_package
from sam3.runtime.multiplex_state import BUCKET_CAPACITY
from sam3.weights import (
    SAM31_CHECKPOINT_SHA256,
    SAM31_REVISION,
    TRACKER_PREFIX,
    TRI_NECK_PREFIX,
    build_sam31_multiplex_tracker_core,
    build_sam31_multiplex_video_modules,
    build_sam31_tri_neck,
    load_sam31_multiplex_checkpoint,
    map_checkpoint_to_module,
    verify_multiplex_checkpoint_shapes,
)


def _environment() -> dict[str, object]:
    import onnxruntime as ort

    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "onnxruntime": ort.__version__,
        "providers": ort.get_available_providers(),
        "gpu": torch.cuda.get_device_name(0),
    }


def _mask_iou(expected: np.ndarray, actual: np.ndarray) -> float:
    left = expected > 0
    right = actual > 0
    intersection = np.logical_and(left, right).sum(axis=(-2, -1))
    union = np.logical_or(left, right).sum(axis=(-2, -1))
    values = np.divide(
        intersection,
        union,
        out=np.ones_like(intersection, dtype=np.float64),
        where=union != 0,
    )
    return float(np.mean(values))


def _mask_ious(expected: np.ndarray, actual: np.ndarray) -> np.ndarray:
    left = expected > 0
    right = actual > 0
    intersection = np.logical_and(left, right).sum(axis=(-2, -1))
    union = np.logical_or(left, right).sum(axis=(-2, -1))
    return np.divide(
        intersection,
        union,
        out=np.ones_like(intersection, dtype=np.float64),
        where=union != 0,
    )


def _logit_metrics(
    expected: np.ndarray,
    actual: np.ndarray,
    *,
    require_mean_abs: bool = True,
) -> dict[str, object]:
    shape_match = expected.shape == actual.shape
    if not shape_match:
        return {
            "shape_match": False,
            "task_mask_iou": 0.0,
            "low_res_logit_mean_abs": float("inf"),
            "low_res_logit_max_abs": float("inf"),
            "pass": False,
        }
    difference = np.abs(expected.astype(np.float32) - actual.astype(np.float32))
    result = {
        "shape_match": True,
        "task_mask_iou": _mask_iou(expected, actual),
        "low_res_logit_mean_abs": float(np.mean(difference)),
        "low_res_logit_max_abs": float(np.max(difference)),
    }
    result["pass"] = bool(
        result["task_mask_iou"] >= 0.98
        and (not require_mean_abs or result["low_res_logit_mean_abs"] <= 0.10)
    )
    return result


def _public_mask_metrics(
    expected_logits: np.ndarray,
    actual_masks: np.ndarray,
    output_size: tuple[int, int],
) -> dict[str, object]:
    expected = _resize_and_threshold(expected_logits, output_size, 0.0)
    shape_match = expected.shape == actual_masks.shape
    iou = _mask_iou(expected, actual_masks) if shape_match else 0.0
    return {
        "shape_match": shape_match,
        "task_mask_iou": iou,
        "pass": bool(shape_match and iou >= 0.98),
    }


def _value_metrics(
    expected: np.ndarray, actual: np.ndarray, maximum: float
) -> dict[str, object]:
    difference = np.abs(expected.astype(np.float32) - actual.astype(np.float32))
    mean_abs = float(np.mean(difference))
    max_abs = float(np.max(difference))
    return {
        "mean_abs": mean_abs,
        "max_abs": max_abs,
        "pass": mean_abs <= maximum,
    }


def _pointer_metrics(expected: np.ndarray, actual: np.ndarray) -> dict[str, object]:
    left = expected.reshape(-1).astype(np.float64)
    right = actual.reshape(-1).astype(np.float64)
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    cosine = 1.0 if denominator == 0 else float(np.dot(left, right) / denominator)
    return {"cosine_similarity": cosine, "pass": cosine >= 0.999}


def _prompt_values(
    points: list[list[float]],
    labels: list[int],
    size: tuple[int, int],
) -> tuple[dict[str, np.ndarray], InteractivePrompt]:
    height, width = size
    prompt = InteractivePrompt(
        points_xy=np.asarray(points, dtype=np.float32)
        * np.asarray([width, height], dtype=np.float32),
        point_labels=np.asarray(labels, dtype=np.int64),
    )
    values, _ = _prompt_arrays(prompt, size)
    return values, prompt


def _torch_prompt(values: dict[str, np.ndarray]) -> tuple[torch.Tensor, ...]:
    return tuple(torch.from_numpy(value).to("cuda") for value in values.values())


def _active_rows(value: torch.Tensor, count: int) -> torch.Tensor:
    rows = [
        value[offset // BUCKET_CAPACITY, offset % BUCKET_CAPACITY]
        for offset in range(count)
    ]
    return torch.stack(rows)


def _pack_torch_state(
    entries: list[dict[str, Any]],
    *,
    frame_index: int,
    bucket_count: int,
) -> tuple[torch.Tensor, ...]:
    device = torch.device("cuda")
    memory = torch.zeros(
        (bucket_count, 10, 256, 72, 72), dtype=torch.float16, device=device
    )
    memory_position = torch.zeros_like(memory)
    memory_image = torch.zeros((10, 256, 72, 72), dtype=torch.float16, device=device)
    memory_image_position = torch.zeros_like(memory_image)
    memory_valid = torch.zeros((bucket_count, 10), dtype=torch.uint8, device=device)
    memory_age = torch.zeros((bucket_count, 10), dtype=torch.int64, device=device)
    pointers = torch.zeros(
        (bucket_count, 16, 16, 256), dtype=torch.float16, device=device
    )
    pointer_valid = torch.zeros(
        (bucket_count, 16, 16), dtype=torch.uint8, device=device
    )
    pointer_age = torch.zeros((bucket_count, 16), dtype=torch.int64, device=device)
    conditioning = [entry for entry in entries if entry["conditioning"]]
    non_conditioning = {
        int(entry["frame_index"]): entry
        for entry in entries
        if not entry["conditioning"]
    }
    spatial: list[tuple[int, dict[str, Any]]] = [
        (column, entry) for column, entry in enumerate(conditioning)
    ]
    for distance in range(1, 7):
        entry = non_conditioning.get(frame_index - distance)
        if entry is not None:
            spatial.append((4 + distance - 1, entry))
    for column, entry in spatial:
        memory[:, column].copy_(entry["memory"])
        memory_position[:, column].copy_(entry["memory_position"])
        memory_image[column].copy_(entry["image"][0])
        memory_image_position[column].copy_(entry["image_position"][0])
        bucket_valid = torch.any(entry["slot_validity"].to(torch.bool), dim=1)
        memory_valid[:, column].copy_(bucket_valid)
        memory_age[:, column] = (
            frame_index - int(entry["frame_index"])
            if entry["conditioning"]
            else 7 - (frame_index - int(entry["frame_index"]))
        )
    pointer_entries = list(conditioning)
    pointer_entries.extend(
        non_conditioning[index]
        for index in range(frame_index - 1, -1, -1)
        if index in non_conditioning
    )
    for column, entry in enumerate(pointer_entries):
        pointers[:, column].copy_(entry["pointers"])
        pointer_valid[:, column].copy_(entry["slot_validity"])
        pointer_age[:, column] = frame_index - int(entry["frame_index"])
    return (
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


def _local_trajectory(
    modules: Any,
    encoded: list[tuple[torch.Tensor, ...]],
    fixtures: dict[str, Any],
    official: Any,
    count: int,
    size: tuple[int, int],
    *,
    apply_correction: bool = True,
    reference_prefix: str | None = None,
) -> tuple[dict[str, object], dict[str, tuple[torch.Tensor, ...]]]:
    bucket_count = 1 if count <= BUCKET_CAPACITY else 2
    if bucket_count != 1:
        raise ValueError("local M5 parity composes independent B1 trajectories")
    slot_validity = torch.zeros(
        (bucket_count, BUCKET_CAPACITY), dtype=torch.uint8, device="cuda"
    )
    low = torch.full(
        (bucket_count, BUCKET_CAPACITY, 1, 288, 288),
        -1024.0,
        dtype=torch.float32,
        device="cuda",
    )
    high = torch.full(
        (bucket_count, BUCKET_CAPACITY, 1, 1008, 1008),
        -1024.0,
        dtype=torch.float32,
        device="cuda",
    )
    pointers = torch.zeros(
        (bucket_count, BUCKET_CAPACITY, 256),
        dtype=torch.float16,
        device="cuda",
    )
    object_scores = torch.full(
        (bucket_count, BUCKET_CAPACITY, 1),
        -1024.0,
        dtype=torch.float32,
        device="cuda",
    )
    selected_scores = torch.zeros(
        (bucket_count, BUCKET_CAPACITY), dtype=torch.float32, device="cuda"
    )
    preview_args: tuple[torch.Tensor, ...] | None = None
    with torch.inference_mode():
        for offset in range(count):
            bucket, slot = divmod(offset, BUCKET_CAPACITY)
            slot_validity[bucket, slot] = 1
            prompt, _ = _prompt_values(
                [fixtures["object_points_xy_relative"][offset % BUCKET_CAPACITY]],
                [1],
                size,
            )
            args = (
                encoded[0][0],
                encoded[0][2],
                encoded[0][3],
                *_torch_prompt(prompt),
            )
            preview_args = args
            preview = modules.preview_single1(*args)
            low[bucket, slot] = preview[0][0]
            high[bucket, slot] = preview[2][0]
            pointers[bucket, slot] = preview[3][0]
            object_scores[bucket, slot] = preview[4][0]
            selected_scores[bucket, slot] = preview[1][0, 0]
        memory_module = (
            modules.memory_commit_bucket1
            if bucket_count == 1
            else modules.memory_commit_bucket2
        )
        propagation_module = (
            modules.propagation_bucket1
            if bucket_count == 1
            else modules.propagation_bucket2
        )
        scatter_module = (
            modules.scatter_commit_bucket1
            if bucket_count == 1
            else modules.scatter_commit_bucket2
        )
        memory0 = memory_module(
            encoded[0][4],
            high,
            object_scores,
            slot_validity,
            slot_validity,
        )
        entry0 = {
            "frame_index": 0,
            "conditioning": True,
            "conditioning_validity": slot_validity,
            "low": low,
            "high": high,
            "pointers": pointers,
            "object_scores": object_scores,
            "selected_scores": selected_scores,
            "memory": memory0[0],
            "memory_position": memory0[1],
            "image": encoded[0][4],
            "image_position": encoded[0][5],
            "slot_validity": slot_validity,
        }
        packed1 = _pack_torch_state([entry0], frame_index=1, bucket_count=bucket_count)
        propagation_args1 = (
            encoded[1][4],
            encoded[1][5],
            encoded[1][6],
            encoded[1][7],
            slot_validity,
            *packed1,
        )
        propagated1 = propagation_module(*propagation_args1)
        best1 = torch.argmax(propagated1[1], dim=2, keepdim=True)
        selected_scores1 = torch.gather(propagated1[1], 2, best1).squeeze(2)

        correction_values, _ = _prompt_values(
            fixtures["correction"]["points_xy_relative"],
            fixtures["correction"]["point_labels"],
            size,
        )
        correction = modules.preview_single1(
            encoded[1][0],
            encoded[1][2],
            encoded[1][3],
            *_torch_prompt(correction_values),
        )
        conditioning_validity = torch.zeros_like(slot_validity)
        conditioning_validity[0, 0] = 1
        scatter_args = (
            encoded[1][4],
            propagated1[2],
            propagated1[3],
            propagated1[4],
            propagated1[5],
            correction[0],
            correction[2],
            correction[3],
            correction[4],
            torch.tensor([[0, 0]], dtype=torch.int64, device="cuda"),
            slot_validity,
            conditioning_validity,
        )
        scattered = scatter_module(*scatter_args)
        if not apply_correction:
            memory1 = memory_module(
                encoded[1][4],
                propagated1[3],
                propagated1[5],
                slot_validity,
                torch.zeros_like(slot_validity),
            )
            scattered = (
                propagated1[2],
                propagated1[3],
                propagated1[4],
                propagated1[5],
                memory1[0],
                memory1[1],
            )
        selected_scores1 = selected_scores1.clone()
        if apply_correction:
            selected_scores1[0, 0] = correction[1][0, 0]
        entry1 = {
            "frame_index": 1,
            "conditioning": apply_correction,
            "conditioning_validity": (
                conditioning_validity
                if apply_correction
                else torch.zeros_like(slot_validity)
            ),
            "low": scattered[0],
            "high": scattered[1],
            "pointers": scattered[2],
            "object_scores": scattered[3],
            "selected_scores": selected_scores1,
            "memory": scattered[4],
            "memory_position": scattered[5],
            "image": encoded[1][4],
            "image_position": encoded[1][5],
            "slot_validity": slot_validity,
        }
        packed2 = _pack_torch_state(
            [entry0, entry1], frame_index=2, bucket_count=bucket_count
        )
        propagation_args2 = (
            encoded[2][4],
            encoded[2][5],
            encoded[2][6],
            encoded[2][7],
            slot_validity,
            *packed2,
        )
        propagated2 = propagation_module(*propagation_args2)
        memory2 = memory_module(
            encoded[2][4],
            propagated2[3],
            propagated2[5],
            slot_validity,
            torch.zeros_like(slot_validity),
        )

    if preview_args is None:
        raise RuntimeError("local trajectory requires active objects")
    local0 = _active_rows(low[:, :, 0], count).float().cpu().numpy()
    local1_pre = _active_rows(propagated1[2][:, :, 0], count).float().cpu().numpy()
    local1 = _active_rows(scattered[0][:, :, 0], count).float().cpu().numpy()
    local2 = _active_rows(propagated2[2][:, :, 0], count).float().cpu().numpy()
    local_scores = _active_rows(propagated2[5], count).float().cpu().numpy()
    local_pointer = _active_rows(propagated2[4], count).float().cpu().numpy()
    non_target = torch.ones_like(slot_validity, dtype=torch.bool)
    non_target[0, 0] = False
    scatter_isolation = all(
        torch.equal(before[non_target], after[non_target])
        for before, after in (
            (propagated1[2], scattered[0]),
            (propagated1[3], scattered[1]),
            (propagated1[4], scattered[2]),
            (propagated1[5], scattered[3]),
        )
    )
    frames = {
        "frame0": _logit_metrics(
            official[f"{reference_prefix or f'count{count}'}_frame0_low"],
            local0,
            require_mean_abs=False,
        ),
        "frame1_pre_correction": _logit_metrics(
            official[
                (
                    f"{reference_prefix or f'count{count}'}_"
                    + (
                        "frame1_pre_correction_low"
                        if apply_correction
                        else "frame1_low"
                    )
                )
            ],
            local1_pre,
            require_mean_abs=False,
        ),
        "frame1_correction": _logit_metrics(
            official[f"{reference_prefix or f'count{count}'}_frame1_low"],
            local1,
            require_mean_abs=False,
        ),
        "frame2": _logit_metrics(
            official[f"{reference_prefix or f'count{count}'}_frame2_low"],
            local2,
            require_mean_abs=False,
        ),
    }
    final_prefix = reference_prefix or f"count{count}"
    final = {
        "object_score": _value_metrics(
            official[f"{final_prefix}_frame2_object_score"][:, None],
            local_scores,
            0.05,
        ),
        "pointer": _pointer_metrics(
            official[f"{final_prefix}_final_pointer"], local_pointer
        ),
        "memory": _value_metrics(
            official[f"{final_prefix}_final_memory"],
            memory2[0].float().cpu().numpy(),
            0.05,
        ),
        "memory_position": _value_metrics(
            official[f"{final_prefix}_final_memory_position"],
            memory2[1].float().cpu().numpy(),
            0.05,
        ),
    }
    passed = (
        all(value["pass"] for value in frames.values())
        and final["memory"]["pass"]
        and final["memory_position"]["pass"]
        and scatter_isolation
    )
    ep_samples = {
        "preview": preview_args,
        "propagation": propagation_args1,
        "memory": (
            encoded[2][4],
            propagated2[3],
            propagated2[5],
            slot_validity,
            torch.zeros_like(slot_validity),
        ),
        "scatter": scatter_args,
    }
    return (
        {
            "status": "pass" if passed else "fail",
            "bucket_count": bucket_count,
            "frames": frames,
            "final_state": final,
            "scatter_non_target_byte_equal": scatter_isolation,
        },
        ep_samples,
    )


def _ep_case(
    module: torch.nn.Module, args: tuple[torch.Tensor, ...]
) -> dict[str, object]:
    cloned = tuple(value.detach().clone() for value in args)
    with torch.no_grad():
        expected = module(*cloned)
        program = torch.export.export(module, cloned, strict=False)
        actual = program.module()(*cloned)
    expected_values = expected if isinstance(expected, tuple) else (expected,)
    actual_values = actual if isinstance(actual, tuple) else (actual,)
    metrics = []
    for left, right in zip(expected_values, actual_values):
        difference = (left.float() - right.float()).abs()
        metrics.append(
            {
                "shape": list(left.shape),
                "max_abs": float(difference.max().cpu()),
                "mean_abs": float(difference.mean().cpu()),
                "within_fp16_component_tolerance": bool(
                    difference.max() <= 0.125 and difference.mean() <= 0.02
                ),
            }
        )
    passed = len(expected_values) == len(actual_values) and all(
        value["within_fp16_component_tolerance"] for value in metrics
    )
    return {"status": "pass" if passed else "fail", "outputs": metrics}


def _ep_report(
    modules: Any,
    pixel_values: torch.Tensor,
    samples: dict[int, dict[str, tuple[torch.Tensor, ...]]],
) -> dict[str, object]:
    report = {
        "multiplex-frame-encode": _ep_case(modules.frame_encode, (pixel_values,)),
        "multiplex-interaction-preview-single1": _ep_case(
            modules.preview_single1, samples[1]["preview"]
        ),
    }
    multi_args = samples[1]["preview"]
    report["multiplex-interaction-preview-multimask3"] = _ep_case(
        modules.preview_multimask3, multi_args
    )
    modules_by_operation = {
        "propagation": modules.propagation_bucket1,
        "memory-commit": modules.memory_commit_bucket1,
        "scatter-replace-commit": modules.scatter_commit_bucket1,
    }
    sample = samples[1]
    for operation, module in modules_by_operation.items():
        sample_name = {
            "propagation": "propagation",
            "memory-commit": "memory",
            "scatter-replace-commit": "scatter",
        }[operation]
        report[f"multiplex-{operation}-bucket1"] = _ep_case(module, sample[sample_name])
    passed = all(value["status"] == "pass" for value in report.values())
    return {"status": "pass" if passed else "fail", "artifacts": report}


def _expect_error(call: Any, error: type[BaseException]) -> bool:
    try:
        call()
    except error:
        return True
    return False


def _public_trajectory(
    session: Any,
    frames: list[Image.Image],
    fixtures: dict[str, Any],
    official: Any,
    count: int,
) -> dict[str, object]:
    counters_before = dict(session._adapter.counters)
    session.set_video(frames)
    size = (frames[0].height, frames[0].width)
    initial_logits: list[np.ndarray] = []
    used_handle_rejected = False
    duplicate_rejected = False
    for offset in range(count):
        object_id = 100 + offset
        session.add_object(object_id)
        if offset == 0:
            duplicate_rejected = _expect_error(
                lambda: session.add_object(object_id), ObjectStateError
            )
        _, prompt = _prompt_values(
            [fixtures["object_points_xy_relative"][offset % BUCKET_CAPACITY]],
            [1],
            size,
        )
        preview = session.preview(
            object_id,
            0,
            prompt,
            InteractivePredictOptions(multimask_output=False),
        )
        if preview.preview_handle is None:
            raise RuntimeError("single preview omitted its commit handle")
        committed = session.commit(preview.preview_handle)
        if offset == 0:
            used_handle_rejected = _expect_error(
                lambda: session.commit(preview.preview_handle),
                PreviewHandleError,
            )
        initial_logits.append(committed.low_res_logits)

    lifecycle = {
        "duplicate_object_rejected": duplicate_rejected,
        "used_preview_rejected": used_handle_rejected,
        "unknown_remove_rejected": _expect_error(
            lambda: session.remove_object(-999), ObjectStateError
        ),
        "foreign_preview_rejected": _expect_error(
            lambda: session.commit(PreviewHandle("foreign", "foreign")),
            PreviewHandleError,
        ),
        "invalid_frame_rejected": _expect_error(
            lambda: session.preview(100, len(frames)), IndexError
        ),
        "invalid_prompt_range_rejected": _expect_error(
            lambda: session.preview(
                100,
                0,
                InteractivePrompt(
                    points_xy=np.asarray([[-1.0, 0.0]], dtype=np.float32),
                    point_labels=np.asarray([1], dtype=np.int64),
                ),
            ),
            ValueError,
        ),
    }
    replacement_map = list(range(count))
    stale_after_assignment = True
    replacement_object_id: int | None = None
    if count > BUCKET_CAPACITY:
        _, stale_prompt = _prompt_values(
            [fixtures["object_points_xy_relative"][15]], [1], size
        )
        stale = session.preview(
            115,
            0,
            stale_prompt,
            InteractivePredictOptions(multimask_output=False),
        )
        session.remove_object(115)
        replacement_object_id = 1000 + count
        session.add_object(replacement_object_id)
        stale_after_assignment = bool(
            stale.preview_handle is not None
            and _expect_error(
                lambda: session.commit(stale.preview_handle),
                PreviewHandleError,
            )
        )
        replacement = session.preview(
            replacement_object_id,
            0,
            stale_prompt,
            InteractivePredictOptions(multimask_output=False),
        )
        if replacement.preview_handle is None:
            raise RuntimeError("replacement preview omitted its commit handle")
        session.commit(replacement.preview_handle)
        replacement_map = [value for value in replacement_map if value != 15]
        replacement_map.append(15)
    lifecycle["stale_after_assignment_rejected"] = stale_after_assignment
    if count == 32:
        lifecycle["capacity_33_rejected"] = _expect_error(
            lambda: session.add_object(99999), ObjectStateError
        )

    revision_before = session._state.revision
    propagated = session.propagate(start_frame=0, end_frame=1)
    propagation_kept_assignment_revision = session._state.revision == revision_before
    frame0 = propagated[0]
    frame1_pre = propagated[1]
    expected_ids = [100 + value for value in replacement_map]
    if replacement_object_id is not None:
        expected_ids[-1] = replacement_object_id
    order_match = frame1_pre.object_ids.tolist() == expected_ids

    non_target_before: tuple[torch.Tensor, ...] | None = None
    non_target_bucket_memory_before: tuple[torch.Tensor, ...] | None = None
    if count > 1:
        entry = session._adapter._non_conditioning[1]
        non_target_before = tuple(
            session._adapter._as_torch(value).clone()
            for value in (
                entry.low_res,
                entry.high_res,
                entry.pointers,
                entry.object_scores,
            )
        )
        if count > BUCKET_CAPACITY:
            non_target_bucket_memory_before = tuple(
                session._adapter._as_torch(value)[1:].clone()
                for value in (entry.memory, entry.memory_position)
            )
    _, correction_prompt = _prompt_values(
        fixtures["correction"]["points_xy_relative"],
        fixtures["correction"]["point_labels"],
        size,
    )
    correction = session.preview(
        100,
        1,
        correction_prompt,
        InteractivePredictOptions(multimask_output=False),
    )
    if correction.preview_handle is None:
        raise RuntimeError("correction preview omitted its commit handle")
    corrected = session.commit(correction.preview_handle)
    scatter_isolation = True
    if non_target_before is not None:
        entry = session._adapter._conditioning[1]
        non_target = torch.ones(
            (entry.slot_validity.shape[0], BUCKET_CAPACITY),
            dtype=torch.bool,
            device="cuda",
        )
        non_target[0, 0] = False
        after_values = tuple(
            session._adapter._as_torch(value)
            for value in (
                entry.low_res,
                entry.high_res,
                entry.pointers,
                entry.object_scores,
            )
        )
        scatter_isolation = all(
            torch.equal(before[non_target], after[non_target])
            for before, after in zip(non_target_before, after_values)
        )
        if non_target_bucket_memory_before is not None:
            scatter_isolation = scatter_isolation and all(
                torch.equal(before, session._adapter._as_torch(after)[1:])
                for before, after in zip(
                    non_target_bucket_memory_before,
                    (entry.memory, entry.memory_position),
                )
            )
        del non_target_before, after_values
        torch.cuda.empty_cache()
    frame2 = session.propagate(start_frame=2, end_frame=2)[0]
    counters = {
        name: value - counters_before[name]
        for name, value in session._adapter.counters.items()
    }
    public_order = np.asarray(replacement_map, dtype=np.int64)
    initial = _logit_metrics(
        official[f"count{count}_frame0_low"],
        np.stack(initial_logits),
    )
    frame0_metrics = _public_mask_metrics(
        official[f"count{count}_frame0_low"][public_order],
        frame0.masks,
        size,
    )
    frame1_metrics = _public_mask_metrics(
        official[f"count{count}_frame1_pre_correction_low"][public_order],
        frame1_pre.masks,
        size,
    )
    correction_metrics = _logit_metrics(
        official[f"count{count}_frame1_low"][0],
        corrected.low_res_logits,
    )
    frame2_metrics = _public_mask_metrics(
        official[f"count{count}_frame2_low"][public_order],
        frame2.masks,
        size,
    )
    frame2_expected = _resize_and_threshold(
        official[f"count{count}_frame2_low"][public_order], size, 0.0
    )
    frame2_ious = _mask_ious(frame2_expected, frame2.masks)
    frame2_entry = session._adapter._non_conditioning[2]
    frame2_locations = torch.as_tensor(
        session._state.assignment_array(), dtype=torch.int64, device="cuda"
    )
    frame2_object_scores = session._adapter._to_public_numpy(
        session._adapter._from_torch(
            session._adapter._as_torch(frame2_entry.object_scores)[
                frame2_locations[:, 0], frame2_locations[:, 1], 0
            ].contiguous()
        )
    )
    expected_object_scores = official[f"count{count}_frame2_object_score"][public_order]
    frame2_metrics["minimum_task_mask_iou"] = float(frame2_ious.min())
    frame2_metrics["below_gate_objects"] = [
        {
            "object_id": int(object_id),
            "iou": float(iou),
            "expected_object_score": float(expected_score),
            "actual_object_score": float(actual_score),
        }
        for object_id, iou, expected_score, actual_score in zip(
            frame2.object_ids,
            frame2_ious,
            expected_object_scores,
            frame2_object_scores,
        )
        if iou < 0.98
    ]
    boundary = {}
    if count > BUCKET_CAPACITY:
        object_17_expected = _resize_and_threshold(
            official[f"count{count}_frame2_low"][16:17], size, 0.0
        )
        object_17_actual = frame2.masks[
            replacement_map.index(16) : replacement_map.index(16) + 1
        ]
        boundary = {
            "object_16_iou": _public_mask_metrics(
                official[f"count{count}_frame2_low"][15:16],
                frame2.masks[replacement_map.index(15) : replacement_map.index(15) + 1],
                size,
            )["task_mask_iou"],
            "object_17_iou": _public_mask_metrics(
                official[f"count{count}_frame2_low"][16:17],
                object_17_actual,
                size,
            )["task_mask_iou"],
            "object_17_expected_foreground_pixels": int(object_17_expected.sum()),
            "object_17_actual_foreground_pixels": int(object_17_actual.sum()),
        }
        boundary["pass"] = bool(
            boundary["object_16_iou"] >= 0.98 and boundary["object_17_iou"] >= 0.98
        )
    lifecycle_pass = all(lifecycle.values())
    residency = {
        "cuda_iobinding": counters["session_launches"] > 0,
        "state_d2h_bytes": counters["state_d2h_bytes"],
        "state_h2d_bytes": counters["state_h2d_bytes"],
        "state_demuxes": counters["state_demuxes"],
        "state_remuxes": counters["state_remuxes"],
        "final_d2h_bytes": counters["final_d2h_bytes"],
        "assignment_revision_unchanged_during_propagation": (
            propagation_kept_assignment_revision
        ),
    }
    residency["pass"] = bool(
        residency["cuda_iobinding"]
        and residency["state_d2h_bytes"] == 0
        and residency["state_h2d_bytes"] == 0
        and residency["state_demuxes"] == 0
        and residency["state_remuxes"] == 0
        and residency["final_d2h_bytes"] > 0
        and propagation_kept_assignment_revision
    )
    passed = all(
        (
            initial["pass"],
            frame0_metrics["pass"],
            frame1_metrics["pass"],
            correction_metrics["pass"],
            frame2_metrics["pass"],
            scatter_isolation,
            order_match,
            lifecycle_pass,
            residency["pass"],
            boundary.get("pass", True),
        )
    )
    return {
        "status": "pass" if passed else "fail",
        "objects": count,
        "initial_single_commit": initial,
        "frame0": frame0_metrics,
        "frame1_pre_correction": frame1_metrics,
        "frame1_correction_commit": correction_metrics,
        "frame2": frame2_metrics,
        "bucket_boundary": boundary,
        "final_object_id_order": frame2.object_ids.tolist(),
        "expected_object_id_order": expected_ids,
        "object_id_order_match": order_match,
        "scatter_non_target_byte_equal": scatter_isolation,
        "lifecycle": lifecycle,
        "residency": residency,
        "counters": counters,
    }


def _public_group(
    bundle_dir: Path,
    frames: list[Image.Image],
    fixtures: dict[str, Any],
    official: Any,
    counts: list[int],
) -> dict[str, object]:
    session = create_multiplex_video_session(
        MULTIPLEX_VIDEO_PLAN_ID, bundle_dir=bundle_dir
    )
    cases = {
        str(count): _public_trajectory(
            session,
            frames,
            fixtures,
            official,
            count,
        )
        for count in counts
    }
    session.close()
    close_lifecycle = {
        "double_close_rejected": _expect_error(session.close, SessionClosedError),
        "use_after_close_rejected": _expect_error(
            lambda: session.add_object(12345), SessionClosedError
        ),
    }
    passed = all(value["status"] == "pass" for value in cases.values()) and all(
        close_lifecycle.values()
    )
    return {
        "status": "pass" if passed else "fail",
        "cases": cases,
        "close_lifecycle": close_lifecycle,
    }


def _public_subprocess(bundle_dir: Path, counts: list[int]) -> dict[str, object]:
    environment = dict(os.environ)
    source_path = str((Path(__file__).resolve().parent.parent / "src").resolve())
    scripts_path = str(Path(__file__).resolve().parent)
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{source_path}:{scripts_path}:{existing}"
        if existing
        else f"{source_path}:{scripts_path}"
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--bundle-dir",
            str(bundle_dir),
            "--public-only",
            ",".join(str(value) for value in counts),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "isolated Public API validation failed:\n"
            + completed.stderr[-4000:]
            + completed.stdout[-4000:]
        )
    return json.loads(completed.stdout)


def validate_bundle(bundle_dir: Path, checkpoint: Path) -> dict[str, object]:
    resolved = validate_manifest_package(
        bundle_dir / "manifests" / f"{MULTIPLEX_VIDEO_PLAN_ID}.json",
        expected_plan_id=MULTIPLEX_VIDEO_PLAN_ID,
    )
    fixtures = json.loads(
        (bundle_dir / "fixtures/cases.json").read_text(encoding="utf-8")
    )
    official = np.load(bundle_dir / "fixtures/official_reference.npz")
    frames = [
        Image.open(
            bundle_dir / "fixtures" / "frames" / Path(item["path"]).name
        ).convert("RGB")
        for item in fixtures["frames"]
    ]
    checkpoint_state = load_sam31_multiplex_checkpoint(checkpoint)
    verify_multiplex_checkpoint_shapes(checkpoint_state)
    mapping = {
        "tri_neck": map_checkpoint_to_module(
            checkpoint_state,
            build_sam31_tri_neck(),
            prefix=TRI_NECK_PREFIX,
            load=True,
        ),
        "multiplex_tracker": map_checkpoint_to_module(
            checkpoint_state,
            build_sam31_multiplex_tracker_core(),
            prefix=TRACKER_PREFIX,
            load=True,
        ),
    }
    mapping_gate = {
        name: {
            "checkpoint_prefix": value.checkpoint_prefix,
            "checkpoint_key_count": value.checkpoint_key_count,
            "module_key_count": value.module_key_count,
            "missing_keys": list(value.missing_keys),
            "unexpected_keys": list(value.unexpected_keys),
            "shape_mismatches": list(value.shape_mismatches),
            "value_mismatches": list(value.value_mismatches),
            "exact": value.exact,
        }
        for name, value in mapping.items()
    }
    del checkpoint_state, mapping
    gc.collect()
    modules = build_sam31_multiplex_video_modules(checkpoint)
    encoded: list[tuple[torch.Tensor, ...]] = []
    pixels: list[torch.Tensor] = []
    for frame in frames:
        values, _ = _preprocess_interactive_image(frame)
        pixel = torch.from_numpy(values).to("cuda")
        pixels.append(pixel)
        with torch.inference_mode():
            encoded.append(modules.frame_encode(pixel))
    size = (frames[0].height, frames[0].width)
    local: dict[str, object] = {}
    ep_samples: dict[int, dict[str, tuple[torch.Tensor, ...]]] = {}
    for count in fixtures["active_slot_cases"]:
        trajectory, samples = _local_trajectory(
            modules,
            encoded,
            fixtures,
            official,
            int(count),
            size,
        )
        local[str(count)] = trajectory
        ep_samples[1] = samples
    for count in fixtures["two_bucket_cases"]:
        second_count = int(count) - BUCKET_CAPACITY
        second, unused_samples = _local_trajectory(
            modules,
            encoded,
            fixtures,
            official,
            second_count,
            size,
            apply_correction=False,
            reference_prefix=f"count{second_count}_no_correction",
        )
        del unused_samples
        first = local[str(BUCKET_CAPACITY)]
        composed_pass = first["status"] == "pass" and second["status"] == "pass"
        local[str(count)] = {
            "status": "pass" if composed_pass else "fail",
            "composition": "independent fixed B1 bucket trajectories",
            "bucket0_selected_correction": first,
            "bucket1_no_correction": second,
        }
    ep = _ep_report(modules, pixels[0], ep_samples)
    del modules, encoded, pixels, ep_samples
    gc.collect()
    torch.cuda.empty_cache()

    public_groups = [
        _public_subprocess(
            bundle_dir, [int(value) for value in fixtures["active_slot_cases"]]
        ),
        _public_subprocess(
            bundle_dir, [int(value) for value in fixtures["two_bucket_cases"]]
        ),
    ]
    public = {
        count: value
        for group in public_groups
        for count, value in group["cases"].items()
    }
    close_lifecycle = [group["close_lifecycle"] for group in public_groups]
    profile = json.loads(
        (bundle_dir / "reports/profile_decision.json").read_text(encoding="utf-8")
    )
    dynamic_gates_pass = bool(
        profile["residency_gate"]["pass"]
        and profile["artifact_size_bytes"]["pass"]
        and all(row["pass"] for row in profile["gate"].values())
    )
    profile_gate = {
        "Decision": profile["Decision"],
        "Applicable profiles": profile["Applicable profiles"],
        "bounded_dynamic_residency": profile["residency_gate"],
        "fixed_measurements": profile["measurements"]["fixed"],
        "artifact_size_bytes": profile["artifact_size_bytes"],
        "fixed_recipe": profile["fixed_recipe"],
        "pass": profile["Decision"] == "fixed-one-two" and not dynamic_gates_pass,
    }
    signatures = json.loads(
        (bundle_dir / "capture/graph_signatures.json").read_text(encoding="utf-8")
    )
    graph_signature_gate = {
        "artifact_count": len(signatures["graphs"]),
        "manifest_artifact_count": len(resolved.manifest["artifacts"]),
        "pass": len(signatures["graphs"]) == len(resolved.manifest["artifacts"]),
    }
    graph_hashes = {
        record["path"]: record["digest"]["value"]
        for record in resolved.manifest["files"]
        if record["role"] in {"graph", "external-data"}
    }
    local_pass = all(value["status"] == "pass" for value in local.values())
    public_pass = all(group["status"] == "pass" for group in public_groups)
    mapping_pass = all(value["exact"] for value in mapping_gate.values())
    passed = all(
        (
            mapping_pass,
            local_pass,
            ep["status"] == "pass",
            public_pass,
            profile_gate["pass"],
            graph_signature_gate["pass"],
        )
    )
    return {
        "format": "sam3-sam31-multiplex-m5-release-validation-v1",
        "status": "pass" if passed else "fail",
        "plan_id": resolved.plan_id,
        "profile_id": resolved.profile_id,
        "scope_label": resolved.manifest["scope"]["scope_label"],
        "environment": _environment(),
        "checkpoint": {
            "revision": SAM31_REVISION,
            "sha256": SAM31_CHECKPOINT_SHA256,
            "actual_sha256": sha256_file(checkpoint),
        },
        "mapping": mapping_gate,
        "artifact_hashes": graph_hashes,
        "graph_signatures": graph_signature_gate,
        "stages": {
            "official_eager_to_local_eager": local,
            "local_eager_to_exported_program": ep,
            "exported_program_to_ort_cuda_to_public_api": {
                "cases": public,
                "close_lifecycle": close_lifecycle,
                "status": "pass" if public_pass else "fail",
            },
        },
        "profile_decision": profile_gate,
        "residency_and_copies": {
            "handoff": "CUDA OrtValue / DLPack D2D bucket-state packing",
            "public_d2h": "public masks, scores, and metadata only",
            "fallback": None,
            "cases": {count: value["residency"] for count, value in public.items()},
        },
        "fixture_hash": sha256_file(bundle_dir / "fixtures/cases.json"),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--update-report", action="store_true")
    parser.add_argument("--public-only")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    bundle_dir = args.bundle_dir.resolve()
    if args.public_only is not None:
        validate_manifest_package(
            bundle_dir / "manifests" / f"{MULTIPLEX_VIDEO_PLAN_ID}.json",
            expected_plan_id=MULTIPLEX_VIDEO_PLAN_ID,
        )
        fixtures = json.loads(
            (bundle_dir / "fixtures/cases.json").read_text(encoding="utf-8")
        )
        official = np.load(bundle_dir / "fixtures/official_reference.npz")
        frames = [
            Image.open(
                bundle_dir / "fixtures" / "frames" / Path(item["path"]).name
            ).convert("RGB")
            for item in fixtures["frames"]
        ]
        counts = [int(value) for value in args.public_only.split(",") if value]
        report = _public_group(bundle_dir, frames, fixtures, official, counts)
        if report["status"] != "pass":
            raise RuntimeError(json.dumps(report, indent=2))
        print(json.dumps(report))
        return 0
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required for full release validation")
    report = validate_bundle(bundle_dir, args.checkpoint.resolve())
    if report["status"] != "pass":
        raise RuntimeError(json.dumps(report, indent=2))
    if args.update_report:
        fixture_report = {
            "format": "m5-sam31-multiplex-fixture-report-v1",
            "status": report["status"],
            "fixture_hash": report["fixture_hash"],
            "stages": report["stages"],
        }
        (bundle_dir / "reports/fixture_report.json").write_text(
            json.dumps(fixture_report, indent=2) + "\n",
            encoding="utf-8",
        )
        (bundle_dir / "reports/m5_release_validation.json").write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )
        refresh_manifest_file_records(bundle_dir)
        validate_manifest_package(
            bundle_dir / "manifests" / f"{MULTIPLEX_VIDEO_PLAN_ID}.json",
            expected_plan_id=MULTIPLEX_VIDEO_PLAN_ID,
        )
    else:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
