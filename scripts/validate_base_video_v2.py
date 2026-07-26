"""Validate the shipped M4 base-video bundle on ORT CUDA and Public API."""

from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path
import platform
from typing import Any

from export_base_video_v2 import (
    _prompt,
    refresh_manifest_file_records,
)
import numpy as np
from PIL import Image
import torch

from sam3.runtime import (
    BASE_VIDEO_PLAN_ID,
    InteractivePredictOptions,
    InteractivePrompt,
    PreviewHandleError,
    create_video_session,
)
from sam3.runtime.interactive_image import (
    _preprocess_interactive_image,
    _prompt_arrays,
)
from sam3.runtime.manifest import sha256_file, validate_manifest_package
from sam3.weights import build_base_video_modules


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
    return float(np.mean(np.where(union == 0, 1.0, intersection / union)))


def _prediction_metrics(
    expected_logits: np.ndarray,
    expected_scores: np.ndarray,
    actual_logits: np.ndarray,
    actual_scores: np.ndarray,
) -> dict[str, object]:
    result = {
        "shape_match": expected_logits.shape == actual_logits.shape,
        "top_score_index_match": int(np.argmax(expected_scores))
        == int(np.argmax(actual_scores)),
        "task_mask_iou": _mask_iou(expected_logits, actual_logits),
        "score_max_abs": float(np.max(np.abs(expected_scores - actual_scores))),
        "low_res_logit_mean_abs": float(
            np.mean(np.abs(expected_logits - actual_logits))
        ),
    }
    result["pass"] = bool(
        result["shape_match"]
        and result["top_score_index_match"]
        and result["task_mask_iou"] >= 0.98
        and result["score_max_abs"] <= 0.02
        and result["low_res_logit_mean_abs"] <= 0.05
    )
    return result


def _memory_metrics(expected: np.ndarray, actual: np.ndarray) -> dict[str, object]:
    mean_abs = float(np.mean(np.abs(expected - actual)))
    return {"mean_abs": mean_abs, "pass": mean_abs <= 0.05}


def _pointer_metrics(expected: np.ndarray, actual: np.ndarray) -> dict[str, object]:
    left = expected.reshape(-1).astype(np.float64)
    right = actual.reshape(-1).astype(np.float64)
    cosine = float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))
    return {"cosine_similarity": cosine, "pass": cosine >= 0.999}


def _empty_state(batch: int = 1) -> tuple[torch.Tensor, ...]:
    return (
        torch.ones(batch, dtype=torch.bool, device="cuda"),
        torch.zeros(batch, 10, 64, 72, 72, dtype=torch.float16, device="cuda"),
        torch.zeros(batch, 10, 64, 72, 72, dtype=torch.float16, device="cuda"),
        torch.zeros(batch, 10, dtype=torch.bool, device="cuda"),
        torch.zeros(batch, 10, dtype=torch.int64, device="cuda"),
        torch.zeros(batch, 10, dtype=torch.bool, device="cuda"),
        torch.zeros(batch, 16, 256, dtype=torch.float16, device="cuda"),
        torch.zeros(batch, 16, dtype=torch.bool, device="cuda"),
        torch.zeros(batch, 16, dtype=torch.int64, device="cuda"),
        torch.zeros(batch, 16, dtype=torch.bool, device="cuda"),
        torch.full((batch,), 2.0, dtype=torch.float32, device="cuda"),
    )


def _torch_prompt(
    values: dict[str, np.ndarray], batch: int = 1
) -> tuple[torch.Tensor, ...]:
    result = []
    for value in values.values():
        repeats = (batch,) + (1,) * (value.ndim - 1)
        result.append(torch.from_numpy(np.tile(value, repeats)).to("cuda"))
    return tuple(result)


def _local_reference(
    modules: Any,
    frames: list[Image.Image],
    fixtures: dict[str, Any],
) -> dict[str, np.ndarray]:
    encoded: list[tuple[torch.Tensor, ...]] = []
    for frame in frames:
        values, _ = _preprocess_interactive_image(frame)
        with torch.inference_mode():
            encoded.append(modules.frame_encode(torch.from_numpy(values).to("cuda")))
    size = (frames[0].height, frames[0].width)
    initial_prompt = _prompt(fixtures, "initial_prompt", size)
    correction_prompt = _prompt(fixtures, "correction_prompt", size)
    arrays: dict[str, np.ndarray] = {
        "frame0_image_embedding": encoded[0][0].float().cpu().numpy(),
        "frame0_image_position": encoded[0][1].float().cpu().numpy(),
        "frame0_high_res_0": encoded[0][2].float().cpu().numpy(),
        "frame0_high_res_1": encoded[0][3].float().cpu().numpy(),
    }
    with torch.inference_mode():
        multi = modules.preview_multimask3(
            *encoded[0], *_empty_state(), *_torch_prompt(initial_prompt)
        )
        best = int(torch.argmax(multi[1], dim=-1)[0])
        correction_prompt["mask-input"][0, 0] = multi[0][0, best].float().cpu().numpy()
        correction_prompt["has-mask"][0] = True
        single = modules.preview_single1(
            *encoded[0], *_empty_state(), *_torch_prompt(correction_prompt)
        )
        committed = modules.memory_commit(
            encoded[0][0],
            single[2],
            single[4],
            torch.ones(1, dtype=torch.bool, device="cuda"),
        )
    arrays.update(
        {
            "memory0_multimask_logits": multi[0].float().cpu().numpy(),
            "memory0_multimask_scores": multi[1].float().cpu().numpy(),
            "memory0_single_logits": single[0].float().cpu().numpy(),
            "memory0_single_scores": single[1].float().cpu().numpy(),
            "memory0_pointer": single[3].float().cpu().numpy(),
            "memory0_object_score": single[4].float().cpu().numpy(),
            "memory0_features": committed[0].float().cpu().numpy(),
            "memory0_position": committed[1].float().cpu().numpy(),
        }
    )

    memory_features = torch.zeros(1, 10, 64, 72, 72, dtype=torch.float16, device="cuda")
    memory_position = torch.zeros_like(memory_features)
    memory_features[:, 0] = committed[0]
    memory_position[:, 0] = committed[1]
    memory_valid = torch.zeros(1, 10, dtype=torch.bool, device="cuda")
    memory_valid[:, 0] = True
    memory_age = torch.zeros(1, 10, dtype=torch.int64, device="cuda")
    memory_age[:, 0] = 1
    memory_cond = torch.zeros(1, 10, dtype=torch.bool, device="cuda")
    memory_cond[:, 0] = True
    pointers = torch.zeros(1, 16, 256, dtype=torch.float16, device="cuda")
    pointers[:, 0] = single[3]
    pointer_valid = torch.zeros(1, 16, dtype=torch.bool, device="cuda")
    pointer_valid[:, 0] = True
    pointer_age = torch.zeros(1, 16, dtype=torch.int64, device="cuda")
    pointer_age[:, 0] = 1
    empty_prompt, _ = _prompt_arrays(InteractivePrompt(), size)
    state_one = (
        torch.ones(1, dtype=torch.bool, device="cuda"),
        memory_features,
        memory_position,
        memory_valid,
        memory_age,
        memory_cond,
        pointers,
        pointer_valid,
        pointer_age,
        memory_cond.new_zeros((1, 16)),
        torch.full((1,), 2.0, dtype=torch.float32, device="cuda"),
    )
    state_one[-2][:, 0] = True
    with torch.inference_mode():
        one = modules.preview_single1(
            *encoded[1], *state_one, *_torch_prompt(empty_prompt)
        )
        memory_one = modules.memory_commit(
            encoded[1][0],
            one[2],
            one[4],
            torch.zeros(1, dtype=torch.bool, device="cuda"),
        )
    arrays.update(
        {
            "memory1_single_logits": one[0].float().cpu().numpy(),
            "memory1_single_scores": one[1].float().cpu().numpy(),
            "memory1_pointer": one[3].float().cpu().numpy(),
            "memory1_features": memory_one[0].float().cpu().numpy(),
            "memory1_position": memory_one[1].float().cpu().numpy(),
        }
    )

    max_memory = committed[0][:, None].expand(-1, 10, -1, -1, -1).contiguous()
    max_position = committed[1][:, None].expand_as(max_memory).contiguous()
    max_age = torch.tensor([[20, 15, 10, 5, 1, 2, 3, 4, 5, 6]], device="cuda")
    max_cond = torch.tensor(
        [[True, True, True, True, False, False, False, False, False, False]],
        device="cuda",
    )
    max_pointers = single[3][:, None].expand(-1, 16, -1).contiguous()
    pointer_age_max = torch.tensor(
        [[20, 15, 10, 5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]],
        device="cuda",
    )
    pointer_valid_max = pointer_age_max != 0
    pointer_cond_max = torch.zeros_like(pointer_valid_max)
    pointer_cond_max[:, :4] = True
    with torch.inference_mode():
        maximum = modules.preview_single1(
            *encoded[2],
            torch.ones(1, dtype=torch.bool, device="cuda"),
            max_memory,
            max_position,
            torch.ones(1, 10, dtype=torch.bool, device="cuda"),
            max_age,
            max_cond,
            max_pointers,
            pointer_valid_max,
            pointer_age_max,
            pointer_cond_max,
            torch.full((1,), 15.0, dtype=torch.float32, device="cuda"),
            *_torch_prompt(empty_prompt),
        )
        absent = modules.memory_commit(
            encoded[0][0],
            single[2],
            -torch.ones_like(single[4]),
            torch.ones(1, dtype=torch.bool, device="cuda"),
        )
    arrays.update(
        {
            "memory_max_single_logits": maximum[0].float().cpu().numpy(),
            "memory_max_single_scores": maximum[1].float().cpu().numpy(),
            "memory_max_pointer": maximum[3].float().cpu().numpy(),
            "absent_memory_features": absent[0].float().cpu().numpy(),
            "absent_memory_position": absent[1].float().cpu().numpy(),
        }
    )
    return arrays


def _official_local_report(
    official: Any, local: dict[str, np.ndarray]
) -> dict[str, object]:
    prediction_cases = {
        "memory0_multimask": ("memory0_multimask_logits", "memory0_multimask_scores"),
        "memory0_single": ("memory0_single_logits", "memory0_single_scores"),
        "memory1_single": ("memory1_single_logits", "memory1_single_scores"),
        "memory_max_single": ("memory_max_single_logits", "memory_max_single_scores"),
    }
    predictions = {
        name: _prediction_metrics(
            official[logits], official[scores], local[logits], local[scores]
        )
        for name, (logits, scores) in prediction_cases.items()
    }
    memories = {
        name: _memory_metrics(official[name], local[name])
        for name in (
            "memory0_features",
            "memory0_position",
            "memory1_features",
            "memory1_position",
            "absent_memory_features",
            "absent_memory_position",
        )
    }
    pointers = {
        name: _pointer_metrics(official[name], local[name])
        for name in ("memory0_pointer", "memory1_pointer", "memory_max_pointer")
    }
    features = {
        name: _memory_metrics(official[name], local[name])
        for name in (
            "frame0_image_embedding",
            "frame0_image_position",
            "frame0_high_res_0",
            "frame0_high_res_1",
        )
    }
    passed = all(
        value["pass"]
        for group in (predictions, memories, pointers, features)
        for value in group.values()
    )
    return {
        "status": "pass" if passed else "fail",
        "predictions": predictions,
        "memories": memories,
        "pointers": pointers,
        "frame_features": features,
    }


def _public_trajectory(
    bundle_dir: Path,
    frames: list[Image.Image],
    fixtures: dict[str, Any],
    official: Any,
) -> dict[str, object]:
    session = create_video_session(BASE_VIDEO_PLAN_ID, bundle_dir=bundle_dir)
    session.set_video(frames)
    session.add_object(1)
    height, width = frames[0].height, frames[0].width

    def public_prompt(name: str, mask: np.ndarray | None = None) -> InteractivePrompt:
        record = fixtures[name]
        relative = np.asarray(record["points_xy"], dtype=np.float32)
        return InteractivePrompt(
            points_xy=relative * np.asarray([width, height], dtype=np.float32),
            point_labels=np.asarray(record["point_labels"], dtype=np.int64),
            mask_logits=mask,
        )

    multi = session.preview(1, 0, public_prompt("initial_prompt"))
    selected = multi.low_res_logits[int(np.argmax(multi.scores))]
    single = session.preview(
        1,
        0,
        public_prompt("correction_prompt", selected),
        InteractivePredictOptions(multimask_output=False),
    )
    if single.preview_handle is None:
        raise RuntimeError("single preview did not return a commit handle")
    record = session._previews[single.preview_handle._token]
    committed = session.commit(single.preview_handle)
    next_frame = session.propagate(start_frame=1, end_frame=1)[0]
    next_entry = session._state.require_object(1).non_conditioning[1]
    commits_before = session._adapter.counters["memory_commits"]
    revision_before = session._state.revision
    for _ in range(3):
        session.preview(1, 2, public_prompt("correction_prompt", selected))
    preview_non_mutating = (
        session._adapter.counters["memory_commits"] == commits_before
        and session._state.revision == revision_before
    )

    negative_scores = session._adapter._upload(
        -np.ones((session._adapter.batch_capacity, 1), dtype=np.float32)
    )
    absent_preview = replace(record.backend_preview, object_scores=negative_scores)
    absent_commit = session._adapter.commit(
        record.frame_cache,
        record.packed,
        absent_preview,
        is_mask_from_points=True,
    )[0]
    if absent_commit is None:
        raise RuntimeError("absent-object commit was omitted")
    absent_memory = (
        torch.from_dlpack(absent_commit.memory_features).float().cpu().numpy()
    )

    source_entry = session._state.require_object(1).conditioning[0]
    stale = session.preview(
        1,
        0,
        public_prompt("correction_prompt", selected),
        InteractivePredictOptions(multimask_output=False),
    )
    replacement = session.preview(
        1,
        0,
        public_prompt("correction_prompt", selected),
        InteractivePredictOptions(multimask_output=False),
    )
    if stale.preview_handle is None or replacement.preview_handle is None:
        raise RuntimeError("replacement previews must be commit-capable")
    session.commit(replacement.preview_handle)
    stale_rejected = False
    try:
        session.commit(stale.preview_handle)
    except PreviewHandleError:
        stale_rejected = True
    replacement_invalidated = 1 not in session._state.require_object(1).non_conditioning
    repropagated = session.propagate(start_frame=1, end_frame=1)[0]
    replacement_entry = session._state.require_object(1).non_conditioning[1]
    replacement_metric = _prediction_metrics(
        official["memory1_single_logits"][0],
        official["memory1_single_scores"][0],
        replacement_entry.low_res_logits[None],
        repropagated.scores[0][None],
    )
    primary_counters = dict(session._adapter.counters)
    session.close()

    max_session = create_video_session(BASE_VIDEO_PLAN_ID, bundle_dir=bundle_dir)
    max_session.set_video([frames[0]] * 20 + [frames[2]])
    max_session.add_object(1)
    for frame_index in (0, 5, 10, 15):
        max_session._state.commit(
            1, replace(source_entry, frame_index=frame_index, conditioning=True)
        )
    for frame_index in range(8, 20):
        max_session._state.commit(
            1, replace(source_entry, frame_index=frame_index, conditioning=False)
        )
    packed_max = max_session._state.pack(
        [1], frame_index=20, reverse=False, video_frame_count=21
    )[0]
    maximum = max_session.propagate(start_frame=20, end_frame=20)[0]
    maximum_entry = max_session._state.require_object(1).non_conditioning[20]
    maximum_metric = _prediction_metrics(
        official["memory_max_single_logits"][0],
        official["memory_max_single_scores"][0],
        maximum_entry.low_res_logits[None],
        maximum.scores[0][None],
    )
    maximum_packing = {
        "spatial_valid": int(packed_max.memory_valid[0].sum()),
        "pointer_valid": int(packed_max.pointer_valid[0].sum()),
        "conditioning_spatial": int(packed_max.memory_conditioning[0].sum()),
        "non_conditioning_ages": packed_max.memory_age[0, 4:].tolist(),
        "pointer_tpos_denominator": float(packed_max.pointer_tpos_denominator[0]),
    }
    maximum_counters = dict(max_session._adapter.counters)
    max_session.close()

    reverse_session = create_video_session(BASE_VIDEO_PLAN_ID, bundle_dir=bundle_dir)
    reverse_session.set_video(frames)
    reverse_session.add_object(1)
    reverse_session._state.commit(
        1, replace(source_entry, frame_index=2, conditioning=True)
    )
    reverse_packed = reverse_session._state.pack(
        [1], frame_index=1, reverse=True, video_frame_count=3
    )[0]
    reverse = reverse_session.propagate(start_frame=1, end_frame=1, reverse=True)[0]
    reverse_entry = reverse_session._state.require_object(1).non_conditioning[1]
    reverse_metric = _prediction_metrics(
        official["memory1_single_logits"][0],
        official["memory1_single_scores"][0],
        reverse_entry.low_res_logits[None],
        reverse.scores[0][None],
    )
    reverse_age = int(reverse_packed.memory_age[0, 0])
    reverse_session.close()

    def batch_case(object_count: int) -> dict[str, object]:
        batch_session = create_video_session(BASE_VIDEO_PLAN_ID, bundle_dir=bundle_dir)
        batch_session.set_video(frames[:2])
        for object_id in range(object_count):
            batch_session.add_object(object_id)
            batch_session._state.commit(
                object_id,
                replace(source_entry, frame_index=0, conditioning=True),
            )
        before = dict(batch_session._adapter.counters)
        prediction = batch_session.propagate(start_frame=1, end_frame=1)[0]
        after = dict(batch_session._adapter.counters)
        result = {
            "object_count": object_count,
            "frame_encode_delta": after["frame_encodes"] - before["frame_encodes"],
            "tracker_launch_delta": after["tracker_launches"]
            - before["tracker_launches"],
            "memory_commit_delta": after["memory_commits"] - before["memory_commits"],
            "first_mask": prediction.masks[0],
            "first_score": float(prediction.scores[0]),
            "counters": after,
        }
        batch_session.close()
        return result

    batch_raw = [batch_case(count) for count in (1, 4, 5)]
    reference_mask = batch_raw[0].pop("first_mask")
    reference_score = batch_raw[0].pop("first_score")
    batch_isolation = True
    for value in batch_raw[1:]:
        batch_isolation = bool(
            batch_isolation
            and np.array_equal(reference_mask, value.pop("first_mask"))
            and reference_score == value.pop("first_score")
        )
    batch_launches = {
        "cases": batch_raw,
        "isolation": batch_isolation,
        "pass": bool(
            batch_isolation
            and [value["frame_encode_delta"] for value in batch_raw] == [1, 1, 1]
            and [value["tracker_launch_delta"] for value in batch_raw] == [1, 1, 2]
            and [value["memory_commit_delta"] for value in batch_raw] == [1, 4, 5]
        ),
    }

    report = {
        "memory0_multimask": _prediction_metrics(
            official["memory0_multimask_logits"][0],
            official["memory0_multimask_scores"][0],
            multi.low_res_logits,
            multi.scores,
        ),
        "memory0_single": _prediction_metrics(
            official["memory0_single_logits"][0],
            official["memory0_single_scores"][0],
            single.low_res_logits,
            single.scores,
        ),
        "memory1_single": _prediction_metrics(
            official["memory1_single_logits"][0],
            official["memory1_single_scores"][0],
            next_entry.low_res_logits[None],
            next_frame.scores[0][None],
        ),
        "absent_memory": _memory_metrics(
            official["absent_memory_features"], absent_memory
        ),
        "memory_max_single": maximum_metric,
        "memory_max_packing": maximum_packing,
        "forward_reverse": {
            "forward": _prediction_metrics(
                official["memory1_single_logits"][0],
                official["memory1_single_scores"][0],
                next_entry.low_res_logits[None],
                next_frame.scores[0][None],
            ),
            "reverse": reverse_metric,
            "reverse_signed_age": reverse_age,
        },
        "correction_replacement": {
            "stale_handle_rejected": stale_rejected,
            "downstream_invalidated": replacement_invalidated,
            "repropagated": replacement_metric,
        },
        "batch_and_chunk": batch_launches,
        "preview_non_mutating": preview_non_mutating,
        "initial_commit": {
            "object_id": committed.object_id,
            "frame_index": committed.frame_index,
        },
        "counters": {
            "primary": primary_counters,
            "memory_max": maximum_counters,
        },
    }
    report["status"] = (
        "pass"
        if all(
            [
                report["memory0_multimask"]["pass"],
                report["memory0_single"]["pass"],
                report["memory1_single"]["pass"],
                report["memory_max_single"]["pass"],
                report["absent_memory"]["pass"],
                report["forward_reverse"]["reverse"]["pass"],
                report["forward_reverse"]["reverse_signed_age"] == -1,
                report["correction_replacement"]["stale_handle_rejected"],
                report["correction_replacement"]["downstream_invalidated"],
                report["correction_replacement"]["repropagated"]["pass"],
                report["batch_and_chunk"]["pass"],
                report["memory_max_packing"]["spatial_valid"] == 10,
                report["memory_max_packing"]["pointer_valid"] == 16,
                preview_non_mutating,
            ]
        )
        else "fail"
    )
    return report


def validate_bundle(bundle_dir: Path, checkpoint: Path) -> dict[str, object]:
    resolved = validate_manifest_package(
        bundle_dir / "manifests" / f"{BASE_VIDEO_PLAN_ID}.json",
        expected_plan_id=BASE_VIDEO_PLAN_ID,
    )
    fixtures = json.loads((bundle_dir / "fixtures/cases.json").read_text())
    official = np.load(bundle_dir / "fixtures/official_reference.npz")
    frames = [
        Image.open(bundle_dir / f"fixtures/frames/frame_{index:03d}.png").convert("RGB")
        for index in range(int(fixtures["video"]["frame_count"]))
    ]
    modules = build_base_video_modules(checkpoint, device="cuda", dtype="fp16")
    local = _local_reference(modules, frames, fixtures)
    official_local = _official_local_report(official, local)
    del modules, local
    gc.collect()
    torch.cuda.empty_cache()
    public = _public_trajectory(bundle_dir, frames, fixtures, official)
    export_report = json.loads(
        (bundle_dir / "reports/export_report.json").read_text(encoding="utf-8")
    )
    ep = {
        role: value["eager_to_exported_program"]
        for role, value in export_report["graphs"].items()
    }
    hashes = {
        record["path"]: record["digest"]["value"]
        for record in resolved.manifest["files"]
        if record["role"] in {"graph", "external-data"}
    }
    passed = official_local["status"] == "pass" and public["status"] == "pass"
    return {
        "format": "sam3-base-video-m4-release-validation-v1",
        "status": "pass" if passed else "fail",
        "plan_id": resolved.plan_id,
        "profile_id": resolved.profile_id,
        "scope_label": resolved.manifest["scope"]["scope_label"],
        "environment": _environment(),
        "artifact_hashes": hashes,
        "stages": {
            "official_to_local_eager": official_local,
            "local_eager_to_exported_program": ep,
            "exported_program_to_ort_cuda_and_public_api": public,
        },
        "trajectory_gates": {
            "memory0": public["memory0_single"],
            "memory1": public["memory1_single"],
            "memory_max": public["memory_max_single"],
            "object_absence": public["absent_memory"],
            "repeated_correction_preview_non_mutating": public["preview_non_mutating"],
            "forward_reverse": public["forward_reverse"],
            "correction_replacement": public["correction_replacement"],
            "batch_and_chunk": public["batch_and_chunk"],
        },
        "residency_and_copies": {
            "frame_memory_pointer": "CUDA OrtValue / DLPack D2D packing",
            "public_d2h": "scores, final low-resolution logits, final masks only",
            "fallback_plan": None,
            "counters": public["counters"],
        },
        "m5_exclusions": {
            "mux_artifacts": 0,
            "demux_artifacts": 0,
            "bucket_state": 0,
            "multiplex_sessions": 0,
        },
        "fixture_hash": sha256_file(bundle_dir / "fixtures/cases.json"),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--update-report", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    bundle_dir = args.bundle_dir.resolve()
    report = validate_bundle(bundle_dir, args.checkpoint.resolve())
    if report["status"] != "pass":
        raise RuntimeError(json.dumps(report, indent=2))
    if args.update_report:
        report_path = bundle_dir / "reports/m4_release_validation.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        refresh_manifest_file_records(bundle_dir)
        validate_manifest_package(
            bundle_dir / "manifests" / f"{BASE_VIDEO_PLAN_ID}.json",
            expected_plan_id=BASE_VIDEO_PLAN_ID,
        )
    else:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
