"""Generate the fixed-revision official SAM3.1 Multiplex trajectory fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image
import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(raw, dict) and isinstance(raw.get("model"), dict):
        raw = raw["model"]
    if not isinstance(raw, dict):
        raise TypeError("official checkpoint is not a flat state dictionary")
    return raw


def _load_model(official_root: Path, checkpoint: Path) -> Any:
    sys.path.insert(0, str(official_root))
    from sam3.model_builder import build_sam3_multiplex_video_model

    raw = _checkpoint_state(checkpoint)
    model = build_sam3_multiplex_video_model(
        checkpoint_path=None,
        load_from_HF=False,
        multiplex_count=16,
        use_fa3=False,
        use_rope_real=False,
        device="cuda",
    )
    tracker_prefix = "tracker.model."
    neck_prefix = "detector.backbone.vision_backbone."
    mapped: dict[str, torch.Tensor] = {}
    for name, value in raw.items():
        if name.startswith(tracker_prefix):
            mapped[name[len(tracker_prefix) :]] = value
        elif name.startswith(neck_prefix):
            mapped["backbone.vision_backbone." + name[len(neck_prefix) :]] = value
    missing, unexpected = model.load_state_dict(mapped, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"official mapping mismatch: missing={missing} unexpected={unexpected}"
        )
    return model.eval()


def _frame_cache(
    model: Any, frame_paths: list[Path]
) -> tuple[dict[int, tuple[torch.Tensor, Any]], tuple[int, int]]:
    from sam3.model.data_misc import NestedTensor

    result: dict[int, tuple[torch.Tensor, Any]] = {}
    original_size: tuple[int, int] | None = None
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for index, path in enumerate(frame_paths):
            image = Image.open(path).convert("RGB")
            original_size = (image.height, image.width)
            resized = np.asarray(
                image.resize((1008, 1008), Image.Resampling.BILINEAR),
                dtype=np.float32,
            )
            resized = resized / 255.0
            tensor = (
                torch.from_numpy((resized - 0.5) / 0.5)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to("cuda")
            )
            features = model.forward_image(
                NestedTensor(tensor, None),
                need_sam3_out=False,
                need_interactive_out=True,
                need_propagation_out=True,
            )
            result[index] = (tensor, features)
    if original_size is None:
        raise ValueError("fixture requires at least one frame")
    return result, original_size


def _trajectory(
    model: Any,
    cached: dict[int, tuple[torch.Tensor, Any]],
    original_size: tuple[int, int],
    points: list[list[float]],
    count: int,
    correction: dict[str, Any] | None,
) -> dict[str, np.ndarray]:
    object_ids = list(range(100, 100 + count))
    multiplex_state = model.multiplex_controller.get_state(
        num_valid_entries=count,
        device="cuda",
        dtype=torch.float32,
        random=False,
        object_ids=object_ids,
    )
    output_dict: dict[str, dict[int, dict[str, Any]]] = {
        "cond_frame_outputs": {},
        "non_cond_frame_outputs": {},
    }

    def frame_views(
        features: Any,
    ) -> tuple[
        dict[str, Any],
        dict[str, Any],
        list[torch.Tensor],
        list[torch.Tensor],
    ]:
        prepared = model._prepare_backbone_features(features)
        interactive = prepared["interactive"]
        propagation = prepared["sam2_backbone_out"]
        interactive_high = [
            value.permute(1, 2, 0).view(value.size(1), value.size(2), *size)
            for value, size in zip(
                interactive["vision_feats"][:-1],
                interactive["feat_sizes"][:-1],
            )
        ]
        propagation_high = [
            value.permute(1, 2, 0).view(value.size(1), value.size(2), *size)
            for value, size in zip(
                propagation["vision_feats"][:-1],
                propagation["feat_sizes"][:-1],
            )
        ]
        return interactive, propagation, interactive_high, propagation_high

    def interactive_output(
        interactive: dict[str, Any],
        interactive_high: list[torch.Tensor],
        point_values: list[list[float]],
        labels: list[int],
        objects: list[int],
    ) -> dict[str, torch.Tensor]:
        coords = torch.tensor(
            point_values,
            dtype=torch.float32,
            device="cuda",
        ).unsqueeze(0)
        coords = coords * model.image_size
        point_labels = torch.tensor(
            labels,
            dtype=torch.int32,
            device="cuda",
        ).unsqueeze(0)
        image = model._get_interactive_pix_mem(
            interactive["vision_feats"],
            interactive["feat_sizes"],
        )
        return model._forward_sam_heads(
            backbone_features=image,
            point_inputs={
                "point_coords": coords,
                "point_labels": point_labels,
            },
            interactive_high_res_features=interactive_high,
            multimask_output=False,
            objects_to_interact=objects,
            multiplex_state=multiplex_state,
        )

    def store_frame(
        frame_index: int,
        image: torch.Tensor,
        propagation: dict[str, Any],
        sam_output: dict[str, torch.Tensor],
        conditioning_objects: set[int],
    ) -> dict[str, Any]:
        memory, memory_position = model._encode_new_memory(
            image=image,
            current_vision_feats=propagation["vision_feats"],
            feat_sizes=propagation["feat_sizes"],
            pred_masks_high_res=sam_output["high_res_masks"],
            object_score_logits=sam_output["object_score_logits"],
            is_mask_from_pts=bool(conditioning_objects),
            conditioning_objects=conditioning_objects,
            multiplex_state=multiplex_state,
        )
        entry: dict[str, Any] = {
            "maskmem_features": memory,
            "maskmem_pos_enc": memory_position,
            "image_features": propagation["vision_feats"][-1],
            "image_pos_enc": propagation["vision_pos_embeds"][-1],
            "pred_masks": sam_output["low_res_masks"],
            "pred_masks_high_res": sam_output["high_res_masks"],
            "obj_ptr": multiplex_state.mux(sam_output["obj_ptr"]),
            "object_score_logits": sam_output["object_score_logits"],
            "conditioning_objects": conditioning_objects,
        }
        destination = (
            output_dict["cond_frame_outputs"]
            if conditioning_objects
            else output_dict["non_cond_frame_outputs"]
        )
        destination[frame_index] = entry
        if conditioning_objects:
            output_dict["non_cond_frame_outputs"].pop(frame_index, None)
        return entry

    arrays: dict[str, np.ndarray] = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        final: dict[str, Any] | None = None
        for frame_index, (image, features) in cached.items():
            interactive, propagation, interactive_high, propagation_high = frame_views(
                features
            )
            if frame_index == 0:
                initial_outputs = [
                    interactive_output(
                        interactive,
                        interactive_high,
                        [points[offset]],
                        [1],
                        [offset],
                    )
                    for offset in range(count)
                ]
                sam_output = {
                    key: torch.cat([item[key] for item in initial_outputs], dim=0)
                    for key in initial_outputs[0]
                }
                conditioning_objects = set(range(count))
            else:
                conditioned = model._prepare_memory_conditioned_features(
                    frame_idx=frame_index,
                    is_init_cond_frame=False,
                    current_vision_feats=propagation["vision_feats"][-1:],
                    current_vision_masks=propagation["vision_masks"][-1:],
                    current_vision_pos_embeds=propagation["vision_pos_embeds"][-1:],
                    feat_sizes=propagation["feat_sizes"][-1:],
                    output_dict=output_dict,
                    num_frames=len(cached),
                    track_in_reverse=False,
                    multiplex_state=multiplex_state,
                )
                sam_output = model._forward_sam_heads(
                    backbone_features=conditioned,
                    propagation_high_res_features=propagation_high,
                    multimask_output=True,
                    objects_to_interact=list(range(count)),
                    multiplex_state=multiplex_state,
                )
                conditioning_objects = set()
                if correction is not None and frame_index == int(
                    correction["frame_index"]
                ):
                    arrays[f"count{count}_frame{frame_index}_pre_correction_low"] = (
                        sam_output["low_res_masks"][:, 0].float().cpu().numpy()
                    )
                    corrected = interactive_output(
                        interactive,
                        interactive_high,
                        correction["points_xy_relative"],
                        correction["point_labels"],
                        [0],
                    )
                    for key in (
                        "low_res_multimasks",
                        "high_res_multimasks",
                        "low_res_masks",
                        "high_res_masks",
                        "ious",
                        "object_score_logits",
                        "obj_ptr",
                    ):
                        sam_output[key][0:1] = corrected[key]
                    conditioning_objects = {0}
            final = store_frame(
                frame_index,
                image,
                propagation,
                sam_output,
                conditioning_objects,
            )
            arrays[f"count{count}_frame{frame_index}_low"] = (
                sam_output["low_res_masks"][:, 0].float().cpu().numpy()
            )
            arrays[f"count{count}_frame{frame_index}_object_score"] = (
                sam_output["object_score_logits"][:, 0].float().cpu().numpy()
            )
        if final is None:
            raise RuntimeError("official trajectory produced no frames")
        arrays[f"count{count}_final_pointer"] = (
            multiplex_state.demux(final["obj_ptr"]).float().cpu().numpy()
        )
        arrays[f"count{count}_final_memory"] = (
            final["maskmem_features"].float().cpu().numpy()
        )
        arrays[f"count{count}_final_memory_position"] = (
            final["maskmem_pos_enc"][0].float().cpu().numpy()
        )
    return arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    args = parser.parse_args()
    fixtures = json.loads(args.fixtures.read_text(encoding="utf-8"))
    if _sha256(args.checkpoint) != fixtures["checkpoint_sha256"]:
        raise RuntimeError("SAM3.1 checkpoint digest mismatch")
    model = _load_model(args.official_root.resolve(), args.checkpoint.resolve())
    frame_paths = sorted(args.frames_dir.glob("frame_*.png"))
    cached, original_size = _frame_cache(model, frame_paths)
    arrays: dict[str, np.ndarray] = {}
    for count in fixtures["active_slot_cases"]:
        arrays.update(
            _trajectory(
                model,
                cached,
                original_size,
                fixtures["object_points_xy_relative"],
                int(count),
                fixtures["correction"],
            )
        )
    # The official model is one native bucket.  The M5 two-bucket reference is
    # deliberately the composition of independent one-bucket trajectories.
    no_correction: dict[int, dict[str, np.ndarray]] = {}
    for second_count in sorted(
        {int(count) - 16 for count in fixtures["two_bucket_cases"]}
    ):
        values = _trajectory(
            model,
            cached,
            original_size,
            fixtures["object_points_xy_relative"],
            second_count,
            None,
        )
        no_correction[second_count] = values
        prefix = f"count{second_count}_"
        for key, value in values.items():
            arrays[f"count{second_count}_no_correction_{key.removeprefix(prefix)}"] = (
                value
            )
    for count in fixtures["two_bucket_cases"]:
        first_count = 16
        second_count = int(count) - first_count
        second = no_correction[second_count]
        for frame_index in range(len(cached)):
            for suffix in ("low", "object_score"):
                first = arrays[f"count16_frame{frame_index}_{suffix}"]
                second_value = second[
                    f"count{second_count}_frame{frame_index}_{suffix}"
                ]
                arrays[f"count{count}_frame{frame_index}_{suffix}"] = np.concatenate(
                    (first, second_value), axis=0
                )
            if frame_index == int(fixtures["correction"]["frame_index"]):
                first_pre = arrays[f"count16_frame{frame_index}_pre_correction_low"]
                second_pre = second[f"count{second_count}_frame{frame_index}_low"]
                arrays[f"count{count}_frame{frame_index}_pre_correction_low"] = (
                    np.concatenate((first_pre, second_pre), axis=0)
                )
        for suffix in (
            "final_pointer",
            "final_memory",
            "final_memory_position",
        ):
            arrays[f"count{count}_{suffix}"] = np.concatenate(
                (
                    arrays[f"count16_{suffix}"],
                    second[f"count{second_count}_{suffix}"],
                ),
                axis=0,
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    metadata = {
        "format": "m5-sam31-multiplex-official-reference-v1",
        "official_source_commit": fixtures["official_source_commit"],
        "model_revision": fixtures["model_revision"],
        "checkpoint_sha256": fixtures["checkpoint_sha256"],
        "mapping": {
            "tracker.model": 457,
            "detector.backbone.vision_backbone": 474,
            "missing": [],
            "unexpected": [],
        },
        "active_slot_cases": fixtures["active_slot_cases"],
        "two_bucket_reference": "independent native one-bucket composition",
        "array_count": len(arrays),
    }
    args.metadata_output.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
