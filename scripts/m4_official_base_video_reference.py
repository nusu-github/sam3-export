"""Generate official eager reference tensors for the M4 base-video fixture."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import types

import numpy as np
from PIL import Image
import torch
from torchvision.transforms import v2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    return parser.parse_args()


def _enable_real_rope(module: torch.nn.Module) -> None:
    for child in module.modules():
        if not getattr(child, "use_rope", False) or not hasattr(child, "use_rope_real"):
            continue
        frequencies = getattr(child, "freqs_cis", None)
        if frequencies is None:
            continue
        if getattr(child, "freqs_cis_real", None) is None:
            child.register_buffer("freqs_cis_real", frequencies.real.float())
            child.register_buffer("freqs_cis_imag", frequencies.imag.float())
        child.use_rope_real = True


def main() -> int:
    args = _parse_args()
    official_root = args.official_root.resolve()
    sys.path.insert(0, str(official_root))
    if "pkg_resources" not in sys.modules:
        pkg_resources = types.ModuleType("pkg_resources")
        pkg_resources.resource_filename = lambda package, name: str(
            official_root / package / name
        )
        pkg_resources.get_distribution = lambda name: types.SimpleNamespace(
            version=importlib.metadata.version(name)
        )
        sys.modules["pkg_resources"] = pkg_resources

    from sam3.model.sam3_image_processor import Sam3Processor  # noqa: PLC0415
    from sam3.model_builder import build_sam3_image_model  # noqa: PLC0415

    fixtures = json.loads(args.fixtures.read_text(encoding="utf-8"))
    if _sha256(args.checkpoint) != fixtures["checkpoint_sha256"]:
        raise RuntimeError("checkpoint does not match the M4 fixture")
    model = (
        build_sam3_image_model(
            bpe_path=str(official_root / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"),
            device="cuda",
            eval_mode=True,
            checkpoint_path=str(args.checkpoint.resolve()),
            load_from_HF=False,
            enable_segmentation=False,
            enable_inst_interactivity=True,
        )
        .eval()
        .half()
    )
    _enable_real_rope(model)
    processor = Sam3Processor(model, resolution=1008, device="cuda")
    tracker = model.inst_interactive_predictor.model
    frames = [
        Image.open(args.frames_dir / f"frame_{index:03d}.png").convert("RGB")
        for index in range(int(fixtures["video"]["frame_count"]))
    ]
    arrays: dict[str, np.ndarray] = {}

    def encode_frame(index: int):
        pixel = (
            processor.transform(v2.functional.to_image(frames[index]))
            .unsqueeze(0)
            .half()
            .to("cuda")
        )
        backbone = model.backbone.forward_image(pixel)["sam2_backbone_out"]
        backbone["backbone_fpn"][0] = tracker.sam_mask_decoder.conv_s0(
            backbone["backbone_fpn"][0]
        )
        backbone["backbone_fpn"][1] = tracker.sam_mask_decoder.conv_s1(
            backbone["backbone_fpn"][1]
        )
        _, features, positions, sizes = tracker._prepare_backbone_features(backbone)
        return pixel, features, positions, sizes

    def point_inputs(name: str) -> dict[str, torch.Tensor]:
        record = fixtures[name]
        coords = np.asarray(record["points_xy"], dtype=np.float32) * 1008.0
        labels = np.asarray(record["point_labels"], dtype=np.int32)
        return {
            "point_coords": torch.from_numpy(coords)[None].to("cuda"),
            "point_labels": torch.from_numpy(labels)[None].to("cuda"),
        }

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        pixel0, features0, positions0, sizes0 = encode_frame(0)
        raw0 = features0[-1].permute(1, 2, 0).reshape(1, 256, 72, 72)
        pos0 = positions0[-1].permute(1, 2, 0).reshape_as(raw0)
        high0 = features0[0].permute(1, 2, 0).reshape(1, 32, 288, 288)
        high1 = features0[1].permute(1, 2, 0).reshape(1, 64, 144, 144)
        arrays["frame0_image_embedding"] = raw0.float().cpu().numpy()
        arrays["frame0_image_position"] = pos0.float().cpu().numpy()
        arrays["frame0_high_res_0"] = high0.float().cpu().numpy()
        arrays["frame0_high_res_1"] = high1.float().cpu().numpy()

        conditioned0 = (features0[-1] + tracker.no_mem_embed).permute(1, 2, 0)
        conditioned0 = conditioned0.reshape(1, 256, 72, 72)
        multi = tracker._forward_sam_heads(
            conditioned0,
            point_inputs=point_inputs("initial_prompt"),
            high_res_features=[high0, high1],
            multimask_output=True,
        )
        arrays["memory0_multimask_logits"] = multi[0].float().cpu().numpy()
        arrays["memory0_multimask_scores"] = multi[2].float().cpu().numpy()
        selected = multi[3]
        single = tracker._forward_sam_heads(
            conditioned0,
            point_inputs=point_inputs("correction_prompt"),
            mask_inputs=selected,
            high_res_features=[high0, high1],
            multimask_output=False,
        )
        arrays["memory0_single_logits"] = single[0].float().cpu().numpy()
        arrays["memory0_single_scores"] = single[2].float().cpu().numpy()
        arrays["memory0_pointer"] = single[5].float().cpu().numpy()
        arrays["memory0_object_score"] = single[6].float().cpu().numpy()
        memory0, memory0_pos = tracker._encode_new_memory(
            image=pixel0,
            current_vision_feats=features0,
            feat_sizes=sizes0,
            pred_masks_high_res=single[4],
            object_score_logits=single[6],
            is_mask_from_pts=True,
        )
        arrays["memory0_features"] = memory0.float().cpu().numpy()
        arrays["memory0_position"] = memory0_pos[0].float().cpu().numpy()
        entry0 = {
            "maskmem_features": memory0,
            "maskmem_pos_enc": memory0_pos,
            "obj_ptr": single[5],
            "pred_masks": single[3],
            "pred_masks_high_res": single[4],
            "object_score_logits": single[6],
        }

        pixel1, features1, positions1, sizes1 = encode_frame(1)
        output_dict = {"cond_frame_outputs": {0: entry0}, "non_cond_frame_outputs": {}}
        conditioned1 = tracker._prepare_memory_conditioned_features(
            frame_idx=1,
            is_init_cond_frame=False,
            current_vision_feats=features1[-1:],
            current_vision_pos_embeds=positions1[-1:],
            feat_sizes=sizes1[-1:],
            output_dict=output_dict,
            num_frames=3,
            track_in_reverse=False,
        )
        one = tracker._forward_sam_heads(
            conditioned1,
            point_inputs=None,
            high_res_features=[
                features1[0].permute(1, 2, 0).reshape(1, 32, 288, 288),
                features1[1].permute(1, 2, 0).reshape(1, 64, 144, 144),
            ],
            multimask_output=False,
        )
        arrays["memory1_single_logits"] = one[0].float().cpu().numpy()
        arrays["memory1_single_scores"] = one[2].float().cpu().numpy()
        arrays["memory1_pointer"] = one[5].float().cpu().numpy()
        memory1, memory1_pos = tracker._encode_new_memory(
            image=pixel1,
            current_vision_feats=features1,
            feat_sizes=sizes1,
            pred_masks_high_res=one[4],
            object_score_logits=one[6],
            is_mask_from_pts=False,
        )
        arrays["memory1_features"] = memory1.float().cpu().numpy()
        arrays["memory1_position"] = memory1_pos[0].float().cpu().numpy()

        _, features2, positions2, sizes2 = encode_frame(2)
        cond = {frame: entry0 for frame in (0, 5, 10, 15)}
        non_cond = {frame: entry0 for frame in range(8, 20)}
        max_conditioned = tracker._prepare_memory_conditioned_features(
            frame_idx=20,
            is_init_cond_frame=False,
            current_vision_feats=features2[-1:],
            current_vision_pos_embeds=positions2[-1:],
            feat_sizes=sizes2[-1:],
            output_dict={
                "cond_frame_outputs": cond,
                "non_cond_frame_outputs": non_cond,
            },
            num_frames=21,
            track_in_reverse=False,
        )
        maximum = tracker._forward_sam_heads(
            max_conditioned,
            point_inputs=None,
            high_res_features=[
                features2[0].permute(1, 2, 0).reshape(1, 32, 288, 288),
                features2[1].permute(1, 2, 0).reshape(1, 64, 144, 144),
            ],
            multimask_output=False,
        )
        arrays["memory_max_single_logits"] = maximum[0].float().cpu().numpy()
        arrays["memory_max_single_scores"] = maximum[2].float().cpu().numpy()
        arrays["memory_max_pointer"] = maximum[5].float().cpu().numpy()

        absent_memory, absent_position = tracker._encode_new_memory(
            image=pixel0,
            current_vision_feats=features0,
            feat_sizes=sizes0,
            pred_masks_high_res=single[4],
            object_score_logits=-torch.ones_like(single[6]),
            is_mask_from_pts=True,
        )
        arrays["absent_memory_features"] = absent_memory.float().cpu().numpy()
        arrays["absent_memory_position"] = absent_position[0].float().cpu().numpy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    args.metadata_output.write_text(
        json.dumps(
            {
                "format": "sam3-base-video-m4-official-reference-v1",
                "checkpoint_sha256": fixtures["checkpoint_sha256"],
                "array_shapes": {
                    name: list(value.shape) for name, value in arrays.items()
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
