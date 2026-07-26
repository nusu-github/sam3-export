"""Capture official eager references for the owned M3 interactive fixture."""

from __future__ import annotations

import argparse
import hashlib
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


def _radial_logits() -> np.ndarray:
    axis = np.linspace(-1.0, 1.0, 288, dtype=np.float32)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    return ((0.58**2 - (xx * xx + yy * yy)) * 12.0).astype(np.float32)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    official_root = args.official_root.resolve()
    sys.path.insert(0, str(official_root))
    if "pkg_resources" not in sys.modules:
        pkg_resources = types.ModuleType("pkg_resources")
        pkg_resources.resource_filename = lambda package, relative: str(  # type: ignore[attr-defined]
            official_root / package / relative
        )
        sys.modules["pkg_resources"] = pkg_resources

    from sam3.model.sam3_image_processor import Sam3Processor  # noqa: PLC0415
    from sam3.model_builder import build_sam3_image_model  # noqa: PLC0415

    fixtures = json.loads(args.fixtures.read_text(encoding="utf-8"))
    if _sha256(args.checkpoint) != fixtures["checkpoint_sha256"]:
        raise RuntimeError("checkpoint does not match the M3 fixture")
    if _sha256(args.image) != fixtures["image"]["sha256"]:
        raise RuntimeError("image does not match the M3 fixture")

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
    for module in model.modules():
        if not getattr(module, "use_rope", False) or not hasattr(
            module, "use_rope_real"
        ):
            continue
        frequencies = getattr(module, "freqs_cis", None)
        if frequencies is None:
            continue
        if getattr(module, "freqs_cis_real", None) is None:
            module.register_buffer("freqs_cis_real", frequencies.real.float())
            module.register_buffer("freqs_cis_imag", frequencies.imag.float())
        module.use_rope_real = True
    processor = Sam3Processor(model, resolution=1008, device="cuda")
    image = Image.open(args.image).convert("RGB")
    arrays: dict[str, np.ndarray] = {}
    records: list[dict[str, object]] = []

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        # The Public API owns preprocessing on the host.  Feed those exact
        # fp16 values to official eager as well so every parity stage compares
        # the same fixture bytes instead of different CPU/CUDA resize kernels.
        pixel_values = (
            processor.transform(v2.functional.to_image(image))
            .unsqueeze(0)
            .half()
            .to("cuda")
        )
        arrays["pixel_values"] = pixel_values.float().cpu().numpy()
        state = {
            "original_height": image.height,
            "original_width": image.width,
            "backbone_out": model.backbone.forward_image(pixel_values),
        }
        sam2_backbone_out = state["backbone_out"]["sam2_backbone_out"]
        official_tracker = model.inst_interactive_predictor.model
        sam2_backbone_out["backbone_fpn"][0] = (
            official_tracker.sam_mask_decoder.conv_s0(
                sam2_backbone_out["backbone_fpn"][0]
            )
        )
        sam2_backbone_out["backbone_fpn"][1] = (
            official_tracker.sam_mask_decoder.conv_s1(
                sam2_backbone_out["backbone_fpn"][1]
            )
        )
        tracker = model.inst_interactive_predictor.model
        backbone_out = state["backbone_out"]["sam2_backbone_out"]
        _, vision_feats, _, _ = tracker._prepare_backbone_features(backbone_out)
        vision_feats[-1] = vision_feats[-1] + tracker.no_mem_embed
        sizes = model.inst_interactive_predictor._bb_feat_sizes
        features = [
            feat.permute(1, 2, 0).view(1, -1, *size)
            for feat, size in zip(vision_feats[::-1], sizes[::-1])
        ][::-1]
        arrays["image_embedding"] = features[-1].float().cpu().numpy()
        arrays["high_res_0"] = features[0].float().cpu().numpy()
        arrays["high_res_1"] = features[1].float().cpu().numpy()

        radial = _radial_logits()
        for case in fixtures["cases"]:
            points = np.asarray(case["points_xy"], dtype=np.float32)
            labels = np.asarray(case["point_labels"], dtype=np.int32)
            box = (
                None
                if case["box_xyxy"] is None
                else np.asarray(case["box_xyxy"], dtype=np.float32)
            )
            mask = radial[None] if case["mask_source"] is not None else None
            _masks, scores, low_res = model.predict_inst(
                state,
                point_coords=points if points.size else None,
                point_labels=labels if labels.size else None,
                box=box,
                mask_input=mask,
                multimask_output=bool(case["multimask_output"]),
                return_logits=True,
                normalize_coords=True,
            )
            prefix = case["id"]
            arrays[f"{prefix}__scores"] = scores.astype(np.float32)
            arrays[f"{prefix}__low_res"] = low_res.astype(np.float32)
            records.append(
                {
                    "id": prefix,
                    "mask_count": int(low_res.shape[0]),
                    "top_score_index": int(np.argmax(scores)),
                }
            )

        repeated = fixtures["repeated_click"]
        first_points = np.asarray(repeated["first_points_xy"], dtype=np.float32)
        first_labels = np.asarray(repeated["first_point_labels"], dtype=np.int32)
        _masks, first_scores, first_low_res = model.predict_inst(
            state,
            point_coords=first_points,
            point_labels=first_labels,
            multimask_output=True,
            return_logits=True,
            normalize_coords=True,
        )
        selected_index = int(np.argmax(first_scores))
        second_points = np.asarray(repeated["second_points_xy"], dtype=np.float32)
        second_labels = np.asarray(repeated["second_point_labels"], dtype=np.int32)
        _masks, second_scores, second_low_res = model.predict_inst(
            state,
            point_coords=second_points,
            point_labels=second_labels,
            mask_input=first_low_res[selected_index][None],
            multimask_output=False,
            return_logits=True,
            normalize_coords=True,
        )
        prefix = repeated["id"]
        arrays[f"{prefix}__first_scores"] = first_scores.astype(np.float32)
        arrays[f"{prefix}__first_low_res"] = first_low_res.astype(np.float32)
        arrays[f"{prefix}__selected_index"] = np.asarray(selected_index, dtype=np.int64)
        arrays[f"{prefix}__second_scores"] = second_scores.astype(np.float32)
        arrays[f"{prefix}__second_low_res"] = second_low_res.astype(np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    metadata = {
        "format": "sam3-m3-official-interactive-reference-v1",
        "official_commit": fixtures["official_source_commit"],
        "checkpoint_sha256": fixtures["checkpoint_sha256"],
        "fixture_version": fixtures["fixture_version"],
        "dtype": "permanent float16 weights with autocast for float32 PE edges",
        "image_encode_count": 1,
        "predict_launch_count": len(fixtures["cases"]) + 2,
        "memory_encode_count": 0,
        "memory_commit_count": 0,
        "cases": records,
        "npz_sha256": _sha256(args.output),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
