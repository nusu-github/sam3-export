"""Capture compact official-eager references for the M1 image PCS fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch


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
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    official_root = args.official_root.resolve()
    sys.path.insert(0, str(official_root))

    from sam3.model.data_misc import FindStage  # noqa: PLC0415
    from sam3.model.sam3_image_processor import Sam3Processor  # noqa: PLC0415
    from sam3.model_builder import build_sam3_image_model  # noqa: PLC0415

    fixtures = json.loads(args.fixtures.read_text(encoding="utf-8"))
    image_records = {record["id"]: record for record in fixtures["images"]}
    workspace = official_root.parent
    bpe_path = official_root / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    model = build_sam3_image_model(
        bpe_path=str(bpe_path),
        device="cuda",
        eval_mode=True,
        checkpoint_path=str(args.checkpoint.resolve()),
        load_from_HF=False,
        enable_segmentation=True,
        enable_inst_interactivity=False,
    ).eval()
    processor = Sam3Processor(model, resolution=1008, device="cuda")
    find_stage = FindStage(
        img_ids=torch.tensor([0], device="cuda", dtype=torch.long),
        text_ids=torch.tensor([0], device="cuda", dtype=torch.long),
        input_boxes=None,
        input_boxes_mask=None,
        input_boxes_label=None,
        input_points=None,
        input_points_mask=None,
    )

    arrays: dict[str, np.ndarray] = {}
    case_records: list[dict[str, object]] = []
    image_states: dict[str, dict[str, object]] = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for case in fixtures["cases"]:
            image_id = case["image"]
            if image_id not in image_states:
                image_record = image_records[image_id]
                image_path = workspace / image_record["workspace_path"]
                if _sha256(image_path) != image_record["sha256"]:
                    raise RuntimeError(f"fixture image hash mismatch: {image_path}")
                image_states[image_id] = processor.set_image(
                    Image.open(image_path).convert("RGB")
                )

            state = image_states[image_id]
            text_outputs = model.backbone.forward_text([case["text"]], device="cuda")
            backbone_out = dict(state["backbone_out"])
            backbone_out.update(text_outputs)
            outputs = model.forward_grounding(
                backbone_out=backbone_out,
                find_input=find_stage,
                geometric_prompt=model._get_dummy_prompt(),
                find_target=None,
            )
            logits = outputs["pred_logits"]
            boxes = outputs["pred_boxes"]
            masks = outputs["pred_masks"]
            presence = outputs["presence_logit_dec"]
            scores = (logits.sigmoid() * presence.sigmoid().unsqueeze(1)).squeeze(-1)
            if scores.ndim == 2:
                scores = scores[0]
            if boxes.ndim == 3:
                boxes = boxes[0]
            if masks.ndim == 4:
                masks = masks[0]
            indices = torch.argsort(scores, descending=True, stable=True)[:16]
            prefix = case["id"]
            arrays[f"{prefix}__scores"] = scores.float().cpu().numpy()
            arrays[f"{prefix}__indices"] = indices.cpu().numpy()
            arrays[f"{prefix}__boxes"] = boxes[indices].float().cpu().numpy()
            arrays[f"{prefix}__masks"] = masks[indices].float().cpu().numpy()
            case_records.append(
                {
                    "id": prefix,
                    "positive_over_0_5": int(torch.count_nonzero(scores > 0.5)),
                    "top_score": float(scores[indices[0]]),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    metadata = {
        "format": "sam3-m1-official-reference-v1",
        "official_commit": fixtures["official_source_commit"],
        "checkpoint_sha256": _sha256(args.checkpoint),
        "fixture_version": fixtures["fixture_version"],
        "dtype": "torch.autocast(cuda, float16)",
        "cases": case_records,
        "npz_sha256": _sha256(args.output),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
