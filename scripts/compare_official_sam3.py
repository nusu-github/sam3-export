"""Manually compare this implementation against an official SAM3 checkout.

This is deliberately a script, rather than a pytest module.  The two projects
both expose a top-level ``sam3`` package, so each comparison temporarily loads
the official package in an isolated ``sys.modules`` namespace.

Examples:

    uv run python scripts/compare_official_sam3.py \
        --official-root /path/to/sam3 --all
    uv run python scripts/compare_official_sam3.py \
        --official-root /path/to/sam3 --check nms
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
import importlib
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sam3.grounding.text_encoder_ve import TextTransformer  # noqa: E402
from sam3.grounding.tokenizer_ve import SimpleTokenizer  # noqa: E402
from sam3.runtime import mask_ops  # noqa: E402
from sam3.runtime.connected_components import connected_components  # noqa: E402
from sam3.runtime.nms import nms_masks  # noqa: E402
from sam3.weights.load_sam3 import (  # noqa: E402
    build_production_interactive,
    resolve_sam3_checkpoint,
)


@contextmanager
def official_sam3_imports(official_root: Path) -> Iterator[None]:
    """Expose an official checkout as ``sam3`` without leaking it into ours."""
    package_root = official_root / "sam3"
    if not package_root.is_dir():
        raise FileNotFoundError(
            f"{official_root} does not contain an official sam3 package"
        )

    original_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "sam3" or name.startswith("sam3.")
    }
    for name in original_modules:
        del sys.modules[name]

    official_path = str(official_root)
    sys.path.insert(0, official_path)
    importlib.invalidate_caches()
    try:
        yield
    finally:
        for name in tuple(sys.modules):
            if name == "sam3" or name.startswith("sam3."):
                del sys.modules[name]
        sys.modules.update(original_modules)
        sys.path.remove(official_path)
        importlib.invalidate_caches()


def _canonicalize_labels_cpu(labels: torch.Tensor) -> torch.Tensor:
    flat = labels.reshape(-1).to(torch.int64).cpu().tolist()
    mapping: dict[int, int] = {}
    next_id = 1
    canonical: list[int] = []
    for value in flat:
        if value == 0:
            canonical.append(0)
            continue
        key = int(value)
        if key not in mapping:
            mapping[key] = next_id
            next_id += 1
        canonical.append(mapping[key])
    return torch.tensor(canonical, dtype=torch.int64).reshape(labels.shape)


def compare_masks_to_boxes(official_root: Path, _: argparse.Namespace) -> None:
    masks = torch.tensor(
        [
            [[0, 1, 2], [0, 0, 0], [1, 0, 0]],
            [[0, 0, 0], [3, 3, 0], [0, 0, 3]],
        ],
        dtype=torch.float32,
    )
    ours = mask_ops.masks_to_boxes(masks)
    with official_sam3_imports(official_root):
        from sam3.perflib.masks_ops import masks_to_boxes as official_masks_to_boxes

        reference = official_masks_to_boxes(masks, list(range(masks.shape[0])))
    torch.testing.assert_close(ours, reference.to(ours.dtype))


def compare_connected_components(official_root: Path, _: argparse.Namespace) -> None:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        try:
            import skimage  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "official CPU connected-components requires scikit-image"
            ) from exc
        device = torch.device("cpu")

    torch.manual_seed(0)
    masks = torch.randint(0, 4, (2, 1, 5, 6), device=device, dtype=torch.int32)
    ours_labels, ours_counts = connected_components(masks)
    with official_sam3_imports(official_root):
        from sam3.perflib.connected_components import (
            connected_components as official_connected_components,
        )

        reference_labels, reference_counts = official_connected_components(masks)

    assert ours_labels.shape == reference_labels.shape
    assert ours_counts.shape == reference_counts.shape
    for batch_index in range(masks.shape[0]):
        assert torch.equal(
            _canonicalize_labels_cpu(ours_labels[batch_index]),
            _canonicalize_labels_cpu(reference_labels[batch_index]),
        )
        torch.testing.assert_close(
            ours_counts[batch_index].to(torch.int32).cpu(),
            reference_counts[batch_index].to(torch.int32).cpu(),
        )


def compare_nms(official_root: Path, _: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("the official NMS implementation requires CUDA")

    probabilities = torch.tensor([0.95, 0.80], device="cuda")
    masks = torch.tensor(
        [
            [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ],
        device="cuda",
    )
    ours = nms_masks(probabilities, masks, prob_threshold=0.5, iou_threshold=0.5)
    with official_sam3_imports(official_root):
        from sam3.perflib.nms import nms_masks as official_nms_masks

        reference = official_nms_masks(probabilities, masks, 0.5, 0.5)
    torch.testing.assert_close(ours, reference)


def _initialize_text_transformer(module: nn.Module) -> None:
    for parameter in module.parameters():
        if parameter.dim() >= 2:
            nn.init.xavier_uniform_(parameter)
        else:
            nn.init.zeros_(parameter)


def compare_text_encoder(official_root: Path, _: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("the text-encoder comparison requires CUDA")

    device = torch.device("cuda")
    context_length = 16
    local_tokenizer = SimpleTokenizer(context_length=context_length)
    texts = ["a photo of a dog", "red car on the street", "The Quick Brown Fox"]
    local_tokens = local_tokenizer(texts, context_length=context_length)

    settings = dict(
        context_length=context_length,
        vocab_size=49408,
        width=64,
        heads=4,
        layers=2,
        output_dim=32,
        pool_type="none",
        output_tokens=True,
        use_ln_post=True,
        use_act_checkpoint=False,
    )
    ours = TextTransformer(**settings).to(device).eval()
    _initialize_text_transformer(ours)
    token_ids = torch.tensor(
        [
            [49406, 320, 1125, 539, 49407, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [49406, 2368, 49407, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        device=device,
        dtype=torch.long,
    )

    bpe_path = official_root / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    with official_sam3_imports(official_root):
        from sam3.model.text_encoder_ve import (
            TextTransformer as OfficialTextTransformer,
        )
        from sam3.model.tokenizer_ve import SimpleTokenizer as OfficialTokenizer

        official_tokenizer = OfficialTokenizer(
            bpe_path=str(bpe_path), context_length=context_length
        )
        official_tokens = official_tokenizer(texts, context_length=context_length)
        reference = OfficialTextTransformer(**settings).to(device).eval()
        reference.load_state_dict(ours.state_dict())
        with torch.no_grad():
            ours_pooled, ours_tokens = ours(token_ids)
            reference_pooled, reference_tokens = reference(token_ids)

    assert torch.equal(local_tokens, official_tokens)
    torch.testing.assert_close(ours_tokens, reference_tokens, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(ours_pooled, reference_pooled, rtol=1e-5, atol=1e-5)


def compare_interactive(official_root: Path, args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("the interactive comparison requires CUDA")

    checkpoint = resolve_sam3_checkpoint(args.checkpoint)
    image_path = official_root / "assets" / "images" / "truck.jpg"
    if not image_path.is_file():
        raise FileNotFoundError(f"official comparison image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    point = np.array([[500.0, 375.0]], dtype=np.float32)
    labels = np.array([1], dtype=np.int32)
    dtype = torch.bfloat16

    predictor = build_production_interactive(
        dtype=dtype, device="cuda", load_weights=True, checkpoint_path=str(checkpoint)
    ).eval()
    with official_sam3_imports(official_root):
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        official = build_sam3_image_model(
            bpe_path=str(
                official_root / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
            ),
            device="cuda",
            eval_mode=True,
            checkpoint_path=str(checkpoint),
            load_from_HF=False,
            enable_inst_interactivity=True,
        )
        with torch.autocast("cuda", dtype=dtype):
            state = Sam3Processor(official, device="cuda").set_image(image)
            official_masks, official_scores, _ = official.predict_inst(
                state,
                point_coords=point,
                point_labels=labels,
                multimask_output=True,
            )

    predictor.set_image(image, dtype=dtype)
    local_masks, local_scores, _ = predictor.predict(
        point_coords=point.tolist(),
        point_labels=labels.tolist(),
        multimask_output=True,
    )
    official_best = official_masks[np.argsort(official_scores)[-1]].astype(bool)
    local_best = local_masks[int(local_scores.argmax())].cpu().numpy().astype(bool)
    union = np.logical_or(official_best, local_best).sum()
    iou = (
        float(np.logical_and(official_best, local_best).sum() / union) if union else 1.0
    )
    assert iou >= 0.95, f"best-mask IoU {iou:.4f} < 0.95"


CHECKS = {
    "masks-to-boxes": compare_masks_to_boxes,
    "connected-components": compare_connected_components,
    "nms": compare_nms,
    "text": compare_text_encoder,
    "interactive": compare_interactive,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--official-root",
        type=Path,
        required=True,
        help="Checkout root of the official SAM3 repository.",
    )
    parser.add_argument(
        "--checkpoint",
        help="Optional local sam3.pt path; used only by --check interactive.",
    )
    parser.add_argument(
        "--check",
        action="append",
        choices=tuple(CHECKS),
        help="Comparison to run; repeat to run more than one.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run every comparison, including CUDA and real-checkpoint checks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = list(CHECKS) if args.all else args.check or ["masks-to-boxes"]
    for name in selected:
        print(f"[compare] {name}")
        CHECKS[name](args.official_root.resolve(), args)
        print(f"[ok] {name}")


if __name__ == "__main__":
    main()
