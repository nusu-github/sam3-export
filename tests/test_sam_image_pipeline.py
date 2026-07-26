"""Full image pipeline: ViT → SamImageHead."""

from __future__ import annotations

import pytest
import torch

from sam3.vision.sam_image_pipeline import SamImagePipeline

if not torch.cuda.is_available():
    pytest.skip("CUDA is required for image pipeline tests", allow_module_level=True)

DEVICE = torch.device("cuda")


def _make_pipeline(dtype: torch.dtype = torch.float32) -> SamImagePipeline:
    torch.manual_seed(0)
    pipe = SamImagePipeline(
        img_size=64,
        patch_size=16,
        embed_dim=32,
        vit_depth=2,
        vit_heads=4,
        vit_window_size=2,
        transformer_depth=1,
        transformer_heads=4,
        transformer_mlp_dim=64,
    ).to(device=DEVICE, dtype=dtype)
    pipe.eval()
    return pipe


def test_encode_image_shape():
    pipe = _make_pipeline()
    image = torch.randn(2, 3, 64, 64, device=DEVICE)
    with torch.no_grad():
        emb = pipe.encode_image(image)
    assert emb.shape == (2, 32, 4, 4)
    assert torch.isfinite(emb).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("multimask", [False, True])
def test_pipeline_points_end_to_end(dtype: torch.dtype, multimask: bool):
    torch.manual_seed(7)
    pipe = _make_pipeline(dtype)
    b = 2
    image = torch.randn(b, 3, 64, 64, device=DEVICE, dtype=dtype)
    coords = torch.rand(b, 3, 2, device=DEVICE, dtype=dtype) * 64
    labels = torch.randint(0, 2, (b, 3), device=DEVICE)

    with torch.no_grad():
        masks, iou, tokens, obj = pipe(
            image, points=(coords, labels), multimask_output=multimask
        )

    n_masks = 3 if multimask else 1
    # MaskDecoder upsamples 4× from 4×4 → 16×16
    assert masks.shape == (b, n_masks, 16, 16)
    assert iou.shape == (b, n_masks)
    assert tokens.shape[0] == b and tokens.shape[-1] == 32
    assert obj.shape[0] == b
    assert torch.isfinite(masks).all()
    assert torch.isfinite(iou).all()
    assert torch.isfinite(tokens).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_pipeline_boxes_end_to_end(dtype: torch.dtype):
    torch.manual_seed(11)
    pipe = _make_pipeline(dtype)
    b = 2
    image = torch.randn(b, 3, 64, 64, device=DEVICE, dtype=dtype)
    boxes = torch.tensor(
        [[4.0, 5.0, 40.0, 50.0], [2.0, 3.0, 30.0, 35.0]],
        device=DEVICE,
        dtype=dtype,
    )
    with torch.no_grad():
        masks, iou, tokens, obj = pipe(image, boxes=boxes, multimask_output=True)
    assert masks.shape == (b, 3, 16, 16)
    assert torch.isfinite(masks).all()


def test_pipeline_deterministic_in_eval():
    pipe = _make_pipeline(torch.float32)
    image = torch.randn(1, 3, 64, 64, device=DEVICE)
    coords = torch.tensor([[[10.0, 12.0], [40.0, 45.0]]], device=DEVICE)
    labels = torch.tensor([[1, 1]], device=DEVICE)
    with torch.no_grad():
        a = pipe(image, points=(coords, labels), multimask_output=False)
        b = pipe(image, points=(coords, labels), multimask_output=False)
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, rtol=0, atol=0)


def test_pipeline_rejects_bad_image_size():
    pipe = _make_pipeline()
    bad = torch.randn(1, 3, 32, 32, device=DEVICE)
    with pytest.raises(ValueError, match="spatial size"):
        pipe.encode_image(bad)


def test_pipeline_matches_manual_compose():
    """encode_image + mask_head equals forward()."""
    pipe = _make_pipeline()
    image = torch.randn(1, 3, 64, 64, device=DEVICE)
    coords = torch.rand(1, 2, 2, device=DEVICE) * 64
    labels = torch.ones(1, 2, device=DEVICE)
    with torch.no_grad():
        emb = pipe.encode_image(image)
        manual = pipe.mask_head(emb, points=(coords, labels), multimask_output=True)
        full = pipe(image, points=(coords, labels), multimask_output=True)
    for a, b in zip(manual, full):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
