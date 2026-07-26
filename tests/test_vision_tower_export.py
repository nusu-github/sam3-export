"""Cut A: VisionTower contracts + torch.export smoke."""

from __future__ import annotations

import pytest
import torch
from torch.export import export

from sam3.export import (
    VISION_D_MODEL,
    VISION_IMAGE_SIZE,
    VISION_NUM_LEVELS,
    VISION_SPATIAL,
    VisionTower,
    VisionTowerFlat,
    VisionTowerSpec,
    flat_vision_keys,
    validate_vision_output,
    vision_output_from_flat,
    vision_output_to_flat,
)
from sam3.weights.load_sam3 import build_production_vision_backbone

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

DEVICE = torch.device("cuda")


@pytest.fixture(scope="module")
def neck():
    m = build_production_vision_backbone(
        load_weights=False,
        add_sam2_neck=True,
        device=DEVICE,
    )
    m.eval()
    return m


@pytest.fixture(scope="module")
def pixels(neck):
    return torch.randn(
        1,
        3,
        VISION_IMAGE_SIZE,
        VISION_IMAGE_SIZE,
        device=DEVICE,
        dtype=next(neck.parameters()).dtype,
    )


@torch.no_grad()
def test_vision_tower_shapes(neck, pixels):
    tower = VisionTower(neck, validate=True)
    out = tower(pixels)
    validate_vision_output(
        out,
        batch=1,
        dtype=pixels.dtype,
        spec=VisionTowerSpec(add_sam2=True),
    )
    assert len(out.sam3_fpn) == VISION_NUM_LEVELS
    for i, hw in enumerate(VISION_SPATIAL):
        assert out.sam3_fpn[i].shape == (1, VISION_D_MODEL, hw, hw)
        assert out.sam2_fpn is not None
        assert out.sam2_fpn[i].shape == (1, VISION_D_MODEL, hw, hw)


@torch.no_grad()
def test_flat_pack_unpack(neck, pixels):
    tower = VisionTower(neck)
    out = tower(pixels)
    flat = vision_output_to_flat(out)
    assert len(flat) == len(flat_vision_keys(True))
    back = vision_output_from_flat(flat, add_sam2=True)
    for a, b in zip(out.sam3_fpn, back.sam3_fpn):
        assert torch.equal(a, b)
    assert out.sam2_fpn is not None and back.sam2_fpn is not None
    for a, b in zip(out.sam2_fpn, back.sam2_fpn):
        assert torch.equal(a, b)


@torch.no_grad()
def test_vision_tower_flat_export(neck, pixels):
    """Export path is pure ATen RoPE; re-run matches eager."""
    flat_mod = VisionTowerFlat(neck).eval()
    eager = flat_mod(pixels)
    assert len(eager) == 16
    ep = export(flat_mod, (pixels,), strict=False)
    exported = ep.module()(pixels)

    assert len(exported) == len(eager)
    for e, x in zip(eager, exported):
        assert torch.isfinite(x).all()
        torch.testing.assert_close(e, x, rtol=1e-4, atol=1e-4)


@torch.no_grad()
def test_validate_rejects_bad_input(neck):
    tower = VisionTower(neck, validate=True)
    bad = torch.randn(1, 3, 224, 224, device=DEVICE)
    with pytest.raises(ValueError, match="spatial"):
        tower(bad)
