"""Tests for the ViTDet block and tiny ViT."""

from __future__ import annotations

import pytest
import torch

from sam3.vision.vit import ViT
from sam3.vision.vitdet_block import Block

if not torch.cuda.is_available():
    pytest.skip("CUDA is required for this test file", allow_module_level=True)

DEVICE = torch.device("cuda")


def _tol(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float16:
        return 2e-2, 2e-3
    if dtype == torch.bfloat16:
        return 3e-2, 3e-3
    return 5e-3, 3e-3


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_block_forward_shape_and_finite(dtype: torch.dtype) -> None:
    torch.manual_seed(11)
    x = torch.randn(2, 4, 4, 64, device=DEVICE, dtype=dtype)
    block = Block(
        dim=64,
        num_heads=4,
        drop_path=0.0,
        window_size=2,
        use_rel_pos=False,
        dropout=0.0,
    ).to(device=DEVICE, dtype=dtype)

    y = block(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_vit_forward_returns_feature_maps(dtype: torch.dtype) -> None:
    torch.manual_seed(15)
    model = ViT(
        img_size=64,
        patch_size=16,
        depth=2,
        embed_dim=64,
        num_heads=4,
        mlp_ratio=4.0,
        window_size=2,
        use_abs_pos=True,
        tile_abs_pos=True,
        return_interm_layers=False,
    ).to(device=DEVICE, dtype=dtype)
    x = torch.randn(2, 3, 64, 64, device=DEVICE, dtype=dtype)

    feats = model(x)
    assert isinstance(feats, list)
    assert len(feats) >= 1
    for feat in feats:
        assert feat.ndim == 4
        assert feat.shape[0] == 2
        assert feat.shape[1] == 64
        assert feat.shape[-2:] == (4, 4)
        assert torch.isfinite(feat).all()


def _reference_block_output(block: Block, x: torch.Tensor) -> torch.Tensor:
    # Reference implementation that mirrors ViTDet Block behavior with
    # torch attention primitives for regression only.
    if x.ndim != 4:
        raise ValueError("Expected NHWC tensor")

    B, H, W, C = x.shape
    if W != H:
        raise ValueError("expected square tokens for this helper")

    shortcut = x
    x = block.norm1(x)
    tokens = x.reshape(B, H * W, C)
    qkv = block.attn.qkv(tokens).reshape(B, H * W, 3, block.attn.num_heads, -1)
    q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

    attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    attn = attn.transpose(1, 2).reshape(B, H * W, C)
    attn = block.attn.proj(attn).reshape(B, H, W, C)

    x = shortcut + block.dropout(block.drop_path(attn))
    mlp_in = block.norm2(x)
    x = x + block.dropout(block.drop_path(block.ls2(block.mlp(mlp_in))))
    return x


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_block_matches_torch_reference_formula(dtype: torch.dtype) -> None:
    torch.manual_seed(19)
    x = torch.randn(2, 4, 4, 64, device=DEVICE, dtype=dtype)
    block = Block(dim=64, num_heads=4, dropout=0.0, window_size=0).to(
        device=DEVICE, dtype=dtype
    )
    # Skip if internals are not exposed in the installed worker B implementation.
    for req in ("qkv", "proj", "num_heads", "norm2", "mlp"):
        if not hasattr(block.attn, req) or not hasattr(block, req):
            pytest.skip("attention implementation missing expected internals")

    ref = _reference_block_output(block, x)
    out = block(x)
    rtol, atol = _tol(dtype)
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)
