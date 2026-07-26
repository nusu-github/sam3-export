"""Gate: core ATen-path modules must torch.export; legacy lab paths are not required."""

from __future__ import annotations

import pytest
import torch
from torch.export import export
import torch.nn as nn

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA export smoke"
)


def _export_roundtrip(module: torch.nn.Module, args: tuple, kwargs: dict | None = None):
    kwargs = kwargs or {}
    module = module.eval()
    with torch.no_grad():
        ep = export(module, args, kwargs, strict=False)
        return ep.module()(*args, **kwargs)


@torch.no_grad()
def test_export_linear_cublas():
    m = nn.Linear(32, 64).cuda()
    x = torch.randn(2, 8, 32, device="cuda")
    y = _export_roundtrip(m, (x,))
    assert y.shape == (2, 8, 64)


@torch.no_grad()
def test_export_attention_sdpa():
    from sam3.primitives.attention import Attention

    m = Attention(64, num_heads=4).cuda()
    x = torch.randn(2, 16, 64, device="cuda")
    y = _export_roundtrip(m, (x, x, x))
    assert y.shape == x.shape


@torch.no_grad()
def test_export_mlp_block():
    from sam3.primitives.mlp import MLPBlock

    m = MLPBlock(64, 128).cuda()
    x = torch.randn(2, 16, 64, device="cuda")
    y = _export_roundtrip(m, (x,))
    assert y.shape == x.shape


@torch.no_grad()
def test_export_layernorm_aten():
    m = nn.LayerNorm(64).cuda()
    x = torch.randn(2, 16, 64, device="cuda")
    y = _export_roundtrip(m, (x,))
    assert y.shape == x.shape


@torch.no_grad()
def test_export_prompt_encoder():
    from sam3.vision.prompt_encoder import PromptEncoder

    m = PromptEncoder(
        embed_dim=64,
        image_embedding_size=(8, 8),
        input_image_size=(128, 128),
        mask_in_chans=16,
    ).cuda()
    coords = torch.tensor([[[10.0, 10.0], [20.0, 20.0]]], device="cuda")
    labels = torch.tensor([[1, 0]], device="cuda")
    sparse, dense = _export_roundtrip(
        m, (), {"points": (coords, labels), "boxes": None, "masks": None}
    )
    assert sparse.ndim == 3
    assert dense.ndim == 4


@torch.no_grad()
def test_torch_linear_export_roundtrip_smoke():
    m = nn.Linear(32, 32).cuda()
    x = torch.randn(2, 4, 32, device="cuda")
    y = _export_roundtrip(m, (x,))
    assert y.shape == (2, 4, 32)
