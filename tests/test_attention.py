"""Attention parity: SDPA core (eval + train dropout)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from sam3.primitives import Attention

if not torch.cuda.is_available():
    pytest.skip("CUDA is required for attention tests", allow_module_level=True)

DEVICE = torch.device("cuda")


def _tol(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float16:
        return 2e-2, 2e-3
    if dtype == torch.bfloat16:
        return 3e-2, 3e-3
    return 5e-3, 3e-3


def _reference_attention_layer_output(
    layer: Attention,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float,
) -> torch.Tensor:
    q = F.linear(q, layer.q_proj.weight, layer.q_proj.bias)
    k = F.linear(k, layer.k_proj.weight, layer.k_proj.bias)
    v = F.linear(v, layer.v_proj.weight, layer.v_proj.bias)

    q = layer._separate_heads(q, layer.num_heads)
    k = layer._separate_heads(k, layer.num_heads)
    v = layer._separate_heads(v, layer.num_heads)

    out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
    out = layer._recombine_heads(out)
    out = F.linear(out, layer.out_proj.weight, layer.out_proj.bias)
    return out


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_attention_layer_matches_sdpa_reference(dtype: torch.dtype) -> None:
    torch.manual_seed(11)
    layer = Attention(
        embedding_dim=64,
        num_heads=8,
        dropout=0.0,
    ).to(device=DEVICE, dtype=dtype)
    layer.eval()
    q = torch.randn(4, 12, 64, device=DEVICE, dtype=dtype)
    k = torch.randn(4, 12, 64, device=DEVICE, dtype=dtype)
    v = torch.randn(4, 12, 64, device=DEVICE, dtype=dtype)

    with torch.no_grad():
        out = layer(q, k, v)
        ref = _reference_attention_layer_output(layer, q, k, v, dropout_p=0.0)

    rtol, atol = _tol(dtype)
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_attention_odd_head_dim_works(dtype: torch.dtype) -> None:
    torch.manual_seed(13)
    layer = Attention(
        embedding_dim=50,
        num_heads=5,
        dropout=0.0,
    ).to(device=DEVICE, dtype=dtype)
    layer.eval()
    q = torch.randn(2, 10, 50, device=DEVICE, dtype=dtype)
    k = torch.randn(2, 10, 50, device=DEVICE, dtype=dtype)
    v = torch.randn(2, 10, 50, device=DEVICE, dtype=dtype)

    with torch.no_grad():
        out = layer(q, k, v)
        ref = _reference_attention_layer_output(layer, q, k, v, dropout_p=0.0)

    rtol, atol = _tol(dtype)
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)


def test_training_dropout_runs() -> None:
    """Training with dropout>0 uses SDPA dropout_p (finite output)."""
    torch.manual_seed(19)
    layer = Attention(
        embedding_dim=32,
        num_heads=4,
        dropout=0.5,
    ).to(device=DEVICE, dtype=torch.float32)
    layer.train()
    q = torch.randn(2, 8, 32, device=DEVICE)
    k = torch.randn(2, 8, 32, device=DEVICE)
    v = torch.randn(2, 8, 32, device=DEVICE)

    out = layer(q, k, v)
    assert out.shape == (2, 8, 32)
    assert torch.isfinite(out).all()


def test_eval_disables_dropout() -> None:
    torch.manual_seed(5)
    layer = Attention(
        embedding_dim=32,
        num_heads=4,
        dropout=0.9,
    ).to(device=DEVICE, dtype=torch.float32)
    layer.eval()
    q = torch.randn(2, 8, 32, device=DEVICE)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    with torch.no_grad():
        a = layer(q, k, v)
        b = layer(q, k, v)
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_backward_with_dropout() -> None:
    """SDPA path is differentiable under training dropout."""
    torch.manual_seed(42)
    layer = Attention(32, 4, dropout=0.1).to(device=DEVICE, dtype=torch.float32)
    layer.train()
    q = torch.randn(2, 8, 32, device=DEVICE, requires_grad=True)
    k = torch.randn(2, 8, 32, device=DEVICE, requires_grad=True)
    v = torch.randn(2, 8, 32, device=DEVICE, requires_grad=True)
    out = layer(q, k, v)
    out.sum().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert k.grad is not None and torch.isfinite(k.grad).all()
    assert v.grad is not None and torch.isfinite(v.grad).all()
