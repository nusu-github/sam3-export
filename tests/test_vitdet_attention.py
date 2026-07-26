"""Parity tests for ViTDet Attention."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from sam3.primitives.rope import apply_rotary_enc_real
from sam3.vision.vitdet_attention import Attention, concat_rel_pos

if not torch.cuda.is_available():
    pytest.skip("CUDA is required for ViTDet Attention tests", allow_module_level=True)

DEVICE = torch.device("cuda")


def _tol(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float16:
        return 2e-2, 2e-3
    if dtype == torch.bfloat16:
        return 3e-2, 3e-3
    return 5e-3, 1e-3


def _reference_forward(layer: Attention, x: torch.Tensor) -> torch.Tensor:
    batch, h, w, _ = x.shape
    qkv = F.linear(x, layer.qkv.weight, layer.qkv.bias).reshape(
        batch, h * w, 3, layer.num_heads, -1
    )
    q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

    if layer.use_rope:
        q, k = apply_rotary_enc_real(
            q,
            k,
            freqs_cis_real=layer.freqs_cis_real,
            freqs_cis_imag=layer.freqs_cis_imag,
            repeat_freqs_k=False,
        )

    if layer.use_rel_pos:
        q, k = concat_rel_pos(
            q.flatten(0, 1),
            k.flatten(0, 1),
            (h, w),
            (h, w),
            layer.rel_pos_h,
            layer.rel_pos_w,
            rescale=True,
            relative_coords=layer.relative_coords,
        )
        q = q.reshape(batch, layer.num_heads, h * w, -1)
        k = k.reshape(batch, layer.num_heads, h * w, -1)

    ref = F.scaled_dot_product_attention(q, k, v, scale=layer.scale)
    ref = (
        ref.view(batch, layer.num_heads, h, w, -1)
        .permute(0, 2, 3, 1, 4)
        .reshape(batch, h, w, -1)
    )
    return F.linear(ref, layer.proj.weight, layer.proj.bias)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("use_rope", [False, True])
@pytest.mark.parametrize("use_rel_pos", [False, True])
def test_vitdet_attention_matches_torch_sdpa(
    dtype: torch.dtype, use_rope: bool, use_rel_pos: bool
) -> None:
    torch.manual_seed(7)
    layer = Attention(
        dim=64,
        num_heads=4,
        use_rel_pos=use_rel_pos,
        input_size=(8, 8),
        use_rope=use_rope,
        rope_pt_size=(8, 8),
    ).to(device=DEVICE, dtype=dtype)
    layer.eval()
    x = torch.randn((2, 8, 8, 64), device=DEVICE, dtype=dtype)

    with torch.no_grad():
        out = layer(x)
        ref = _reference_forward(layer, x)

    rtol, atol = _tol(dtype)
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_vitdet_attention_real_rope_matches_torch_sdpa(dtype: torch.dtype) -> None:
    torch.manual_seed(11)
    layer = Attention(
        dim=64,
        num_heads=4,
        use_rel_pos=False,
        input_size=(8, 8),
        use_rope=True,
        use_rope_real=True,
        rope_pt_size=(8, 8),
    ).to(device=DEVICE, dtype=dtype)
    layer.eval()
    x = torch.randn((2, 8, 8, 64), device=DEVICE, dtype=dtype)

    with torch.no_grad():
        out = layer(x)
        ref = _reference_forward(layer, x)

    rtol, atol = _tol(dtype)
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)
