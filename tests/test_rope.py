"""RoPE parity tests for pure PyTorch reference implementations."""

from __future__ import annotations

import pytest
import torch

from sam3.primitives import rope

if not torch.cuda.is_available():
    pytest.skip("CUDA is required for RoPE tests", allow_module_level=True)

DEVICE = torch.device("cuda")


def _tol(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float16:
        return 2e-2, 2e-3
    return 5e-3, 3e-3


def _reference_init_t_xy(
    end_x: int,
    end_y: int,
    scale: float = 1.0,
    offset: int = 0,
    device: torch.device | None = None,
):
    device = device or DEVICE
    t = torch.arange(end_x * end_y, dtype=torch.float32, device=device)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode="floor").float()
    return t_x * scale + offset, t_y * scale + offset


def _reference_compute_axial_cis(
    dim: int,
    end_x: int,
    end_y: int,
    theta: float = 10000.0,
    scale_pos: float = 1.0,
    offset: int = 0,
    device: torch.device | None = None,
):
    device = device or DEVICE
    freqs_x = 1.0 / (
        theta ** (torch.arange(0, dim, 4, device=device)[: (dim // 4)].float() / dim)
    )
    freqs_y = 1.0 / (
        theta ** (torch.arange(0, dim, 4, device=device)[: (dim // 4)].float() / dim)
    )
    t_x, t_y = _reference_init_t_xy(end_x, end_y, scale_pos, offset, device=device)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)


def _reference_reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[-2], x.shape[-1])
    shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def _reference_complex_mult(
    x_real: torch.Tensor,
    x_imag: torch.Tensor,
    freqs_real: torch.Tensor,
    freqs_imag: torch.Tensor,
) -> torch.Tensor:
    real_part = x_real * freqs_real - x_imag * freqs_imag
    imag_part = x_real * freqs_imag + x_imag * freqs_real
    return torch.stack([real_part, imag_part], dim=-1)


def _reference_apply_rotary_enc(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    repeat_freqs_k: bool = False,
):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = (
        torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        if xk.shape[-2] != 0
        else None
    )
    freqs_cis = _reference_reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    if xk_ is None:
        return xq_out.type_as(xq).to(xq.device), xk
    if repeat_freqs_k:
        r = xk_.shape[-2] // xq_.shape[-2]
        freqs_cis = freqs_cis.repeat(*([1] * (freqs_cis.ndim - 2)), r, 1)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)


def _reference_apply_rotary_enc_real(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis_real: torch.Tensor,
    freqs_cis_imag: torch.Tensor,
    repeat_freqs_k: bool = False,
):
    assert xk.shape[-2] != 0
    xq_real = xq.float().reshape(*xq.shape[:-1], -1, 2)[..., 0]
    xq_imag = xq.float().reshape(*xq.shape[:-1], -1, 2)[..., 1]
    xk_real = xk.float().reshape(*xk.shape[:-1], -1, 2)[..., 0]
    xk_imag = xk.float().reshape(*xk.shape[:-1], -1, 2)[..., 1]
    freqs_cis_real = _reference_reshape_for_broadcast(freqs_cis_real, xq_real)
    freqs_cis_imag = _reference_reshape_for_broadcast(freqs_cis_imag, xq_imag)
    xq_out = _reference_complex_mult(
        xq_real, xq_imag, freqs_cis_real, freqs_cis_imag
    ).flatten(3)
    if repeat_freqs_k:
        r = xk_real.shape[-2] // xq_real.shape[-2]
        freqs_cis_real = freqs_cis_real.repeat(*([1] * (freqs_cis_real.ndim - 2)), r, 1)
        freqs_cis_imag = freqs_cis_imag.repeat(*([1] * (freqs_cis_imag.ndim - 2)), r, 1)
    xk_out = _reference_complex_mult(
        xk_real, xk_imag, freqs_cis_real, freqs_cis_imag
    ).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)


def test_init_and_compute_axial_cis_matches_reference() -> None:
    end_x, end_y = 8, 8
    dim = 64
    tx_ref, ty_ref = _reference_init_t_xy(end_x, end_y, device=DEVICE)
    tx, ty = rope.init_t_xy(end_x, end_y, device=DEVICE)
    torch.testing.assert_close(tx, tx_ref)
    torch.testing.assert_close(ty, ty_ref)

    ref = _reference_compute_axial_cis(dim=dim, end_x=end_x, end_y=end_y, device=DEVICE)
    got = rope.compute_axial_cis(dim=dim, end_x=end_x, end_y=end_y, device=DEVICE)
    torch.testing.assert_close(got, ref)


def test_reshape_for_broadcast_matches_reference() -> None:
    x = torch.randn((2, 3, 16, 32), device=DEVICE)
    freqs = torch.randn((16, 16), device=DEVICE)
    expected = _reference_reshape_for_broadcast(freqs, x[..., :16])
    got = rope.reshape_for_broadcast(freqs, x[..., :16])
    torch.testing.assert_close(got, expected)


def test_apply_rotary_enc_matches_reference() -> None:
    dtype = torch.float16
    end_x, end_y = 4, 4
    dim = 32
    q = torch.randn((2, 4, end_x * end_y, dim), device=DEVICE, dtype=dtype)
    k = torch.randn((2, 4, end_x * end_y, dim), device=DEVICE, dtype=dtype)
    freqs = rope.compute_axial_cis(dim=dim, end_x=end_x, end_y=end_y, device=DEVICE)

    out_q, out_k = rope.apply_rotary_enc(q, k, freqs)
    ref_q, ref_k = _reference_apply_rotary_enc(q, k, freqs)

    rtol, atol = _tol(dtype)
    torch.testing.assert_close(out_q, ref_q, rtol=rtol, atol=atol)
    torch.testing.assert_close(out_k, ref_k, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_apply_rotary_enc_matches_reference_parametrized(dtype: torch.dtype) -> None:
    torch.manual_seed(7)
    end_x, end_y = 4, 4
    dim = 64
    q = torch.randn((2, 4, end_x * end_y, dim), device=DEVICE, dtype=dtype)
    k = torch.randn((2, 4, end_x * end_y, dim), device=DEVICE, dtype=dtype)
    freqs = rope.compute_axial_cis(dim=dim, end_x=end_x, end_y=end_y, device=DEVICE)

    out_q, out_k = rope.apply_rotary_enc(q, k, freqs)
    ref_q, ref_k = _reference_apply_rotary_enc(q, k, freqs)
    rtol, atol = _tol(dtype)
    torch.testing.assert_close(out_q, ref_q, rtol=rtol, atol=atol)
    torch.testing.assert_close(out_k, ref_k, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_apply_rotary_enc_repeat_k_lengths(dtype: torch.dtype) -> None:
    torch.manual_seed(11)
    dim = 64
    q_len_x, q_len_y = 2, 4
    q_len = q_len_x * q_len_y
    q = torch.randn((2, 3, q_len, dim), device=DEVICE, dtype=dtype)
    k = torch.randn((2, 3, q_len * 2, dim), device=DEVICE, dtype=dtype)
    freqs = rope.compute_axial_cis(dim=dim, end_x=q_len_x, end_y=q_len_y, device=DEVICE)

    out_q, out_k = rope.apply_rotary_enc(q, k, freqs, repeat_freqs_k=True)
    ref_q, ref_k = _reference_apply_rotary_enc(q, k, freqs, repeat_freqs_k=True)

    rtol, atol = _tol(dtype)
    torch.testing.assert_close(out_q, ref_q, rtol=rtol, atol=atol)
    torch.testing.assert_close(out_k, ref_k, rtol=rtol, atol=atol)


def test_apply_rotary_enc_keeps_empty_k_tensor() -> None:
    dtype = torch.float16
    q = torch.randn((2, 4, 16, 64), device=DEVICE, dtype=dtype)
    k = torch.empty((2, 4, 0, 64), device=DEVICE, dtype=dtype)
    freqs = rope.compute_axial_cis(dim=64, end_x=4, end_y=4, device=DEVICE)

    out_q, out_k = rope.apply_rotary_enc(q, k, freqs)
    ref_q, ref_k = _reference_apply_rotary_enc(q, k, freqs)

    torch.testing.assert_close(out_q, ref_q)
    assert out_k is k
    torch.testing.assert_close(ref_k, k)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_apply_rotary_enc_real_matches_reference(dtype: torch.dtype) -> None:
    torch.manual_seed(19)
    end_x, end_y = 4, 4
    q = torch.randn((1, 2, end_x * end_y, 32), device=DEVICE, dtype=dtype)
    k = torch.randn((1, 2, end_x * end_y, 32), device=DEVICE, dtype=dtype)
    freqs = rope.compute_axial_cis(dim=32, end_x=end_x, end_y=end_y, device=DEVICE)

    out_q, out_k = rope.apply_rotary_enc_real(q, k, freqs.real, freqs.imag)
    ref_q, ref_k = _reference_apply_rotary_enc_real(q, k, freqs.real, freqs.imag)

    rtol, atol = _tol(dtype)
    torch.testing.assert_close(out_q, ref_q, rtol=rtol, atol=atol)
    torch.testing.assert_close(out_k, ref_k, rtol=rtol, atol=atol)
