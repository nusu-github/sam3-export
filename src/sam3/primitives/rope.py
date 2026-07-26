"""Pure-ATen rotary position-embedding helpers."""

from __future__ import annotations

from jaxtyping import Complex, Float
import torch
from torch import Tensor


def init_t_xy(
    end_x: int,
    end_y: int,
    scale: float = 1.0,
    offset: int = 0,
    device: torch.device | str | None = None,
) -> tuple[Float[Tensor, "seq"], Float[Tensor, "seq"]]:
    t = torch.arange(end_x * end_y, dtype=torch.float32, device=device)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode="floor").float()
    return t_x * scale + offset, t_y * scale + offset


def compute_axial_cis(
    dim: int,
    end_x: int,
    end_y: int,
    theta: float = 10000.0,
    scale_pos: float = 1.0,
    offset: int = 0,
    device: torch.device | str | None = None,
) -> Complex[Tensor, "seq half"]:
    freqs_x = 1.0 / (
        theta ** (torch.arange(0, dim, 4, device=device)[: (dim // 4)].float() / dim)
    )
    freqs_y = 1.0 / (
        theta ** (torch.arange(0, dim, 4, device=device)[: (dim // 4)].float() / dim)
    )
    t_x, t_y = init_t_xy(end_x, end_y, scale_pos, offset, device=device)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)


def reshape_for_broadcast(
    freqs_cis: Float[Tensor, "seq dim"] | Complex[Tensor, "seq dim"],
    x: Float[Tensor, "*leading seq dim"] | Complex[Tensor, "*leading seq dim"],
) -> Float[Tensor, "*leading seq dim"] | Complex[Tensor, "*leading seq dim"]:
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[-2], x.shape[-1])
    shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def complex_mult(
    xq_real: Float[Tensor, "*batch"],
    xq_imag: Float[Tensor, "*batch"],
    freqs_cis_real: Float[Tensor, "*batch"],
    freqs_cis_imag: Float[Tensor, "*batch"],
) -> Float[Tensor, "*batch 2"]:
    real_part = xq_real * freqs_cis_real - xq_imag * freqs_cis_imag
    imag_part = xq_real * freqs_cis_imag + xq_imag * freqs_cis_real
    return torch.stack([real_part, imag_part], dim=-1)


def apply_rotary_enc(
    xq: Float[Tensor, "*leading seq_q dim"],
    xk: Float[Tensor, "*leading seq_k dim"],
    freqs_cis: Complex[Tensor, "seq half"],
    repeat_freqs_k: bool = False,
) -> tuple[Float[Tensor, "*leading seq_q dim"], Float[Tensor, "*leading seq_k dim"]]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = (
        torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        if xk.shape[-2] != 0
        else None
    )

    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    q_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    if xk_ is None:
        return q_out.type_as(xq).to(xq.device), xk
    if repeat_freqs_k:
        r = xk_.shape[-2] // xq_.shape[-2]
        freqs_cis = freqs_cis.repeat(*([1] * (freqs_cis.ndim - 2)), r, 1)
    k_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return q_out.type_as(xq).to(xq.device), k_out.type_as(xk).to(xk.device)


def apply_rotary_enc_real(
    xq: Float[Tensor, "*leading seq_q dim"],
    xk: Float[Tensor, "*leading seq_k dim"],
    freqs_cis_real: Float[Tensor, "freq_seq half"],
    freqs_cis_imag: Float[Tensor, "freq_seq half"],
    repeat_freqs_k: bool = False,
) -> tuple[Float[Tensor, "*leading seq_q dim"], Float[Tensor, "*leading seq_k dim"]]:
    """Real-valued RoPE; freqs are broadcast against **query** length then
    optionally repeated for longer keys (matches official SAM3).
    """
    assert xk is not None
    assert xk.shape[-2] != 0

    xq_real = xq.float().reshape(*xq.shape[:-1], -1, 2)[..., 0]
    xq_imag = xq.float().reshape(*xq.shape[:-1], -1, 2)[..., 1]
    xk_real = xk.float().reshape(*xk.shape[:-1], -1, 2)[..., 0]
    xk_imag = xk.float().reshape(*xk.shape[:-1], -1, 2)[..., 1]

    freqs_real = reshape_for_broadcast(freqs_cis_real, xq_real)
    freqs_imag = reshape_for_broadcast(freqs_cis_imag, xq_imag)
    q_out = complex_mult(xq_real, xq_imag, freqs_real, freqs_imag).flatten(3)

    if repeat_freqs_k:
        r = xk_real.shape[-2] // xq_real.shape[-2]
        freqs_real = freqs_real.repeat(*([1] * (freqs_real.ndim - 2)), r, 1)
        freqs_imag = freqs_imag.repeat(*([1] * (freqs_imag.ndim - 2)), r, 1)
    k_out = complex_mult(xk_real, xk_imag, freqs_real, freqs_imag).flatten(3)
    return q_out.type_as(xq).to(xq.device), k_out.type_as(xk).to(xk.device)
