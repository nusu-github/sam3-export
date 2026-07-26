"""Parity tests for ``RoPEAttention``."""

from __future__ import annotations

from functools import partial
import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from sam3.primitives.rope_attention import RoPEAttention

if not torch.cuda.is_available():
    pytest.skip("CUDA is required for RoPEAttention tests", allow_module_level=True)

DEVICE = torch.device("cuda")


def _tol(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float16:
        return 2e-2, 2e-3
    if dtype == torch.bfloat16:
        return 3e-2, 3e-3
    return 1e-2, 3e-3


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
    shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


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
    q_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    if xk_ is None:
        return q_out.type_as(xq).to(xq.device), xk
    if repeat_freqs_k:
        r = xk_.shape[-2] // xq_.shape[-2]
        freqs_cis = freqs_cis.repeat(*([1] * (freqs_cis.ndim - 2)), r, 1)
    k_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return q_out.type_as(xq).to(xq.device), k_out.type_as(xk).to(xk.device)


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
    q_out = torch.stack(
        [
            xq_real * freqs_cis_real - xq_imag * freqs_cis_imag,
            xq_real * freqs_cis_imag + xq_imag * freqs_cis_real,
        ],
        dim=-1,
    ).flatten(3)
    if repeat_freqs_k:
        r = xk_real.shape[-2] // xq_real.shape[-2]
        freqs_cis_real = freqs_cis_real.repeat(*([1] * (freqs_cis_real.ndim - 2)), r, 1)
        freqs_cis_imag = freqs_cis_imag.repeat(*([1] * (freqs_cis_imag.ndim - 2)), r, 1)
    k_out = torch.stack(
        [
            xk_real * freqs_cis_real - xk_imag * freqs_cis_imag,
            xk_real * freqs_cis_imag + xk_imag * freqs_cis_real,
        ],
        dim=-1,
    ).flatten(3)
    return q_out.type_as(xq).to(xq.device), k_out.type_as(xk).to(xk.device)


class _ReferenceRoPEAttention(nn.Module):
    """Pure torch RoPEAttention mirror with shared projection weights."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        rope_k_repeat: bool = False,
        feat_sizes: tuple[int, int] = (64, 64),
        use_rope_real: bool = False,
        rope_theta: float = 10000.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.kv_in_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.embedding_dim = embedding_dim
        self.use_rope_real = use_rope_real
        self.rope_k_repeat = rope_k_repeat
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)
        self.compute_cis = partial(
            _reference_compute_axial_cis,
            dim=self.internal_dim // self.num_heads,
            theta=rope_theta,
        )
        self.freqs_cis = self.compute_cis(
            end_x=feat_sizes[0], end_y=feat_sizes[1], device=DEVICE
        )
        if self.use_rope_real:
            self.freqs_cis_real = self.freqs_cis.real
            self.freqs_cis_imag = self.freqs_cis.imag

    def _separate_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)

    def _recombine_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        num_k_exclude_rope: int = 0,
    ) -> torch.Tensor:
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        w = h = int(math.sqrt(q.shape[-2]))
        if self.freqs_cis.shape[0] != q.shape[-2]:
            self.freqs_cis = self.compute_cis(end_x=w, end_y=h, device=q.device)
            if self.use_rope_real:
                self.freqs_cis_real = self.freqs_cis.real
                self.freqs_cis_imag = self.freqs_cis.imag

        if q.shape[-2] != k.shape[-2]:
            assert self.rope_k_repeat

        num_k_rope = k.size(-2) - num_k_exclude_rope
        if self.use_rope_real:
            q, k[:, :, :num_k_rope] = _reference_apply_rotary_enc_real(
                q,
                k[:, :, :num_k_rope],
                self.freqs_cis_real,
                self.freqs_cis_imag,
                repeat_freqs_k=self.rope_k_repeat,
            )
        else:
            q, k[:, :, :num_k_rope] = _reference_apply_rotary_enc(
                q,
                k[:, :, :num_k_rope],
                self.freqs_cis,
                repeat_freqs_k=self.rope_k_repeat,
            )

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        out = self._recombine_heads(out)
        return self.out_proj(out)


def _copy_weights(source: RoPEAttention, target: _ReferenceRoPEAttention) -> None:
    target.q_proj.weight.data.copy_(source.q_proj.weight.data)
    target.q_proj.bias.data.copy_(source.q_proj.bias.data)
    target.k_proj.weight.data.copy_(source.k_proj.weight.data)
    target.k_proj.bias.data.copy_(source.k_proj.bias.data)
    target.v_proj.weight.data.copy_(source.v_proj.weight.data)
    target.v_proj.bias.data.copy_(source.v_proj.bias.data)
    target.out_proj.weight.data.copy_(source.out_proj.weight.data)
    target.out_proj.bias.data.copy_(source.out_proj.bias.data)


def _build_reference(
    source: RoPEAttention, *, use_rope_real: bool, rope_k_repeat: bool
) -> _ReferenceRoPEAttention:
    ref = _ReferenceRoPEAttention(
        embedding_dim=source.embedding_dim,
        num_heads=source.num_heads,
        downsample_rate=source.embedding_dim // source.internal_dim,
        rope_k_repeat=rope_k_repeat,
        use_rope_real=use_rope_real,
    )
    _copy_weights(source, ref)
    return ref


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("use_rope_real", [False, True])
def test_rope_attention_matches_torch_reference(
    dtype: torch.dtype, use_rope_real: bool
) -> None:
    torch.manual_seed(0)
    layer = RoPEAttention(
        embedding_dim=64,
        num_heads=4,
        downsample_rate=1,
        use_rope_real=use_rope_real,
        feat_sizes=(8, 8),
        rope_k_repeat=False,
        dropout=0.0,
    ).to(device=DEVICE, dtype=dtype)
    layer.eval()

    ref = _build_reference(layer, use_rope_real=use_rope_real, rope_k_repeat=False).to(
        DEVICE, dtype=dtype
    )
    ref.eval()

    q = torch.randn(2, 64, 64, device=DEVICE, dtype=dtype)
    k = torch.randn(2, 64, 64, device=DEVICE, dtype=dtype)
    v = torch.randn(2, 64, 64, device=DEVICE, dtype=dtype)

    with torch.no_grad():
        out = layer(q, k, v, num_k_exclude_rope=0)
        ref_out = ref(q, k, v, num_k_exclude_rope=0)

    rtol, atol = _tol(dtype)
    torch.testing.assert_close(out, ref_out, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("use_rope_real", [False, True])
def test_rope_attention_supports_longer_k_with_rope_k_repeat(
    dtype: torch.dtype, use_rope_real: bool
) -> None:
    torch.manual_seed(1)
    layer = RoPEAttention(
        embedding_dim=64,
        num_heads=4,
        downsample_rate=1,
        use_rope_real=use_rope_real,
        feat_sizes=(8, 8),
        rope_k_repeat=True,
        dropout=0.0,
    ).to(DEVICE, dtype=dtype)
    layer.eval()

    ref = _build_reference(layer, use_rope_real=use_rope_real, rope_k_repeat=True).to(
        DEVICE, dtype=dtype
    )
    ref.eval()

    q = torch.randn(2, 64, 64, device=DEVICE, dtype=dtype)
    k = torch.randn(2, 128, 64, device=DEVICE, dtype=dtype)
    v = torch.randn(2, 128, 64, device=DEVICE, dtype=dtype)

    with torch.no_grad():
        out = layer(q, k, v)
        ref_out = ref(q, k, v)

    rtol, atol = _tol(dtype)
    torch.testing.assert_close(out, ref_out, rtol=rtol, atol=atol)


def test_rope_attention_rebuilds_freqs_for_different_seq_len() -> None:
    torch.manual_seed(2)
    dtype = torch.float32
    layer = RoPEAttention(
        embedding_dim=64,
        num_heads=4,
        downsample_rate=1,
        use_rope_real=False,
        feat_sizes=(4, 4),
        dropout=0.0,
    ).to(DEVICE, dtype=dtype)
    layer.eval()

    ref = _build_reference(layer, use_rope_real=False, rope_k_repeat=False).to(
        DEVICE, dtype=dtype
    )
    ref.eval()

    q0 = torch.randn(1, 16, 64, device=DEVICE, dtype=dtype)
    k0 = torch.randn(1, 16, 64, device=DEVICE, dtype=dtype)
    v0 = torch.randn(1, 16, 64, device=DEVICE, dtype=dtype)
    q1 = torch.randn(1, 36, 64, device=DEVICE, dtype=dtype)
    k1 = torch.randn(1, 36, 64, device=DEVICE, dtype=dtype)
    v1 = torch.randn(1, 36, 64, device=DEVICE, dtype=dtype)

    with torch.no_grad():
        out0 = layer(q0, k0, v0)
        ref0 = ref(q0, k0, v0)
        out1 = layer(q1, k1, v1)
        ref1 = ref(q1, k1, v1)

    rtol, atol = _tol(dtype)
    torch.testing.assert_close(out0, ref0, rtol=rtol, atol=atol)
    torch.testing.assert_close(out1, ref1, rtol=rtol, atol=atol)
    assert layer.freqs_cis.shape[0] == 36
