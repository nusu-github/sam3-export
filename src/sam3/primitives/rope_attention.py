"""Rotary-position attention primitive."""

from __future__ import annotations

from functools import partial
import math

from jaxtyping import Float
import torch
from torch import Tensor

from .attention import Attention
from .rope import apply_rotary_enc, apply_rotary_enc_real, compute_axial_cis


class RoPEAttention(Attention):
    """Attention with rotary position encoding."""

    def __init__(
        self,
        *args,
        rope_theta: float = 10000.0,
        rope_k_repeat: bool = False,
        feat_sizes: tuple[int, int] = (64, 64),
        use_rope_real: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.use_rope_real = use_rope_real
        self.rope_k_repeat = rope_k_repeat
        self.compute_cis = partial(
            compute_axial_cis,
            dim=self.internal_dim // self.num_heads,
            theta=rope_theta,
        )
        device = torch.device("cuda") if torch.cuda.is_available() else None
        self.freqs_cis = self.compute_cis(
            end_x=feat_sizes[0], end_y=feat_sizes[1], device=device
        )
        if self.use_rope_real:
            self.freqs_cis_real = self.freqs_cis.real
            self.freqs_cis_imag = self.freqs_cis.imag

    def forward(
        self,
        q: Float[Tensor, "b n_q c_q"],
        k: Float[Tensor, "b n_k c_k"],
        v: Float[Tensor, "b n_k c_v"],
        num_k_exclude_rope: int = 0,
    ) -> Float[Tensor, "b n_q c_out"]:
        wdtype = self.q_proj.weight.dtype
        if q.dtype != wdtype:
            q = q.to(dtype=wdtype)
        if k.dtype != wdtype:
            k = k.to(dtype=wdtype)
        if v.dtype != wdtype:
            v = v.to(dtype=wdtype)
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
            q, k[:, :, :num_k_rope] = apply_rotary_enc_real(
                q,
                k[:, :, :num_k_rope],
                freqs_cis_real=self.freqs_cis_real,
                freqs_cis_imag=self.freqs_cis_imag,
                repeat_freqs_k=self.rope_k_repeat,
            )
        else:
            q, k[:, :, :num_k_rope] = apply_rotary_enc(
                q,
                k[:, :, :num_k_rope],
                self.freqs_cis,
                repeat_freqs_k=self.rope_k_repeat,
            )

        dropout_p = self.dropout_p if self.training else 0.0
        out = self._attention_core(q, k, v, dropout_p)

        out = self._recombine_heads(out)
        out = self.out_proj(out)
        return out
