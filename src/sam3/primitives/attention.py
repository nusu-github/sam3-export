"""Attention primitive with ``nn.Linear`` projections and an SDPA core.

Source of truth: ``sam3.sam.transformer.Attention`` (API shape).

Core policy
-----------
Always use ``torch.nn.functional.scaled_dot_product_attention`` (Flash /
Efficient / Math backends chosen by PyTorch for the device).

* **Eval**: ``dropout_p=0`` — deterministic fused kernels when available.
* **Train**: pass ``dropout_p`` through SDPA (same as official SAM3).

Projections are plain ``nn.Linear`` so the default path stays
``torch.export``-friendly (cuBLAS / ATen).
"""

from __future__ import annotations

import math

from jaxtyping import Float
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """Attention with optional internal-dim downsampling after projection."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        dropout: float = 0.0,
        kv_in_dim: int | None = None,
    ) -> None:
        super().__init__()
        if num_heads <= 0:
            raise ValueError("num_heads must be > 0")
        if downsample_rate <= 0:
            raise ValueError("downsample_rate must be >= 1")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.embedding_dim = embedding_dim
        self.kv_in_dim = kv_in_dim if kv_in_dim is not None else embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        self.dropout_p = dropout

        if self.internal_dim <= 0:
            raise ValueError("downsample_rate is too large for embedding_dim")
        if self.internal_dim % num_heads != 0:
            raise ValueError("num_heads must divide internal dimension")
        if self.kv_in_dim <= 0:
            raise ValueError("kv_in_dim must be > 0")

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.v_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(
        self, x: Float[Tensor, "b n c"], num_heads: int
    ) -> Float[Tensor, "b h n d"]:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x heads x tokens x dim

    def _recombine_heads(self, x: Float[Tensor, "b h n d"]) -> Float[Tensor, "b n c"]:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)

    def _attention_core(
        self,
        q: Float[Tensor, "b h lq d"],
        k: Float[Tensor, "b h lk d"],
        v: Float[Tensor, "b h lk d"],
        dropout_p: float | int,
        attn_mask: Tensor | None = None,
    ) -> Float[Tensor, "b h lq d"]:
        scale = 1.0 / math.sqrt(q.shape[-1])
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, scale=scale, dropout_p=float(dropout_p)
        )

    def forward(
        self,
        q: Float[Tensor, "b n_q c_q"],
        k: Float[Tensor, "b n_k c_k"],
        v: Float[Tensor, "b n_k c_v"],
    ) -> Float[Tensor, "b n_q c_out"]:
        # Permanent bf16/fp16 weights + float PE (or autocast edges) otherwise
        # fail with "mat1 and mat2 must have the same dtype".
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

        dropout_p = self.dropout_p if self.training else 0.0
        out = self._attention_core(q, k, v, dropout_p)

        out = self._recombine_heads(out)
        out = self.out_proj(out)
        return out
