"""ViTDet attention with 2D RoPE and relative-position bias."""

from __future__ import annotations

from functools import partial
import math
from typing import Optional

from jaxtyping import Float
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from ..primitives.rope import apply_rotary_enc_real, compute_axial_cis
from .vitdet_ops import concat_rel_pos


class Attention(nn.Module):
    """Multi-head Attention block with relative position and 2D RoPE support."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[tuple[int, int]] = None,
        attn_type: object = "Vanilla",
        cls_token: bool = False,
        use_rope: bool = False,
        rope_theta: float | int = 10000.0,
        rope_pt_size: Optional[tuple[int, int]] = None,
        rope_interp: bool = False,
        rope_tiled: bool = False,
        use_rope_real: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        if self.head_dim * num_heads != dim:
            raise ValueError("dim must be divisible by num_heads")
        self.scale = self.head_dim**-0.5
        self.cls_token = cls_token
        self.attn_type = attn_type
        attn_name = str(attn_type)
        if attn_name.lower() not in {"attentiontype.vanilla", "vanilla"}:
            raise NotImplementedError("only Vanilla attention is supported")

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        self.input_size = input_size

        self.use_rope = use_rope
        self.rope_theta = float(rope_theta)
        self.rope_pt_size = rope_pt_size
        self.rope_interp = rope_interp
        self.rope_tiled = rope_tiled
        self.use_rope_real = use_rope_real

        self._setup_rel_pos(rel_pos_zero_init)
        self._setup_rope_freqs()

    def _setup_rel_pos(self, rel_pos_zero_init: bool = True) -> None:
        if not self.use_rel_pos:
            self.rel_pos_h = None
            self.rel_pos_w = None
            return

        assert self.input_size is not None
        assert self.cls_token is False, "not supported"
        self.rel_pos_h = nn.Parameter(
            torch.zeros(2 * self.input_size[0] - 1, self.head_dim)
        )
        self.rel_pos_w = nn.Parameter(
            torch.zeros(2 * self.input_size[1] - 1, self.head_dim)
        )
        if not rel_pos_zero_init:
            nn.init.trunc_normal_(self.rel_pos_h, std=0.02)
            nn.init.trunc_normal_(self.rel_pos_w, std=0.02)

        h, w = self.input_size
        q_coords = torch.arange(h)[:, None]
        k_coords = torch.arange(w)[None, :]
        relative_coords = (q_coords - k_coords) + (h - 1)
        self.register_buffer("relative_coords", relative_coords.long())

    def _setup_rope_freqs(self) -> None:
        if not self.use_rope:
            self.freqs_cis_real = None
            self.freqs_cis_imag = None
            return

        assert self.input_size is not None
        if self.rope_pt_size is None:
            self.rope_pt_size = self.input_size

        compute_cis = partial(
            compute_axial_cis,
            dim=self.head_dim,
            theta=self.rope_theta,
        )

        if self.rope_pt_size != self.input_size and self.rope_tiled:
            assert not self.rope_interp, "cannot both tile and interpolate rope"
            freqs_cis = compute_cis(
                end_x=self.rope_pt_size[0], end_y=self.rope_pt_size[1]
            )
            rh = self.input_size[0] // self.rope_pt_size[0]
            rw = self.input_size[1] // self.rope_pt_size[1]
            assert rh >= 1
            assert rw >= 1
            assert (
                self.input_size[0] % self.rope_pt_size[0] == 0
                and self.input_size[1] % self.rope_pt_size[1] == 0
            )
            freqs_cis = (
                freqs_cis.reshape(self.rope_pt_size[0], self.rope_pt_size[1], -1)
                .tile(rh, rw, 1)
                .reshape(-1, freqs_cis.shape[-1])
            )
        else:
            scale_pos = 1.0
            if self.rope_interp:
                scale_pos = self.rope_pt_size[0] / self.input_size[0]
            freqs_cis = compute_cis(
                end_x=self.input_size[0],
                end_y=self.input_size[1],
                scale_pos=scale_pos,
            )

        if self.cls_token:
            t = torch.zeros(
                self.head_dim // 2, dtype=torch.float32, device=freqs_cis.device
            )
            cls_freqs_cis = torch.polar(torch.ones_like(t), t)[None, :]
            freqs_cis = torch.cat([cls_freqs_cis, freqs_cis], dim=0)

        # Keep RoPE tables in float32; casting them with model.to(bf16/fp16)
        # loses angle precision and diverges from official complex64 tables.
        self.register_buffer("freqs_cis_real", freqs_cis.real.float().contiguous())
        self.register_buffer("freqs_cis_imag", freqs_cis.imag.float().contiguous())

    def _apply_rope(
        self,
        q: Float[Tensor, "b h n d"],
        k: Float[Tensor, "b h n d"],
    ) -> tuple[Float[Tensor, "b h n d"], Float[Tensor, "b h n d"]]:
        if not self.use_rope:
            return q, k

        assert self.freqs_cis_real is not None
        assert self.freqs_cis_imag is not None
        # Ensure float32 tables even after module.to(low_precision).
        freqs_r = self.freqs_cis_real
        freqs_i = self.freqs_cis_imag
        if freqs_r.dtype != torch.float32:
            freqs_r = freqs_r.float()
            freqs_i = freqs_i.float()
        return apply_rotary_enc_real(
            q,
            k,
            freqs_cis_real=freqs_r,
            freqs_cis_imag=freqs_i,
            repeat_freqs_k=False,
        )

    def forward(
        self, x: Float[Tensor, "b h w c"] | Float[Tensor, "b l c"]
    ) -> Float[Tensor, "b h w c"] | Float[Tensor, "b l c"]:
        s = 1 if self.cls_token else 0
        if x.ndim == 4:
            batch, h, w, _ = x.shape
            assert s == 0
            token_count = h * w
            ndim = 4
        elif x.ndim == 3:
            batch, token_count, _ = x.shape
            h = w = int(math.sqrt(token_count - s))
            assert h * w == token_count - s
            ndim = 3
        else:
            raise ValueError(f"Unsupported input rank {x.ndim}; expected 3 or 4")

        qkv = self.qkv(x).reshape(batch, token_count, 3, self.num_heads, -1)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        q, k = self._apply_rope(q, k)

        if self.use_rel_pos:
            assert self.rel_pos_h is not None and self.rel_pos_w is not None
            q, k = concat_rel_pos(
                q.flatten(0, 1),
                k.flatten(0, 1),
                (h, w),
                (h, w),
                self.rel_pos_h,
                self.rel_pos_w,
                rescale=True,
                relative_coords=self.relative_coords,
            )
            q = q.reshape(batch, self.num_heads, h * w, -1)
            k = k.reshape(batch, self.num_heads, h * w, -1)

        x = F.scaled_dot_product_attention(q, k, v, scale=self.scale, dropout_p=0.0)

        if ndim == 4:
            x = (
                x.view(batch, self.num_heads, h, w, -1)
                .permute(0, 2, 3, 1, 4)
                .reshape(batch, h, w, -1)
            )
        else:
            x = (
                x.view(batch, self.num_heads, token_count, -1)
                .permute(0, 2, 1, 3)
                .reshape(batch, token_count, -1)
            )

        return self.proj(x)
