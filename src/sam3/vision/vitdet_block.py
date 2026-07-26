"""ViTDet transformer blocks."""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

from jaxtyping import Float
from timm.layers import Mlp
import torch
from torch import Tensor
import torch.nn as nn

from .vitdet_attention import Attention
from .vitdet_ops import DropPath, LayerScale, window_partition, window_unpartition


class OfficialInferenceMlp(Mlp):
    """Match the inference-only MLP used by the official SAM3 ViT trunk."""

    def forward(self, x: Tensor) -> Tensor:
        if torch.is_grad_enabled():
            raise ValueError("OfficialInferenceMlp requires gradients to be disabled")
        bias = self.fc1.bias.detach().to(torch.bfloat16)
        weight = self.fc1.weight.detach().to(torch.bfloat16)
        flat = x.to(torch.bfloat16).view(-1, x.shape[-1])
        hidden = torch.ops.aten._addmm_activation(
            bias,
            flat,
            weight.t(),
            beta=1,
            alpha=1,
            use_gelu=True,
        ).view(*x.shape[:-1], weight.shape[0])
        hidden = self.drop1(hidden)
        hidden = self.norm(hidden)
        hidden = hidden.to(self.fc2.weight.dtype)
        hidden = self.fc2(hidden)
        return self.drop2(hidden)


class Block(nn.Module):
    """Transformer block with optional windowed attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float | int = 4.0,
        qkv_bias: bool = True,
        drop_path: float | int = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[tuple[int, int]] = None,
        use_rope: bool = False,
        rope_pt_size: Optional[tuple[int, int]] = None,
        rope_tiled: bool = False,
        rope_interp: bool = False,
        cls_token: bool = False,
        dropout: float | int = 0.0,
        init_values: Optional[float | int] = None,
        attn_type: str = "vanilla",
        use_rope_real: bool = False,
        official_inference_mlp: bool = False,
    ) -> None:
        if dim <= 0:
            raise ValueError("dim must be > 0")
        if num_heads <= 0:
            raise ValueError("num_heads must be > 0")
        dropout = float(dropout)
        drop_path = float(drop_path)
        mlp_ratio = float(mlp_ratio)
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
            use_rope=use_rope,
            rope_pt_size=rope_pt_size,
            rope_tiled=rope_tiled,
            rope_interp=rope_interp,
            cls_token=cls_token,
            use_rope_real=use_rope_real,
            attn_type=attn_type,
        )
        init_vals = float(init_values) if init_values is not None else None
        self.ls1 = (
            LayerScale(dim, init_values=init_vals) if init_vals else nn.Identity()
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_type = OfficialInferenceMlp if official_inference_mlp else Mlp
        self.mlp = mlp_type(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=(dropout, 0.0),
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_vals) if init_vals else nn.Identity()
        )
        self.dropout = nn.Dropout(dropout)
        self.window_size = int(window_size)

    def forward(self, x: Float[Tensor, "b h w c"]) -> Float[Tensor, "b h w c"]:
        if x.ndim != 4:
            raise ValueError("Block expects NHWC input: (B, H, W, C)")
        shortcut = x
        x = self.norm1(x)

        pad_hw = None
        if self.window_size > 0:
            h, w = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.ls1(self.attn(x))

        if self.window_size > 0:
            if pad_hw is None:
                raise RuntimeError("window_partition did not return padding metadata")
            x = window_unpartition(x, self.window_size, pad_hw, (h, w))

        x = shortcut + self.dropout(self.drop_path(x))
        x = x + self.dropout(self.drop_path(self.ls2(self.mlp(self.norm2(x)))))
        return x
