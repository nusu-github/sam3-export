"""ViTDet-compatible vision backbone."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Optional

from jaxtyping import Float
from timm.layers import PatchEmbed, trunc_normal_
import torch
from torch import Tensor
import torch.nn as nn

from .vitdet_block import Block
from .vitdet_ops import get_abs_pos


class ViT(nn.Module):
    """ViTDet-compatible backbone (SAM3 production trunk settings).

    Structure can diverge from a pure Detectron2 port as long as checkpoint
    keys under ``trunk.*`` still load (or can be remapped). Core blocks use
    timm ``Mlp`` / ``DropPath`` / ``LayerScale`` and NHWC patch embed.
    """

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 64,
        depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float | int = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float | int = 0.0,
        norm_layer: Callable[..., nn.Module] | str = "LayerNorm",
        act_layer: Callable[..., nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        tile_abs_pos: bool = True,
        window_size: int = 2,
        global_att_blocks: tuple[int, ...] = (),
        use_rope: bool = False,
        use_tiled_rope: bool = False,
        rope_pt_size: Optional[int] = None,
        use_interp_rope: bool = False,
        use_rel_pos_blocks: bool = False,
        rel_pos_zero_init: bool = True,
        pretrain_img_size: int = 224,
        pretrain_use_cls_token: bool = True,
        retain_cls_token: bool = False,
        dropout: float | int = 0.0,
        return_interm_layers: bool = False,
        init_values: Optional[float | int] = None,
        attn_type: str = "vanilla",
        ln_pre: bool = False,
        ln_post: bool = False,
        bias_patch_embed: bool = True,
        use_rope_real: bool = False,
    ) -> None:
        drop_path_rate = float(drop_path_rate)
        dropout = float(dropout)
        mlp_ratio = float(mlp_ratio)
        if not 0.0 <= drop_path_rate <= 1.0:
            raise ValueError("drop_path_rate must be in [0, 1]")
        super().__init__()

        if isinstance(norm_layer, str):
            norm_layer = partial(getattr(nn, norm_layer), eps=1e-5)

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.pretrain_use_cls_token = pretrain_use_cls_token
        self.retain_cls_token = retain_cls_token
        self.use_abs_pos = use_abs_pos
        self.tile_abs_pos = tile_abs_pos
        self.return_interm_layers = return_interm_layers
        self.pretrain_img_size = pretrain_img_size
        self.window_size = window_size
        self.full_attn_ids = (
            list(global_att_blocks) if global_att_blocks else [depth - 1]
        )
        window_block_indexes = [i for i in range(depth) if i not in self.full_attn_ids]

        if self.retain_cls_token:
            self.class_embedding = nn.Parameter(
                trunc_normal_(torch.empty(1, 1, embed_dim), std=0.02)
            )
            if window_size > 0:
                raise ValueError("window attention is not supported with cls token")
        else:
            self.register_parameter("class_embedding", None)

        self.patch_embed = PatchEmbed(
            img_size=None,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            flatten=False,
            output_fmt="NHWC",
            bias=bias_patch_embed,
            strict_img_size=False,
            dynamic_img_pad=False,
        )

        num_patches = (pretrain_img_size // patch_size) ** 2
        num_positions = (num_patches + 1) if pretrain_use_cls_token else num_patches
        if use_abs_pos:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim))
        else:
            self.pos_embed = None

        dpr = [x.item() for x in torch.linspace(0.0, drop_path_rate, depth)]
        grid = (img_size // patch_size, img_size // patch_size)
        self.blocks = nn.ModuleList()
        for i in range(depth):
            blk_window = window_size if i in window_block_indexes else 0
            if rope_pt_size is not None:
                rope_size = (rope_pt_size, rope_pt_size)
            elif blk_window > 0:
                rope_size = (window_size, window_size)
            else:
                rope_size = grid

            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=bool(use_rel_pos_blocks),
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=blk_window,
                input_size=grid if blk_window == 0 else (window_size, window_size),
                use_rope=use_rope,
                rope_pt_size=rope_size,
                rope_tiled=use_tiled_rope,
                rope_interp=use_interp_rope,
                cls_token=self.retain_cls_token,
                dropout=dropout,
                init_values=init_values,
                attn_type=attn_type,
                use_rope_real=use_rope_real,
            )
            self.blocks.append(block)

        self.ln_pre = norm_layer(embed_dim) if ln_pre else nn.Identity()
        self.ln_post = norm_layer(embed_dim) if ln_post else nn.Identity()

        self.channel_list = (
            [embed_dim] * len(self.full_attn_ids)
            if return_interm_layers
            else [embed_dim]
        )

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(
        self, tensor: Float[Tensor, "b c h w"] | object
    ) -> list[Float[Tensor, "b c h w"]]:
        x = tensor
        if hasattr(tensor, "tensors"):
            x = tensor.tensors

        x = self.patch_embed(x)
        h, w = x.shape[1], x.shape[2]

        s = 0
        if self.retain_cls_token:
            x = torch.cat(
                [self.class_embedding, x.reshape(x.shape[0], -1, x.shape[-1])], dim=1
            )
            s = 1

        if self.pos_embed is not None:
            x = x + get_abs_pos(
                self.pos_embed,
                self.pretrain_use_cls_token,
                (h, w),
                self.retain_cls_token,
                tiling=self.tile_abs_pos,
            )

        x = self.ln_pre(x)

        outputs: list[Float[Tensor, "b c h w"]] = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            is_output = (i == self.full_attn_ids[-1]) or (
                self.return_interm_layers and i in self.full_attn_ids
            )
            if is_output:
                if i == self.full_attn_ids[-1]:
                    x = self.ln_post(x)
                feats = x[:, s:]
                if feats.ndim != 4:
                    raise RuntimeError("ViT block output expected NHWC features")
                feats = feats.permute(0, 3, 1, 2)
                outputs.append(feats)

        if not outputs:
            feats = x[:, s:].permute(0, 3, 1, 2)
            outputs.append(feats)
        return outputs

    def get_num_layers(self) -> int:
        return len(self.blocks)
