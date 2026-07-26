"""Mask-memory encoder stack for SAM3 video tracking.

Port of ``sam3.model.memory``:
  - ``SimpleMaskDownSampler``
  - ``CXBlock`` (ConvNeXt-style residual)
  - ``SimpleFuser``
  - ``SimpleMaskEncoder``

Uses timm normalization and ConvNeXt blocks so checkpoint keys match official
``tracker.maskmem_backbone.*`` after the loader's key remap.
"""

from __future__ import annotations

from collections.abc import Sequence
import copy
import math
from typing import Any

from jaxtyping import Float
from timm.layers import LayerNorm2d
from timm.models.convnext import ConvNeXtBlock
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


def get_clones(module: nn.Module, n: int) -> nn.ModuleList:
    """Deep-copy ``module`` ``n`` times into a ModuleList (matches model_misc)."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class SimpleMaskDownSampler(nn.Module):
    """
    Progressively downsample a mask by total_stride, each time by stride.
    Note that LayerNorm is applied per *token*, like in ViT.

    With each downsample (by a factor stride**2), channel capacity increases by
    the same factor. In the end, we linearly project to embed_dim channels.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        kernel_size: int = 4,
        stride: int = 4,
        padding: int = 0,
        total_stride: int = 16,
        activation: type[nn.Module] = nn.GELU,
        # Option to interpolate the input mask first before downsampling using
        # convs. In that case, the total_stride is assumed to be after
        # interpolation. If set to input resolution or None, we don't
        # interpolate. We default to None to be safe (for older configs or if
        # not explicitly set).
        interpol_size: list[int] | tuple[int, int] | None = None,
        # options for incorporating multiplex memory encoding
        multiplex_count: int = 1,
        starting_out_chan: int = 1,
        input_channel_multiplier: int = 1,
    ) -> None:
        super().__init__()
        num_layers = int(math.log2(total_stride) // math.log2(stride))
        multiplex_count = multiplex_count * input_channel_multiplier
        assert stride**num_layers == total_stride
        self.encoder = nn.Sequential()
        mask_in_chans, mask_out_chans = multiplex_count, starting_out_chan
        for _ in range(num_layers):
            mask_out_chans = mask_out_chans * (stride**2)
            self.encoder.append(
                nn.Conv2d(
                    mask_in_chans,
                    mask_out_chans,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
            self.encoder.append(LayerNorm2d(mask_out_chans))
            self.encoder.append(activation())
            mask_in_chans = mask_out_chans

        self.encoder.append(nn.Conv2d(mask_out_chans, embed_dim, kernel_size=1))
        self.multiplex_count = multiplex_count
        self.interpol_size = interpol_size
        if self.interpol_size is not None:
            assert isinstance(self.interpol_size, (list, tuple)), (
                f"Unsupported type {type(self.interpol_size)}. Should be a list or tuple."
            )
            self.interpol_size = list(interpol_size)
            assert len(self.interpol_size) == 2

    def forward(
        self, x: Float[Tensor, "n c h w"]
    ) -> Float[Tensor, "n out_c out_h out_w"]:
        if self.interpol_size is not None and self.interpol_size != list(x.shape[-2:]):
            # bilinear+antialias prefers fp32; restore weight dtype for Conv/LN
            wdtype = next(self.encoder.parameters()).dtype
            x = F.interpolate(
                x.float(),
                size=self.interpol_size,
                align_corners=False,
                mode="bilinear",
                antialias=True,
            ).to(dtype=wdtype)
        return self.encoder(x)


class SimpleFuser(nn.Module):
    def __init__(
        self,
        layer: nn.Module,
        num_layers: int,
        dim: int | None = None,
        input_projection: bool = False,
    ) -> None:
        super().__init__()
        self.proj = nn.Identity()
        self.layers = get_clones(layer, num_layers)

        if input_projection:
            assert dim is not None
            self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x: Float[Tensor, "n c h w"]) -> Float[Tensor, "n c h w"]:
        # normally x: (N, C, H, W)
        x = self.proj(x)
        for layer in self.layers:
            x = layer(x)
        return x


class SimpleMaskEncoder(nn.Module):
    def __init__(
        self,
        out_dim: int,
        mask_downsampler: nn.Module,
        fuser: nn.Module,
        position_encoding: nn.Module,
        in_dim: int = 256,  # in_dim of pix_feats
    ) -> None:
        super().__init__()

        self.mask_downsampler = mask_downsampler

        self.pix_feat_proj = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.fuser = fuser
        self.position_encoding = position_encoding
        self.out_proj = nn.Identity()
        if out_dim != in_dim:
            self.out_proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)

    def forward(
        self,
        pix_feat: Float[Tensor, "n c h w"],
        masks: Float[Tensor, "n c_mask h_mask w_mask"],
        skip_mask_sigmoid: bool = False,
    ) -> dict[str, Any]:
        ## Process masks
        # sigmoid, so that less domain shift from gt masks which are bool
        if not skip_mask_sigmoid:
            masks = F.sigmoid(masks)
        masks = self.mask_downsampler(masks)

        ## Fuse pix_feats and downsampled masks
        # in case the visual features are on CPU, cast them to CUDA
        pix_feat = pix_feat.to(masks.device)

        x = self.pix_feat_proj(pix_feat)
        x = x + masks
        x = self.fuser(x)
        x = self.out_proj(x)

        pos = self.position_encoding(x).to(x.dtype)

        return {"vision_features": x, "vision_pos_enc": [pos]}


def create_maskmem_backbone(
    out_dim: int = 64,
    in_dim: int = 256,
    fuser_layers: int = 2,
    pe_dim: int = 64,
    interpol_size: Sequence[int] | None = (1152, 1152),
    precompute_resolution: int | None = 1008,
) -> SimpleMaskEncoder:
    """Factory matching ``model_builder._create_tracker_maskmem_backbone``.

    Production: out_dim=64, CXBlock dim=256, 2 fuser layers, PE 64-d,
    interpol 1152, image 1008 → feat 72×72 after total_stride=16.
    """
    from ..primitives.position_encoding import PositionEmbeddingSine

    position_encoding = PositionEmbeddingSine(
        num_pos_feats=pe_dim,
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=precompute_resolution,
    )
    mask_downsampler = SimpleMaskDownSampler(
        embed_dim=in_dim,
        kernel_size=3,
        stride=2,
        padding=1,
        total_stride=16,
        interpol_size=list(interpol_size) if interpol_size is not None else None,
    )
    fuser = SimpleFuser(
        layer=ConvNeXtBlock(
            in_chs=in_dim,
            out_chs=in_dim,
            kernel_size=7,
            ls_init_value=1.0e-06,
        ),
        num_layers=fuser_layers,
    )
    return SimpleMaskEncoder(
        out_dim=out_dim,
        position_encoding=position_encoding,
        mask_downsampler=mask_downsampler,
        fuser=fuser,
        in_dim=in_dim,
    )


__all__ = [
    "SimpleMaskDownSampler",
    "SimpleFuser",
    "SimpleMaskEncoder",
    "create_maskmem_backbone",
    "get_clones",
]
