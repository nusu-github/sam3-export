"""Sine positional encoding shared by SAM3 feature towers."""

from __future__ import annotations

import math
from typing import Optional

from jaxtyping import Float
import torch
from torch import Tensor, nn


class PositionEmbeddingSine(nn.Module):
    """Standard 2D sine positional encoding for feature maps."""

    def __init__(
        self,
        num_pos_feats: int,
        temperature: int = 10000,
        normalize: bool = True,
        scale: Optional[float] = None,
        precompute_resolution: Optional[int] = None,
    ) -> None:
        super().__init__()
        if num_pos_feats % 2 != 0:
            raise ValueError("Expecting even model width")
        self.num_pos_feats = num_pos_feats // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and not normalize:
            raise ValueError("normalize should be True if scale is passed")
        self.scale = 2 * math.pi if scale is None else scale
        self._cache_names: dict[tuple[int, int], str] = {}

        if precompute_resolution is not None:
            # Sizes used by Dual/Tri necks on 1008 input at stride 14.
            precompute_sizes = [
                (int(precompute_resolution // 3.5), int(precompute_resolution // 3.5)),
                (precompute_resolution // 4, precompute_resolution // 4),
                (int(precompute_resolution // 7), int(precompute_resolution // 7)),
                (precompute_resolution // 8, precompute_resolution // 8),
                (int(precompute_resolution // 14), int(precompute_resolution // 14)),
                (precompute_resolution // 16, precompute_resolution // 16),
                (int(precompute_resolution // 28), int(precompute_resolution // 28)),
                (precompute_resolution // 32, precompute_resolution // 32),
            ]
            device = "cuda" if torch.cuda.is_available() else "cpu"
            for size in precompute_sizes:
                if size in self._cache_names:
                    continue
                tensors = torch.zeros((1, 1) + size, device=device)
                cache_name = f"_cache_{size[0]}_{size[1]}"
                self._cache_names[size] = cache_name
                setattr(
                    self,
                    cache_name,
                    nn.Buffer(self._position_encoding(tensors)[0], persistent=False),
                )

    def _encode_xy(
        self,
        x: Float[Tensor, "n"],
        y: Float[Tensor, "n"],
    ) -> tuple[Float[Tensor, "n c"], Float[Tensor, "n c"]]:
        """Encode normalized 1-d coordinates (used by geometry encoder)."""
        assert len(x) == len(y) and x.ndim == y.ndim == 1
        x_embed = x * self.scale
        y_embed = y * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, None] / dim_t
        pos_y = y_embed[:, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2
        ).flatten(1)
        pos_y = torch.stack(
            (pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2
        ).flatten(1)
        return pos_x, pos_y

    @torch.no_grad()
    def encode_boxes(
        self,
        x: Float[Tensor, "n"],
        y: Float[Tensor, "n"],
        w: Float[Tensor, "n"],
        h: Float[Tensor, "n"],
    ) -> Float[Tensor, "n c"]:
        """Box center sine PE + raw (h, w); used by SequenceGeometryEncoder."""
        pos_x, pos_y = self._encode_xy(x, y)
        return torch.cat((pos_y, pos_x, h[:, None], w[:, None]), dim=1)

    @torch.no_grad()
    def _position_encoding(
        self, x: Float[Tensor, "b c h w"]
    ) -> Float[Tensor, "b c_pe h w"]:
        y_embed = (
            torch.arange(1, x.shape[-2] + 1, dtype=torch.float32, device=x.device)
            .view(1, -1, 1)
            .repeat(x.shape[0], 1, x.shape[-1])
        )
        x_embed = (
            torch.arange(1, x.shape[-1] + 1, dtype=torch.float32, device=x.device)
            .view(1, 1, -1)
            .repeat(x.shape[0], x.shape[-2], 1)
        )

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

    @torch.no_grad()
    def forward(self, x: Float[Tensor, "b c h w"]) -> Float[Tensor, "b c_pe h w"]:
        cache_name = self._cache_names.get((x.shape[-2], x.shape[-1]))
        if cache_name is not None:
            return getattr(self, cache_name)[None].repeat(x.shape[0], 1, 1, 1)
        return self._position_encoding(x)
