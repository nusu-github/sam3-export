"""ViTDet FPN necks bridging the trunk to SAM3 feature consumers."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy

from jaxtyping import Float
from torch import Tensor
import torch.nn as nn


class Sam3DualViTDetNeck(nn.Module):
    """SimpleFPN neck a la ViTDet (SAM3 dual neck: sam3 + optional sam2).

    Source: ``sam3.model.necks.Sam3DualViTDetNeck``.
    """

    def __init__(
        self,
        trunk: nn.Module,
        position_encoding: nn.Module,
        d_model: int = 256,
        scale_factors: Sequence[float | int] = (4.0, 2.0, 1.0, 0.5),
        add_sam2_neck: bool = False,
    ) -> None:
        super().__init__()
        self.trunk = trunk
        self.position_encoding = position_encoding
        self.scale_factors = tuple(scale_factors)
        self.convs = nn.ModuleList()

        # Prefer explicit channel_list; fall back to embed_dim.
        if hasattr(trunk, "channel_list") and trunk.channel_list:
            dim = int(trunk.channel_list[-1])
        elif hasattr(trunk, "embed_dim"):
            dim = int(trunk.embed_dim)
        else:
            raise ValueError("trunk must expose channel_list or embed_dim")

        use_bias = True
        for scale in self.scale_factors:
            current = nn.Sequential()
            if scale == 4.0:
                current.add_module(
                    "dconv_2x2_0",
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                )
                current.add_module("gelu", nn.GELU())
                current.add_module(
                    "dconv_2x2_1",
                    nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
                )
                out_dim = dim // 4
            elif scale == 2.0:
                current.add_module(
                    "dconv_2x2",
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                )
                out_dim = dim // 2
            elif scale == 1.0:
                out_dim = dim
            elif scale == 0.5:
                current.add_module("maxpool_2x2", nn.MaxPool2d(kernel_size=2, stride=2))
                out_dim = dim
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported yet.")

            current.add_module(
                "conv_1x1",
                nn.Conv2d(out_dim, d_model, kernel_size=1, bias=use_bias),
            )
            current.add_module(
                "conv_3x3",
                nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, bias=use_bias),
            )
            self.convs.append(current)

        self.sam2_convs: nn.ModuleList | None = None
        if add_sam2_neck:
            self.sam2_convs = deepcopy(self.convs)

    def forward(
        self, tensor_list: Float[Tensor, "..."] | list[Float[Tensor, "..."]]
    ) -> tuple[
        list[Float[Tensor, "..."]],
        list[Float[Tensor, "..."]],
        list[Float[Tensor, "..."]] | None,
        list[Float[Tensor, "..."]] | None,
    ]:
        xs = self.trunk(tensor_list)
        if not isinstance(xs, (list, tuple)):
            xs = [xs]
        # Handle NestedTensor-like objects
        x = xs[-1]
        x = getattr(x, "tensors", x)

        sam3_out: list[Tensor] = []
        sam3_pos: list[Tensor] = []
        sam2_out: list[Tensor] | None = [] if self.sam2_convs is not None else None
        sam2_pos: list[Tensor] | None = [] if self.sam2_convs is not None else None

        for i, conv in enumerate(self.convs):
            sam3_x = conv(x)
            sam3_p = self.position_encoding(sam3_x).to(dtype=sam3_x.dtype)
            sam3_out.append(sam3_x)
            sam3_pos.append(sam3_p)
            if (
                self.sam2_convs is not None
                and sam2_out is not None
                and sam2_pos is not None
            ):
                sam2_x = self.sam2_convs[i](x)
                sam2_p = self.position_encoding(sam2_x).to(dtype=sam2_x.dtype)
                sam2_out.append(sam2_x)
                sam2_pos.append(sam2_p)

        return sam3_out, sam3_pos, sam2_out, sam2_pos
