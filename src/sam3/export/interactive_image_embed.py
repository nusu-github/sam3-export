"""Cut C' -- SAM2 FPN view for the interactive / tracker mask head."""

from __future__ import annotations

from torch import Tensor, nn


class InteractiveImageEmbed(nn.Module):
    """Turn cached SAM2 FPN levels into SAM mask-head inputs.

    ``sam2_fpn_0..3`` are the four tensors emitted by :class:`VisionTowerFlat`.
    The final (coarsest) level is the scalp level and is intentionally omitted,
    matching ``SAM3VLBackbone(..., scalp=1)``.  Temporal scheduling is outside
    this cut; the learned no-memory embedding represents an initial frame.
    """

    def __init__(self, tracker: nn.Module) -> None:
        super().__init__()
        required = ("sam_mask_decoder", "no_mem_embed")
        if not all(hasattr(tracker, name) for name in required):
            raise TypeError("tracker must expose sam_mask_decoder and no_mem_embed")
        decoder = tracker.sam_mask_decoder
        if not hasattr(decoder, "conv_s0") or not hasattr(decoder, "conv_s1"):
            raise TypeError("tracker SAM mask decoder must use high-res features")
        self.conv_s0 = decoder.conv_s0
        self.conv_s1 = decoder.conv_s1
        self.no_mem_embed = tracker.no_mem_embed

    def forward(
        self,
        sam2_fpn_0: Tensor,
        sam2_fpn_1: Tensor,
        sam2_fpn_2: Tensor,
        sam2_fpn_3: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        # Keep the fourth input explicit: this makes the VisionTowerFlat wiring
        # stable and documents that the scalp is not consumed by this view.
        del sam2_fpn_3
        image_embed = sam2_fpn_2 + self.no_mem_embed.permute(0, 2, 1).unsqueeze(-1)
        return image_embed, self.conv_s0(sam2_fpn_0), self.conv_s1(sam2_fpn_1)


__all__ = ["InteractiveImageEmbed"]
