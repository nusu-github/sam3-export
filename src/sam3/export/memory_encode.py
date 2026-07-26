"""Cut H -- tensor-only tracker mask-memory encoder."""

from __future__ import annotations

from torch import Tensor, nn


class MemoryEncode(nn.Module):
    """Encode a predicted mask and image feature map into cacheable memory.

    By default masks are logits, matching tracker output.  The sigmoid/scale
    transform is kept here so callers can cache the exact tracker memory while
    still keeping bank selection and object policy outside the graph.
    """

    def __init__(
        self,
        encoder: nn.Module,
        *,
        mask_is_logits: bool = True,
        sigmoid_scale: float = 20.0,
        sigmoid_bias: float = -10.0,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.mask_is_logits = bool(mask_is_logits)
        self.sigmoid_scale = float(sigmoid_scale)
        self.sigmoid_bias = float(sigmoid_bias)

    def forward(self, image_features: Tensor, masks: Tensor) -> tuple[Tensor, Tensor]:
        mask_for_memory = masks.sigmoid() if self.mask_is_logits else masks
        mask_for_memory = mask_for_memory * self.sigmoid_scale + self.sigmoid_bias
        out = self.encoder(image_features, mask_for_memory, skip_mask_sigmoid=True)
        return out["vision_features"], out["vision_pos_enc"][0]


__all__ = ["MemoryEncode"]
