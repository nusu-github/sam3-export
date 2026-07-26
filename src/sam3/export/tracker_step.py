"""Cut I -- one fixed-shape tracker step with an externally managed memory bank."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class TrackerStep(nn.Module):
    """Run memory attention and the SAM mask head for one object/frame.

    The caller supplies padded spatial-memory and object-pointer slots.  Slot
    selection, temporal-position construction, appending new memory, and the
    video/object loops remain L3 runtime responsibilities.  Use
    :class:`MemoryEncode` on the returned high-resolution mask to append a slot.
    """

    def __init__(self, tracker: nn.Module, *, multimask_output: bool = False) -> None:
        super().__init__()
        required = ("transformer", "_forward_sam_heads")
        if not all(hasattr(tracker, name) for name in required):
            raise TypeError("tracker must be a SAM3 tracker core")
        self.tracker = tracker
        self.multimask_output = bool(multimask_output)

    def forward(
        self,
        image_embeddings: Tensor,
        image_pos: Tensor,
        high_res_0: Tensor,
        high_res_1: Tensor,
        memory_features: Tensor,
        memory_pos: Tensor,
        memory_padding_mask: Tensor,
        object_memory: Tensor,
        object_memory_pos: Tensor,
        object_padding_mask: Tensor,
        point_coords: Tensor,
        point_labels: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        # [B,M,Cm,H,W] -> [M*H*W,B,Cm].  Memory positions already include the
        # runtime-selected temporal embedding, keeping scheduling out of export.
        b, slots, _mem_dim, mh, mw = memory_features.shape
        memory_tokens = memory_features.permute(1, 3, 4, 0, 2).reshape(
            slots * mh * mw, b, -1
        )
        memory_pos_tokens = memory_pos.permute(1, 3, 4, 0, 2).reshape(
            slots * mh * mw, b, -1
        )
        memory_mask = memory_padding_mask.repeat_interleave(mh * mw, dim=1)

        object_tokens = object_memory.permute(1, 0, 2)
        object_pos_tokens = object_memory_pos.permute(1, 0, 2)
        prompt = torch.cat((memory_tokens, object_tokens), dim=0)
        prompt_pos = torch.cat((memory_pos_tokens, object_pos_tokens), dim=0)
        prompt_mask = torch.cat(
            (memory_mask, object_padding_mask.to(torch.bool)), dim=1
        )

        src = image_embeddings.flatten(2).permute(2, 0, 1)
        src_pos = image_pos.flatten(2).permute(2, 0, 1)
        encoder_out = self.tracker.transformer.encoder(
            src=[src],
            src_key_padding_mask=[None],
            src_pos=[src_pos],
            prompt=prompt,
            prompt_pos=prompt_pos,
            prompt_key_padding_mask=prompt_mask,
            feat_sizes=[image_embeddings.shape[-2:]],
            num_obj_ptr_tokens=object_memory.shape[1],
        )
        pix_feat = encoder_out["memory"].permute(1, 2, 0).reshape_as(image_embeddings)
        outputs = self.tracker._forward_sam_heads(
            backbone_features=pix_feat,
            point_inputs={"point_coords": point_coords, "point_labels": point_labels},
            high_res_features=[high_res_0, high_res_1],
            multimask_output=self.multimask_output,
        )
        _multi_low, _multi_high, _ious, low, high, obj_ptr, obj_score = outputs
        return low, high, obj_ptr, obj_score


__all__ = ["TrackerStep"]
