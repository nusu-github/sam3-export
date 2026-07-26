"""Cuts F/G -- fixed-tensor open-vocabulary grounding encoder and decoder."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from sam3.grounding.det_decoder import inverse_sigmoid


class GroundingEncode(nn.Module):
    """Fuse fixed FPN tensors and batch-first text memory into DETR memory.

    Feature tuples are part of the static export signature. ``image_masks`` and
    ``text_mask`` use PyTorch's key-padding convention (``True`` = ignored).
    """

    def __init__(self, encoder: nn.Module, num_feature_levels: int) -> None:
        super().__init__()
        if num_feature_levels <= 0:
            raise ValueError("num_feature_levels must be positive")
        self.encoder = encoder
        self.num_feature_levels = int(num_feature_levels)

    def forward(
        self,
        image_features: tuple[Tensor, ...],
        image_pos: tuple[Tensor, ...],
        image_masks: tuple[Tensor, ...],
        text_memory: Tensor,
        text_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        if len(image_features) != self.num_feature_levels:
            raise ValueError("image_features has an unexpected number of levels")
        if len(image_pos) != self.num_feature_levels:
            raise ValueError("image_pos has an unexpected number of levels")
        if len(image_masks) != self.num_feature_levels:
            raise ValueError("image_masks has an unexpected number of levels")
        if text_memory.ndim != 3 or text_mask.shape != text_memory.shape[:2]:
            raise ValueError("text inputs must be [B,L,D] and [B,L]")

        prompt = text_memory.transpose(0, 1)
        out = self.encoder(
            src=list(image_features),
            src_key_padding_mask=list(image_masks),
            src_pos=list(image_pos),
            prompt=prompt,
            prompt_pos=prompt.new_zeros(prompt.shape),
            prompt_key_padding_mask=text_mask.to(torch.bool),
            feat_sizes=None,
            encoder_extra_kwargs=None,
        )
        padding = out["padding_mask"]
        if padding is None:
            # The public export contract always returns a tensor.  This branch is
            # retained for generic TransformerEncoderFusion construction.
            padding = torch.zeros(
                text_memory.shape[0],
                out["memory"].shape[0],
                device=text_memory.device,
                dtype=torch.bool,
            )
        else:
            # TransformerEncoder keeps a legacy sequence-first return for this
            # value.  The decoder / exported public contract use the standard
            # PyTorch key-padding layout [B, S].
            padding = padding.transpose(0, 1)
        return (
            out["memory"],
            out["pos_embed"],
            padding,
            out["level_start_index"],
            out["spatial_shapes"],
            out["valid_ratios"],
            out["memory_text"].transpose(0, 1),
        )


class GroundingDecode(nn.Module):
    """Decode fused memory to fixed query boxes, scores, masks, and presence."""

    def __init__(
        self,
        decoder: nn.Module,
        dot_product_scoring: nn.Module,
        segmentation_head: nn.Module,
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self.dot_product_scoring = dot_product_scoring
        self.segmentation_head = segmentation_head

    def forward(
        self,
        image_features: tuple[Tensor, ...],
        memory: Tensor,
        pos_embed: Tensor,
        padding_mask: Tensor,
        level_start_index: Tensor,
        spatial_shapes: Tensor,
        valid_ratios: Tensor,
        text_memory: Tensor,
        text_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch = memory.shape[1]
        prompt = text_memory.transpose(0, 1)
        tgt = self.decoder.query_embed.weight.unsqueeze(1).expand(-1, batch, -1)
        hs, reference_boxes, presence, _presence_features = self.decoder(
            tgt=tgt,
            memory=memory,
            # The decoder implementation retains the reference code's
            # sequence-first internal convention; public inputs are [B, S].
            memory_key_padding_mask=padding_mask.transpose(0, 1),
            pos=pos_embed,
            reference_boxes=None,
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            feature_size=image_features[-1].shape[-2:],
            tgt_mask=None,
            memory_text=prompt,
            text_attention_mask=text_mask.to(torch.bool),
            apply_dac=False,
        )
        hs_bq = hs.transpose(1, 2)
        refs_bq = reference_boxes.transpose(1, 2)
        logits = self.dot_product_scoring(hs_bq, prompt, text_mask.to(torch.bool))
        boxes = (inverse_sigmoid(refs_bq) + self.decoder.bbox_embed(hs_bq)).sigmoid()

        image_ids = torch.arange(batch, device=memory.device)
        seg_out = self.segmentation_head(
            backbone_feats=list(image_features),
            obj_queries=hs_bq,
            image_ids=image_ids,
            encoder_hidden_states=memory,
            prompt=prompt,
            prompt_mask=text_mask.to(torch.bool),
        )
        masks = seg_out["pred_masks"]
        if presence is None:
            presence_out = logits.new_zeros((batch, logits.shape[2]))
        else:
            presence_out = presence.transpose(1, 2)[-1]
        return logits[-1], boxes[-1], masks, presence_out


__all__ = ["GroundingEncode", "GroundingDecode"]
