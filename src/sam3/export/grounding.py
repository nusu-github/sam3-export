"""Fixed-tensor open-vocabulary grounding experiment components."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from sam3.grounding.det_decoder import inverse_sigmoid
from sam3.grounding.geometry_encoders import activation_ckpt_wrapper


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


class TextOnlyPromptEncode(nn.Module):
    """Append the official image-conditioned empty-geometry CLS token."""

    def __init__(self, geometry_encoder: nn.Module) -> None:
        super().__init__()
        self.geometry_encoder = geometry_encoder

    def forward(
        self,
        image_feature_low: Tensor,
        image_pos_low: Tensor,
        text_memory: Tensor,
        text_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch = image_feature_low.shape[0]
        feature_seq = image_feature_low.flatten(2).permute(2, 0, 1)
        position_seq = image_pos_low.flatten(2).permute(2, 0, 1)
        cls_embed = self.geometry_encoder.cls_embed
        if cls_embed is None:
            raise RuntimeError("text-only prompt encoding requires geometry CLS")
        geometry = cls_embed.weight.view(1, 1, -1).repeat(1, batch, 1)
        geometry_mask = torch.zeros(
            (batch, 1), device=image_feature_low.device, dtype=torch.bool
        )
        if self.geometry_encoder.final_proj is not None:
            geometry = self.geometry_encoder.norm(
                self.geometry_encoder.final_proj(geometry)
            )
        if self.geometry_encoder.encode is not None:
            for layer in self.geometry_encoder.encode:
                geometry = activation_ckpt_wrapper(layer)(
                    tgt=geometry,
                    memory=feature_seq,
                    tgt_key_padding_mask=geometry_mask,
                    pos=position_seq,
                    act_ckpt_enable=False,
                )
            geometry = self.geometry_encoder.encode_norm(geometry)
        prompt = torch.cat((text_memory.transpose(0, 1), geometry), dim=0)
        prompt_mask = torch.cat((text_mask.to(torch.bool), geometry_mask), dim=1)
        return prompt.transpose(0, 1), prompt_mask


class GroundingEncodeTextOnly(nn.Module):
    """M1 text-only encoder with the official prompt-before-encoder contract."""

    def __init__(
        self, encoder: GroundingEncode, prompt_encoder: TextOnlyPromptEncode
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.prompt_encoder = prompt_encoder
        self.num_feature_levels = encoder.num_feature_levels

    def forward(
        self,
        image_features: tuple[Tensor, ...],
        image_pos: tuple[Tensor, ...],
        image_masks: tuple[Tensor, ...],
        text_memory: Tensor,
        text_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        prompt, prompt_mask = self.prompt_encoder(
            image_features[-1], image_pos[-1], text_memory, text_mask
        )
        encoded = self.encoder(
            image_features,
            image_pos,
            image_masks,
            prompt,
            prompt_mask,
        )
        return (*encoded[:6], prompt, prompt_mask)


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


class GroundingFull(nn.Module):
    """M1 E1 candidate that removes the encoder/decoder deployment cut."""

    def __init__(self, encoder: nn.Module, decoder: GroundingDecode) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        image_features: tuple[Tensor, ...],
        image_pos: tuple[Tensor, ...],
        image_masks: tuple[Tensor, ...],
        text_memory: Tensor,
        text_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        encoded = self.encoder(
            image_features[-self.encoder.num_feature_levels :],
            image_pos,
            image_masks,
            text_memory,
            text_mask,
        )
        if len(encoded) == 8:
            return self.decoder(image_features, *encoded[:7], encoded[7])
        return self.decoder(image_features, *encoded, text_mask)


class GroundingFullFeatureOnly(nn.Module):
    """M1 E2 candidate that owns the required position tensor downstream."""

    def __init__(self, full: GroundingFull, position_encoding: nn.Module) -> None:
        super().__init__()
        self.full = full
        self.position_encoding = position_encoding

    def forward(
        self,
        image_features: tuple[Tensor, ...],
        image_masks: tuple[Tensor, ...],
        text_memory: Tensor,
        text_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        required_pos = self.position_encoding(image_features[-1]).to(
            dtype=image_features[-1].dtype
        )
        return self.full(
            image_features,
            (required_pos,),
            image_masks,
            text_memory,
            text_mask,
        )


class GroundingQueryCore(nn.Module):
    """M1 E3 query stage ending after final all-query interaction."""

    def __init__(self, decoder: nn.Module, dot_product_scoring: nn.Module) -> None:
        super().__init__()
        self.decoder = decoder
        self.dot_product_scoring = dot_product_scoring

    def forward(
        self,
        image_feature_low: Tensor,
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
            memory_key_padding_mask=padding_mask.transpose(0, 1),
            pos=pos_embed,
            reference_boxes=None,
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            feature_size=image_feature_low.shape[-2:],
            tgt_mask=None,
            memory_text=prompt,
            text_attention_mask=text_mask.to(torch.bool),
            apply_dac=False,
        )
        hs_bq = hs.transpose(1, 2)
        refs_bq = reference_boxes.transpose(1, 2)
        logits = self.dot_product_scoring(hs_bq, prompt, text_mask.to(torch.bool))
        boxes = (inverse_sigmoid(refs_bq) + self.decoder.bbox_embed(hs_bq)).sigmoid()
        if presence is None:
            presence_out = logits.new_zeros((batch, logits.shape[2]))
        else:
            presence_out = presence.transpose(1, 2)[-1]
        return logits[-1], boxes[-1], presence_out, hs_bq[-1]


class GroundingMaskSelectedK(nn.Module):
    """M1 E3 mask stage with device-side gather and fixed-K validity."""

    def __init__(self, segmentation_head: nn.Module) -> None:
        super().__init__()
        self.segmentation_head = segmentation_head

    def forward(
        self,
        image_features: tuple[Tensor, ...],
        memory: Tensor,
        text_memory: Tensor,
        text_mask: Tensor,
        query_embeddings: Tensor,
        selected_indices: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor]:
        gather_index = (
            selected_indices.to(torch.int64)
            .unsqueeze(-1)
            .expand(-1, -1, query_embeddings.shape[-1])
        )
        selected_queries = torch.gather(query_embeddings, 1, gather_index)
        image_ids = torch.arange(memory.shape[1], device=memory.device)
        seg_out = self.segmentation_head(
            backbone_feats=list(image_features),
            obj_queries=selected_queries.unsqueeze(0),
            image_ids=image_ids,
            encoder_hidden_states=memory,
            prompt=text_memory.transpose(0, 1),
            prompt_mask=text_mask.to(torch.bool),
        )
        masks = seg_out["pred_masks"]
        return (torch.where(valid_mask.to(torch.bool)[..., None, None], masks, 0.0),)


__all__ = [
    "GroundingDecode",
    "GroundingEncode",
    "GroundingEncodeTextOnly",
    "GroundingFull",
    "GroundingFullFeatureOnly",
    "GroundingMaskSelectedK",
    "GroundingQueryCore",
    "TextOnlyPromptEncode",
]
