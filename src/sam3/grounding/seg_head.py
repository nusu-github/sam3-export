"""MaskFormer-style segmentation head for text grounding.

Port of ``sam3.model.maskformer_segmentation``:

* ``MaskPredictor``
* ``PixelDecoder``
* ``SegmentationHead``
* ``UniversalSegmentationHead``
"""

from __future__ import annotations

import math

from jaxtyping import Bool, Float, Integer
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from ..primitives.mlp import MLP


def _unwrap_feat(x: Float[Tensor, "..."] | object) -> Tensor:
    """Unwrap NestedTensor-like containers to plain tensors if needed."""
    if torch.is_tensor(x):
        return x
    # NestedTensor (official) or duck-typed containers expose ``.tensors``.
    tensors = getattr(x, "tensors", None)
    if tensors is not None:
        return tensors
    return x  # type: ignore[return-value]


class MaskPredictor(nn.Module):
    """Project object queries and einsum against pixel embeddings → masks."""

    def __init__(self, hidden_dim: int, mask_dim: int) -> None:
        super().__init__()
        # model_misc.MLP uses layers.N keys; our MLP matches that layout.
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

    def forward(
        self,
        obj_queries: Float[Tensor, "..."],
        pixel_embed: Float[Tensor, "..."],
    ) -> Float[Tensor, "..."]:
        if len(obj_queries.shape) == 3:
            if pixel_embed.ndim == 3:
                # batch size was omitted on pixel_embed
                mask_preds = torch.einsum(
                    "bqc,chw->bqhw", self.mask_embed(obj_queries), pixel_embed
                )
            else:
                mask_preds = torch.einsum(
                    "bqc,bchw->bqhw", self.mask_embed(obj_queries), pixel_embed
                )
        else:
            # Assumed to have aux masks: (L, B, Q, C)
            if pixel_embed.ndim == 3:
                mask_preds = torch.einsum(
                    "lbqc,chw->lbqhw", self.mask_embed(obj_queries), pixel_embed
                )
            else:
                mask_preds = torch.einsum(
                    "lbqc,bchw->lbqhw", self.mask_embed(obj_queries), pixel_embed
                )

        return mask_preds


class PixelDecoder(nn.Module):
    """Top-down FPN-style upsample with 3×3 conv + GroupNorm + ReLU per stage."""

    def __init__(
        self,
        hidden_dim: int,
        num_upsampling_stages: int,
        interpolation_mode: str = "nearest",
        shared_conv: bool = False,
        compile_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_upsampling_stages = num_upsampling_stages
        self.interpolation_mode = interpolation_mode
        conv_layers: list[nn.Module] = []
        norms: list[nn.Module] = []
        num_convs = 1 if shared_conv else num_upsampling_stages
        for _ in range(num_convs):
            conv_layers.append(nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, 1, 1))
            norms.append(nn.GroupNorm(8, self.hidden_dim))

        self.conv_layers = nn.ModuleList(conv_layers)
        self.norms = nn.ModuleList(norms)
        self.shared_conv = shared_conv
        self.out_dim = self.conv_layers[-1].out_channels
        if compile_mode is not None:
            self.forward = torch.compile(  # type: ignore[method-assign]
                self.forward, mode=compile_mode, dynamic=True, fullgraph=True
            )
            # Needed to make checkpointing happy when compile is on.
            torch._dynamo.config.optimize_ddp = False

    def forward(
        self, backbone_feats: list[Float[Tensor, "..."]]
    ) -> Float[Tensor, "b c h w"]:
        # Assumes backbone features are already projected (C == hidden dim)
        prev_fpn = backbone_feats[-1]
        fpn_feats = backbone_feats[:-1]
        for layer_idx, bb_feat in enumerate(fpn_feats[::-1]):
            curr_fpn = bb_feat
            prev_fpn = curr_fpn + F.interpolate(
                prev_fpn, size=curr_fpn.shape[-2:], mode=self.interpolation_mode
            )
            if self.shared_conv:
                layer_idx = 0
            prev_fpn = self.conv_layers[layer_idx](prev_fpn)
            prev_fpn = F.relu(self.norms[layer_idx](prev_fpn))

        return prev_fpn


class SegmentationHead(nn.Module):
    """Base mask head: pixel decode + MaskPredictor (or 1×1 conv if ``no_dec``)."""

    def __init__(
        self,
        hidden_dim: int,
        upsampling_stages: int,
        use_encoder_inputs: bool = False,
        aux_masks: bool = False,
        no_dec: bool = False,
        pixel_decoder: nn.Module | None = None,
        act_ckpt: bool = False,
        shared_conv: bool = False,
        compile_mode_pixel_decoder: str | None = None,
    ) -> None:
        super().__init__()
        self.use_encoder_inputs = use_encoder_inputs
        self.aux_masks = aux_masks
        if pixel_decoder is not None:
            self.pixel_decoder = pixel_decoder
        else:
            self.pixel_decoder = PixelDecoder(
                hidden_dim,
                upsampling_stages,
                shared_conv=shared_conv,
                compile_mode=compile_mode_pixel_decoder,
            )
        self.no_dec = no_dec
        if no_dec:
            self.mask_predictor = nn.Conv2d(
                hidden_dim, 1, kernel_size=3, stride=1, padding=1
            )
        else:
            self.mask_predictor = MaskPredictor(hidden_dim, mask_dim=hidden_dim)

        self.act_ckpt = act_ckpt
        self.instance_keys = ["pred_masks"]

    @property
    def device(self) -> torch.device:
        cached = getattr(self, "_device", None)
        return cached if cached is not None else next(self.parameters()).device

    def to(self, *args, **kwargs):  # type: ignore[override]
        self._device = None
        return super().to(*args, **kwargs)

    def _embed_pixels(
        self,
        backbone_feats: list[Tensor],
        image_ids: Tensor,
        encoder_hidden_states: Tensor | None,
    ) -> Tensor:
        feature_device = backbone_feats[0].device
        model_device = self.device
        image_ids_ = image_ids.to(feature_device)
        if self.use_encoder_inputs:
            assert encoder_hidden_states is not None
            if _unwrap_feat(backbone_feats[0]).shape[0] > 1:
                backbone_visual_feats = []
                for feat in backbone_feats:
                    backbone_visual_feats.append(
                        _unwrap_feat(feat)[image_ids_, ...].to(model_device)
                    )
            else:
                backbone_visual_feats = [
                    _unwrap_feat(bb_feat).clone() for bb_feat in backbone_feats
                ]
            encoder_hidden_states = encoder_hidden_states.permute(1, 2, 0)
            spatial_dim = math.prod(_unwrap_feat(backbone_feats[-1]).shape[-2:])
            encoder_visual_embed = encoder_hidden_states[..., :spatial_dim].reshape(
                -1, *_unwrap_feat(backbone_feats[-1]).shape[1:]
            )

            backbone_visual_feats[-1] = encoder_visual_embed
            if self.act_ckpt:
                pixel_embed = checkpoint.checkpoint(
                    self.pixel_decoder, backbone_visual_feats, use_reentrant=False
                )
            else:
                pixel_embed = self.pixel_decoder(backbone_visual_feats)
        else:
            backbone_feats_t = [
                _unwrap_feat(x).to(model_device) for x in backbone_feats
            ]
            pixel_embed = self.pixel_decoder(backbone_feats_t)
            if pixel_embed.shape[0] == 1:
                pixel_embed = pixel_embed.squeeze(0)
            else:
                pixel_embed = pixel_embed[image_ids, ...]
        return pixel_embed

    def forward(
        self,
        backbone_feats: list[Float[Tensor, "..."]],
        obj_queries: Float[Tensor, "..."],
        image_ids: Integer[Tensor, "..."] | Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."] | None = None,
        **kwargs,
    ) -> dict[str, Tensor]:
        if self.use_encoder_inputs:
            assert encoder_hidden_states is not None

        pixel_embed = self._embed_pixels(
            backbone_feats=backbone_feats,
            image_ids=image_ids,
            encoder_hidden_states=encoder_hidden_states,
        )

        if self.no_dec:
            mask_pred = self.mask_predictor(pixel_embed)
        elif self.aux_masks:
            mask_pred = self.mask_predictor(obj_queries, pixel_embed)
        else:
            mask_pred = self.mask_predictor(obj_queries[-1], pixel_embed)

        return {"pred_masks": mask_pred}


class UniversalSegmentationHead(SegmentationHead):
    """Semantic + instance segmentation head used by the image text detector."""

    def __init__(
        self,
        hidden_dim: int,
        upsampling_stages: int,
        pixel_decoder: nn.Module,
        aux_masks: bool = False,
        no_dec: bool = False,
        act_ckpt: bool = False,
        presence_head: bool = False,
        dot_product_scorer: nn.Module | None = None,
        cross_attend_prompt: nn.Module | None = None,
    ) -> None:
        super().__init__(
            hidden_dim=hidden_dim,
            upsampling_stages=upsampling_stages,
            use_encoder_inputs=True,
            aux_masks=aux_masks,
            no_dec=no_dec,
            pixel_decoder=pixel_decoder,
            act_ckpt=act_ckpt,
        )
        self.d_model = hidden_dim

        if dot_product_scorer is not None:
            assert presence_head, (
                "Specifying a dot product scorer without a presence head is "
                "likely a mistake"
            )

        self.presence_head: nn.Module | None = None
        if presence_head:
            self.presence_head = (
                dot_product_scorer
                if dot_product_scorer is not None
                else nn.Linear(self.d_model, 1)
            )

        self.cross_attend_prompt = cross_attend_prompt
        if self.cross_attend_prompt is not None:
            self.cross_attn_norm = nn.LayerNorm(self.d_model)

        self.semantic_seg_head = nn.Conv2d(self.pixel_decoder.out_dim, 1, kernel_size=1)
        self.instance_seg_head = nn.Conv2d(
            self.pixel_decoder.out_dim, self.d_model, kernel_size=1
        )

    def forward(
        self,
        backbone_feats: list[Float[Tensor, "..."]],
        obj_queries: Float[Tensor, "..."],
        image_ids: Integer[Tensor, "..."] | Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."] | None = None,
        prompt: Float[Tensor, "..."] | None = None,
        prompt_mask: Bool[Tensor, "..."] | Float[Tensor, "..."] | None = None,
        **kwargs,
    ) -> dict[str, Tensor | None]:
        assert encoder_hidden_states is not None
        bs = encoder_hidden_states.shape[1]

        if self.cross_attend_prompt is not None:
            tgt2 = self.cross_attn_norm(encoder_hidden_states)
            tgt2 = self.cross_attend_prompt(
                query=tgt2,
                key=prompt,
                value=prompt,
                key_padding_mask=prompt_mask,
                need_weights=False,
            )[0]
            encoder_hidden_states = tgt2 + encoder_hidden_states

        presence_logit = None
        if self.presence_head is not None:
            pooled_enc = encoder_hidden_states.mean(0)
            pooled_enc = pooled_enc.view(1, bs, 1, self.d_model)
            if isinstance(self.presence_head, nn.Linear):
                presence_logit = self.presence_head(pooled_enc)
            else:
                presence_logit = self.presence_head(
                    pooled_enc,
                    prompt=prompt,
                    prompt_mask=prompt_mask,
                )
            presence_logit = presence_logit.squeeze(0).squeeze(1)

        pixel_embed = self._embed_pixels(
            backbone_feats=backbone_feats,
            image_ids=image_ids,
            encoder_hidden_states=encoder_hidden_states,
        )

        instance_embeds = self.instance_seg_head(pixel_embed)

        if self.no_dec:
            mask_pred = self.mask_predictor(instance_embeds)
        elif self.aux_masks:
            mask_pred = self.mask_predictor(obj_queries, instance_embeds)
        else:
            mask_pred = self.mask_predictor(obj_queries[-1], instance_embeds)

        return {
            "pred_masks": mask_pred,
            "semantic_seg": self.semantic_seg_head(pixel_embed),
            "presence_logit": presence_logit,
        }
