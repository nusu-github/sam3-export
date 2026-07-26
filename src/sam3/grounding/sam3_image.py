"""SAM3 image detector for text and geometry grounding.

Port of the eval path used by ``Sam3Processor.set_text_prompt`` from
``sam3.model.sam3_image.Sam3Image``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from jaxtyping import Bool, Float, Integer
import torch
from torch import Tensor
import torch.nn as nn
from torchvision.ops import box_convert

from sam3.dtype_policy import module_param_dtype

from .det_decoder import inverse_sigmoid
from .geometry_encoders import Prompt


@dataclass
class FindStage:
    """Minimal find-stage indices for single-image text grounding."""

    img_ids: Integer[Tensor, "..."]
    text_ids: Integer[Tensor, "..."]
    input_boxes: Any = None
    input_boxes_mask: Any = None
    input_boxes_label: Any = None
    input_points: Any = None
    input_points_mask: Any = None


def _update_out(
    out: dict[str, Any],
    out_name: str,
    out_value: Any,
    auxiliary: bool = True,
    update_aux: bool = True,
) -> None:
    out[out_name] = out_value[-1] if auxiliary else out_value
    if auxiliary and update_aux:
        if "aux_outputs" not in out:
            out["aux_outputs"] = [{} for _ in range(len(out_value) - 1)]
        assert len(out["aux_outputs"]) == len(out_value) - 1
        for aux_output, aux_value in zip(out["aux_outputs"], out_value[:-1]):
            aux_output[out_name] = aux_value


class Sam3Image(nn.Module):
    """Text-conditioned open-vocab detector head stack (eval)."""

    def __init__(
        self,
        backbone: nn.Module,
        transformer: nn.Module,
        input_geometry_encoder: nn.Module,
        segmentation_head: nn.Module | None = None,
        num_feature_levels: int = 1,
        o2m_mask_predict: bool = True,
        dot_prod_scoring: nn.Module | None = None,
        use_instance_query: bool = False,
        multimask_output: bool = True,
        use_act_checkpoint_seg_head: bool = False,
        interactivity_in_encoder: bool = True,
        matcher: Any = None,
        use_dot_prod_scoring: bool = True,
        supervise_joint_box_scores: bool = False,
        detach_presence_in_joint_score: bool = False,
        separate_scorer_for_instance: bool = False,
        num_interactive_steps_val: int = 0,
        inst_interactive_predictor: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        del kwargs  # accept extra official kwargs
        self.backbone = backbone
        self.geometry_encoder = input_geometry_encoder
        self.transformer = transformer
        self.hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.segmentation_head = segmentation_head
        self.o2m_mask_predict = o2m_mask_predict
        self.dot_prod_scoring = dot_prod_scoring
        self.use_act_checkpoint_seg_head = use_act_checkpoint_seg_head
        self.interactivity_in_encoder = interactivity_in_encoder
        self.matcher = matcher
        self.num_interactive_steps_val = num_interactive_steps_val
        self.use_dot_prod_scoring = use_dot_prod_scoring
        self.supervise_joint_box_scores = supervise_joint_box_scores
        self.detach_presence_in_joint_score = detach_presence_in_joint_score
        self.use_instance_query = use_instance_query
        self.multimask_output = multimask_output
        self.inst_interactive_predictor = inst_interactive_predictor
        self.instance_dot_prod_scoring = None
        self._device_ref = nn.Buffer(torch.empty(0), persistent=False)

        num_o2o_static = self.transformer.decoder.num_queries
        num_o2m_static = self.transformer.decoder.num_o2m_queries
        assert num_o2m_static == (num_o2o_static if self.transformer.decoder.dac else 0)
        self.dac = self.transformer.decoder.dac

    @property
    def device(self) -> torch.device:
        return self._device_ref.device

    def _get_img_feats(
        self,
        backbone_out: Mapping[str, Any],
        img_ids: Integer[Tensor, "..."],
    ) -> tuple[Mapping[str, Any], list[Tensor], list[Tensor], list[tuple[int, int]]]:
        if "backbone_fpn" not in backbone_out:
            raise KeyError(
                "backbone_out missing backbone_fpn; call forward_image first"
            )
        if "id_mapping" in backbone_out and backbone_out["id_mapping"] is not None:
            img_ids = backbone_out["id_mapping"][img_ids]
            torch._assert_async((img_ids >= 0).all())

        vis_feats = backbone_out["backbone_fpn"][-self.num_feature_levels :]
        vis_pos_enc = backbone_out["vision_pos_enc"][-self.num_feature_levels :]
        vis_feat_sizes = [x.shape[-2:] for x in vis_pos_enc]
        img_feats = [x[img_ids].flatten(2).permute(2, 0, 1) for x in vis_feats]
        img_pos_embeds = [x[img_ids].flatten(2).permute(2, 0, 1) for x in vis_pos_enc]
        return backbone_out, img_feats, img_pos_embeds, vis_feat_sizes

    def _encode_prompt(
        self,
        backbone_out: Mapping[str, Any],
        find_input: FindStage,
        geometric_prompt: Prompt,
        visual_prompt_embed: Float[Tensor, "..."] | None = None,
        visual_prompt_mask: Bool[Tensor, "..."] | Integer[Tensor, "..."] | None = None,
        encode_text: bool = True,
        prev_mask_pred: Float[Tensor, "..."] | None = None,
    ) -> tuple[
        Float[Tensor, "..."],
        Bool[Tensor, "..."] | Integer[Tensor, "..."],
        Mapping[str, Any],
    ]:
        txt_ids = find_input.text_ids
        txt_feats = backbone_out["language_features"][:, txt_ids]
        txt_masks = backbone_out["language_mask"][txt_ids]

        feat_tuple = self._get_img_feats(backbone_out, find_input.img_ids)
        backbone_out, img_feats, img_pos_embeds, vis_feat_sizes = feat_tuple

        if prev_mask_pred is not None:
            img_feats = [img_feats[-1] + prev_mask_pred]

        geo_feats, geo_masks = self.geometry_encoder(
            geo_prompt=geometric_prompt,
            img_feats=img_feats,
            img_sizes=vis_feat_sizes,
            img_pos_embeds=img_pos_embeds,
        )
        if visual_prompt_embed is None:
            visual_prompt_embed = torch.zeros(
                (0, *geo_feats.shape[1:]), device=geo_feats.device
            )
            visual_prompt_mask = torch.zeros(
                (*geo_masks.shape[:-1], 0),
                device=geo_masks.device,
                dtype=geo_masks.dtype,
            )
        if encode_text:
            prompt = torch.cat([txt_feats, geo_feats, visual_prompt_embed], dim=0)
            prompt_mask = torch.cat([txt_masks, geo_masks, visual_prompt_mask], dim=1)
        else:
            prompt = torch.cat([geo_feats, visual_prompt_embed], dim=0)
            prompt_mask = torch.cat([geo_masks, visual_prompt_mask], dim=1)
        return prompt, prompt_mask, backbone_out

    def _run_encoder(
        self,
        backbone_out: Mapping[str, Any],
        find_input: FindStage,
        prompt: Float[Tensor, "..."],
        prompt_mask: Bool[Tensor, "..."] | Integer[Tensor, "..."],
        encoder_extra_kwargs: Mapping[str, Any] | None = None,
    ) -> tuple[Mapping[str, Any], dict[str, Any], tuple[Any, ...]]:
        feat_tuple = self._get_img_feats(backbone_out, find_input.img_ids)
        backbone_out, img_feats, img_pos_embeds, vis_feat_sizes = feat_tuple

        prompt_pos_embed = torch.zeros_like(prompt)
        memory = self.transformer.encoder(
            src=img_feats.copy(),
            src_key_padding_mask=None,
            src_pos=img_pos_embeds.copy(),
            prompt=prompt,
            prompt_pos=prompt_pos_embed,
            prompt_key_padding_mask=prompt_mask,
            feat_sizes=vis_feat_sizes,
            encoder_extra_kwargs=encoder_extra_kwargs,
        )
        encoder_out = {
            "encoder_hidden_states": memory["memory"],
            "pos_embed": memory["pos_embed"],
            "padding_mask": memory["padding_mask"],
            "level_start_index": memory["level_start_index"],
            "spatial_shapes": memory["spatial_shapes"],
            "valid_ratios": memory["valid_ratios"],
            "vis_feat_sizes": vis_feat_sizes,
            "prompt_before_enc": prompt,
            "prompt_after_enc": memory.get("memory_text", prompt),
            "prompt_mask": prompt_mask,
        }
        return backbone_out, encoder_out, feat_tuple

    def _update_scores_and_boxes(
        self,
        out,
        hs,
        reference_boxes,
        prompt,
        prompt_mask,
        dec_presence_out=None,
        is_instance_prompt=False,
    ):
        apply_dac = self.transformer.decoder.dac and self.training
        num_o2o = (hs.size(2) // 2) if apply_dac else hs.size(2)
        num_o2m = hs.size(2) - num_o2o
        assert num_o2m == (num_o2o if apply_dac else 0)
        out["queries"] = hs[-1][:, :num_o2o]
        if self.use_dot_prod_scoring:
            outputs_class = self.dot_prod_scoring(hs, prompt, prompt_mask)
        else:
            raise NotImplementedError("class_embed path not used in production eval")

        box_head = self.transformer.decoder.bbox_embed
        anchor_box_offsets = box_head(hs)
        reference_boxes_inv_sig = inverse_sigmoid(reference_boxes)
        outputs_coord = (reference_boxes_inv_sig + anchor_box_offsets).sigmoid()
        outputs_boxes_xyxy = box_convert(
            outputs_coord.reshape(-1, 4), in_fmt="cxcywh", out_fmt="xyxy"
        ).reshape_as(outputs_coord)

        if dec_presence_out is not None:
            _update_out(
                out, "presence_logit_dec", dec_presence_out, update_aux=self.training
            )

        if self.supervise_joint_box_scores:
            assert dec_presence_out is not None
            prob_dec_presence_out = dec_presence_out.clone().sigmoid()
            if self.detach_presence_in_joint_score:
                prob_dec_presence_out = prob_dec_presence_out.detach()
            outputs_class = inverse_sigmoid(
                outputs_class.sigmoid() * prob_dec_presence_out.unsqueeze(2)
            ).clamp(min=-10.0, max=10.0)

        _update_out(
            out, "pred_logits", outputs_class[:, :, :num_o2o], update_aux=self.training
        )
        _update_out(
            out, "pred_boxes", outputs_coord[:, :, :num_o2o], update_aux=self.training
        )
        _update_out(
            out,
            "pred_boxes_xyxy",
            outputs_boxes_xyxy[:, :, :num_o2o],
            update_aux=self.training,
        )

    def _run_decoder(
        self,
        pos_embed,
        memory,
        src_mask,
        out,
        prompt,
        prompt_mask,
        encoder_out,
    ):
        bs = memory.shape[1]
        query_embed = self.transformer.decoder.query_embed.weight
        tgt = query_embed.unsqueeze(1).repeat(1, bs, 1)

        apply_dac = self.transformer.decoder.dac and self.training
        hs, reference_boxes, dec_presence_out, dec_presence_feats = (
            self.transformer.decoder(
                tgt=tgt,
                memory=memory,
                memory_key_padding_mask=src_mask,
                pos=pos_embed,
                reference_boxes=None,
                level_start_index=encoder_out["level_start_index"],
                spatial_shapes=encoder_out["spatial_shapes"],
                valid_ratios=encoder_out["valid_ratios"],
                feature_size=encoder_out["vis_feat_sizes"][-1],
                tgt_mask=None,
                memory_text=prompt,
                text_attention_mask=prompt_mask,
                apply_dac=apply_dac,
            )
        )
        hs = hs.transpose(1, 2)
        reference_boxes = reference_boxes.transpose(1, 2)
        if dec_presence_out is not None:
            dec_presence_out = dec_presence_out.transpose(1, 2)

        out["presence_feats"] = dec_presence_feats
        self._update_scores_and_boxes(
            out,
            hs,
            reference_boxes,
            prompt,
            prompt_mask,
            dec_presence_out=dec_presence_out,
        )
        return out, hs

    def _run_segmentation_heads(
        self,
        out,
        backbone_out,
        img_ids,
        vis_feat_sizes,
        encoder_hidden_states,
        prompt,
        prompt_mask,
        hs,
    ):
        del vis_feat_sizes
        apply_dac = self.transformer.decoder.dac and self.training
        if self.segmentation_head is not None:
            num_o2o = (hs.size(2) // 2) if apply_dac else hs.size(2)
            num_o2m = hs.size(2) - num_o2o
            obj_queries = hs if self.o2m_mask_predict else hs[:, :, :num_o2o]
            seg_head_outputs = self.segmentation_head(
                backbone_feats=backbone_out["backbone_fpn"],
                obj_queries=obj_queries,
                image_ids=img_ids,
                encoder_hidden_states=encoder_hidden_states,
                prompt=prompt,
                prompt_mask=prompt_mask,
            )
            aux_masks = False
            for k, v in seg_head_outputs.items():
                if k in self.segmentation_head.instance_keys:
                    _update_out(out, k, v[:, :num_o2o], auxiliary=aux_masks)
                    if self.o2m_mask_predict and num_o2m > 0:
                        _update_out(
                            out, f"{k}_o2m", v[:, num_o2o:], auxiliary=aux_masks
                        )
                else:
                    out[k] = v
        else:
            backbone_out.pop("backbone_fpn", None)

    def _get_dummy_prompt(self, num_prompts: int = 1) -> Prompt:
        device = self.device
        dtype = module_param_dtype(self)
        return Prompt(
            box_embeddings=torch.zeros(0, num_prompts, 4, device=device, dtype=dtype),
            box_mask=torch.zeros(num_prompts, 0, device=device, dtype=torch.bool),
        )

    def forward_grounding(
        self,
        backbone_out: dict[str, Any],
        find_input: FindStage,
        find_target: Any,
        geometric_prompt: Prompt,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del find_target, kwargs
        # Unify floating dtypes after PE / tokenizer paths (often stay fp32).
        dtype = module_param_dtype(self)
        for key in ("backbone_fpn", "vision_pos_enc", "language_features"):
            if key not in backbone_out:
                continue
            val = backbone_out[key]
            if isinstance(val, (list, tuple)):
                backbone_out[key] = [
                    t.to(dtype=dtype) if torch.is_floating_point(t) else t for t in val
                ]
            elif torch.is_tensor(val) and torch.is_floating_point(val):
                backbone_out[key] = val.to(dtype=dtype)

        prompt, prompt_mask, backbone_out = self._encode_prompt(
            backbone_out, find_input, geometric_prompt
        )
        if torch.is_floating_point(prompt):
            prompt = prompt.to(dtype=dtype)
        backbone_out, encoder_out, _ = self._run_encoder(
            backbone_out, find_input, prompt, prompt_mask
        )
        out = {
            "encoder_hidden_states": encoder_out["encoder_hidden_states"],
            "prev_encoder_out": {
                "encoder_out": encoder_out,
                "backbone_out": backbone_out,
            },
        }
        out, hs = self._run_decoder(
            memory=out["encoder_hidden_states"],
            pos_embed=encoder_out["pos_embed"],
            src_mask=encoder_out["padding_mask"],
            out=out,
            prompt=prompt,
            prompt_mask=prompt_mask,
            encoder_out=encoder_out,
        )
        seg_img_ids = find_input.img_ids
        if "id_mapping" in backbone_out and backbone_out["id_mapping"] is not None:
            seg_img_ids = backbone_out["id_mapping"][seg_img_ids]
        self._run_segmentation_heads(
            out=out,
            backbone_out=backbone_out,
            img_ids=seg_img_ids,
            vis_feat_sizes=encoder_out["vis_feat_sizes"],
            encoder_hidden_states=out["encoder_hidden_states"],
            prompt=prompt,
            prompt_mask=prompt_mask,
            hs=hs,
        )
        return out
