"""Core SAM3 per-object tracker.

Provides the memory-bank fusion path used for multi-frame point/mask tracking:
``_prepare_memory_conditioned_features``, ``_encode_new_memory``, ``track_step``.

The production tracker always uses the concrete memory encoder and tracker
transformer in this package; incomplete alternatives are intentionally absent.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from jaxtyping import Float, Integer
from tensordict import TensorDict, TensorDictBase
from timm.layers import trunc_normal_
from timm.models.convnext import ConvNeXtBlock
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from sam3.dtype_policy import PrecisionConfig

from ..grounding.transformer_wrapper import TransformerWrapper
from ..primitives.mlp import MLP
from ..primitives.position_encoding import PositionEmbeddingSine
from ..primitives.two_way_transformer import TwoWayTransformer
from ..vision.mask_decoder import MaskDecoder
from ..vision.prompt_encoder import PromptEncoder
from .memory import SimpleFuser, SimpleMaskDownSampler, SimpleMaskEncoder
from .tracker_transformer import (
    create_tracker_transformer as _create_tracker_encoder,
)
from .tracker_utils import get_1d_sine_pe, select_closest_cond_frames


def create_tracker_transformer(
    d_model: int = 256,
    num_layers: int = 4,
    feat_sizes: tuple[int, int] = (72, 72),
    **kwargs: Any,
) -> TransformerWrapper:
    """Wrap the tracker encoder in the standard transformer container."""
    encoder = _create_tracker_encoder(
        d_model=d_model,
        num_layers=num_layers,
        feat_sizes=feat_sizes,
        **kwargs,
    )
    return TransformerWrapper(encoder=encoder, decoder=None, d_model=d_model)


# A large negative score used for absent object predictions.
NO_OBJ_SCORE = -1024.0


class Sam3TrackerBase(nn.Module):
    """Eval-critical core of SAM3 video tracker (memory bank + SAM heads + track_step)."""

    def __init__(
        self,
        backbone: nn.Module | None,
        transformer: nn.Module,
        maskmem_backbone: nn.Module,
        num_maskmem: int = 7,  # default 1 input frame + 6 previous frames
        image_size: int = 1008,
        backbone_stride: int = 14,
        max_cond_frames_in_attn: int = -1,
        keep_first_cond_frame: bool = False,
        multimask_output_in_sam: bool = False,
        multimask_min_pt_num: int = 1,
        multimask_max_pt_num: int = 1,
        multimask_output_for_tracking: bool = False,
        forward_backbone_per_frame_for_eval: bool = False,
        memory_temporal_stride_for_eval: int = 1,
        offload_output_to_cpu_for_eval: bool = False,
        trim_past_non_cond_mem_for_eval: bool = False,
        non_overlap_masks_for_mem_enc: bool = False,
        max_obj_ptrs_in_encoder: int = 16,
        sam_mask_decoder_extra_args: dict[str, Any] | None = None,
        # SAM head construction knobs (for tiny tests)
        sam_num_heads: int = 8,
        sam_mlp_dim: int = 2048,
        sam_twoway_depth: int = 2,
        *,
        precision: PrecisionConfig | None = None,
    ) -> None:
        super().__init__()

        # Part 1: the image backbone (optional for synthetic track_step tests)
        self.backbone = backbone
        self.num_feature_levels = 3
        self.max_obj_ptrs_in_encoder = max_obj_ptrs_in_encoder
        self.mask_downsample = nn.Conv2d(1, 1, kernel_size=4, stride=4)

        # Part 2: encoder-only transformer to fuse current frame visual features
        # with memories from past frames
        assert getattr(transformer, "decoder", None) is None, (
            "transformer should be encoder-only"
        )
        self.transformer = transformer
        self.hidden_dim = transformer.d_model

        # Part 3: memory encoder for previous frame outputs
        self.maskmem_backbone = maskmem_backbone
        self.mem_dim = self.hidden_dim
        if hasattr(self.maskmem_backbone, "out_proj") and hasattr(
            self.maskmem_backbone.out_proj, "weight"
        ):
            self.mem_dim = self.maskmem_backbone.out_proj.weight.shape[0]
        self.num_maskmem = num_maskmem

        self.maskmem_tpos_enc = nn.Parameter(
            torch.zeros(num_maskmem, 1, 1, self.mem_dim)
        )
        trunc_normal_(self.maskmem_tpos_enc, std=0.02)

        self.no_mem_embed = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.no_mem_pos_enc = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        trunc_normal_(self.no_mem_embed, std=0.02)
        trunc_normal_(self.no_mem_pos_enc, std=0.02)

        self.sigmoid_scale_for_mem_enc = 20.0
        self.sigmoid_bias_for_mem_enc = -10.0
        self.non_overlap_masks_for_mem_enc = non_overlap_masks_for_mem_enc
        self.memory_temporal_stride_for_eval = memory_temporal_stride_for_eval
        self.multimask_output_in_sam = multimask_output_in_sam
        self.multimask_min_pt_num = multimask_min_pt_num
        self.multimask_max_pt_num = multimask_max_pt_num
        self.multimask_output_for_tracking = multimask_output_for_tracking
        self.precision = precision

        # Part 4: SAM-style prompt encoder + mask decoder
        self.image_size = image_size
        self.backbone_stride = backbone_stride
        self.low_res_mask_size = self.image_size // self.backbone_stride * 4
        self.input_mask_size = self.low_res_mask_size * 4
        self.forward_backbone_per_frame_for_eval = forward_backbone_per_frame_for_eval
        self.offload_output_to_cpu_for_eval = offload_output_to_cpu_for_eval
        self.trim_past_non_cond_mem_for_eval = trim_past_non_cond_mem_for_eval
        self.sam_mask_decoder_extra_args = sam_mask_decoder_extra_args
        self.no_obj_ptr = nn.Parameter(torch.zeros(1, self.hidden_dim))
        trunc_normal_(self.no_obj_ptr, std=0.02)
        self.no_obj_embed_spatial = nn.Parameter(torch.zeros(1, self.mem_dim))
        trunc_normal_(self.no_obj_embed_spatial, std=0.02)

        self._sam_num_heads = sam_num_heads
        self._sam_mlp_dim = sam_mlp_dim
        self._sam_twoway_depth = sam_twoway_depth
        self._build_sam_heads()
        self.max_cond_frames_in_attn = max_cond_frames_in_attn
        self.keep_first_cond_frame = keep_first_cond_frame

        # Training-only flags present on official class (keep inert for eval)
        self.teacher_force_obj_scores_for_mem = False
        self.prob_to_dropout_spatial_mem = 0.0
        self.iter_use_prev_mask_pred = False

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _to_storage(self, tensor: Tensor | None) -> Tensor | None:
        if tensor is None or self.precision is None:
            return tensor
        storage_dtype = self.precision.resolved_storage()
        if torch.is_floating_point(tensor) and storage_dtype != tensor.dtype:
            return tensor.to(dtype=storage_dtype)
        return tensor

    def _get_tpos_enc(
        self,
        rel_pos_list: Sequence[float | int],
        device: torch.device,
        max_abs_pos: int | None = None,
        dummy: bool = False,
    ) -> Float[Tensor, "..."]:
        if dummy:
            return torch.zeros(len(rel_pos_list), self.mem_dim, device=device)

        t_diff_max = max_abs_pos - 1 if max_abs_pos is not None else 1
        pos_enc = (
            torch.tensor(rel_pos_list, dtype=torch.float32, device=device) / t_diff_max
        )
        tpos_dim = self.hidden_dim
        pos_enc = get_1d_sine_pe(pos_enc, dim=tpos_dim)
        # sine PE is fp32; match Linear weight dtype (bf16/fp16)
        pos_enc = pos_enc.to(dtype=self.obj_ptr_tpos_proj.weight.dtype)
        pos_enc = self.obj_ptr_tpos_proj(pos_enc)
        return pos_enc

    def _build_sam_heads(self) -> None:
        """Build SAM-style prompt encoder and mask decoder."""
        self.sam_prompt_embed_dim = self.hidden_dim
        self.sam_image_embedding_size = self.image_size // self.backbone_stride

        self.sam_prompt_encoder = PromptEncoder(
            embed_dim=self.sam_prompt_embed_dim,
            image_embedding_size=(
                self.sam_image_embedding_size,
                self.sam_image_embedding_size,
            ),
            input_image_size=(self.image_size, self.image_size),
            mask_in_chans=16,
        )
        self.sam_mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=self._sam_twoway_depth,
                embedding_dim=self.sam_prompt_embed_dim,
                mlp_dim=self._sam_mlp_dim,
                num_heads=self._sam_num_heads,
            ),
            transformer_dim=self.sam_prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=True,
            iou_prediction_use_sigmoid=True,
            pred_obj_scores=True,
            pred_obj_scores_mlp=True,
            use_multimask_token_for_obj_ptr=True,
            **(self.sam_mask_decoder_extra_args or {}),
        )
        # Official overwrites Linear with MLP(3 layers) for obj_ptr_proj
        self.obj_ptr_proj = MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim, 3)
        self.obj_ptr_tpos_proj = nn.Linear(self.hidden_dim, self.mem_dim)

    def _forward_sam_heads(
        self,
        backbone_features,
        point_inputs=None,
        mask_inputs=None,
        high_res_features=None,
        multimask_output=False,
        gt_masks=None,
    ):
        B = backbone_features.size(0)
        device = backbone_features.device
        assert backbone_features.size(1) == self.sam_prompt_embed_dim
        assert backbone_features.size(2) == self.sam_image_embedding_size
        assert backbone_features.size(3) == self.sam_image_embedding_size

        if point_inputs is not None:
            sam_point_coords = point_inputs["point_coords"]
            sam_point_labels = point_inputs["point_labels"]
            assert sam_point_coords.size(0) == B and sam_point_labels.size(0) == B
        else:
            sam_point_coords = torch.zeros(B, 1, 2, device=device)
            sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=device)

        if mask_inputs is not None:
            assert len(mask_inputs.shape) == 4 and mask_inputs.shape[:2] == (B, 1)
            pe_dtype = next(self.sam_prompt_encoder.parameters()).dtype
            if mask_inputs.shape[-2:] != self.sam_prompt_encoder.mask_input_size:
                sam_mask_prompt = F.interpolate(
                    mask_inputs.float(),
                    size=self.sam_prompt_encoder.mask_input_size,
                    align_corners=False,
                    mode="bilinear",
                    antialias=True,
                ).to(dtype=pe_dtype)
            else:
                sam_mask_prompt = mask_inputs.to(dtype=pe_dtype)
        else:
            sam_mask_prompt = None

        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
            points=(sam_point_coords, sam_point_labels),
            boxes=None,
            masks=sam_mask_prompt,
        )
        image_pe = self.sam_prompt_encoder.get_dense_pe()
        # Align PE / prompt activations to backbone compute dtype (permanent bf16).
        feat_dtype = backbone_features.dtype
        if sparse_embeddings.dtype != feat_dtype:
            sparse_embeddings = sparse_embeddings.to(dtype=feat_dtype)
        if dense_embeddings.dtype != feat_dtype:
            dense_embeddings = dense_embeddings.to(dtype=feat_dtype)
        if image_pe.dtype != feat_dtype:
            image_pe = image_pe.to(dtype=feat_dtype)
        if high_res_features is not None:
            high_res_features = [
                f.to(dtype=feat_dtype) if f.dtype != feat_dtype else f
                for f in high_res_features
            ]
        (
            low_res_multimasks,
            ious,
            sam_output_tokens,
            object_score_logits,
        ) = self.sam_mask_decoder(
            image_embeddings=backbone_features,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=False,
            high_res_features=high_res_features,
        )
        if self.training and self.teacher_force_obj_scores_for_mem:
            is_obj_appearing = torch.any(gt_masks.float().flatten(1) > 0, dim=1)
            is_obj_appearing = is_obj_appearing[..., None]
        else:
            is_obj_appearing = object_score_logits > 0

        low_res_multimasks = torch.where(
            is_obj_appearing[:, None, None],
            low_res_multimasks,
            NO_OBJ_SCORE,
        )

        low_res_multimasks = low_res_multimasks.float()
        high_res_multimasks = F.interpolate(
            low_res_multimasks,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        sam_output_token = sam_output_tokens[:, 0]
        if multimask_output:
            best_iou_inds = torch.argmax(ious, dim=-1)
            batch_inds = torch.arange(B, device=device)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            if sam_output_tokens.size(1) > 1:
                sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
        else:
            low_res_masks, high_res_masks = low_res_multimasks, high_res_multimasks

        obj_ptr = self.obj_ptr_proj(sam_output_token)
        lambda_is_obj_appearing = is_obj_appearing.float()
        obj_ptr = lambda_is_obj_appearing * obj_ptr
        obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        return (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        )

    def _use_mask_as_output(self, backbone_features, high_res_features, mask_inputs):
        out_scale, out_bias = 20.0, -10.0
        # mask_downsample Conv is often bf16; cast inputs to its weight dtype
        md_dtype = self.mask_downsample.weight.dtype
        mask_inputs_f = mask_inputs.to(dtype=md_dtype)
        high_res_masks = mask_inputs_f * out_scale + out_bias
        low_res_masks = F.interpolate(
            high_res_masks.float(),
            size=(
                high_res_masks.size(-2) // self.backbone_stride * 4,
                high_res_masks.size(-1) // self.backbone_stride * 4,
            ),
            align_corners=False,
            mode="bilinear",
            antialias=True,
        ).to(dtype=md_dtype)
        ious = mask_inputs.new_ones(mask_inputs.size(0), 1).float()
        _, _, _, _, _, obj_ptr, _ = self._forward_sam_heads(
            backbone_features=backbone_features,
            mask_inputs=self.mask_downsample(mask_inputs_f),
            high_res_features=high_res_features,
            gt_masks=mask_inputs,
        )
        is_obj_appearing = torch.any(mask_inputs.flatten(1).float() > 0.0, dim=1)
        is_obj_appearing = is_obj_appearing[..., None]
        lambda_is_obj_appearing = is_obj_appearing.float()
        object_score_logits = out_scale * lambda_is_obj_appearing + out_bias
        obj_ptr = lambda_is_obj_appearing * obj_ptr
        obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        return (
            low_res_masks,
            high_res_masks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        )

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "Use track_step (or a video predictor wrapper) for inference."
        )

    def forward_image(self, img_batch: Float[Tensor, "b 3 h w"]) -> dict[str, Any]:
        """Get image features via the attached backbone (optional)."""
        if self.backbone is None:
            raise RuntimeError(
                "No backbone attached; provide precomputed vision features to track_step."
            )
        backbone_out = self.backbone.forward_image(img_batch)["sam2_backbone_out"]
        backbone_out["backbone_fpn"][0] = self.sam_mask_decoder.conv_s0(
            backbone_out["backbone_fpn"][0]
        )
        backbone_out["backbone_fpn"][1] = self.sam_mask_decoder.conv_s1(
            backbone_out["backbone_fpn"][1]
        )
        return backbone_out

    def _prepare_backbone_features(
        self, backbone_out: dict[str, Any]
    ) -> tuple[
        dict[str, Any],
        list[Tensor],
        list[Tensor],
        list[tuple[int, int]],
    ]:
        backbone_out = backbone_out.copy()
        assert len(backbone_out["backbone_fpn"]) == len(backbone_out["vision_pos_enc"])
        assert len(backbone_out["backbone_fpn"]) >= self.num_feature_levels

        feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels :]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels :]

        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]
        return backbone_out, vision_feats, vision_pos_embeds, feat_sizes

    def _prepare_memory_conditioned_features(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        output_dict,
        num_frames,
        track_in_reverse=False,
        use_prev_mem_frame=True,
    ):
        """Fuse the current frame's visual feature map with previous memory."""
        B = current_vision_feats[-1].size(1)
        C = self.hidden_dim
        H, W = feat_sizes[-1]
        device = current_vision_feats[-1].device

        if self.num_maskmem == 0:
            pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
            return pix_feat

        num_obj_ptr_tokens = 0
        tpos_sign_mul = -1 if track_in_reverse else 1

        if not is_init_cond_frame and use_prev_mem_frame:
            to_cat_prompt, to_cat_prompt_mask, to_cat_prompt_pos_embed = [], [], []
            assert len(output_dict["cond_frame_outputs"]) > 0
            cond_outputs = output_dict["cond_frame_outputs"]
            selected_cond_outputs, unselected_cond_outputs = select_closest_cond_frames(
                frame_idx,
                cond_outputs,
                self.max_cond_frames_in_attn,
                keep_first_cond_frame=self.keep_first_cond_frame,
            )
            t_pos_and_prevs = [
                ((frame_idx - t) * tpos_sign_mul, out, True)
                for t, out in selected_cond_outputs.items()
            ]
            r = 1 if self.training else self.memory_temporal_stride_for_eval

            for t_pos in range(1, self.num_maskmem):
                t_rel = self.num_maskmem - t_pos
                if t_rel == 1:
                    if not track_in_reverse:
                        prev_frame_idx = frame_idx - t_rel
                    else:
                        prev_frame_idx = frame_idx + t_rel
                else:
                    if not track_in_reverse:
                        prev_frame_idx = ((frame_idx - 2) // r) * r
                        prev_frame_idx = prev_frame_idx - (t_rel - 2) * r
                    else:
                        prev_frame_idx = -(-(frame_idx + 2) // r) * r
                        prev_frame_idx = prev_frame_idx + (t_rel - 2) * r

                out = output_dict["non_cond_frame_outputs"].get(prev_frame_idx, None)
                if out is None:
                    out = unselected_cond_outputs.get(prev_frame_idx, None)
                t_pos_and_prevs.append((t_pos, out, False))

            for t_pos, prev, is_selected_cond_frame in t_pos_and_prevs:
                if prev is None:
                    continue
                feats = prev["maskmem_features"].to(device, non_blocking=True)
                seq_len = feats.shape[-2] * feats.shape[-1]
                to_cat_prompt.append(feats.flatten(2).permute(2, 0, 1))
                to_cat_prompt_mask.append(
                    torch.zeros(B, seq_len, device=device, dtype=torch.bool)
                )
                maskmem_enc = self._get_maskmem_pos_enc_last(prev).to(
                    device, non_blocking=True
                )
                maskmem_enc = maskmem_enc.flatten(2).permute(2, 0, 1)

                if (
                    is_selected_cond_frame
                    and getattr(self, "cond_frame_spatial_embedding", None) is not None
                ):
                    maskmem_enc = maskmem_enc + self.cond_frame_spatial_embedding

                t = t_pos if not is_selected_cond_frame else 0
                maskmem_enc = (
                    maskmem_enc + self.maskmem_tpos_enc[self.num_maskmem - t - 1]
                )
                to_cat_prompt_pos_embed.append(maskmem_enc)

            max_obj_ptrs_in_encoder = min(num_frames, self.max_obj_ptrs_in_encoder)
            if not self.training:
                ptr_cond_outputs = {
                    t: out
                    for t, out in selected_cond_outputs.items()
                    if (t >= frame_idx if track_in_reverse else t <= frame_idx)
                }
            else:
                ptr_cond_outputs = selected_cond_outputs
            pos_and_ptrs = [
                (
                    (frame_idx - t) * tpos_sign_mul,
                    out["obj_ptr"],
                    True,
                )
                for t, out in ptr_cond_outputs.items()
            ]

            for t_diff in range(1, max_obj_ptrs_in_encoder):
                t = frame_idx + t_diff if track_in_reverse else frame_idx - t_diff
                if t < 0 or (num_frames is not None and t >= num_frames):
                    break
                out = output_dict["non_cond_frame_outputs"].get(
                    t, unselected_cond_outputs.get(t, None)
                )
                if out is not None:
                    pos_and_ptrs.append((t_diff, out["obj_ptr"], False))

            if len(pos_and_ptrs) > 0:
                pos_list, ptrs_list, is_selected_cond_frame_list = zip(*pos_and_ptrs)
                obj_ptrs = torch.stack(ptrs_list, dim=0)
                if getattr(self, "cond_frame_obj_ptr_embedding", None) is not None:
                    obj_ptrs = (
                        obj_ptrs
                        + self.cond_frame_obj_ptr_embedding
                        * torch.tensor(is_selected_cond_frame_list, device=device)[
                            ..., None, None
                        ].float()
                    )
                obj_pos = self._get_tpos_enc(
                    pos_list,
                    max_abs_pos=max_obj_ptrs_in_encoder,
                    device=device,
                )
                obj_pos = obj_pos.unsqueeze(1).expand(-1, B, -1)

                if self.mem_dim < C:
                    obj_ptrs = obj_ptrs.reshape(-1, B, C // self.mem_dim, self.mem_dim)
                    obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
                    obj_pos = obj_pos.repeat_interleave(C // self.mem_dim, dim=0)
                to_cat_prompt.append(obj_ptrs)
                to_cat_prompt_mask.append(None)
                to_cat_prompt_pos_embed.append(obj_pos)
                num_obj_ptr_tokens = obj_ptrs.shape[0]
            else:
                num_obj_ptr_tokens = 0
        else:
            # Init cond frame: add no-mem embedding and return (skip transformer)
            pix_feat_with_mem = current_vision_feats[-1] + self.no_mem_embed
            pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
            return pix_feat_with_mem

        prompt = torch.cat(to_cat_prompt, dim=0)
        prompt_mask = None
        prompt_pos_embed = torch.cat(to_cat_prompt_pos_embed, dim=0)
        encoder_out = self.transformer.encoder(
            src=current_vision_feats,
            src_key_padding_mask=[None],
            src_pos=current_vision_pos_embeds,
            prompt=prompt,
            prompt_pos=prompt_pos_embed,
            prompt_key_padding_mask=prompt_mask,
            feat_sizes=feat_sizes,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )
        pix_feat_with_mem = encoder_out["memory"].permute(1, 2, 0).view(B, C, H, W)
        return pix_feat_with_mem

    def _encode_new_memory(
        self,
        image,
        current_vision_feats,
        feat_sizes,
        pred_masks_high_res,
        object_score_logits,
        is_mask_from_pts,
        output_dict=None,
        is_init_cond_frame=False,
    ):
        B = current_vision_feats[-1].size(1)
        C = self.hidden_dim
        H, W = feat_sizes[-1]
        pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
        if self.non_overlap_masks_for_mem_enc and not self.training:
            pred_masks_high_res = self._apply_non_overlapping_constraints(
                pred_masks_high_res
            )
        if is_mask_from_pts and not self.training:
            mask_for_mem = (pred_masks_high_res > 0).float()
        else:
            mask_for_mem = torch.sigmoid(pred_masks_high_res)
        if self.sigmoid_scale_for_mem_enc != 1.0:
            mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
        if self.sigmoid_bias_for_mem_enc != 0.0:
            mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc

        if isinstance(self.maskmem_backbone, SimpleMaskEncoder):
            maskmem_out = self.maskmem_backbone(
                pix_feat, mask_for_mem, skip_mask_sigmoid=True
            )
        else:
            maskmem_out = self.maskmem_backbone(image, pix_feat, mask_for_mem)

        maskmem_features = maskmem_out["vision_features"]
        maskmem_pos_enc = list(maskmem_out["vision_pos_enc"])
        is_obj_appearing = (object_score_logits > 0).float()
        maskmem_features = maskmem_features + (
            1 - is_obj_appearing[..., None, None]
        ) * self.no_obj_embed_spatial[..., None, None].expand(*maskmem_features.shape)

        return maskmem_features, maskmem_pos_enc

    @staticmethod
    def _get_maskmem_pos_enc_last(prev) -> Tensor:
        pos_enc = prev["maskmem_pos_enc"]
        if isinstance(pos_enc, Tensor):
            return pos_enc
        if isinstance(pos_enc, list):
            return pos_enc[-1]
        if isinstance(pos_enc, TensorDictBase):
            if len(pos_enc) == 0:
                raise IndexError("maskmem_pos_enc is empty")
            keys = [key for key in pos_enc.keys() if isinstance(key, str)]
            if not keys:
                raise IndexError("maskmem_pos_enc has no string keys")
            numeric_keys = [int(k) for k in keys if k.isdigit()]
            if not numeric_keys:
                key = max(keys)
            else:
                key = str(max(numeric_keys))
            return pos_enc[key]
        if isinstance(pos_enc, dict):
            if not pos_enc:
                raise IndexError("maskmem_pos_enc is empty")
            keys = list(pos_enc.keys())
            str_keys = [key for key in keys if isinstance(key, str)]
            numeric_keys = [int(key) for key in str_keys if key.isdigit()]
            if numeric_keys:
                return pos_enc[str(max(numeric_keys))]
            int_keys = [key for key in keys if isinstance(key, int)]
            if int_keys:
                return pos_enc[max(int_keys)]
            if len(pos_enc) == 1 and 0 in pos_enc:
                return pos_enc[0]
            if len(pos_enc) == 1 and "0" in pos_enc:
                return pos_enc["0"]
            last_key = list(pos_enc.keys())[-1]
            return pos_enc[last_key]
        return pos_enc[-1]

    def track_step(
        self,
        frame_idx: int,
        is_init_cond_frame: bool,
        current_vision_feats: Sequence[Float[Tensor, "..."]],
        current_vision_pos_embeds: Sequence[Float[Tensor, "..."]],
        feat_sizes: Sequence[tuple[int, int]],
        image: Float[Tensor, "..."] | None,
        point_inputs: Mapping[str, Tensor] | None,
        mask_inputs: Float[Tensor, "..."] | None,
        output_dict: Mapping[str, Any],
        num_frames: int,
        track_in_reverse: bool = False,
        run_mem_encoder: bool = True,
        prev_sam_mask_logits: Float[Tensor, "..."] | None = None,
        use_prev_mem_frame: bool = True,
    ) -> TensorDict | dict[str, Any]:
        B = current_vision_feats[-1].size(1)
        current_out: TensorDict = TensorDict(
            {"point_inputs": point_inputs, "mask_inputs": mask_inputs},
            batch_size=(B,),
        )
        if len(current_vision_feats) > 1:
            high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(current_vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            high_res_features = None

        if mask_inputs is not None:
            pix_feat = current_vision_feats[-1].permute(1, 2, 0)
            pix_feat = pix_feat.view(-1, self.hidden_dim, *feat_sizes[-1])
            sam_outputs = self._use_mask_as_output(
                pix_feat, high_res_features, mask_inputs
            )
        else:
            pix_feat_with_mem = self._prepare_memory_conditioned_features(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=current_vision_feats[-1:],
                current_vision_pos_embeds=current_vision_pos_embeds[-1:],
                feat_sizes=feat_sizes[-1:],
                output_dict=output_dict,
                num_frames=num_frames,
                track_in_reverse=track_in_reverse,
                use_prev_mem_frame=use_prev_mem_frame,
            )
            if prev_sam_mask_logits is not None:
                assert self.iter_use_prev_mask_pred
                assert point_inputs is not None and mask_inputs is None
                mask_inputs = prev_sam_mask_logits
            multimask_output = self._use_multimask(is_init_cond_frame, point_inputs)
            sam_outputs = self._forward_sam_heads(
                backbone_features=pix_feat_with_mem,
                point_inputs=point_inputs,
                mask_inputs=mask_inputs,
                high_res_features=high_res_features,
                multimask_output=multimask_output,
            )
        (
            _,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        ) = sam_outputs
        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr
        if not self.training:
            current_out["object_score_logits"] = object_score_logits

        if run_mem_encoder and self.num_maskmem > 0:
            maskmem_features, maskmem_pos_enc = self._encode_new_memory(
                image=image,
                current_vision_feats=current_vision_feats,
                feat_sizes=feat_sizes,
                pred_masks_high_res=high_res_masks,
                object_score_logits=object_score_logits,
                is_mask_from_pts=(point_inputs is not None),
                output_dict=output_dict,
                is_init_cond_frame=is_init_cond_frame,
            )
            current_out["maskmem_features"] = self._to_storage(maskmem_features)
            current_out["maskmem_pos_enc"] = TensorDict(
                {str(i): x for i, x in enumerate(maskmem_pos_enc)},
                batch_size=(B,),
            )
        else:
            current_out["maskmem_features"] = None
            current_out["maskmem_pos_enc"] = None

        if self.offload_output_to_cpu_for_eval and not self.training:
            keep_on_device = current_out.select("obj_ptr", "object_score_logits")
            current_out = current_out.update(
                current_out.select(
                    "pred_masks",
                    "pred_masks_high_res",
                    "maskmem_features",
                    "maskmem_pos_enc",
                ).cpu()
            )
            current_out["obj_ptr"] = keep_on_device["obj_ptr"]
            current_out["object_score_logits"] = keep_on_device["object_score_logits"]

        if self.trim_past_non_cond_mem_for_eval and not self.training:
            r = self.memory_temporal_stride_for_eval
            past_frame_idx = frame_idx - r * self.num_maskmem
            past_out = output_dict["non_cond_frame_outputs"].get(past_frame_idx, None)
            if past_out is not None:
                output_dict["non_cond_frame_outputs"][past_frame_idx] = TensorDict(
                    {
                        "pred_masks": past_out["pred_masks"],
                        "obj_ptr": past_out["obj_ptr"],
                        "object_score_logits": past_out["object_score_logits"],
                    },
                    batch_size=(B,),
                )

        return current_out

    def _use_multimask(self, is_init_cond_frame, point_inputs):
        num_pts = 0 if point_inputs is None else point_inputs["point_labels"].size(1)
        return (
            self.multimask_output_in_sam
            and (is_init_cond_frame or self.multimask_output_for_tracking)
            and (self.multimask_min_pt_num <= num_pts <= self.multimask_max_pt_num)
        )

    def _apply_non_overlapping_constraints(self, pred_masks):
        batch_size = pred_masks.size(0)
        if batch_size == 1:
            return pred_masks
        device = pred_masks.device
        max_obj_inds = torch.argmax(pred_masks, dim=0, keepdim=True)
        batch_obj_inds = torch.arange(batch_size, device=device)[:, None, None, None]
        keep = max_obj_inds == batch_obj_inds
        pred_masks = torch.where(keep, pred_masks, torch.clamp(pred_masks, max=-10.0))
        return pred_masks


def concat_points(
    old_point_inputs: Mapping[str, Tensor] | None,
    new_points: Float[Tensor, "..."],
    new_labels: Integer[Tensor, "..."] | Float[Tensor, "..."],
) -> dict[str, Tensor]:
    """Add new points and labels to previous point inputs (append at the end)."""
    if old_point_inputs is None:
        points, labels = new_points, new_labels
    else:
        points = torch.cat([old_point_inputs["point_coords"], new_points], dim=1)
        labels = torch.cat([old_point_inputs["point_labels"], new_labels], dim=1)
    return {"point_coords": points, "point_labels": labels}


# ---------------------------------------------------------------------------
# Factories for production-ish and tiny synthetic trackers
# ---------------------------------------------------------------------------


def _build_maskmem_backbone(
    in_dim: int,
    out_dim: int,
    *,
    image_size: int = 1008,
    backbone_stride: int = 14,
):
    """Build the mask-memory encoder for production or small test dimensions.

    Mask spatial size after downsampling must equal ``image_size // backbone_stride``.
    Production uses interpol→1152 then total_stride=16 → 72 (=1008/14). For tiny
    synthetic configs we set ``total_stride = backbone_stride`` with no interpol so
    ``image_size / total_stride`` matches the feature map.
    """
    # Production path (1008 / 14 = 72): interpolate to 1152, then stride 16.
    if image_size == 1008 and backbone_stride == 14:
        total_stride = 16
        interpol_size: list[int] | None = [1152, 1152]
        precompute_resolution: int | None = 1008
    else:
        if backbone_stride < 2 or backbone_stride & (backbone_stride - 1):
            raise ValueError(
                f"custom backbone_stride must be a power of two; got {backbone_stride}"
            )
        total_stride = backbone_stride
        interpol_size = None
        precompute_resolution = None

    position_encoding = PositionEmbeddingSine(
        num_pos_feats=out_dim,
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
        total_stride=total_stride,
        interpol_size=interpol_size,
    )
    fuser = SimpleFuser(
        layer=ConvNeXtBlock(
            in_chs=in_dim,
            out_chs=in_dim,
            kernel_size=7,
            ls_init_value=1.0e-06,
        ),
        num_layers=2,
    )
    return SimpleMaskEncoder(
        out_dim=out_dim,
        position_encoding=position_encoding,
        mask_downsampler=mask_downsampler,
        fuser=fuser,
        in_dim=in_dim,
    )


def build_sam3_tracker(
    *,
    image_size: int = 1008,
    backbone_stride: int = 14,
    d_model: int = 256,
    mem_dim: int = 64,
    num_maskmem: int = 7,
    num_layers: int = 4,
    multimask_output_in_sam: bool = True,
    multimask_output_for_tracking: bool = True,
    multimask_min_pt_num: int = 0,
    multimask_max_pt_num: int = 1,
    max_cond_frames_in_attn: int = 4,
    sam_num_heads: int = 8,
    sam_mlp_dim: int = 2048,
    sam_twoway_depth: int = 2,
    backbone: nn.Module | None = None,
    **tracker_kwargs: Any,
) -> Sam3TrackerBase:
    """Wire transformer + maskmem + Sam3TrackerBase (tiny or production dims)."""
    feat = image_size // backbone_stride
    # Production uses 2048; tiny trackers may pass a smaller dim_feedforward.
    dim_ff = tracker_kwargs.pop(
        "dim_feedforward", 2048 if d_model >= 256 else max(4 * d_model, 128)
    )
    transformer = create_tracker_transformer(
        d_model=d_model,
        num_layers=num_layers,
        feat_sizes=(feat, feat),
        kv_in_dim=mem_dim,
        dim_feedforward=dim_ff,
    )
    maskmem_backbone = _build_maskmem_backbone(
        in_dim=d_model,
        out_dim=mem_dim,
        image_size=image_size,
        backbone_stride=backbone_stride,
    )
    return Sam3TrackerBase(
        backbone=backbone,
        transformer=transformer,
        maskmem_backbone=maskmem_backbone,
        num_maskmem=num_maskmem,
        image_size=image_size,
        backbone_stride=backbone_stride,
        multimask_output_in_sam=multimask_output_in_sam,
        multimask_output_for_tracking=multimask_output_for_tracking,
        multimask_min_pt_num=multimask_min_pt_num,
        multimask_max_pt_num=multimask_max_pt_num,
        max_cond_frames_in_attn=max_cond_frames_in_attn,
        sam_num_heads=sam_num_heads,
        sam_mlp_dim=sam_mlp_dim,
        sam_twoway_depth=sam_twoway_depth,
        **tracker_kwargs,
    )


Sam3Tracker = Sam3TrackerBase


def build_tiny_sam3_tracker(
    *,
    image_size: int = 64,
    backbone_stride: int = 8,
    d_model: int = 64,
    mem_dim: int = 16,
    num_maskmem: int = 3,
    num_layers: int = 2,
    **kwargs: Any,
) -> Sam3TrackerBase:
    """Small-dim tracker for CUDA synthetic ``track_step`` tests."""
    # MaskDecoder / TwoWayTransformer need num_heads that divide embedding_dim.
    sam_num_heads = kwargs.pop("sam_num_heads", 4)
    return build_sam3_tracker(
        image_size=image_size,
        backbone_stride=backbone_stride,
        d_model=d_model,
        mem_dim=mem_dim,
        num_maskmem=num_maskmem,
        num_layers=num_layers,
        sam_num_heads=sam_num_heads,
        sam_mlp_dim=kwargs.pop("sam_mlp_dim", 128),
        sam_twoway_depth=kwargs.pop("sam_twoway_depth", 2),
        multimask_output_in_sam=kwargs.pop("multimask_output_in_sam", True),
        multimask_output_for_tracking=kwargs.pop("multimask_output_for_tracking", True),
        multimask_min_pt_num=kwargs.pop("multimask_min_pt_num", 0),
        multimask_max_pt_num=kwargs.pop("multimask_max_pt_num", 1),
        max_cond_frames_in_attn=kwargs.pop("max_cond_frames_in_attn", 4),
        **kwargs,
    )


def make_synthetic_vision_feats(
    tracker: Sam3TrackerBase,
    batch: int = 1,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[list[Tensor], list[Tensor], list[tuple[int, int]]]:
    """Build 3-level synthetic vision features for ``track_step``.

    High-res levels use post-``conv_s0`` / ``conv_s1`` channel counts so they can
    be fed straight into the mask decoder (same as official ``forward_image``).
    Layout is seq-first ``(HW, B, C)`` matching the official tracker.
    """
    if device is None:
        device = tracker.device
    H = tracker.sam_image_embedding_size
    C = tracker.hidden_dim
    # Level sizes: 4H, 2H, H  (matches SAM high-res + top-level)
    sizes = [(4 * H, 4 * H), (2 * H, 2 * H), (H, H)]
    # Channels after conv_s0 / conv_s1 / raw top
    chans = [C // 8, C // 4, C]
    vision_feats = []
    vision_pos = []
    for (h, w), c in zip(sizes, chans):
        feat = torch.randn(h * w, batch, c, device=device, dtype=dtype)
        pos = torch.randn(h * w, batch, c if c == C else C, device=device, dtype=dtype)
        # pos enc for high-res levels is unused in track_step memory path (only top
        # level is fused); keep same spatial tokens with C channels for consistency.
        if c != C:
            # high-res feats keep their projected channel count
            pos = torch.randn(h * w, batch, c, device=device, dtype=dtype)
        vision_feats.append(feat)
        vision_pos.append(pos)
    return vision_feats, vision_pos, sizes
