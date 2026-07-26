"""Mask decoder used by interactive image inference."""

from __future__ import annotations

from typing import Optional, Type

from jaxtyping import Float
from timm.layers import LayerNorm2d
import torch
from torch import Tensor, nn

from ..primitives.mlp import MLP


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        use_high_res_features: bool = False,
        iou_prediction_use_sigmoid: bool = False,
        dynamic_multimask_via_stability: bool = False,
        dynamic_multimask_stability_delta: float = 0.05,
        dynamic_multimask_stability_thresh: float = 0.98,
        pred_obj_scores: bool = False,
        pred_obj_scores_mlp: bool = False,
        use_multimask_token_for_obj_ptr: bool = False,
    ) -> None:
        """Predict masks from image embeddings and prompt embeddings using a transformer."""
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.pred_obj_scores = pred_obj_scores
        if self.pred_obj_scores:
            self.obj_score_token = nn.Embedding(1, transformer_dim)
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            ),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(
                transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
            ),
            activation(),
        )
        self.use_high_res_features = use_high_res_features
        if use_high_res_features:
            self.conv_s0 = nn.Conv2d(
                transformer_dim, transformer_dim // 8, kernel_size=1, stride=1
            )
            self.conv_s1 = nn.Conv2d(
                transformer_dim, transformer_dim // 4, kernel_size=1, stride=1
            )

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for _ in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim,
            iou_head_hidden_dim,
            self.num_mask_tokens,
            iou_head_depth,
            sigmoid_output=iou_prediction_use_sigmoid,
        )
        if self.pred_obj_scores:
            self.pred_obj_score_head = nn.Linear(transformer_dim, 1)
            if pred_obj_scores_mlp:
                self.pred_obj_score_head = MLP(transformer_dim, transformer_dim, 1, 3)

        self.dynamic_multimask_via_stability = dynamic_multimask_via_stability
        self.dynamic_multimask_stability_delta = dynamic_multimask_stability_delta
        self.dynamic_multimask_stability_thresh = dynamic_multimask_stability_thresh

    def forward(
        self,
        image_embeddings: Float[Tensor, "b c h w"],
        image_pe: Float[Tensor, "b_pe c h w"],
        sparse_prompt_embeddings: Float[Tensor, "b n c"],
        dense_prompt_embeddings: Float[Tensor, "b c h w"],
        multimask_output: bool,
        repeat_image: bool,
        high_res_features: Optional[list[Float[Tensor, "b c_hr h_hr w_hr"]]] = None,
        sparse_prompt_valid: Tensor | None = None,
    ) -> tuple[
        Float[Tensor, "b n_masks h_out w_out"],
        Float[Tensor, "b n_masks"],
        Float[Tensor, "b n_tok c"],
        Float[Tensor, "b 1"],
    ]:
        masks, iou_pred, mask_tokens_out, object_score_logits = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            repeat_image=repeat_image,
            high_res_features=high_res_features,
            sparse_prompt_valid=sparse_prompt_valid,
        )

        if multimask_output:
            masks = masks[:, 1:, :, :]
            iou_pred = iou_pred[:, 1:]
        elif self.dynamic_multimask_via_stability and not self.training:
            masks, iou_pred = self._dynamic_multimask_via_stability(masks, iou_pred)
        else:
            masks = masks[:, 0:1, :, :]
            iou_pred = iou_pred[:, 0:1]

        if multimask_output and self.use_multimask_token_for_obj_ptr:
            sam_tokens_out = mask_tokens_out[:, 1:]
        else:
            sam_tokens_out = mask_tokens_out[:, 0:1]

        return masks, iou_pred, sam_tokens_out, object_score_logits

    def predict_masks(
        self,
        image_embeddings: Float[Tensor, "b_img c h w"],
        image_pe: Float[Tensor, "b_pe c h w"],
        sparse_prompt_embeddings: Float[Tensor, "b n c"],
        dense_prompt_embeddings: Float[Tensor, "b c h w"],
        repeat_image: bool,
        high_res_features: Optional[list[Float[Tensor, "b c_hr h_hr w_hr"]]] = None,
        sparse_prompt_valid: Tensor | None = None,
    ) -> tuple[
        Float[Tensor, "b n_masks h_out w_out"],
        Float[Tensor, "b n_masks"],
        Float[Tensor, "b n_mask_tokens c"],
        Float[Tensor, "b 1"],
    ]:
        """Predict raw masks and IoU/object-score logits before final output slicing."""
        s = 0
        if self.pred_obj_scores:
            output_tokens = torch.cat(
                [
                    self.obj_score_token.weight,
                    self.iou_token.weight,
                    self.mask_tokens.weight,
                ],
                dim=0,
            )
            s = 1
        else:
            output_tokens = torch.cat(
                [self.iou_token.weight, self.mask_tokens.weight], dim=0
            )

        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        token_valid = None
        if sparse_prompt_valid is not None:
            output_valid = torch.ones(
                (sparse_prompt_valid.shape[0], output_tokens.shape[1]),
                dtype=torch.bool,
                device=sparse_prompt_valid.device,
            )
            token_valid = torch.cat((output_valid, sparse_prompt_valid), dim=1)

        if repeat_image:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            if image_embeddings.shape[0] != tokens.shape[0]:
                raise AssertionError(
                    "image_embeddings and tokens batch dimensions must align when repeat_image is False"
                )
            src = image_embeddings
        src = src + dense_prompt_embeddings

        if image_pe.size(0) != 1:
            raise AssertionError(
                "image_pe should have size 1 in batch dim (from `get_dense_pe()`)"
            )
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)

        b, c, h, w = src.shape
        if token_valid is None:
            hs, src = self.transformer(src, pos_src, tokens)
        else:
            hs, src = self.transformer(src, pos_src, tokens, point_valid=token_valid)
        iou_token_out = hs[:, s, :]
        mask_tokens_out = hs[:, s + 1 : s + 1 + self.num_mask_tokens, :]

        src = src.transpose(1, 2).view(b, c, h, w)
        if not self.use_high_res_features:
            upscaled_embedding = self.output_upscaling(src)
        else:
            dc1, ln1, act1, dc2, act2 = self.output_upscaling
            if high_res_features is None:
                raise ValueError(
                    "high_res_features must be provided when use_high_res_features=True"
                )
            feat_s0, feat_s1 = high_res_features
            upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
            upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)

        _, c, h, w = upscaled_embedding.shape
        hyper_in = torch.stack(
            [
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
                for i in range(self.num_mask_tokens)
            ],
            dim=1,
        )

        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        iou_pred = self.iou_prediction_head(iou_token_out)

        if self.pred_obj_scores:
            object_score_logits = self.pred_obj_score_head(hs[:, 0, :])
        else:
            object_score_logits = 10.0 * iou_pred.new_ones(iou_pred.shape[0], 1)

        return masks, iou_pred, mask_tokens_out, object_score_logits

    def _get_stability_scores(
        self, mask_logits: Float[Tensor, "b n h w"]
    ) -> Float[Tensor, "b n"]:
        mask_logits = mask_logits.flatten(-2)
        stability_delta = self.dynamic_multimask_stability_delta
        area_i = torch.sum(mask_logits > stability_delta, dim=-1).float()
        area_u = torch.sum(mask_logits > -stability_delta, dim=-1).float()
        return torch.where(
            area_u > 0, area_i / area_u, torch.tensor(1.0, device=mask_logits.device)
        )

    def _dynamic_multimask_via_stability(
        self,
        all_mask_logits: Float[Tensor, "b n_all h w"],
        all_iou_scores: Float[Tensor, "b n_all"],
    ) -> tuple[Float[Tensor, "b 1 h w"], Float[Tensor, "b 1"]]:
        multimask_logits = all_mask_logits[:, 1:, :, :]
        multimask_iou_scores = all_iou_scores[:, 1:]
        best_scores_inds = torch.argmax(multimask_iou_scores, dim=-1, keepdim=True)
        best_multimask_iou_scores = torch.gather(
            multimask_iou_scores, 1, best_scores_inds
        )
        mask_indices = best_scores_inds[:, :, None, None].expand(
            -1, -1, multimask_logits.shape[-2], multimask_logits.shape[-1]
        )
        best_multimask_logits = torch.gather(multimask_logits, 1, mask_indices)

        singlemask_logits = all_mask_logits[:, 0:1, :, :]
        singlemask_iou_scores = all_iou_scores[:, 0:1]
        stability_scores = self._get_stability_scores(singlemask_logits)
        is_stable = stability_scores >= self.dynamic_multimask_stability_thresh

        mask_logits_out = torch.where(
            is_stable[..., None, None].expand_as(singlemask_logits),
            singlemask_logits,
            best_multimask_logits,
        )
        iou_scores_out = torch.where(
            is_stable.expand_as(singlemask_iou_scores),
            singlemask_iou_scores,
            best_multimask_iou_scores,
        )
        return mask_logits_out, iou_scores_out
