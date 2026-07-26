"""Integrated SAM image head: prompt encoder → two-way transformer → mask decoder.

This is the first end-to-end composition for migration parity. Image embeddings
are still assumed to come from an external encoder (ViTDet / etc.).
"""

from __future__ import annotations

from typing import Optional, Type

from jaxtyping import Float, Integer
import torch
from torch import Tensor, nn

from sam3.dtype_policy import module_param_dtype

from ..primitives.two_way_transformer import TwoWayTransformer
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder


class SamImageHead(nn.Module):
    """Prompted mask head assembled from vision and primitive components.

    Parameters match the common SAM image-task wiring (not the full SAM3
    multiplex detector). Defaults are small enough for CUDA unit tests.
    """

    def __init__(
        self,
        *,
        embed_dim: int = 32,
        image_embedding_size: tuple[int, int] = (8, 8),
        input_image_size: tuple[int, int] = (32, 32),
        mask_in_chans: int = 16,
        transformer_depth: int = 1,
        transformer_heads: int = 4,
        transformer_mlp_dim: int = 64,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        use_high_res_features: bool = False,
        iou_prediction_use_sigmoid: bool = False,
        pred_obj_scores: bool = False,
        pred_obj_scores_mlp: bool = False,
        use_multimask_token_for_obj_ptr: bool = False,
        dynamic_multimask_via_stability: bool = False,
        dynamic_multimask_stability_delta: float = 0.05,
        dynamic_multimask_stability_thresh: float = 0.98,
        iou_head_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if embed_dim % transformer_heads != 0:
            raise ValueError("embed_dim must be divisible by transformer_heads")

        self.embed_dim = embed_dim
        self.image_embedding_size = image_embedding_size
        self.input_image_size = input_image_size

        self.prompt_encoder = PromptEncoder(
            embed_dim=embed_dim,
            image_embedding_size=image_embedding_size,
            input_image_size=input_image_size,
            mask_in_chans=mask_in_chans,
            activation=activation,
        )
        self.transformer = TwoWayTransformer(
            depth=transformer_depth,
            embedding_dim=embed_dim,
            num_heads=transformer_heads,
            mlp_dim=transformer_mlp_dim,
            activation=nn.ReLU,
            attention_downsample_rate=2,
        )
        self.mask_decoder = MaskDecoder(
            transformer_dim=embed_dim,
            transformer=self.transformer,
            num_multimask_outputs=num_multimask_outputs,
            activation=activation,
            iou_head_depth=3,
            iou_head_hidden_dim=(
                iou_head_hidden_dim
                if iou_head_hidden_dim is not None
                else max(embed_dim, 32)
            ),
            use_high_res_features=use_high_res_features,
            iou_prediction_use_sigmoid=iou_prediction_use_sigmoid,
            pred_obj_scores=pred_obj_scores,
            pred_obj_scores_mlp=pred_obj_scores_mlp,
            use_multimask_token_for_obj_ptr=use_multimask_token_for_obj_ptr,
            dynamic_multimask_via_stability=dynamic_multimask_via_stability,
            dynamic_multimask_stability_delta=dynamic_multimask_stability_delta,
            dynamic_multimask_stability_thresh=dynamic_multimask_stability_thresh,
        )

    def forward(
        self,
        image_embeddings: Float[Tensor, "b c h w"],
        points: Optional[
            tuple[
                Float[Tensor, "b n 2"],
                Float[Tensor, "b n"] | Integer[Tensor, "b n"],
            ]
        ] = None,
        boxes: Optional[Float[Tensor, "b 4"]] = None,
        masks: Optional[Float[Tensor, "b 1 h_m w_m"]] = None,
        multimask_output: bool = True,
        repeat_image: bool = False,
        high_res_features: Optional[list[Float[Tensor, "b c_hr h_hr w_hr"]]] = None,
    ) -> tuple[
        Float[Tensor, "b n_masks h_out w_out"],
        Float[Tensor, "b n_masks"],
        Float[Tensor, "b n_tok c"],
        Float[Tensor, "b 1"],
    ]:
        """Encode prompts and decode masks.

        Args:
            image_embeddings: ``(B, C, H, W)`` image tokens (C == embed_dim).
            points: optional ``(coords, labels)`` with coords ``(B, N, 2)``.
            boxes: optional ``(B, 4)`` boxes in XYXY image coordinates.
            masks: optional dense mask inputs ``(B, 1, 4H, 4W)``.
            multimask_output: see :class:`MaskDecoder`.
            repeat_image: see :class:`MaskDecoder`.
            high_res_features: optional ``[feat_s0, feat_s1]`` for high-res
                skip connections (already projected by ``conv_s0`` / ``conv_s1``).

        Returns:
            ``(masks, iou_pred, sam_tokens, object_score_logits)``
        """
        if image_embeddings.shape[1] != self.embed_dim:
            raise ValueError(
                f"image_embeddings channels {image_embeddings.shape[1]} "
                f"!= embed_dim {self.embed_dim}"
            )
        sparse, dense = self.prompt_encoder(points=points, boxes=boxes, masks=masks)
        # PE / embeddings may be built in float32 (sin/cos + buffers). Match the
        # image activation dtype so projections and SDPA see a single dtype.
        # (Do not force param dtype: under autocast, masters may be fp32 while
        # activations are compute dtype.)
        dtype = image_embeddings.dtype
        device = image_embeddings.device
        # Permanent-cast path: if image somehow differs from weights and we are
        # not under autocast, prefer weight dtype for the whole head.
        param_dtype = module_param_dtype(self)
        if dtype != param_dtype and not torch.is_autocast_enabled():
            image_embeddings = image_embeddings.to(dtype=param_dtype)
            dtype = param_dtype
        sparse = sparse.to(device=device, dtype=dtype)
        dense = dense.to(device=device, dtype=dtype)
        image_pe = self.prompt_encoder.get_dense_pe().to(device=device, dtype=dtype)
        if high_res_features is not None:
            high_res_features = [
                f.to(device=device, dtype=dtype) for f in high_res_features
            ]
        return self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=multimask_output,
            repeat_image=repeat_image,
            high_res_features=high_res_features,
        )
