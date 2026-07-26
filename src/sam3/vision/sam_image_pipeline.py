"""Full interactive-image path: ViT encoder → SAM image head.

Tiny defaults are sized for CUDA parity / smoke tests. Production SAM3 uses
much larger ``embed_dim`` / depth; swap kwargs only.
"""

from __future__ import annotations

from typing import Type

from jaxtyping import Float, Integer
from torch import Tensor, nn

from .sam_image_head import SamImageHead
from .vit import ViT


class SamImagePipeline(nn.Module):
    """End-to-end image → masks stack from sam3 building blocks.

    Flow::

        image (B,3,H,W)
            → ViT → image_embeddings (B,C,h,w)
            → SamImageHead(prompts) → masks, iou, tokens, obj_scores
    """

    def __init__(
        self,
        *,
        img_size: int = 64,
        patch_size: int = 16,
        embed_dim: int = 32,
        vit_depth: int = 2,
        vit_heads: int = 4,
        vit_window_size: int = 2,
        vit_mlp_ratio: float | int = 4.0,
        use_rope: bool = False,
        use_rope_real: bool = False,
        transformer_depth: int = 1,
        transformer_heads: int = 4,
        transformer_mlp_dim: int = 64,
        mask_in_chans: int = 16,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError("img_size must be divisible by patch_size")
        if embed_dim % vit_heads != 0:
            raise ValueError("embed_dim must be divisible by vit_heads")
        if embed_dim % transformer_heads != 0:
            raise ValueError("embed_dim must be divisible by transformer_heads")
        # TwoWayAttention uses downsample_rate=2 → internal_dim must be multiple of heads.
        if (embed_dim // 2) % transformer_heads != 0:
            raise ValueError(
                "embed_dim//2 must be divisible by transformer_heads "
                "(attention_downsample_rate=2)"
            )

        grid = img_size // patch_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.image_embedding_size = (grid, grid)

        # Match pretrain pos size to runtime grid to avoid awkward cls-token abs pos.
        self.image_encoder = ViT(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            embed_dim=embed_dim,
            depth=vit_depth,
            num_heads=vit_heads,
            mlp_ratio=vit_mlp_ratio,
            window_size=vit_window_size,
            use_abs_pos=True,
            tile_abs_pos=False,
            use_rope=use_rope,
            use_rope_real=use_rope_real,
            pretrain_img_size=img_size,
            pretrain_use_cls_token=False,
            retain_cls_token=False,
            ln_pre=False,
            ln_post=True,
        )
        self.mask_head = SamImageHead(
            embed_dim=embed_dim,
            image_embedding_size=self.image_embedding_size,
            input_image_size=(img_size, img_size),
            mask_in_chans=mask_in_chans,
            transformer_depth=transformer_depth,
            transformer_heads=transformer_heads,
            transformer_mlp_dim=transformer_mlp_dim,
            num_multimask_outputs=num_multimask_outputs,
            activation=activation,
        )

    def encode_image(
        self, image: Float[Tensor, "b 3 h w"]
    ) -> Float[Tensor, "b c h_e w_e"]:
        """Return final encoder feature map ``(B, C, h, w)``."""
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"expected image (B,3,H,W), got {tuple(image.shape)}")
        if image.shape[-2] != self.img_size or image.shape[-1] != self.img_size:
            raise ValueError(
                f"expected spatial size {(self.img_size, self.img_size)}, "
                f"got {tuple(image.shape[-2:])}"
            )
        feats = self.image_encoder(image)
        if not feats:
            raise RuntimeError("ViT returned no feature maps")
        return feats[-1]

    def forward(
        self,
        image: Float[Tensor, "b 3 h w"],
        points: tuple[
            Float[Tensor, "b n 2"],
            Float[Tensor, "b n"] | Integer[Tensor, "b n"],
        ]
        | None = None,
        boxes: Float[Tensor, "b 4"] | None = None,
        masks: Float[Tensor, "b 1 h_m w_m"] | None = None,
        multimask_output: bool = True,
    ) -> tuple[
        Float[Tensor, "b n_masks h_out w_out"],
        Float[Tensor, "b n_masks"],
        Float[Tensor, "b n_tok c"],
        Float[Tensor, "b 1"],
    ]:
        """Image + prompts → masks.

        Returns:
            ``(masks, iou_pred, sam_tokens, object_score_logits)``
        """
        image_embeddings = self.encode_image(image)
        return self.mask_head(
            image_embeddings,
            points=points,
            boxes=boxes,
            masks=masks,
            multimask_output=multimask_output,
            repeat_image=False,
        )
