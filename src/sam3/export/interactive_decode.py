"""Cut E — InteractiveDecode (image embed + fixed points → multimask)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from torch import Tensor
import torch.nn as nn

from sam3.export.prompt_encode import (
    PROMPT_EMBED_DIM,
    PROMPT_IMAGE_EMBED_HW,
    PROMPT_INPUT_IMAGE_HW,
    PROMPT_N_POINTS,
    PromptEncodeSpec,
)
from sam3.vision.sam_image_head import SamImageHead

INTERACTIVE_NUM_MASKS: Final[int] = 3  # multimask_output=True default


@dataclass(frozen=True)
class InteractiveDecodeSpec:
    embed_dim: int = PROMPT_EMBED_DIM
    image_embedding_size: tuple[int, int] = PROMPT_IMAGE_EMBED_HW
    input_image_size: tuple[int, int] = PROMPT_INPUT_IMAGE_HW
    n_points: int = PROMPT_N_POINTS
    num_masks: int = INTERACTIVE_NUM_MASKS
    transformer_depth: int = 1
    transformer_heads: int = 4
    transformer_mlp_dim: int = 64


def validate_interactive_decode_input(
    image_embeddings: Tensor,
    point_coords: Tensor,
    point_labels: Tensor,
    spec: InteractiveDecodeSpec = InteractiveDecodeSpec(),
) -> None:
    if image_embeddings.ndim != 4:
        raise ValueError(
            f"image_embeddings must be [B,C,H,W], got {tuple(image_embeddings.shape)}"
        )
    b, c, h, w = image_embeddings.shape
    eh, ew = spec.image_embedding_size
    if c != spec.embed_dim or h != eh or w != ew:
        raise ValueError(
            f"image_embeddings expected {(b, spec.embed_dim, eh, ew)} channels/spatial, "
            f"got {tuple(image_embeddings.shape)}"
        )
    if point_coords.shape != (b, spec.n_points, 2):
        raise ValueError(
            f"point_coords expected {(b, spec.n_points, 2)}, got {tuple(point_coords.shape)}"
        )
    if point_labels.shape != (b, spec.n_points):
        raise ValueError(
            f"point_labels expected {(b, spec.n_points)}, got {tuple(point_labels.shape)}"
        )


def validate_interactive_decode_output(
    masks: Tensor,
    iou: Tensor,
    *,
    batch: int,
    spec: InteractiveDecodeSpec = InteractiveDecodeSpec(),
) -> None:
    # MaskDecoder upsamples 4x from embedding spatial → input_image_size for tiny head
    oh, ow = spec.input_image_size
    if masks.shape[0] != batch or masks.shape[1] != spec.num_masks:
        raise ValueError(f"masks unexpected shape {tuple(masks.shape)}")
    if masks.shape[-2:] != (oh, ow):
        raise ValueError(
            f"masks spatial expected {(oh, ow)}, got {tuple(masks.shape[-2:])}"
        )
    if iou.shape != (batch, spec.num_masks):
        raise ValueError(
            f"iou expected {(batch, spec.num_masks)}, got {tuple(iou.shape)}"
        )


class InteractiveDecode(nn.Module):
    """Fixed-shape click path: image tokens + points → multimask logits + IoU."""

    def __init__(
        self,
        head: SamImageHead | None = None,
        spec: InteractiveDecodeSpec | None = None,
        *,
        validate: bool = True,
    ) -> None:
        super().__init__()
        self.spec = spec or InteractiveDecodeSpec()
        if head is None:
            head = SamImageHead(
                embed_dim=self.spec.embed_dim,
                image_embedding_size=self.spec.image_embedding_size,
                input_image_size=self.spec.input_image_size,
                transformer_depth=self.spec.transformer_depth,
                transformer_heads=self.spec.transformer_heads,
                transformer_mlp_dim=self.spec.transformer_mlp_dim,
                num_multimask_outputs=self.spec.num_masks,
                use_high_res_features=False,
            )
        self.head = head
        self.validate = bool(validate)
        self.prompt_spec = PromptEncodeSpec(
            embed_dim=self.spec.embed_dim,
            image_embedding_size=self.spec.image_embedding_size,
            input_image_size=self.spec.input_image_size,
            n_points=self.spec.n_points,
        )

    def forward(
        self,
        image_embeddings: Tensor,
        point_coords: Tensor,
        point_labels: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if self.validate:
            validate_interactive_decode_input(
                image_embeddings, point_coords, point_labels, self.spec
            )
        masks, iou, _tokens, _obj = self.head(
            image_embeddings,
            points=(point_coords, point_labels),
            boxes=None,
            masks=None,
            multimask_output=True,
            repeat_image=False,
            high_res_features=None,
        )
        if self.validate:
            validate_interactive_decode_output(
                masks,
                iou,
                batch=int(image_embeddings.shape[0]),
                spec=self.spec,
            )
        return masks, iou


__all__ = [
    "INTERACTIVE_NUM_MASKS",
    "InteractiveDecodeSpec",
    "InteractiveDecode",
    "validate_interactive_decode_input",
    "validate_interactive_decode_output",
]
