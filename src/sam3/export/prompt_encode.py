"""Cut D — fixed-shape PromptEncode for torch.export."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from torch import Tensor
import torch.nn as nn

from sam3.vision.prompt_encoder import PromptEncoder

# Tiny-head defaults used by unit tests / export smoke (not production 256-d).
PROMPT_EMBED_DIM: Final[int] = 32
PROMPT_IMAGE_EMBED_HW: Final[tuple[int, int]] = (8, 8)
PROMPT_INPUT_IMAGE_HW: Final[tuple[int, int]] = (32, 32)
PROMPT_N_POINTS: Final[int] = 2
# When boxes is None, PromptEncoder pads one extra point → N+1 sparse tokens.
PROMPT_N_SPARSE: Final[int] = PROMPT_N_POINTS + 1


@dataclass(frozen=True)
class PromptEncodeSpec:
    embed_dim: int = PROMPT_EMBED_DIM
    image_embedding_size: tuple[int, int] = PROMPT_IMAGE_EMBED_HW
    input_image_size: tuple[int, int] = PROMPT_INPUT_IMAGE_HW
    n_points: int = PROMPT_N_POINTS
    mask_in_chans: int = 16


def validate_prompt_encode_input(
    point_coords: Tensor,
    point_labels: Tensor,
    spec: PromptEncodeSpec = PromptEncodeSpec(),
) -> None:
    if point_coords.ndim != 3 or point_coords.shape[-1] != 2:
        raise ValueError(
            f"point_coords must be [B,N,2], got {tuple(point_coords.shape)}"
        )
    if point_labels.ndim != 2:
        raise ValueError(f"point_labels must be [B,N], got {tuple(point_labels.shape)}")
    if point_coords.shape[0] != point_labels.shape[0]:
        raise ValueError("batch mismatch between coords and labels")
    if point_coords.shape[1] != spec.n_points or point_labels.shape[1] != spec.n_points:
        raise ValueError(
            f"expected N={spec.n_points} points, got coords N={point_coords.shape[1]} "
            f"labels N={point_labels.shape[1]}"
        )


def validate_prompt_encode_output(
    sparse: Tensor,
    dense: Tensor,
    *,
    batch: int,
    spec: PromptEncodeSpec = PromptEncodeSpec(),
) -> None:
    eh, ew = spec.image_embedding_size
    if sparse.shape != (batch, PROMPT_N_SPARSE, spec.embed_dim):
        # Allow encoder pad rule: n_points or n_points+1 depending on pad path
        if (
            sparse.ndim != 3
            or sparse.shape[0] != batch
            or sparse.shape[2] != spec.embed_dim
        ):
            raise ValueError(f"sparse unexpected shape {tuple(sparse.shape)}")
    if dense.shape != (batch, spec.embed_dim, eh, ew):
        raise ValueError(
            f"dense must be {(batch, spec.embed_dim, eh, ew)}, got {tuple(dense.shape)}"
        )


class PromptEncode(nn.Module):
    """Tensor-only prompt encoding with fixed point count (export cut D)."""

    def __init__(
        self,
        encoder: PromptEncoder | None = None,
        spec: PromptEncodeSpec | None = None,
        *,
        validate: bool = True,
    ) -> None:
        super().__init__()
        self.spec = spec or PromptEncodeSpec()
        if encoder is None:
            encoder = PromptEncoder(
                embed_dim=self.spec.embed_dim,
                image_embedding_size=self.spec.image_embedding_size,
                input_image_size=self.spec.input_image_size,
                mask_in_chans=self.spec.mask_in_chans,
            )
        self.encoder = encoder
        self.validate = bool(validate)

    def forward(
        self,
        point_coords: Tensor,
        point_labels: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            point_coords: ``[B, N, 2]`` in model-frame absolute coords
            point_labels: ``[B, N]`` int labels (-1/0/1/…)
        Returns:
            sparse ``[B, N_sparse, C]``, dense ``[B, C, H, W]``
        """
        if self.validate:
            validate_prompt_encode_input(point_coords, point_labels, self.spec)
        sparse, dense = self.encoder(
            points=(point_coords, point_labels),
            boxes=None,
            masks=None,
        )
        if self.validate:
            validate_prompt_encode_output(
                sparse,
                dense,
                batch=int(point_coords.shape[0]),
                spec=self.spec,
            )
        return sparse, dense


__all__ = [
    "PROMPT_EMBED_DIM",
    "PROMPT_IMAGE_EMBED_HW",
    "PROMPT_INPUT_IMAGE_HW",
    "PROMPT_N_POINTS",
    "PROMPT_N_SPARSE",
    "PromptEncodeSpec",
    "PromptEncode",
    "validate_prompt_encode_input",
    "validate_prompt_encode_output",
]
