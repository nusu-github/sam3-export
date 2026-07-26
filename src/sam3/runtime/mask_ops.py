"""Mask geometry helpers for runtime postprocessing and association."""

from __future__ import annotations

from collections.abc import Sequence

from jaxtyping import Bool, Float
import torch
from torch import Tensor
import torch.nn.functional as F


def mask_iou(
    masks1: Bool[Tensor, "n h w"] | Float[Tensor, "n h w"],
    masks2: Bool[Tensor, "m h w"] | Float[Tensor, "m h w"],
) -> Float[Tensor, "n m"]:
    """Pairwise IoU for binary/float masks.

    Uses flattened matmul (cuBLAS) rather than an explicit N×M×HW loop.

    Args:
        masks1: ``(N, H, W)``
        masks2: ``(M, H, W)``
    Returns:
        ``(N, M)`` IoU matrix.
    """
    # Bool/float → float01; matmul is the library-backed path (cuBLAS).
    m1 = (masks1.flatten(1) > 0.5).to(dtype=torch.float32)
    m2 = (masks2.flatten(1) > 0.5).to(dtype=torch.float32)
    inter = m1 @ m2.t()
    area1 = m1.sum(dim=1, keepdim=True)
    area2 = m2.sum(dim=1, keepdim=True)
    union = area1 + area2.t() - inter
    return inter / union.clamp(min=1.0)


def resize_masks(
    masks: Float[Tensor, "n h w"],
    size: tuple[int, int],
) -> Float[Tensor, "n out_h out_w"]:
    """Bilinear resize ``(N,H,W)`` float masks to ``size``."""

    if masks.numel() == 0:
        return torch.zeros(
            (masks.shape[0], *size), device=masks.device, dtype=masks.dtype
        )
    return F.interpolate(
        masks.unsqueeze(1).float(),
        size=size,
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)


def masks_to_boxes(
    masks: Tensor,
    obj_ids: Sequence[int] | None = None,
) -> Float[Tensor, "n 4"]:
    """Compute one (x_min, y_min, x_max, y_max) box per binary mask.

    Empty masks map to zeros.
    """
    if masks.ndim != 3:
        raise ValueError(f"masks must be (N, H, W); received {tuple(masks.shape)}")

    if obj_ids is not None and len(obj_ids) != masks.shape[0]:
        raise ValueError(
            "obj_ids must have one entry per mask; "
            f"received {len(obj_ids)} ids for {masks.shape[0]} masks"
        )

    batch, height, width = masks.shape
    if batch == 0 or height == 0 or width == 0:
        return torch.zeros((batch, 4), dtype=torch.float32, device=masks.device)

    # Match sam3.perflib.masks_ops: nonzero = foreground; xyxy with *inclusive* max.
    column_indices = torch.arange(width, device=masks.device).view(1, width)
    row_indices = torch.arange(height, device=masks.device).view(1, height)

    foreground = masks != 0
    occupied_columns = foreground.amax(dim=1)
    occupied_rows = foreground.amax(dim=2)

    x_min = torch.amin(
        (~occupied_columns * width) + (occupied_columns * column_indices),
        dim=1,
    )
    y_min = torch.amin(
        (~occupied_rows * height) + (occupied_rows * row_indices),
        dim=1,
    )
    x_max = torch.amax(occupied_columns * column_indices, dim=1)
    y_max = torch.amax(occupied_rows * row_indices, dim=1)

    boxes = torch.stack((x_min, y_min, x_max, y_max), dim=1).to(dtype=torch.float32)
    empty = ~foreground.flatten(1).any(dim=1)
    boxes = boxes * (~empty).to(dtype=boxes.dtype).unsqueeze(1)
    return boxes
