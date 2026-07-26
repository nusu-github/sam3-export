"""Mask NMS for the host-side runtime.

Matches upstream `sam3.perflib.nms.nms_masks` behavior: apply score filtering
first, then IoU-based suppression on the surviving masks.
"""

from __future__ import annotations

from jaxtyping import Bool, Float
import torch
from torch import Tensor

from .mask_ops import mask_iou


def _generic_nms_cpu(
    ious: Tensor,
    scores: Tensor,
    iou_threshold: float,
) -> Tensor:
    if ious.numel() == 0:
        return torch.empty((0,), device=scores.device, dtype=torch.long)

    order = torch.argsort(scores, descending=True)
    keep: list[int] = []
    while order.numel() > 0:
        current = order[0].item()
        keep.append(current)
        if order.numel() == 1:
            break
        rest = order[1:]
        keep_mask = ious[current, rest] <= iou_threshold
        order = rest[keep_mask]
    return torch.tensor(keep, device=scores.device, dtype=torch.long)


def _nms_masks_torch(
    pred_probs: Float[Tensor, "n"],
    pred_masks: Float[Tensor, "n h w"],
    prob_threshold: float,
    iou_threshold: float,
) -> Bool[Tensor, "n"]:
    is_valid = pred_probs > prob_threshold
    probs = pred_probs[is_valid]
    masks_binary = pred_masks[is_valid] > 0

    if probs.numel() == 0:
        return is_valid

    ious = mask_iou(masks_binary, masks_binary)
    kept_inds = _generic_nms_cpu(ious, probs, iou_threshold)

    valid_inds = torch.nonzero(is_valid, as_tuple=False).squeeze(1)
    keep = torch.zeros_like(is_valid, dtype=torch.bool)
    if kept_inds.numel() > 0:
        keep[valid_inds[kept_inds]] = True
    return keep


def nms_masks(
    pred_probs: Float[Tensor, "n"],
    pred_masks: Float[Tensor, "n h w"],
    prob_threshold: float,
    iou_threshold: float,
) -> Bool[Tensor, "n"]:
    """Filter and suppress mask detections.

    Args:
        pred_probs: (N,) scores.
        pred_masks: (N, H, W) raw mask logits / values.
        prob_threshold: score threshold applied before NMS.
        iou_threshold: IoU threshold for suppression.

    Returns:
        bool mask of shape (N,) where True indicates kept detections.
    """
    if pred_probs.dim() != 1:
        raise ValueError(f"pred_probs must be 1D, got {tuple(pred_probs.shape)}")
    if pred_masks.dim() != 3:
        raise ValueError(f"pred_masks must be 3D, got {tuple(pred_masks.shape)}")
    if pred_masks.shape[0] != pred_probs.shape[0]:
        raise ValueError(
            "pred_masks and pred_probs must agree on proposal count, "
            f"got pred_masks={pred_masks.shape[0]} pred_probs={pred_probs.shape[0]}"
        )
    if pred_masks.device != pred_probs.device:
        raise ValueError("pred_masks and pred_probs must be on the same device")

    return _nms_masks_torch(pred_probs, pred_masks, prob_threshold, iou_threshold)
