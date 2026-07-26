"""Runtime detection-to-track association by mask IoU.

Port of ``sam3.perflib.associate_det_trk.associate_det_trk`` for single-GPU use.
"""

from __future__ import annotations

from collections import defaultdict

from jaxtyping import Bool, Float
from scipy.optimize import linear_sum_assignment
from torch import Tensor
import torch.nn.functional as F

from .mask_ops import mask_iou

AssociationResult = tuple[
    list[int],
    list[int],
    dict[int, list[int]],
    dict[int, list[float]],
]


def associate_det_trk(
    det_masks: Bool[Tensor, "n h w"] | Float[Tensor, "n h w"],
    track_masks: Bool[Tensor, "m h w"] | Float[Tensor, "m h w"],
    iou_threshold: float = 0.5,
    iou_threshold_trk: float = 0.5,
    det_scores: Float[Tensor, "n"] | None = None,
    new_det_thresh: float = 0.0,
) -> AssociationResult:
    """Associate detections with existing tracks.

    Returns:
        new_det_indices, unmatched_trk_indices, det_to_matched_trk, matched_det_scores
    """
    if det_masks.size(0) == 0 or track_masks.size(0) == 0:
        return list(range(det_masks.size(0))), [], {}, {}

    if det_masks.shape[-2:] != track_masks.shape[-2:]:
        if det_masks.shape[-2] * det_masks.shape[-1] < (
            track_masks.shape[-2] * track_masks.shape[-1]
        ):
            track_masks = (
                F.interpolate(
                    track_masks.unsqueeze(1).float(),
                    size=det_masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
                > 0
            )
        else:
            det_masks = (
                F.interpolate(
                    det_masks.unsqueeze(1).float(),
                    size=track_masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
                > 0
            )

    det_masks = det_masks > 0
    track_masks = track_masks > 0
    iou = mask_iou(det_masks.float(), track_masks.float())  # (N, M)

    cost = (1.0 - iou).detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost)

    det_scores_list = (
        [1.0] * det_masks.size(0)
        if det_scores is None
        else det_scores.detach().float().cpu().tolist()
    )
    iou_list = iou.detach().cpu().tolist()
    igeit = (iou >= iou_threshold).cpu().tolist()
    igeit_any = (iou >= iou_threshold).any(dim=1).cpu().tolist()
    igeit_trk = (iou >= iou_threshold_trk).cpu().tolist()

    matched_trk: set[int] = set()
    matched_det_scores: dict[int, list[float]] = {}
    for d, t in zip(row_ind.tolist(), col_ind.tolist()):
        matched_det_scores[t] = [
            float(det_scores_list[d]),
            float(det_scores_list[d] * iou_list[d][t]),
        ]
        if igeit_trk[d][t]:
            matched_trk.add(t)

    unmatched_trk = [t for t in range(track_masks.size(0)) if t not in matched_trk]
    new_det = [
        d
        for d in range(det_masks.size(0))
        if (not igeit_any[d]) and det_scores_list[d] >= new_det_thresh
    ]
    det_to_matched: dict[int, list[int]] = defaultdict(list)
    for d in range(det_masks.size(0)):
        for t in range(track_masks.size(0)):
            if igeit[d][t]:
                det_to_matched[d].append(t)

    return new_det, unmatched_trk, dict(det_to_matched), matched_det_scores
