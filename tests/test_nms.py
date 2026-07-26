"""Parity and behavior tests for mask NMS."""

from __future__ import annotations

import pytest
import torch

from sam3.runtime.nms import nms_masks

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for NMS tests",
)

DEVICE = torch.device("cuda")


def test_nms_empty_inputs() -> None:
    pred_probs = torch.empty((0,), device=DEVICE, dtype=torch.float32)
    pred_masks = torch.empty((0, 2, 2), device=DEVICE, dtype=torch.float32)

    keep = nms_masks(pred_probs, pred_masks, prob_threshold=0.5, iou_threshold=0.5)
    assert keep.shape == pred_probs.shape
    assert keep.dtype == torch.bool
    assert not keep.any()


def test_nms_single() -> None:
    pred_probs = torch.tensor([0.9], device=DEVICE)
    pred_masks = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]],
        device=DEVICE,
    )
    keep = nms_masks(pred_probs, pred_masks[0:1], prob_threshold=0.5, iou_threshold=0.5)
    assert keep.shape == (1,)
    assert torch.equal(keep, torch.tensor([True], device=DEVICE))


def test_nms_overlap_keep_highest() -> None:
    pred_probs = torch.tensor([0.95, 0.80], device=DEVICE)
    pred_masks = torch.tensor(
        [
            [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ],
        device=DEVICE,
    )

    keep = nms_masks(pred_probs, pred_masks, prob_threshold=0.5, iou_threshold=0.5)
    assert torch.equal(keep, torch.tensor([True, False], device=DEVICE))


def test_nms_threshold_filter() -> None:
    pred_probs = torch.tensor([0.2, 0.9], device=DEVICE)
    pred_masks = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 0.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ],
        device=DEVICE,
    )

    keep = nms_masks(pred_probs, pred_masks, prob_threshold=0.5, iou_threshold=0.5)
    assert torch.equal(keep, torch.tensor([False, True], device=DEVICE))


def test_nms_all_below_threshold() -> None:
    pred_probs = torch.tensor([0.1, 0.2], device=DEVICE)
    pred_masks = torch.tensor(
        [
            [[1.0, 1.0], [1.0, 1.0]],
            [[1.0, 1.0], [1.0, 1.0]],
        ],
        device=DEVICE,
    )

    keep = nms_masks(
        pred_probs,
        pred_masks,
        prob_threshold=0.5,
        iou_threshold=0.5,
    )
    assert torch.equal(keep, torch.tensor([False, False], device=DEVICE))


def test_nms_cpu() -> None:
    pred_probs = torch.tensor([0.95, 0.8], device="cpu")
    pred_masks = torch.tensor(
        [
            [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ],
        device="cpu",
    )

    keep = nms_masks(pred_probs, pred_masks, prob_threshold=0.5, iou_threshold=0.5)
    assert keep.device == pred_probs.device
    assert torch.equal(keep, torch.tensor([True, False]))
