"""Tests for det↔track association."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")


def test_associate_empty():
    from sam3.runtime.associate_det_trk import associate_det_trk

    det = torch.zeros(0, 8, 8, device="cuda")
    trk = torch.zeros(0, 8, 8, device="cuda")
    new, unmatched, d2t, scores = associate_det_trk(det, trk)
    assert new == []
    assert unmatched == []


def test_associate_all_new():
    from sam3.runtime.associate_det_trk import associate_det_trk

    det = torch.zeros(2, 16, 16, device="cuda")
    det[0, 2:6, 2:6] = 1
    det[1, 10:14, 10:14] = 1
    trk = torch.zeros(0, 16, 16, device="cuda")
    new, unmatched, _, _ = associate_det_trk(det, trk, new_det_thresh=0.0)
    assert new == [0, 1]


def test_associate_match():
    from sam3.runtime.associate_det_trk import associate_det_trk

    det = torch.zeros(1, 16, 16, device="cuda")
    det[0, 2:8, 2:8] = 1
    trk = torch.zeros(1, 16, 16, device="cuda")
    trk[0, 3:9, 3:9] = 1  # high overlap
    scores = torch.tensor([0.9], device="cuda")
    new, unmatched, d2t, _ = associate_det_trk(
        det, trk, iou_threshold=0.1, iou_threshold_trk=0.1, det_scores=scores
    )
    assert new == []  # matched, not new
    assert 0 in d2t
