"""Tests for connected components and mask->box conversion."""

from __future__ import annotations

import torch

from sam3.runtime import mask_ops
from sam3.runtime.connected_components import connected_components


def test_connected_components_empty() -> None:
    masks = torch.zeros((2, 3, 4), dtype=torch.int32)
    labels, counts = connected_components(masks)
    assert labels.shape == masks.shape
    assert counts.shape == masks.shape
    torch.testing.assert_close(labels, torch.zeros_like(labels))
    torch.testing.assert_close(counts, torch.zeros_like(counts))


def test_connected_components_single_blob() -> None:
    masks = torch.zeros((1, 4, 6), dtype=torch.int32)
    masks[0, 1:3, 1:4] = 7

    labels, counts = connected_components(masks)
    lbl = labels[0]
    cnt = counts[0]
    assert int((lbl[masks[0] == 0] != 0).sum()) == 0
    assert int((lbl[masks[0] != 0] > 0).sum()) == 6
    assert int((cnt[masks[0] != 0] == 6).sum()) == 6
    assert int((cnt[masks[0] == 0] == 0).sum()) == lbl.numel() - 6


def test_connected_components_two_blobs() -> None:
    masks = torch.tensor(
        [
            [
                [0, 1, 1, 0, 0],
                [0, 1, 0, 0, 2],
                [0, 0, 0, 2, 2],
                [0, 0, 3, 0, 0],
            ]
        ],
        dtype=torch.int32,
    )
    labels, counts = connected_components(masks)
    components = labels[masks != 0].unique(sorted=True)
    components = components[components != 0]
    assert components.shape[0] == 3
    component_counts = sorted(counts[masks != 0].unique(sorted=True).tolist())
    assert component_counts == [1, 3]
    assert labels.shape == masks.shape
    assert counts.shape == masks.shape
    assert labels[masks == 0].max() == 0
    assert int((labels > 0).sum()) == 7


def test_masks_to_boxes_matches_extents() -> None:
    masks = torch.tensor(
        [
            [
                [0, 0, 0, 0],
                [0, 5, 1, 0],
                [0, 0, 0, 0],
            ],
            [
                [3, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 3, 3, 0],
            ],
            torch.zeros((3, 4), dtype=torch.int32),
        ],
        dtype=torch.int32,
    )

    boxes = mask_ops.masks_to_boxes(masks)
    # Coordinates use an inclusive maximum endpoint.
    expected = torch.tensor(
        [
            [1.0, 1.0, 2.0, 1.0],
            [0.0, 0.0, 2.0, 2.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(boxes, expected)
