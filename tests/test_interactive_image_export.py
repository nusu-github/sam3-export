"""M3 production interactive components and fixed prompt ABI tests."""

from __future__ import annotations

import pytest
import torch

from sam3.export.interactive_image import (
    InteractivePredictMultimask3,
    InteractivePredictSingle1,
)
from sam3.vision.sam_image_head import SamImageHead


def _head() -> SamImageHead:
    torch.manual_seed(7)
    return SamImageHead(
        embed_dim=32,
        image_embedding_size=(8, 8),
        input_image_size=(32, 32),
        transformer_depth=1,
        transformer_heads=4,
        transformer_mlp_dim=64,
        use_high_res_features=True,
        iou_prediction_use_sigmoid=True,
        dynamic_multimask_via_stability=True,
    ).eval()


def _fixed_prompt(
    points: list[tuple[float, float]],
    labels: list[int],
    box: tuple[float, float, float, float] | None,
    mask: torch.Tensor | None,
) -> tuple[torch.Tensor, ...]:
    point_coords = torch.zeros((1, 16, 2), dtype=torch.float32)
    point_labels = torch.full((1, 16), -1, dtype=torch.int64)
    point_valid = torch.zeros((1, 16), dtype=torch.bool)
    if points:
        count = len(points)
        point_coords[0, :count] = torch.tensor(points)
        point_labels[0, :count] = torch.tensor(labels)
        point_valid[0, :count] = True
    box_xyxy = torch.zeros((1, 4), dtype=torch.float32)
    has_box = torch.tensor([box is not None], dtype=torch.bool)
    if box is not None:
        box_xyxy[0] = torch.tensor(box)
    mask_input = torch.zeros((1, 1, 32, 32), dtype=torch.float32)
    has_mask = torch.tensor([mask is not None], dtype=torch.bool)
    if mask is not None:
        mask_input[0] = mask
    return (
        point_coords,
        point_labels,
        point_valid,
        box_xyxy,
        has_box,
        mask_input,
        has_mask,
    )


def _variable_reference(
    head: SamImageHead,
    image: torch.Tensor,
    high_res: list[torch.Tensor],
    points: list[tuple[float, float]],
    labels: list[int],
    box: tuple[float, float, float, float] | None,
    mask: torch.Tensor | None,
    *,
    multimask: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    concat_points = None
    if points:
        concat_points = (
            torch.tensor(points, dtype=torch.float32).unsqueeze(0),
            torch.tensor(labels, dtype=torch.int64).unsqueeze(0),
        )
    if box is not None:
        box_coords = torch.tensor(box, dtype=torch.float32).reshape(1, 2, 2)
        box_labels = torch.tensor([[2, 3]], dtype=torch.int64)
        if concat_points is None:
            concat_points = (box_coords, box_labels)
        else:
            concat_points = (
                torch.cat((box_coords, concat_points[0]), dim=1),
                torch.cat((box_labels, concat_points[1]), dim=1),
            )
    mask_input = None if mask is None else mask.unsqueeze(0)
    low_res, scores, _tokens, _object = head(
        image_embeddings=image,
        points=concat_points,
        boxes=None,
        masks=mask_input,
        multimask_output=multimask,
        high_res_features=high_res,
    )
    return low_res.float().clamp(-32, 32), scores.float()


@pytest.mark.parametrize(
    ("points", "labels", "box", "with_mask"),
    [
        ([], [], None, False),
        ([(8.0, 12.0)], [1], None, False),
        ([], [], (3.0, 4.0, 24.0, 25.0), False),
        (
            [(float(index), float(index + 1)) for index in range(16)],
            [index % 2 for index in range(16)],
            (1.0, 2.0, 28.0, 29.0),
            False,
        ),
        ([], [], None, True),
        ([(8.0, 12.0)], [1], (3.0, 4.0, 24.0, 25.0), True),
    ],
)
@pytest.mark.parametrize("multimask", [True, False])
def test_fixed_prompt_validity_matches_variable_official_semantics(
    points: list[tuple[float, float]],
    labels: list[int],
    box: tuple[float, float, float, float] | None,
    with_mask: bool,
    multimask: bool,
) -> None:
    head = _head()
    image = torch.randn(1, 32, 8, 8)
    high_res = [torch.randn(1, 4, 32, 32), torch.randn(1, 8, 16, 16)]
    mask = torch.randn(1, 32, 32) if with_mask else None
    wrapper = (
        InteractivePredictMultimask3(head)
        if multimask
        else InteractivePredictSingle1(head)
    )
    actual = wrapper(image, *high_res, *_fixed_prompt(points, labels, box, mask))
    expected = _variable_reference(
        head,
        image,
        high_res,
        points,
        labels,
        box,
        mask,
        multimask=multimask,
    )
    torch.testing.assert_close(actual[0], expected[0], atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(actual[1], expected[1], atol=2e-6, rtol=2e-6)


def test_fixed_prompt_exports_with_static_capacity() -> None:
    head = _head()
    wrapper = InteractivePredictMultimask3(head)
    args = (
        torch.randn(1, 32, 8, 8),
        torch.randn(1, 4, 32, 32),
        torch.randn(1, 8, 16, 16),
        *_fixed_prompt([(8.0, 12.0)], [1], None, None),
    )
    exported = torch.export.export(wrapper, args, strict=False)
    low_res, scores = exported.module()(*args)
    assert low_res.shape == (1, 3, 32, 32)
    assert scores.shape == (1, 3)
