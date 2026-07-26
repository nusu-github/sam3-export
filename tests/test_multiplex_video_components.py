"""M5 canonical component export, slot isolation, and bucket-boundary gates."""

from __future__ import annotations

import pytest
import torch

from sam3.export.multiplex import ScatterReplace
from sam3.export.multiplex_video import MultiplexPropagation
from sam3.runtime.multiplex_state import MultiplexVariantParameters
from sam3.weights.multiplex import (
    build_sam31_multiplex_tracker_core,
    load_sam31_multiplex_checkpoint,
    resolve_sam31_multiplex_checkpoint,
)


def test_private_scatter_replace_exports_and_preserves_non_target_bytes() -> None:
    module = ScatterReplace(2, validate_assignments=False)
    values = torch.arange(2 * 16 * 4, dtype=torch.float32).reshape(2, 16, 4)
    replacement = torch.full((1, 4), -7.0)
    assignment = torch.tensor([[1, 0]], dtype=torch.int64)
    exported = torch.export.export(
        module, (values, replacement, assignment), strict=False
    )
    output = exported.module()(values, replacement, assignment)
    expected = values.clone()
    expected[1, 0] = replacement[0]
    assert torch.equal(output, expected)


@pytest.fixture(scope="module")
def tracker() -> torch.nn.Module:
    if not torch.cuda.is_available():
        pytest.skip("M5 native component gate requires CUDA")
    try:
        checkpoint = load_sam31_multiplex_checkpoint(
            resolve_sam31_multiplex_checkpoint()
        )
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    return (
        build_sam31_multiplex_tracker_core(checkpoint)
        .eval()
        .to(device="cuda", dtype=torch.float16)
    )


def _args(bucket_count: int) -> list[torch.Tensor]:
    device = "cuda"
    dtype = torch.float16
    torch.manual_seed(17)
    values = [
        torch.randn(1, 256, 72, 72, device=device, dtype=dtype),
        torch.randn(1, 256, 72, 72, device=device, dtype=dtype),
        torch.randn(1, 32, 288, 288, device=device, dtype=dtype),
        torch.randn(1, 64, 144, 144, device=device, dtype=dtype),
        torch.zeros(bucket_count, 16, device=device, dtype=torch.bool),
        torch.randn(
            bucket_count, 10, 256, 72, 72, device=device, dtype=dtype
        ),
        torch.randn(
            bucket_count, 10, 256, 72, 72, device=device, dtype=dtype
        ),
        torch.randn(10, 256, 72, 72, device=device, dtype=dtype),
        torch.randn(10, 256, 72, 72, device=device, dtype=dtype),
        torch.zeros(bucket_count, 10, device=device, dtype=torch.bool),
        torch.zeros(bucket_count, 10, device=device, dtype=torch.int64),
        torch.randn(
            bucket_count, 16, 16, 256, device=device, dtype=dtype
        ),
        torch.zeros(bucket_count, 16, device=device, dtype=torch.bool),
        torch.zeros(bucket_count, 16, device=device, dtype=torch.int64),
    ]
    values[4][:, 0] = True
    values[9][:, 0] = True
    values[10][:, 0] = 1
    values[12][:, 0] = True
    values[13][:, 0] = 1
    return values


@torch.inference_mode()
def test_padding_values_do_not_change_active_slot(tracker: torch.nn.Module) -> None:
    variant = MultiplexVariantParameters.native()
    module = MultiplexPropagation(
        tracker, variant, bucket_count=1, use_cuda_autocast=True
    ).eval()
    first_args = _args(1)
    second_args = [value.clone() for value in first_args]
    second_args[5][:, 1:] = 100
    second_args[6][:, 1:] = -100
    second_args[7][1:] = 100
    second_args[8][1:] = -100
    second_args[11][:, 1:] = 100
    first = module(*first_args)
    second = module(*second_args)
    torch.testing.assert_close(
        first[2][:, :1], second[2][:, :1], atol=2e-3, rtol=2e-3
    )
    torch.testing.assert_close(
        first[4][:, :1], second[4][:, :1], atol=2e-3, rtol=2e-3
    )


@torch.inference_mode()
def test_object_16_and_17_are_isolated_across_bucket_boundary(
    tracker: torch.nn.Module,
) -> None:
    variant = MultiplexVariantParameters.native()
    one_args = _args(1)
    two_args = _args(2)
    for index in (5, 6, 11):
        two_args[index][0] = one_args[index][0]
    for index in (4, 9, 10, 12, 13):
        two_args[index][0] = one_args[index][0]
    for index in (0, 1, 2, 3, 7, 8):
        two_args[index] = one_args[index]
    one = MultiplexPropagation(
        tracker, variant, bucket_count=1, use_cuda_autocast=True
    ).eval()(*one_args)
    two = MultiplexPropagation(
        tracker, variant, bucket_count=2, use_cuda_autocast=True
    ).eval()(*two_args)
    torch.testing.assert_close(one[2][0], two[2][0], atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(one[4][0], two[4][0], atol=1e-2, rtol=1e-2)
