"""Production builders accept dtype/precision and cast after weight load."""

from __future__ import annotations

import pytest
import torch

from sam3.weights.load_sam3 import (
    build_production_interactive,
    build_production_text_detector,
    build_production_tracker,
    build_production_video_tracker,
    resolve_sam3_checkpoint,
)

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)


def _first_float_dtype(module: torch.nn.Module) -> torch.dtype:
    for name, p in module.named_parameters():
        if not p.is_floating_point():
            continue
        if "freqs_" in name or "freqs_cis" in name:
            continue
        return p.dtype
    raise AssertionError("no floating non-rope parameter")


def test_default_load_weights_is_inference_dtype() -> None:
    """load_weights=True without dtype → DEFAULT_INFERENCE_DTYPE permanent cast."""
    from sam3.dtype_policy import DEFAULT_INFERENCE_DTYPE, parse_dtype

    try:
        resolve_sam3_checkpoint()
    except FileNotFoundError as e:
        pytest.skip(str(e))
    expected = parse_dtype(DEFAULT_INFERENCE_DTYPE)
    pred = build_production_interactive(load_weights=True)
    assert pred.precision is not None
    assert pred.precision.compute_dtype == expected
    assert _first_float_dtype(pred) == expected
    assert next(pred.parameters()).is_cuda


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_interactive_builder_precision(dtype: torch.dtype) -> None:
    try:
        resolve_sam3_checkpoint()
    except FileNotFoundError as e:
        pytest.skip(str(e))
    pred = build_production_interactive(dtype=dtype, device="cuda", load_weights=True)
    assert pred.precision is not None
    assert pred.precision.compute_dtype == dtype
    assert _first_float_dtype(pred) == dtype
    # RoPE-ish buffers stay fp32 when present
    for name, buf in pred.named_buffers():
        if "freqs_" in name or "freqs_cis" in name:
            if torch.is_floating_point(buf) and not torch.is_complex(buf):
                assert buf.dtype == torch.float32, name


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_text_detector_builder_precision(dtype: torch.dtype) -> None:
    try:
        resolve_sam3_checkpoint()
    except FileNotFoundError as e:
        pytest.skip(str(e))
    model = build_production_text_detector(
        dtype=dtype, device="cuda", load_weights=True
    )
    assert _first_float_dtype(model) == dtype


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_tracker_builder_precision(dtype: torch.dtype) -> None:
    try:
        resolve_sam3_checkpoint()
    except FileNotFoundError as e:
        pytest.skip(str(e))
    tracker = build_production_tracker(
        with_backbone=False,
        dtype=dtype,
        device="cuda",
        load_weights=True,
    )
    assert tracker.precision is not None
    assert tracker.precision.compute_dtype == dtype
    assert _first_float_dtype(tracker) == dtype

    vt = build_production_video_tracker(
        with_backbone=False,
        dtype=dtype,
        device="cuda",
        load_weights=True,
    )
    assert vt.precision is not None
    assert vt.precision.compute_dtype == dtype
    assert _first_float_dtype(vt.tracker) == dtype
