"""Unit tests for dtype policy helpers."""

from __future__ import annotations

import contextlib

import pytest
import torch
import torch.nn as nn

from sam3.dtype_policy import (
    DEFAULT_INFERENCE_DTYPE,
    PrecisionConfig,
    apply_precision,
    cast_module_,
    finalize_module,
    make_precision,
    maybe_autocast,
    module_param_dtype,
    parse_dtype,
    resolve_device,
    resolve_precision,
)


class _PolicyFixture(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2, 2, dtype=torch.float32))
        self.freqs_param = nn.Parameter(torch.ones(4, dtype=torch.float32))
        self.register_parameter(
            "freqs_param_table",
            nn.Parameter(torch.ones(3, dtype=torch.float32)),
        )
        self.register_buffer("freqs_cis_buf", torch.ones(2, dtype=torch.float32))
        self.register_buffer("freqs_sinusoid", torch.ones(2, dtype=torch.float32))
        self.register_buffer("other", torch.ones(2, dtype=torch.float32))
        self.register_buffer("complex_buffer", torch.ones(2, dtype=torch.complex64))


@pytest.mark.parametrize(
    ("name", "dtype"),
    [
        ("fp16", torch.float16),
        ("float16", torch.float16),
        ("half", torch.float16),
        ("bf16", torch.bfloat16),
        ("bfloat16", torch.bfloat16),
        ("fp32", torch.float32),
        ("float32", torch.float32),
        ("single", torch.float32),
    ],
)
def test_parse_dtype_aliases(name: str, dtype: torch.dtype) -> None:
    assert parse_dtype(name) is dtype
    assert parse_dtype(dtype) is dtype


def test_parse_dtype_invalid_alias() -> None:
    with pytest.raises(ValueError, match="Unsupported dtype"):
        parse_dtype("int8")


def test_cast_module_keeps_rope_and_complex() -> None:
    module = _PolicyFixture()
    cast_module_(module, torch.float16)

    assert module.weight.dtype == torch.float16
    assert module.freqs_param.dtype == torch.float32
    assert module.freqs_param_table.dtype == torch.float32
    assert module.freqs_cis_buf.dtype == torch.float32
    assert module.freqs_sinusoid.dtype == torch.float32
    assert module.other.dtype == torch.float16
    assert module.complex_buffer.dtype == torch.complex64

    cast_module_(module, torch.float16, keep_rope=False)
    assert module.freqs_param_table.dtype == torch.float16
    assert module.freqs_cis_buf.dtype == torch.float16
    assert module.freqs_sinusoid.dtype == torch.float16


def test_apply_precision_permanent_vs_autocast() -> None:
    perm = _PolicyFixture()
    apply_precision(
        perm, PrecisionConfig(compute_dtype=torch.bfloat16, use_autocast=False)
    )
    assert perm.weight.dtype == torch.bfloat16
    assert perm.freqs_cis_buf.dtype == torch.float32
    assert perm.other.dtype == torch.bfloat16

    auto = _PolicyFixture()
    apply_precision(
        auto,
        PrecisionConfig(
            compute_dtype=torch.bfloat16,
            master_weight_dtype=torch.float32,
            use_autocast=True,
        ),
    )
    assert auto.weight.dtype == torch.float32
    assert auto.freqs_cis_buf.dtype == torch.float32
    assert auto.other.dtype == torch.float32


def test_module_param_dtype() -> None:
    fixture = _PolicyFixture()
    assert module_param_dtype(fixture) == torch.float32
    no_float = nn.Module()
    no_float.register_parameter(
        "weight", nn.Parameter(torch.ones(1, dtype=torch.int64), requires_grad=False)
    )
    with pytest.raises(ValueError, match="no floating-point"):
        module_param_dtype(no_float)


def test_resolved_storage_defaults() -> None:
    assert PrecisionConfig().resolved_storage() == torch.bfloat16
    assert (
        PrecisionConfig(
            compute_dtype=torch.float16,
            storage_dtype=torch.float16,
        ).resolved_storage()
        == torch.float16
    )


def test_make_precision_and_finalize() -> None:
    cfg = make_precision("fp16", use_autocast=False)
    assert cfg.compute_dtype == torch.float16
    assert cfg.use_autocast is False
    module = nn.Linear(2, 2)
    finalize_module(module, cfg)
    assert module.weight.dtype == torch.float16
    cpu_mod = nn.Linear(2, 2)
    finalize_module(cpu_mod, None, device="cpu")
    assert cpu_mod.weight.device.type == "cpu"


def test_resolve_precision_default_on_load() -> None:
    assert resolve_precision(load_weights=False) is None
    cfg = resolve_precision(load_weights=True)
    assert cfg is not None
    assert cfg.compute_dtype == parse_dtype(DEFAULT_INFERENCE_DTYPE)
    cfg_fp16 = resolve_precision(dtype="fp16", load_weights=True)
    assert cfg_fp16 is not None and cfg_fp16.compute_dtype == torch.float16
    # explicit fp32 overrides default
    cfg_fp32 = resolve_precision(dtype="fp32", load_weights=True)
    assert cfg_fp32 is not None and cfg_fp32.compute_dtype == torch.float32
    # device defaults when precision active
    if torch.cuda.is_available():
        assert resolve_device(None, precision=cfg) == "cuda"
    assert resolve_device("cpu", precision=cfg) == "cpu"


def test_maybe_autocast_respects_cfg(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_autocast(*args: object, **kwargs: object):
        calls.append((args, kwargs))
        return contextlib.nullcontext()

    monkeypatch.setattr(torch.amp, "autocast", fake_autocast)

    with maybe_autocast(
        PrecisionConfig(
            use_autocast=True, device_type="cpu", compute_dtype=torch.bfloat16
        )
    ):
        pass
    with maybe_autocast(
        PrecisionConfig(
            use_autocast=False, device_type="cpu", compute_dtype=torch.bfloat16
        )
    ):
        pass

    assert len(calls) == 1
    _, kwargs = calls[0]
    assert kwargs["device_type"] == "cpu"
    assert kwargs["dtype"] == torch.bfloat16
    assert kwargs["enabled"] is True
