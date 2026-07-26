"""Precision policy helpers for dtype/autocast behavior."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

SUPPORTED_COMPUTE = (torch.float32, torch.float16, torch.bfloat16)

# Default for production inference builders when ``load_weights=True`` and no
# explicit ``dtype``/``precision`` is passed. Matches official SAM3 deployment
# habit (bf16) while remaining overridable (``dtype="fp16"`` / ``"fp32"``).
DEFAULT_INFERENCE_DTYPE: str = "bf16"


def _is_rope_name(name: str) -> bool:
    return "freqs_cis" in name or "freqs_" in name


def is_low_precision(dtype: torch.dtype) -> bool:
    """True for fp16 / bf16 (and any future low-precision compute dtypes)."""
    return dtype in (torch.float16, torch.bfloat16)


@dataclass(frozen=True)
class PrecisionConfig:
    """Canonical precision policy for sam3."""

    compute_dtype: torch.dtype = torch.bfloat16
    master_weight_dtype: torch.dtype = torch.float32
    storage_dtype: torch.dtype | None = None
    rope_dtype: torch.dtype = torch.float32
    use_autocast: bool = False
    device_type: str = "cuda"

    def resolved_storage(self) -> torch.dtype:
        return self.compute_dtype if self.storage_dtype is None else self.storage_dtype

    def autocast(self, *, enabled: bool | None = None) -> AbstractContextManager[Any]:
        if enabled is None:
            enabled = self.use_autocast
        if not enabled:
            return nullcontext()
        return torch.amp.autocast(
            device_type=self.device_type,
            dtype=self.compute_dtype,
            enabled=enabled,
        )


def parse_dtype(name: str | torch.dtype) -> torch.dtype:
    """Parse dtype aliases used by the runner CLI and policy helpers."""
    if isinstance(name, torch.dtype):
        return name
    if not isinstance(name, str):
        raise TypeError(
            f"dtype must be torch.dtype or string alias, got {type(name)!r}"
        )
    key = name.strip().lower()
    alias = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "single": torch.float32,
    }.get(key)
    if alias is None:
        raise ValueError(f"Unsupported dtype alias: {name!r}")
    return alias


def module_param_dtype(module: nn.Module) -> torch.dtype:
    """Return first floating-point parameter dtype."""
    for parameter in module.parameters():
        if torch.is_floating_point(parameter):
            return parameter.dtype
    raise ValueError("module has no floating-point parameters")


def cast_floating_to(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """No-op if dtype already matches; cast only floating tensors."""
    if not torch.is_floating_point(t):
        return t
    if t.dtype == dtype:
        return t
    return t.to(dtype=dtype)


def cast_module_(
    module: nn.Module,
    dtype: torch.dtype,
    *,
    keep_rope: bool = True,
    device: torch.device | str | None = None,
    rope_dtype: torch.dtype = torch.float32,
) -> nn.Module:
    """In-place: cast floating params/buffers to ``dtype`` with RoPE/complex exceptions.

    Rules:
      - optional device move first
      - complex buffers: left untouched
      - names containing ``freqs_cis`` or ``freqs_``: keep ``rope_dtype`` (default
        float32) when ``keep_rope`` is true
      - other floating params/buffers: cast to ``dtype`` via ``.data`` (preserves
        Parameter identity for optimizers / shared tensors)
    """
    if device is not None:
        module.to(device=device)

    for name, parameter in module.named_parameters():
        if not torch.is_floating_point(parameter):
            continue
        target = rope_dtype if (keep_rope and _is_rope_name(name)) else dtype
        if parameter.dtype != target:
            parameter.data = parameter.data.to(dtype=target)

    for name, buffer in module.named_buffers():
        if not torch.is_tensor(buffer):
            continue
        if torch.is_complex(buffer) or not torch.is_floating_point(buffer):
            continue
        target = rope_dtype if (keep_rope and _is_rope_name(name)) else dtype
        if buffer.dtype != target:
            buffer.data = buffer.data.to(dtype=target)

    return module


def apply_precision(
    module: nn.Module,
    cfg: PrecisionConfig,
    *,
    device: torch.device | str | None = None,
) -> nn.Module:
    """Apply precision policy with optional permanent cast or fp32-weight autocast setup."""
    dtype = cfg.master_weight_dtype if cfg.use_autocast else cfg.compute_dtype
    cast_module_(
        module,
        dtype=dtype,
        keep_rope=True,
        device=device,
        rope_dtype=cfg.rope_dtype,
    )
    return module


def maybe_autocast(cfg: PrecisionConfig):
    """Convenience alias for :meth:`PrecisionConfig.autocast`."""
    return cfg.autocast()


def make_precision(
    dtype: str | torch.dtype | None = None,
    *,
    use_autocast: bool = False,
    storage_dtype: torch.dtype | str | None = None,
    master_weight_dtype: torch.dtype | str | None = None,
    device_type: str = "cuda",
) -> PrecisionConfig:
    """Build a :class:`PrecisionConfig` from CLI-friendly aliases.

    ``dtype=None`` defaults to :data:`DEFAULT_INFERENCE_DTYPE` (bf16 permanent).
    """
    compute = (
        parse_dtype(dtype)
        if dtype is not None
        else parse_dtype(DEFAULT_INFERENCE_DTYPE)
    )
    storage: torch.dtype | None
    if storage_dtype is None:
        storage = None
    else:
        storage = parse_dtype(storage_dtype)
    master = (
        parse_dtype(master_weight_dtype)
        if master_weight_dtype is not None
        else torch.float32
    )
    return PrecisionConfig(
        compute_dtype=compute,
        master_weight_dtype=master,
        storage_dtype=storage,
        use_autocast=use_autocast,
        device_type=device_type,
    )


def resolve_precision(
    precision: PrecisionConfig | None = None,
    *,
    dtype: torch.dtype | str | None = None,
    use_autocast: bool = False,
    load_weights: bool = False,
    default_on_load: bool = True,
) -> PrecisionConfig | None:
    """Resolve an optional precision policy for production builders.

    Priority: explicit ``precision`` → ``dtype`` → when ``load_weights`` and
    ``default_on_load``, :data:`DEFAULT_INFERENCE_DTYPE` permanent cast.
    """
    if precision is not None:
        return precision
    if dtype is not None:
        return make_precision(dtype, use_autocast=use_autocast)
    if load_weights and default_on_load:
        return make_precision(DEFAULT_INFERENCE_DTYPE, use_autocast=use_autocast)
    return None


def resolve_device(
    device: torch.device | str | None = None,
    *,
    precision: PrecisionConfig | None = None,
    prefer_cuda: bool = True,
) -> torch.device | str | None:
    """Default device to CUDA when a precision policy is active and CUDA exists."""
    if device is not None:
        return device
    if precision is not None and prefer_cuda and torch.cuda.is_available():
        return "cuda"
    return None


def finalize_module(
    module: nn.Module,
    precision: PrecisionConfig | None = None,
    *,
    device: torch.device | str | None = None,
) -> nn.Module:
    """Apply precision policy and/or move to device.

    - ``precision`` set → :func:`apply_precision` (includes optional device move)
    - only ``device`` → ``module.to(device)``
    - neither → no-op
    """
    if precision is not None:
        return apply_precision(module, precision, device=device)
    if device is not None:
        module.to(device=device)
    return module


def bf16_permanent() -> PrecisionConfig:
    """BF16 permanent-cast mode."""
    return PrecisionConfig(compute_dtype=torch.bfloat16, use_autocast=False)


def fp16_permanent() -> PrecisionConfig:
    """FP16 permanent-cast mode."""
    return PrecisionConfig(compute_dtype=torch.float16, use_autocast=False)


def bf16_autocast() -> PrecisionConfig:
    """BF16 compute under fp32 master-weight autocast mode."""
    return PrecisionConfig(
        compute_dtype=torch.bfloat16,
        master_weight_dtype=torch.float32,
        use_autocast=True,
    )


def fp16_autocast() -> PrecisionConfig:
    """FP16 compute under fp32 master-weight autocast mode."""
    return PrecisionConfig(
        compute_dtype=torch.float16,
        master_weight_dtype=torch.float32,
        use_autocast=True,
    )
