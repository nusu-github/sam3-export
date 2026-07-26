"""Shared helpers (CUDA checks only; no vendor kernels)."""

from __future__ import annotations

from torch import Tensor


def ensure_cuda(x: Tensor, op_name: str) -> None:
    if not x.is_cuda:
        raise RuntimeError(f"{op_name} requires CUDA tensors; got device={x.device}")
