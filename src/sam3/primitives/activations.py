"""ATen activation helpers shared by SAM3 model components."""

from __future__ import annotations

from typing import Protocol, TypeAlias, runtime_checkable

from jaxtyping import Float
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

_ACT_GELU = "gelu"
_ACT_RELU = "relu"


@runtime_checkable
class _LinearLikeProtocol(Protocol):
    weight: torch.Tensor
    bias: torch.Tensor | None


LinearLike: TypeAlias = (
    nn.Linear
    | tuple[torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor | None]
    | _LinearLikeProtocol
)

ActivationSpec: TypeAlias = str | type[nn.Module] | nn.Module | object


def _resolve_activation(activation: ActivationSpec) -> str:
    if isinstance(activation, str):
        normalized = activation.lower()
        if normalized in {"gelu", _ACT_GELU}:
            return _ACT_GELU
        if normalized in {"relu", _ACT_RELU}:
            return _ACT_RELU
    if activation in (torch.nn.functional.relu, torch.relu, nn.ReLU):
        return _ACT_RELU
    if activation in (torch.nn.functional.gelu, nn.GELU):
        return _ACT_GELU
    if isinstance(activation, nn.Module):
        if isinstance(activation, nn.ReLU):
            return _ACT_RELU
        if isinstance(activation, nn.GELU):
            return _ACT_GELU
    if isinstance(activation, type):
        if issubclass(activation, nn.ReLU):
            return _ACT_RELU
        if issubclass(activation, nn.GELU):
            return _ACT_GELU
    raise ValueError(f"Unsupported activation {activation!r}")


def _extract_linear_params(
    linear_module_or_weights: LinearLike,
) -> tuple[Tensor, Tensor | None]:
    if isinstance(linear_module_or_weights, tuple):
        if len(linear_module_or_weights) == 1:
            return linear_module_or_weights[0], None
        return linear_module_or_weights[0], linear_module_or_weights[1]
    return linear_module_or_weights.weight, linear_module_or_weights.bias


def linear_act_forward(
    x: Float[Tensor, "*batch in_features"],
    weight: Float[Tensor, "out_features in_features"],
    bias: Float[Tensor, "out_features"] | None,
    activation: ActivationSpec,
) -> Float[Tensor, "*batch out_features"]:
    y = F.linear(x, weight, bias)
    act = _resolve_activation(activation)
    if act == _ACT_GELU:
        return F.gelu(y, approximate="tanh")
    if act == _ACT_RELU:
        return F.relu(y)
    raise ValueError(act)


def addmm_act(
    activation: ActivationSpec,
    linear_module_or_weights: LinearLike,
    x: Float[Tensor, "*batch in_features"],
) -> Float[Tensor, "*batch out_features"]:
    """Apply an ATen linear projection followed by ReLU or tanh-GELU.

    This intentionally does not select an optional fused kernel: the same
    implementation is used for eager inference, training, and export capture.
    """
    weight, bias = _extract_linear_params(linear_module_or_weights)
    return linear_act_forward(x, weight, bias, activation)
