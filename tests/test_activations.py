"""Activation and fused Linear+activation parity tests (ATen only)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from sam3.primitives.activations import addmm_act

if not torch.cuda.is_available():
    pytest.skip("CUDA is required for activation tests", allow_module_level=True)

DEVICE = torch.device("cuda")


def _tol(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float16:
        return 3e-2, 3e-3
    return 1e-2, 5e-3


def _grad_tol(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float16:
        return 5e-2, 1e-2
    return 3e-3, 5e-3


def _reference_gelu(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="tanh")


def _activation_tag(activation: object) -> str:
    if activation in [torch.nn.functional.relu, nn.ReLU, nn.ReLU()]:
        return "relu"
    if activation in [torch.nn.functional.gelu, nn.GELU, nn.GELU()]:
        return "gelu"
    raise ValueError(f"Unexpected activation {activation}")


def _apply_reference_activation(activation: object, x: torch.Tensor) -> torch.Tensor:
    tag = _activation_tag(activation)
    if tag == "relu":
        return F.relu(x)
    return _reference_gelu(x)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(8, 64), (2, 4, 64)])
@pytest.mark.parametrize(
    "activation",
    [torch.nn.functional.relu, torch.nn.functional.gelu, nn.ReLU, nn.GELU],
)
def test_addmm_act_forward_and_backward_matches_torch(
    dtype: torch.dtype,
    shape: tuple[int, ...],
    activation: object,
) -> None:
    torch.manual_seed(11)
    in_features = shape[-1]
    out_features = in_features + 32

    linear = nn.Linear(in_features, out_features).to(device=DEVICE, dtype=dtype)
    linear_ref = nn.Linear(in_features, out_features).to(device=DEVICE, dtype=dtype)
    linear_ref.weight.data.copy_(linear.weight.data)
    linear_ref.bias.data.copy_(linear.bias.data)

    x = torch.randn(*shape, in_features, device=DEVICE, dtype=dtype, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)

    y_t = addmm_act(activation, linear, x)
    y_ref = _apply_reference_activation(
        activation,
        F.linear(x_ref, linear_ref.weight, linear_ref.bias),
    )

    grad = torch.randn_like(y_t)
    y_t.backward(grad)
    y_ref.backward(grad)

    rtol, atol = _tol(dtype)
    torch.testing.assert_close(y_t, y_ref, rtol=rtol, atol=atol)
    grad_rtol, grad_atol = _grad_tol(dtype)
    torch.testing.assert_close(x.grad, x_ref.grad, rtol=grad_rtol, atol=grad_atol)
    torch.testing.assert_close(
        linear.weight.grad, linear_ref.weight.grad, rtol=grad_rtol, atol=grad_atol
    )
    if linear.bias is not None:
        torch.testing.assert_close(
            linear.bias.grad, linear_ref.bias.grad, rtol=grad_rtol, atol=grad_atol
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(2, 4, 64)])
@pytest.mark.parametrize("activation", [torch.nn.functional.gelu, nn.GELU])
def test_addmm_act_accepts_weight_bias_tuple(
    dtype: torch.dtype, shape: tuple[int, ...], activation: object
) -> None:
    torch.manual_seed(19)
    in_features = shape[-1]
    out_features = 80
    linear = nn.Linear(in_features, out_features).to(device=DEVICE, dtype=dtype)

    x = torch.randn(*shape, in_features, device=DEVICE, dtype=dtype)
    y_t = addmm_act(activation, (linear.weight, linear.bias), x)
    y_ref = _apply_reference_activation(
        activation,
        F.linear(x, linear.weight, linear.bias),
    )
    rtol, atol = _tol(dtype)
    torch.testing.assert_close(y_t, y_ref, rtol=rtol, atol=atol)
