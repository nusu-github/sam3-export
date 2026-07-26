"""MLP building blocks shared by SAM3 model components.

* ``Mlp``     – ViT FFN (same API as ``timm.layers.Mlp`` / SAM3 vitdet)
* ``MLP``     – multi-layer head MLP (``sam3.model.model_misc.MLP`` API)
* ``MLPBlock``– 2-layer transformer FFN (``sam3.sam.common.MLPBlock``)

All use plain ``nn.Linear`` (``torch.export`` / cuBLAS). Structure matches
timm so ``fc1``/``fc2`` checkpoint keys load without remapping.
"""

from __future__ import annotations

from typing import Optional, Type

from jaxtyping import Float
import torch
from torch import Tensor
import torch.nn as nn

_ACT_GELU = "gelu"
_ACT_RELU = "relu"


def _activation_tag(activation: nn.Module) -> str | None:
    if isinstance(activation, nn.GELU):
        return _ACT_GELU
    if isinstance(activation, nn.ReLU):
        return _ACT_RELU
    return None


def _try_addmm_activation(
    linear: nn.Module,
    x: Float[Tensor, "*batch features"],
    activation: nn.Module,
) -> Float[Tensor, "*batch out_features"] | None:
    if torch.is_grad_enabled():
        return None
    if not x.is_cuda:
        return None
    act = _activation_tag(activation)
    if act is None:
        return None
    if not isinstance(linear, nn.Linear):
        return None
    if type(linear.weight) is not nn.Parameter:
        return None
    if x.dtype not in (torch.float16, torch.bfloat16):
        return None
    if x.shape[-1] != linear.weight.shape[1]:
        return None
    if x.device != linear.weight.device:
        return None

    addmm_act_op = getattr(torch.ops.aten, "_addmm_activation", None)
    if addmm_act_op is None:
        return None

    try:
        bias = linear.bias
        if bias is None:
            bias = torch.zeros(
                (linear.weight.shape[0],), device=x.device, dtype=torch.bfloat16
            )
            beta = 0
        else:
            bias = bias.to(torch.bfloat16)
            beta = 1

        y2d = addmm_act_op(
            bias,
            x.to(torch.bfloat16).reshape(-1, x.shape[-1]),
            linear.weight.to(torch.bfloat16).t(),
            beta=beta,
            alpha=1.0,
            use_gelu=(act == _ACT_GELU),
        )
        return y2d.reshape(*x.shape[:-1], linear.weight.shape[0]).to(x.dtype)
    except Exception:
        return None


def _apply_linear_act(
    linear: nn.Module,
    x: Float[Tensor, "*batch features"],
    activation: nn.Module,
) -> Float[Tensor, "*batch out_features"]:
    fused = _try_addmm_activation(linear, x, activation)
    if fused is not None:
        return fused
    return activation(linear(x))


class MLP(nn.Module):
    """Multi-layer perceptron used across SAM3 detector / mask heads.

    Matches ``sam3.model.model_misc.MLP`` (ReLU + optional residual / out_norm)
    and the mask-decoder variant that adds ``sigmoid_output``.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float = 0.0,
        residual: bool = False,
        activation: Type[nn.Module] = nn.ReLU,
        out_norm: Optional[nn.Module] = None,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if residual and input_dim != output_dim:
            raise ValueError("residual is only supported if input_dim == output_dim")
        if out_norm is not None and not isinstance(out_norm, nn.Module):
            raise TypeError("out_norm must be an nn.Module or None")

        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.activation = activation()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.residual = residual
        self.sigmoid_output = sigmoid_output
        self.out_norm = out_norm or nn.Identity()

    def forward(
        self, x: Float[Tensor, "*batch features"]
    ) -> Float[Tensor, "*batch out_features"]:
        orig_x = x
        for i, layer in enumerate(self.layers):
            if i < self.num_layers - 1:
                x = _apply_linear_act(layer, x, self.activation)
                x = self.drop(x)
            else:
                x = layer(x)
        if self.residual:
            x = x + orig_x
        x = self.out_norm(x)
        if self.sigmoid_output:
            x = torch.sigmoid(x)
        return x


class MLPBlock(nn.Module):
    """2-layer MLP used by SAM3 transformer blocks (``sam3.sam.common.MLPBlock``)."""

    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(
        self, x: Float[Tensor, "*batch features"]
    ) -> Float[Tensor, "*batch features"]:
        wdtype = self.lin1.weight.dtype
        if x.dtype != wdtype:
            x = x.to(dtype=wdtype)
        x = _apply_linear_act(self.lin1, x, self.act)
        return self.lin2(x)
