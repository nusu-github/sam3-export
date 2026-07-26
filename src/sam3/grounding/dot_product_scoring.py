"""Dot-product scoring head for text-to-query matching.

Port of ``sam3.model.model_misc.DotProductScoring``.
"""

from __future__ import annotations

import math

from jaxtyping import Bool, Float
import torch
from torch import Tensor
import torch.nn as nn


class DotProductScoring(nn.Module):
    """Mean-pool prompt tokens and score against projected object queries.

    Args:
        d_model: Channel dim of ``hs`` / prompt features.
        d_proj: Projection dim used for the scaled dot product.
        prompt_mlp: Optional MLP applied to prompt before pooling.
        clamp_logits: If True, clamp scores to ``[-clamp_max_val, clamp_max_val]``.
        clamp_max_val: Clamp magnitude when ``clamp_logits`` is True.
    """

    def __init__(
        self,
        d_model: int,
        d_proj: int,
        prompt_mlp: nn.Module | None = None,
        clamp_logits: bool = True,
        clamp_max_val: float | int = 12.0,
    ) -> None:
        super().__init__()
        self.d_proj = d_proj
        assert isinstance(prompt_mlp, nn.Module) or prompt_mlp is None
        self.prompt_mlp = prompt_mlp  # optional MLP projection for prompt
        # Keep nn.Linear so state_dict keys match official checkpoints.
        self.prompt_proj = nn.Linear(d_model, d_proj)
        self.hs_proj = nn.Linear(d_model, d_proj)
        self.scale = float(1.0 / math.sqrt(d_proj))
        self.clamp_logits = clamp_logits
        if self.clamp_logits:
            self.clamp_max_val = float(clamp_max_val)

    def mean_pool_text(
        self,
        prompt: Float[Tensor, "seq bs d"],
        prompt_mask: Bool[Tensor, "bs seq"] | Float[Tensor, "bs seq"],
    ) -> Float[Tensor, "bs d"]:
        # is_valid has shape (seq, bs, 1); prompt_mask is True/1 for padding.
        # Keep is_valid in prompt dtype so bf16 * mask does not promote to fp32.
        is_valid = (~prompt_mask).to(dtype=prompt.dtype).permute(1, 0)[..., None]
        # num_valid has shape (bs, 1)
        num_valid = torch.clamp(torch.sum(is_valid, dim=0), min=1.0)
        # mean pool over valid tokens → (bs, d_model) or (bs, d_proj) after MLP
        pooled_prompt = (prompt * is_valid).sum(dim=0) / num_valid
        return pooled_prompt

    def forward(
        self,
        hs: Float[Tensor, "num_layer bs num_query d_model"],
        prompt: Float[Tensor, "seq bs d_model"],
        prompt_mask: Bool[Tensor, "bs seq"] | Float[Tensor, "bs seq"],
    ) -> Float[Tensor, "num_layer bs num_query 1"]:
        # hs: (num_layer, bs, num_query, d_model)
        # prompt: (seq, bs, d_model)
        # prompt_mask: (bs, seq) — True = padding (key_padding_mask convention)
        assert hs.dim() == 4 and prompt.dim() == 3 and prompt_mask.dim() == 2

        if self.prompt_mlp is not None:
            prompt = self.prompt_mlp(prompt)

        pooled_prompt = self.mean_pool_text(prompt, prompt_mask)

        proj_pooled_prompt = self.prompt_proj(pooled_prompt)  # (bs, d_proj)
        proj_hs = self.hs_proj(hs)  # (num_layer, bs, num_query, d_proj)

        # (num_layer, bs, num_query, 1)
        scores = torch.matmul(proj_hs, proj_pooled_prompt.unsqueeze(-1))
        scores *= self.scale

        if self.clamp_logits:
            scores.clamp_(min=-self.clamp_max_val, max=self.clamp_max_val)

        return scores
