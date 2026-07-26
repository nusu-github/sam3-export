"""Cut B -- token-id-only text tower for ``torch.export``.

The tokenizer deliberately stays in the runtime.  ``attention_mask`` follows
the common tokeniser convention (one/``True`` means a real token); the returned
mask follows PyTorch attention's convention (one/``True`` means padding).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from sam3.grounding.text_encoder_ve import VETextEncoder


@dataclass(frozen=True)
class TextTowerSpec:
    """Static TextTower dimensions used by an exported artifact."""

    context_length: int = 32
    d_model: int = 256


class TextTower(nn.Module):
    """Encode pre-tokenized text without a tokenizer or Python strings.

    The tensor layout is batch first to make it convenient for runtimes:
    ``input_ids[B, L]`` -> ``text_memory[B, L, D]``.  GroundingEncode performs
    the one required sequence-first transpose at its boundary.
    """

    def __init__(self, encoder: VETextEncoder, *, validate: bool = True) -> None:
        super().__init__()
        if not isinstance(encoder, VETextEncoder):
            raise TypeError(f"encoder must be VETextEncoder, got {type(encoder)!r}")
        self.encoder = encoder
        self.validate = bool(validate)

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        if self.validate:
            if input_ids.ndim != 2:
                raise ValueError(
                    f"input_ids must be [B,L], got {tuple(input_ids.shape)}"
                )
            if attention_mask.shape != input_ids.shape:
                raise ValueError(
                    "attention_mask must have the same [B,L] shape as input_ids"
                )
            if input_ids.dtype not in (torch.int32, torch.int64):
                raise ValueError(f"input_ids must be integer, got {input_ids.dtype}")

        # TextTransformer returns (pooled, token features) when output_tokens=True.
        _pooled, tokens = self.encoder.encoder(input_ids)
        memory = self.encoder.resizer(tokens)
        padding_mask = ~attention_mask.to(torch.bool)
        return memory, padding_mask


__all__ = ["TextTower", "TextTowerSpec"]
