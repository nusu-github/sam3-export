"""Encoder/decoder holder for text-grounding stacks."""

from __future__ import annotations

import torch.nn as nn


class TransformerWrapper(nn.Module):
    def __init__(
        self,
        encoder: nn.Module | None,
        decoder: nn.Module | None,
        d_model: int,
        two_stage_type: str = "none",
        pos_enc_at_input_dec: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.num_queries = decoder.num_queries if decoder is not None else None
        self.pos_enc_at_input_dec = pos_enc_at_input_dec
        assert two_stage_type in ["none"], f"unknown two_stage_type {two_stage_type}"
        self.two_stage_type = two_stage_type
        self.d_model = d_model
        # Official resets non-box/query params; skip here so builders can load ckpt
        # without re-init fighting loaded weights. Call ``reset_parameters()`` if needed.
