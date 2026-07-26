"""Fixed-Tensor detector encoder/decoder subgraph for ``torch.export``.

Text tokenization, geometry-prompt construction, image preprocessing, and
segmentation stay outside this contract.  The caller supplies the final prompt
sequence and one already-selected FPN level.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from sam3.grounding.det_decoder import inverse_sigmoid


class DetectorEncoderDecoder(nn.Module):
    """Run the single-level SAM3 detector on fixed Tensor inputs.

    Inputs are ``image_features`` and ``image_pos`` in ``(B, C, H, W)``, plus
    tokenized-and-encoded ``prompt`` in ``(T, B, C)`` and ``prompt_mask`` in
    ``(B, T)``.  Outputs are the last decoder layer's logits, ``cxcywh`` boxes,
    and presence logits.
    """

    def __init__(self, transformer: nn.Module, dot_prod_scoring: nn.Module) -> None:
        super().__init__()
        if transformer.encoder is None or transformer.decoder is None:
            raise ValueError("DetectorEncoderDecoder requires an encoder and decoder")
        if transformer.decoder.presence_token is None:
            raise ValueError("DetectorEncoderDecoder requires presence-token decoding")
        self.encoder = transformer.encoder
        self.decoder = transformer.decoder
        self.dot_prod_scoring = dot_prod_scoring

    def forward(
        self,
        image_features: Tensor,
        image_pos: Tensor,
        prompt: Tensor,
        prompt_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        src = image_features.flatten(2).permute(2, 0, 1)
        src_pos = image_pos.flatten(2).permute(2, 0, 1)
        memory_out = self.encoder(
            src=[src],
            src_key_padding_mask=None,
            src_pos=[src_pos],
            prompt=prompt,
            prompt_pos=torch.zeros_like(prompt),
            prompt_key_padding_mask=prompt_mask,
            feat_sizes=[image_features.shape[-2:]],
            encoder_extra_kwargs=None,
        )

        batch = image_features.shape[0]
        tgt = self.decoder.query_embed.weight.unsqueeze(1).expand(-1, batch, -1)
        hidden_states, reference_boxes, presence_logits, _ = self.decoder(
            tgt=tgt,
            memory=memory_out["memory"],
            memory_key_padding_mask=memory_out["padding_mask"],
            pos=memory_out["pos_embed"],
            reference_boxes=None,
            level_start_index=memory_out["level_start_index"],
            spatial_shapes=memory_out["spatial_shapes"],
            valid_ratios=memory_out["valid_ratios"],
            feature_size=image_features.shape[-2:],
            tgt_mask=None,
            memory_text=prompt,
            text_attention_mask=prompt_mask,
            apply_dac=False,
        )

        hidden_states = hidden_states.transpose(1, 2)
        reference_boxes = reference_boxes.transpose(1, 2)
        presence_logits = presence_logits.transpose(1, 2)
        logits = self.dot_prod_scoring(hidden_states, prompt, prompt_mask)
        boxes = (
            inverse_sigmoid(reference_boxes) + self.decoder.bbox_embed(hidden_states)
        ).sigmoid()
        return logits[-1], boxes[-1], presence_logits[-1]
