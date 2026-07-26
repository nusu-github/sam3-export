"""Visual-language backbone combiner for image/text grounding.

Port of ``sam3.model.vl_combiner.SAM3VLBackbone`` (dual-neck path only).
"""

from __future__ import annotations

from copy import copy
from typing import Any

from jaxtyping import Float
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel


class SAM3VLBackbone(nn.Module):
    """Combine a DualViTDet neck with a language backbone (no early fusion)."""

    def __init__(
        self,
        visual: nn.Module,
        text: nn.Module | None,
        compile_visual: bool = False,
        act_ckpt_whole_vision_backbone: bool = False,
        act_ckpt_whole_language_backbone: bool = False,
        scalp: int = 0,
    ) -> None:
        super().__init__()
        self.vision_backbone = torch.compile(visual) if compile_visual else visual
        self.language_backbone = text
        self.scalp = scalp
        self.act_ckpt_whole_vision_backbone = act_ckpt_whole_vision_backbone
        self.act_ckpt_whole_language_backbone = act_ckpt_whole_language_backbone

    def forward(
        self,
        samples: Float[Tensor, "..."],
        captions: list[str],
        input_boxes: Float[Tensor, "..."] | None = None,
        additional_text: list[str] | None = None,
    ) -> dict[str, Any]:
        output = self.forward_image(samples)
        device = output["vision_features"].device
        output.update(self.forward_text(captions, input_boxes, additional_text, device))
        return output

    def forward_image(self, samples: Float[Tensor, "..."]) -> dict[str, Any]:
        return self._forward_image_no_act_ckpt(samples)

    def _forward_image_no_act_ckpt(
        self, samples: Float[Tensor, "..."]
    ) -> dict[str, Any]:
        sam3_features, sam3_pos, sam2_features, sam2_pos = self.vision_backbone.forward(
            samples
        )
        if self.scalp > 0:
            sam3_features, sam3_pos = (
                sam3_features[: -self.scalp],
                sam3_pos[: -self.scalp],
            )
            if sam2_features is not None and sam2_pos is not None:
                sam2_features, sam2_pos = (
                    sam2_features[: -self.scalp],
                    sam2_pos[: -self.scalp],
                )

        sam2_output = None
        if sam2_features is not None and sam2_pos is not None:
            sam2_src = sam2_features[-1]
            sam2_output = {
                "vision_features": sam2_src,
                "vision_pos_enc": sam2_pos,
                "backbone_fpn": sam2_features,
            }

        sam3_src = sam3_features[-1]
        return {
            "vision_features": sam3_src,
            "vision_pos_enc": sam3_pos,
            "backbone_fpn": sam3_features,
            "sam2_backbone_out": sam2_output,
        }

    def forward_text(
        self,
        captions: list[str],
        input_boxes: Float[Tensor, "..."] | None = None,
        additional_text: list[str] | None = None,
        device: str | torch.device = "cuda",
    ) -> dict[str, Any]:
        return self._forward_text_no_ack_ckpt(
            captions=captions,
            input_boxes=input_boxes,
            additional_text=additional_text,
            device=device,
        )

    def _forward_text_no_ack_ckpt(
        self,
        captions: list[str],
        input_boxes: Float[Tensor, "..."] | None = None,
        additional_text: list[str] | None = None,
        device: str | torch.device = "cuda",
    ) -> dict[str, Any]:
        if self.language_backbone is None:
            raise RuntimeError("language_backbone is None; cannot encode text")

        output: dict[str, Any] = {}
        text_to_encode = copy(captions)
        if additional_text is not None:
            text_to_encode += additional_text

        sdpa_context = sdpa_kernel(
            [
                SDPBackend.MATH,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.FLASH_ATTENTION,
            ]
        )
        with sdpa_context:
            text_attention_mask, text_memory, text_embeds = self.language_backbone(
                text_to_encode, input_boxes, device=device
            )

        if additional_text is not None:
            output["additional_text_features"] = text_memory[:, -len(additional_text) :]
            output["additional_text_mask"] = text_attention_mask[
                -len(additional_text) :
            ]

        text_memory = text_memory[:, : len(captions)]
        text_attention_mask = text_attention_mask[: len(captions)]
        text_embeds = text_embeds[:, : len(captions)]
        output["language_features"] = text_memory
        output["language_mask"] = text_attention_mask
        output["language_embeds"] = text_embeds
        return output
