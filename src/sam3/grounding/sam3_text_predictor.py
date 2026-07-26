"""High-level image-plus-text grounding API."""

from __future__ import annotations

from typing import Any

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from torchvision.ops import box_convert
from torchvision.transforms import v2

from sam3.dtype_policy import PrecisionConfig, module_param_dtype
from sam3.runtime.nms import nms_masks

from .sam3_image import FindStage, Sam3Image


class Sam3TextPredictor:
    """Preprocess images and run text open-vocab detection/segmentation."""

    def __init__(
        self,
        model: Sam3Image,
        resolution: int = 1008,
        device: str = "cuda",
        confidence_threshold: float | int = 0.5,
        precision: PrecisionConfig | None = None,
        nms_iou_threshold: float | None = None,
    ) -> None:
        self.model = model
        self.resolution = resolution
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.precision = precision
        self.nms_iou_threshold = nms_iou_threshold
        self.transform = v2.Compose(
            [
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(size=(resolution, resolution)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.find_stage = FindStage(
            img_ids=torch.tensor([0], device=device, dtype=torch.long),
            text_ids=torch.tensor([0], device=device, dtype=torch.long),
        )

    def _compute_dtype(self) -> torch.dtype:
        if self.precision is not None:
            return self.precision.compute_dtype
        try:
            return module_param_dtype(self.model)
        except ValueError:
            return next(self.model.parameters()).dtype

    @torch.inference_mode()
    def set_image(
        self,
        image: Any,
        state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if state is None:
            state = {}
        if isinstance(image, PIL.Image.Image):
            width, height = image.size
        elif isinstance(image, (torch.Tensor, np.ndarray)):
            height, width = image.shape[-2:]
        else:
            raise TypeError("Image must be a PIL image or a tensor")

        image_t = v2.functional.to_image(image).to(self.device)
        image_t = self.transform(image_t).unsqueeze(0)
        # Match backbone compute / weight dtype (fp16/bf16 permanent cast).
        image_t = image_t.to(device=self.device, dtype=self._compute_dtype())
        state["original_height"] = height
        state["original_width"] = width
        state["backbone_out"] = self.model.backbone.forward_image(image_t)
        return state

    @torch.inference_mode()
    def set_text_prompt(self, prompt: str, state: dict[str, Any]) -> dict[str, Any]:
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before set_text_prompt")
        text_outputs = self.model.backbone.forward_text([prompt], device=self.device)
        state["backbone_out"].update(text_outputs)
        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()
        return self._forward_grounding(state)

    @torch.inference_mode()
    def _forward_grounding(self, state: dict[str, Any]) -> dict[str, Any]:
        outputs = self.model.forward_grounding(
            backbone_out=state["backbone_out"],
            find_input=self.find_stage,
            geometric_prompt=state["geometric_prompt"],
            find_target=None,
        )
        out_bbox = outputs["pred_boxes"]
        out_logits = outputs["pred_logits"]
        out_masks = outputs["pred_masks"]
        out_probs = out_logits.sigmoid()
        presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
        out_probs = (out_probs * presence_score).squeeze(-1)

        keep = out_probs > self.confidence_threshold
        if self.nms_iou_threshold is not None:
            keep = keep & nms_masks(
                pred_probs=out_probs,
                pred_masks=out_masks,
                prob_threshold=float(self.confidence_threshold),
                iou_threshold=self.nms_iou_threshold,
            )

        out_probs = out_probs[keep]
        out_masks = out_masks[keep]
        out_bbox = out_bbox[keep]

        boxes = box_convert(
            out_bbox.reshape(-1, 4), in_fmt="cxcywh", out_fmt="xyxy"
        ).reshape_as(out_bbox)
        img_h = state["original_height"]
        img_w = state["original_width"]
        scale_fct = torch.tensor(
            [img_w, img_h, img_w, img_h], device=self.device, dtype=boxes.dtype
        )
        boxes = boxes * scale_fct[None, :]

        out_masks = F.interpolate(
            out_masks.unsqueeze(1),
            (img_h, img_w),
            mode="bilinear",
            align_corners=False,
        ).sigmoid()

        state["masks_logits"] = out_masks
        state["masks"] = out_masks > 0.5
        state["boxes"] = boxes
        state["scores"] = out_probs
        state["raw_outputs"] = outputs
        return state
