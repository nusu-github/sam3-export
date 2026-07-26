"""Production interactive image path built from SAM3 components.

Mirrors official ``SAM3InteractiveImagePredictor`` / ``Sam3Image.predict_inst``:

    image → DualViTDetNeck (sam2 branch) → scalp → conv_s0/s1
          → no_mem_embed on lowest map → PromptEncoder → MaskDecoder
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from jaxtyping import Float, Integer
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from sam3.dtype_policy import PrecisionConfig, module_param_dtype

from .necks import Sam3DualViTDetNeck
from .sam_image_head import SamImageHead

# After scalp=1 on 1008 / stride-14 DualFPN: levels 288, 144, 72.
_DEFAULT_BB_FEAT_SIZES = ((288, 288), (144, 144), (72, 72))
_DEFAULT_IMG_SIZE = 1008
_DEFAULT_SCALP = 1


class SamInteractivePredictor(nn.Module):
    """End-to-end point/box interactive masks with real SAM3 weights.

    Components:
      * ``backbone``: ``Sam3DualViTDetNeck`` with ``add_sam2_neck=True``
      * ``head``: production ``SamImageHead`` (high-res features on)
      * ``no_mem_embed``: tracker spatial bias on the lowest-res map
    """

    def __init__(
        self,
        backbone: Sam3DualViTDetNeck,
        head: SamImageHead,
        *,
        no_mem_embed: Float[Tensor, "1 1 c"] | None = None,
        image_size: int = _DEFAULT_IMG_SIZE,
        scalp: int = _DEFAULT_SCALP,
        bb_feat_sizes: Sequence[tuple[int, int]] = _DEFAULT_BB_FEAT_SIZES,
        mask_threshold: float | int = 0.0,
        precision: PrecisionConfig | None = None,
    ) -> None:
        super().__init__()
        if backbone.sam2_convs is None:
            raise ValueError("backbone must be built with add_sam2_neck=True")
        self.backbone = backbone
        self.head = head
        self.image_size = int(image_size)
        self.scalp = int(scalp)
        self.bb_feat_sizes = tuple(bb_feat_sizes)
        self.mask_threshold = float(mask_threshold)
        self.precision = precision
        hidden = head.embed_dim
        if no_mem_embed is None:
            no_mem_embed = torch.zeros(1, 1, hidden)
        if tuple(no_mem_embed.shape) != (1, 1, hidden):
            raise ValueError(
                f"no_mem_embed expected shape (1,1,{hidden}), got {tuple(no_mem_embed.shape)}"
            )
        self.no_mem_embed = nn.Parameter(no_mem_embed.clone().detach())

        # Cached per set_image
        self._is_image_set = False
        self._orig_hw: tuple[int, int] | None = None
        self._image_embed: Tensor | None = None
        self._high_res: list[Tensor] | None = None

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _compute_dtype(self) -> torch.dtype:
        if self.precision is not None:
            return self.precision.compute_dtype
        try:
            return module_param_dtype(self)
        except Exception:
            return next(self.parameters()).dtype

    def reset_image(self) -> None:
        self._is_image_set = False
        self._orig_hw = None
        self._image_embed = None
        self._high_res = None

    def preprocess(
        self,
        image: Any,
        *,
        dtype: torch.dtype | None = None,
    ) -> tuple[Float[Tensor, "..."], tuple[int, int]]:
        """PIL / HWC uint8 / BCHW float → model tensor + original (H, W)."""
        from torchvision.transforms import v2

        if dtype is None:
            dtype = self._compute_dtype()

        if hasattr(image, "size") and not torch.is_tensor(image):
            # PIL
            ow, oh = image.size
            img_t = v2.functional.to_image(image)
        elif torch.is_tensor(image):
            if image.ndim == 3 and image.shape[0] in (1, 3):
                # CHW
                img_t = image
                oh, ow = int(image.shape[-2]), int(image.shape[-1])
            elif image.ndim == 3:
                # HWC
                oh, ow = int(image.shape[0]), int(image.shape[1])
                img_t = image.permute(2, 0, 1)
            elif image.ndim == 4:
                oh, ow = int(image.shape[-2]), int(image.shape[-1])
                # already batched; handle below
                x = image.to(device=self.device, dtype=dtype)
                if x.shape[-2:] != (self.image_size, self.image_size):
                    x = F.interpolate(
                        x,
                        size=(self.image_size, self.image_size),
                        mode="bilinear",
                        align_corners=False,
                    ).to(dtype=dtype)
                # assume already normalized if float and in ~[-1,1] range is caller responsibility
                return x, (oh, ow)
            else:
                raise ValueError(f"unsupported tensor shape {tuple(image.shape)}")
        else:
            raise TypeError(f"unsupported image type {type(image)}")

        t = v2.Compose(
            [
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(size=(self.image_size, self.image_size)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        x = t(img_t.to(self.device)).unsqueeze(0).to(dtype=dtype)
        return x, (oh, ow)

    def encode_image_tensor(
        self, image_bchw: Float[Tensor, "b 3 h w"]
    ) -> tuple[Float[Tensor, "b c h_e w_e"], list[Float[Tensor, "b c_hr h_hr w_hr"]]]:
        """Run backbone + high-res projs. ``image_bchw`` is preprocessed model input."""
        _sam3, _pos, sam2, _pos2 = self.backbone(image_bchw)
        if sam2 is None:
            raise RuntimeError("sam2 neck produced no features")
        fpn = list(sam2)
        if self.scalp > 0:
            fpn = fpn[: -self.scalp]
        n_levels = len(self.bb_feat_sizes)
        if len(fpn) < n_levels:
            raise RuntimeError(
                f"expected >= {n_levels} FPN levels after scalp, got {len(fpn)}"
            )
        fpn = fpn[-n_levels:]

        # Project high-res levels once (official does this at set_image time).
        fpn[0] = self.head.mask_decoder.conv_s0(fpn[0])
        fpn[1] = self.head.mask_decoder.conv_s1(fpn[1])
        no_mem = self.no_mem_embed.to(device=fpn[2].device, dtype=fpn[2].dtype)
        image_embed = fpn[2] + no_mem.reshape(1, -1, 1, 1)
        high_res = [fpn[0], fpn[1]]
        return image_embed, high_res

    @torch.inference_mode()
    def set_image(
        self,
        image: Any,
        *,
        dtype: torch.dtype | None = None,
    ) -> tuple[int, int]:
        """Compute and cache image embeddings. Returns original (H, W)."""
        self.reset_image()
        x, orig_hw = self.preprocess(image, dtype=dtype)
        image_embed, high_res = self.encode_image_tensor(x)
        self._image_embed = image_embed
        self._high_res = high_res
        self._orig_hw = orig_hw
        self._is_image_set = True
        return orig_hw

    def _coords_to_model(
        self,
        point_coords: Float[Tensor, "... 2"],
        *,
        normalize: bool,
        orig_hw: tuple[int, int],
    ) -> Float[Tensor, "... 2"]:
        """Pixel (or normalized) coords → model-frame absolute coords on image_size."""
        coords = point_coords.to(device=self.device, dtype=torch.float32).clone()
        if normalize:
            oh, ow = orig_hw
            coords[..., 0] = coords[..., 0] / ow
            coords[..., 1] = coords[..., 1] / oh
        coords = coords * self.image_size
        return coords

    @torch.inference_mode()
    def predict(
        self,
        point_coords: Float[Tensor, "..."]
        | Sequence[Sequence[float | int]]
        | None = None,
        point_labels: Integer[Tensor, "..."] | Sequence[int] | None = None,
        box: Float[Tensor, "..."] | Sequence[float | int] | None = None,
        multimask_output: bool = True,
        normalize_coords: bool = True,
        return_logits: bool = False,
    ) -> tuple[Tensor, Float[Tensor, "..."], Float[Tensor, "..."]]:
        """Predict masks for the cached image.

        Returns:
            masks: ``(C, H, W)`` bool (or float logits if ``return_logits``)
            iou: ``(C,)`` quality scores
            low_res: ``(C, h, w)`` low-res logits (for iterative prompts)
        """
        if not self._is_image_set or self._image_embed is None or self._orig_hw is None:
            raise RuntimeError("call set_image(...) before predict(...)")

        dtype = self._image_embed.dtype
        orig_hw = self._orig_hw
        points = None
        boxes = None

        if point_coords is not None:
            if point_labels is None:
                raise ValueError("point_labels required when point_coords is set")
            coords = torch.as_tensor(point_coords, dtype=torch.float32)
            labels = torch.as_tensor(point_labels, dtype=torch.int64)
            if coords.ndim == 2:
                coords = coords.unsqueeze(0)
                labels = labels.unsqueeze(0)
            coords = self._coords_to_model(
                coords, normalize=normalize_coords, orig_hw=orig_hw
            ).to(dtype=dtype)
            labels = labels.to(device=self.device)
            points = (coords, labels)

        if box is not None:
            b = torch.as_tensor(box, dtype=torch.float32).view(1, 4)
            # box as two corners → transform each corner
            corners = b.view(1, 2, 2)
            corners = self._coords_to_model(
                corners, normalize=normalize_coords, orig_hw=orig_hw
            )
            boxes = corners.view(1, 4).to(device=self.device, dtype=dtype)

        low_res, iou, _tokens, _obj = self.head(
            image_embeddings=self._image_embed,
            points=points,
            boxes=boxes,
            multimask_output=multimask_output,
            high_res_features=self._high_res,
        )
        # low_res: (1, C, h, w) → upsample
        full = F.interpolate(
            low_res.float(), size=orig_hw, mode="bilinear", align_corners=False
        )
        low_res = torch.clamp(low_res, -32.0, 32.0)
        if return_logits:
            masks = full.squeeze(0)
        else:
            masks = full.squeeze(0) > self.mask_threshold
        return masks, iou.squeeze(0).float(), low_res.squeeze(0).float()

    def forward(
        self,
        image: Any,
        point_coords: Float[Tensor, "..."] | None = None,
        point_labels: Integer[Tensor, "..."] | None = None,
        **kwargs: Any,
    ) -> tuple[Tensor, Float[Tensor, "..."], Float[Tensor, "..."]]:
        """One-shot set_image + predict (not cached across calls)."""
        self.set_image(image)
        return self.predict(
            point_coords=point_coords, point_labels=point_labels, **kwargs
        )
