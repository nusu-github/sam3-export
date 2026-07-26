"""Production-shape interactive image deployment components for M3."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


class InteractiveFeatureProject(nn.Module):
    """Project the production SAM2 FPN view without temporal conditioning."""

    def __init__(self, mask_decoder: nn.Module) -> None:
        super().__init__()
        if not hasattr(mask_decoder, "conv_s0") or not hasattr(mask_decoder, "conv_s1"):
            raise TypeError("production mask decoder must expose conv_s0 and conv_s1")
        self.conv_s0 = mask_decoder.conv_s0
        self.conv_s1 = mask_decoder.conv_s1

    def forward(
        self, fpn_0: Tensor, fpn_1: Tensor, fpn_2: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        return fpn_2, self.conv_s0(fpn_0), self.conv_s1(fpn_1)


class InitialNoMemoryCondition(nn.Module):
    """Apply the checkpoint-owned initial/no-memory image condition."""

    def __init__(self, no_mem_embed: nn.Parameter) -> None:
        super().__init__()
        if tuple(no_mem_embed.shape) != (1, 1, 256):
            raise ValueError("no_mem_embed must have shape [1,1,256]")
        self.no_mem_embed = no_mem_embed

    def forward(self, image_embedding: Tensor) -> Tensor:
        condition = self.no_mem_embed.to(
            device=image_embedding.device, dtype=image_embedding.dtype
        )
        return image_embedding + condition.reshape(1, -1, 1, 1)


class InteractiveImageEncodeInitial(nn.Module):
    """Fused image-only recipe: vision + FPN projection + initial condition."""

    def __init__(
        self,
        backbone: nn.Module,
        feature_project: InteractiveFeatureProject,
        initial_condition: InitialNoMemoryCondition,
        *,
        scalp: int = 1,
        feature_sizes: Sequence[tuple[int, int]] = (
            (288, 288),
            (144, 144),
            (72, 72),
        ),
        use_cuda_autocast: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.feature_project = feature_project
        self.initial_condition = initial_condition
        self.scalp = int(scalp)
        self.feature_sizes = tuple(feature_sizes)
        self.use_cuda_autocast = bool(use_cuda_autocast)

    def forward(self, pixel_values: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if self.use_cuda_autocast:
            with torch.autocast("cuda", dtype=torch.float16):
                return self._forward(pixel_values)
        return self._forward(pixel_values)

    def _forward(self, pixel_values: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        _detector, _detector_pos, sam2, _sam2_pos = self.backbone(pixel_values)
        if sam2 is None:
            raise RuntimeError("SAM2 FPN is required by the interactive image plan")
        levels = list(sam2)
        if self.scalp:
            levels = levels[: -self.scalp]
        levels = levels[-len(self.feature_sizes) :]
        if len(levels) != len(self.feature_sizes):
            raise RuntimeError("interactive image plan requires three FPN levels")
        base, high_res_0, high_res_1 = self.feature_project(
            levels[0], levels[1], levels[2]
        )
        image_embedding = self.initial_condition(base)
        return image_embedding, high_res_0, high_res_1


class InteractivePredict(nn.Module):
    """Fixed-capacity production prompt encoder plus one static mask policy."""

    def __init__(
        self,
        head: nn.Module,
        *,
        multimask_output: bool,
        use_cuda_autocast: bool = False,
    ) -> None:
        super().__init__()
        self.prompt_encoder = head.prompt_encoder
        self.mask_decoder = head.mask_decoder
        self.multimask_output = bool(multimask_output)
        self.use_cuda_autocast = bool(use_cuda_autocast)

    def _encode_prompts(
        self,
        point_coords: Tensor,
        point_labels: Tensor,
        point_valid: Tensor,
        box_xyxy: Tensor,
        has_box: Tensor,
        mask_input: Tensor,
        has_mask: Tensor,
        activation_dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor, Tensor]:
        box_coords = box_xyxy.reshape(1, 2, 2)
        box_labels = torch.tensor(
            [[2, 3]], dtype=point_labels.dtype, device=point_labels.device
        )
        box_valid = has_box[:, None].expand(-1, 2)
        box_labels = torch.where(box_valid, box_labels, torch.full_like(box_labels, -1))
        labels = torch.where(
            point_valid, point_labels, torch.full_like(point_labels, -1)
        )
        coords = torch.cat((box_coords, point_coords), dim=1)
        labels = torch.cat((box_labels, labels), dim=1)
        sparse = self.prompt_encoder._embed_points(coords, labels, pad=True)

        prompt_valid = torch.cat((box_valid, point_valid), dim=1)
        sentinel_valid = torch.any(prompt_valid, dim=1, keepdim=True)
        sparse_valid = torch.cat((prompt_valid, sentinel_valid), dim=1)

        mask_embedding = self.prompt_encoder._embed_masks(
            mask_input.to(dtype=activation_dtype)
        )
        no_mask = self.prompt_encoder.no_mask_embed.weight.reshape(
            1, -1, 1, 1
        ).expand_as(mask_embedding)
        dense = torch.where(has_mask[:, None, None, None], mask_embedding, no_mask)
        return sparse, dense, sparse_valid

    def forward(
        self,
        image_embedding: Tensor,
        high_res_0: Tensor,
        high_res_1: Tensor,
        point_coords: Tensor,
        point_labels: Tensor,
        point_valid: Tensor,
        box_xyxy: Tensor,
        has_box: Tensor,
        mask_input: Tensor,
        has_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if self.use_cuda_autocast:
            with torch.autocast("cuda", dtype=torch.float16):
                return self._forward(
                    image_embedding,
                    high_res_0,
                    high_res_1,
                    point_coords,
                    point_labels,
                    point_valid,
                    box_xyxy,
                    has_box,
                    mask_input,
                    has_mask,
                )
        return self._forward(
            image_embedding,
            high_res_0,
            high_res_1,
            point_coords,
            point_labels,
            point_valid,
            box_xyxy,
            has_box,
            mask_input,
            has_mask,
        )

    def _forward(
        self,
        image_embedding: Tensor,
        high_res_0: Tensor,
        high_res_1: Tensor,
        point_coords: Tensor,
        point_labels: Tensor,
        point_valid: Tensor,
        box_xyxy: Tensor,
        has_box: Tensor,
        mask_input: Tensor,
        has_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        dtype = image_embedding.dtype
        sparse, dense, sparse_valid = self._encode_prompts(
            point_coords,
            point_labels,
            point_valid,
            box_xyxy,
            has_box,
            mask_input,
            has_mask,
            dtype,
        )
        image_pe = self.prompt_encoder.get_dense_pe().to(
            device=image_embedding.device, dtype=dtype
        )
        low_res, scores, _tokens, _object_score = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse.to(dtype=dtype),
            dense_prompt_embeddings=dense.to(dtype=dtype),
            multimask_output=self.multimask_output,
            repeat_image=False,
            high_res_features=[high_res_0, high_res_1],
            sparse_prompt_valid=sparse_valid,
        )
        return torch.clamp(low_res.float(), -32.0, 32.0), scores.float()


class InteractivePredictMultimask3(InteractivePredict):
    def __init__(self, head: nn.Module, *, use_cuda_autocast: bool = False) -> None:
        super().__init__(
            head,
            multimask_output=True,
            use_cuda_autocast=use_cuda_autocast,
        )


class InteractivePredictSingle1(InteractivePredict):
    def __init__(self, head: nn.Module, *, use_cuda_autocast: bool = False) -> None:
        super().__init__(
            head,
            multimask_output=False,
            use_cuda_autocast=use_cuda_autocast,
        )


__all__ = [
    "InitialNoMemoryCondition",
    "InteractiveFeatureProject",
    "InteractiveImageEncodeInitial",
    "InteractivePredict",
    "InteractivePredictMultimask3",
    "InteractivePredictSingle1",
]
