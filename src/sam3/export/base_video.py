"""Production tensor components for M4 SAM3 base video tracking."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from sam3.runtime.base_video_state import BaseVideoVariantParameters
from sam3.tracking.tracker_utils import get_1d_sine_pe


class TrackerFrameEncode(nn.Module):
    """Encode one frame into the unconditioned tracker image view."""

    def __init__(self, tracker: nn.Module, *, use_cuda_autocast: bool = False) -> None:
        super().__init__()
        if getattr(tracker, "backbone", None) is None:
            raise TypeError("TrackerFrameEncode requires a backbone-attached tracker")
        self.tracker = tracker
        self.use_cuda_autocast = bool(use_cuda_autocast)

    def forward(self, pixel_values: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if self.use_cuda_autocast:
            with torch.autocast("cuda", dtype=torch.float16):
                return self._forward(pixel_values)
        return self._forward(pixel_values)

    def _forward(self, pixel_values: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        backbone = self.tracker.forward_image(pixel_values)
        _, features, positions, sizes = self.tracker._prepare_backbone_features(
            backbone
        )
        high_res_0 = (
            features[0]
            .permute(1, 2, 0)
            .reshape(pixel_values.shape[0], features[0].shape[2], *sizes[0])
        )
        high_res_1 = (
            features[1]
            .permute(1, 2, 0)
            .reshape(pixel_values.shape[0], features[1].shape[2], *sizes[1])
        )
        image_embedding = (
            features[-1]
            .permute(1, 2, 0)
            .reshape(pixel_values.shape[0], features[-1].shape[2], *sizes[-1])
        )
        image_position = positions[-1].permute(1, 2, 0).reshape_as(image_embedding)
        return image_embedding, image_position, high_res_0, high_res_1


class BaseTrackerPreview(nn.Module):
    """Memory-aware fixed-capacity preview without a state mutation."""

    def __init__(
        self,
        tracker: nn.Module,
        variant: BaseVideoVariantParameters,
        *,
        multimask_output: bool,
        use_cuda_autocast: bool = False,
    ) -> None:
        super().__init__()
        variant.validate()
        self.tracker = tracker
        self.variant = variant
        self.multimask_output = bool(multimask_output)
        self.use_cuda_autocast = bool(use_cuda_autocast)

    def _memory_condition(
        self,
        image_embedding: Tensor,
        image_position: Tensor,
        object_valid: Tensor,
        memory_features: Tensor,
        memory_position: Tensor,
        memory_valid: Tensor,
        memory_age: Tensor,
        memory_conditioning: Tensor,
        object_pointers: Tensor,
        pointer_valid: Tensor,
        pointer_age: Tensor,
        pointer_conditioning: Tensor,
        pointer_tpos_denominator: Tensor,
    ) -> Tensor:
        batch, slots, _channels, height, width = memory_features.shape
        valid_objects = object_valid.to(torch.bool)
        valid_memory = memory_valid.to(torch.bool) & valid_objects[:, None]
        valid_pointers = pointer_valid.to(torch.bool) & valid_objects[:, None]
        has_memory = torch.any(valid_memory, dim=1)

        # Conditioning frames always use the final temporal embedding. The six
        # non-conditioning entries are packed at signed ages +/-1..+/-6.
        temporal_index = memory_age.abs() - 1
        temporal_index = temporal_index.clamp(0, self.variant.num_maskmem - 1)
        temporal_index = torch.where(
            memory_conditioning.to(torch.bool),
            torch.full_like(temporal_index, self.variant.num_maskmem - 1),
            temporal_index,
        )
        temporal_table = self.tracker.maskmem_tpos_enc[:, 0, 0]
        temporal = F.embedding(temporal_index, temporal_table)
        memory_position = memory_position + temporal[:, :, :, None, None]

        memory_tokens = memory_features.permute(1, 3, 4, 0, 2).reshape(
            slots * height * width, batch, -1
        )
        memory_position_tokens = memory_position.permute(1, 3, 4, 0, 2).reshape(
            slots * height * width, batch, -1
        )
        memory_padding = (~valid_memory).repeat_interleave(height * width, dim=1)

        # A masked-out object with no memory still needs one numerically safe key;
        # the transformer result is discarded in favor of the official no-memory
        # route below.
        safe_first = memory_padding[:, :1] & has_memory[:, None]
        memory_padding = torch.cat((safe_first, memory_padding[:, 1:]), dim=1)

        pointer_position = get_1d_sine_pe(
            pointer_age.abs().to(torch.float32)
            / pointer_tpos_denominator[:, None].to(torch.float32).clamp_min(1.0),
            dim=self.variant.hidden_dimension,
        ).to(dtype=self.tracker.obj_ptr_tpos_proj.weight.dtype)
        pointer_position = self.tracker.obj_ptr_tpos_proj(pointer_position)
        # Base SAM3 has no learned conditioning-pointer offset. Retain the flag
        # as a first-class ABI input while producing the official base value.
        pointer_position = pointer_position + pointer_conditioning.to(
            pointer_position.dtype
        )[..., None] * torch.zeros_like(pointer_position)

        split = self.variant.hidden_dimension // self.variant.memory_dimension
        pointer_tokens = object_pointers.reshape(
            batch,
            self.variant.object_pointer_capacity,
            split,
            self.variant.memory_dimension,
        ).permute(1, 2, 0, 3)
        pointer_tokens = pointer_tokens.flatten(0, 1)
        pointer_position_tokens = (
            pointer_position[:, :, None, :]
            .expand(-1, -1, split, -1)
            .permute(1, 2, 0, 3)
        )
        pointer_position_tokens = pointer_position_tokens.flatten(0, 1)
        pointer_padding = (~valid_pointers).repeat_interleave(split, dim=1)

        prompt = torch.cat((memory_tokens, pointer_tokens), dim=0)
        prompt_position = torch.cat(
            (memory_position_tokens, pointer_position_tokens), dim=0
        )
        prompt_padding = torch.cat((memory_padding, pointer_padding), dim=1)
        source = image_embedding.flatten(2).permute(2, 0, 1)
        source_position = image_position.flatten(2).permute(2, 0, 1)
        encoded = self.tracker.transformer.encoder(
            src=[source],
            src_key_padding_mask=[None],
            src_pos=[source_position],
            prompt=prompt,
            prompt_pos=prompt_position,
            prompt_key_padding_mask=prompt_padding,
            feat_sizes=[image_embedding.shape[-2:]],
            num_obj_ptr_tokens=pointer_tokens.shape[0],
        )["memory"]
        memory_view = encoded.permute(1, 2, 0).reshape_as(image_embedding)
        no_memory = image_embedding + self.tracker.no_mem_embed.reshape(1, -1, 1, 1)
        return torch.where(has_memory[:, None, None, None], memory_view, no_memory)

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
        batch = point_coords.shape[0]
        box_coords = box_xyxy.reshape(batch, 2, 2)
        box_labels = (
            torch.tensor([2, 3], dtype=point_labels.dtype, device=point_labels.device)
            .reshape(1, 2)
            .expand(batch, -1)
        )
        box_valid = has_box[:, None].expand(-1, 2)
        box_labels = torch.where(box_valid, box_labels, torch.full_like(box_labels, -1))
        labels = torch.where(
            point_valid, point_labels, torch.full_like(point_labels, -1)
        )
        coords = torch.cat((box_coords, point_coords), dim=1)
        labels = torch.cat((box_labels, labels), dim=1)
        sparse = self.tracker.sam_prompt_encoder._embed_points(coords, labels, pad=True)
        prompt_valid = torch.cat((box_valid, point_valid), dim=1)
        sentinel_valid = torch.ones(
            (batch, 1), dtype=torch.bool, device=point_valid.device
        )
        sparse_valid = torch.cat((prompt_valid, sentinel_valid), dim=1)

        mask_embedding = self.tracker.sam_prompt_encoder._embed_masks(
            mask_input.to(dtype=activation_dtype)
        )
        no_mask = self.tracker.sam_prompt_encoder.no_mask_embed.weight.reshape(
            1, -1, 1, 1
        ).expand_as(mask_embedding)
        dense = torch.where(has_mask[:, None, None, None], mask_embedding, no_mask)
        return sparse, dense, sparse_valid

    def _predict(
        self,
        conditioned: Tensor,
        high_res_0: Tensor,
        high_res_1: Tensor,
        point_coords: Tensor,
        point_labels: Tensor,
        point_valid: Tensor,
        box_xyxy: Tensor,
        has_box: Tensor,
        mask_input: Tensor,
        has_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        dtype = conditioned.dtype
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
        batch = point_coords.shape[0]
        empty_coords = torch.zeros(
            (batch, 1, 2), dtype=point_coords.dtype, device=point_coords.device
        )
        empty_labels = torch.full(
            (batch, 1), -1, dtype=point_labels.dtype, device=point_labels.device
        )
        empty_sparse = self.tracker.sam_prompt_encoder._embed_points(
            empty_coords, empty_labels, pad=True
        )
        empty_dense = self.tracker.sam_prompt_encoder.no_mask_embed.weight.reshape(
            1, -1, 1, 1
        ).expand_as(dense)
        empty_valid = torch.ones(
            empty_sparse.shape[:2], dtype=torch.bool, device=point_valid.device
        )
        image_pe = self.tracker.sam_prompt_encoder.get_dense_pe().to(
            device=conditioned.device, dtype=dtype
        )
        decoder_kwargs = {
            "image_embeddings": conditioned,
            "image_pe": image_pe,
            "multimask_output": self.multimask_output,
            "repeat_image": False,
            "high_res_features": [high_res_0, high_res_1],
        }
        prompted = self.tracker.sam_mask_decoder(
            sparse_prompt_embeddings=sparse.to(dtype=dtype),
            dense_prompt_embeddings=dense.to(dtype=dtype),
            sparse_prompt_valid=sparse_valid,
            **decoder_kwargs,
        )
        empty = self.tracker.sam_mask_decoder(
            sparse_prompt_embeddings=empty_sparse.to(dtype=dtype),
            dense_prompt_embeddings=empty_dense.to(dtype=dtype),
            sparse_prompt_valid=empty_valid,
            **decoder_kwargs,
        )
        has_prompt = (
            point_valid.any(dim=1) | has_box.to(torch.bool) | has_mask.to(torch.bool)
        )
        low_res, scores, output_tokens, object_score = tuple(
            torch.where(
                has_prompt.reshape((batch,) + (1,) * (prompted_value.ndim - 1)),
                prompted_value,
                empty_value,
            )
            for prompted_value, empty_value in zip(prompted, empty)
        )
        appearing = object_score > 0
        low_res = torch.where(appearing[:, None, None], low_res, -1024.0).float()
        high_res = F.interpolate(
            low_res,
            size=(self.variant.memory_spatial_size[0] * 14,) * 2,
            mode="bilinear",
            align_corners=False,
        )
        output_token = output_tokens[:, 0]
        if self.multimask_output:
            best = torch.argmax(scores, dim=-1)
            batch_indices = torch.arange(low_res.shape[0], device=low_res.device)
            commit_mask = high_res[batch_indices, best].unsqueeze(1)
            if output_tokens.shape[1] > 1:
                output_token = output_tokens[batch_indices, best]
        else:
            commit_mask = high_res
        object_pointer = self.tracker.obj_ptr_proj(output_token)
        appearing_float = appearing.float()
        object_pointer = appearing_float * object_pointer
        object_pointer = (
            object_pointer + (1 - appearing_float) * self.tracker.no_obj_ptr
        )
        return (
            torch.clamp(low_res, -32.0, 32.0),
            scores.float(),
            commit_mask,
            object_pointer,
            object_score.float(),
        )

    def forward(
        self,
        image_embedding: Tensor,
        image_position: Tensor,
        high_res_0: Tensor,
        high_res_1: Tensor,
        object_valid: Tensor,
        memory_features: Tensor,
        memory_position: Tensor,
        memory_valid: Tensor,
        memory_age: Tensor,
        memory_conditioning: Tensor,
        object_pointers: Tensor,
        pointer_valid: Tensor,
        pointer_age: Tensor,
        pointer_conditioning: Tensor,
        pointer_tpos_denominator: Tensor,
        point_coords: Tensor,
        point_labels: Tensor,
        point_valid: Tensor,
        box_xyxy: Tensor,
        has_box: Tensor,
        mask_input: Tensor,
        has_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if self.use_cuda_autocast:
            with torch.autocast("cuda", dtype=torch.float16):
                return self._forward_impl(
                    image_embedding,
                    image_position,
                    high_res_0,
                    high_res_1,
                    object_valid,
                    memory_features,
                    memory_position,
                    memory_valid,
                    memory_age,
                    memory_conditioning,
                    object_pointers,
                    pointer_valid,
                    pointer_age,
                    pointer_conditioning,
                    pointer_tpos_denominator,
                    point_coords,
                    point_labels,
                    point_valid,
                    box_xyxy,
                    has_box,
                    mask_input,
                    has_mask,
                )
        return self._forward_impl(
            image_embedding,
            image_position,
            high_res_0,
            high_res_1,
            object_valid,
            memory_features,
            memory_position,
            memory_valid,
            memory_age,
            memory_conditioning,
            object_pointers,
            pointer_valid,
            pointer_age,
            pointer_conditioning,
            pointer_tpos_denominator,
            point_coords,
            point_labels,
            point_valid,
            box_xyxy,
            has_box,
            mask_input,
            has_mask,
        )

    def _forward_impl(
        self, *values: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        conditioned = self._memory_condition(values[0], values[1], *values[4:15])
        return self._predict(conditioned, values[2], values[3], *values[15:])


class BaseTrackerPreviewMultimask3(BaseTrackerPreview):
    def __init__(
        self,
        tracker: nn.Module,
        variant: BaseVideoVariantParameters,
        *,
        use_cuda_autocast: bool = False,
    ) -> None:
        super().__init__(
            tracker,
            variant,
            multimask_output=True,
            use_cuda_autocast=use_cuda_autocast,
        )


class BaseTrackerPreviewSingle1(BaseTrackerPreview):
    def __init__(
        self,
        tracker: nn.Module,
        variant: BaseVideoVariantParameters,
        *,
        use_cuda_autocast: bool = False,
    ) -> None:
        super().__init__(
            tracker,
            variant,
            multimask_output=False,
            use_cuda_autocast=use_cuda_autocast,
        )


class BaseMemoryCommit(nn.Module):
    """Convert one final single preview into conditioning/non-conditioning memory."""

    def __init__(
        self,
        tracker: nn.Module,
        variant: BaseVideoVariantParameters,
        *,
        use_cuda_autocast: bool = False,
    ) -> None:
        super().__init__()
        variant.validate()
        self.tracker = tracker
        self.variant = variant
        self.use_cuda_autocast = bool(use_cuda_autocast)

    def forward(
        self,
        image_embedding: Tensor,
        commit_mask: Tensor,
        object_score: Tensor,
        is_mask_from_points: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if self.use_cuda_autocast:
            with torch.autocast("cuda", dtype=torch.float16):
                return self._forward_impl(
                    image_embedding, commit_mask, object_score, is_mask_from_points
                )
        return self._forward_impl(
            image_embedding, commit_mask, object_score, is_mask_from_points
        )

    def _forward_impl(
        self,
        image_embedding: Tensor,
        commit_mask: Tensor,
        object_score: Tensor,
        is_mask_from_points: Tensor,
    ) -> tuple[Tensor, Tensor]:
        binary = (commit_mask > 0).float()
        probability = torch.sigmoid(commit_mask)
        mask = torch.where(
            is_mask_from_points[:, None, None, None], binary, probability
        )
        mask = (
            mask * self.variant.memory_sigmoid_scale + self.variant.memory_sigmoid_bias
        )
        output: Mapping[str, object] = self.tracker.maskmem_backbone(
            image_embedding, mask, skip_mask_sigmoid=True
        )
        memory = output["vision_features"]
        position = output["vision_pos_enc"][0]
        appearing = (object_score > 0).float()
        memory = memory + (1 - appearing[..., None, None]) * (
            self.tracker.no_obj_embed_spatial[..., None, None].expand_as(memory)
        )
        return memory, position


class BaseTrackerStepAndCommitSingle1(nn.Module):
    """Candidate fused steady-state single preview and memory commit."""

    def __init__(
        self,
        preview: BaseTrackerPreviewSingle1,
        commit: BaseMemoryCommit,
    ) -> None:
        super().__init__()
        self.preview = preview
        self.commit = commit

    def forward(self, *values: Tensor) -> tuple[Tensor, ...]:
        low_res, scores, commit_mask, pointer, object_score = self.preview(*values)
        is_mask_from_points = torch.zeros_like(values[-1], dtype=torch.bool)
        memory, position = self.commit(
            values[0], commit_mask, object_score, is_mask_from_points
        )
        return (
            low_res,
            scores,
            commit_mask,
            pointer,
            object_score,
            memory,
            position,
        )


__all__ = [
    "BaseMemoryCommit",
    "BaseTrackerPreview",
    "BaseTrackerPreviewMultimask3",
    "BaseTrackerPreviewSingle1",
    "BaseTrackerStepAndCommitSingle1",
    "TrackerFrameEncode",
]
