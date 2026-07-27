"""SAM3.1 native Multiplex video canonical tensor components."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from sam3.runtime.multiplex_state import MultiplexVariantParameters
from sam3.tracking.tracker_utils import get_1d_sine_pe

from .multiplex import ScatterReplace


class MultiplexFrameEncode(nn.Module):
    """Encode one 1008 frame into distinct interactive and propagation views."""

    def __init__(
        self,
        tri_neck: nn.Module,
        tracker: nn.Module,
        *,
        use_cuda_autocast: bool = False,
    ) -> None:
        super().__init__()
        self.trunk = tri_neck.trunk
        self.position_encoding = tri_neck.position_encoding
        self.interactive_convs = tri_neck.interactive_convs
        self.propagation_convs = tri_neck.propagation_convs
        self.interactive_conv_s0 = tracker.interactive_sam_mask_decoder.conv_s0
        self.interactive_conv_s1 = tracker.interactive_sam_mask_decoder.conv_s1
        self.propagation_conv_s0 = tracker.sam_mask_decoder.conv_s0
        self.propagation_conv_s1 = tracker.sam_mask_decoder.conv_s1
        self.use_cuda_autocast = bool(use_cuda_autocast)

    def forward(self, pixel_values: Tensor) -> tuple[Tensor, ...]:
        if self.use_cuda_autocast:
            with torch.autocast("cuda", dtype=torch.float16):
                return self._forward(pixel_values)
        return self._forward(pixel_values)

    def _forward(self, pixel_values: Tensor) -> tuple[Tensor, ...]:
        trunk_values = self.trunk(pixel_values)
        if not isinstance(trunk_values, (list, tuple)):
            trunk_values = [trunk_values]
        trunk = getattr(trunk_values[-1], "tensors", trunk_values[-1])
        interactive = [layer(trunk) for layer in self.interactive_convs]
        propagation = [layer(trunk) for layer in self.propagation_convs]
        interactive_position = self.position_encoding(interactive[-1]).to(
            interactive[-1].dtype
        )
        propagation_position = self.position_encoding(propagation[-1]).to(
            propagation[-1].dtype
        )
        return (
            interactive[-1],
            interactive_position,
            self.interactive_conv_s0(interactive[0]),
            self.interactive_conv_s1(interactive[1]),
            propagation[-1],
            propagation_position,
            self.propagation_conv_s0(propagation[0]),
            self.propagation_conv_s1(propagation[1]),
        )


class MultiplexInteractionPreview(nn.Module):
    """Non-destructive selected-object interaction in object space."""

    def __init__(
        self,
        tracker: nn.Module,
        variant: MultiplexVariantParameters,
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
        prompt_encoder = self.tracker.interactive_sam_prompt_encoder
        sparse = prompt_encoder._embed_points(coords, labels, pad=True)
        prompt_valid = torch.cat((box_valid, point_valid), dim=1)
        sentinel_valid = torch.ones(
            (batch, 1), dtype=torch.bool, device=point_valid.device
        )
        sparse_valid = torch.cat((prompt_valid, sentinel_valid), dim=1)
        mask_embedding = prompt_encoder._embed_masks(
            mask_input.to(dtype=activation_dtype)
        )
        no_mask = prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand_as(
            mask_embedding
        )
        dense = torch.where(has_mask[:, None, None, None], mask_embedding, no_mask)
        return sparse, dense, sparse_valid

    def forward(
        self,
        interactive_image: Tensor,
        interactive_high_res_0: Tensor,
        interactive_high_res_1: Tensor,
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
                return self._forward(
                    interactive_image,
                    interactive_high_res_0,
                    interactive_high_res_1,
                    point_coords,
                    point_labels,
                    point_valid,
                    box_xyxy,
                    has_box,
                    mask_input,
                    has_mask,
                )
        return self._forward(
            interactive_image,
            interactive_high_res_0,
            interactive_high_res_1,
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
        interactive_image: Tensor,
        interactive_high_res_0: Tensor,
        interactive_high_res_1: Tensor,
        point_coords: Tensor,
        point_labels: Tensor,
        point_valid: Tensor,
        box_xyxy: Tensor,
        has_box: Tensor,
        mask_input: Tensor,
        has_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        dtype = interactive_image.dtype
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
        conditioned = interactive_image + self.tracker.interactivity_no_mem_embed.to(
            dtype=dtype
        ).reshape(1, -1, 1, 1)
        prompt_encoder = self.tracker.interactive_sam_prompt_encoder
        image_pe = prompt_encoder.get_dense_pe().to(
            device=conditioned.device, dtype=dtype
        )
        low_res, scores, output_tokens, object_score = (
            self.tracker.interactive_sam_mask_decoder(
                image_embeddings=conditioned,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse.to(dtype=dtype),
                dense_prompt_embeddings=dense.to(dtype=dtype),
                multimask_output=self.multimask_output,
                repeat_image=False,
                high_res_features=[
                    interactive_high_res_0,
                    interactive_high_res_1,
                ],
                sparse_prompt_valid=sparse_valid,
            )
        )
        appearing = object_score > 0
        low_res = torch.where(
            appearing[..., None, None], low_res, torch.full_like(low_res, -1024.0)
        ).float()
        high_res = F.interpolate(
            low_res,
            size=(self.variant.image_size, self.variant.image_size),
            mode="bilinear",
            align_corners=False,
        )
        if self.multimask_output:
            best = torch.argmax(scores, dim=-1)
            batch_indices = torch.arange(low_res.shape[0], device=low_res.device)
            commit_mask = high_res[batch_indices, best].unsqueeze(1)
            output_token = output_tokens[batch_indices, best]
        else:
            commit_mask = high_res
            output_token = output_tokens[:, 0]
        raw_pointer = self.tracker.interactive_obj_ptr_proj(output_token)
        appearing_float = appearing.to(raw_pointer.dtype)
        pointer = appearing_float * raw_pointer + (
            1 - appearing_float
        ) * self.tracker.no_obj_ptr_linear(raw_pointer)
        return (
            torch.clamp(low_res, -32.0, 32.0),
            scores.float(),
            commit_mask,
            pointer,
            object_score.float(),
        )


class MultiplexInteractionPreviewMultimask3(MultiplexInteractionPreview):
    def __init__(
        self,
        tracker: nn.Module,
        variant: MultiplexVariantParameters,
        *,
        use_cuda_autocast: bool = False,
    ) -> None:
        super().__init__(
            tracker,
            variant,
            multimask_output=True,
            use_cuda_autocast=use_cuda_autocast,
        )


class MultiplexInteractionPreviewSingle1(MultiplexInteractionPreview):
    def __init__(
        self,
        tracker: nn.Module,
        variant: MultiplexVariantParameters,
        *,
        use_cuda_autocast: bool = False,
    ) -> None:
        super().__init__(
            tracker,
            variant,
            multimask_output=False,
            use_cuda_autocast=use_cuda_autocast,
        )


class MultiplexPropagation(nn.Module):
    """Condition and decode one fixed one- or two-bucket propagation step."""

    def __init__(
        self,
        tracker: nn.Module,
        variant: MultiplexVariantParameters,
        *,
        bucket_count: int | None,
        use_cuda_autocast: bool = False,
    ) -> None:
        super().__init__()
        variant.validate()
        if bucket_count not in (1, 2, None):
            raise ValueError("bucket_count must be 1, 2, or bounded-dynamic")
        self.tracker = tracker
        self.variant = variant
        self.bucket_count = bucket_count
        self.use_cuda_autocast = bool(use_cuda_autocast)

    def _memory_condition(
        self,
        image: Tensor,
        image_position: Tensor,
        slot_validity: Tensor,
        memory_features: Tensor,
        memory_position: Tensor,
        memory_image_features: Tensor,
        memory_image_position: Tensor,
        memory_valid: Tensor,
        memory_age: Tensor,
        object_pointers: Tensor,
        pointer_valid: Tensor,
        pointer_age: Tensor,
    ) -> Tensor:
        bucket_count, spatial_capacity, channels, height, width = memory_features.shape
        if self.bucket_count is not None and bucket_count != self.bucket_count:
            raise ValueError("memory bucket count disagrees with fixed profile")
        if spatial_capacity != self.variant.total_spatial_input_capacity:
            raise ValueError("memory input capacity disagrees with MultiplexStateV1")
        valid_memory = memory_valid.to(torch.bool)
        temporal_index = torch.where(
            (memory_age.abs() > 0) & (memory_age.abs() < self.variant.num_maskmem),
            self.variant.num_maskmem - memory_age.abs() - 1,
            torch.full_like(memory_age, self.variant.num_maskmem - 1),
        )
        temporal_table = self.tracker.maskmem_tpos_enc[:, 0, 0]
        temporal = F.embedding(temporal_index, temporal_table)
        positioned_memory = memory_position + temporal[..., None, None]
        memory_tokens = memory_features.permute(1, 3, 4, 0, 2).reshape(
            spatial_capacity * height * width, bucket_count, channels
        )
        memory_position_tokens = positioned_memory.permute(1, 3, 4, 0, 2).reshape(
            spatial_capacity * height * width, bucket_count, channels
        )
        memory_padding = (~valid_memory).repeat_interleave(height * width, dim=1)

        image_temporal = temporal[0]
        positioned_memory_image = (
            memory_image_position + image_temporal[..., None, None]
        )
        memory_image_tokens = memory_image_features.permute(0, 2, 3, 1).reshape(
            spatial_capacity * height * width, 1, channels
        )
        memory_image_position_tokens = positioned_memory_image.permute(
            0, 2, 3, 1
        ).reshape(spatial_capacity * height * width, 1, channels)

        pointer_frames = object_pointers.shape[1]
        pointer_position = get_1d_sine_pe(
            pointer_age.abs().to(torch.float32)
            / float(self.variant.object_pointer_frame_capacity - 1),
            dim=self.variant.hidden_dimension,
        ).to(dtype=self.tracker.obj_ptr_tpos_proj.weight.dtype)
        pointer_position = self.tracker.obj_ptr_tpos_proj(pointer_position)
        pointer_tokens = object_pointers.permute(1, 2, 0, 3).flatten(0, 1)
        pointer_position_tokens = (
            pointer_position[:, :, None, :]
            .expand(-1, -1, self.variant.bucket_capacity, -1)
            .permute(1, 2, 0, 3)
            .flatten(0, 1)
        )
        valid_pointer = pointer_valid.to(torch.bool)
        if valid_pointer.ndim == 2:
            valid_pointer = valid_pointer[..., None]
        valid_pointer = valid_pointer & slot_validity.to(torch.bool)[:, None, :]
        pointer_padding = (~valid_pointer).flatten(1)

        prompt = torch.cat((memory_tokens, pointer_tokens), dim=0)
        prompt_position = torch.cat(
            (memory_position_tokens, pointer_position_tokens), dim=0
        )
        prompt_padding = torch.cat((memory_padding, pointer_padding), dim=1)
        source = image.expand(bucket_count, -1, -1, -1)
        source_position = image_position.expand(bucket_count, -1, -1, -1)
        encoded = self.tracker.transformer.encoder(
            image=image.flatten(2).permute(2, 0, 1),
            src=source.flatten(2).permute(2, 0, 1),
            memory_image=memory_image_tokens,
            memory=prompt,
            image_pos=image_position.flatten(2).permute(2, 0, 1),
            src_pos=source_position.flatten(2).permute(2, 0, 1),
            memory_image_pos=memory_image_position_tokens,
            memory_pos=prompt_position,
            num_obj_ptr_tokens=pointer_frames * self.variant.bucket_capacity,
            memory_key_padding_mask=prompt_padding,
        )["memory"]
        conditioned = encoded.permute(1, 2, 0).reshape_as(source)
        has_state = torch.any(valid_memory, dim=1) | torch.any(
            valid_pointer, dim=(1, 2)
        )
        return torch.where(has_state[:, None, None, None], conditioned, source)

    def forward(
        self,
        propagation_image: Tensor,
        propagation_position: Tensor,
        propagation_high_res_0: Tensor,
        propagation_high_res_1: Tensor,
        slot_validity: Tensor,
        memory_features: Tensor,
        memory_position: Tensor,
        memory_image_features: Tensor,
        memory_image_position: Tensor,
        memory_valid: Tensor,
        memory_age: Tensor,
        object_pointers: Tensor,
        pointer_valid: Tensor,
        pointer_age: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        values = (
            propagation_image,
            propagation_position,
            propagation_high_res_0,
            propagation_high_res_1,
            slot_validity,
            memory_features,
            memory_position,
            memory_image_features,
            memory_image_position,
            memory_valid,
            memory_age,
            object_pointers,
            pointer_valid,
            pointer_age,
        )
        if self.use_cuda_autocast:
            with torch.autocast("cuda", dtype=torch.float16):
                return self._forward(*values)
        return self._forward(*values)

    def _forward(
        self,
        propagation_image: Tensor,
        propagation_position: Tensor,
        propagation_high_res_0: Tensor,
        propagation_high_res_1: Tensor,
        slot_validity: Tensor,
        memory_features: Tensor,
        memory_position: Tensor,
        memory_image_features: Tensor,
        memory_image_position: Tensor,
        memory_valid: Tensor,
        memory_age: Tensor,
        object_pointers: Tensor,
        pointer_valid: Tensor,
        pointer_age: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        conditioned = self._memory_condition(
            propagation_image,
            propagation_position,
            slot_validity,
            memory_features,
            memory_position,
            memory_image_features,
            memory_image_position,
            memory_valid,
            memory_age,
            object_pointers,
            pointer_valid,
            pointer_age,
        )
        validity = slot_validity.to(conditioned.dtype)[..., None]
        bucket_count = conditioned.shape[0]
        extra_embeddings = (
            validity * self.tracker.output_valid_embed
            + (1 - validity) * self.tracker.output_invalid_embed
        )
        image_pe = self.tracker.image_pe_layer(
            self.variant.memory_spatial_size
        ).unsqueeze(0)
        image_pe = image_pe.to(device=conditioned.device, dtype=conditioned.dtype)
        high_res_0 = propagation_high_res_0.expand(bucket_count, -1, -1, -1)
        high_res_1 = propagation_high_res_1.expand(bucket_count, -1, -1, -1)
        low_res, scores, output_tokens, object_score = self.tracker.sam_mask_decoder(
            image_embeddings=conditioned,
            image_pe=image_pe,
            high_res_features=[high_res_0, high_res_1],
            extra_per_object_embeddings=extra_embeddings,
        )
        appearing = (object_score > 0) & slot_validity.to(torch.bool)[..., None]
        low_res = torch.where(
            appearing[..., None, None], low_res, torch.full_like(low_res, -1024.0)
        ).float()
        scores = torch.where(
            slot_validity.to(torch.bool)[..., None], scores, torch.zeros_like(scores)
        ).float()
        object_score = torch.where(
            slot_validity.to(torch.bool)[..., None],
            object_score,
            torch.full_like(object_score, -1024.0),
        ).float()
        best = torch.argmax(scores, dim=2, keepdim=True)
        selected_low = torch.gather(
            low_res,
            2,
            best[..., None, None].expand(-1, -1, -1, *low_res.shape[-2:]),
        )
        selected_token = torch.gather(
            output_tokens,
            2,
            best[..., None].expand(-1, -1, -1, output_tokens.shape[-1]),
        ).squeeze(2)
        high_res = F.interpolate(
            selected_low.flatten(0, 1),
            size=(self.variant.image_size, self.variant.image_size),
            mode="bilinear",
            align_corners=False,
        ).view(
            bucket_count,
            self.variant.bucket_capacity,
            1,
            self.variant.image_size,
            self.variant.image_size,
        )
        raw_pointer = self.tracker.obj_ptr_proj(selected_token)
        appearing_float = appearing.to(raw_pointer.dtype)
        pointer = appearing_float * raw_pointer + (
            1 - appearing_float
        ) * self.tracker.no_obj_ptr_linear(raw_pointer)
        return (
            low_res,
            scores,
            selected_low,
            high_res,
            pointer,
            object_score,
        )


class MultiplexMemoryCommit(nn.Module):
    """Encode bucket-space masks into native shared spatial memory."""

    def __init__(
        self,
        tracker: nn.Module,
        variant: MultiplexVariantParameters,
        *,
        bucket_count: int | None,
        use_cuda_autocast: bool = False,
    ) -> None:
        super().__init__()
        variant.validate()
        if bucket_count not in (1, 2, None):
            raise ValueError("bucket_count must be 1, 2, or bounded-dynamic")
        self.tracker = tracker
        self.variant = variant
        self.bucket_count = bucket_count
        self.use_cuda_autocast = bool(use_cuda_autocast)

    def forward(
        self,
        propagation_image: Tensor,
        bucket_masks: Tensor,
        object_score: Tensor,
        slot_validity: Tensor,
        conditioning_validity: Tensor,
    ) -> tuple[Tensor, Tensor]:
        values = (
            propagation_image,
            bucket_masks,
            object_score,
            slot_validity,
            conditioning_validity,
        )
        if self.use_cuda_autocast:
            with torch.autocast("cuda", dtype=torch.float16):
                return self._forward(*values)
        return self._forward(*values)

    def _forward(
        self,
        propagation_image: Tensor,
        bucket_masks: Tensor,
        object_score: Tensor,
        slot_validity: Tensor,
        conditioning_validity: Tensor,
    ) -> tuple[Tensor, Tensor]:
        masks = bucket_masks.squeeze(2)
        bucket_count = bucket_masks.shape[0]
        validity = slot_validity.to(torch.bool)
        masks = torch.where(validity[..., None, None], masks, torch.zeros_like(masks))
        memory_masks = (
            torch.sigmoid(masks) * self.variant.memory_sigmoid_scale
            + self.variant.memory_sigmoid_bias
        )
        condition_flag = (conditioning_validity.to(torch.bool) & validity).to(
            memory_masks.dtype
        )
        condition = (
            condition_flag * self.variant.condition_mask_foreground
            + (1 - condition_flag) * self.variant.condition_mask_background
        )
        condition = condition[..., None, None].expand_as(memory_masks)
        mask_channels = torch.cat((memory_masks, condition), dim=1)
        frame = propagation_image.expand(bucket_count, -1, -1, -1)
        output = self.tracker.maskmem_backbone(
            frame, mask_channels, skip_mask_sigmoid=True
        )
        memory = output["vision_features"]
        position = output["vision_pos_enc"][0]
        appearing = ((object_score > 0) & validity[..., None]).to(memory.dtype)
        no_object = (
            (1 - appearing) * self.tracker.no_obj_embed_spatial.unsqueeze(0)
        ).sum(dim=1)
        memory = (memory + no_object[..., None, None]).to(dtype=propagation_image.dtype)
        return memory, position


class MultiplexScatterReplaceCommit(nn.Module):
    """Replace one slot, then encode the resulting bucket memory once."""

    def __init__(
        self,
        tracker: nn.Module,
        variant: MultiplexVariantParameters,
        *,
        bucket_count: int | None,
        use_cuda_autocast: bool = False,
    ) -> None:
        super().__init__()
        # The Host Runtime owns and validates this private assignment. Keeping
        # data-dependent assertions out of the graph makes the static cut
        # exportable without weakening the public assignment boundary.
        self.replace_low = ScatterReplace(bucket_count, validate_assignments=False)
        self.replace_high = ScatterReplace(bucket_count, validate_assignments=False)
        self.replace_pointer = ScatterReplace(bucket_count, validate_assignments=False)
        self.replace_score = ScatterReplace(bucket_count, validate_assignments=False)
        self.memory_commit = MultiplexMemoryCommit(
            tracker,
            variant,
            bucket_count=bucket_count,
            use_cuda_autocast=use_cuda_autocast,
        )

    def forward(
        self,
        propagation_image: Tensor,
        bucket_low_res: Tensor,
        bucket_high_res: Tensor,
        bucket_pointers: Tensor,
        bucket_object_scores: Tensor,
        replacement_low_res: Tensor,
        replacement_high_res: Tensor,
        replacement_pointer: Tensor,
        replacement_object_score: Tensor,
        assignment: Tensor,
        slot_validity: Tensor,
        conditioning_validity: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        low_res = self.replace_low(bucket_low_res, replacement_low_res, assignment)
        high_res = self.replace_high(bucket_high_res, replacement_high_res, assignment)
        pointer = self.replace_pointer(bucket_pointers, replacement_pointer, assignment)
        object_score = self.replace_score(
            bucket_object_scores, replacement_object_score, assignment
        )
        memory, position = self.memory_commit(
            propagation_image,
            high_res,
            object_score,
            slot_validity,
            conditioning_validity,
        )
        return low_res, high_res, pointer, object_score, memory, position


__all__ = [
    "MultiplexFrameEncode",
    "MultiplexInteractionPreview",
    "MultiplexInteractionPreviewMultimask3",
    "MultiplexInteractionPreviewSingle1",
    "MultiplexMemoryCommit",
    "MultiplexPropagation",
    "MultiplexScatterReplaceCommit",
]
