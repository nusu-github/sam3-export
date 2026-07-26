"""SAM3.1 decoupled transformer mapping and tensor-shape tests."""

from __future__ import annotations

import torch

from sam3.tracking.multiplex_transformer import create_multiplex_transformer
from sam3.weights.multiplex import (
    load_sam31_multiplex_checkpoint,
    map_checkpoint_to_module,
)


def test_multiplex_transformer_checkpoint_mapping_is_exact() -> None:
    checkpoint = load_sam31_multiplex_checkpoint()
    module = create_multiplex_transformer()
    report = map_checkpoint_to_module(
        checkpoint,
        module,
        prefix="tracker.model.transformer.",
        load=True,
    )
    assert report.exact
    assert report.checkpoint_key_count == report.module_key_count == 122


@torch.inference_mode()
def test_multiplex_transformer_bucket_and_padding_shapes() -> None:
    module = create_multiplex_transformer().eval()
    sequence = 4
    buckets = 2
    memories = 8
    image = torch.randn(sequence, 1, 256)
    src = torch.randn(sequence, buckets, 256)
    memory_image = torch.randn(memories, 1, 256)
    memory = torch.randn(memories, buckets, 256)
    image_pos = torch.randn_like(image)
    src_pos = torch.randn_like(src)
    memory_image_pos = torch.randn_like(memory_image)
    memory_pos = torch.randn_like(memory)
    padding = torch.zeros((buckets, memories), dtype=torch.bool)
    padding[:, -1] = True
    output = module.encoder(
        image=image,
        src=src,
        memory_image=memory_image,
        memory=memory,
        image_pos=image_pos,
        src_pos=src_pos,
        memory_image_pos=memory_image_pos,
        memory_pos=memory_pos,
        memory_key_padding_mask=padding,
    )
    assert output["memory"].shape == src.shape
    assert torch.isfinite(output["memory"]).all()


@torch.inference_mode()
def test_pointer_position_padding_uses_shared_image_batch_for_two_buckets() -> None:
    module = create_multiplex_transformer().eval()
    sequence = 4
    buckets = 2
    spatial_memories = 8
    pointer_tokens = 16
    image = torch.randn(sequence, 1, 256)
    src = torch.randn(sequence, buckets, 256)
    memory_image = torch.randn(spatial_memories, 1, 256)
    memory = torch.randn(spatial_memories + pointer_tokens, buckets, 256)
    output = module.encoder(
        image=image,
        src=src,
        memory_image=memory_image,
        memory=memory,
        image_pos=torch.randn_like(image),
        src_pos=torch.randn_like(src),
        memory_image_pos=torch.randn_like(memory_image),
        memory_pos=torch.randn_like(memory),
        num_obj_ptr_tokens=pointer_tokens,
        memory_key_padding_mask=torch.zeros(
            (buckets, spatial_memories + pointer_tokens), dtype=torch.bool
        ),
    )
    assert output["memory"].shape == src.shape
