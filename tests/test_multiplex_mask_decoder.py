"""SAM3.1 16-slot mask decoder checkpoint and isolation gates."""

from __future__ import annotations

import torch

from sam3.vision.multiplex_mask_decoder import create_multiplex_mask_decoder
from sam3.weights.multiplex import (
    load_sam31_multiplex_checkpoint,
    map_checkpoint_to_module,
)


def test_multiplex_decoder_checkpoint_mapping_is_exact() -> None:
    checkpoint = load_sam31_multiplex_checkpoint()
    module = create_multiplex_mask_decoder()
    report = map_checkpoint_to_module(
        checkpoint,
        module,
        prefix="tracker.model.sam_mask_decoder.",
        load=True,
    )
    assert report.exact
    assert report.checkpoint_key_count == report.module_key_count == 125


@torch.inference_mode()
def test_multiplex_decoder_has_native_slot_axis() -> None:
    module = create_multiplex_mask_decoder().eval()
    image = torch.randn(1, 256, 4, 4)
    position = torch.randn_like(image)
    high_resolution = [
        torch.randn(1, 32, 16, 16),
        torch.randn(1, 64, 8, 8),
    ]
    validity_embeddings = torch.randn(1, 16, 256)
    masks, scores, tokens, object_scores = module(
        image, position, high_resolution, validity_embeddings
    )
    assert masks.shape == (1, 16, 3, 16, 16)
    assert scores.shape == (1, 16, 3)
    assert tokens.shape == (1, 16, 3, 256)
    assert object_scores.shape == (1, 16, 1)
