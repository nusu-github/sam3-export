"""SAM3.1 shared mask-memory encoder mapping tests."""

from __future__ import annotations

import torch

from sam3.tracking.memory import create_maskmem_backbone
from sam3.weights.multiplex import (
    load_sam31_multiplex_checkpoint,
    map_checkpoint_to_module,
)


def _module(interpol_size: tuple[int, int] = (1152, 1152)):
    return create_maskmem_backbone(
        out_dim=256,
        in_dim=256,
        fuser_layers=2,
        pe_dim=256,
        interpol_size=interpol_size,
        precompute_resolution=None,
        multiplex_count=16,
        starting_out_chan=4,
        input_channel_multiplier=2,
        official_fuser_names=True,
    )


def test_multiplex_memory_checkpoint_mapping_is_exact() -> None:
    checkpoint = load_sam31_multiplex_checkpoint()
    module = _module()
    report = map_checkpoint_to_module(
        checkpoint,
        module,
        prefix="tracker.model.maskmem_backbone.",
        load=True,
    )
    assert report.exact
    assert report.checkpoint_key_count == report.module_key_count == 38


@torch.inference_mode()
def test_multiplex_memory_has_shared_bucket_output() -> None:
    module = _module((64, 64)).eval()
    image = torch.randn(1, 256, 4, 4)
    masks_and_condition = torch.randn(1, 32, 64, 64)
    output = module(image, masks_and_condition, skip_mask_sigmoid=True)
    assert output["vision_features"].shape == (1, 256, 4, 4)
    assert output["vision_pos_enc"][0].shape == (1, 256, 4, 4)
