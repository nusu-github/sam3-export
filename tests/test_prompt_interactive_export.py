"""Cut D/E: test-only PromptEncode and InteractiveDecode export fixtures."""

from __future__ import annotations

import pytest
import torch
from torch.export import export

from sam3.export.interactive_decode import (
    INTERACTIVE_NUM_MASKS,
    InteractiveDecode,
    InteractiveDecodeSpec,
)
from sam3.export.prompt_encode import (
    PROMPT_EMBED_DIM,
    PROMPT_IMAGE_EMBED_HW,
    PROMPT_N_SPARSE,
    PromptEncode,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")

DEVICE = torch.device("cuda")


@pytest.fixture
def prompt_mod() -> PromptEncode:
    return PromptEncode(validate=True).to(DEVICE).eval()


@pytest.fixture
def decode_mod() -> InteractiveDecode:
    return InteractiveDecode(validate=True).to(DEVICE).eval()


@torch.no_grad()
def test_prompt_encode_shapes(prompt_mod: PromptEncode):
    b = 1
    coords = torch.tensor([[[4.0, 4.0], [8.0, 8.0]]], device=DEVICE)
    labels = torch.tensor([[1, 0]], device=DEVICE)
    sparse, dense = prompt_mod(coords, labels)
    assert sparse.shape[0] == b
    assert sparse.shape[1] == PROMPT_N_SPARSE
    assert sparse.shape[2] == PROMPT_EMBED_DIM
    eh, ew = PROMPT_IMAGE_EMBED_HW
    assert dense.shape == (b, PROMPT_EMBED_DIM, eh, ew)


@torch.no_grad()
def test_prompt_encode_export(prompt_mod: PromptEncode):
    coords = torch.tensor([[[4.0, 4.0], [8.0, 8.0]]], device=DEVICE)
    labels = torch.tensor([[1, 0]], device=DEVICE)
    eager_s, eager_d = prompt_mod(coords, labels)
    ep = export(prompt_mod, (coords, labels), strict=False)
    s, d = ep.module()(coords, labels)
    assert torch.isfinite(s).all() and torch.isfinite(d).all()
    torch.testing.assert_close(eager_s, s, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(eager_d, d, rtol=1e-4, atol=1e-4)


@torch.no_grad()
def test_interactive_decode_shapes(decode_mod: InteractiveDecode):
    spec = InteractiveDecodeSpec()
    img = torch.randn(1, spec.embed_dim, *spec.image_embedding_size, device=DEVICE)
    coords = torch.tensor([[[4.0, 4.0], [8.0, 8.0]]], device=DEVICE)
    labels = torch.tensor([[1, 0]], device=DEVICE)
    masks, iou = decode_mod(img, coords, labels)
    assert masks.shape == (1, INTERACTIVE_NUM_MASKS, *spec.input_image_size)
    assert iou.shape == (1, INTERACTIVE_NUM_MASKS)


@torch.no_grad()
def test_interactive_decode_export(decode_mod: InteractiveDecode):
    spec = InteractiveDecodeSpec()
    img = torch.randn(1, spec.embed_dim, *spec.image_embedding_size, device=DEVICE)
    coords = torch.tensor([[[4.0, 4.0], [8.0, 8.0]]], device=DEVICE)
    labels = torch.tensor([[1, 0]], device=DEVICE)
    eager_m, eager_i = decode_mod(img, coords, labels)
    ep = export(decode_mod, (img, coords, labels), strict=False)
    m, i = ep.module()(img, coords, labels)
    assert torch.isfinite(m).all() and torch.isfinite(i).all()
    torch.testing.assert_close(eager_m, m, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(eager_i, i, rtol=1e-3, atol=1e-3)


@torch.no_grad()
def test_prompt_encode_rejects_wrong_n(prompt_mod: PromptEncode):
    coords = torch.zeros(1, 3, 2, device=DEVICE)
    labels = torch.zeros(1, 3, device=DEVICE, dtype=torch.long)
    with pytest.raises(ValueError, match="N="):
        prompt_mod(coords, labels)
