"""Parity and shape tests for the tokenizer and VE text encoder."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from sam3.grounding.text_encoder_ve import (
    TextTransformer,
    VETextEncoder,
    text_global_pool,
)
from sam3.grounding.tokenizer_ve import SimpleTokenizer

if not torch.cuda.is_available():
    pytest.skip("CUDA is required for text encoder tests", allow_module_level=True)

DEVICE = torch.device("cuda")


@pytest.fixture(scope="module")
def tokenizer() -> SimpleTokenizer:
    return SimpleTokenizer(context_length=32)


def test_tokenize_shapes(tokenizer: SimpleTokenizer) -> None:
    texts = ["a cat", "person riding a bicycle", ""]
    tokens = tokenizer(texts, context_length=32)
    assert tokens.dtype == torch.long
    assert tokens.shape == (3, 32)
    # SOT is always first non-pad token
    assert (tokens[:, 0] == tokenizer.sot_token_id).all()
    # Empty string still has SOT + EOT
    assert tokens[2, 0] == tokenizer.sot_token_id
    assert tokens[2, 1] == tokenizer.eot_token_id
    assert (tokens[2, 2:] == 0).all()


def test_tokenizer_default_bpe_path() -> None:
    tok = SimpleTokenizer()  # uses DEFAULT_BPE_PATH
    out = tok(["hello world"], context_length=16)
    assert out.shape == (1, 16)
    assert out[0, 0] == tok.sot_token_id


def test_text_global_pool_argmax() -> None:
    # token ids: pad=0, eot has largest id at positions 2 and 3
    text = torch.tensor([[49406, 100, 49407, 0], [49406, 1, 2, 49407]], device=DEVICE)
    x = torch.randn(2, 4, 8, device=DEVICE)
    pooled, tokens = text_global_pool(x, text, pool_type="argmax")
    assert pooled.shape == (2, 8)
    assert torch.allclose(pooled[0], x[0, 2])
    assert torch.allclose(pooled[1], x[1, 3])
    assert tokens is x


def _init_text_transformer_like_clip(m: nn.Module) -> None:
    """Deterministic-ish init so empty params are finite."""
    for name, p in m.named_parameters():
        if p.dim() >= 2:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.zeros_(p)
    # causal mask buffer already set


def test_text_transformer_state_dict_keys() -> None:
    """Checkpoint-compatible module names (MHA / Linear / LayerNorm)."""
    m = TextTransformer(
        context_length=8,
        vocab_size=1000,
        width=32,
        heads=4,
        layers=1,
        use_act_checkpoint=False,
    )
    keys = m.state_dict().keys()
    assert "token_embedding.weight" in keys
    assert "positional_embedding" in keys
    assert "transformer.resblocks.0.attn.in_proj_weight" in keys
    assert "transformer.resblocks.0.attn.out_proj.weight" in keys
    assert "transformer.resblocks.0.mlp.c_fc.weight" in keys
    assert "transformer.resblocks.0.mlp.c_proj.weight" in keys
    assert "transformer.resblocks.0.ln_1.weight" in keys
    assert "ln_final.weight" in keys
    assert "text_projection" in keys


def test_vetext_encoder_tiny_forward_shapes(tokenizer: SimpleTokenizer) -> None:
    d_model = 32
    width = 64
    heads = 4
    layers = 2
    ctx = 16
    enc = (
        VETextEncoder(
            d_model=d_model,
            tokenizer=tokenizer,
            width=width,
            heads=heads,
            layers=layers,
            context_length=ctx,
            use_act_checkpoint=False,
        )
        .to(DEVICE)
        .eval()
    )
    _init_text_transformer_like_clip(enc)

    texts = ["a cat", "two dogs playing"]
    with torch.no_grad():
        mask, memory, embeds = enc(texts, device=DEVICE)

    b = len(texts)
    # mask: padding True (inverted for pytorch) — [B, S]
    assert mask.shape == (b, ctx)
    assert mask.dtype == torch.bool
    # memory sequence-first after resizer: [S, B, d_model]
    assert memory.shape == (ctx, b, d_model)
    assert torch.isfinite(memory).all()
    # inputs_embeds sequence-first: [S, B, width]
    assert embeds.shape == (ctx, b, width)
    assert torch.isfinite(embeds).all()
    # pad positions should be True in inverted mask
    # non-pad tokens have False
    assert mask[:, 0].eq(False).all()  # SOT always present
