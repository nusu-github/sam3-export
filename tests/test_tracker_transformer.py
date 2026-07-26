"""CUDA tests for the tracker memory-attention transformer."""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip(
        "CUDA is required for tracker_transformer tests", allow_module_level=True
    )

from sam3.primitives.rope_attention import RoPEAttention
from sam3.tracking.tracker_transformer import (
    TransformerDecoderLayerv2,
    TransformerEncoderCrossAttention,
    create_tracker_transformer,
)

DEVICE = torch.device("cuda")


def _build_tiny_layer(
    *,
    d_model: int = 64,
    num_heads: int = 1,
    dim_ff: int = 128,
    feat_hw: int = 8,
    kv_in_dim: int = 16,
    dropout: float = 0.0,
    use_rope_real: bool = False,
) -> TransformerDecoderLayerv2:
    feat_sizes = (feat_hw, feat_hw)
    self_attn = RoPEAttention(
        embedding_dim=d_model,
        num_heads=num_heads,
        downsample_rate=1,
        dropout=dropout,
        rope_theta=10000.0,
        feat_sizes=feat_sizes,
        use_rope_real=use_rope_real,
    )
    cross_attn = RoPEAttention(
        embedding_dim=d_model,
        num_heads=num_heads,
        downsample_rate=1,
        dropout=dropout,
        kv_in_dim=kv_in_dim,
        rope_theta=10000.0,
        feat_sizes=feat_sizes,
        rope_k_repeat=True,
        use_rope_real=use_rope_real,
    )
    return TransformerDecoderLayerv2(
        cross_attention_first=False,
        activation="relu",
        dim_feedforward=dim_ff,
        dropout=dropout,
        pos_enc_at_attn=False,
        pre_norm=True,
        self_attention=self_attn,
        d_model=d_model,
        pos_enc_at_cross_attn_keys=True,
        pos_enc_at_cross_attn_queries=False,
        cross_attention=cross_attn,
    )


def _build_tiny_encoder(
    *,
    d_model: int = 64,
    num_layers: int = 2,
    num_heads: int = 1,
    dim_ff: int = 128,
    feat_hw: int = 8,
    kv_in_dim: int = 16,
    dropout: float = 0.0,
    use_rope_real: bool = False,
    batch_first: bool = True,
) -> TransformerEncoderCrossAttention:
    layer = _build_tiny_layer(
        d_model=d_model,
        num_heads=num_heads,
        dim_ff=dim_ff,
        feat_hw=feat_hw,
        kv_in_dim=kv_in_dim,
        dropout=dropout,
        use_rope_real=use_rope_real,
    )
    return TransformerEncoderCrossAttention(
        d_model=d_model,
        frozen=False,
        pos_enc_at_input=True,
        layer=layer,
        num_layers=num_layers,
        use_act_checkpoint=False,
        batch_first=batch_first,
        remove_cross_attention_layers=[],
    )


def test_layer_v2_forward_finite():
    """Single Layerv2: batch-first tiny dims, finite residual path."""
    d_model, feat_hw, kv_in_dim = 64, 8, 16
    n_src = feat_hw * feat_hw  # 64
    n_mem = n_src * 2  # longer memory → rope_k_repeat
    b = 2

    layer = _build_tiny_layer(d_model=d_model, feat_hw=feat_hw, kv_in_dim=kv_in_dim).to(
        DEVICE
    )
    layer.eval()

    tgt = torch.randn(b, n_src, d_model, device=DEVICE)
    memory = torch.randn(b, n_mem, kv_in_dim, device=DEVICE)
    query_pos = torch.randn(b, n_src, d_model, device=DEVICE)
    pos = torch.randn(b, n_mem, kv_in_dim, device=DEVICE)

    with torch.no_grad():
        out = layer(
            tgt=tgt,
            memory=memory,
            query_pos=query_pos,
            pos=pos,
            num_k_exclude_rope=0,
        )

    assert out.shape == (b, n_src, d_model)
    assert torch.isfinite(out).all()


def test_layer_v2_num_k_exclude_rope():
    """Cross-attn can exclude trailing object-pointer keys from RoPE."""
    d_model, feat_hw, kv_in_dim = 64, 8, 16
    n_src = feat_hw * feat_hw
    n_ptr = 3
    n_mem = n_src + n_ptr
    b = 1

    layer = _build_tiny_layer(d_model=d_model, feat_hw=feat_hw, kv_in_dim=kv_in_dim).to(
        DEVICE
    )
    layer.eval()

    tgt = torch.randn(b, n_src, d_model, device=DEVICE)
    memory = torch.randn(b, n_mem, kv_in_dim, device=DEVICE)
    query_pos = torch.zeros(b, n_src, d_model, device=DEVICE)
    # PE only meaningful for spatial mem keys; zeros on ptrs is fine
    pos = torch.randn(b, n_mem, kv_in_dim, device=DEVICE)

    with torch.no_grad():
        out = layer(
            tgt=tgt,
            memory=memory,
            query_pos=query_pos,
            pos=pos,
            num_k_exclude_rope=n_ptr,
        )

    assert out.shape == (b, n_src, d_model)
    assert torch.isfinite(out).all()


def test_encoder_batch_first_finite_and_shapes():
    """Full encoder with batch_first=True and sequence-first public I/O."""
    d_model, feat_hw, kv_in_dim = 64, 8, 16
    n_src = feat_hw * feat_hw
    n_mem = n_src * 2
    b = 2
    num_layers = 2

    enc = _build_tiny_encoder(
        d_model=d_model,
        num_layers=num_layers,
        feat_hw=feat_hw,
        kv_in_dim=kv_in_dim,
        batch_first=True,
    ).to(DEVICE)
    enc.eval()

    # The public API is sequence-first when batch_first=True (internally transposed).
    src = torch.randn(n_src, b, d_model, device=DEVICE)
    prompt = torch.randn(n_mem, b, kv_in_dim, device=DEVICE)
    src_pos = torch.randn(n_src, b, d_model, device=DEVICE)
    prompt_pos = torch.randn(n_mem, b, kv_in_dim, device=DEVICE)

    with torch.no_grad():
        out = enc(
            src=src,
            prompt=prompt,
            src_pos=src_pos,
            prompt_pos=prompt_pos,
            num_obj_ptr_tokens=0,
        )

    assert set(out.keys()) == {"memory", "pos_embed", "padding_mask"}
    assert out["memory"].shape == (n_src, b, d_model)
    assert out["pos_embed"].shape == (n_src, b, d_model)
    assert out["padding_mask"] is None
    assert torch.isfinite(out["memory"]).all()


def test_encoder_with_obj_ptr_tokens():
    """num_obj_ptr_tokens forwarded as num_k_exclude_rope to RoPE cross-attn."""
    d_model, feat_hw, kv_in_dim = 64, 8, 16
    n_src = feat_hw * feat_hw
    n_ptr = 2
    n_mem = n_src + n_ptr
    b = 1

    enc = _build_tiny_encoder(
        d_model=d_model,
        num_layers=2,
        feat_hw=feat_hw,
        kv_in_dim=kv_in_dim,
        batch_first=True,
    ).to(DEVICE)
    enc.eval()

    src = torch.randn(n_src, b, d_model, device=DEVICE)
    prompt = torch.randn(n_mem, b, kv_in_dim, device=DEVICE)
    src_pos = torch.randn(n_src, b, d_model, device=DEVICE)
    prompt_pos = torch.randn(n_mem, b, kv_in_dim, device=DEVICE)

    with torch.no_grad():
        out = enc(
            src=src,
            prompt=prompt,
            src_pos=src_pos,
            prompt_pos=prompt_pos,
            num_obj_ptr_tokens=n_ptr,
        )

    assert out["memory"].shape == (n_src, b, d_model)
    assert torch.isfinite(out["memory"]).all()


def test_create_tracker_transformer_config():
    """Production factory defaults match model_builder._create_tracker_transformer."""
    enc = create_tracker_transformer()
    assert enc.d_model == 256
    assert enc.num_layers == 4
    assert enc.batch_first is True
    assert enc.pos_enc_at_input is True
    assert len(enc.layers) == 4

    layer0 = enc.layers[0]
    assert isinstance(layer0, TransformerDecoderLayerv2)
    assert layer0.pre_norm is True
    assert layer0.cross_attention_first is False
    assert layer0.d_model == 256
    assert layer0.dim_feedforward == 2048

    self_attn = layer0.self_attn
    cross_attn = layer0.cross_attn_image
    assert isinstance(self_attn, RoPEAttention)
    assert isinstance(cross_attn, RoPEAttention)
    assert self_attn.embedding_dim == 256
    assert self_attn.num_heads == 1
    assert self_attn.rope_k_repeat is False
    assert self_attn.use_rope_real is True  # production preference
    assert cross_attn.kv_in_dim == 64
    assert cross_attn.rope_k_repeat is True
    assert cross_attn.use_rope_real is True


def test_create_tracker_transformer_tiny_forward():
    """Factory with tiny dims runs end-to-end on CUDA."""
    d_model, feat_hw = 64, 8
    n_src = feat_hw * feat_hw
    n_mem = n_src * 2
    b = 1
    kv_in_dim = 16

    enc = create_tracker_transformer(
        d_model=d_model,
        num_layers=2,
        feat_sizes=(feat_hw, feat_hw),
        num_heads=1,
        dim_feedforward=128,
        dropout=0.0,
        kv_in_dim=kv_in_dim,
        use_rope_real=True,
    ).to(DEVICE)
    enc.eval()

    src = torch.randn(n_src, b, d_model, device=DEVICE)
    prompt = torch.randn(n_mem, b, kv_in_dim, device=DEVICE)
    src_pos = torch.randn(n_src, b, d_model, device=DEVICE)
    prompt_pos = torch.randn(n_mem, b, kv_in_dim, device=DEVICE)

    with torch.no_grad():
        out = enc(src=src, prompt=prompt, src_pos=src_pos, prompt_pos=prompt_pos)

    assert out["memory"].shape == (n_src, b, d_model)
    assert torch.isfinite(out["memory"]).all()


def test_remove_cross_attention_layers():
    """Layers marked for removal skip cross-attn (self-attn + FFN only)."""
    d_model, feat_hw, kv_in_dim = 64, 8, 16
    n_src = feat_hw * feat_hw
    n_mem = n_src
    b = 1

    layer = _build_tiny_layer(d_model=d_model, feat_hw=feat_hw, kv_in_dim=kv_in_dim)
    enc = TransformerEncoderCrossAttention(
        d_model=d_model,
        frozen=False,
        pos_enc_at_input=True,
        layer=layer,
        num_layers=2,
        batch_first=True,
        remove_cross_attention_layers=[1],  # second layer: no CA
    ).to(DEVICE)
    enc.eval()

    assert enc.layers[0].cross_attn_image is not None
    assert enc.layers[1].cross_attn_image is None

    src = torch.randn(n_src, b, d_model, device=DEVICE)
    prompt = torch.randn(n_mem, b, kv_in_dim, device=DEVICE)
    src_pos = torch.randn(n_src, b, d_model, device=DEVICE)
    prompt_pos = torch.randn(n_mem, b, kv_in_dim, device=DEVICE)

    with torch.no_grad():
        out = enc(src=src, prompt=prompt, src_pos=src_pos, prompt_pos=prompt_pos)

    assert out["memory"].shape == (n_src, b, d_model)
    assert torch.isfinite(out["memory"]).all()
