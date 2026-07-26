"""Parity tests for Triton two-way attention block composition."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from sam3.primitives import Attention, MLPBlock
from sam3.primitives.two_way_transformer import TwoWayAttentionBlock

if not torch.cuda.is_available():
    pytest.skip("CUDA is required for two-way attention tests", allow_module_level=True)

DEVICE = torch.device("cuda")


def _tol(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float16:
        return 2e-2, 2e-3
    if dtype == torch.bfloat16:
        return 3e-2, 3e-3
    return 1e-2, 5e-3


class _ReferenceAttention(nn.Module):
    """Reference attention module matching the Triton wrapper wiring."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.kv_in_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.v_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)
        self.dropout_p = 0.0

    def _separate_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)

    def _recombine_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_p)
        out = self._recombine_heads(out)
        return self.out_proj(out)


class _ReferenceTwoWayAttentionBlock(nn.Module):
    """Torch mirror of :class:`TwoWayAttentionBlock` with shared weights."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        super().__init__()
        self.self_attn = _ReferenceAttention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = _ReferenceAttention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),
            activation(),
            nn.Linear(mlp_dim, embedding_dim),
        )
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = _ReferenceAttention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        query_pe: torch.Tensor,
        key_pe: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


def _copy_attention_weights(source: Attention, target: _ReferenceAttention) -> None:
    target.q_proj.weight.data.copy_(source.q_proj.weight.data)
    target.q_proj.bias.data.copy_(source.q_proj.bias.data)
    target.k_proj.weight.data.copy_(source.k_proj.weight.data)
    target.k_proj.bias.data.copy_(source.k_proj.bias.data)
    target.v_proj.weight.data.copy_(source.v_proj.weight.data)
    target.v_proj.bias.data.copy_(source.v_proj.bias.data)
    target.out_proj.weight.data.copy_(source.out_proj.weight.data)
    target.out_proj.bias.data.copy_(source.out_proj.bias.data)


def _copy_layernorm_weights(source: nn.LayerNorm, target: nn.LayerNorm) -> None:
    target.weight.data.copy_(source.weight.data)
    target.bias.data.copy_(source.bias.data)


def _copy_mlp_weights(source: MLPBlock, target: nn.Sequential) -> None:
    target[0].weight.data.copy_(source.lin1.weight.data)
    target[0].bias.data.copy_(source.lin1.bias.data)
    target[2].weight.data.copy_(source.lin2.weight.data)
    target[2].bias.data.copy_(source.lin2.bias.data)


def _build_torch_reference(
    source: TwoWayAttentionBlock,
) -> _ReferenceTwoWayAttentionBlock:
    attention_downsample_rate = (
        source.cross_attn_token_to_image.kv_in_dim
        // source.cross_attn_token_to_image.internal_dim
    )
    target = _ReferenceTwoWayAttentionBlock(
        embedding_dim=source.self_attn.embedding_dim,
        num_heads=source.self_attn.num_heads,
        mlp_dim=source.mlp.lin1.out_features,
        activation=nn.ReLU,
        attention_downsample_rate=attention_downsample_rate,
        skip_first_layer_pe=source.skip_first_layer_pe,
    )

    _copy_attention_weights(source.self_attn, target.self_attn)
    _copy_attention_weights(
        source.cross_attn_token_to_image, target.cross_attn_token_to_image
    )
    _copy_attention_weights(
        source.cross_attn_image_to_token, target.cross_attn_image_to_token
    )
    _copy_layernorm_weights(source.norm1, target.norm1)
    _copy_layernorm_weights(source.norm2, target.norm2)
    _copy_layernorm_weights(source.norm3, target.norm3)
    _copy_layernorm_weights(source.norm4, target.norm4)
    _copy_mlp_weights(source.mlp, target.mlp)
    return target


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_two_way_attention_block_matches_torch(dtype: torch.dtype) -> None:
    torch.manual_seed(0)

    triton_block = TwoWayAttentionBlock(
        embedding_dim=32,
        num_heads=4,
        mlp_dim=64,
        activation=nn.ReLU,
        attention_downsample_rate=2,
        skip_first_layer_pe=False,
    ).to(DEVICE, dtype=dtype)
    triton_block.eval()

    torch_block = _build_torch_reference(triton_block).to(DEVICE, dtype=dtype)
    torch_block.eval()

    queries = torch.randn(2, 8, 32, device=DEVICE, dtype=dtype)
    keys = torch.randn(2, 16, 32, device=DEVICE, dtype=dtype)
    query_pe = torch.randn(2, 8, 32, device=DEVICE, dtype=dtype)
    key_pe = torch.randn(2, 16, 32, device=DEVICE, dtype=dtype)

    with torch.no_grad():
        triton_queries, triton_keys = triton_block(
            queries=queries,
            keys=keys,
            query_pe=query_pe,
            key_pe=key_pe,
        )
        ref_queries, ref_keys = torch_block(
            queries=queries,
            keys=keys,
            query_pe=query_pe,
            key_pe=key_pe,
        )

    rtol, atol = _tol(dtype)
    torch.testing.assert_close(triton_queries, ref_queries, rtol=rtol, atol=atol)
    torch.testing.assert_close(triton_keys, ref_keys, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_two_way_attention_block_skip_first_layer_pe_matches_torch(
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(1)

    triton_block = TwoWayAttentionBlock(
        embedding_dim=32,
        num_heads=4,
        mlp_dim=64,
        activation=nn.ReLU,
        attention_downsample_rate=2,
        skip_first_layer_pe=True,
    ).to(DEVICE, dtype=dtype)
    triton_block.eval()

    torch_block = _build_torch_reference(triton_block).to(DEVICE, dtype=dtype)
    torch_block.eval()

    queries = torch.randn(2, 8, 32, device=DEVICE, dtype=dtype)
    keys = torch.randn(2, 16, 32, device=DEVICE, dtype=dtype)
    query_pe = torch.randn(2, 8, 32, device=DEVICE, dtype=dtype)
    key_pe = torch.randn(2, 16, 32, device=DEVICE, dtype=dtype)

    with torch.no_grad():
        triton_queries, triton_keys = triton_block(
            queries=queries,
            keys=keys,
            query_pe=query_pe,
            key_pe=key_pe,
        )
        ref_queries, ref_keys = torch_block(
            queries=queries,
            keys=keys,
            query_pe=query_pe,
            key_pe=key_pe,
        )

    rtol, atol = _tol(dtype)
    torch.testing.assert_close(triton_queries, ref_queries, rtol=rtol, atol=atol)
    torch.testing.assert_close(triton_keys, ref_keys, rtol=rtol, atol=atol)
