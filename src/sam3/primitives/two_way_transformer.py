"""Two-way transformer primitives used by prompted-mask heads."""

from __future__ import annotations

from typing import Type

from jaxtyping import Float
from torch import Tensor, nn

from .attention import Attention
from .mlp import MLPBlock


class TwoWayAttentionBlock(nn.Module):
    """Transformer block used by SAM3 mask decoder two-way attention.

    This is a pure composition wrapper around the shared ``Attention``,
    ``LayerNorm``, and ``MLPBlock`` modules.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim,
            num_heads,
            downsample_rate=attention_downsample_rate,
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim,
            num_heads,
            downsample_rate=attention_downsample_rate,
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self,
        queries: Float[Tensor, "b n_q c"],
        keys: Float[Tensor, "b n_k c"],
        query_pe: Float[Tensor, "b n_q c"],
        key_pe: Float[Tensor, "b n_k c"],
        query_valid: Tensor | None = None,
    ) -> tuple[Float[Tensor, "b n_q c"], Float[Tensor, "b n_k c"]]:
        if self.skip_first_layer_pe:
            queries = self.self_attn(
                q=queries, k=queries, v=queries, key_valid=query_valid
            )
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries, key_valid=query_valid)
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
        attn_out = self.cross_attn_image_to_token(
            q=k, k=q, v=queries, key_valid=query_valid
        )
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class TwoWayTransformer(nn.Module):
    """Optional two-way transformer using :class:`TwoWayAttentionBlock` layers."""

    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList(
            [
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
                for i in range(depth)
            ]
        )

        self.final_attn_token_to_image = Attention(
            embedding_dim,
            num_heads,
            downsample_rate=attention_downsample_rate,
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Float[Tensor, "b c h w"],
        image_pe: Float[Tensor, "b c h w"],
        point_embedding: Float[Tensor, "b n c"],
        point_valid: Tensor | None = None,
    ) -> tuple[Float[Tensor, "b n c"], Float[Tensor, "b hw c"]]:
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        queries = point_embedding
        keys = image_embedding

        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
                query_valid=point_valid,
            )

        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys
