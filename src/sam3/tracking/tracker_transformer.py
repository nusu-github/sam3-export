"""Memory-attention transformer for video tracking.

Port of inference-critical modules from ``sam3.model.decoder``:

* ``TransformerDecoderLayerv1`` (base; supports RoPE or MHA-style inject)
* ``TransformerDecoderLayerv2`` (used by the production tracker)
* ``TransformerEncoderCrossAttention`` (stack with optional batch_first)

Attention modules are **injected** :class:`RoPEAttention` instances (not
``nn.MultiheadAttention``). Factory ``create_tracker_transformer`` mirrors
``_create_tracker_transformer`` in ``sam3.model_builder``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import copy
from typing import Any

from jaxtyping import Bool, Float
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ..primitives.rope_attention import RoPEAttention

# ---------------------------------------------------------------------------
# Helpers (from model_misc) — kept private to this module
# ---------------------------------------------------------------------------


def get_activation_fn(activation: str) -> Callable[..., Tensor]:
    """Return an activation function given a string."""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def get_clones(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


# ---------------------------------------------------------------------------
# Decoder layers
# ---------------------------------------------------------------------------


class TransformerDecoderLayerv1(nn.Module):
    """Pre/post-norm decoder layer with injected self + cross attention.

    Official tracker path uses :class:`TransformerDecoderLayerv2` (pre-norm only
    with RoPE ``q,k,v`` kwargs). This base keeps the MHA-style ``value=`` /
    ``[0]`` API for completeness / parity with ``sam3.model.decoder``.
    """

    def __init__(
        self,
        activation: str,
        cross_attention: nn.Module,
        d_model: int,
        dim_feedforward: int,
        dropout: float | int,
        pos_enc_at_attn: bool,
        pos_enc_at_cross_attn_keys: bool,
        pos_enc_at_cross_attn_queries: bool,
        pre_norm: bool,
        self_attention: nn.Module,
    ):
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout_value = dropout
        self.self_attn = self_attention
        self.cross_attn_image = cross_attention

        # Feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation_str = activation
        self.activation = get_activation_fn(activation)
        self.pre_norm = pre_norm

        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys

    def forward_post(
        self,
        tgt: Float[Tensor, "..."],
        memory: Float[Tensor, "..."],
        tgt_mask: Float[Tensor, "..."] | Bool[Tensor, "..."] | None = None,
        memory_mask: Float[Tensor, "..."] | Bool[Tensor, "..."] | None = None,
        tgt_key_padding_mask: Float[Tensor, "..."] | Bool[Tensor, "..."] | None = None,
        memory_key_padding_mask: Float[Tensor, "..."]
        | Bool[Tensor, "..."]
        | None = None,
        pos: Float[Tensor, "..."] | None = None,
        query_pos: Float[Tensor, "..."] | None = None,
        **kwargs: Any,
    ) -> Float[Tensor, "..."]:
        q = k = tgt + query_pos if self.pos_enc_at_attn else tgt

        # Self attention (MHA-style: returns (out, weights))
        tgt2 = self.self_attn(
            q,
            k,
            value=tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross attention to image
        tgt2 = self.cross_attn_image(
            query=tgt + query_pos if self.pos_enc_at_cross_attn_queries else tgt,
            key=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(
        self,
        tgt: Float[Tensor, "..."],
        memory: Float[Tensor, "..."],
        dac: bool = False,
        tgt_mask: Float[Tensor, "..."] | Bool[Tensor, "..."] | None = None,
        memory_mask: Float[Tensor, "..."] | Bool[Tensor, "..."] | None = None,
        tgt_key_padding_mask: Float[Tensor, "..."] | Bool[Tensor, "..."] | None = None,
        memory_key_padding_mask: Float[Tensor, "..."]
        | Bool[Tensor, "..."]
        | None = None,
        pos: Float[Tensor, "..."] | None = None,
        query_pos: Float[Tensor, "..."] | None = None,
        attn_bias: Float[Tensor, "..."] | None = None,
        **kwargs: Any,
    ) -> Float[Tensor, "..."]:
        if dac:
            assert tgt.shape[0] % 2 == 0
            other_tgt = tgt[tgt.shape[0] // 2 :]
            tgt = tgt[: tgt.shape[0] // 2]
        tgt2 = self.norm1(tgt)
        q = k = tgt2 + query_pos if self.pos_enc_at_attn else tgt2
        tgt2 = self.self_attn(
            q,
            k,
            value=tgt2,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        if dac:
            tgt = torch.cat((tgt, other_tgt), dim=0)
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn_image(
            query=tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2,
            key=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            attn_bias=attn_bias,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(
        self,
        tgt: Float[Tensor, "..."],
        memory: Float[Tensor, "..."],
        dac: bool = False,
        tgt_mask: Float[Tensor, "..."] | Bool[Tensor, "..."] | None = None,
        memory_mask: Float[Tensor, "..."] | Bool[Tensor, "..."] | None = None,
        tgt_key_padding_mask: Float[Tensor, "..."] | Bool[Tensor, "..."] | None = None,
        memory_key_padding_mask: Float[Tensor, "..."]
        | Bool[Tensor, "..."]
        | None = None,
        pos: Float[Tensor, "..."] | None = None,
        query_pos: Float[Tensor, "..."] | None = None,
        attn_bias: Float[Tensor, "..."] | None = None,
        **kwds: Any,
    ) -> Float[Tensor, "..."]:
        fwd_fn = self.forward_pre if self.pre_norm else self.forward_post
        return fwd_fn(
            tgt,
            memory,
            dac=dac,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            pos=pos,
            query_pos=query_pos,
            attn_bias=attn_bias,
            **kwds,
        )


class TransformerDecoderLayerv2(TransformerDecoderLayerv1):
    """Tracker layer: pre-norm SA/CA via RoPEAttention ``(q,k,v=...)`` API."""

    def __init__(self, cross_attention_first: bool = False, *args: Any, **kwds: Any):
        super().__init__(*args, **kwds)
        self.cross_attention_first = cross_attention_first

    def _forward_sa(
        self, tgt: Float[Tensor, "..."], query_pos: Float[Tensor, "..."] | None
    ) -> Float[Tensor, "..."]:
        # Self-Attention
        tgt2 = self.norm1(tgt)
        q = k = tgt2 + query_pos if self.pos_enc_at_attn else tgt2
        tgt2 = self.self_attn(q, k, v=tgt2)
        tgt = tgt + self.dropout1(tgt2)
        return tgt

    def _forward_ca(
        self,
        tgt: Float[Tensor, "..."],
        memory: Float[Tensor, "..."],
        query_pos: Float[Tensor, "..."] | None,
        pos: Float[Tensor, "..."] | None,
        num_k_exclude_rope: int = 0,
    ) -> Float[Tensor, "..."]:
        if self.cross_attn_image is None:
            return tgt

        kwds: dict[str, Any] = {}
        if num_k_exclude_rope > 0:
            assert isinstance(self.cross_attn_image, RoPEAttention)
            kwds = {"num_k_exclude_rope": num_k_exclude_rope}

        # Cross-Attention
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn_image(
            q=tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2,
            k=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
            v=memory,
            **kwds,
        )
        tgt = tgt + self.dropout2(tgt2)
        return tgt

    def forward_pre(
        self,
        tgt: Float[Tensor, "..."],
        memory: Float[Tensor, "..."],
        dac: bool = False,
        tgt_mask: Float[Tensor, "..."] | Bool[Tensor, "..."] | None = None,
        memory_mask: Float[Tensor, "..."] | Bool[Tensor, "..."] | None = None,
        tgt_key_padding_mask: Float[Tensor, "..."] | Bool[Tensor, "..."] | None = None,
        memory_key_padding_mask: Float[Tensor, "..."]
        | Bool[Tensor, "..."]
        | None = None,
        pos: Float[Tensor, "..."] | None = None,
        query_pos: Float[Tensor, "..."] | None = None,
        attn_bias: Float[Tensor, "..."] | None = None,
        num_k_exclude_rope: int = 0,
    ) -> Float[Tensor, "..."]:
        assert dac is False
        assert tgt_mask is None
        assert memory_mask is None
        assert tgt_key_padding_mask is None
        assert memory_key_padding_mask is None
        assert attn_bias is None

        if self.cross_attention_first:
            tgt = self._forward_ca(tgt, memory, query_pos, pos, num_k_exclude_rope)
            tgt = self._forward_sa(tgt, query_pos)
        else:
            tgt = self._forward_sa(tgt, query_pos)
            tgt = self._forward_ca(tgt, memory, query_pos, pos, num_k_exclude_rope)

        # MLP
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, *args: Any, **kwds: Any) -> Float[Tensor, "..."]:
        if self.pre_norm:
            return self.forward_pre(*args, **kwds)
        raise NotImplementedError("TransformerDecoderLayerv2 requires pre_norm=True")


# ---------------------------------------------------------------------------
# Encoder stack (memory attention)
# ---------------------------------------------------------------------------


class TransformerEncoderCrossAttention(nn.Module):
    """Stack of decoder-style layers: self-attn on src + cross-attn to prompt.

    Matches ``sam3.model.decoder.TransformerEncoderCrossAttention`` including
    ``batch_first=True`` (tracker) and ``num_obj_ptr_tokens`` →
    ``num_k_exclude_rope`` for RoPE cross-attention.
    """

    def __init__(
        self,
        d_model: int,
        frozen: bool,
        pos_enc_at_input: bool,
        layer: nn.Module,
        num_layers: int,
        use_act_checkpoint: bool = False,
        batch_first: bool = False,
        remove_cross_attention_layers: list[int] | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.layers = get_clones(layer, num_layers)
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.pos_enc_at_input = pos_enc_at_input
        self.use_act_checkpoint = use_act_checkpoint

        if frozen:
            for p in self.parameters():
                p.requires_grad_(False)

        self.batch_first = batch_first

        # remove cross attention layers if specified
        self.remove_cross_attention_layers = [False] * self.num_layers
        if remove_cross_attention_layers is not None:
            for i in remove_cross_attention_layers:
                self.remove_cross_attention_layers[i] = True
        assert len(self.remove_cross_attention_layers) == len(self.layers)

        for i, remove_cross_attention in enumerate(self.remove_cross_attention_layers):
            if remove_cross_attention:
                self.layers[i].cross_attn_image = None
                self.layers[i].norm2 = None
                self.layers[i].dropout2 = None

    def forward(
        self,
        src: Float[Tensor, "..."] | list[Float[Tensor, "..."]],  # self-attention inputs
        prompt: Float[Tensor, "..."],  # cross-attention inputs
        src_mask: Float[Tensor, "..."] | Bool[Tensor, "..."] | None = None,
        prompt_mask: Float[Tensor, "..."] | Bool[Tensor, "..."] | None = None,
        src_key_padding_mask: (
            Float[Tensor, "..."]
            | Bool[Tensor, "..."]
            | list[Float[Tensor, "..."] | Bool[Tensor, "..."] | None]
            | None
        ) = None,
        prompt_key_padding_mask: Float[Tensor, "..."]
        | Bool[Tensor, "..."]
        | None = None,
        src_pos: Float[Tensor, "..."] | list[Float[Tensor, "..."]] | None = None,
        prompt_pos: Float[Tensor, "..."] | None = None,
        feat_sizes: Sequence[int]
        | Sequence[tuple[int, int]]
        | list[int]
        | list[tuple[int, int]]
        | None = None,
        num_obj_ptr_tokens: int = 0,
    ) -> dict[str, Any]:
        if isinstance(src, list):
            assert isinstance(src_key_padding_mask, list) and isinstance(src_pos, list)
            assert len(src) == len(src_key_padding_mask) == len(src_pos) == 1
            src, src_key_padding_mask, src_pos = (
                src[0],
                src_key_padding_mask[0],
                src_pos[0],
            )

        assert src.shape[1] == prompt.shape[1], (
            "Batch size must be the same for src and prompt"
        )

        output = src

        if self.pos_enc_at_input and src_pos is not None:
            output = output + 0.1 * src_pos

        if self.batch_first:
            # Convert to batch first
            output = output.transpose(0, 1)
            src_pos = src_pos.transpose(0, 1)
            prompt = prompt.transpose(0, 1)
            prompt_pos = prompt_pos.transpose(0, 1)

        # Activation checkpointing is not wired (eval/inference first).
        # When training with use_act_checkpoint=True, layers still run directly.
        _ = self.use_act_checkpoint  # documented no-op for now

        for layer in self.layers:
            kwds: dict[str, Any] = {}
            if isinstance(layer.cross_attn_image, RoPEAttention):
                kwds = {"num_k_exclude_rope": num_obj_ptr_tokens}

            output = layer(
                tgt=output,
                memory=prompt,
                tgt_mask=src_mask,
                memory_mask=prompt_mask,
                tgt_key_padding_mask=src_key_padding_mask,
                memory_key_padding_mask=prompt_key_padding_mask,
                pos=prompt_pos,
                query_pos=src_pos,
                dac=False,
                attn_bias=None,
                **kwds,
            )
            normed_output = self.norm(output)

        if self.batch_first:
            # Convert back to seq first
            normed_output = normed_output.transpose(0, 1)
            src_pos = src_pos.transpose(0, 1)

        return {
            "memory": normed_output,
            "pos_embed": src_pos,
            "padding_mask": src_key_padding_mask,
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_tracker_transformer(
    d_model: int = 256,
    num_layers: int = 4,
    feat_sizes: Sequence[int] | tuple[int, ...] = (72, 72),
    num_heads: int = 1,
    dim_feedforward: int = 2048,
    dropout: float | int = 0.1,
    kv_in_dim: int = 64,
    rope_theta: float | int = 10000.0,
    use_rope_real: bool = True,
    pos_enc_at_input: bool = True,
    pos_enc_at_attn: bool = False,
    pos_enc_at_cross_attn_keys: bool = True,
    pos_enc_at_cross_attn_queries: bool = False,
    cross_attention_first: bool = False,
    batch_first: bool = True,
    frozen: bool = False,
    use_act_checkpoint: bool = False,
    remove_cross_attention_layers: list[int] | None = None,
) -> TransformerEncoderCrossAttention:
    """Build the production tracker memory-attention transformer.

    Mirrors ``_create_tracker_transformer`` in ``sam3.model_builder`` with
    defaults ``d_model=256``, 4 layers, ``feat_sizes=(72, 72)``, cross
    ``kv_in_dim=64`` and ``rope_k_repeat=True``. Prefers
    ``use_rope_real=True`` for the ATen tracker stack.
    """
    feat_sizes = tuple(feat_sizes)

    self_attention = RoPEAttention(
        embedding_dim=d_model,
        num_heads=num_heads,
        downsample_rate=1,
        dropout=dropout,
        rope_theta=rope_theta,
        feat_sizes=feat_sizes,
        use_rope_real=use_rope_real,
    )

    cross_attention = RoPEAttention(
        embedding_dim=d_model,
        num_heads=num_heads,
        downsample_rate=1,
        dropout=dropout,
        kv_in_dim=kv_in_dim,
        rope_theta=rope_theta,
        feat_sizes=feat_sizes,
        rope_k_repeat=True,
        use_rope_real=use_rope_real,
    )

    encoder_layer = TransformerDecoderLayerv2(
        cross_attention_first=cross_attention_first,
        activation="relu",
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        pos_enc_at_attn=pos_enc_at_attn,
        pre_norm=True,
        self_attention=self_attention,
        d_model=d_model,
        pos_enc_at_cross_attn_keys=pos_enc_at_cross_attn_keys,
        pos_enc_at_cross_attn_queries=pos_enc_at_cross_attn_queries,
        cross_attention=cross_attention,
    )

    encoder = TransformerEncoderCrossAttention(
        remove_cross_attention_layers=remove_cross_attention_layers or [],
        batch_first=batch_first,
        d_model=d_model,
        frozen=frozen,
        pos_enc_at_input=pos_enc_at_input,
        layer=encoder_layer,
        num_layers=num_layers,
        use_act_checkpoint=use_act_checkpoint,
    )
    return encoder
