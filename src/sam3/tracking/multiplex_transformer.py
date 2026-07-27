"""Checkpoint-exact SAM3.1 decoupled Multiplex memory transformer."""

from __future__ import annotations

import copy
from functools import partial
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from sam3.grounding.transformer_wrapper import TransformerWrapper
from sam3.primitives.rope import apply_rotary_enc_real, compute_axial_cis


def _multiplex_chunked_attention_impl(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    key_validity: Tensor,
    num_heads: int,
) -> Tensor:
    """Exact online softmax without materializing the 52k-token score matrix."""

    batch, query_length, channels = query.shape
    key_length = key.shape[1]
    head_dimension = channels // num_heads
    q = query.reshape(
        batch, query_length, num_heads, head_dimension
    ).transpose(1, 2)
    k = key.reshape(
        batch, key_length, num_heads, head_dimension
    ).transpose(1, 2)
    v = value.reshape(
        batch, key_length, num_heads, head_dimension
    ).transpose(1, 2)
    scale = head_dimension**-0.5
    running_max = torch.full(
        (batch, num_heads, query_length, 1),
        -torch.inf,
        dtype=query.dtype,
        device=query.device,
    )
    running_sum = torch.zeros_like(running_max)
    running_value = torch.zeros(
        (batch, num_heads, query_length, head_dimension),
        dtype=query.dtype,
        device=query.device,
    )
    chunk_size = 512
    for start in range(0, key_length, chunk_size):
        end = min(start + chunk_size, key_length)
        scores = torch.matmul(q, k[:, :, start:end].transpose(-1, -2))
        scores = scores * scale
        valid = key_validity[:, None, None, start:end].to(torch.bool)
        scores = torch.where(valid, scores, torch.full_like(scores, -torch.inf))
        chunk_max = torch.amax(scores, dim=-1, keepdim=True)
        new_max = torch.maximum(running_max, chunk_max)
        old_scale = torch.exp(running_max - new_max)
        weights = torch.exp(scores - new_max)
        weights = torch.where(valid, weights, torch.zeros_like(weights))
        running_value = running_value * old_scale + torch.matmul(
            weights, v[:, :, start:end]
        )
        running_sum = running_sum * old_scale + torch.sum(
            weights, dim=-1, keepdim=True
        )
        running_max = new_max
    attended = running_value / running_sum.clamp_min(1e-12)
    return (
        attended.transpose(1, 2)
        .reshape(batch, query_length, channels)
    )


def multiplex_chunked_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    key_validity: Tensor,
    num_heads: int,
) -> Tensor:
    with torch.autocast(device_type=query.device.type, enabled=False):
        return _multiplex_chunked_attention_impl(
            query, key, value, key_validity, num_heads
        )


class SimpleRoPEAttention(nn.Module):
    """Projection-free RoPE attention used by the SAM3.1 tracker."""

    def __init__(
        self,
        *,
        d_model: int,
        num_heads: int,
        dropout_p: float,
        rope_theta: float = 10000.0,
        rope_k_repeat: bool = False,
        feat_sizes: tuple[int, int] = (72, 72),
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.rope_k_repeat = rope_k_repeat
        self.compute_cis = partial(
            compute_axial_cis, dim=d_model // num_heads, theta=rope_theta
        )
        frequencies = self.compute_cis(
            end_x=feat_sizes[0], end_y=feat_sizes[1], device=None
        )
        # The official checkpoint owns no transformer RoPE buffers.
        self.freqs_cis_real = frequencies.real
        self.freqs_cis_imag = frequencies.imag

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        num_k_exclude_rope: int = 0,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        batch, query_length, q_channels = q.shape
        _, key_length, k_channels = k.shape
        value_batch, value_length, v_channels = v.shape
        if value_length != key_length:
            raise ValueError("key and value sequence lengths differ")
        q = q.reshape(
            batch, query_length, self.num_heads, q_channels // self.num_heads
        ).transpose(1, 2)
        k = k.reshape(
            k.shape[0], key_length, self.num_heads, k_channels // self.num_heads
        ).transpose(1, 2)
        v = v.reshape(
            value_batch, value_length, self.num_heads, v_channels // self.num_heads
        ).transpose(1, 2)
        if q.shape[-2] != k.shape[-2] and not self.rope_k_repeat:
            raise ValueError("cross-attention requires repeated key RoPE")
        frequencies_real = self.freqs_cis_real
        frequencies_imag = self.freqs_cis_imag
        if frequencies_real.shape[0] != q.shape[-2]:
            side = int(math.sqrt(q.shape[-2]))
            if side * side != q.shape[-2]:
                raise ValueError("query RoPE sequence must have a square spatial shape")
            frequencies = self.compute_cis(end_x=side, end_y=side, device=q.device)
            frequencies_real = frequencies.real
            frequencies_imag = frequencies.imag
        num_k_rope = k.shape[-2] - num_k_exclude_rope
        q, rotated = apply_rotary_enc_real(
            q,
            k[:, :, :num_k_rope],
            freqs_cis_real=frequencies_real.to(q.device),
            freqs_cis_imag=frequencies_imag.to(q.device),
            repeat_freqs_k=self.rope_k_repeat,
        )
        k = torch.cat((rotated, k[:, :, num_k_rope:]), dim=2)
        if self.training and self.dropout_p:
            raise RuntimeError("Multiplex export attention is inference-only")
        validity = (
            torch.ones(
                (batch, key_length),
                dtype=torch.int32,
                device=q.device,
            )
            if key_padding_mask is None
            else (~key_padding_mask.to(torch.bool)).to(torch.int32)
        )
        q_flat = q.transpose(1, 2).reshape(batch, query_length, q_channels)
        k_flat = k.transpose(1, 2).reshape(batch, key_length, k_channels)
        v_flat = v.transpose(1, 2).reshape(batch, value_length, v_channels)
        return multiplex_chunked_attention(
            q_flat,
            k_flat,
            v_flat,
            validity,
            self.num_heads,
        )


class DecoupledTransformerDecoderLayerv2(nn.Module):
    """SAM3.1 layer with distinct shared-image and bucket-object projections."""

    def __init__(
        self,
        *,
        d_model: int,
        dim_feedforward: int,
        dropout: float,
        self_attention_rope: SimpleRoPEAttention,
        cross_attention_rope: SimpleRoPEAttention,
    ) -> None:
        super().__init__()
        self.self_attn_q_proj = nn.Linear(d_model, d_model)
        self.self_attn_k_proj = nn.Linear(d_model, d_model)
        self.self_attn_v_proj = nn.Linear(d_model, d_model)
        self.self_attn_out_proj = nn.Linear(d_model, d_model)
        self.cross_attn_q_proj = nn.Linear(d_model, d_model)
        self.cross_attn_k_proj = nn.Linear(d_model, d_model)
        self.cross_attn_v_proj = nn.Linear(d_model, d_model)
        self.cross_attn_out_proj = nn.Linear(d_model, d_model)
        self.image_cross_attn_q_proj = nn.Linear(d_model, d_model)
        self.image_cross_attn_k_proj = nn.Linear(d_model, d_model)
        self.self_attention_rope = self_attention_rope
        self.cross_attention_rope = cross_attention_rope
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        *,
        image: Tensor,
        tgt: Tensor,
        memory_image: Tensor,
        memory: Tensor,
        query_pos: Tensor,
        memory_image_pos: Tensor,
        num_k_exclude_rope: int = 0,
        memory_key_padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        normalized = self.norm1(tgt)
        attended = self.self_attention_rope(
            self.self_attn_q_proj(normalized),
            self.self_attn_k_proj(normalized),
            self.self_attn_v_proj(normalized),
        )
        tgt = tgt + self.dropout1(self.self_attn_out_proj(attended))

        normalized = self.norm2(tgt)
        q = self.image_cross_attn_q_proj(image) + self.cross_attn_q_proj(normalized)
        k = (
            self.image_cross_attn_k_proj(memory_image)
            + self.cross_attn_k_proj(memory)
            + memory_image_pos
        )
        attended = self.cross_attention_rope(
            q,
            k,
            self.cross_attn_v_proj(memory),
            num_k_exclude_rope=num_k_exclude_rope,
            key_padding_mask=memory_key_padding_mask,
        )
        tgt = tgt + self.dropout2(self.cross_attn_out_proj(attended))
        normalized = self.norm3(tgt)
        tgt = tgt + self.dropout3(
            self.linear2(self.dropout(F.gelu(self.linear1(normalized))))
        )
        return image, tgt


class TransformerEncoderDecoupledCrossAttention(nn.Module):
    """Four-layer bucket-space memory attention with shared frame features."""

    def __init__(
        self, *, layer: nn.Module, num_layers: int = 4, d_model: int = 256
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        *,
        image: Tensor,
        src: Tensor,
        memory_image: Tensor,
        memory: Tensor,
        image_pos: Tensor,
        src_pos: Tensor,
        memory_image_pos: Tensor,
        memory_pos: Tensor,
        num_obj_ptr_tokens: int = 0,
        memory_key_padding_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        del image_pos
        output = src + 0.1 * src_pos
        image = image.transpose(0, 1)
        output = output.transpose(0, 1)
        memory = memory.transpose(0, 1)
        memory_image = memory_image.transpose(0, 1)
        memory_image_pos = memory_image_pos.transpose(0, 1)
        if memory_image.shape[1] != memory.shape[1]:
            pad = memory.shape[1] - memory_image.shape[1]
            if pad != num_obj_ptr_tokens:
                raise ValueError("pointer token count disagrees with memory shape")
            memory_image = torch.cat(
                (
                    memory_image,
                    memory_image.new_zeros(
                        memory_image.shape[0], pad, memory_image.shape[2]
                    ),
                ),
                dim=1,
            )
            memory_image_pos = torch.cat(
                (
                    memory_image_pos,
                    memory_pos[-num_obj_ptr_tokens:, :1].transpose(0, 1),
                ),
                dim=1,
            )
        query_pos = src_pos.transpose(0, 1)
        for layer in self.layers:
            image, output = layer(
                image=image,
                tgt=output,
                memory_image=memory_image,
                memory=memory,
                query_pos=query_pos,
                memory_image_pos=memory_image_pos,
                num_k_exclude_rope=num_obj_ptr_tokens,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        return {
            "memory": self.norm(output).transpose(0, 1),
            "pos_embed": src_pos,
        }


def create_multiplex_transformer() -> TransformerWrapper:
    self_attention = SimpleRoPEAttention(
        d_model=256, num_heads=8, dropout_p=0.1, feat_sizes=(72, 72)
    )
    cross_attention = SimpleRoPEAttention(
        d_model=256,
        num_heads=8,
        dropout_p=0.1,
        feat_sizes=(72, 72),
        rope_k_repeat=True,
    )
    layer = DecoupledTransformerDecoderLayerv2(
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        self_attention_rope=self_attention,
        cross_attention_rope=cross_attention,
    )
    encoder = TransformerEncoderDecoupledCrossAttention(
        layer=layer, num_layers=4, d_model=256
    )
    return TransformerWrapper(encoder=encoder, decoder=None, d_model=256)


__all__ = [
    "DecoupledTransformerDecoderLayerv2",
    "SimpleRoPEAttention",
    "TransformerEncoderDecoupledCrossAttention",
    "create_multiplex_transformer",
    "multiplex_chunked_attention",
]
