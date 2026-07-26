"""Transformer decoder for the SAM3 text-grounding detector.

Port of the inference-critical path from ``sam3.model.decoder``:
``TransformerDecoderLayer`` + ``TransformerDecoder`` as constructed by
``_create_transformer_decoder`` in ``sam3.model_builder``.

Optional paths (decoupled / RoI / v1/v2 layers) raise ``NotImplementedError``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import copy
import math
from typing import Any

from jaxtyping import Bool, Float, Integer
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torchvision.ops import box_convert

# ---------------------------------------------------------------------------
# Helpers (from model_misc / box_ops) — kept private to this module
# ---------------------------------------------------------------------------


def inverse_sigmoid(
    x: Float[Tensor, "..."], eps: float | int = 1e-3
) -> Float[Tensor, "..."]:
    """Inverse of sigmoid; matches ``sam3.model.model_misc.inverse_sigmoid``."""
    eps = float(eps)
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


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


def gen_sineembed_for_position(
    pos_tensor: Float[Tensor, "nq bs coords"],
    num_feats: int = 256,
) -> Float[Tensor, "nq bs embed"]:
    """Sine positional embedding for reference points (2- or 4-d last dim)."""
    assert num_feats % 2 == 0
    num_feats = num_feats // 2
    scale = 2 * math.pi
    dim_t = torch.arange(num_feats, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (torch.div(dim_t, 2, rounding_mode="floor")) / num_feats)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack(
        (pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3
    ).flatten(2)
    pos_y = torch.stack(
        (pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3
    ).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack(
            (pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3
        ).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack(
            (pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3
        ).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError(f"Unknown pos_tensor shape(-1):{pos_tensor.size(-1)}")
    return pos


class _MLP(nn.Module):
    """Official ``model_misc.MLP`` (nn.Linear + ReLU) for checkpoint parity."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float | int = 0.0,
        residual: bool = False,
        out_norm: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        dropout = float(dropout)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if residual and input_dim != output_dim:
            raise ValueError("residual is only supported if input_dim == output_dim")
        self.residual = residual
        self.out_norm = out_norm or nn.Identity()

    def forward(
        self, x: Float[Tensor, "*batch features"]
    ) -> Float[Tensor, "*batch out"]:
        orig_x = x
        for i, layer in enumerate(self.layers):
            x = self.drop(F.relu(layer(x))) if i < self.num_layers - 1 else layer(x)
        if self.residual:
            x = x + orig_x
        x = self.out_norm(x)
        return x


def _call_module(module: nn.Module, *args, act_ckpt_enable: bool = False, **kwargs):
    """Drop-in for activation_ckpt_wrapper in eval / non-ckpt paths.

    When ``act_ckpt_enable`` is False (the common case for inference tests),
    calls the module with remaining kwargs. Training-time checkpointing is not
    required for the image-detector inference path ported here.
    """
    del act_ckpt_enable  # unused; kept for API compatibility with official calls
    return module(*args, **kwargs)


# ---------------------------------------------------------------------------
# TransformerDecoderLayer
# ---------------------------------------------------------------------------


class TransformerDecoderLayer(nn.Module):
    """Pre-norm DETR decoder layer: SA → text CA → image CA → FFN."""

    def __init__(
        self,
        activation: str,
        d_model: int,
        dim_feedforward: int,
        dropout: float | int,
        cross_attention: nn.Module,
        n_heads: int,
        use_text_cross_attention: bool = False,
    ) -> None:
        super().__init__()
        dropout = float(dropout)

        # cross attention (image)
        self.cross_attn = cross_attention
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention text
        self.use_text_cross_attention = use_text_cross_attention
        if use_text_cross_attention:
            self.ca_text = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
            self.catext_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            self.catext_norm = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm3 = nn.LayerNorm(d_model)

        self.layer_idx: int | None = None

    @staticmethod
    def with_pos_embed(
        tensor: Float[Tensor, "..."] | None,
        pos: Float[Tensor, "..."] | None,
    ) -> Float[Tensor, "..."] | None:
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt: Float[Tensor, "nq bs d"]) -> Float[Tensor, "nq bs d"]:
        # Keep autocast disabled here for parity with official fp32-master
        # behavior. For permanent-cast weights (fp16/bf16), cast inputs to the
        # linear weight dtype and cast FFN output back to the residual dtype.
        with torch.amp.autocast(device_type="cuda", enabled=False):
            ffn_dtype = self.linear1.weight.dtype
            x = tgt if tgt.dtype == ffn_dtype else tgt.to(dtype=ffn_dtype)
            tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(x))))
            if tgt2.dtype != tgt.dtype:
                tgt2 = tgt2.to(dtype=tgt.dtype)
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(
        self,
        # for tgt
        tgt: Float[Tensor, "nq bs d"] | None,  # nq, bs, d_model
        tgt_query_pos: Float[Tensor, "nq bs d"]
        | None = None,  # pos for query. MLP(Sine(pos))
        tgt_query_sine_embed: Float[Tensor, "..."]
        | None = None,  # pos for query. Sine(pos)
        tgt_key_padding_mask: Bool[Tensor, "..."] | Float[Tensor, "..."] | None = None,
        tgt_reference_points: Float[Tensor, "..."] | None = None,  # nq, bs, 4
        memory_text: Float[Tensor, "ntoken bs d"]
        | None = None,  # num_token, bs, d_model
        text_attention_mask: Bool[Tensor, "bs ntoken"]
        | Float[Tensor, "bs ntoken"]
        | None = None,
        # for memory
        memory: Float[Tensor, "hw bs d"] | None = None,  # hw, bs, d_model
        memory_key_padding_mask: Bool[Tensor, "..."]
        | Float[Tensor, "..."]
        | None = None,
        memory_level_start_index: Integer[Tensor, "..."] | None = None,  # num_levels
        memory_spatial_shapes: Integer[Tensor, "..."]
        | None = None,  # bs, num_levels, 2
        memory_pos: Float[Tensor, "..."] | None = None,  # pos for memory
        # sa
        self_attn_mask: Bool[Tensor, "..."] | Float[Tensor, "..."] | None = None,
        cross_attn_mask: Bool[Tensor, "..."] | Float[Tensor, "..."] | None = None,
        # dac
        dac: bool = False,
        dac_use_selfatt_ln: bool = True,
        presence_token: Float[Tensor, "1 bs d"] | None = None,
        # skip inside deformable attn
        identity: float | int = 0.0,
        **kwargs: Any,
    ) -> tuple[Float[Tensor, "nq bs d"], Float[Tensor, "1 bs d"] | None]:
        del (
            tgt_query_sine_embed,
            tgt_key_padding_mask,
            tgt_reference_points,
            memory_level_start_index,
            memory_spatial_shapes,
            identity,
            kwargs,
        )

        # self attention
        if self.self_attn is not None:
            if dac:
                # only apply self attention to the first half of the queries
                assert tgt is not None and tgt.shape[0] % 2 == 0
                num_o2o_queries = tgt.shape[0] // 2
                tgt_o2o = tgt[:num_o2o_queries]
                tgt_query_pos_o2o = (
                    tgt_query_pos[:num_o2o_queries]
                    if tgt_query_pos is not None
                    else None
                )
                tgt_o2m = tgt[num_o2o_queries:]
            else:
                tgt_o2o = tgt
                tgt_query_pos_o2o = tgt_query_pos

            if presence_token is not None:
                tgt_o2o = torch.cat([presence_token, tgt_o2o], dim=0)
                tgt_query_pos_o2o = torch.cat(
                    [torch.zeros_like(presence_token), tgt_query_pos_o2o], dim=0
                )
                tgt_query_pos = torch.cat(
                    [torch.zeros_like(presence_token), tgt_query_pos], dim=0
                )

            q = k = self.with_pos_embed(tgt_o2o, tgt_query_pos_o2o)
            tgt2 = self.self_attn(
                q, k, tgt_o2o, attn_mask=self_attn_mask, need_weights=False
            )[0]
            tgt_o2o = tgt_o2o + self.dropout2(tgt2)
            if dac:
                if not dac_use_selfatt_ln:
                    tgt_o2o = self.norm2(tgt_o2o)
                tgt = torch.cat((tgt_o2o, tgt_o2m), dim=0)  # Recombine
                if dac_use_selfatt_ln:
                    tgt = self.norm2(tgt)
            else:
                tgt = tgt_o2o
                tgt = self.norm2(tgt)

        if self.use_text_cross_attention:
            tgt2 = self.ca_text(
                self.with_pos_embed(tgt, tgt_query_pos),
                memory_text,
                memory_text,
                key_padding_mask=text_attention_mask,
                need_weights=False,
            )[0]
            tgt = tgt + self.catext_dropout(tgt2)
            tgt = self.catext_norm(tgt)

        if presence_token is not None:
            assert cross_attn_mask is not None
            presence_token_mask = torch.zeros_like(cross_attn_mask[:, :1, :])
            cross_attn_mask = torch.cat(
                [presence_token_mask, cross_attn_mask], dim=1
            )  # (bs*nheads, 1+nq, hw)

        # Cross attention to image
        tgt2 = self.cross_attn(
            query=self.with_pos_embed(tgt, tgt_query_pos),
            key=self.with_pos_embed(memory, memory_pos),
            value=memory,
            attn_mask=cross_attn_mask,
            key_padding_mask=(
                memory_key_padding_mask.transpose(0, 1)
                if memory_key_padding_mask is not None
                else None
            ),
            need_weights=False,
        )[0]

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        presence_token_out = None
        if presence_token is not None:
            presence_token_out = tgt[:1]
            tgt = tgt[1:]

        return tgt, presence_token_out


# ---------------------------------------------------------------------------
# TransformerDecoder
# ---------------------------------------------------------------------------


class TransformerDecoder(nn.Module):
    """SAM3 DETR decoder with iterative box refine, boxRPB, presence token, DAC."""

    def __init__(
        self,
        d_model: int,
        frozen: bool,
        interaction_layer,
        layer: nn.Module,
        num_layers: int,
        num_queries: int,
        return_intermediate: bool,
        box_refine: bool = False,
        num_o2m_queries: int = 0,
        dac: bool = False,
        boxRPB: str = "none",
        instance_query: bool = False,
        num_instances: int = 1,
        dac_use_selfatt_ln: bool = True,
        use_act_checkpoint: bool = False,
        compile_mode=None,
        presence_token: bool = False,
        clamp_presence_logits: bool = True,
        clamp_presence_logit_max_val: float | int = 10.0,
        use_normed_output_consistently: bool = True,
        separate_box_head_instance: bool = False,
        separate_norm_instance: bool = False,
        resolution: int | None = None,
        stride: int | None = None,
    ) -> None:
        super().__init__()
        if interaction_layer is not None:
            raise NotImplementedError(
                "interaction_layer / RoI fine layers are not ported in sam3"
            )
        if instance_query:
            raise NotImplementedError(
                "instance_query path is not ported in sam3 det_decoder"
            )

        self.d_model = d_model
        self.layers = get_clones(layer, num_layers)
        self.fine_layers = [None] * num_layers
        self.num_layers = num_layers
        self.num_queries = num_queries
        self.dac = dac
        if dac:
            self.num_o2m_queries = num_queries
            tot_num_queries = num_queries
        else:
            self.num_o2m_queries = num_o2m_queries
            tot_num_queries = num_queries + num_o2m_queries
        self.norm = nn.LayerNorm(d_model)
        self.return_intermediate = return_intermediate
        self.bbox_embed = _MLP(d_model, d_model, 4, 3)
        self.query_embed = nn.Embedding(tot_num_queries, d_model)
        self.instance_query_embed = None
        self.instance_query_reference_points = None
        self.use_instance_query = instance_query
        self.num_instances = num_instances
        self.use_normed_output_consistently = use_normed_output_consistently

        self.instance_norm = nn.LayerNorm(d_model) if separate_norm_instance else None
        self.instance_bbox_embed = None
        if separate_box_head_instance:
            self.instance_bbox_embed = _MLP(d_model, d_model, 4, 3)
        self.box_refine = box_refine
        if box_refine:
            nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

            self.reference_points = nn.Embedding(num_queries, 4)

        assert boxRPB in ["none", "log", "linear", "both"]
        self.boxRPB = boxRPB
        if boxRPB != "none":
            try:
                nheads = self.layers[0].cross_attn_image.num_heads  # type: ignore[attr-defined]
            except AttributeError:
                nheads = self.layers[0].cross_attn.num_heads  # type: ignore[attr-defined]

            n_input = 4 if boxRPB == "both" else 2
            self.boxRPB_embed_x = _MLP(n_input, d_model, nheads, 2)
            self.boxRPB_embed_y = _MLP(n_input, d_model, nheads, 2)

        self.roi_pooler = None
        if frozen:
            for p in self.parameters():
                p.requires_grad_(False)

        self.presence_token = None
        self.clamp_presence_logits = clamp_presence_logits
        self.clamp_presence_logit_max_val = float(clamp_presence_logit_max_val)
        if presence_token:
            self.presence_token = nn.Embedding(1, d_model)
            self.presence_token_head = _MLP(d_model, d_model, 1, 3)
            self.presence_token_out_norm = nn.LayerNorm(d_model)

        self.ref_point_head = _MLP(2 * self.d_model, self.d_model, self.d_model, 2)
        self.dac_use_selfatt_ln = dac_use_selfatt_ln
        self.use_act_checkpoint = use_act_checkpoint

        nn.init.normal_(self.query_embed.weight.data)

        assert self.roi_pooler is None
        assert self.return_intermediate, "support return_intermediate only"
        assert self.box_refine, "support box refine only"

        self.compile_mode = compile_mode
        self.compiled = False

        for layer_idx, lyr in enumerate(self.layers):
            lyr.layer_idx = layer_idx  # type: ignore[attr-defined]

    @staticmethod
    def _get_coords(H: int, W: int, device) -> tuple[Tensor, Tensor]:
        coords_h = torch.arange(0, H, device=device, dtype=torch.float32) / H
        coords_w = torch.arange(0, W, device=device, dtype=torch.float32) / W
        return coords_h, coords_w

    def _get_rpb_matrix(
        self,
        reference_boxes: Tensor,
        coords_h: Tensor,
        coords_w: Tensor,
    ) -> Tensor:
        boxes_xyxy = (
            box_convert(reference_boxes.reshape(-1, 4), in_fmt="cxcywh", out_fmt="xyxy")
            .reshape_as(reference_boxes)
            .transpose(0, 1)
        )
        bs, num_queries, _ = boxes_xyxy.shape

        deltas_y = coords_h.view(1, -1, 1) - boxes_xyxy.reshape(-1, 1, 4)[:, :, 1:4:2]
        deltas_y = deltas_y.view(bs, num_queries, -1, 2)
        deltas_x = coords_w.view(1, -1, 1) - boxes_xyxy.reshape(-1, 1, 4)[:, :, 0:3:2]
        deltas_x = deltas_x.view(bs, num_queries, -1, 2)

        if self.boxRPB in ["log", "both"]:
            deltas_x_log = deltas_x * 8  # normalize to -8, 8
            deltas_x_log = (
                torch.sign(deltas_x_log)
                * torch.log2(torch.abs(deltas_x_log) + 1.0)
                / np.log2(8)
            )

            deltas_y_log = deltas_y * 8  # normalize to -8, 8
            deltas_y_log = (
                torch.sign(deltas_y_log)
                * torch.log2(torch.abs(deltas_y_log) + 1.0)
                / np.log2(8)
            )
            if self.boxRPB == "log":
                deltas_x = deltas_x_log
                deltas_y = deltas_y_log
            else:
                deltas_x = torch.cat([deltas_x, deltas_x_log], dim=-1)
                deltas_y = torch.cat([deltas_y, deltas_y_log], dim=-1)

        # Skip act-ckpt wrapper; call MLPs directly (inference / unit tests).
        rpb_dtype = next(self.boxRPB_embed_x.parameters()).dtype
        deltas_x = self.boxRPB_embed_x(deltas_x.to(dtype=rpb_dtype))
        deltas_y = self.boxRPB_embed_y(deltas_y.to(dtype=rpb_dtype))

        B = deltas_y.unsqueeze(3) + deltas_x.unsqueeze(
            2
        )  # bs, num_queries, H, W, n_heads
        B = B.flatten(2, 3)  # bs, num_queries, H*W, n_heads
        B = B.permute(0, 3, 1, 2)  # bs, n_heads, num_queries, H*W
        return B.contiguous()

    def forward(
        self,
        tgt: Float[Tensor, "nq bs d"],
        memory: Float[Tensor, "hw bs d"],
        tgt_mask: Bool[Tensor, "..."] | Float[Tensor, "..."] | None = None,
        memory_mask: Bool[Tensor, "..."] | Float[Tensor, "..."] | None = None,
        tgt_key_padding_mask: Bool[Tensor, "..."] | Float[Tensor, "..."] | None = None,
        memory_key_padding_mask: Bool[Tensor, "..."]
        | Float[Tensor, "..."]
        | None = None,
        pos: Float[Tensor, "..."] | None = None,
        reference_boxes: Float[Tensor, "nq bs 4"] | None = None,  # num_queries, bs, 4
        # for memory
        level_start_index: Integer[Tensor, "..."] | None = None,  # num_levels
        spatial_shapes: Integer[Tensor, "..."]
        | None = None,  # num_levels, 2  (or bs, nlevel, 2)
        valid_ratios: Float[Tensor, "..."] | None = None,
        feature_size: tuple[int, int] | None = None,
        # for text
        memory_text: Float[Tensor, "ntoken bs d"] | None = None,
        text_attention_mask: Bool[Tensor, "bs ntoken"]
        | Float[Tensor, "bs ntoken"]
        | None = None,
        # if `apply_dac` is None, it will default to `self.dac`
        apply_dac: bool | None = None,
        is_instance_prompt: bool = False,
        decoder_extra_kwargs: Mapping[str, object] | None = None,
        # ROI memory bank — not supported
        obj_roi_memory_feat: object | None = None,
        obj_roi_memory_mask: object | None = None,
        box_head_trk: object | None = None,
    ) -> tuple[
        Float[Tensor, "layers nq bs d"],
        Float[Tensor, "layers nq bs 4"],
        Float[Tensor, "layers 1 bs"] | None,
        Float[Tensor, "1 bs d"] | None,
    ]:
        if obj_roi_memory_feat is not None or obj_roi_memory_mask is not None:
            raise NotImplementedError("ROI memory bank path is not ported")
        if box_head_trk is not None:
            raise NotImplementedError("tracking box_head_trk path is not ported")
        if is_instance_prompt:
            raise NotImplementedError("instance_prompt path is not ported")
        if decoder_extra_kwargs:
            # Official may pass empty/None; reject unknown non-empty extras that
            # would alter tracking / ROI behavior.
            unknown = set(decoder_extra_kwargs.keys()) - set()
            if unknown and any(
                k in decoder_extra_kwargs
                for k in ("Q_det", "obj_roi_memory_feat", "obj_roi_memory_mask")
            ):
                raise NotImplementedError(
                    f"decoder_extra_kwargs keys not supported: {decoder_extra_kwargs.keys()}"
                )

        if memory_mask is not None:
            assert self.boxRPB == "none", (
                "inputting a memory_mask in the presence of boxRPB is unexpected/not implemented"
            )

        apply_dac = apply_dac if apply_dac is not None else self.dac
        if apply_dac:
            assert tgt.shape[0] == self.num_queries

            tgt = tgt.repeat(2, 1, 1)
            if reference_boxes is not None:
                assert reference_boxes.shape[0] == self.num_queries
                reference_boxes = reference_boxes.repeat(2, 1, 1)

        bs = tgt.shape[1]
        intermediate = []
        intermediate_presence_logits = []
        presence_feats = None

        if self.box_refine:
            if reference_boxes is None:
                # one-stage model: learnable reference points
                reference_boxes = self.reference_points.weight.unsqueeze(1)
                reference_boxes = (
                    reference_boxes.repeat(2, bs, 1)
                    if apply_dac
                    else reference_boxes.repeat(1, bs, 1)
                )
                reference_boxes = reference_boxes.sigmoid()
            intermediate_ref_boxes = [reference_boxes]
        else:
            reference_boxes = None
            intermediate_ref_boxes = None

        output = tgt
        presence_out = None
        if self.presence_token is not None and not is_instance_prompt:
            presence_out = self.presence_token.weight[None].expand(1, bs, -1)

        box_head = self.bbox_embed
        out_norm = self.norm
        if self.boxRPB != "none":
            if feature_size is None:
                assert spatial_shapes is not None
                feature_size = (
                    int(spatial_shapes[0, 0]),
                    int(spatial_shapes[0, 1]),
                )
            coords_h, coords_w = self._get_coords(
                feature_size[0], feature_size[1], memory.device
            )

        for layer_idx, layer in enumerate(self.layers):
            reference_points_input = (
                reference_boxes[:, :, None]
                * torch.cat([valid_ratios, valid_ratios], -1)[None, :]
            )  # nq, bs, nlevel, 4

            query_sine_embed = gen_sineembed_for_position(
                reference_points_input[:, :, 0, :], self.d_model
            )  # nq, bs, d_model*2
            # gen_sineembed is fp32; match ref_point_head weights (bf16/fp16)
            query_sine_embed = query_sine_embed.to(
                dtype=next(self.ref_point_head.parameters()).dtype
            )

            query_pos = self.ref_point_head(query_sine_embed)  # nq, bs, d_model

            if self.boxRPB != "none" and reference_boxes is not None:
                assert spatial_shapes is not None
                assert spatial_shapes.shape[0] == 1, (
                    "only single scale support implemented"
                )
                memory_mask = self._get_rpb_matrix(
                    reference_boxes,
                    coords_h,
                    coords_w,
                )
                memory_mask = memory_mask.flatten(0, 1)  # (bs*n_heads, nq, H*W)

            output, presence_out = _call_module(
                layer,
                tgt=output,
                tgt_query_pos=query_pos,
                tgt_query_sine_embed=query_sine_embed,
                tgt_key_padding_mask=tgt_key_padding_mask,
                tgt_reference_points=reference_points_input,
                memory_text=memory_text,
                text_attention_mask=text_attention_mask,
                memory=memory,
                memory_key_padding_mask=memory_key_padding_mask,
                memory_level_start_index=level_start_index,
                memory_spatial_shapes=spatial_shapes,
                memory_pos=pos,
                self_attn_mask=tgt_mask,
                cross_attn_mask=memory_mask,
                dac=apply_dac,
                dac_use_selfatt_ln=self.dac_use_selfatt_ln,
                presence_token=presence_out,
                act_ckpt_enable=self.training and self.use_act_checkpoint,
            )

            # iterative box refine
            if self.box_refine:
                reference_before_sigmoid = inverse_sigmoid(reference_boxes)
                if not self.use_normed_output_consistently:
                    delta_unsig = box_head(output)
                else:
                    delta_unsig = box_head(out_norm(output))
                outputs_unsig = delta_unsig + reference_before_sigmoid
                new_reference_points = outputs_unsig.sigmoid()

                reference_boxes = new_reference_points.detach()
                if layer_idx != self.num_layers - 1:
                    intermediate_ref_boxes.append(new_reference_points)
            else:
                raise NotImplementedError("box_refine=False is not supported")

            intermediate.append(out_norm(output))
            if self.presence_token is not None and not is_instance_prompt:
                intermediate_layer_presence_logits = self.presence_token_head(
                    self.presence_token_out_norm(presence_out)
                ).squeeze(-1)

                # Match official: non-in-place clamp (result unused unless reassigned).
                # Keep identical call for behavioral parity with sam3.model.decoder.
                if self.clamp_presence_logits:
                    intermediate_layer_presence_logits.clamp(
                        min=-self.clamp_presence_logit_max_val,
                        max=self.clamp_presence_logit_max_val,
                    )

                intermediate_presence_logits.append(intermediate_layer_presence_logits)
                presence_feats = presence_out.clone()

        if not self.compiled and self.compile_mode is not None:
            self.forward = torch.compile(  # type: ignore[method-assign]
                self.forward, mode=self.compile_mode, fullgraph=True
            )
            self.compiled = True

        return (
            torch.stack(intermediate),
            torch.stack(intermediate_ref_boxes),
            (
                torch.stack(intermediate_presence_logits)
                if self.presence_token is not None and not is_instance_prompt
                else None
            ),
            presence_feats,
        )


def create_sam3_image_decoder(
    d_model: int = 256,
    n_heads: int = 8,
    dim_feedforward: int = 2048,
    dropout: float | int = 0.1,
    num_layers: int = 6,
    num_queries: int = 200,
    resolution: int = 1008,
    stride: int = 14,
    use_act_checkpoint: bool = False,
) -> TransformerDecoder:
    """Factory matching ``_create_transformer_decoder`` (SAM3 image detector)."""
    dropout = float(dropout)
    cross_attention = nn.MultiheadAttention(
        embed_dim=d_model, num_heads=n_heads, dropout=dropout
    )
    decoder_layer = TransformerDecoderLayer(
        activation="relu",
        d_model=d_model,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        cross_attention=cross_attention,
        n_heads=n_heads,
        use_text_cross_attention=True,
    )
    return TransformerDecoder(
        layer=decoder_layer,
        num_layers=num_layers,
        num_queries=num_queries,
        return_intermediate=True,
        box_refine=True,
        num_o2m_queries=0,
        dac=True,
        boxRPB="log",
        d_model=d_model,
        frozen=False,
        interaction_layer=None,
        dac_use_selfatt_ln=True,
        resolution=resolution,
        stride=stride,
        use_act_checkpoint=use_act_checkpoint,
        presence_token=True,
    )
