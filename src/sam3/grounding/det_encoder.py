"""Grounding transformer encoder for fused image and text features.

Source of truth: ``sam3.model.encoder`` + helpers from ``sam3.model.model_misc``.

Ports:
* ``TransformerEncoderLayer`` (pre-norm path used by SAM3)
* ``TransformerEncoder`` multilevel flatten helpers
* ``TransformerEncoderFusion`` + ``pool_text_feat``

Helpers (``get_activation_fn``, ``get_clones``, ``get_valid_ratio``, minimal
activation-ckpt wrapper) live here to avoid ownership conflicts with other workers.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import copy
from functools import wraps
from typing import TypeVar

from jaxtyping import Bool, Float, Integer
import torch
from torch import Tensor, nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Local helpers (from model_misc / act_ckpt_utils — keep self-contained)
# ---------------------------------------------------------------------------

T = TypeVar("T")
Module = TypeVar("Module", bound=nn.Module)


def get_clones(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def get_activation_fn(activation: str) -> Callable[..., Tensor]:
    """Return an activation function given a string."""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def get_valid_ratio(
    mask: Bool[Tensor, "b h w"] | Float[Tensor, "b h w"],
) -> Float[Tensor, "b 2"]:
    """Valid width/height ratios from a spatial padding mask ``(B, H, W)``."""
    _, h, w = mask.shape
    valid_h = torch.sum(~mask[:, :, 0], 1)
    valid_w = torch.sum(~mask[:, 0, :], 1)
    valid_ratio_h = valid_h.float() / h
    valid_ratio_w = valid_w.float() / w
    return torch.stack([valid_ratio_w, valid_ratio_h], -1)


def activation_ckpt_wrapper(module: nn.Module | Callable) -> Callable:
    """Minimal activation-checkpoint wrapper (eval path is a plain call)."""

    @wraps(module)
    def act_ckpt_wrapper(
        *args, act_ckpt_enable: bool = True, use_reentrant: bool = False, **kwargs
    ):
        if act_ckpt_enable:
            if len(args) > 0:
                raise ValueError(
                    "This wrapper expects keyword arguments only when `act_ckpt_enable=True`"
                )
            callable_fn = module.forward if isinstance(module, nn.Module) else module
            import inspect

            sig = inspect.signature(callable_fn)
            param_defaults = {
                name: param.default for name, param in sig.parameters.items()
            }
            pos_args = []
            for p_name in param_defaults.keys():
                if p_name in kwargs:
                    pos_args.append(kwargs.pop(p_name))
                elif param_defaults[p_name] is not inspect.Parameter.empty:
                    pos_args.append(param_defaults[p_name])
                elif sig.parameters[p_name].kind is not inspect.Parameter.VAR_KEYWORD:
                    raise ValueError(f"Missing positional argument: {p_name}")
            return torch.utils.checkpoint.checkpoint(
                module if not isinstance(module, nn.Module) else module.__call__,
                *pos_args,
                use_reentrant=use_reentrant,
                **kwargs,
            )
        return module(*args, **kwargs)

    return act_ckpt_wrapper


# ---------------------------------------------------------------------------
# Encoder layer / stack / fusion
# ---------------------------------------------------------------------------


class TransformerEncoderLayer(nn.Module):
    """Self-attention + cross-attention (to text/prompt) + FFN encoder layer."""

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
        self.dropout_value = float(dropout)
        self.self_attn = self_attention
        self.cross_attn_image = cross_attention

        # FFN — nn.Linear keeps official state_dict keys (linear1 / linear2)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(float(dropout))
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(float(dropout))
        self.dropout2 = nn.Dropout(float(dropout))
        self.dropout3 = nn.Dropout(float(dropout))

        self.activation_str = activation
        self.activation = get_activation_fn(activation)
        self.pre_norm = pre_norm

        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys

        self.layer_idx = None

    def forward_post(
        self,
        tgt: Float[Tensor, "b hw d"],
        memory: Float[Tensor, "b txt d"],
        tgt_mask: Float[Tensor, "..."] | Bool[Tensor, "..."] | None = None,
        memory_mask: Float[Tensor, "..."] | Bool[Tensor, "..."] | None = None,
        tgt_key_padding_mask: Bool[Tensor, "..."] | Float[Tensor, "..."] | None = None,
        memory_key_padding_mask: Bool[Tensor, "..."]
        | Float[Tensor, "..."]
        | None = None,
        pos: Float[Tensor, "..."] | None = None,
        query_pos: Float[Tensor, "..."] | None = None,
        **kwargs,
    ) -> Float[Tensor, "b hw d"]:
        q = k = tgt + query_pos if self.pos_enc_at_attn else tgt

        tgt2 = self.self_attn(
            q,
            k,
            value=tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False,
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.cross_attn_image(
            query=tgt + query_pos if self.pos_enc_at_cross_attn_queries else tgt,
            key=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(
        self,
        tgt: Float[Tensor, "b hw d"],
        memory: Float[Tensor, "b txt d"],
        dac: bool = False,
        tgt_mask: Float[Tensor, "..."] | Bool[Tensor, "..."] | None = None,
        memory_mask: Float[Tensor, "..."] | Bool[Tensor, "..."] | None = None,
        tgt_key_padding_mask: Bool[Tensor, "..."] | Float[Tensor, "..."] | None = None,
        memory_key_padding_mask: Bool[Tensor, "..."]
        | Float[Tensor, "..."]
        | None = None,
        pos: Float[Tensor, "..."] | None = None,
        query_pos: Float[Tensor, "..."] | None = None,
    ) -> Float[Tensor, "b hw d"]:
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
            need_weights=False,
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
            need_weights=False,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(
        self,
        tgt: Float[Tensor, "b hw d"],
        memory: Float[Tensor, "b txt d"],
        dac: bool = False,
        tgt_mask: Float[Tensor, "..."] | Bool[Tensor, "..."] | None = None,
        memory_mask: Float[Tensor, "..."] | Bool[Tensor, "..."] | None = None,
        tgt_key_padding_mask: Bool[Tensor, "..."] | Float[Tensor, "..."] | None = None,
        memory_key_padding_mask: Bool[Tensor, "..."]
        | Float[Tensor, "..."]
        | None = None,
        pos: Float[Tensor, "..."] | None = None,
        query_pos: Float[Tensor, "..."] | None = None,
    ) -> Float[Tensor, "b hw d"]:
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
        )


class TransformerEncoder(nn.Module):
    """Stack of encoder layers over multi-level image features."""

    def __init__(
        self,
        layer: nn.Module,
        num_layers: int,
        d_model: int,
        num_feature_levels: int,
        frozen: bool = False,
        use_act_checkpoint: bool = False,
    ):
        super().__init__()
        self.layers = get_clones(layer, num_layers)
        self.num_layers = num_layers

        self.num_feature_levels = num_feature_levels
        self.level_embed = None
        if num_feature_levels > 1:
            self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        if frozen:
            for p in self.parameters():
                p.requires_grad_(False)

        self.use_act_checkpoint = use_act_checkpoint

        for layer_idx, layer_mod in enumerate(self.layers):
            layer_mod.layer_idx = layer_idx

    @staticmethod
    def get_reference_points(
        spatial_shapes,
        valid_ratios,
        device,
    ):
        with torch.no_grad():
            reference_points_list = []
            for lvl, (h_, w_) in enumerate(spatial_shapes):
                ref_y, ref_x = torch.meshgrid(
                    torch.linspace(
                        0.5, h_ - 0.5, h_, dtype=torch.float32, device=device
                    ),
                    torch.linspace(
                        0.5, w_ - 0.5, w_, dtype=torch.float32, device=device
                    ),
                )
                ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * h_)
                ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * w_)
                ref = torch.stack((ref_x, ref_y), -1)
                reference_points_list.append(ref)
            reference_points = torch.cat(reference_points_list, 1)
            reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def _prepare_multilevel_features(self, srcs, masks, pos_embeds):
        assert len(srcs) == self.num_feature_levels, (
            "mismatch between expected and received # of feature levels"
        )
        # Normalize None masks / pos to per-level lists (production path may pass None).
        if masks is None:
            masks = [None] * len(srcs)
        if pos_embeds is None:
            raise ValueError("pos embeddings are required for multilevel features")

        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        has_mask = masks is not None and masks[0] is not None
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            src = src.flatten(2).transpose(1, 2)  # bs, hw, c
            if has_mask:
                mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # bs, hw, c
            if self.level_embed is not None:
                lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            else:
                lvl_pos_embed = pos_embed
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            if has_mask:
                mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)  # bs, sum{hxw}, c
        mask_flatten = torch.cat(mask_flatten, 1) if has_mask else None
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.tensor(
            spatial_shapes, dtype=torch.long, device=src_flatten.device
        )
        level_start_index = torch.cat(
            (
                spatial_shapes.new_zeros((1,)),
                spatial_shapes.prod(1).cumsum(0)[:-1],
            )
        )
        if has_mask:
            valid_ratios = torch.stack([get_valid_ratio(m) for m in masks], 1)
        else:
            valid_ratios = torch.ones(
                (src_flatten.shape[0], self.num_feature_levels, 2),
                device=src_flatten.device,
            )

        return (
            src_flatten,
            mask_flatten,
            lvl_pos_embed_flatten,
            level_start_index,
            valid_ratios,
            spatial_shapes,
        )

    def forward(
        self,
        src: list[Float[Tensor, "..."]],
        src_key_padding_masks: (
            list[Bool[Tensor, "..."] | Float[Tensor, "..."] | None] | None
        ) = None,
        pos: list[Float[Tensor, "..."]] | None = None,
        prompt: Float[Tensor, "..."] | None = None,
        prompt_key_padding_mask: Bool[Tensor, "..."]
        | Float[Tensor, "..."]
        | None = None,
        encoder_extra_kwargs: Mapping[str, object] | None = None,
    ) -> tuple[
        Float[Tensor, "..."],
        Bool[Tensor, "..."] | Float[Tensor, "..."] | None,
        Float[Tensor, "..."],
        Integer[Tensor, "..."],
        Integer[Tensor, "..."],
        Float[Tensor, "..."],
    ]:
        assert len(src) == self.num_feature_levels, (
            "must be equal to num_feature_levels"
        )
        if src_key_padding_masks is not None:
            assert len(src_key_padding_masks) == self.num_feature_levels
        if pos is not None:
            assert len(pos) == self.num_feature_levels

        (
            src_flatten,
            key_padding_masks_flatten,
            lvl_pos_embed_flatten,
            level_start_index,
            valid_ratios,
            spatial_shapes,
        ) = self._prepare_multilevel_features(src, src_key_padding_masks, pos)

        self.get_reference_points(
            spatial_shapes, valid_ratios, device=src_flatten.device
        )

        output = src_flatten
        for layer in self.layers:
            layer_kwargs = {}

            assert isinstance(layer, TransformerEncoderLayer)
            layer_kwargs["memory"] = prompt
            layer_kwargs["memory_key_padding_mask"] = prompt_key_padding_mask
            layer_kwargs["query_pos"] = lvl_pos_embed_flatten
            layer_kwargs["tgt"] = output
            layer_kwargs["tgt_key_padding_mask"] = key_padding_masks_flatten

            if self.training:
                assert self.use_act_checkpoint, "activation ckpt not enabled in encoder"
            if encoder_extra_kwargs is not None:
                layer_kwargs.update(encoder_extra_kwargs)
            output = activation_ckpt_wrapper(layer)(
                **layer_kwargs,
                act_ckpt_enable=self.training and self.use_act_checkpoint,
            )
        # return as seq first
        return (
            output.transpose(0, 1),
            (
                key_padding_masks_flatten.transpose(0, 1)
                if key_padding_masks_flatten is not None
                else None
            ),
            lvl_pos_embed_flatten.transpose(0, 1),
            level_start_index,
            spatial_shapes,
            valid_ratios,
        )


class TransformerEncoderFusion(TransformerEncoder):
    """Encoder that optionally fuses pooled text features into image features."""

    def __init__(
        self,
        layer: nn.Module,
        num_layers: int,
        d_model: int,
        num_feature_levels: int,
        add_pooled_text_to_img_feat: bool = True,
        pool_text_with_mask: bool = False,
        compile_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(
            layer,
            num_layers,
            d_model,
            num_feature_levels,
            **kwargs,
        )
        self.add_pooled_text_to_img_feat = add_pooled_text_to_img_feat
        if self.add_pooled_text_to_img_feat:
            self.text_pooling_proj = nn.Linear(d_model, d_model)
        self.pool_text_with_mask = pool_text_with_mask
        if compile_mode is not None:
            self.forward = torch.compile(
                self.forward, mode=compile_mode, fullgraph=True
            )

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        return None

    def forward(
        self,
        src: list[Float[Tensor, "..."]],
        prompt: Float[Tensor, "s b d"],
        src_key_padding_mask: list[Bool[Tensor, "..."] | Float[Tensor, "..."] | None]
        | None = None,
        src_pos: list[Float[Tensor, "..."]] | None = None,
        prompt_key_padding_mask: Bool[Tensor, "b s"]
        | Float[Tensor, "b s"]
        | None = None,
        prompt_pos: Float[Tensor, "..."] | None = None,
        feat_sizes: list[tuple[int, int]] | None = None,
        encoder_extra_kwargs: Mapping[str, object] | None = None,
    ) -> dict[str, Tensor | None]:
        # Restore spatial shapes of vision when given as seq-first flat tokens.
        bs = src[0].shape[1]  # seq first when feat_sizes is set
        if feat_sizes is not None:
            assert len(feat_sizes) == len(src)
            if src_key_padding_mask is None:
                src_key_padding_mask = [None] * len(src)
            for i, (h, w) in enumerate(feat_sizes):
                src[i] = src[i].reshape(h, w, bs, -1).permute(2, 3, 0, 1)
                src_pos[i] = src_pos[i].reshape(h, w, bs, -1).permute(2, 3, 0, 1)
                src_key_padding_mask[i] = (
                    src_key_padding_mask[i].reshape(h, w, bs).permute(2, 0, 1)
                    if src_key_padding_mask[i] is not None
                    else None
                )
        else:
            # Official has a typo ``x.dim == 4``; use correct ``dim()``.
            assert all(x.dim() == 4 for x in src), (
                "expected list of (bs, c, h, w) tensors"
            )
            if src_key_padding_mask is None:
                src_key_padding_mask = [None] * len(src)

        if self.add_pooled_text_to_img_feat:
            pooled_text = pool_text_feat(
                prompt, prompt_key_padding_mask, self.pool_text_with_mask
            )
            pooled_text = self.text_pooling_proj(pooled_text)[..., None, None]
            src = [x.add_(pooled_text) for x in src]

        (
            out,
            key_padding_masks_flatten,
            lvl_pos_embed_flatten,
            level_start_index,
            spatial_shapes,
            valid_ratios,
        ) = super().forward(
            src,
            src_key_padding_masks=src_key_padding_mask,
            pos=src_pos,
            prompt=prompt.transpose(0, 1),
            prompt_key_padding_mask=prompt_key_padding_mask,
            encoder_extra_kwargs=encoder_extra_kwargs,
        )

        return {
            "memory": out,
            "padding_mask": key_padding_masks_flatten,
            "pos_embed": lvl_pos_embed_flatten,
            "memory_text": prompt,
            "level_start_index": level_start_index,
            "spatial_shapes": spatial_shapes,
            "valid_ratios": valid_ratios,
        }


def pool_text_feat(
    prompt: Float[Tensor, "s b c"],
    prompt_mask: Bool[Tensor, "b s"] | Float[Tensor, "b s"] | None,
    pool_with_mask: bool,
) -> Float[Tensor, "b c"]:
    """Mean-pool text tokens. ``prompt`` is seq-first ``(S, B, C)``."""
    if not pool_with_mask:
        return prompt.mean(dim=0)

    # prompt_mask: (B, S), False=valid, True=padding
    assert prompt_mask is not None and prompt_mask.dim() == 2
    is_valid = (~prompt_mask).float().permute(1, 0)[..., None]
    num_valid = torch.clamp(torch.sum(is_valid, dim=0), min=1.0)
    pooled_text = (prompt * is_valid).sum(dim=0) / num_valid
    return pooled_text
