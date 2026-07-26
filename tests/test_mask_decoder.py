"""Parity tests for ``MaskDecoder`` against a pure-``torch`` reference."""

from __future__ import annotations

from typing import List, Tuple

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from sam3.primitives.mlp import MLP
from sam3.primitives.two_way_transformer import TwoWayTransformer
from sam3.vision.mask_decoder import MaskDecoder

if not torch.cuda.is_available():
    pytest.skip("CUDA is required for mask decoder tests", allow_module_level=True)

DEVICE = torch.device("cuda")


def _tol(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float16:
        return 2e-2, 2e-3
    return 2e-2, 3e-3


class _ReferenceAttention(nn.Module):
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
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
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
        q = self._separate_heads(self.q_proj(q), self.num_heads)
        k = self._separate_heads(self.k_proj(k), self.num_heads)
        v = self._separate_heads(self.v_proj(v), self.num_heads)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_p)
        out = self._recombine_heads(out)
        return self.out_proj(out)


class _ReferenceLayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        return (
            self.weight[:, None, None] * ((x - u) / torch.sqrt(s + self.eps))
            + self.bias[:, None, None]
        )


class _ReferenceMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        hidden = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + hidden, hidden + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < len(self.layers) - 1 else layer(x)
        if self.sigmoid_output:
            x = torch.sigmoid(x)
        return x


class _ReferenceTwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
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
        self.mlp = _ReferenceMLP(embedding_dim, mlp_dim, embedding_dim, 2)
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        queries = self.norm2(queries + attn_out)

        queries = self.norm3(queries + self.mlp(queries))

        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = self.norm4(keys + attn_out)

        return queries, keys


class _ReferenceTwoWayTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        attention_downsample_rate: int = 2,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList(
            [
                _ReferenceTwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
                for i in range(depth)
            ]
        )
        self.final_attn_token_to_image = _ReferenceAttention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: torch.Tensor,
        image_pe: torch.Tensor,
        point_embedding: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bs, _, h, w = image_embedding.shape
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
            )

        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


class _ReferenceMaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        use_high_res_features: bool = False,
        iou_prediction_use_sigmoid: bool = False,
        dynamic_multimask_via_stability: bool = False,
        dynamic_multimask_stability_delta: float = 0.05,
        dynamic_multimask_stability_thresh: float = 0.98,
        pred_obj_scores: bool = False,
        pred_obj_scores_mlp: bool = False,
        use_multimask_token_for_obj_ptr: bool = False,
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_multimask_outputs = num_multimask_outputs
        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)
        self.pred_obj_scores = pred_obj_scores
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr
        if pred_obj_scores:
            self.obj_score_token = nn.Embedding(1, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            ),
            _ReferenceLayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(
                transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
            ),
            activation(),
        )
        self.use_high_res_features = use_high_res_features
        if use_high_res_features:
            self.conv_s0 = nn.Conv2d(
                transformer_dim, transformer_dim // 8, kernel_size=1, stride=1
            )
            self.conv_s1 = nn.Conv2d(
                transformer_dim, transformer_dim // 4, kernel_size=1, stride=1
            )

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                _ReferenceMLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for _ in range(self.num_mask_tokens)
            ]
        )
        self.iou_prediction_head = _ReferenceMLP(
            transformer_dim,
            iou_head_hidden_dim,
            self.num_mask_tokens,
            iou_head_depth,
            sigmoid_output=iou_prediction_use_sigmoid,
        )
        if pred_obj_scores:
            if pred_obj_scores_mlp:
                self.pred_obj_score_head = _ReferenceMLP(
                    transformer_dim, transformer_dim, 1, 3
                )
            else:
                self.pred_obj_score_head = nn.Linear(transformer_dim, 1)

        self.dynamic_multimask_via_stability = dynamic_multimask_via_stability
        self.dynamic_multimask_stability_delta = dynamic_multimask_stability_delta
        self.dynamic_multimask_stability_thresh = dynamic_multimask_stability_thresh

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        repeat_image: bool,
        high_res_features: List[torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        masks, iou_pred, mask_tokens_out, object_score_logits = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            repeat_image=repeat_image,
            high_res_features=high_res_features,
        )

        if multimask_output:
            masks = masks[:, 1:, :, :]
            iou_pred = iou_pred[:, 1:]
        elif self.dynamic_multimask_via_stability and not self.training:
            masks, iou_pred = self._dynamic_multimask_via_stability(masks, iou_pred)
        else:
            masks = masks[:, 0:1, :, :]
            iou_pred = iou_pred[:, 0:1]

        if multimask_output and self.use_multimask_token_for_obj_ptr:
            sam_tokens_out = mask_tokens_out[:, 1:]
        else:
            sam_tokens_out = mask_tokens_out[:, 0:1]

        return masks, iou_pred, sam_tokens_out, object_score_logits

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        repeat_image: bool,
        high_res_features: List[torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s = 0
        if self.pred_obj_scores:
            output_tokens = torch.cat(
                [
                    self.obj_score_token.weight,
                    self.iou_token.weight,
                    self.mask_tokens.weight,
                ],
                dim=0,
            )
            s = 1
        else:
            output_tokens = torch.cat(
                [self.iou_token.weight, self.mask_tokens.weight], dim=0
            )
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        if repeat_image:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            src = image_embeddings
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)

        b, c, h, w = src.shape
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, s, :]
        mask_tokens_out = hs[:, s + 1 : s + 1 + self.num_mask_tokens, :]

        src = src.transpose(1, 2).view(b, c, h, w)
        if not self.use_high_res_features:
            upscaled_embedding = self.output_upscaling(src)
        else:
            if high_res_features is None:
                raise ValueError("high_res_features must be provided")
            dc1, ln1, act1, dc2, act2 = self.output_upscaling
            feat_s0, feat_s1 = high_res_features
            upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
            upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)

        _, c, h, w = upscaled_embedding.shape
        hyper_in = torch.stack(
            [
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
                for i in range(self.num_mask_tokens)
            ],
            dim=1,
        )
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        iou_pred = self.iou_prediction_head(iou_token_out)
        if self.pred_obj_scores:
            object_score_logits = self.pred_obj_score_head(hs[:, 0, :])
        else:
            object_score_logits = 10.0 * iou_pred.new_ones(iou_pred.shape[0], 1)
        return masks, iou_pred, mask_tokens_out, object_score_logits

    def _get_stability_scores(self, mask_logits: torch.Tensor) -> torch.Tensor:
        mask_logits = mask_logits.flatten(-2)
        area_i = torch.sum(
            mask_logits > self.dynamic_multimask_stability_delta, dim=-1
        ).float()
        area_u = torch.sum(
            mask_logits > -self.dynamic_multimask_stability_delta, dim=-1
        ).float()
        return torch.where(
            area_u > 0, area_i / area_u, torch.tensor(1.0, device=mask_logits.device)
        )

    def _dynamic_multimask_via_stability(
        self, all_mask_logits: torch.Tensor, all_iou_scores: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        multimask_logits = all_mask_logits[:, 1:, :, :]
        multimask_iou_scores = all_iou_scores[:, 1:]
        best_scores_inds = torch.argmax(multimask_iou_scores, dim=-1)
        batch_inds = torch.arange(
            multimask_iou_scores.size(0), device=all_iou_scores.device
        )
        best_multimask_logits = multimask_logits[batch_inds, best_scores_inds]
        best_multimask_iou_scores = multimask_iou_scores[batch_inds, best_scores_inds]

        singlemask_logits = all_mask_logits[:, 0:1, :, :]
        singlemask_iou_scores = all_iou_scores[:, 0:1]
        stability_scores = self._get_stability_scores(singlemask_logits)
        is_stable = stability_scores >= self.dynamic_multimask_stability_thresh

        return (
            torch.where(
                is_stable[..., None, None].expand_as(singlemask_logits),
                singlemask_logits,
                best_multimask_logits[:, None, :, :],
            ),
            torch.where(
                is_stable.expand_as(singlemask_iou_scores),
                singlemask_iou_scores,
                best_multimask_iou_scores[:, None],
            ),
        )


def _copy_attention_weights(source: nn.Module, target: _ReferenceAttention) -> None:
    target.q_proj.weight.data.copy_(source.q_proj.weight.data)
    target.q_proj.bias.data.copy_(source.q_proj.bias.data)
    target.k_proj.weight.data.copy_(source.k_proj.weight.data)
    target.k_proj.bias.data.copy_(source.k_proj.bias.data)
    target.v_proj.weight.data.copy_(source.v_proj.weight.data)
    target.v_proj.bias.data.copy_(source.v_proj.bias.data)
    target.out_proj.weight.data.copy_(source.out_proj.weight.data)
    target.out_proj.bias.data.copy_(source.out_proj.bias.data)


def _copy_layernorm_weights(
    source: nn.Module, target: nn.LayerNorm | _ReferenceLayerNorm2d
) -> None:
    target.weight.data.copy_(source.weight.data)
    target.bias.data.copy_(source.bias.data)


def _copy_mlp_weights(source: MLP | _ReferenceMLP, target: _ReferenceMLP) -> None:
    if hasattr(source, "layers"):
        for src_layer, dst_layer in zip(source.layers, target.layers):
            dst_layer.weight.data.copy_(src_layer.weight.data)
            dst_layer.bias.data.copy_(src_layer.bias.data)
        return

    src_lin1 = source.lin1
    src_lin2 = source.lin2
    target.layers[0].weight.data.copy_(src_lin1.weight.data)
    target.layers[0].bias.data.copy_(src_lin1.bias.data)
    target.layers[1].weight.data.copy_(src_lin2.weight.data)
    target.layers[1].bias.data.copy_(src_lin2.bias.data)


def _copy_conv2d(source: nn.Conv2d, target: nn.Conv2d) -> None:
    target.weight.data.copy_(source.weight.data)
    if source.bias is not None:
        target.bias.data.copy_(source.bias.data)


def _build_reference_two_way_transformer(
    source: TwoWayTransformer,
) -> _ReferenceTwoWayTransformer:
    attention_downsample_rate = (
        source.layers[0].cross_attn_token_to_image.kv_in_dim
        // source.layers[0].cross_attn_token_to_image.internal_dim
    )
    target = _ReferenceTwoWayTransformer(
        depth=source.depth,
        embedding_dim=source.embedding_dim,
        num_heads=source.num_heads,
        mlp_dim=source.mlp_dim,
        attention_downsample_rate=attention_downsample_rate,
    )
    for src_layer, tgt_layer in zip(source.layers, target.layers):
        _copy_attention_weights(src_layer.self_attn, tgt_layer.self_attn)
        _copy_attention_weights(
            src_layer.cross_attn_token_to_image, tgt_layer.cross_attn_token_to_image
        )
        _copy_attention_weights(
            src_layer.cross_attn_image_to_token, tgt_layer.cross_attn_image_to_token
        )
        _copy_layernorm_weights(src_layer.norm1, tgt_layer.norm1)
        _copy_layernorm_weights(src_layer.norm2, tgt_layer.norm2)
        _copy_layernorm_weights(src_layer.norm3, tgt_layer.norm3)
        _copy_layernorm_weights(src_layer.norm4, tgt_layer.norm4)
        _copy_mlp_weights(src_layer.mlp, tgt_layer.mlp)

    _copy_attention_weights(
        source.final_attn_token_to_image, target.final_attn_token_to_image
    )
    _copy_layernorm_weights(source.norm_final_attn, target.norm_final_attn)
    return target


def _build_reference_mask_decoder(source: MaskDecoder) -> _ReferenceMaskDecoder:
    transformer = _build_reference_two_way_transformer(source.transformer)
    source_pred_obj_score = source.pred_obj_scores
    source_pred_obj_score_mlp = source_pred_obj_score and not isinstance(
        source.pred_obj_score_head, nn.Linear
    )

    target = _ReferenceMaskDecoder(
        transformer_dim=source.transformer_dim,
        transformer=transformer,
        num_multimask_outputs=source.num_multimask_outputs,
        iou_head_depth=source.iou_prediction_head.num_layers,
        iou_head_hidden_dim=source.iou_prediction_head.layers[0].out_features,
        iou_prediction_use_sigmoid=source.iou_prediction_head.sigmoid_output,
        use_high_res_features=source.use_high_res_features,
        dynamic_multimask_via_stability=source.dynamic_multimask_via_stability,
        dynamic_multimask_stability_delta=source.dynamic_multimask_stability_delta,
        dynamic_multimask_stability_thresh=source.dynamic_multimask_stability_thresh,
        pred_obj_scores=source_pred_obj_score,
        pred_obj_scores_mlp=source_pred_obj_score_mlp,
        use_multimask_token_for_obj_ptr=source.use_multimask_token_for_obj_ptr,
    )

    target.iou_token.weight.data.copy_(source.iou_token.weight.data)
    target.mask_tokens.weight.data.copy_(source.mask_tokens.weight.data)
    if source_pred_obj_score:
        target.obj_score_token.weight.data.copy_(source.obj_score_token.weight.data)

    _copy_conv2d(source.output_upscaling[0], target.output_upscaling[0])
    _copy_layernorm_weights(source.output_upscaling[1], target.output_upscaling[1])
    _copy_conv2d(source.output_upscaling[3], target.output_upscaling[3])

    for src_mlp, tgt_mlp in zip(
        source.output_hypernetworks_mlps, target.output_hypernetworks_mlps
    ):
        _copy_mlp_weights(src_mlp, tgt_mlp)
    _copy_mlp_weights(source.iou_prediction_head, target.iou_prediction_head)

    if source_pred_obj_score:
        if source_pred_obj_score_mlp:
            _copy_mlp_weights(source.pred_obj_score_head, target.pred_obj_score_head)  # type: ignore[arg-type]
        else:
            target.pred_obj_score_head.weight.data.copy_(  # type: ignore[union-attr]
                source.pred_obj_score_head.weight.data
            )
            target.pred_obj_score_head.bias.data.copy_(  # type: ignore[union-attr]
                source.pred_obj_score_head.bias.data
            )

    return target


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("multimask_output", [False, True])
def test_mask_decoder_matches_torch_reference(
    dtype: torch.dtype, multimask_output: bool
) -> None:
    torch.manual_seed(0)

    transformer = TwoWayTransformer(
        depth=1,
        embedding_dim=32,
        num_heads=4,
        mlp_dim=64,
    ).to(device=DEVICE, dtype=dtype)
    transformer.eval()
    decoder = MaskDecoder(
        transformer_dim=32,
        transformer=transformer,
        num_multimask_outputs=3,
        iou_head_depth=3,
        iou_head_hidden_dim=64,
        pred_obj_scores=False,
    ).to(device=DEVICE, dtype=dtype)
    decoder.eval()

    reference = _build_reference_mask_decoder(decoder).to(device=DEVICE, dtype=dtype)
    reference.eval()

    image_embeddings = torch.randn(2, 32, 8, 8, device=DEVICE, dtype=dtype)
    image_pe = torch.randn(1, 32, 8, 8, device=DEVICE, dtype=dtype)
    sparse_prompt_embeddings = torch.randn(2, 4, 32, device=DEVICE, dtype=dtype)
    dense_prompt_embeddings = torch.randn(2, 32, 8, 8, device=DEVICE, dtype=dtype)

    with torch.no_grad():
        triton_out = decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            multimask_output=multimask_output,
            repeat_image=False,
        )
        torch_out = reference(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            multimask_output=multimask_output,
            repeat_image=False,
        )

    rtol, atol = _tol(dtype)
    torch.testing.assert_close(triton_out[0], torch_out[0], rtol=rtol, atol=atol)
    torch.testing.assert_close(triton_out[1], torch_out[1], rtol=rtol, atol=atol)
    torch.testing.assert_close(triton_out[2], torch_out[2], rtol=rtol, atol=atol)
    torch.testing.assert_close(triton_out[3], torch_out[3], rtol=rtol, atol=atol)


def test_stability_helpers_match_torch_reference() -> None:
    torch.manual_seed(1)
    transformer = TwoWayTransformer(
        depth=1,
        embedding_dim=32,
        num_heads=4,
        mlp_dim=64,
    ).to(device=DEVICE)
    transformer.eval()
    decoder = MaskDecoder(
        transformer_dim=32,
        transformer=transformer,
        num_multimask_outputs=3,
        pred_obj_scores=False,
        dynamic_multimask_via_stability=True,
    ).to(device=DEVICE)
    decoder.eval()
    reference = _build_reference_mask_decoder(decoder).to(device=DEVICE)
    reference.eval()

    mask_logits = torch.randn(2, 4, 12, 13, device=DEVICE, dtype=torch.float32)
    iou_scores = torch.randn(2, 4, device=DEVICE, dtype=torch.float32)

    torch.testing.assert_close(
        decoder._get_stability_scores(mask_logits),
        reference._get_stability_scores(mask_logits),
        rtol=1e-4,
        atol=1e-4,
    )
    out_mask_dec = decoder._dynamic_multimask_via_stability(mask_logits, iou_scores)
    out_mask_ref = reference._dynamic_multimask_via_stability(mask_logits, iou_scores)
    torch.testing.assert_close(out_mask_dec[0], out_mask_ref[0], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(out_mask_dec[1], out_mask_ref[1], rtol=1e-4, atol=1e-4)
