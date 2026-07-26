"""End-to-end integration: PromptEncoder → TwoWayTransformer → MaskDecoder."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from sam3.primitives.mlp import MLP
from sam3.vision.sam_image_head import SamImageHead

if not torch.cuda.is_available():
    pytest.skip(
        "CUDA is required for SAM head integration tests", allow_module_level=True
    )

DEVICE = torch.device("cuda")


def _tol(dtype: torch.dtype) -> tuple[float, float]:
    # Attention stacks accumulate numerical error; keep practical tolerances for fp16 head.
    if dtype == torch.float16:
        return 5e-2, 5e-3
    return 2e-2, 3e-3


# ---------------------------------------------------------------------------
# Minimal pure-torch reference stack (mirrors structures used in unit tests)
# ---------------------------------------------------------------------------


class _RefAttention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, downsample_rate: int = 1):
        super().__init__()
        self.num_heads = num_heads
        self.internal_dim = embedding_dim // downsample_rate
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _sep(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        return x.reshape(b, n, self.num_heads, c // self.num_heads).transpose(1, 2)

    def _comb(self, x: torch.Tensor) -> torch.Tensor:
        b, h, n, d = x.shape
        return x.transpose(1, 2).reshape(b, n, h * d)

    def forward(self, q, k, v):
        q, k, v = (
            self._sep(self.q_proj(q)),
            self._sep(self.k_proj(k)),
            self._sep(self.v_proj(v)),
        )
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        return self.out_proj(self._comb(out))


class _RefLN2d(nn.Module):
    def __init__(self, c: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        return (
            self.weight[:, None, None] * (x - u) / torch.sqrt(s + self.eps)
            + self.bias[:, None, None]
        )


class _RefMLP(nn.Module):
    def __init__(self, i, h, o, n, sigmoid_output=False):
        super().__init__()
        dims = [i] + [h] * (n - 1) + [o]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims, dims[1:]))
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i + 1 < len(self.layers) else layer(x)
        return torch.sigmoid(x) if self.sigmoid_output else x


class _RefMLPBlock(nn.Module):
    def __init__(self, d, m):
        super().__init__()
        self.lin1 = nn.Linear(d, m)
        self.lin2 = nn.Linear(m, d)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.lin2(self.act(self.lin1(x)))


class _RefTwoWayBlock(nn.Module):
    def __init__(self, d, heads, mlp_dim, downsample, skip_first):
        super().__init__()
        self.self_attn = _RefAttention(d, heads)
        self.norm1 = nn.LayerNorm(d)
        self.cross_attn_token_to_image = _RefAttention(d, heads, downsample)
        self.norm2 = nn.LayerNorm(d)
        self.mlp = _RefMLPBlock(d, mlp_dim)
        self.norm3 = nn.LayerNorm(d)
        self.norm4 = nn.LayerNorm(d)
        self.cross_attn_image_to_token = _RefAttention(d, heads, downsample)
        self.skip_first_layer_pe = skip_first

    def forward(self, queries, keys, query_pe, key_pe):
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            queries = queries + self.self_attn(q=q, k=q, v=queries)
        queries = self.norm1(queries)
        q, k = queries + query_pe, keys + key_pe
        queries = self.norm2(queries + self.cross_attn_token_to_image(q=q, k=k, v=keys))
        queries = self.norm3(queries + self.mlp(queries))
        q, k = queries + query_pe, keys + key_pe
        keys = self.norm4(keys + self.cross_attn_image_to_token(q=k, k=q, v=queries))
        return queries, keys


class _RefTwoWay(nn.Module):
    def __init__(self, depth, d, heads, mlp_dim, downsample=2):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _RefTwoWayBlock(d, heads, mlp_dim, downsample, skip_first=(i == 0))
                for i in range(depth)
            ]
        )
        self.final_attn_token_to_image = _RefAttention(d, heads, downsample)
        self.norm_final_attn = nn.LayerNorm(d)

    def forward(self, image_embedding, image_pe, point_embedding):
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)
        queries, keys = point_embedding, image_embedding
        for layer in self.layers:
            queries, keys = layer(queries, keys, point_embedding, image_pe)
        q, k = queries + point_embedding, keys + image_pe
        queries = self.norm_final_attn(
            queries + self.final_attn_token_to_image(q=q, k=k, v=keys)
        )
        return queries, keys


class _RefPosEnc(nn.Module):
    def __init__(self, num_pos_feats: int = 16):
        super().__init__()
        self.register_buffer(
            "positional_encoding_gaussian_matrix", torch.randn(2, num_pos_feats)
        )

    def _pe(self, coords):
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix.to(coords.dtype)
        coords = coords * (2 * torch.pi)
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size):
        h, w = size
        g = torch.ones(h, w, device=self.positional_encoding_gaussian_matrix.device)
        y = (g.cumsum(0) - 0.5) / h
        x = (g.cumsum(1) - 0.5) / w
        return self._pe(torch.stack([x, y], -1)).permute(2, 0, 1)

    def forward_with_coords(self, coords_input, image_size):
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe(coords.float())


class _RefPromptEncoder(nn.Module):
    def __init__(
        self, embed_dim, image_embedding_size, input_image_size, mask_in_chans
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.image_embedding_size = image_embedding_size
        self.input_image_size = input_image_size
        self.pe_layer = _RefPosEnc(embed_dim // 2)
        self.point_embeddings = nn.ModuleList(
            [nn.Embedding(1, embed_dim) for _ in range(4)]
        )
        self.not_a_point_embed = nn.Embedding(1, embed_dim)
        self.mask_input_size = (
            4 * image_embedding_size[0],
            4 * image_embedding_size[1],
        )
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, 2, 2),
            _RefLN2d(mask_in_chans // 4),
            nn.GELU(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, 2, 2),
            _RefLN2d(mask_in_chans),
            nn.GELU(),
            nn.Conv2d(mask_in_chans, embed_dim, 1),
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self):
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(self, points, labels, pad):
        points = points + 0.5
        if pad:
            points = torch.cat(
                [points, torch.zeros(points.shape[0], 1, 2, device=points.device)],
                dim=1,
            )
            labels = torch.cat(
                [labels, -torch.ones(labels.shape[0], 1, device=labels.device)], dim=1
            )
        pe = self.pe_layer.forward_with_coords(points, self.input_image_size)
        pe = torch.where(
            (labels == -1).unsqueeze(-1), self.not_a_point_embed.weight + 0, pe
        )
        for i in range(4):
            pe = torch.where(
                (labels == i).unsqueeze(-1), pe + self.point_embeddings[i].weight, pe
            )
        return pe

    def _embed_boxes(self, boxes):
        boxes = boxes + 0.5
        coords = boxes.reshape(-1, 2, 2)
        corner = self.pe_layer.forward_with_coords(coords, self.input_image_size)
        corner[:, 0, :] += self.point_embeddings[2].weight
        corner[:, 1, :] += self.point_embeddings[3].weight
        return corner

    def forward(self, points=None, boxes=None, masks=None):
        if points is not None:
            bs = points[0].shape[0]
        elif boxes is not None:
            bs = boxes.shape[0]
        elif masks is not None:
            bs = masks.shape[0]
        else:
            bs = 1
        sparse = torch.empty(
            bs, 0, self.embed_dim, device=self.no_mask_embed.weight.device
        )
        if points is not None:
            sparse = torch.cat(
                [sparse, self._embed_points(points[0], points[1], pad=(boxes is None))],
                dim=1,
            )
        if boxes is not None:
            sparse = torch.cat([sparse, self._embed_boxes(boxes)], dim=1)
        if masks is not None:
            dense = self.mask_downscaling(masks)
        else:
            h, w = self.image_embedding_size
            dense = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(bs, -1, h, w)
        return sparse, dense


class _RefMaskDecoder(nn.Module):
    def __init__(self, transformer_dim, transformer, num_multimask_outputs=3):
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_multimask_outputs = num_multimask_outputs
        self.num_mask_tokens = num_multimask_outputs + 1
        self.iou_token = nn.Embedding(1, transformer_dim)
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)
        self.pred_obj_scores = False
        self.use_multimask_token_for_obj_ptr = False
        self.dynamic_multimask_via_stability = False
        self.use_high_res_features = False
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, 2, 2),
            _RefLN2d(transformer_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, 2, 2),
            nn.GELU(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                _RefMLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for _ in range(self.num_mask_tokens)
            ]
        )
        self.iou_prediction_head = _RefMLP(
            transformer_dim, max(transformer_dim, 32), self.num_mask_tokens, 3
        )

    def forward(
        self,
        image_embeddings,
        image_pe,
        sparse_prompt_embeddings,
        dense_prompt_embeddings,
        multimask_output,
        repeat_image,
        high_res_features=None,
    ):
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], 0)
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        tokens = torch.cat([output_tokens, sparse_prompt_embeddings], 1)
        if repeat_image:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], 0)
        else:
            src = image_embeddings
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], 0)
        b, c, h, w = src.shape
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : 1 + self.num_mask_tokens, :]
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled = self.output_upscaling(src)
        hyper = torch.stack(
            [
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
                for i in range(self.num_mask_tokens)
            ],
            1,
        )
        _, c, h, w = upscaled.shape
        masks = (hyper @ upscaled.view(b, c, h * w)).view(b, -1, h, w)
        iou_pred = self.iou_prediction_head(iou_token_out)
        obj = 10.0 * iou_pred.new_ones(iou_pred.shape[0], 1)
        if multimask_output:
            masks, iou_pred = masks[:, 1:], iou_pred[:, 1:]
            sam = mask_tokens_out[:, 0:1]
        else:
            masks, iou_pred = masks[:, 0:1], iou_pred[:, 0:1]
            sam = mask_tokens_out[:, 0:1]
        return masks, iou_pred, sam, obj


# ---------------------------------------------------------------------------
# Weight copy
# ---------------------------------------------------------------------------


def _copy_linear(src, dst):
    dst.weight.data.copy_(src.weight.data)
    if src.bias is not None and dst.bias is not None:
        dst.bias.data.copy_(src.bias.data)


def _copy_attn(src, dst: _RefAttention):
    _copy_linear(src.q_proj, dst.q_proj)
    _copy_linear(src.k_proj, dst.k_proj)
    _copy_linear(src.v_proj, dst.v_proj)
    _copy_linear(src.out_proj, dst.out_proj)


def _copy_ln(src, dst):
    dst.weight.data.copy_(src.weight.data)
    dst.bias.data.copy_(src.bias.data)


def _copy_mlp_block(src, dst: _RefMLPBlock):
    _copy_linear(src.lin1, dst.lin1)
    _copy_linear(src.lin2, dst.lin2)


def _copy_mlp_list(src: MLP | _RefMLP, dst: _RefMLP):
    for a, b in zip(src.layers, dst.layers):
        _copy_linear(a, b)


def _build_ref_head(src: SamImageHead) -> tuple[_RefPromptEncoder, _RefMaskDecoder]:
    pe = src.prompt_encoder
    mask_in_chans = pe.mask_downscaling[3].out_channels
    ref_pe = _RefPromptEncoder(
        pe.embed_dim, pe.image_embedding_size, pe.input_image_size, mask_in_chans
    )
    ref_pe.pe_layer.positional_encoding_gaussian_matrix.copy_(
        pe.pe_layer.positional_encoding_gaussian_matrix
    )
    for i in range(4):
        ref_pe.point_embeddings[i].weight.data.copy_(pe.point_embeddings[i].weight.data)
    ref_pe.not_a_point_embed.weight.data.copy_(pe.not_a_point_embed.weight.data)
    ref_pe.no_mask_embed.weight.data.copy_(pe.no_mask_embed.weight.data)

    # mask downscaling: Conv, LN, GELU, Conv, LN, GELU, Conv
    def _copy_linear_conv(source, destination) -> None:
        destination.weight.data.copy_(source.weight.data)
        if source.bias is not None:
            assert destination.bias is not None
            destination.bias.data.copy_(source.bias.data)

    _copy_linear_conv(pe.mask_downscaling[0], ref_pe.mask_downscaling[0])
    _copy_ln(pe.mask_downscaling[1], ref_pe.mask_downscaling[1])
    _copy_linear_conv(pe.mask_downscaling[3], ref_pe.mask_downscaling[3])
    _copy_ln(pe.mask_downscaling[4], ref_pe.mask_downscaling[4])
    _copy_linear_conv(pe.mask_downscaling[6], ref_pe.mask_downscaling[6])

    tr = src.transformer
    ref_tr = _RefTwoWay(
        tr.depth, tr.embedding_dim, tr.num_heads, tr.mlp_dim, downsample=2
    )
    for sl, tl in zip(tr.layers, ref_tr.layers):
        _copy_attn(sl.self_attn, tl.self_attn)
        _copy_attn(sl.cross_attn_token_to_image, tl.cross_attn_token_to_image)
        _copy_attn(sl.cross_attn_image_to_token, tl.cross_attn_image_to_token)
        _copy_ln(sl.norm1, tl.norm1)
        _copy_ln(sl.norm2, tl.norm2)
        _copy_ln(sl.norm3, tl.norm3)
        _copy_ln(sl.norm4, tl.norm4)
        _copy_mlp_block(sl.mlp, tl.mlp)
    _copy_attn(tr.final_attn_token_to_image, ref_tr.final_attn_token_to_image)
    _copy_ln(tr.norm_final_attn, ref_tr.norm_final_attn)

    md = src.mask_decoder
    ref_md = _RefMaskDecoder(md.transformer_dim, ref_tr, md.num_multimask_outputs)
    ref_md.iou_token.weight.data.copy_(md.iou_token.weight.data)
    ref_md.mask_tokens.weight.data.copy_(md.mask_tokens.weight.data)
    ref_md.output_upscaling[0].weight.data.copy_(md.output_upscaling[0].weight.data)
    ref_md.output_upscaling[0].bias.data.copy_(md.output_upscaling[0].bias.data)
    _copy_ln(md.output_upscaling[1], ref_md.output_upscaling[1])
    ref_md.output_upscaling[3].weight.data.copy_(md.output_upscaling[3].weight.data)
    ref_md.output_upscaling[3].bias.data.copy_(md.output_upscaling[3].bias.data)
    for a, b in zip(md.output_hypernetworks_mlps, ref_md.output_hypernetworks_mlps):
        _copy_mlp_list(a, b)
    _copy_mlp_list(md.iou_prediction_head, ref_md.iou_prediction_head)
    return ref_pe, ref_md


def _run_ref(
    ref_pe,
    ref_md,
    image_embeddings,
    points=None,
    boxes=None,
    masks=None,
    multimask=True,
):
    sparse, dense = ref_pe(points=points, boxes=boxes, masks=masks)
    dtype = image_embeddings.dtype
    device = image_embeddings.device
    sparse = sparse.to(device=device, dtype=dtype)
    dense = dense.to(device=device, dtype=dtype)
    image_pe = ref_pe.get_dense_pe().to(device=device, dtype=dtype)
    return ref_md(
        image_embeddings,
        image_pe,
        sparse,
        dense,
        multimask_output=multimask,
        repeat_image=False,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _make_head(dtype: torch.dtype = torch.float32) -> SamImageHead:
    torch.manual_seed(0)
    head = SamImageHead(
        embed_dim=32,
        image_embedding_size=(8, 8),
        input_image_size=(32, 32),
        mask_in_chans=16,
        transformer_depth=1,
        transformer_heads=4,
        transformer_mlp_dim=64,
    ).to(device=DEVICE, dtype=dtype)
    head.eval()
    return head


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("multimask", [False, True])
def test_sam_head_points_chain_parity(dtype: torch.dtype, multimask: bool):
    torch.manual_seed(11)
    head = _make_head(dtype)
    b, c, h, w = 2, 32, 8, 8
    image = torch.randn(b, c, h, w, device=DEVICE, dtype=dtype)
    coords = torch.rand(b, 3, 2, device=DEVICE, dtype=dtype) * 32
    labels = torch.randint(0, 2, (b, 3), device=DEVICE)

    ref_pe, ref_md = _build_ref_head(head)
    ref_pe = ref_pe.to(DEVICE).to(dtype)
    ref_md = ref_md.to(DEVICE).to(dtype)
    ref_pe.eval()
    ref_md.eval()

    with torch.no_grad():
        out_t = head(image, points=(coords, labels), multimask_output=multimask)
        out_r = _run_ref(
            ref_pe, ref_md, image, points=(coords, labels), multimask=multimask
        )

    rtol, atol = _tol(dtype)
    for a, b_ in zip(out_t, out_r):
        assert a.shape == b_.shape
        torch.testing.assert_close(a, b_, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_sam_head_boxes_chain_parity(dtype: torch.dtype):
    torch.manual_seed(13)
    head = _make_head(dtype)
    b = 2
    image = torch.randn(b, 32, 8, 8, device=DEVICE, dtype=dtype)
    # XYXY boxes
    boxes = torch.tensor(
        [[2.0, 3.0, 20.0, 22.0], [1.0, 1.0, 15.0, 18.0]], device=DEVICE, dtype=dtype
    )

    ref_pe, ref_md = _build_ref_head(head)
    ref_pe = ref_pe.to(DEVICE).to(dtype)
    ref_md = ref_md.to(DEVICE).to(dtype)

    with torch.no_grad():
        out_t = head(image, boxes=boxes, multimask_output=True)
        out_r = _run_ref(ref_pe, ref_md, image, boxes=boxes, multimask=True)

    rtol, atol = _tol(dtype)
    for a, b_ in zip(out_t, out_r):
        torch.testing.assert_close(a, b_, rtol=rtol, atol=atol)


def test_sam_head_smoke_shapes():
    """Sanity: full chain produces finite outputs with expected ranks."""
    torch.manual_seed(3)
    head = _make_head(torch.float32)
    b, c, h, w = 2, 32, 8, 8
    image = torch.randn(b, c, h, w, device=DEVICE)
    coords = torch.rand(b, 2, 2, device=DEVICE) * 32
    labels = torch.ones(b, 2, device=DEVICE)
    with torch.no_grad():
        masks, iou, tokens, obj = head(
            image, points=(coords, labels), multimask_output=True
        )
    # multimask: 3 masks, upscaled 4x → 32x32
    assert masks.shape == (b, 3, 32, 32)
    assert iou.shape == (b, 3)
    assert tokens.shape[0] == b and tokens.shape[-1] == c
    assert obj.shape[0] == b
    assert torch.isfinite(masks).all()
    assert torch.isfinite(iou).all()


def test_sam_head_empty_prompts_smoke():
    head = _make_head(torch.float32)
    image = torch.randn(1, 32, 8, 8, device=DEVICE)
    with torch.no_grad():
        masks, iou, tokens, obj = head(image, multimask_output=False)
    assert masks.shape == (1, 1, 32, 32)
    assert torch.isfinite(masks).all()
