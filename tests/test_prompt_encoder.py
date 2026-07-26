"""Parity tests for :class:`sam3.vision.prompt_encoder.PromptEncoder`."""

from __future__ import annotations

from typing import Optional, Tuple, Type

import pytest
import torch
import torch.nn as nn

from sam3.vision.prompt_encoder import PromptEncoder

if not torch.cuda.is_available():
    pytest.skip("CUDA is required for prompt encoder tests", allow_module_level=True)

DEVICE = torch.device("cuda")


class _TorchLayerNorm2d(nn.Module):
    """Pure torch LayerNorm over channels for NCHW input."""

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class _TorchPositionEmbeddingRandom(nn.Module):
    """Torch reference for random positional embedding used by SAM."""

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        coords = 2 * coords - 1
        matrix = self.positional_encoding_gaussian_matrix.to(coords.dtype)
        coords = coords @ matrix
        coords = coords * (2 * torch.pi)
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        h, w = size
        device = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(
            coords.to(self.positional_encoding_gaussian_matrix.dtype)
        )


class _TorchPromptEncoder(nn.Module):
    """Reference PromptEncoder implementation with plain torch operators."""

    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = _TorchPositionEmbeddingRandom(embed_dim // 2)

        self.num_point_embeddings = 4  # pos/neg point + 2 box corners
        self.point_embeddings = nn.ModuleList(
            nn.Embedding(1, embed_dim) for _ in range(self.num_point_embeddings)
        )
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        self.mask_input_size = (
            4 * image_embedding_size[0],
            4 * image_embedding_size[1],
        )
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            _TorchLayerNorm2d(mask_in_chans // 4),
            activation(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            _TorchLayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self) -> torch.Tensor:
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
        pad: bool,
    ) -> torch.Tensor:
        points = points + 0.5
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)

        point_embedding = self.pe_layer.forward_with_coords(
            points, self.input_image_size
        )

        point_embedding = torch.where(
            (labels == -1).unsqueeze(-1),
            torch.zeros_like(point_embedding) + self.not_a_point_embed.weight,
            point_embedding,
        )
        point_embedding = torch.where(
            (labels == 0).unsqueeze(-1),
            point_embedding + self.point_embeddings[0].weight,
            point_embedding,
        )
        point_embedding = torch.where(
            (labels == 1).unsqueeze(-1),
            point_embedding + self.point_embeddings[1].weight,
            point_embedding,
        )
        point_embedding = torch.where(
            (labels == 2).unsqueeze(-1),
            point_embedding + self.point_embeddings[2].weight,
            point_embedding,
        )
        point_embedding = torch.where(
            (labels == 3).unsqueeze(-1),
            point_embedding + self.point_embeddings[3].weight,
            point_embedding,
        )
        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        boxes = boxes + 0.5
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(
            coords, self.input_image_size
        )
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        return self.mask_downscaling(masks)

    def _get_batch_size(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> int:
        if points is not None:
            return points[0].shape[0]
        if boxes is not None:
            return boxes.shape[0]
        if masks is not None:
            return masks.shape[0]
        return 1

    def _get_device(self) -> torch.device:
        return self.point_embeddings[0].weight.device

    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bs = self._get_batch_size(points, boxes, masks)
        sparse_embeddings = torch.empty(
            (bs, 0, self.embed_dim), device=self._get_device()
        )

        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )
        return sparse_embeddings, dense_embeddings


def _copy_prompt_encoder_weights(
    source: PromptEncoder,
    destination: _TorchPromptEncoder,
) -> None:
    destination.pe_layer.positional_encoding_gaussian_matrix.data.copy_(
        source.pe_layer.positional_encoding_gaussian_matrix.data
    )
    destination.not_a_point_embed.weight.data.copy_(
        source.not_a_point_embed.weight.data
    )
    for i in range(source.num_point_embeddings):
        destination.point_embeddings[i].weight.data.copy_(
            source.point_embeddings[i].weight.data
        )

    destination.mask_downscaling[0].weight.data.copy_(
        source.mask_downscaling[0].weight.data
    )
    destination.mask_downscaling[0].bias.data.copy_(
        source.mask_downscaling[0].bias.data
    )
    destination.mask_downscaling[1].weight.data.copy_(
        source.mask_downscaling[1].weight.data
    )
    destination.mask_downscaling[1].bias.data.copy_(
        source.mask_downscaling[1].bias.data
    )
    destination.mask_downscaling[3].weight.data.copy_(
        source.mask_downscaling[3].weight.data
    )
    destination.mask_downscaling[3].bias.data.copy_(
        source.mask_downscaling[3].bias.data
    )
    destination.mask_downscaling[4].weight.data.copy_(
        source.mask_downscaling[4].weight.data
    )
    destination.mask_downscaling[4].bias.data.copy_(
        source.mask_downscaling[4].bias.data
    )
    destination.mask_downscaling[6].weight.data.copy_(
        source.mask_downscaling[6].weight.data
    )
    destination.mask_downscaling[6].bias.data.copy_(
        source.mask_downscaling[6].bias.data
    )
    destination.no_mask_embed.weight.data.copy_(source.no_mask_embed.weight.data)


def _build_torch_reference(source: PromptEncoder) -> _TorchPromptEncoder:
    activation = source.mask_downscaling[2].__class__
    destination = _TorchPromptEncoder(
        embed_dim=source.embed_dim,
        image_embedding_size=source.image_embedding_size,
        input_image_size=source.input_image_size,
        mask_in_chans=source.mask_downscaling[3].weight.shape[0],
        activation=activation,
    )
    _copy_prompt_encoder_weights(source, destination)
    return destination


def _tol(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float16:
        return 2e-2, 2e-3
    return 2e-3, 1e-3


def _run_parity_case(
    dtype: torch.dtype,
    points: Optional[tuple[torch.Tensor, torch.Tensor]],
    boxes: Optional[torch.Tensor],
    masks: Optional[torch.Tensor],
) -> None:
    torch.manual_seed(21)
    image_embedding_size = (4, 4)
    input_image_size = (16, 16)
    encoder = PromptEncoder(
        embed_dim=16,
        image_embedding_size=image_embedding_size,
        input_image_size=input_image_size,
        mask_in_chans=32,
        activation=nn.GELU,
    ).to(DEVICE, dtype=dtype)

    ref_encoder = _build_torch_reference(encoder).to(DEVICE, dtype=dtype)

    with torch.no_grad():
        sparse, dense = encoder(points, boxes, masks)
        sparse_ref, dense_ref = ref_encoder(points, boxes, masks)

    rtol, atol = _tol(dtype)
    torch.testing.assert_close(sparse, sparse_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(dense, dense_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_prompt_encoder_points_only(dtype: torch.dtype) -> None:
    points = torch.tensor(
        [
            [[1.0, 2.0], [5.0, 6.0], [9.0, 1.0]],
            [[2.0, 3.0], [4.0, 12.0], [8.0, 11.0]],
        ],
        device=DEVICE,
        dtype=dtype,
    )
    labels = torch.tensor(
        [[-1.0, 0.0, 1.0], [2.0, 3.0, 1.0]], device=DEVICE, dtype=dtype
    )

    _run_parity_case(dtype=dtype, points=(points, labels), boxes=None, masks=None)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_prompt_encoder_boxes_only(dtype: torch.dtype) -> None:
    boxes = torch.tensor(
        [[1.0, 2.0, 13.0, 14.0], [4.0, 1.0, 15.0, 10.0]],
        device=DEVICE,
        dtype=dtype,
    )

    _run_parity_case(dtype=dtype, points=None, boxes=boxes, masks=None)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_prompt_encoder_points_and_boxes(dtype: torch.dtype) -> None:
    points = torch.tensor(
        [
            [[2.0, 4.0], [7.0, 8.0], [11.0, 3.0]],
            [[1.0, 2.0], [6.0, 9.0], [10.0, 13.0]],
        ],
        device=DEVICE,
        dtype=dtype,
    )
    labels = torch.tensor(
        [[1.0, 0.0, 1.0], [1.0, -1.0, 0.0]], device=DEVICE, dtype=dtype
    )
    boxes = torch.tensor(
        [[2.0, 2.0, 12.0, 12.0], [4.0, 4.0, 14.0, 14.0]],
        device=DEVICE,
        dtype=dtype,
    )

    _run_parity_case(dtype=dtype, points=(points, labels), boxes=boxes, masks=None)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_prompt_encoder_masks_only(dtype: torch.dtype) -> None:
    masks = torch.randn((2, 1, 16, 16), device=DEVICE, dtype=dtype)

    _run_parity_case(dtype=dtype, points=None, boxes=None, masks=masks)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_prompt_encoder_empty_prompts(dtype: torch.dtype) -> None:
    _run_parity_case(dtype=dtype, points=None, boxes=None, masks=None)
