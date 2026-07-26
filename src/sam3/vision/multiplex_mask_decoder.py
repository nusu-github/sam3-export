"""Checkpoint-native 16-slot SAM3.1 Multiplex mask decoder."""

from __future__ import annotations

from timm.layers import LayerNorm2d
import torch
from torch import Tensor, nn

from sam3.primitives.mlp import MLP
from sam3.primitives.two_way_transformer import TwoWayTransformer


class MultiplexMaskDecoder(nn.Module):
    """Decode three masks per slot from one shared bucket feature map."""

    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        multiplex_count: int = 16,
        num_multimask_outputs: int = 3,
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.multiplex_count = multiplex_count
        self.num_multimask_outputs = num_multimask_outputs
        self.num_mask_output_per_object = num_multimask_outputs
        self.num_mask_tokens = multiplex_count * num_multimask_outputs
        self.iou_token = nn.Embedding(multiplex_count, transformer_dim)
        self.obj_score_token = nn.Embedding(multiplex_count, transformer_dim)
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            ),
            LayerNorm2d(transformer_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(
                transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
            ),
            nn.GELU(),
        )
        self.conv_s0 = nn.Conv2d(
            transformer_dim, transformer_dim // 8, kernel_size=1, stride=1
        )
        self.conv_s1 = nn.Conv2d(
            transformer_dim, transformer_dim // 4, kernel_size=1, stride=1
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for _ in range(num_multimask_outputs)
            ]
        )
        self.iou_prediction_head = MLP(
            transformer_dim,
            transformer_dim,
            num_multimask_outputs,
            3,
            sigmoid_output=False,
        )
        self.pred_obj_score_head = MLP(transformer_dim, transformer_dim, 1, 3)

    def forward(
        self,
        image_embeddings: Tensor,
        image_pe: Tensor,
        high_res_features: tuple[Tensor, Tensor] | list[Tensor],
        extra_per_object_embeddings: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch = image_embeddings.shape[0]
        tokens = torch.cat((self.obj_score_token.weight, self.iou_token.weight), dim=0)
        tokens = tokens.unsqueeze(0).expand(batch, -1, -1)
        mask_tokens = self.mask_tokens.weight.view(
            1,
            self.multiplex_count,
            self.num_mask_output_per_object,
            self.transformer_dim,
        ).expand(batch, -1, -1, -1)
        mask_tokens = mask_tokens + extra_per_object_embeddings.unsqueeze(2)
        tokens = torch.cat((tokens, mask_tokens.flatten(1, 2)), dim=1)
        if image_pe.shape[0] != 1:
            raise ValueError("image_pe must contain one shared frame position")
        pos_src = image_pe.expand(batch, -1, -1, -1)
        hidden, source = self.transformer(image_embeddings, pos_src, tokens)
        offset = 0
        object_tokens = hidden[:, offset : offset + self.multiplex_count]
        offset += self.multiplex_count
        iou_tokens = hidden[:, offset : offset + self.multiplex_count]
        offset += self.multiplex_count
        mask_tokens_out = hidden[:, offset:]
        mask_tokens_out = mask_tokens_out.view(
            batch,
            self.multiplex_count,
            self.num_mask_output_per_object,
            self.transformer_dim,
        )
        source = source.transpose(1, 2).view_as(image_embeddings)
        dc1, ln1, act1, dc2, act2 = self.output_upscaling
        feat_s0, feat_s1 = high_res_features
        upscaled = act1(ln1(dc1(source) + feat_s1))
        upscaled = act2(dc2(upscaled) + feat_s0)
        hyper = torch.stack(
            [
                network(mask_tokens_out[:, :, index])
                for index, network in enumerate(self.output_hypernetworks_mlps)
            ],
            dim=2,
        )
        _, channels, height, width = upscaled.shape
        masks = torch.bmm(
            hyper.flatten(1, 2), upscaled.view(batch, channels, height * width)
        ).view(
            batch,
            self.multiplex_count,
            self.num_mask_output_per_object,
            height,
            width,
        )
        iou = self.iou_prediction_head(iou_tokens).view(
            batch, self.multiplex_count, self.num_mask_output_per_object
        )
        object_scores = self.pred_obj_score_head(object_tokens)
        return masks, iou, mask_tokens_out, object_scores


def create_multiplex_mask_decoder() -> MultiplexMaskDecoder:
    return MultiplexMaskDecoder(
        transformer_dim=256,
        transformer=TwoWayTransformer(
            depth=2,
            embedding_dim=256,
            mlp_dim=2048,
            num_heads=8,
        ),
        multiplex_count=16,
        num_multimask_outputs=3,
    )


__all__ = ["MultiplexMaskDecoder", "create_multiplex_mask_decoder"]
