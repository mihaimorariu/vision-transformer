from typing import Any

import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Linear, ModuleList, Parameter, Sequential, Softmax
from torch.optim import Adam

from .block import VisionTransformerBlock


class VisionTransformer(LightningModule):
    def __init__(
        self,
        n_patches: int = 7,
        n_encoder_blocks: int = 2,
        n_mhsa_heads: int = 2,
        n_channels: int = 1,
        n_hidden_dim: int = 8,
        image_height: int = 28,
        image_width: int = 28,
        n_classes: int = 10,
    ) -> None:
        super().__init__()

        if image_height % n_patches != 0 or image_width % n_patches != 0:
            raise ValueError(
                f"Image height ({image_height}) and width "
                f"({image_width}) must be divisible by {n_patches}!"
            )

        self._n_patches = n_patches
        self._n_hidden_dim = n_hidden_dim
        self._patch_height = image_height // n_patches
        self._patch_width = image_width // n_patches

        self._linear = Linear(
            n_channels * self._patch_height * self._patch_width,
            n_hidden_dim,
        )

        self._cls_embed = Parameter(torch.rand(1, n_hidden_dim))
        self._pos_embed = Parameter(self._get_positional_embedding())
        self._pos_embed.requires_grad = False

        self._encoder_blocks = ModuleList(
            [
                VisionTransformerBlock(n_hidden_dim, n_mhsa_heads)
                for _ in range(n_encoder_blocks)
            ]
        )

        self._mlp = Sequential(
            Linear(n_hidden_dim, n_classes),
            Softmax(dim=-1),
        )

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        batch_size = x.shape[0]

        h = self._patchify(x)
        h = self._linear(h)
        h = torch.cat((self._cls_embed.repeat(batch_size, 1, 1), h), dim=1)
        h = h + self._pos_embed.repeat(batch_size, 1, 1)

        for block in self._encoder_blocks:
            h = block(h)

        cls_token = h[..., 0, :]
        y = self._mlp(cls_token)

        return y

    def _get_positional_embedding(self):
        n_tokens = self._n_patches**2 + 1

        y, x = torch.meshgrid(
            torch.arange(n_tokens),
            torch.arange(self._n_hidden_dim),
            indexing="ij",
        )
        embedding = torch.cos(
            y / torch.pow(10000, (x - 1) / self._n_hidden_dim))
        mask = x % 2 == 0
        embedding[mask] = torch.sin(
            y / torch.pow(10000, x / self._n_hidden_dim))[mask]
        return embedding

    def _patchify(self, x: Tensor) -> Tensor:
        assert len(x.shape) == 4
        batch_size, n_channels = x.shape[:2]

        patches = x.reshape(
            batch_size,
            n_channels,
            self._patch_height,
            self._n_patches,
            self._patch_width,
            self._n_patches,
        )
        patches = torch.permute(patches, (0, 3, 5, 2, 4, 1))
        patches = patches.reshape(batch_size, self._n_patches**2, -1)

        return patches

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer
