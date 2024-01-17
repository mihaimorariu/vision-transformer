"""Implements the main vision transformer network."""

from typing import Tuple

import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Linear, ModuleList, Parameter, Sequential, Softmax
from torch.nn.functional import cross_entropy
from torch.optim import Adam

from .block import VisionTransformerBlock
from .helper import get_positional_embedding, patchify


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
        self._pos_embed = Parameter(
            get_positional_embedding(
                self._n_patches**2 + 1,
                self._n_hidden_dim,
            )
        )
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

        h = patchify(
            x,
            self._n_patches,
            self._patch_height,
            self._patch_width,
        )
        h = h.reshape(batch_size, self._n_patches**2, -1)

        h = self._linear(h)
        h = torch.cat((self._cls_embed.repeat(batch_size, 1, 1), h), dim=1)
        h = h + self._pos_embed.repeat(batch_size, 1, 1)

        for block in self._encoder_blocks:
            h = block(h)

        cls_token = h[..., 0, :]
        y = self._mlp(cls_token)

        return y

    def training_step(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        batch: Tuple[Tensor, Tensor],
    ) -> Tensor:
        x, y = batch
        y_hat = self.forward(x)
        loss = cross_entropy(y_hat, y)
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=0.005)
        return optimizer
