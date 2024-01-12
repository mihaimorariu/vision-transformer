"""Implements the multi-headed self attention module."""

import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Linear, ModuleList, Softmax


class MultiHeadedSelfAttention(LightningModule):
    def __init__(
        self,
        n_hidden_dim: int = 8,
        n_heads: int = 2,
    ) -> None:
        super().__init__()

        if n_hidden_dim % n_heads != 0:
            raise ValueError(
                f"Dimensionality ({n_hidden_dim}) needs to be a multiple "
                f"of the number of heads ({n_heads})!"
            )

        self._n_heads = n_heads
        self._n_hidden_dim_head = n_hidden_dim // n_heads
        self._q_linear = ModuleList(
            [
                Linear(self._n_hidden_dim_head, self._n_hidden_dim_head)
                for _ in range(n_heads)
            ]
        )
        self._k_linear = ModuleList(
            [
                Linear(self._n_hidden_dim_head, self._n_hidden_dim_head)
                for _ in range(n_heads)
            ]
        )
        self._v_linear = ModuleList(
            [
                Linear(self._n_hidden_dim_head, self._n_hidden_dim_head)
                for _ in range(n_heads)
            ]
        )
        self._softmax = Softmax(dim=-1)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        batch_size, n_channels = x.shape[:2]
        x_reshaped = x.reshape(
            batch_size,
            n_channels,
            -1,
            self._n_hidden_dim_head,
        )

        attention = []

        for head in range(self._n_heads):
            x_head = x_reshaped[..., head, :]

            q = self._q_linear[head](x_head)
            k = self._k_linear[head](x_head)
            v = self._v_linear[head](x_head)
            a = self._softmax(
                q @ torch.swapaxes(k, -1, -2) / (self._n_hidden_dim_head**0.5)
            )

            attention.append(a @ v)

        attention = torch.cat(attention, dim=-1)
        return attention
