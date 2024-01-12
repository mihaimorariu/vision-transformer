"""Implements the vision transformer encoder block."""

from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import GELU, LayerNorm, Linear, Sequential

from vit.attention import MultiHeadedSelfAttention


class VisionTransformerBlock(LightningModule):
    def __init__(
        self,
        n_hidden_dim: int = 8,
        n_heads: int = 2,
        mlp_ratio: int = 4,
    ) -> None:
        super().__init__()

        self._norm1 = LayerNorm(n_hidden_dim)
        self._mhsa = MultiHeadedSelfAttention(n_hidden_dim, n_heads)
        self._norm2 = LayerNorm(n_hidden_dim)
        self._mlp = Sequential(
            Linear(n_hidden_dim, n_hidden_dim * mlp_ratio),
            GELU(),
            Linear(n_hidden_dim * mlp_ratio, n_hidden_dim),
        )

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        h = x + self._mhsa(self._norm1(x))
        h = h + self._mlp(self._norm2(h))
        return h
