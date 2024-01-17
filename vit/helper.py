import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor


def get_positional_embedding(
    n_tokens: int,
    n_hidden_dim: int,
) -> Tensor:
    y, x = torch.meshgrid(
        torch.arange(n_tokens),
        torch.arange(n_hidden_dim),
        indexing="ij",
    )
    embedding = torch.cos(y / torch.pow(10000, (x - 1) / n_hidden_dim))
    mask = x % 2 == 0
    embedding[mask] = torch.sin(y / torch.pow(10000, x / n_hidden_dim))[mask]
    return embedding


def patchify(
    batch: Tensor,
    n_patches: int,
    patch_height: int,
    patch_width: int,
    visualize: bool = False,
) -> Tensor:
    if len(batch.shape) != 4:
        raise ValueError("Only 4D tensors are supported as input!")

    batch_size, n_channels, image_height, image_width = batch.shape

    patches = batch.reshape(
        batch_size,
        n_channels,
        n_patches,
        patch_height,
        n_patches,
        patch_width,
    )
    patches = torch.permute(patches, (0, 2, 4, 3, 5, 1))

    if visualize:
        image_np = batch[0].cpu().numpy()
        image_np = np.transpose(image_np, (1, 2, 0))

        plt.imshow(image_np)
        plt.title(f"Image ({image_width} x {image_height})")

        patches_np = patches[0].cpu().numpy()
        fig, axs = plt.subplots(n_patches, n_patches)
        fig.suptitle(f"Patches ({patch_width} x {patch_height})")

        for i in range(n_patches):
            for j in range(n_patches):
                axs[i, j].imshow(patches_np[i][j])

        plt.show()

    return patches
