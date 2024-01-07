from torch import Tensor
from torch.nn import Module


class VisionTransformer(Module):
    def __init__(self):
        super().__init__()

    def forward(self, images: Tensor) -> Tensor:
        return images
