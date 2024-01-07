from typing import Any

from pytorch_lightning import LightningModule


class VisionTransformer(LightningModule):
    def __init__(self):
        super().__init__()

    def training_step(self, *args: Any, **kwargs: Any) -> None:
        pass

    def configure_optimizers(self):
        pass
