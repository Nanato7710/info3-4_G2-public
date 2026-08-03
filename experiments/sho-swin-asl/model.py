import timm
from torch import nn


class ExperimentModel(nn.Module):
    def __init__(self, num_classes: int = 19):
        super().__init__()
        # 転移学習アリ
        self.backbone = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=True,
            num_classes=num_classes
        )

    def forward(self, x):
        return self.backbone(x)
