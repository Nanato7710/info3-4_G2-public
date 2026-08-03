from torch import nn
from torchvision import models


class ExperimentModel(nn.Module):
    def __init__(self, num_classes: int = 19):
        super().__init__()

        self.backbone = models.efficientnet_b3(
            weights=models.EfficientNet_B3_Weights.DEFAULT
        )

        # EfficientNet-B3 の最後の分類層を置き換える
        num_ftrs = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(num_ftrs, num_classes)

        # まず全層を凍結
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 最後の特徴抽出ブロックだけ学習可能
        for param in self.backbone.features[-1].parameters():
            param.requires_grad = True

        # classifierだけ学習可能
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)