from torch import nn
from torchvision import models


class ExperimentModel(nn.Module):

    def __init__(self, num_classes: int = 19):
        super().__init__()
        # 事前学習をオンにする
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.backbone(x)