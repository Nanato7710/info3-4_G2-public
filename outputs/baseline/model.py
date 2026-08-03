from torch import nn
from torchvision import models


class ExperimentModel(nn.Module):
    def __init__(self, num_classes: int = 19):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, inputs):
        return self.backbone(inputs)


def build_model(config: dict) -> nn.Module:
    return ExperimentModel(num_classes=19)
