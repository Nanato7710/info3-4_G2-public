from torch import nn
from torchvision import models
import timm


class ExperimentModel(nn.Module):
    def __init__(self, num_classes: int = 19):
        super().__init__()
        self.backbone = timm.create_model('tresnet_v2_l.miil_in21k', num_classes=0, pretrained=True)
        num_features = self.backbone.num_features
        self.head = nn.Linear(num_features, num_classes, bias=False)

    def forward(self, x):
        features = self.backbone(x)
        output = self.head(features)
        return output
