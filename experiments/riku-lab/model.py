from torch import nn
from torchvision import models


class ExperimentModel(nn.Module):
    """
    ConvNeXt-Tiny + ImageNet事前学習。
    最後のCNBlockとclassifier/headのみ学習する。
    """

    def __init__(self, num_classes: int = 19):
        super().__init__()

        weights = models.ConvNeXt_Tiny_Weights.DEFAULT
        self.backbone = models.convnext_tiny(weights=weights)
        self.feature_dim = self.backbone.classifier[2].in_features

        self.backbone.classifier[2] = nn.Linear(self.feature_dim, num_classes)

        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in self.backbone.features[-1][-1].parameters():
            param.requires_grad = True

        for param in self.backbone.classifier.parameters():
            param.requires_grad = True

    def forward_features(self, x):
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = self.backbone.classifier[0](x)
        x = self.backbone.classifier[1](x)
        return x

    def forward(self, x, return_features: bool = False):
        features = self.forward_features(x)
        logits = self.backbone.classifier[2](features)

        if return_features:
            return logits, features

        return logits
