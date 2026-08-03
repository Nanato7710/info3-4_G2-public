from __future__ import annotations

import timm
from torch import nn


class ExperimentModel(nn.Module):
    """Checkpoint-compatible final-tri-model definition."""

    def __init__(self, num_classes: int = 19):
        super().__init__()
        self.backbone = timm.create_model(
            "convnext_base.fb_in22k",
            pretrained=False,
            num_classes=0,
        )
        self.head = nn.Linear(self.backbone.num_features, num_classes, bias=True)

    def forward(self, inputs):
        return self.head(self.backbone(inputs))


def build_model(num_classes: int = 19) -> nn.Module:
    return ExperimentModel(num_classes=num_classes)
