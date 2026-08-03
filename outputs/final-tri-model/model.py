import timm
from torch import nn


class ExperimentModel(nn.Module):
    def __init__(self, num_classes: int = 19, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            "convnext_base.fb_in22k",
            pretrained=pretrained,
            num_classes=0,
        )
        self.head = nn.Linear(self.backbone.num_features, num_classes, bias=True)

    def forward(self, inputs):
        return self.head(self.backbone(inputs))


def build_model(config: dict) -> nn.Module:
    return ExperimentModel(
        num_classes=19,
        pretrained=bool(config.get("pretrained", True)),
    )
