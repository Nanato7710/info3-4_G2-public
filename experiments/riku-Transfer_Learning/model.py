from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


class ExperimentModel(nn.Module):

    def __init__(self, num_classes=19):
        super().__init__()

        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)

        # 出力層を置き換え
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)

        # ★全層学習OK（optimizerで制御するため）
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)