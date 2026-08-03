import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class AnimeResNet(nn.Module):

    def __init__(self, num_classes=19):
        super().__init__()

        # ImageNet学習済みResNet18
        self.resnet = resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

        # 特徴抽出部分を固定
        for param in self.resnet.parameters():
            param.requires_grad = False

        # 分類層だけ置き換え
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(
            num_ftrs,
            num_classes
        )

    def forward(self, x):
        return self.resnet(x)