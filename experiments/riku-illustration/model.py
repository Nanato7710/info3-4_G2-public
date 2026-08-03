from torch import nn

from danbooru_resnet import resnet50


class ExperimentModel(nn.Module):
    def __init__(self, num_classes: int = 19):
        super().__init__()

        # Danbooru事前学習済み ResNet50
        self.backbone = resnet50(pretrained=True, top_n=6000)

        # 最終分類器を19クラス用に置き換える
        in_features = self.backbone[1][-1].in_features
        self.backbone[1][-1] = nn.Linear(in_features, num_classes)

    @property
    def body(self) -> nn.Module:
        return self.backbone[0]

    @property
    def layer4(self) -> nn.Module:
        return self.backbone[0][7]

    @property
    def fc(self) -> nn.Module:
        return self.backbone[1][-1]

    def forward(self, x):
        return self.backbone(x)