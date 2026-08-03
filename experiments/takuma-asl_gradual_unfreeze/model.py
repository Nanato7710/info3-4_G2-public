from torch import nn
from torchvision import models


class ExperimentModel(nn.Module):

    def __init__(self, num_classes: int = 19):
        super().__init__()
        # 1. 事前学習済みモデルを読み込む
        self.backbone = models.resnet101(
            weights=models.ResNet101_Weights.IMAGENET1K_V1
        )

        # 2. 最後の全結合層（fc）を新しい層に置き換える
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)

        # 3. 初期状態は「全固定（Freeze）」からスタート
        #（最初の1歩はヘッドだけを動かすため）
        self.freeze_backbone()

    def freeze_backbone(self):
        """バックボーンの全レイヤーを固定し、最後の fc 層だけを解凍する"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        # fc層だけは常に学習させる
        for param in self.backbone.fc.parameters():
            param.requires_grad = True

    def unfreeze_stage(self, epoch: int):
        """エポック数に応じて、出力に近い層から段階的に解凍していく"""
        # --- ステージ1: 1〜5エポック目 ---
        # fc層のみ学習（初期化時点で設定済みなので何もしない）
        if epoch < 5:
            pass

        # --- ステージ2: 6〜10エポック目 ---
        # 最も出力に近い「layer4」を解凍
        elif 5 <= epoch < 10:
            for param in self.backbone.layer4.parameters():
                param.requires_grad = True

        # --- ステージ3: 11〜15エポック目 ---
        # 次に手前にある「layer3」も追加で解凍
        elif 10 <= epoch < 15:
            for param in self.backbone.layer3.parameters():
                param.requires_grad = True

        # --- ステージ4: 16エポック目以降 ---
        # すべてのレイヤー（layer1, layer2, conv1等含む）を全解凍
        else:
            for param in self.backbone.parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)