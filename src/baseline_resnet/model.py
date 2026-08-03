import torch.nn as nn
from torchvision import models

class AnimeResNet(nn.Module):
    """
    ResNet18をベースにした、19クラスのマルチラベル分類モデル。
    ベースライン評価のため、事前学習済み重みを使用せずスクラッチで学習する。
    """
    def __init__(self, num_classes=19):
        super().__init__()
        
        # 1. ベースライン用モデルのロード
        # 事前学習の知識を使わず（weights=None）、純粋に今回のデータのみで
        # スクラッチ学習を行うため、軽量で収束しやすいResNet18を採用しています。
        self.resnet = models.resnet18(weights=None)
        
        # 2. 最終層（分類層）のカスタマイズ
        # 元のモデルの最終層（fc）が受け取る特徴量のサイズを取得
        num_ftrs = self.resnet.fc.in_features
        
        # 自分のデータセットに合わせて、19クラスを出力する新しい層に置き換え
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        """
        順伝播：モデルにデータを入力した時の処理
        """
        # モデルに入力し、19クラス分の数値（Logits）を出力
        # ※活性化関数(Sigmoid)を通さないことで、学習時の安定性を向上させています
        return self.resnet(x)
