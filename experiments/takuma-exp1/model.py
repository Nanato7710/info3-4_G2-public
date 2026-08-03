import torch
import torch.nn as nn
import timm

class ExperimentModel(nn.Module):
    def __init__(self, num_classes=19):
        super().__init__()
        
        self.backbone = timm.create_model(
            'convnext_base.fb_in22k', 
            pretrained=True, 
            num_classes=0  # 元の出力層を削除して特徴量抽出器として使う
        )
      
        in_features = self.backbone.num_features
        
        self.head = nn.Linear(in_features, num_classes, bias=True)
        
    def forward(self, x):
        features = self.backbone(x)
        logits = self.head(features)
        return logits


def build_model(config=None) -> nn.Module:
    """
    学習スクリプトから呼び出されるモデル構築関数
    """
    return ExperimentModel(num_classes=19)