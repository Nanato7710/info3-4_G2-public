import torch
from torch import nn

class HuberBCEWithLogitsLoss(nn.Module):
    def __init__(self, delta: float):
        super().__init__()
        self.huber = nn.HuberLoss(delta=delta)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # ロジットを確率(0.0~1.0)に変換してからHuberLossを適用
        probs = torch.sigmoid(logits)
        return self.huber(probs, targets)

def build_criterion(config: dict) -> nn.Module:
    # config.yaml から delta の値だけを取得する（デフォルトは0.15に設定）
    delta = float(config.get("huber_delta", 0.15))
    return HuberBCEWithLogitsLoss(delta=delta)