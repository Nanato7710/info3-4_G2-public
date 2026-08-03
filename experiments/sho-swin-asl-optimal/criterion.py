import torch
import torch.nn as nn

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4.0, gamma_pos=0.0, clip=0.05, eps=1e-8):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x: logits, y: targets (0 or 1)
        # 確率に変換
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1.0 - x_sigmoid

        # Asymmetric Clipping (マージン m=0.05)
        # 確率が非常に低い負例 (0.05以下) は loss を完全に 0 にする
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg - self.clip).clamp(min=self.eps)

        # 基本となる Log Loss
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))

        # Asymmetric Focusing (非対称な減衰)
        if self.gamma_pos > 0:
            los_pos = los_pos * (1 - xs_pos) ** self.gamma_pos
        if self.gamma_neg > 0:
            los_neg = los_neg * (1 - xs_neg) ** self.gamma_neg

        # 最終的な損失 (平均)
        loss = -(los_pos + los_neg)
        return loss.mean()

# ★引数に config を追加し、yaml の値を読み込む
def build_criterion(config: dict) -> nn.Module:
    asl_cfg = config.get("asymmetric_loss", {})
    gamma_neg = asl_cfg.get("gamma_neg", 4.0)
    gamma_pos = asl_cfg.get("gamma_pos", 0.0)
    clip = asl_cfg.get("clip", 0.05)
    
    return AsymmetricLoss(gamma_neg=gamma_neg, gamma_pos=gamma_pos, clip=clip)