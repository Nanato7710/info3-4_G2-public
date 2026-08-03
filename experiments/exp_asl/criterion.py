import torch
from torch import nn


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss (ASL) for Multi-Label Classification"""

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [batch_size, num_classes], targets: [batch_size, num_classes]
        probs = torch.sigmoid(logits)

        # Positive loss
        xs_pos = probs
        loss_pos = targets * torch.log(xs_pos.clamp(min=self.eps)) * ((1 - xs_pos) ** self.gamma_pos)

        # Negative loss with asymmetric shifting (clipping)
        xs_neg = 1 - probs
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)
        loss_neg = (
            (1 - targets) * torch.log(xs_neg.clamp(min=self.eps)) * ((1 - xs_neg) ** self.gamma_neg)
        )

        # Combined loss
        loss = loss_pos + loss_neg
        return -loss.mean()


def build_criterion() -> nn.Module:
    # デフォルトのASL推奨パラメータを設定
    return AsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0, clip=0.05)