import torch
from torch import nn


class AsymmetricLoss(nn.Module):
    def __init__(
        self,
        *,
        gamma_neg: float,
        gamma_pos: float,
        clip: float,
        eps: float,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        positive_scores = torch.sigmoid(logits)
        negative_scores = 1.0 - positive_scores
        if self.clip > 0:
            negative_scores = (negative_scores - self.clip).clamp(min=self.eps)
        positive_loss = targets * torch.log(positive_scores.clamp(min=self.eps))
        negative_loss = (1 - targets) * torch.log(
            negative_scores.clamp(min=self.eps)
        )
        if self.gamma_pos > 0:
            positive_loss *= (1 - positive_scores) ** self.gamma_pos
        if self.gamma_neg > 0:
            negative_loss *= (1 - negative_scores) ** self.gamma_neg
        return -(positive_loss + negative_loss).mean()


def build_criterion(config: dict) -> nn.Module:
    criterion = config["criterion"]
    if criterion["name"] != "AsymmetricLoss":
        raise ValueError("final-tri-model criterion must be AsymmetricLoss")
    return AsymmetricLoss(
        gamma_pos=float(criterion["gamma_pos"]),
        gamma_neg=float(criterion["gamma_neg"]),
        clip=float(criterion["clip"]),
        eps=float(criterion["eps"]),
    )
