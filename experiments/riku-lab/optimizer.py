import torch

from model import ExperimentModel


def build_optimizer(
    model: ExperimentModel,
    head_lr: float = 3e-4,
    backbone_lr: float = 1e-5,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        [
            {
                "params": model.backbone.features[-1][-1].parameters(),
                "lr": backbone_lr,
            },
            {
                "params": model.backbone.classifier.parameters(),
                "lr": head_lr,
            },
        ],
        weight_decay=weight_decay,
    )
