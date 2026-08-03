from torch.optim.lr_scheduler import OneCycleLR
import torch
from torch import nn


def build_scheduler(optimizer: torch.optim.Optimizer, steps_per_epoch: int, epochs: int, max_lr: float, pct_start: float = 0.2) -> torch.optim.lr_scheduler._LRScheduler:
    scheduler = OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch, epochs=epochs, pct_start=pct_start)
    return scheduler