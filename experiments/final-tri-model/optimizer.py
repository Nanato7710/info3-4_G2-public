import torch


def build_optimizer(model: torch.nn.Module, learning_rate: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=learning_rate)
