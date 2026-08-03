import torch


def build_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    # config から weight_decay を取得（デフォルト値は 0.05）
    wd = config.get("weight_decay", 0.05)
    lr = float(config.get("learning_rate", 1e-5))
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
