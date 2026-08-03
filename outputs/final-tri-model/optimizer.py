import torch


def build_optimizer(
    model: torch.nn.Module,
    config: dict,
) -> torch.optim.Optimizer:
    if config["optimizer"]["name"] != "AdamW":
        raise ValueError("final-tri-model optimizer must be AdamW")
    return torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]))
