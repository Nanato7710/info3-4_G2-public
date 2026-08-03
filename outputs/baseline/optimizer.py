import torch


def build_optimizer(
    model: torch.nn.Module,
    config: dict,
) -> torch.optim.Optimizer:
    if config["optimizer"]["name"] != "Adam":
        raise ValueError("baseline optimizer must be Adam")
    return torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
