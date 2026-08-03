import torch


def freeze_all(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_fc(model: torch.nn.Module) -> None:
    for param in model.fc.parameters():
        param.requires_grad = True


def unfreeze_layer4_and_fc(model: torch.nn.Module) -> None:
    for param in model.layer4.parameters():
        param.requires_grad = True

    for param in model.fc.parameters():
        param.requires_grad = True


def unfreeze_body_and_fc(model: torch.nn.Module) -> None:
    for param in model.body.parameters():
        param.requires_grad = True

    for param in model.fc.parameters():
        param.requires_grad = True


def build_optimizer_fc_only(
    model: torch.nn.Module,
    learning_rate: float,
) -> torch.optim.Optimizer:
    return torch.optim.Adam(
        model.fc.parameters(),
        lr=learning_rate,
    )


def build_optimizer_layer4_and_fc(
    model: torch.nn.Module,
    fc_lr: float,
    layer4_lr: float,
) -> torch.optim.Optimizer:
    return torch.optim.Adam(
        [
            {"params": model.layer4.parameters(), "lr": layer4_lr},
            {"params": model.fc.parameters(), "lr": fc_lr},
        ]
    )


def build_optimizer_body_and_fc(
    model: torch.nn.Module,
    fc_lr: float,
    body_lr: float,
) -> torch.optim.Optimizer:
    return torch.optim.Adam(
        [
            {"params": model.body.parameters(), "lr": body_lr},
            {"params": model.fc.parameters(), "lr": fc_lr},
        ]
    )


def build_optimizer(
    model: torch.nn.Module,
    training_mode: str,
    learning_rate: float,
    layer4_learning_rate: float = 1e-4,
    body_learning_rate: float = 1e-6,
) -> torch.optim.Optimizer:
    freeze_all(model)

    if training_mode == "fc_only":
        unfreeze_fc(model)
        return build_optimizer_fc_only(
            model=model,
            learning_rate=learning_rate,
        )

    if training_mode == "layer4_and_fc":
        unfreeze_layer4_and_fc(model)
        return build_optimizer_layer4_and_fc(
            model=model,
            fc_lr=learning_rate,
            layer4_lr=layer4_learning_rate,
        )

    if training_mode == "body_and_fc":
        unfreeze_body_and_fc(model)
        return build_optimizer_body_and_fc(
            model=model,
            fc_lr=learning_rate,
            body_lr=body_learning_rate,
        )

    raise ValueError(f"Unknown training_mode: {training_mode}")