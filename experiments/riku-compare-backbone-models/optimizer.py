import torch


def build_optimizer(
    model: torch.nn.Module,
    classifier_learning_rate: float,
    feature_learning_rate: float,
) -> torch.optim.Optimizer:
    
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.features[-1].parameters(), "lr": feature_learning_rate},
        {"params": model.backbone.classifier.parameters(), "lr": classifier_learning_rate},
    ], weight_decay=1e-4)

    return optimizer