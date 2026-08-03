import torch

#特徴量は守りつつ出力層は大きく変更できる設計
def build_optimizer(model: torch.nn.Module, learning_rate: float) -> torch.optim.Optimizer:
    return torch.optim.Adam([
        {"params": model.backbone.fc.parameters(), "lr": 1e-3},
        {"params": model.backbone.layer4.parameters(), "lr": 1e-4},
        {"params": list(model.backbone.layer1.parameters())
         
                   + list(model.backbone.layer2.parameters())
                    + list(model.backbone.layer3.parameters()), "lr": 1e-6},
    ])