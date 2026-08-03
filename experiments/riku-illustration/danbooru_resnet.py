from pathlib import Path

import torch
import torch.nn as nn
from torch.hub import download_url_to_file
from torchvision import models


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        self.output_size = sz or 1
        self.ap = nn.AdaptiveAvgPool2d(self.output_size)
        self.mp = nn.AdaptiveMaxPool2d(self.output_size)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def bn_drop_lin(n_in: int, n_out: int, bn: bool = True, p: float = 0.0, actn=None):
    layers = [nn.BatchNorm1d(n_in)] if bn else []

    if p != 0:
        layers.append(nn.Dropout(p))

    layers.append(nn.Linear(n_in, n_out))

    if actn is not None:
        layers.append(actn)

    return layers


def create_head(top_n_tags: int, nf: int):
    lin_ftrs = [nf, 512, top_n_tags]

    layers = [
        AdaptiveConcatPool2d(),
        Flatten(),
    ]

    layers += [
        *bn_drop_lin(lin_ftrs[0], lin_ftrs[1], bn=True, p=0.25, actn=nn.ReLU(inplace=True)),
        *bn_drop_lin(lin_ftrs[1], lin_ftrs[2], bn=True, p=0.5),
    ]

    return nn.Sequential(*layers)


def _resnet(base_arch, top_n: int, **kwargs):
    model = base_arch(weights=None, **kwargs)

    body = nn.Sequential(*list(model.children())[:-2])

    if base_arch in [models.resnet18, models.resnet34]:
        num_features_model = 512
    elif base_arch in [models.resnet50, models.resnet101]:
        num_features_model = 2048
    else:
        raise ValueError("Unsupported ResNet architecture")

    head = create_head(top_n_tags=top_n, nf=num_features_model * 2)

    return nn.Sequential(body, head)


def load_state_dict_from_url_cpu(url: str, filename: str, progress: bool = True):
    checkpoint_dir = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    weight_path = checkpoint_dir / filename

    if not weight_path.exists():
        download_url_to_file(url, str(weight_path), progress=progress)

    return torch.load(weight_path, map_location="cpu")


def resnet18(pretrained: bool = True, progress: bool = True, top_n: int = 100, **kwargs):
    model = _resnet(models.resnet18, top_n, **kwargs)

    if pretrained:
        if top_n != 100:
            raise ValueError("resnet18 only supports top_n=100")

        state = load_state_dict_from_url_cpu(
            url="https://github.com/RF5/danbooru-pretrained/releases/download/v0.1/resnet18-3f77756f.pth",
            filename="resnet18-3f77756f.pth",
            progress=progress,
        )

        model.load_state_dict(state)

    return model

def resnet50(pretrained: bool = True, progress: bool = True, top_n: int = 6000, **kwargs):
    model = _resnet(models.resnet50, top_n, **kwargs)

    if pretrained:
        if top_n != 6000:
            raise ValueError("resnet50 only supports top_n=6000")

        state = load_state_dict_from_url_cpu(
            url="https://github.com/RF5/danbooru-pretrained/releases/download/v0.1/resnet50-13306192.pth",
            filename="resnet50-13306192.pth",
            progress=progress,
        )

        model.load_state_dict(state)

    return model