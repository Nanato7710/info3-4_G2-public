from torch import nn


def build_criterion() -> nn.Module:
    return nn.BCEWithLogitsLoss()
