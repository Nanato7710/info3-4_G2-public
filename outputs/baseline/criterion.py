from torch import nn


def build_criterion(config: dict) -> nn.Module:
    if config["criterion"]["name"] != "BCEWithLogitsLoss":
        raise ValueError("baseline criterion must be BCEWithLogitsLoss")
    return nn.BCEWithLogitsLoss()
