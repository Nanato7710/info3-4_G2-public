from torch import nn


def build_criterion(df, genre_cols, device) -> nn.Module:
    return nn.BCEWithLogitsLoss()
