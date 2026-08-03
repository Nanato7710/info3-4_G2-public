import numpy as np
import pandas as pd
import torch
from torch import nn


def build_criterion(train_df: pd.DataFrame, genre_cols: list[str], device: torch.device) -> nn.Module:
    """動的に計算された pos_weight を持つ BCEWithLogitsLoss を構築して返す"""
    pos_weight = calculate_pos_weights(train_df, genre_cols).to(device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# 重み付けをするためのヘルパー関数
def calculate_pos_weights(train_df: pd.DataFrame, genre_cols: list[str]) -> torch.Tensor:
    """データフレームから各ジャンルの不均衡を補正するための pos_weight を計算するヘルパー関数

    式: pos_weight = (負例の数) / (正例の数)
    """
    # 各ジャンルの正例（1）の数をカウント
    pos_counts = train_df[genre_cols].sum().values
    num_samples = len(train_df)

    # 負例（0）の数を計算
    neg_counts = num_samples - pos_counts

    # 正例が0のジャンルがある場合に0除算を防ぐため、最低値を1にする
    pos_weight_np = neg_counts / np.maximum(pos_counts, 1)

    return torch.tensor(pos_weight_np, dtype=torch.float32)