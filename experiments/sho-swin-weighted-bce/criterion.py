import torch
import torch.nn as nn
import pandas as pd
from src.preprocessing.dataset_utils import GENRE_COLS

def build_criterion(train_df: pd.DataFrame, device: torch.device) -> nn.Module:
    """
    train_dfに基づいてジャンルごとの正例比率を計算し、
    BCEWithLogitsLossのpos_weightを設定する。
    """
    # 各ジャンルの正例の数を計算
    genre_counts = train_df[GENRE_COLS].sum()
    total_samples = len(train_df)
    
    # pos_weight = 負例数 / 正例数
    # 正例数 = genre_counts
    # 負例数 = total_samples - genre_counts
    pos_counts = genre_counts.values
    neg_counts = total_samples - pos_counts
    
    # ゼロ除算を防ぐために、正例が0個のクラスがある場合は1にしておく
    pos_counts = torch.tensor(pos_counts, dtype=torch.float32)
    pos_counts = torch.clamp(pos_counts, min=1.0)
    
    pos_weight = torch.tensor(neg_counts, dtype=torch.float32, device=device) / pos_counts.to(device)
    
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)