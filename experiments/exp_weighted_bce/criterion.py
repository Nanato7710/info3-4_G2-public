import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np

def build_criterion(config) -> nn.Module:
    # 1. デバイスの決定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 提供された訓練データのパスを設定
    # (リポジトリのルートからでも、実験フォルダ内からでも動くように自動フォールバック付き)
    csv_rel_path = "data/series_split_outputs/training_data_grouped.csv"
    if os.path.exists(csv_rel_path):
        csv_path = csv_rel_path
    else:
        csv_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../", csv_rel_path)
        )

    # 3. ジャンル列リスト（全19種類）
    genre_cols = [
        "Action",
        "Adventure",
        "Comedy",
        "Drama",
        "Ecchi",
        "Fantasy",
        "Hentai",
        "Horror",
        "Mahou Shoujo",
        "Mecha",
        "Music",
        "Mystery",
        "Psychological",
        "Romance",
        "Sci-Fi",
        "Slice of Life",
        "Sports",
        "Supernatural",
        "Thriller",
    ]

    # 4. データの読み込みと pos_weight の自動計算
    print(f"📦 重み計算のためにデータを読み込んでいます: {csv_path}")
    train_df = pd.read_csv(csv_path)
    labels = train_df[genre_cols].values

    # 陰性数(0の数) / 陽性数(1の数) で各ジャンルの重みを算出
    raw_weight = (1 - labels).sum(axis=0) / (labels.sum(axis=0) + 1e-5)
    pos_weight = np.sqrt(raw_weight)
    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32).to(
        device
    )

    return nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
