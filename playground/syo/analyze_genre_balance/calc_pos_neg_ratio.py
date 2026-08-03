import sys
from pathlib import Path

# 現在のファイルから3つ上のディレクトリ(info3-4_G2)をプロジェクトルートとして取得
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# これ以降に元のimport文を書く
import pandas as pd
from src.preprocessing.dataset_utils import GENRE_COLS, load_dataset

# 1. テンプレートと同じ手順でデータセットを読み込む
train_df, _, _ = load_dataset()

# 2. ジャンルごとの合計（正例の数）を計算
genre_counts = train_df[GENRE_COLS].sum()

# 3. 統計量の表示
total_pos = genre_counts.sum()
total_elements = len(train_df) * len(GENRE_COLS)
pos_ratio = total_pos / total_elements

print(f"--- ジャンル分布統計 ---")
print(f"全ジャンルの正例総数: {total_pos}")
print(f"全ラベル数: {total_elements}")
print(f"正例の全体比率 (pos_ratio): {pos_ratio:.4f}")
print(f"\nジャンルごとの内訳:\n{genre_counts}")