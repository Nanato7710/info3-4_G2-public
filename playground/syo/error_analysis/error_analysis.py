import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
import argparse
from PIL import Image
from sklearn.metrics import average_precision_score

# プロジェクトルートの設定
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.preprocessing.dataset_utils import GENRE_COLS, load_dataset

def show_bad_sample(exp_name, genre=None):
    # パス設定を関数内に移動（または引数で受ける形に整理）
    exp_dir = PROJECT_ROOT / "experiments" / exp_name
    output_dir = exp_dir / "outputs" / "seed_42" / "analysis"
    
    # 1. データの読み込み
    try:
        data = np.load(output_dir / "model_predictions.npz")
    except FileNotFoundError:
        print(f"エラー: 指定されたパスにファイルが見つかりません: {output_dir / 'model_predictions.npz'}")
        return

    y_true = data["y_val"]
    y_pred = data["val_probs"]
    val_ids = data["val_ids"]

    # 検証データの読み込み
    val_df = load_dataset()[1]
    val_df["ID"] = val_df["ID"].astype(val_ids.dtype)

    # 2. 誤差計算
    errors = np.mean(np.abs(y_true - y_pred), axis=1)
    map_score = average_precision_score(y_true, y_pred, average="macro")

    # フィルタリング
    mask = errors > 0.2
    if genre:
        if genre not in GENRE_COLS:
            print(f"エラー: {genre} はジャンルリストに含まれていません。")
            return
        genre_mask = (val_df[genre].values == 1)
        mask = mask & genre_mask

    bad_indices = np.where(mask)[0]
    
    if len(bad_indices) == 0:
        print("条件を満たすサンプルはありませんでした。")
        return

    idx = random.choice(bad_indices)
    target_id = val_ids[idx]
    
    print(f"\n--- サンプル情報 ---")
    print(f"ID: {target_id}")
    print(f"平均誤差: {errors[idx]:.4f}")
    
    # 結果の表示
    print(f"\n--- 予測 vs 正解 (予測確率が高い順) ---")
    results = []
    for i, g in enumerate(GENRE_COLS):
        results.append({
            "genre": g, 
            "pred": y_pred[idx][i], 
            "true": int(y_true[idx][i]), 
            "diff": abs(y_pred[idx][i] - y_true[idx][i])
        })
    
    results.sort(key=lambda x: x["diff"], reverse=True)
    
    for item in results:
        print(f"  {item['genre']:15}: 予測={item['pred']:.3f} | 正解={item['true']} | 誤差={item['diff']:.3f}")

    print(f"\n検証データ全体のmAP: {map_score:.4f}")
    
    # 画像表示
    img_path = PROJECT_ROOT / "data" / "images" / f"{target_id}.jpg"
    if img_path.exists():
        img = Image.open(img_path)
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.title(f"ID: {target_id} (Error: {errors[idx]:.4f})")
        plt.axis("off")
        plt.show()
    else:
        print(f"画像が見つかりません: {img_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="sho-swin-asl", help="実験名 (フォルダ名)")
    parser.add_argument("--genre", type=str, default=None, help="絞り込むジャンル名")
    args = parser.parse_args()
    
    show_bad_sample(args.exp_name, args.genre)