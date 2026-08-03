import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import importlib.util
import argparse
from contextlib import redirect_stdout

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

# --- 変更点: 引数を2つ（実験ルートとサブディレクトリ）に変更 ---
parser = argparse.ArgumentParser(description="Analyze model residuals.")
parser.add_argument("--exp_root", type=str, required=True, help="Experiments root (e.g., sho-swin-asl-optimal)")
parser.add_argument("--data_subdir", type=str, required=True, help="Path to npz file (e.g., outputs_gn3.0_gp0.5_c0.1/seed_42/analysis)")
args = parser.parse_args()

# グラフのタイトルやファイル名に使うための文字列（従来の REL_PATH の代わり）
REL_PATH = f"{args.exp_root}_{args.data_subdir.replace('/', '_')}"

# 実験ルートディレクトリの指定
EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / args.exp_root

# config のパス構築
dummy_config_path = EXPERIMENT_DIR / "config.yaml"
sys.argv = [sys.argv[0], "--config", str(dummy_config_path)]

# analyze.py の読み込み
analyze_path = EXPERIMENT_DIR / "analyze.py"
spec = importlib.util.spec_from_file_location("analyze", str(analyze_path))
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load analyze module from {analyze_path}")
analyze = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyze)
load_split = analyze.load_split

from src.preprocessing.dataset_utils import GENRE_COLS

# outputsディレクトリと保存先の指定
# 保存先は重複を避けるため、実験ルート＋サブディレクトリ名にします
output_dir = PROJECT_ROOT / "playground/syo/analyze_residuals" / args.exp_root / args.data_subdir.replace("/", "_")
output_dir.mkdir(parents=True, exist_ok=True)

# ターゲットファイルの指定
target_file = EXPERIMENT_DIR / args.data_subdir / "model_predictions.npz"
if not target_file.exists():
    raise FileNotFoundError(f"ファイルが見つかりません: {target_file}")

data = np.load(target_file)
y_true = data["y_val"]
y_pred = data["val_probs"]
val_ids = data["val_ids"]

# データ計算
all_residuals_flat = np.abs(y_true - y_pred).flatten()
residuals_per_sample = np.mean(np.abs(y_true - y_pred), axis=1)

# --- 2. 統計量の表示 (ターミナル出力) ---
p95 = np.percentile(all_residuals_flat, 95)
p99 = np.percentile(all_residuals_flat, 99)
mean_err = np.mean(all_residuals_flat)
median_err = np.median(all_residuals_flat)
std_err = np.std(all_residuals_flat)
skewness = (np.mean((all_residuals_flat - mean_err)**3)) / (std_err**3)

report_path = output_dir / f"report_{REL_PATH}.txt"

# ファイルとコンソールの両方に出力するロジック
def print_and_save(text, file_obj):
    print(text)  # コンソールへ
    file_obj.write(text + "\n")  # ファイルへ

with open(report_path, "w", encoding="utf-8") as f:
    # ファイル書き込み用関数を定義
    def log(text):
        print(text)       # コンソール出力
        f.write(text + "\n")  # ファイル出力

    log(f"--- Experiment: {REL_PATH} ---")
    log(f"95%のサンプルは誤差 {p95:.3f} 以内に収まっています。")
    log(f"99%のサンプルは誤差 {p99:.3f} 以内に収まっています。")
    log("-" * 30)
    log("--- 誤差の詳細統計量 ---")
    log(f"平均値 (Mean):     {mean_err:.4f}")
    log(f"中央値 (Median):   {median_err:.4f}")
    log(f"標準偏差 (Std):    {std_err:.4f}")
    log(f"歪度 (Skewness):   {skewness:.4f}")
    log("-" * 30)

    # --- 4. ソースデータとの突き合わせ分析 ---
    mask = residuals_per_sample > 0.2
    val_df = load_split("validation")
    val_df["ID"] = val_df["ID"].astype(val_ids.dtype)
    bad_samples_df = val_df[val_df["ID"].isin(val_ids[mask])]

    if not bad_samples_df.empty:
        log(f"誤差0.2以上のサンプル数: {np.sum(mask)}")
        log("\n--- 誤差が大きいサンプル(>0.2)のジャンル傾向 ---")
        log(bad_samples_df[GENRE_COLS].sum().sort_values(ascending=False).head(10).to_string())
    else:
        log("該当サンプルがありませんでした。")

print(f"分析レポートを保存しました: {report_path}")

# --- 3. 個別にグラフを保存 ---
# A: 全体誤差ヒストグラム (L字型)
plt.figure(figsize=(10, 6))
plt.hist(all_residuals_flat, bins=100, color='blue', alpha=0.7, edgecolor='black')
plt.title(f"Distribution of All Errors ({REL_PATH})")
plt.savefig(output_dir / f"residuals_distribution_all_{REL_PATH}.png")
plt.close()

# B: 作品単位の平均誤差ヒストグラム (山型)
plt.figure(figsize=(10, 6))
plt.hist(residuals_per_sample, bins=50, color='green', alpha=0.7, edgecolor='black')
plt.title(f"Distribution of Average Error per Sample ({REL_PATH})")
plt.savefig(output_dir / f"residuals_distribution_per_sample_{REL_PATH}.png")
plt.close()

# C: 全体誤差の箱ひげ図
plt.figure(figsize=(10, 4))
plt.boxplot(all_residuals_flat, vert=False, patch_artist=True)
plt.title(f"Box Plot of All Errors ({REL_PATH})")
plt.savefig(output_dir / f"residuals_boxplot_all_{REL_PATH}.png")
plt.close()

# D: 作品単位の平均誤差の箱ひげ図
plt.figure(figsize=(10, 4))
plt.boxplot(residuals_per_sample, vert=False, patch_artist=True, boxprops=dict(facecolor='green'))
plt.title(f"Box Plot of Average Error per Sample ({REL_PATH})")
plt.savefig(output_dir / f"residuals_boxplot_per_sample_{REL_PATH}.png")
plt.close()

print("4つのグラフを個別に保存しました。")
print(f"--- 分析完了 ---")
print(f"レポート保存先: {report_path}")