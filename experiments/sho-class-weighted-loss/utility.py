from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve

def plot_learning_curve(history: list[dict], output_dir: Path) -> None:
    """学習履歴（history）からLossとMetricsの2面グラフを生成し、指定ディレクトリに保存する"""
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    macro_f1 = [row["macro_f1"] for row in history]
    samples_f1 = [row["samples_f1"] for row in history]
    map_score = [row["mAP"] for row in history]

    # サブプロットの作成 (横に2つ並べる)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左側：Lossのプロット
    axes[0].plot(epochs, train_loss, label="Train Loss", color="tab:blue")
    axes[0].plot(epochs, val_loss, label="Val Loss", color="tab:orange")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="-", alpha=0.3)
    axes[0].legend()

    # 右側：Metricsのプロット
    axes[1].plot(epochs, macro_f1, label="Macro F1", color="tab:blue")
    axes[1].plot(epochs, samples_f1, label="Samples F1", color="tab:orange")
    axes[1].plot(epochs, map_score, label="mAP", color="tab:green")
    axes[1].set_title("Validation metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].grid(True, linestyle="-", alpha=0.3)
    axes[1].legend()

    plt.tight_layout()

    # 画像として保存
    fig.savefig(output_dir / "learning_curve.png", dpi=150)
    plt.close(fig)

def plot_pr_curve(probs: np.ndarray, targets: np.ndarray, output_dir: Path) -> None:
    """確率と正解ラベルからマイクロ平均のPR曲線を生成し保存する"""
    precision, recall, _ = precision_recall_curve(targets.ravel(), probs.ravel())

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color="tab:purple", lw=2, label="Micro-average PR curve")
    
    ax.set_title("Precision-Recall Curve (Best Model)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, linestyle="-", alpha=0.3)
    ax.legend(loc="lower left")

    plt.tight_layout()
    fig.savefig(output_dir / "pr_curve_best.png", dpi=150)
    plt.close(fig)