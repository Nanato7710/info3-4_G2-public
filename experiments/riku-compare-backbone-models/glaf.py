from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURE_DIR = BASE_DIR / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


MODEL_CSVS = {
    "ResNet50": OUTPUT_DIR / "ResNet50_metrics.csv",
    "ResNet101": OUTPUT_DIR / "ResNet101_metrics.csv",
    "EfficientNet-B0": OUTPUT_DIR / "efficientnet-B0_metrics.csv",
    "EfficientNet-B3": OUTPUT_DIR / "efficientnet-B3_metrics.csv",
}


REQUIRED_COLUMNS = {
    "epoch",
    "train_loss",
    "val_loss",
    "mAP",
}


def load_metrics():
    """
    各モデルのCSVを読み込み、
    グラフ化に必要な列だけを数値として整理する。
    """
    metrics = {}

    for model_name, csv_path in MODEL_CSVS.items():
        if not csv_path.exists():
            print(f"[SKIP] CSVが見つかりません: {csv_path}")
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as error:
            print(f"[ERROR] {csv_path}を読み込めませんでした: {error}")
            continue

        missing_columns = REQUIRED_COLUMNS - set(df.columns)

        if missing_columns:
            print(
                f"[SKIP] {model_name}: 必要な列がありません "
                f"{sorted(missing_columns)}"
            )
            continue

        df = df.copy()

        for column in REQUIRED_COLUMNS:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

        # 途中で壊れた行を削除
        df = df.dropna(subset=list(REQUIRED_COLUMNS))

        # epoch順に並べる
        df = df.sort_values("epoch")

        if df.empty:
            print(f"[SKIP] {model_name}: 有効なデータがありません")
            continue

        metrics[model_name] = df

        print(
            f"[LOAD] {model_name}: "
            f"{len(df)} epoch読み込みました"
        )

    return metrics


def plot_all_model_losses(metrics):
    """
    全モデルのTrain LossとValidation Lossを
    1枚のグラフにまとめる。
    """
    plt.figure(figsize=(12, 7))

    for model_name, df in metrics.items():
        train_line = plt.plot(
            df["epoch"],
            df["train_loss"],
            label=f"{model_name} Train",
            linewidth=2,
        )[0]

        plt.plot(
            df["epoch"],
            df["val_loss"],
            label=f"{model_name} Validation",
            linewidth=2,
            linestyle="--",
            color=train_line.get_color(),
        )

    plt.title("Train and Validation Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    save_path = FIGURE_DIR / "loss_all_models.png"

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"[SAVE] {save_path}")


def plot_each_model_loss(metrics):
    """
    モデルごとにTrain LossとValidation Lossを
    1枚ずつ作成する。
    """
    for model_name, df in metrics.items():
        plt.figure(figsize=(9, 6))

        plt.plot(
            df["epoch"],
            df["train_loss"],
            label="Train Loss",
            linewidth=2,
        )

        plt.plot(
            df["epoch"],
            df["val_loss"],
            label="Validation Loss",
            linewidth=2,
            linestyle="--",
        )

        plt.title(f"{model_name} Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()

        filename = (
            model_name
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )

        save_path = FIGURE_DIR / f"loss_{filename}.png"

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        print(f"[SAVE] {save_path}")


def plot_all_model_map(metrics):
    """
    全モデルのmAP推移を比較する。
    """
    plt.figure(figsize=(10, 6))

    for model_name, df in metrics.items():
        plt.plot(
            df["epoch"],
            df["mAP"],
            label=model_name,
            linewidth=2,
        )

    plt.title("mAP Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("mAP")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    save_path = FIGURE_DIR / "map_all_models.png"

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"[SAVE] {save_path}")


def plot_best_map(metrics):
    """
    各モデルの最高mAPを棒グラフで比較する。
    """
    best_maps = {
        model_name: df["mAP"].max()
        for model_name, df in metrics.items()
    }

    if not best_maps:
        print("[SKIP] 最高mAPを比較できるデータがありません")
        return

    model_names = list(best_maps.keys())
    values = list(best_maps.values())

    plt.figure(figsize=(9, 6))

    bars = plt.bar(
        model_names,
        values,
    )

    plt.title("Best mAP Comparison")
    plt.xlabel("Model")
    plt.ylabel("Best mAP")
    plt.grid(
        axis="y",
        alpha=0.3,
    )

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
        )

    plt.xticks(rotation=15)
    plt.tight_layout()

    save_path = FIGURE_DIR / "best_map.png"

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"[SAVE] {save_path}")


def main():
    metrics = load_metrics()

    if not metrics:
        raise RuntimeError(
            "読み込めるCSVがありません。"
            "MODEL_CSVSのファイル名とoutputs内のCSVを確認してください。"
        )

    plot_all_model_losses(metrics)
    plot_each_model_loss(metrics)
    plot_all_model_map(metrics)
    plot_best_map(metrics)

    print("\nグラフ生成が完了しました")
    print(f"保存先: {FIGURE_DIR}")


if __name__ == "__main__":
    main()