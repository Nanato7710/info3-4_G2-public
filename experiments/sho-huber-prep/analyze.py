from __future__ import annotations

import json
import os
import sys
from pathlib import Path
import argparse

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.append(str(EXPERIMENT_DIR))


from run_exp import build_transform, load_config, resolve_device, resolve_output_dir
from model import ExperimentModel
from src.preprocessing.dataset_utils import GENRE_COLS, load_image

# 引数 --config を受け取る機能を実装
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default=str(EXPERIMENT_DIR / "config.yaml"))
args, _ = parser.parse_known_args()

CONFIG_PATH = Path(args.config).resolve()
config = load_config(CONFIG_PATH)
RUNS_DIR = resolve_output_dir(CONFIG_PATH, str(config["output_dir"]))

# 【超重要】分析結果が上書きされないよう、ANALYSIS_DIR を outputs_dXXX の中に変更
ANALYSIS_DIR = RUNS_DIR / "analysis"


class LocalAnimeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        anime_id = int(row["ID"])
        image = load_image(self.df, anime_id)
        if image is None:
            raise ValueError(f"ID {anime_id} の画像が取得できませんでした。")
        if self.transform is not None:
            image = self.transform(image)
        labels = torch.tensor(row[GENRE_COLS].values.astype("float32"))
        return image, labels, anime_id


def load_split(name: str) -> pd.DataFrame:
    return pd.read_csv(PROJECT_ROOT / "data" / "series_split_outputs" / f"{name}_data_grouped.csv")


def build_validation_transform(image_size: int):
    transform = build_transform(image_size)
    if isinstance(transform, (tuple, list)):
        if len(transform) != 2:
            raise ValueError(
                "build_transform() must return one transform or "
                "a (train_transform, val_transform) pair."
            )
        return transform[1]
    return transform


def predict_split(
    model: torch.nn.Module,
    df: pd.DataFrame,
    device: torch.device,
    image_size: int,
    batch_size: int,
    num_workers: int,
):
    transform = build_validation_transform(image_size)
    loader = DataLoader(
        LocalAnimeDataset(df, transform=transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    all_logits = []
    all_targets = []
    all_ids = []
    model.eval()
    with torch.no_grad():
        for images, targets, anime_ids in loader:
            images = images.to(device)
            logits = model(images)
            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())
            all_ids.extend([int(x) for x in anime_ids])

    logits_tensor = torch.cat(all_logits)
    logits = logits_tensor.numpy()
    targets = torch.cat(all_targets).numpy().astype(int)
    probs = torch.sigmoid(logits_tensor).numpy()
    return all_ids, targets, logits, probs


def overall_metrics(y_true: np.ndarray, y_score: np.ndarray, thresholds) -> dict[str, float]:
    thresholds_arr = np.asarray(thresholds)
    if thresholds_arr.ndim == 0:
        thresholds_arr = np.full(y_true.shape[1], float(thresholds_arr))
    y_pred = (y_score >= thresholds_arr.reshape(1, -1)).astype(int)
    valid = y_true.sum(axis=0) > 0
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "samples_f1": f1_score(y_true, y_pred, average="samples", zero_division=0),
        "hamming_loss": hamming_loss(y_true, y_pred),
        "mAP": average_precision_score(y_true[:, valid], y_score[:, valid], average="macro"),
        "predicted_labels_per_item": float(y_pred.sum(axis=1).mean()),
    }


def genre_metrics(y_true: np.ndarray, y_score: np.ndarray, thresholds) -> pd.DataFrame:
    thresholds_arr = np.asarray(thresholds)
    if thresholds_arr.ndim == 0:
        thresholds_arr = np.full(y_true.shape[1], float(thresholds_arr))
    y_pred = (y_score >= thresholds_arr.reshape(1, -1)).astype(int)

    rows = []
    for idx, genre in enumerate(GENRE_COLS):
        support = int(y_true[:, idx].sum())
        predicted_positive = int(y_pred[:, idx].sum())
        ap = average_precision_score(y_true[:, idx], y_score[:, idx]) if support > 0 else np.nan
        rows.append(
            {
                "genre": genre,
                "support": support,
                "predicted_positive": predicted_positive,
                "threshold": float(thresholds_arr[idx]),
                "precision": precision_score(y_true[:, idx], y_pred[:, idx], zero_division=0),
                "recall": recall_score(y_true[:, idx], y_pred[:, idx], zero_division=0),
                "f1": f1_score(y_true[:, idx], y_pred[:, idx], zero_division=0),
                "ap": ap,
            }
        )
    return pd.DataFrame(rows)


def plot_learning_curves(metrics_path: Path, out_dir: Path) -> None:
    metrics = pd.read_csv(metrics_path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=160)

    axes[0].plot(metrics["epoch"], metrics["train_loss"], label="Train Loss")
    axes[0].plot(metrics["epoch"], metrics["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(metrics["epoch"], metrics["macro_f1"], label="Macro F1")
    axes[1].plot(metrics["epoch"], metrics["samples_f1"], label="Samples F1")
    axes[1].plot(metrics["epoch"], metrics["mAP"], label="mAP")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Validation metrics")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_dir / "learning_curves.png")
    fig.savefig(out_dir / "learning_curves.svg")
    plt.close(fig)


def discover_run_dirs() -> list[Path]:
    seed_dirs = sorted(
        path
        for path in RUNS_DIR.glob("seed_*")
        if path.is_dir() and (path / config["best_model_name"]).exists()
    )
    if seed_dirs:
        return seed_dirs
    if (RUNS_DIR / config["best_model_name"]).exists():
        return [RUNS_DIR]
    return []


def seed_from_run_dir(run_dir: Path) -> int:
    if run_dir.name.startswith("seed_"):
        return int(run_dir.name.removeprefix("seed_"))
    return int(config.get("seed", 0))


def analyze_run(
    run_dir: Path,
    model: torch.nn.Module,
    val_df: pd.DataFrame,
    device: torch.device,
) -> tuple[dict, pd.DataFrame]:
    seed = seed_from_run_dir(run_dir)
    run_analysis_dir = run_dir / "analysis"
    run_analysis_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = run_dir / config["best_model_name"]
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)

    val_ids, y_val, val_logits, val_probs = predict_split(
        model,
        val_df,
        device,
        image_size=int(config["image_size"]),
        batch_size=int(config["batch_size"]),
        num_workers=int(config["num_workers"]),
    )

    np.savez_compressed(
        run_analysis_dir / "model_predictions.npz",
        val_ids=np.array(val_ids),
        y_val=y_val,
        val_logits=val_logits,
        val_probs=val_probs,
    )

    row = overall_metrics(y_val, val_probs, 0.5)
    row["seed"] = seed
    row["split"] = "validation"
    row["method"] = "model_threshold_0.5"
    row["run_dir"] = str(run_dir.relative_to(PROJECT_ROOT))
    overall_df = pd.DataFrame([row])[
        [
            "seed",
            "split",
            "method",
            "macro_f1",
            "samples_f1",
            "hamming_loss",
            "mAP",
            "predicted_labels_per_item",
            "run_dir",
        ]
    ]
    overall_df.to_csv(run_analysis_dir / "overall_model_metrics.csv", index=False)

    genre_df = genre_metrics(y_val, val_probs, 0.5)
    genre_df.insert(0, "seed", seed)
    genre_df.to_csv(run_analysis_dir / "genre_metrics_validation_threshold_0.5.csv", index=False)

    metrics_path = run_dir / config["metrics_name"]
    if metrics_path.exists():
        plot_learning_curves(metrics_path, run_analysis_dir)

    return row, genre_df


def aggregate_overall(seed_overall_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["macro_f1", "samples_f1", "hamming_loss", "mAP", "predicted_labels_per_item"]
    row = {"split": "validation", "method": "model_threshold_0.5"}
    for col in metric_cols:
        row[col] = seed_overall_df[col].mean()
        row[f"{col}_std"] = seed_overall_df[col].std(ddof=1) if len(seed_overall_df) > 1 else 0.0
    return pd.DataFrame([row])


def aggregate_genres(seed_genre_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["support", "predicted_positive", "threshold", "precision", "recall", "f1", "ap"]
    rows = []
    for genre, group in seed_genre_df.groupby("genre", sort=False):
        row = {"genre": genre}
        for col in metric_cols:
            row[col] = group[col].mean()
            if col in {"precision", "recall", "f1", "ap"}:
                row[f"{col}_std"] = group[col].std(ddof=1) if len(group) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def plot_aggregate_learning_curves(run_dirs: list[Path]) -> None:
    frames = []
    for run_dir in run_dirs:
        metrics_path = run_dir / config["metrics_name"]
        if not metrics_path.exists():
            continue
        metrics = pd.read_csv(metrics_path)
        metrics["seed"] = seed_from_run_dir(run_dir)
        frames.append(metrics)
    if not frames:
        return

    metrics_all = pd.concat(frames, ignore_index=True)
    mean_metrics = metrics_all.groupby("epoch", as_index=False)[
        ["train_loss", "val_loss", "macro_f1", "samples_f1", "mAP"]
    ].mean()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=160)
    for _, group in metrics_all.groupby("seed"):
        axes[0].plot(group["epoch"], group["train_loss"], color="tab:blue", alpha=0.18)
        axes[0].plot(group["epoch"], group["val_loss"], color="tab:orange", alpha=0.18)
        axes[1].plot(group["epoch"], group["mAP"], color="tab:green", alpha=0.18)

    axes[0].plot(mean_metrics["epoch"], mean_metrics["train_loss"], label="Train Loss mean", color="tab:blue")
    axes[0].plot(mean_metrics["epoch"], mean_metrics["val_loss"], label="Val Loss mean", color="tab:orange")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(mean_metrics["epoch"], mean_metrics["macro_f1"], label="Macro F1 mean")
    axes[1].plot(mean_metrics["epoch"], mean_metrics["samples_f1"], label="Samples F1 mean")
    axes[1].plot(mean_metrics["epoch"], mean_metrics["mAP"], label="mAP mean")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Validation metrics")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "learning_curves.png")
    fig.savefig(ANALYSIS_DIR / "learning_curves.svg")
    plt.close(fig)


def main() -> None:
    run_dirs = discover_run_dirs()
    if not run_dirs:
        raise FileNotFoundError(f"No checkpoints found under {RUNS_DIR}")
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    device = resolve_device(str(config["device"]))
    val_df = load_split("validation")

    model = ExperimentModel(num_classes=len(GENRE_COLS)).to(device)
    seed_overall_rows = []
    seed_genre_frames = []
    for run_dir in run_dirs:
        row, genre_df = analyze_run(run_dir, model, val_df, device)
        seed_overall_rows.append(row)
        seed_genre_frames.append(genre_df)

    seed_overall_df = pd.DataFrame(seed_overall_rows)
    seed_overall_df.to_csv(ANALYSIS_DIR / "seed_overall_model_metrics.csv", index=False)
    overall_df = aggregate_overall(seed_overall_df)
    overall_df.to_csv(ANALYSIS_DIR / "overall_model_metrics.csv", index=False)

    seed_genre_df = pd.concat(seed_genre_frames, ignore_index=True)
    seed_genre_df.to_csv(ANALYSIS_DIR / "seed_genre_metrics_validation_threshold_0.5.csv", index=False)
    validation_genre_05 = aggregate_genres(seed_genre_df)
    validation_genre_05.to_csv(ANALYSIS_DIR / "genre_metrics_validation_threshold_0.5.csv", index=False)

    plot_aggregate_learning_curves(run_dirs)

    generated_outputs = [
        "analysis_summary.json",
        "genre_metrics_validation_threshold_0.5.csv",
        "learning_curves.png",
        "learning_curves.svg",
        "overall_model_metrics.csv",
        "seed_genre_metrics_validation_threshold_0.5.csv",
        "seed_overall_model_metrics.csv",
    ]
    seed_training_summary_path = RUNS_DIR / "seed_training_summary.csv"
    seed_training_summary = pd.read_csv(seed_training_summary_path).to_dict(orient="records") if seed_training_summary_path.exists() else []
    summary = {
        "device": str(device),
        "inference_config": {
            "image_size": int(config["image_size"]),
            "batch_size": int(config["batch_size"]),
            "num_workers": int(config["num_workers"]),
        },
        "evaluation_policy": "Experiment iteration uses validation only. Do not use the test split until final model selection.",
        "primary_comparison_split": "validation",
        "primary_metric": "mAP",
        "standard_method": "model_threshold_0.5",
        "threshold_policy": "Threshold optimization is intentionally not performed in this experiment-iteration analysis because the primary metric is threshold-independent mAP.",
        "run_dirs": [str(path.relative_to(PROJECT_ROOT)) for path in run_dirs],
        "seed_training_summary": seed_training_summary,
        "outputs": generated_outputs,
        "model_metrics": overall_df.to_dict(orient="records"),
        "seed_model_metrics": seed_overall_df.to_dict(orient="records"),
        "worst_validation_f1_genres_threshold_0.5": validation_genre_05
        .sort_values(["f1", "support"], ascending=[True, True])
        .head(8)
        .to_dict(orient="records"),
        "best_validation_ap_genres_threshold_0.5": validation_genre_05
        .sort_values("ap", ascending=False)
        .head(5)
        .to_dict(orient="records"),
    }
    with open(ANALYSIS_DIR / "analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
