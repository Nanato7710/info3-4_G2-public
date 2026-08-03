from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "playground" / "kazusa" / "baseline" / "analysis"
IMAGE_DIR = ROOT / "data" / "images"
CHECKPOINT_PATH = ROOT / "src" / "baseline_resnet" / "model" / "resnet18_best.pth"
METRICS_PATH = ROOT / "src" / "baseline_resnet" / "model" / "baseline_full_metrics.csv"

sys.path.append(str(ROOT))

from src.baseline_resnet.model import AnimeResNet  # noqa: E402
from src.preprocessing.dataset_utils import GENRE_COLS  # noqa: E402


class LocalAnimeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        anime_id = int(row["ID"])
        image_path = IMAGE_DIR / f"{anime_id}.jpg"
        if not image_path.exists():
            raise FileNotFoundError(f"Missing cached image: {image_path}")
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        labels = torch.tensor(row[GENRE_COLS].values.astype("float32"))
        return image, labels, anime_id


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_split(name: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "series_split_outputs" / f"{name}_data_grouped.csv")


def predict_split(model: torch.nn.Module, df: pd.DataFrame, device: torch.device):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    loader = DataLoader(
        LocalAnimeDataset(df, transform=transform),
        batch_size=64,
        shuffle=False,
        num_workers=0,
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

    logits = torch.cat(all_logits).numpy()
    targets = torch.cat(all_targets).numpy().astype(int)
    probs = 1.0 / (1.0 + np.exp(-logits))
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


def optimize_thresholds(y_true: np.ndarray, y_score: np.ndarray) -> pd.DataFrame:
    grid = np.linspace(0.05, 0.95, 91)
    rows = []
    for idx, genre in enumerate(GENRE_COLS):
        best_threshold = 0.5
        best_f1 = -1.0
        best_precision = 0.0
        best_recall = 0.0
        for threshold in grid:
            y_pred = (y_score[:, idx] >= threshold).astype(int)
            score = f1_score(y_true[:, idx], y_pred, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = float(threshold)
                best_precision = precision_score(y_true[:, idx], y_pred, zero_division=0)
                best_recall = recall_score(y_true[:, idx], y_pred, zero_division=0)
        rows.append(
            {
                "genre": genre,
                "support": int(y_true[:, idx].sum()),
                "threshold": best_threshold,
                "validation_f1": best_f1,
                "validation_precision": best_precision,
                "validation_recall": best_recall,
            }
        )
    return pd.DataFrame(rows)


def fixed_top_k_scores(train_df: pd.DataFrame, n_rows: int, k: int) -> np.ndarray:
    prevalence = train_df[GENRE_COLS].mean().to_numpy(dtype=float)
    order = np.argsort(-prevalence)
    scores = np.zeros((n_rows, len(GENRE_COLS)), dtype=float)
    scores[:, order[:k]] = 1.0
    return scores


def bernoulli_prevalence_baseline(
    train_df: pd.DataFrame,
    y_true: np.ndarray,
    repeats: int = 100,
    seed: int = 42,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    prevalence = train_df[GENRE_COLS].mean().to_numpy(dtype=float)
    rows = []
    for _ in range(repeats):
        scores = rng.binomial(1, prevalence.reshape(1, -1), size=y_true.shape).astype(float)
        rows.append(overall_metrics(y_true, scores, 0.5))
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0].keys()
    }


def build_simple_baselines(train_df: pd.DataFrame, y_true: np.ndarray) -> pd.DataFrame:
    rows = []

    zero_scores = np.zeros_like(y_true, dtype=float)
    row = overall_metrics(y_true, zero_scores, 0.5)
    row["method"] = "always_none"
    rows.append(row)

    for k in [2, 3]:
        scores = fixed_top_k_scores(train_df, len(y_true), k)
        row = overall_metrics(y_true, scores, 0.5)
        row["method"] = f"always_top_{k}_train_genres"
        rows.append(row)

    row = bernoulli_prevalence_baseline(train_df, y_true)
    row["method"] = "bernoulli_by_train_prevalence_mean_100"
    rows.append(row)

    return pd.DataFrame(rows)[
        [
            "method",
            "macro_f1",
            "samples_f1",
            "hamming_loss",
            "mAP",
            "predicted_labels_per_item",
        ]
    ]


def plot_learning_curves() -> None:
    metrics = pd.read_csv(METRICS_PATH)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=160)

    axes[0].plot(metrics["Epoch"], metrics["Train_Loss"], label="Train Loss")
    axes[0].plot(metrics["Epoch"], metrics["Val_Loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(metrics["Epoch"], metrics["Macro_F1"], label="Macro F1")
    axes[1].plot(metrics["Epoch"], metrics["Samples_F1"], label="Samples F1")
    axes[1].plot(metrics["Epoch"], metrics["mAP"], label="mAP")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Validation metrics")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "learning_curves.png")
    fig.savefig(OUT_DIR / "learning_curves.svg")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = select_device()
    train_df = load_split("training")
    val_df = load_split("validation")
    test_df = load_split("test")

    model = AnimeResNet(num_classes=len(GENRE_COLS)).to(device)
    state = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state)

    val_ids, y_val, val_logits, val_probs = predict_split(model, val_df, device)
    test_ids, y_test, test_logits, test_probs = predict_split(model, test_df, device)

    np.savez_compressed(
        OUT_DIR / "model_predictions.npz",
        val_ids=np.array(val_ids),
        y_val=y_val,
        val_logits=val_logits,
        val_probs=val_probs,
        test_ids=np.array(test_ids),
        y_test=y_test,
        test_logits=test_logits,
        test_probs=test_probs,
    )

    thresholds_df = optimize_thresholds(y_val, val_probs)
    thresholds = thresholds_df["threshold"].to_numpy()
    thresholds_df.to_csv(OUT_DIR / "optimized_thresholds_by_genre.csv", index=False)

    overall_rows = []
    for split, y_true, probs in [
        ("validation", y_val, val_probs),
        ("test", y_test, test_probs),
    ]:
        row = overall_metrics(y_true, probs, 0.5)
        row["split"] = split
        row["method"] = "model_threshold_0.5"
        overall_rows.append(row)

        row = overall_metrics(y_true, probs, thresholds)
        row["split"] = split
        row["method"] = "model_validation_optimized_thresholds"
        overall_rows.append(row)

    overall_df = pd.DataFrame(overall_rows)[
        [
            "split",
            "method",
            "macro_f1",
            "samples_f1",
            "hamming_loss",
            "mAP",
            "predicted_labels_per_item",
        ]
    ]
    overall_df.to_csv(OUT_DIR / "overall_model_metrics.csv", index=False)

    genre_metrics(y_val, val_probs, 0.5).to_csv(OUT_DIR / "genre_metrics_validation_threshold_0.5.csv", index=False)
    genre_metrics(y_test, test_probs, 0.5).to_csv(OUT_DIR / "genre_metrics_test_threshold_0.5.csv", index=False)
    genre_metrics(y_val, val_probs, thresholds).to_csv(
        OUT_DIR / "genre_metrics_validation_optimized_thresholds.csv",
        index=False,
    )
    genre_metrics(y_test, test_probs, thresholds).to_csv(
        OUT_DIR / "genre_metrics_test_validation_optimized_thresholds.csv",
        index=False,
    )

    baselines_df = build_simple_baselines(train_df, y_test)
    model_test_row = overall_df[
        (overall_df["split"] == "test") & (overall_df["method"] == "model_threshold_0.5")
    ].drop(columns=["split"])
    comparison_df = pd.concat([model_test_row, baselines_df], ignore_index=True)
    comparison_df.to_csv(OUT_DIR / "simple_baseline_comparison_test.csv", index=False)

    plot_learning_curves()

    metrics_history = pd.read_csv(METRICS_PATH)
    summary = {
        "device": str(device),
        "checkpoint_path": str(CHECKPOINT_PATH.relative_to(ROOT)),
        "best_logged_val_loss_epoch": int(metrics_history.loc[metrics_history["Val_Loss"].idxmin(), "Epoch"]),
        "outputs": sorted(path.name for path in OUT_DIR.iterdir()),
        "model_metrics": overall_df.to_dict(orient="records"),
        "simple_baselines_test": comparison_df.to_dict(orient="records"),
        "worst_test_f1_genres_threshold_0.5": genre_metrics(y_test, test_probs, 0.5)
        .sort_values(["f1", "support"], ascending=[True, True])
        .head(8)
        .to_dict(orient="records"),
        "best_test_ap_genres_threshold_0.5": genre_metrics(y_test, test_probs, 0.5)
        .sort_values("ap", ascending=False)
        .head(5)
        .to_dict(orient="records"),
    }
    with open(OUT_DIR / "analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
