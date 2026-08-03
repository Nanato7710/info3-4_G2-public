from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, f1_score, hamming_loss, precision_score, recall_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.preprocessing.dataset_utils import GENRE_COLS


def load_config() -> dict:
    with (EXPERIMENT_DIR / "config.yaml").open(encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def load_split(name: str) -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "series_split_outputs" / f"{name}_data_grouped.csv"
    return pd.read_csv(path)


def overall_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    valid = y_true.sum(axis=0) > 0
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "samples_f1": f1_score(y_true, y_pred, average="samples", zero_division=0),
        "hamming_loss": hamming_loss(y_true, y_pred),
        "mAP": average_precision_score(y_true[:, valid], y_score[:, valid], average="macro"),
        "predicted_labels_per_item": float(y_pred.sum(axis=1).mean()),
    }


def genre_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> pd.DataFrame:
    y_pred = (y_score >= threshold).astype(int)
    rows = []
    for index, genre in enumerate(GENRE_COLS):
        support = int(y_true[:, index].sum())
        rows.append(
            {
                "genre": genre,
                "support": support,
                "predicted_positive": int(y_pred[:, index].sum()),
                "threshold": threshold,
                "precision": precision_score(y_true[:, index], y_pred[:, index], zero_division=0),
                "recall": recall_score(y_true[:, index], y_pred[:, index], zero_division=0),
                "f1": f1_score(y_true[:, index], y_pred[:, index], zero_division=0),
                "ap": average_precision_score(y_true[:, index], y_score[:, index]) if support else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    config = load_config()
    threshold = float(config.get("threshold", 0.5))
    method = str(config.get("method", "train_prevalence"))
    output_dir = EXPERIMENT_DIR / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_split("training")
    validation_df = load_split("validation")
    prevalence = train_df[GENRE_COLS].mean().to_numpy(dtype=float)
    y_true = validation_df[GENRE_COLS].to_numpy(dtype=int)
    y_score = np.broadcast_to(prevalence, y_true.shape).copy()

    overall = overall_metrics(y_true, y_score, threshold)
    overall.update({"split": "validation", "method": method})
    overall_df = pd.DataFrame([overall])[
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
    overall_df.to_csv(output_dir / "overall_model_metrics.csv", index=False)

    genre_df = genre_metrics(y_true, y_score, threshold)
    genre_df.to_csv(output_dir / "genre_metrics_validation_threshold_0.5.csv", index=False)

    summary = {
        "evaluation_policy": "Validation only. The test split is not used.",
        "primary_metric": "mAP",
        "standard_method": method,
        "description": "Image-free baseline that assigns each genre its training-set prevalence for every item.",
        "outputs": [
            "analysis_summary.json",
            "genre_metrics_validation_threshold_0.5.csv",
            "overall_model_metrics.csv",
        ],
        "model_metrics": overall_df.to_dict(orient="records"),
    }
    with (output_dir / "analysis_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
