from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUTS_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = OUTPUTS_ROOT.parent
if str(OUTPUTS_ROOT) not in sys.path:
    sys.path.insert(0, str(OUTPUTS_ROOT))

from artifact_common import (
    GENRE_NAMES,
    calculate_genre_ap,
    calculate_metrics,
    load_predictions,
    sha256_file,
    utc_now,
    write_json,
)

MODELS = ("baseline", "final-tri-model")
SEEDS = (42, 43, 44)
METRICS = ("mAP", "macro_f1", "samples_f1", "hamming_loss")
RESULTS_DIR = OUTPUTS_ROOT / "evaluation" / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate frozen predictions.")
    parser.add_argument(
        "--split",
        choices=("validation", "test", "all"),
        default="all",
    )
    return parser.parse_args()


def input_paths(split: str) -> list[Path]:
    return [
        OUTPUTS_ROOT
        / model_id
        / "runs"
        / f"seed_{seed}"
        / f"{split}_predictions.npz"
        for model_id in MODELS
        for seed in SEEDS
    ]


def verify_alignment(predictions: list[tuple[str, int, dict]]) -> None:
    first = predictions[0][2]
    for model_id, seed, arrays in predictions[1:]:
        for key in ("ids", "series_groups", "genre_names", "targets"):
            if not np.array_equal(first[key], arrays[key]):
                raise ValueError(f"{key} differs for {model_id} seed {seed}")


def aggregate_split(split: str) -> None:
    paths = input_paths(split)
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing {split} predictions: {missing}")
    loaded: list[tuple[str, int, dict]] = []
    per_seed_rows = []
    genre_rows = []
    for model_id in MODELS:
        for seed in SEEDS:
            path = (
                OUTPUTS_ROOT
                / model_id
                / "runs"
                / f"seed_{seed}"
                / f"{split}_predictions.npz"
            )
            arrays = load_predictions(path)
            loaded.append((model_id, seed, arrays))
            metrics = calculate_metrics(arrays["targets"], arrays["scores"])
            per_seed_rows.append(
                {
                    "split": split,
                    "model_id": model_id,
                    "seed": seed,
                    **metrics,
                    "prediction_path": path.relative_to(PROJECT_ROOT).as_posix(),
                    "prediction_sha256": sha256_file(path),
                }
            )
            for genre in calculate_genre_ap(arrays["targets"], arrays["scores"]):
                genre_rows.append(
                    {
                        "split": split,
                        "model_id": model_id,
                        "seed": seed,
                        **genre,
                    }
                )
    verify_alignment(loaded)
    per_seed = pd.DataFrame(per_seed_rows)
    per_seed.to_csv(RESULTS_DIR / f"{split}-per-seed.csv", index=False)
    summary_models = {}
    for model_id in MODELS:
        subset = per_seed.loc[per_seed["model_id"] == model_id]
        summary_models[model_id] = {}
        for metric in METRICS:
            values = subset[metric].to_numpy(dtype=float)
            summary_models[model_id][metric] = {
                "mean": float(values.mean()),
                "sample_standard_deviation": float(values.std(ddof=1)),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
            }
    write_json(
        RESULTS_DIR / f"{split}-summary.json",
        {
            "created_at": utc_now(),
            "split": split,
            "threshold": 0.5,
            "seed_summary_formula": {
                "mean": "sum(x_i) / 3",
                "sample_standard_deviation": (
                    "sqrt(sum((x_i - mean)^2) / (3 - 1))"
                ),
                "minimum": "min(x_i)",
                "maximum": "max(x_i)",
            },
            "metric_formula": {
                "mAP": "macro mean of per-genre average precision",
                "macro_f1": "macro F1 at score >= 0.5",
                "samples_f1": "sample-averaged F1 at score >= 0.5",
                "hamming_loss": "fraction of labels misclassified at score >= 0.5",
            },
            "inputs": [
                {
                    "path": path.relative_to(PROJECT_ROOT).as_posix(),
                    "sha256": sha256_file(path),
                }
                for path in paths
            ],
            "models": summary_models,
        },
    )

    new_genre = pd.DataFrame(genre_rows)
    genre_per_seed_path = RESULTS_DIR / "genre-ap-per-seed.csv"
    if genre_per_seed_path.exists():
        existing = pd.read_csv(genre_per_seed_path)
        existing = existing.loc[existing["split"] != split]
        new_genre = pd.concat([existing, new_genre], ignore_index=True)
    new_genre.to_csv(genre_per_seed_path, index=False)
    genre_summary = (
        new_genre.groupby(["split", "model_id", "genre"], sort=False)
        .agg(
            positive_count=("positive_count", "first"),
            mean_average_precision=("average_precision", "mean"),
            sample_standard_deviation=("average_precision", lambda x: x.std(ddof=1)),
            minimum=("average_precision", "min"),
            maximum=("average_precision", "max"),
        )
        .reset_index()
    )
    genre_summary.to_csv(RESULTS_DIR / "genre-ap-summary.csv", index=False)

    differences = []
    for seed in SEEDS:
        baseline = per_seed.loc[
            (per_seed["model_id"] == "baseline") & (per_seed["seed"] == seed)
        ].iloc[0]
        final = per_seed.loc[
            (per_seed["model_id"] == "final-tri-model")
            & (per_seed["seed"] == seed)
        ].iloc[0]
        differences.append(
            {
                "split": split,
                "row_type": "seed",
                "seed": seed,
                **{
                    metric: float(final[metric] - baseline[metric])
                    for metric in METRICS
                },
            }
        )
    seed_differences = pd.DataFrame(differences)
    differences.append(
        {
            "split": split,
            "row_type": "mean",
            "seed": "",
            **{
                metric: float(seed_differences[metric].mean()) for metric in METRICS
            },
        }
    )
    difference_frame = pd.DataFrame(differences)
    difference_path = RESULTS_DIR / "model-differences.csv"
    if difference_path.exists():
        existing = pd.read_csv(difference_path)
        existing = existing.loc[existing["split"] != split]
        difference_frame = pd.concat([existing, difference_frame], ignore_index=True)
    difference_frame.to_csv(difference_path, index=False)
    print(f"aggregated {split}: {RESULTS_DIR}")


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    splits = ("validation", "test") if args.split == "all" else (args.split,)
    for split in splits:
        aggregate_split(split)


if __name__ == "__main__":
    main()
