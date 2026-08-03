from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from inference import MODEL_ID, MODEL_SEED, load_genre_names, sha256_file

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parents[1]


def optimize_thresholds(
    targets: np.ndarray,
    scores: np.ndarray,
    genre_names: tuple[str, ...],
    *,
    minimum: float = 0.0,
    maximum: float = 1.0,
    step: float = 0.01,
) -> tuple[dict[str, float], dict[str, float], float]:
    if targets.shape != scores.shape or targets.shape[1:] != (len(genre_names),):
        raise ValueError("targetsとscoresは同じ[N, 19] shapeである必要があります")
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise ValueError("scoresは0以上1以下の有限値である必要があります")
    if not np.isin(targets, [0, 1]).all():
        raise ValueError("targetsは0または1である必要があります")
    if not (0.0 <= minimum <= maximum <= 1.0 and step > 0):
        raise ValueError("探索範囲または刻み幅が不正です")

    grid = np.round(
        np.arange(minimum, maximum + step / 2.0, step, dtype=np.float64),
        10,
    )
    selected_thresholds: dict[str, float] = {}
    selected_f1: dict[str, float] = {}
    prediction_columns = []
    for index, genre_name in enumerate(genre_names):
        candidates = [
            (
                float(
                    f1_score(
                        targets[:, index],
                        scores[:, index] >= threshold,
                        zero_division=0,
                    )
                ),
                float(threshold),
            )
            for threshold in grid
        ]
        best_f1, best_threshold = max(
            candidates,
            key=lambda candidate: (candidate[0], candidate[1]),
        )
        selected_thresholds[genre_name] = best_threshold
        selected_f1[genre_name] = best_f1
        prediction_columns.append(scores[:, index] >= best_threshold)

    predictions = np.column_stack(prediction_columns)
    macro_f1 = float(
        f1_score(targets, predictions, average="macro", zero_division=0)
    )
    return selected_thresholds, selected_f1, macro_f1


def build_threshold_document(
    predictions_path: Path,
    checkpoint_path: Path,
    genres_path: Path,
    *,
    minimum: float = 0.0,
    maximum: float = 1.0,
    step: float = 0.01,
) -> dict[str, object]:
    genre_names = load_genre_names(genres_path)
    with np.load(predictions_path, allow_pickle=False) as values:
        required = {"genre_names", "targets", "scores"}
        if not required.issubset(values.files):
            raise ValueError(f"validation予測に必要なkeyがありません: {sorted(required)}")
        prediction_genres = tuple(values["genre_names"].astype(str))
        if prediction_genres != genre_names:
            raise ValueError("validation予測とgenres.jsonのジャンル順が一致しません")
        targets = values["targets"]
        scores = values["scores"]

    thresholds, validation_f1, macro_f1 = optimize_thresholds(
        targets,
        scores,
        genre_names,
        minimum=minimum,
        maximum=maximum,
        step=step,
    )
    return {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "seed": MODEL_SEED,
        "checkpoint_path": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "validation_predictions_path": str(
            predictions_path.relative_to(PROJECT_ROOT)
        ),
        "validation_predictions_sha256": sha256_file(predictions_path),
        "source_split": "validation",
        "objective": "per_genre_binary_f1",
        "search": {
            "minimum": minimum,
            "maximum": maximum,
            "step": step,
            "tie_break": "highest_threshold",
        },
        "genre_names": list(genre_names),
        "thresholds": thresholds,
        "validation_f1": validation_f1,
        "validation_macro_f1": macro_f1,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune per-genre thresholds on seed 44 validation predictions."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/final-tri-model/runs/seed_44/validation_predictions.npz",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "outputs/final-tri-model/runs/seed_44/best_model.pth",
    )
    parser.add_argument("--genres", type=Path, default=APP_DIR / "genres.json")
    parser.add_argument("--output", type=Path, default=APP_DIR / "threshold.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    document = build_threshold_document(
        args.predictions.resolve(),
        args.checkpoint.resolve(),
        args.genres.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".part")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(
        f"wrote {args.output}: validation Macro F1 "
        f"{document['validation_macro_f1']:.6f}"
    )


if __name__ == "__main__":
    main()
