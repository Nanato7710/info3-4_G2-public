from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

BOOTSTRAP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BOOTSTRAP_DIR.parents[1]
MODELS = ("baseline", "final-tri-model")
SEEDS = (42, 43, 44)
GENRE_NAMES = (
    "Action",
    "Adventure",
    "Comedy",
    "Drama",
    "Ecchi",
    "Fantasy",
    "Hentai",
    "Horror",
    "Mahou Shoujo",
    "Mecha",
    "Music",
    "Mystery",
    "Psychological",
    "Romance",
    "Sci-Fi",
    "Slice of Life",
    "Sports",
    "Supernatural",
    "Thriller",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Independently verify selected Bootstrap rows and optional repeat output."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--repeat-dir", type=Path)
    parser.add_argument(
        "--replicate-ids",
        type=int,
        nargs="+",
        default=(1, 137, 10000),
    )
    parser.add_argument("--atol", type=float, default=1e-12)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_project_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def average_precision_binary(targets: np.ndarray, scores: np.ndarray) -> float:
    """Compute non-interpolated AP independently, grouping tied thresholds."""
    order = np.argsort(-scores, kind="mergesort")
    sorted_targets = targets[order].astype(np.float64)
    positives = float(sorted_targets.sum())
    if positives == 0:
        raise ValueError("average precision is undefined without positives")
    sorted_scores = scores[order]
    threshold_ends = np.r_[
        np.flatnonzero(np.diff(sorted_scores) != 0),
        len(sorted_scores) - 1,
    ]
    true_positives = np.cumsum(sorted_targets)[threshold_ends]
    predicted_positives = threshold_ends + 1
    precision = true_positives / predicted_positives
    recall = true_positives / positives
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def manual_map(targets: np.ndarray, scores: np.ndarray) -> float:
    return float(
        np.mean(
            [
                average_precision_binary(targets[:, index], scores[:, index])
                for index in range(targets.shape[1])
            ]
        )
    )


def load_inputs(config: dict[str, Any]) -> dict[str, dict[int, dict[str, np.ndarray]]]:
    result: dict[str, dict[int, dict[str, np.ndarray]]] = {
        model: {} for model in MODELS
    }
    reference_ids: list[Any] | None = None
    for model in MODELS:
        for seed in SEEDS:
            path = resolve_project_path(config["inputs"][model][f"seed_{seed}"])
            with np.load(path, allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            ids = arrays["ids"].tolist()
            if len(set(ids)) != len(ids):
                raise ValueError(f"duplicate IDs in {path}")
            if reference_ids is None:
                reference_ids = ids
            elif ids != reference_ids:
                if set(ids) != set(reference_ids):
                    raise ValueError(f"ID set differs in {path}")
                by_id = {value: index for index, value in enumerate(ids)}
                order = np.asarray([by_id[value] for value in reference_ids])
                arrays = {
                    key: (value if key == "genre_names" else value[order])
                    for key, value in arrays.items()
                }
            result[model][seed] = arrays
    return result


def group_rows(series_groups: np.ndarray) -> list[np.ndarray]:
    return [
        np.flatnonzero(series_groups == group)
        for group in np.unique(series_groups)
    ]


def rows_for_draw(draw: np.ndarray, rows_by_group: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([rows_by_group[int(index)] for index in draw])


def assert_close(actual: float, expected: float, atol: float, label: str) -> None:
    if not np.isclose(actual, expected, atol=atol, rtol=0):
        raise AssertionError(f"{label}: actual={actual}, expected={expected}")


def verify_selected_replicates(
    config: dict[str, Any],
    result_dir: Path,
    replicate_ids: list[int],
    atol: float,
) -> list[dict[str, Any]]:
    frame = pd.read_csv(result_dir / "replicates.csv")
    selected = frame.set_index("replicate_id").loc[replicate_ids]
    inputs = load_inputs(config)
    reference = inputs["baseline"][42]
    targets = reference["targets"]
    rows_by_group = group_rows(reference["series_groups"])
    rng = np.random.default_rng(int(config["random_seed"]))
    by_attempt = {
        int(row["attempt_id"]): (int(replicate_id), row)
        for replicate_id, row in selected.iterrows()
    }
    last_attempt = max(by_attempt)
    verified = []

    for attempt_id in range(1, last_attempt + 1):
        draw = rng.integers(
            0,
            len(rows_by_group),
            size=len(rows_by_group),
            dtype=np.int64,
        )
        indices = rows_for_draw(draw, rows_by_group)
        if np.any(targets[indices].sum(axis=0) == 0):
            if attempt_id in by_attempt:
                raise AssertionError(
                    f"stored valid replicate points to invalid attempt {attempt_id}"
                )
            continue
        if attempt_id not in by_attempt:
            continue

        replicate_id, stored = by_attempt[attempt_id]
        maps: dict[str, dict[int, float]] = {model: {} for model in MODELS}
        for model in MODELS:
            for seed in SEEDS:
                value = manual_map(
                    targets[indices],
                    inputs[model][seed]["scores"][indices],
                )
                maps[model][seed] = value
                prefix = "final" if model == "final-tri-model" else "baseline"
                assert_close(
                    value,
                    float(stored[f"{prefix}_seed_{seed}_map"]),
                    atol,
                    f"replicate {replicate_id} {model} seed {seed}",
                )
        baseline_mean = float(np.mean(list(maps["baseline"].values())))
        final_mean = float(np.mean(list(maps["final-tri-model"].values())))
        assert_close(
            baseline_mean,
            float(stored["baseline_mean_map"]),
            atol,
            f"replicate {replicate_id} baseline mean",
        )
        assert_close(
            final_mean,
            float(stored["final_mean_map"]),
            atol,
            f"replicate {replicate_id} final mean",
        )
        assert_close(
            final_mean - baseline_mean,
            float(stored["delta_map"]),
            atol,
            f"replicate {replicate_id} delta",
        )
        if int(stored["n_rows"]) != len(indices):
            raise AssertionError(f"replicate {replicate_id} n_rows differs")
        if int(stored["n_unique_groups"]) != len(np.unique(draw)):
            raise AssertionError(f"replicate {replicate_id} group count differs")
        verified.append(
            {
                "replicate_id": replicate_id,
                "attempt_id": attempt_id,
                "delta_map": final_mean - baseline_mean,
            }
        )
    if len(verified) != len(replicate_ids):
        raise AssertionError("not all requested replicates were verified")
    return verified


def verify_summary(result_dir: Path) -> dict[str, Any]:
    summary_path = result_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    replicates_path = result_dir / "replicates.csv"
    actual_hash = sha256_file(replicates_path)
    expected_hash = summary["outputs"]["replicates_csv"]["sha256"]
    if actual_hash != expected_hash:
        raise AssertionError("replicates.csv SHA-256 differs from summary.json")
    frame = pd.read_csv(replicates_path)
    deltas = frame["delta_map"].to_numpy(dtype=float)
    alpha = 1.0 - float(summary["analysis"]["confidence_level"])
    lower, upper = np.quantile(
        deltas,
        [alpha / 2.0, 1.0 - alpha / 2.0],
        method="linear",
    )
    assert_close(
        float(lower),
        float(summary["analysis"]["confidence_interval"]["lower"]),
        1e-15,
        "confidence interval lower",
    )
    assert_close(
        float(upper),
        float(summary["analysis"]["confidence_interval"]["upper"]),
        1e-15,
        "confidence interval upper",
    )
    if len(frame) != int(summary["sampling"]["valid_replicates"]):
        raise AssertionError("valid replicate count differs")
    return {
        "replicates_sha256": actual_hash,
        "valid_replicates": len(frame),
        "confidence_interval": [float(lower), float(upper)],
    }


def verify_repeat(result_dir: Path, repeat_dir: Path) -> dict[str, Any]:
    result_csv = result_dir / "replicates.csv"
    repeat_csv = repeat_dir / "replicates.csv"
    if result_csv.read_bytes() != repeat_csv.read_bytes():
        raise AssertionError("repeat replicates.csv is not byte-identical")
    result_summary = json.loads((result_dir / "summary.json").read_text())
    repeat_summary = json.loads((repeat_dir / "summary.json").read_text())
    for summary in (result_summary, repeat_summary):
        summary.pop("execution", None)
        for output in summary["outputs"].values():
            output.pop("path", None)
    if result_summary != repeat_summary:
        raise AssertionError(
            "repeat summary computational content is not identical"
        )
    return {
        "replicates_byte_identical": True,
        "summary_computational_content_identical": True,
        "result_fingerprint": result_summary["reproducibility"][
            "result_fingerprint"
        ],
    }


def main() -> None:
    args = parse_args()
    with args.config.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    result = {
        "summary": verify_summary(args.result_dir),
        "independent_recalculation": verify_selected_replicates(
            config,
            args.result_dir,
            list(args.replicate_ids),
            args.atol,
        ),
    }
    if args.repeat_dir is not None:
        result["repeatability"] = verify_repeat(
            args.result_dir,
            args.repeat_dir,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
