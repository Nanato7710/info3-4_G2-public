from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import sklearn
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["svg.hashsalt"] = "info3dm-g2-bootstrap-validation"

BOOTSTRAP_DIR = Path(__file__).resolve().parent
OUTPUTS_ROOT = BOOTSTRAP_DIR.parent
PROJECT_ROOT = OUTPUTS_ROOT.parent
if str(OUTPUTS_ROOT) not in sys.path:
    sys.path.insert(0, str(OUTPUTS_ROOT))

from artifact_common import (  # noqa: E402
    EXPECTED_TEST_SERIES_GROUPS,
    GENRE_NAMES,
    calculate_map,
    load_predictions,
    sha256_file,
    validate_prediction_arrays,
)

MODELS = ("baseline", "final-tri-model")
SEEDS = (42, 43, 44)
EXPECTED_ROWS = 1121
EXPECTED_LABELS = 19
REPLICATE_COLUMNS = (
    "replicate_id",
    "attempt_id",
    "n_rows",
    "n_unique_groups",
    "baseline_seed_42_map",
    "baseline_seed_43_map",
    "baseline_seed_44_map",
    "baseline_mean_map",
    "final_seed_42_map",
    "final_seed_43_map",
    "final_seed_44_map",
    "final_mean_map",
    "delta_map",
)
GENERATED_FILENAMES = (
    "replicates.csv",
    "summary.json",
    "delta-map-distribution.png",
    "delta-map-distribution.svg",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a paired SeriesGroup cluster Bootstrap for test mAP."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Override output_dir without changing the computation. "
            "Use a distinct directory for reproducibility runs."
        ),
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def resolve_project_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def project_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def load_and_validate_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("config must be a YAML mapping")

    required = {
        "n_valid_replicates",
        "random_seed",
        "confidence_level",
        "sampling_unit",
        "invalid_if_any_label_has_no_positive",
        "max_attempts",
        "inputs",
        "output_dir",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"config is missing keys: {missing}")
    if int(config["n_valid_replicates"]) <= 0:
        raise ValueError("n_valid_replicates must be positive")
    if int(config["max_attempts"]) < int(config["n_valid_replicates"]):
        raise ValueError("max_attempts must be at least n_valid_replicates")
    if not 0 < float(config["confidence_level"]) < 1:
        raise ValueError("confidence_level must be between 0 and 1")
    if config["sampling_unit"] != "SeriesGroup":
        raise ValueError("sampling_unit must be SeriesGroup")
    if config["invalid_if_any_label_has_no_positive"] is not True:
        raise ValueError("invalid_if_any_label_has_no_positive must be true")

    inputs = config["inputs"]
    if not isinstance(inputs, dict) or set(inputs) != set(MODELS):
        raise ValueError(f"inputs must contain exactly {MODELS}")
    expected_seed_keys = {f"seed_{seed}" for seed in SEEDS}
    for model in MODELS:
        if not isinstance(inputs[model], dict):
            raise ValueError(f"inputs.{model} must be a mapping")
        if set(inputs[model]) != expected_seed_keys:
            raise ValueError(
                f"inputs.{model} must contain exactly {sorted(expected_seed_keys)}"
            )
    return config


def normalized_id(value: Any) -> tuple[str, Any]:
    scalar = value.item() if isinstance(value, np.generic) else value
    return type(scalar).__name__, scalar


def non_empty_strings(values: np.ndarray, name: str) -> None:
    rendered = np.char.lower(np.char.strip(values.astype(str)))
    missing_tokens = np.asarray(["", "nan", "none", "null", "<na>"])
    if np.isin(rendered, missing_tokens).any():
        raise ValueError(f"{name} contains missing or blank values")


def align_to_reference(
    arrays: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    *,
    model: str,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    reference_ids = [normalized_id(value) for value in reference["ids"]]
    current_ids = [normalized_id(value) for value in arrays["ids"]]
    if len(set(current_ids)) != len(current_ids):
        raise ValueError(f"duplicate IDs for {model} seed {seed}")
    if current_ids == reference_ids:
        return arrays, {
            "model": model,
            "seed": seed,
            "action": "already_aligned",
            "rows_reordered": 0,
        }
    if set(current_ids) != set(reference_ids):
        missing = len(set(reference_ids) - set(current_ids))
        extra = len(set(current_ids) - set(reference_ids))
        raise ValueError(
            f"ID set differs for {model} seed {seed}: missing={missing}, extra={extra}"
        )

    row_by_id = {value: index for index, value in enumerate(current_ids)}
    order = np.asarray([row_by_id[value] for value in reference_ids], dtype=np.int64)
    aligned = {
        key: (value if key == "genre_names" else value[order])
        for key, value in arrays.items()
    }
    rows_reordered = int(np.count_nonzero(order != np.arange(len(order))))
    return aligned, {
        "model": model,
        "seed": seed,
        "action": "reordered_by_unique_id",
        "rows_reordered": rows_reordered,
    }


def validate_one_input(
    arrays: dict[str, np.ndarray],
    *,
    model: str,
    seed: int,
) -> dict[str, Any]:
    validate_prediction_arrays(
        arrays,
        expected_rows=EXPECTED_ROWS,
        expected_series_groups=EXPECTED_TEST_SERIES_GROUPS,
    )
    if arrays["ids"].dtype.kind not in {"i", "u", "U", "S"}:
        raise ValueError(f"ids must be integer or string for {model} seed {seed}")
    non_empty_strings(arrays["ids"], "ids")
    non_empty_strings(arrays["series_groups"], "series_groups")
    if arrays["targets"].shape[1] != EXPECTED_LABELS:
        raise ValueError(
            f"expected {EXPECTED_LABELS} labels for {model} seed {seed}"
        )
    target_values = np.unique(arrays["targets"])
    if not np.isin(target_values, (0, 1)).all():
        raise ValueError(f"targets are not binary for {model} seed {seed}")
    label_positives = arrays["targets"].sum(axis=0)
    if np.any(label_positives == 0):
        absent = np.asarray(GENRE_NAMES)[label_positives == 0].tolist()
        raise ValueError(f"full test has labels without positives: {absent}")
    return {
        "schema_valid": True,
        "row_count": int(len(arrays["ids"])),
        "label_count": int(arrays["targets"].shape[1]),
        "series_group_count": int(len(np.unique(arrays["series_groups"]))),
        "unique_id_count": int(len(np.unique(arrays["ids"]))),
        "missing_id_count": 0,
        "missing_series_group_count": 0,
        "target_values": [int(value) for value in target_values.tolist()],
        "score_min": float(arrays["scores"].min()),
        "score_max": float(arrays["scores"].max()),
        "scores_finite": bool(np.isfinite(arrays["scores"]).all()),
        "logits_finite": bool(np.isfinite(arrays["logits"]).all()),
        "positive_counts_by_genre": {
            genre: int(count)
            for genre, count in zip(GENRE_NAMES, label_positives, strict=True)
        },
    }


def load_inputs(
    config: dict[str, Any],
) -> tuple[
    dict[str, dict[int, dict[str, np.ndarray]]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    loaded: dict[str, dict[int, dict[str, np.ndarray]]] = {
        model: {} for model in MODELS
    }
    input_records = []
    alignment_records = []
    reference: dict[str, np.ndarray] | None = None

    for model in MODELS:
        for seed in SEEDS:
            path = resolve_project_path(config["inputs"][model][f"seed_{seed}"])
            if not path.is_file():
                raise FileNotFoundError(path)
            arrays = load_predictions(path)
            schema = validate_one_input(arrays, model=model, seed=seed)
            if reference is None:
                reference = arrays
                alignment = {
                    "model": model,
                    "seed": seed,
                    "action": "reference_order",
                    "rows_reordered": 0,
                }
            else:
                if not np.array_equal(reference["genre_names"], arrays["genre_names"]):
                    raise ValueError(
                        f"genre_names differs for {model} seed {seed}"
                    )
                arrays, alignment = align_to_reference(
                    arrays,
                    reference,
                    model=model,
                    seed=seed,
                )
                if not np.array_equal(
                    reference["series_groups"], arrays["series_groups"]
                ):
                    raise ValueError(
                        f"series_groups differs for {model} seed {seed}"
                    )
                if not np.array_equal(reference["targets"], arrays["targets"]):
                    raise ValueError(f"targets differs for {model} seed {seed}")
                if not np.array_equal(reference["ids"], arrays["ids"]):
                    raise ValueError(f"ids differs after alignment for {model} seed {seed}")
            loaded[model][seed] = arrays
            alignment_records.append(alignment)
            input_records.append(
                {
                    "model": model,
                    "seed": seed,
                    "path": project_relative(path),
                    "sha256": sha256_file(path),
                    "schema_check": schema,
                }
            )

    return loaded, input_records, alignment_records


def build_group_rows(series_groups: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    unique_groups = np.unique(series_groups)
    rows_by_group = [
        np.flatnonzero(series_groups == group).astype(np.int64)
        for group in unique_groups
    ]
    return unique_groups, rows_by_group


def sampled_row_indices(
    drawn_group_indices: np.ndarray,
    rows_by_group: list[np.ndarray],
) -> np.ndarray:
    return np.concatenate(
        [rows_by_group[int(group_index)] for group_index in drawn_group_indices]
    )


def map_for_rows(
    targets: np.ndarray,
    scores: np.ndarray,
    row_indices: np.ndarray,
) -> float:
    return calculate_map(targets[row_indices], scores[row_indices])


def run_bootstrap(
    loaded: dict[str, dict[int, dict[str, np.ndarray]]],
    *,
    n_valid_replicates: int,
    random_seed: int,
    max_attempts: int,
) -> tuple[list[dict[str, Any]], int, Counter[str], Counter[str]]:
    reference = loaded[MODELS[0]][SEEDS[0]]
    targets = reference["targets"]
    unique_groups, rows_by_group = build_group_rows(reference["series_groups"])
    if len(unique_groups) != EXPECTED_TEST_SERIES_GROUPS:
        raise ValueError(
            f"expected {EXPECTED_TEST_SERIES_GROUPS} groups, got {len(unique_groups)}"
        )

    rng = np.random.default_rng(random_seed)
    rows = []
    invalid_reasons: Counter[str] = Counter()
    invalid_labels: Counter[str] = Counter()

    for attempt_id in range(1, max_attempts + 1):
        drawn = rng.integers(
            0,
            len(unique_groups),
            size=len(unique_groups),
            dtype=np.int64,
        )
        row_indices = sampled_row_indices(drawn, rows_by_group)
        positive_counts = targets[row_indices].sum(axis=0)
        missing_mask = positive_counts == 0
        if np.any(missing_mask):
            invalid_reasons["any_label_has_no_positive"] += 1
            for genre in np.asarray(GENRE_NAMES)[missing_mask]:
                invalid_labels[str(genre)] += 1
            continue

        baseline_maps = {
            seed: map_for_rows(
                targets,
                loaded["baseline"][seed]["scores"],
                row_indices,
            )
            for seed in SEEDS
        }
        final_maps = {
            seed: map_for_rows(
                targets,
                loaded["final-tri-model"][seed]["scores"],
                row_indices,
            )
            for seed in SEEDS
        }
        baseline_mean = float(np.mean(list(baseline_maps.values())))
        final_mean = float(np.mean(list(final_maps.values())))
        rows.append(
            {
                "replicate_id": len(rows) + 1,
                "attempt_id": attempt_id,
                "n_rows": int(len(row_indices)),
                "n_unique_groups": int(len(np.unique(drawn))),
                **{
                    f"baseline_seed_{seed}_map": baseline_maps[seed]
                    for seed in SEEDS
                },
                "baseline_mean_map": baseline_mean,
                **{
                    f"final_seed_{seed}_map": final_maps[seed] for seed in SEEDS
                },
                "final_mean_map": final_mean,
                "delta_map": final_mean - baseline_mean,
            }
        )
        if len(rows) == n_valid_replicates:
            return rows, attempt_id, invalid_reasons, invalid_labels
        if len(rows) % 1000 == 0:
            print(
                f"valid replicates: {len(rows)}/{n_valid_replicates} "
                f"(attempts: {attempt_id})",
                flush=True,
            )

    raise RuntimeError(
        f"only {len(rows)} valid replicates after {max_attempts} attempts; "
        "partial results were not written"
    )


def full_test_estimate(
    loaded: dict[str, dict[int, dict[str, np.ndarray]]],
) -> tuple[dict[int, float], dict[int, float], float]:
    reference = loaded[MODELS[0]][SEEDS[0]]
    targets = reference["targets"]
    full_rows = np.arange(len(targets), dtype=np.int64)
    baseline = {
        seed: map_for_rows(targets, loaded["baseline"][seed]["scores"], full_rows)
        for seed in SEEDS
    }
    final = {
        seed: map_for_rows(
            targets,
            loaded["final-tri-model"][seed]["scores"],
            full_rows,
        )
        for seed in SEEDS
    }
    return baseline, final, float(np.mean(list(final.values())) - np.mean(list(baseline.values())))


def write_replicates(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REPLICATE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def plot_distribution(
    deltas: np.ndarray,
    *,
    point_estimate: float,
    confidence_interval: tuple[float, float],
    confidence_level: float,
    png_path: Path,
    svg_path: Path,
) -> None:
    """Render the static chart contract for the formal result.

    Question: what is the paired cluster-Bootstrap distribution of delta mAP?
    Form: histogram for 10,000 replicate-level values with uncertainty references.
    Palette: one blue root, orange point estimate, neutral zero reference; line
    styles duplicate color meaning. Final QA surfaces are the exported PNG and SVG.
    """
    lower, upper = confidence_interval
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.hist(
        deltas,
        bins=50,
        color="#4E79A7",
        edgecolor="#24435C",
        linewidth=0.5,
        alpha=0.82,
    )
    ax.axvline(0, color="#4A4A4A", linestyle=":", linewidth=1.8, label="Zero")
    ax.axvline(
        point_estimate,
        color="#F28E2B",
        linestyle="-",
        linewidth=2.2,
        label=f"Full-test point estimate ({point_estimate:.4f})",
    )
    ax.axvline(
        lower,
        color="#24435C",
        linestyle="--",
        linewidth=1.8,
        label=f"{confidence_level:.0%} CI ({lower:.4f}, {upper:.4f})",
    )
    ax.axvline(upper, color="#24435C", linestyle="--", linewidth=1.8)
    ax.set_title(
        "Paired SeriesGroup Bootstrap Distribution of mAP Difference",
        loc="left",
        fontsize=15,
        color="#222222",
        pad=20,
    )
    ax.text(
        0,
        1.01,
        (
            f"{len(deltas):,} valid replicates; percentile "
            f"{confidence_level:.0%} confidence interval"
        ),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color="#555555",
    )
    ax.set_xlabel("delta mAP (final-tri-model - baseline)")
    ax.set_ylabel("Valid replicate count")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(png_path, dpi=180, facecolor="white")
    fig.savefig(svg_path, facecolor="white", metadata={"Date": None})
    plt.close(fig)


def refuse_generated_overwrite(output_dir: Path) -> None:
    existing = [
        output_dir / filename
        for filename in GENERATED_FILENAMES
        if (output_dir / filename).exists()
    ]
    if existing:
        rendered = "\n".join(str(path) for path in existing)
        raise FileExistsError(
            "refusing to overwrite generated Bootstrap outputs; "
            "choose a distinct --output-dir:\n"
            + rendered
        )


def main() -> None:
    args = parse_args()
    started_at = utc_now()
    monotonic_start = time.monotonic()
    config_path = args.config.resolve()
    config = load_and_validate_config(config_path)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else resolve_project_path(config["output_dir"]).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    refuse_generated_overwrite(output_dir)

    loaded, input_records, alignment_records = load_inputs(config)
    rows, total_attempts, invalid_reasons, invalid_labels = run_bootstrap(
        loaded,
        n_valid_replicates=int(config["n_valid_replicates"]),
        random_seed=int(config["random_seed"]),
        max_attempts=int(config["max_attempts"]),
    )
    baseline_full, final_full, point_estimate = full_test_estimate(loaded)

    replicates_path = output_dir / "replicates.csv"
    write_replicates(replicates_path, rows)
    replicates = pd.DataFrame(rows, columns=REPLICATE_COLUMNS)
    deltas = replicates["delta_map"].to_numpy(dtype=float)
    alpha = 1.0 - float(config["confidence_level"])
    lower, upper = np.quantile(
        deltas,
        [alpha / 2.0, 1.0 - alpha / 2.0],
        method="linear",
    )
    lower = float(lower)
    upper = float(upper)

    png_path = output_dir / "delta-map-distribution.png"
    svg_path = output_dir / "delta-map-distribution.svg"
    plot_distribution(
        deltas,
        point_estimate=point_estimate,
        confidence_interval=(lower, upper),
        confidence_level=float(config["confidence_level"]),
        png_path=png_path,
        svg_path=svg_path,
    )

    stable_config = {
        key: config[key]
        for key in (
            "n_valid_replicates",
            "random_seed",
            "confidence_level",
            "sampling_unit",
            "invalid_if_any_label_has_no_positive",
            "max_attempts",
            "inputs",
        )
    }
    summary = {
        "analysis": {
            "delta_definition": (
                "mean_mAP(final-tri-model seeds 42,43,44) - "
                "mean_mAP(baseline seeds 42,43,44)"
            ),
            "point_estimate": point_estimate,
            "full_test_seed_maps": {
                "baseline": {str(seed): baseline_full[seed] for seed in SEEDS},
                "final-tri-model": {str(seed): final_full[seed] for seed in SEEDS},
            },
            "confidence_level": float(config["confidence_level"]),
            "confidence_interval_method": "percentile",
            "confidence_interval": {"lower": lower, "upper": upper},
            "confidence_interval_crosses_zero": bool(lower <= 0 <= upper),
            "bootstrap_delta_mean": float(deltas.mean()),
            "bootstrap_delta_standard_deviation": float(deltas.std(ddof=1)),
            "bootstrap_delta_standard_deviation_ddof": 1,
            "proportion_delta_greater_than_zero": float(np.mean(deltas > 0)),
        },
        "sampling": {
            "sampling_unit": config["sampling_unit"],
            "population_group_count": EXPECTED_TEST_SERIES_GROUPS,
            "groups_drawn_with_replacement_per_attempt": EXPECTED_TEST_SERIES_GROUPS,
            "random_seed": int(config["random_seed"]),
            "valid_replicates": len(rows),
            "total_attempts": total_attempts,
            "invalid_attempts": total_attempts - len(rows),
            "invalid_reason_counts": {
                "any_label_has_no_positive": int(
                    invalid_reasons.get("any_label_has_no_positive", 0)
                )
            },
            "invalid_label_counts": {
                genre: int(invalid_labels.get(genre, 0)) for genre in GENRE_NAMES
            },
            "max_attempts": int(config["max_attempts"]),
        },
        "input_validation": {
            "status": "passed",
            "expected": {
                "row_count": EXPECTED_ROWS,
                "label_count": EXPECTED_LABELS,
                "series_group_count": EXPECTED_TEST_SERIES_GROUPS,
            },
            "alignment_records": alignment_records,
            "inputs": input_records,
        },
        "reproducibility": {
            "config_path": project_relative(config_path),
            "computational_config_sha256": sha256_text(canonical_json(stable_config)),
            "rng": "numpy.random.Generator",
            "bit_generator": "PCG64",
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "scikit_learn_version": sklearn.__version__,
            "matplotlib_version": matplotlib.__version__,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "outputs": {
            "replicates_csv": {
                "path": project_relative(replicates_path),
                "sha256": sha256_file(replicates_path),
                "row_count": len(rows),
            },
            "distribution_png": {
                "path": project_relative(png_path),
                "sha256": sha256_file(png_path),
            },
            "distribution_svg": {
                "path": project_relative(svg_path),
                "sha256": sha256_file(svg_path),
            },
        },
        "execution": {
            "started_at": started_at,
            "finished_at": utc_now(),
            "duration_seconds": float(time.monotonic() - monotonic_start),
        },
    }
    summary["reproducibility"]["result_fingerprint"] = sha256_text(
        canonical_json(
            {
                "computational_config_sha256": summary["reproducibility"][
                    "computational_config_sha256"
                ],
                "input_sha256": [record["sha256"] for record in input_records],
                "replicates_sha256": summary["outputs"]["replicates_csv"]["sha256"],
                "analysis": summary["analysis"],
                "sampling": summary["sampling"],
            }
        )
    )
    summary_path = output_dir / "summary.json"
    temporary_summary = summary_path.with_suffix(".json.tmp")
    temporary_summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_summary.replace(summary_path)
    print(
        f"completed {len(rows)} valid replicates in {total_attempts} attempts; "
        f"delta={point_estimate:.6f}, "
        f"{float(config['confidence_level']):.0%} CI=[{lower:.6f}, {upper:.6f}]"
    )


if __name__ == "__main__":
    main()
