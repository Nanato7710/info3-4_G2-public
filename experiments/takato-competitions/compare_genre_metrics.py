"""Compare two genre-level metric CSV files produced by analyze.py."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = EXPERIMENT_DIR / "outputs"
DEFAULT_CSV_A = OUTPUTS_DIR / "ctran_v2_best_model_genre_ap.csv"
DEFAULT_CSV_B = OUTPUTS_DIR / "ctran_v2_do2_best_model_genre_ap.csv"
METRICS = ("AP", "Precision", "Recall")
SOURCE_COLUMNS = [
    "genre",
    "support",
    "predicted_positive",
    *METRICS,
]
REQUIRED_COLUMNS = set(SOURCE_COLUMNS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare AP, Precision, and Recall between two genre CSV files."
    )
    parser.add_argument(
        "csv_a",
        type=Path,
        nargs="?",
        default=DEFAULT_CSV_A,
        help="First CSV file used as the baseline.",
    )
    parser.add_argument(
        "csv_b",
        type=Path,
        nargs="?",
        default=DEFAULT_CSV_B,
        help="Second CSV file compared against the baseline.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUTS_DIR / "genre_metrics_comparison.csv",
        help="Path for the detailed comparison CSV.",
    )
    return parser.parse_args()


def load_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV file was not found: {path}")

    dataframe = pd.read_csv(path)
    missing_columns = REQUIRED_COLUMNS - set(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{path} is missing required columns: {missing}")
    if dataframe["genre"].duplicated().any():
        duplicated = dataframe.loc[dataframe["genre"].duplicated(), "genre"].tolist()
        raise ValueError(f"{path} contains duplicate genres: {duplicated}")

    return dataframe[SOURCE_COLUMNS]


def compare_metrics(
    dataframe_a: pd.DataFrame,
    dataframe_b: pd.DataFrame,
) -> pd.DataFrame:
    comparison = dataframe_a.merge(
        dataframe_b,
        on="genre",
        how="outer",
        suffixes=("_a", "_b"),
        validate="one_to_one",
        indicator=True,
    )
    missing_genres = comparison.loc[
        comparison["_merge"] != "both",
        ["genre", "_merge"],
    ]
    if not missing_genres.empty:
        details = ", ".join(
            f"{genre} ({merge_state})"
            for genre, merge_state in missing_genres.itertuples(
                index=False,
                name=None,
            )
        )
        raise ValueError(f"The genre sets do not match: {details}")
    comparison = comparison.drop(columns="_merge")

    for metric in METRICS:
        comparison[f"{metric}_diff"] = (
            comparison[f"{metric}_b"] - comparison[f"{metric}_a"]
        )

    ordered_columns = [
        "genre",
        "support_a",
        "support_b",
        "predicted_positive_a",
        "predicted_positive_b",
    ]
    for metric in METRICS:
        ordered_columns.extend(
            [f"{metric}_a", f"{metric}_b", f"{metric}_diff"]
        )
    return comparison[ordered_columns]


def print_summary(
    comparison: pd.DataFrame,
    csv_a: Path,
    csv_b: Path,
) -> None:
    label_a = csv_a.stem
    label_b = csv_b.stem
    print(f"A (baseline): {csv_a}")
    print(f"B (comparison): {csv_b}")
    print("\nMean metrics")
    print(f"{'Metric':<12}{'A':>12}{'B':>12}{'B - A':>12}")

    for metric in METRICS:
        mean_a = comparison[f"{metric}_a"].mean()
        mean_b = comparison[f"{metric}_b"].mean()
        print(f"{metric:<12}{mean_a:>12.6f}{mean_b:>12.6f}{mean_b - mean_a:>12.6f}")

    print("\nGenres improved by B")
    for metric in METRICS:
        differences = comparison[f"{metric}_diff"]
        improved = int((differences > 0).sum())
        declined = int((differences < 0).sum())
        tied = int((differences == 0).sum())
        print(
            f"{metric:<12} improved={improved:>2}  "
            f"declined={declined:>2}  tied={tied:>2}"
        )

    display_columns = ["genre"]
    for metric in METRICS:
        display_columns.extend([f"{metric}_a", f"{metric}_b", f"{metric}_diff"])
    formatters = {
        column: (lambda value: f"{value:.6f}")
        for column in display_columns
        if column != "genre"
    }
    print(f"\nPer-genre comparison ({label_b} minus {label_a})")
    print(
        comparison[display_columns].to_string(
            index=False,
            formatters=formatters,
        )
    )


def main() -> None:
    args = parse_args()
    csv_a = args.csv_a.resolve()
    csv_b = args.csv_b.resolve()
    output_path = args.output.resolve()

    dataframe_a = load_metrics(csv_a)
    dataframe_b = load_metrics(csv_b)
    comparison = compare_metrics(dataframe_a, dataframe_b)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path, index=False)
    print_summary(comparison, csv_a, csv_b)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
