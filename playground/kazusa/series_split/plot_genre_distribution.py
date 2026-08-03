from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GENRE_COLS = [
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
]

SPLIT_FILES = {
    "train": "training_data_grouped.csv",
    "validation": "validation_data_grouped.csv",
    "test": "test_data_grouped.csv",
}
TARGET_RATIOS = {"train": 0.8, "validation": 0.1, "test": 0.1}

FONT_FAMILY = ["DejaVu Sans", "sans-serif"]
MONO_FONT_FAMILY = ["DejaVu Sans Mono", "monospace"]

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}
COLORS = {
    "train": "#A3BEFA",
    "validation": "#F0986E",
    "test": "#A3D576",
    "train_edge": "#2E4780",
    "validation_edge": "#804126",
    "test_edge": "#386411",
    "negative": "#5477C4",
    "positive": "#CC6F47",
    "neutral": "#FFFFFF",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot genre distribution charts for grouped train/validation/test splits."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("playground/kazusa/series_split/outputs"),
        help="Directory containing *_data_grouped.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("playground/kazusa/series_split/outputs/figures"),
        help="Directory where chart images and chart data are written.",
    )
    return parser.parse_args()


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "axes.titlecolor": TOKENS["ink"],
            "xtick.color": TOKENS["muted"],
            "ytick.color": TOKENS["ink"],
            "grid.color": TOKENS["grid"],
            "font.family": FONT_FAMILY,
            "font.size": 10,
            "axes.titlesize": 17,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "savefig.facecolor": TOKENS["surface"],
            "savefig.bbox": "tight",
        }
    )


def add_chart_header(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.text(0.08, 0.965, title, ha="left", va="top", fontsize=17, fontweight="bold", color=TOKENS["ink"])
    fig.text(0.08, 0.925, subtitle, ha="left", va="top", fontsize=10.5, color=TOKENS["muted"])


def load_split_data(input_dir: Path) -> dict[str, pd.DataFrame]:
    split_data = {}
    for split_name, file_name in SPLIT_FILES.items():
        path = input_dir / file_name
        if not path.exists():
            raise FileNotFoundError(f"missing split file: {path}")
        df = pd.read_csv(path)
        missing = [genre for genre in GENRE_COLS if genre not in df.columns]
        if missing:
            raise ValueError(f"{path} is missing genre columns: {missing}")
        split_data[split_name] = df
    return split_data


def build_chart_data(split_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    total_counts = sum((df[GENRE_COLS].sum() for df in split_data.values()))
    rows = []
    for split_name, df in split_data.items():
        split_counts = df[GENRE_COLS].sum()
        for genre in GENRE_COLS:
            count = int(split_counts[genre])
            rows.append(
                {
                    "genre": genre,
                    "split": split_name,
                    "split_rows": len(df),
                    "genre_count": count,
                    "share_within_split": count / len(df),
                    "share_of_genre_total": count / int(total_counts[genre]),
                    "target_share_of_total": TARGET_RATIOS[split_name],
                    "target_delta_pp": (count / int(total_counts[genre]) - TARGET_RATIOS[split_name]) * 100,
                    "total_genre_count": int(total_counts[genre]),
                }
            )
    return pd.DataFrame(rows)


def write_chart_data(path: Path, chart_data: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    chart_data.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def plot_within_split_distribution(chart_data: pd.DataFrame, output_dir: Path) -> None:
    pivot = (
        chart_data.pivot(index="genre", columns="split", values="share_within_split")
        .loc[GENRE_COLS]
        .sort_values("train", ascending=True)
    )

    y = np.arange(len(pivot))
    bar_height = 0.24
    offsets = {"train": -bar_height, "validation": 0.0, "test": bar_height}

    fig, ax = plt.subplots(figsize=(12, 10))
    fig.subplots_adjust(left=0.22, right=0.96, top=0.85, bottom=0.08)

    for split_name in ["train", "validation", "test"]:
        values = pivot[split_name].to_numpy() * 100
        ax.barh(
            y + offsets[split_name],
            values,
            height=bar_height * 0.86,
            label=split_name,
            color=COLORS[split_name],
            edgecolor=COLORS[f"{split_name}_edge"],
            linewidth=0.8,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Rows with genre within each split (%)")
    ax.set_xlim(0, max(48, float((pivot.max().max() * 100) + 4)))
    ax.grid(axis="x", linestyle="-", linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(TOKENS["axis"])
    ax.spines["bottom"].set_color(TOKENS["axis"])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.055), ncols=3, frameon=False)

    add_chart_header(
        fig,
        "Genre distribution by split",
        "Percent of rows tagged with each genre inside train, validation, and test after AniList relation grouping.",
    )

    for ext in ["png", "svg"]:
        fig.savefig(output_dir / f"genre_distribution_by_split.{ext}", dpi=180)
    plt.close(fig)


def plot_split_balance_heatmap(chart_data: pd.DataFrame, output_dir: Path) -> None:
    pivot = (
        chart_data.pivot(index="genre", columns="split", values="target_delta_pp")
        .loc[GENRE_COLS, ["train", "validation", "test"]]
    )
    order = pivot.abs().max(axis=1).sort_values(ascending=True).index
    pivot = pivot.loc[order]

    values = pivot.to_numpy()
    max_abs = max(1.0, float(np.nanmax(np.abs(values))))

    fig, ax = plt.subplots(figsize=(8.8, 10))
    fig.subplots_adjust(left=0.30, right=0.88, top=0.84, bottom=0.08)
    image = ax.imshow(values, cmap="RdBu_r", vmin=-max_abs, vmax=max_abs, aspect="auto")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.tick_params(axis="both", length=0)
    ax.set_xlabel("Split")
    ax.set_ylabel("Genre")

    for row_index in range(values.shape[0]):
        for col_index in range(values.shape[1]):
            value = values[row_index, col_index]
            label_color = TOKENS["ink"] if abs(value) < max_abs * 0.52 else "#FFFFFF"
            ax.text(
                col_index,
                row_index,
                f"{value:+.1f}",
                ha="center",
                va="center",
                fontsize=8.6,
                color=label_color,
                family=MONO_FONT_FAMILY,
            )

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(pivot.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(pivot.index), 1), minor=True)
    ax.grid(which="minor", color=TOKENS["surface"], linestyle="-", linewidth=1.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.035)
    colorbar.set_label("Difference from target split share (percentage points)")
    colorbar.outline.set_edgecolor(TOKENS["axis"])

    add_chart_header(
        fig,
        "Genre balance versus target split ratios",
        "Cells show how far each split's share of a genre is from the 80% / 10% / 10% target.",
    )

    for ext in ["png", "svg"]:
        fig.savefig(output_dir / f"genre_split_balance_heatmap.{ext}", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_plot_style()

    split_data = load_split_data(args.input_dir)
    chart_data = build_chart_data(split_data)
    write_chart_data(args.output_dir / "genre_distribution_chart_data.csv", chart_data)
    plot_within_split_distribution(chart_data, args.output_dir)
    plot_split_balance_heatmap(chart_data, args.output_dir)

    print(f"generated figures in {args.output_dir}")


if __name__ == "__main__":
    main()
