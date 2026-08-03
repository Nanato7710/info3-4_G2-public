from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "series_split_outputs"
IMAGE_DIR = ROOT / "data" / "images"
OUT_DIR = ROOT / "playground" / "kazusa" / "series_split" / "analysis"

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

LEGACY_SPLIT_FILES = {
    "train": ROOT / "data" / "training_data.csv",
    "validation": ROOT / "data" / "validation_data.csv",
    "test": ROOT / "data" / "test_data.csv",
}

COLORS = {
    "train": "#4E79A7",
    "validation": "#F28E2B",
    "test": "#59A14F",
    "neutral": "#6B7280",
    "accent": "#D37295",
    "grid": "#E5E7EB",
    "ink": "#222222",
}


def configure_plots() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#D1D5DB",
            "axes.labelcolor": COLORS["ink"],
            "axes.titlecolor": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "grid.color": COLORS["grid"],
            "font.family": ["DejaVu Sans", "sans-serif"],
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "savefig.bbox": "tight",
        }
    )


def load_current_data() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    all_df = pd.read_csv(DATA_DIR / "preprocessed_with_series_group.csv")
    splits = {
        split: pd.read_csv(DATA_DIR / file_name)
        for split, file_name in SPLIT_FILES.items()
    }
    return all_df, splits


def build_genre_distribution(all_df: pd.DataFrame, splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    total_rows = len(all_df)
    for genre in GENRE_COLS:
        total = int(all_df[genre].sum())
        row = {
            "genre": genre,
            "total": total,
            "overall_share": total / total_rows,
        }
        for split, df in splits.items():
            count = int(df[genre].sum())
            row[f"{split}_count"] = count
            row[f"{split}_within_split_share"] = count / len(df)
            row[f"{split}_share_of_genre_total"] = count / total if total else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values("total", ascending=False)


def build_label_count_distribution(all_df: pd.DataFrame, splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in {"all": all_df, **splits}.items():
        counts = Counter(df[GENRE_COLS].sum(axis=1).astype(int))
        for label_count in range(1, max(counts) + 1):
            rows.append(
                {
                    "split": name,
                    "label_count": label_count,
                    "rows": int(counts.get(label_count, 0)),
                    "share": counts.get(label_count, 0) / len(df),
                }
            )
    return pd.DataFrame(rows)


def build_group_size_distribution(all_df: pd.DataFrame) -> pd.DataFrame:
    group_sizes = all_df.groupby("SeriesGroup").size()
    bins = [
        ("1", group_sizes == 1),
        ("2", group_sizes == 2),
        ("3-5", (group_sizes >= 3) & (group_sizes <= 5)),
        ("6-10", (group_sizes >= 6) & (group_sizes <= 10)),
        ("11-20", (group_sizes >= 11) & (group_sizes <= 20)),
        ("21+", group_sizes >= 21),
    ]
    rows = []
    for label, mask in bins:
        rows.append(
            {
                "group_size_bin": label,
                "groups": int(mask.sum()),
                "rows": int(group_sizes[mask].sum()),
            }
        )
    return pd.DataFrame(rows)


def build_relation_distribution() -> pd.DataFrame:
    edges = pd.read_csv(DATA_DIR / "anilist_relation_edges.csv")
    total_counts = edges["relation_type"].value_counts()
    used_counts = edges.loc[edges["used_for_grouping"] == True, "relation_type"].value_counts()
    rows = []
    for relation_type in total_counts.index:
        rows.append(
            {
                "relation_type": relation_type,
                "total_edges": int(total_counts[relation_type]),
                "used_for_grouping": int(used_counts.get(relation_type, 0)),
                "unused": int(total_counts[relation_type] - used_counts.get(relation_type, 0)),
            }
        )
    return pd.DataFrame(rows)


def build_legacy_leakage_comparison(all_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    id_to_group = all_df.set_index("ID")["SeriesGroup"].to_dict()
    id_to_title = all_df.set_index("ID")["Title"].to_dict()

    split_groups: dict[str, set[str]] = {}
    split_rows: dict[str, pd.DataFrame] = {}
    for split, path in LEGACY_SPLIT_FILES.items():
        df = pd.read_csv(path)
        df["SeriesGroup"] = df["ID"].map(id_to_group)
        df["Title"] = df["ID"].map(id_to_title)
        split_rows[split] = df
        split_groups[split] = set(df["SeriesGroup"].dropna())

    rows = []
    for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        overlap = split_groups[left] & split_groups[right]
        rows.append(
            {
                "left_split": left,
                "right_split": right,
                "overlapping_series_groups": len(overlap),
                "left_rows_in_overlapping_groups": int(split_rows[left]["SeriesGroup"].isin(overlap).sum()),
                "right_rows_in_overlapping_groups": int(split_rows[right]["SeriesGroup"].isin(overlap).sum()),
            }
        )

    group_to_splits: dict[str, set[str]] = {}
    for split, groups in split_groups.items():
        for group in groups:
            group_to_splits.setdefault(group, set()).add(split)

    leaked_groups = {group for group, groups in group_to_splits.items() if len(groups) >= 2}
    summary_rows = []
    for split, df in split_rows.items():
        summary_rows.append(
            {
                "split": split,
                "rows": len(df),
                "series_groups": df["SeriesGroup"].nunique(),
                "rows_in_any_leaked_group": int(df["SeriesGroup"].isin(leaked_groups).sum()),
            }
        )

    examples = []
    for group in leaked_groups:
        counts = {}
        titles = []
        total_rows = 0
        for split, df in split_rows.items():
            gdf = df[df["SeriesGroup"] == group]
            counts[split] = len(gdf)
            total_rows += len(gdf)
            titles.extend(gdf["Title"].dropna().astype(str).head(3).tolist())
        examples.append(
            {
                "SeriesGroup": group,
                "total_rows_in_legacy_splits": total_rows,
                "train_rows": counts["train"],
                "validation_rows": counts["validation"],
                "test_rows": counts["test"],
                "example_titles": " / ".join(titles[:5]),
            }
        )
    examples_df = pd.DataFrame(examples).sort_values("total_rows_in_legacy_splits", ascending=False)

    comparison_df = pd.DataFrame(rows)
    comparison_df["total_rows_in_overlapping_groups"] = (
        comparison_df["left_rows_in_overlapping_groups"] + comparison_df["right_rows_in_overlapping_groups"]
    )
    summary_df = pd.DataFrame(summary_rows)
    comparison_df.to_csv(OUT_DIR / "legacy_split_pair_leakage.csv", index=False)
    summary_df.to_csv(OUT_DIR / "legacy_split_leakage_summary.csv", index=False)
    examples_df.head(25).to_csv(OUT_DIR / "legacy_split_leakage_examples.csv", index=False)
    return comparison_df, summary_df


def inspect_images(all_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    rows = []
    for anime_id in all_df["ID"].astype(int):
        path = IMAGE_DIR / f"{anime_id}.jpg"
        row = {
            "ID": anime_id,
            "path": str(path.relative_to(ROOT)),
            "exists": path.exists(),
            "readable": False,
            "width": None,
            "height": None,
            "mode": None,
            "format": None,
            "error": "",
        }
        if path.exists():
            try:
                with Image.open(path) as image:
                    row["width"], row["height"] = image.size
                    row["mode"] = image.mode
                    row["format"] = image.format
                    image.verify()
                row["readable"] = True
            except Exception as exc:  # noqa: BLE001
                row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)

    image_df = pd.DataFrame(rows)
    readable = image_df[image_df["readable"]].copy()
    summary: dict[str, object] = {
        "expected_images": len(image_df),
        "existing_images": int(image_df["exists"].sum()),
        "missing_images": int((~image_df["exists"]).sum()),
        "readable_images": int(image_df["readable"].sum()),
        "unreadable_images": int((image_df["exists"] & ~image_df["readable"]).sum()),
    }
    if not readable.empty:
        summary.update(
            {
                "min_width": int(readable["width"].min()),
                "median_width": float(readable["width"].median()),
                "max_width": int(readable["width"].max()),
                "min_height": int(readable["height"].min()),
                "median_height": float(readable["height"].median()),
                "max_height": int(readable["height"].max()),
                "format_counts": readable["format"].value_counts().to_dict(),
                "mode_counts": readable["mode"].value_counts().to_dict(),
            }
        )
    image_df.to_csv(OUT_DIR / "image_file_check.csv", index=False)
    return image_df, summary


def plot_genre_counts(genre_df: pd.DataFrame) -> None:
    plot_df = genre_df.sort_values("total", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 8), dpi=160)
    y = np.arange(len(plot_df))
    left = np.zeros(len(plot_df))
    for split in ["train", "validation", "test"]:
        values = plot_df[f"{split}_count"].to_numpy()
        ax.barh(y, values, left=left, color=COLORS[split], label=split)
        left += values
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["genre"])
    ax.set_xlabel("Rows")
    ax.set_title("Genre Counts by Split")
    ax.grid(axis="x", alpha=0.45)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "genre_counts_by_split.png")
    fig.savefig(OUT_DIR / "genre_counts_by_split.svg")
    plt.close(fig)


def plot_split_ratio_heatmap(genre_df: pd.DataFrame) -> None:
    matrix = genre_df.set_index("genre")[
        ["train_share_of_genre_total", "validation_share_of_genre_total", "test_share_of_genre_total"]
    ].loc[GENRE_COLS]
    target = np.array([0.8, 0.1, 0.1])
    delta = (matrix.to_numpy() - target.reshape(1, -1)) * 100

    fig, ax = plt.subplots(figsize=(9, 8), dpi=160)
    im = ax.imshow(delta, cmap="coolwarm", vmin=-6, vmax=6, aspect="auto")
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(["train", "validation", "test"])
    ax.set_yticks(np.arange(len(GENRE_COLS)))
    ax.set_yticklabels(GENRE_COLS)
    ax.set_title("Split Ratio Delta from 80/10/10 (percentage points)")
    for i in range(delta.shape[0]):
        for j in range(delta.shape[1]):
            ax.text(j, i, f"{delta[i, j]:+.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "genre_split_ratio_delta_heatmap.png")
    fig.savefig(OUT_DIR / "genre_split_ratio_delta_heatmap.svg")
    plt.close(fig)


def plot_label_count_distribution(label_df: pd.DataFrame) -> None:
    plot_df = label_df[label_df["split"] == "all"].sort_values("label_count")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    ax.bar(plot_df["label_count"], plot_df["rows"], color=COLORS["neutral"])
    ax.set_xlabel("Number of genres per row")
    ax.set_ylabel("Rows")
    ax.set_title("Label Count Distribution")
    ax.grid(axis="y", alpha=0.45)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "label_count_distribution.png")
    fig.savefig(OUT_DIR / "label_count_distribution.svg")
    plt.close(fig)


def plot_group_size_distribution(group_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    ax.bar(group_df["group_size_bin"], group_df["groups"], color=COLORS["accent"])
    ax.set_xlabel("Series group size")
    ax.set_ylabel("Series groups")
    ax.set_title("Series Group Size Distribution")
    ax.grid(axis="y", alpha=0.45)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "series_group_size_distribution.png")
    fig.savefig(OUT_DIR / "series_group_size_distribution.svg")
    plt.close(fig)


def plot_image_sizes(image_df: pd.DataFrame) -> None:
    readable = image_df[image_df["readable"]].copy()
    if readable.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 6), dpi=160)
    sample = readable.sample(n=min(2500, len(readable)), random_state=42)
    ax.scatter(sample["width"], sample["height"], s=8, alpha=0.35, color=COLORS["train"])
    ax.set_xlabel("Width")
    ax.set_ylabel("Height")
    ax.set_title("Cached Image Dimensions")
    ax.grid(alpha=0.45)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "image_dimensions_scatter.png")
    fig.savefig(OUT_DIR / "image_dimensions_scatter.svg")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_plots()

    all_df, splits = load_current_data()

    genre_df = build_genre_distribution(all_df, splits)
    label_df = build_label_count_distribution(all_df, splits)
    group_df = build_group_size_distribution(all_df)
    relation_df = build_relation_distribution()
    leakage_pair_df, leakage_summary_df = build_legacy_leakage_comparison(all_df)
    image_df, image_summary = inspect_images(all_df)

    genre_df.to_csv(OUT_DIR / "genre_distribution_from_data_outputs.csv", index=False)
    label_df.to_csv(OUT_DIR / "label_count_distribution.csv", index=False)
    group_df.to_csv(OUT_DIR / "series_group_size_distribution.csv", index=False)
    relation_df.to_csv(OUT_DIR / "relation_type_distribution.csv", index=False)

    plot_genre_counts(genre_df)
    plot_split_ratio_heatmap(genre_df)
    plot_label_count_distribution(label_df)
    plot_group_size_distribution(group_df)
    plot_image_sizes(image_df)

    group_sizes = all_df.groupby("SeriesGroup").size()
    split_overlap = {
        "train_validation": len(set(splits["train"]["SeriesGroup"]) & set(splits["validation"]["SeriesGroup"])),
        "train_test": len(set(splits["train"]["SeriesGroup"]) & set(splits["test"]["SeriesGroup"])),
        "validation_test": len(set(splits["validation"]["SeriesGroup"]) & set(splits["test"]["SeriesGroup"])),
    }
    summary = {
        "source_dir": str(DATA_DIR.relative_to(ROOT)),
        "output_dir": str(OUT_DIR.relative_to(ROOT)),
        "rows": len(all_df),
        "unique_ids": int(all_df["ID"].nunique()),
        "series_groups": int(all_df["SeriesGroup"].nunique()),
        "multi_item_series_groups": int((group_sizes > 1).sum()),
        "max_series_group_size": int(group_sizes.max()),
        "split_overlap": split_overlap,
        "image_summary": image_summary,
        "legacy_split_pair_leakage": leakage_pair_df.to_dict(orient="records"),
        "legacy_split_leakage_summary": leakage_summary_df.to_dict(orient="records"),
        "outputs": sorted(path.name for path in OUT_DIR.iterdir()),
    }
    with open(OUT_DIR / "dataset_analysis_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
