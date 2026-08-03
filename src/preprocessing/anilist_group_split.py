from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests


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

ANILIST_URL = "https://graphql.anilist.co"
DEFAULT_GROUPING_RELATION_TYPES = [
    "PREQUEL",
    "SEQUEL",
    "SIDE_STORY",
    "PARENT",
    "SUMMARY",
    "ALTERNATIVE",
    "SPIN_OFF",
    "COMPILATION",
    "CONTAINS",
]
RELATION_QUERY = """
query ($ids: [Int]) {
  Page(page: 1, perPage: 50) {
    media(id_in: $ids, type: ANIME, sort: ID) {
      id
      relations {
        edges {
          relationType
          node {
            id
            type
          }
        }
      }
    }
  }
}
"""


class UnionFind:
    def __init__(self, values: list[int]) -> None:
        self.parent = {value: value for value in values}
        self.rank = {value: 0 for value in values}

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return

        if self.rank[left_root] < self.rank[right_root]:
            self.parent[left_root] = right_root
        elif self.rank[left_root] > self.rank[right_root]:
            self.parent[right_root] = left_root
        else:
            self.parent[right_root] = left_root
            self.rank[left_root] += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build AniList-relation series groups and create leakage-aware "
            "train/validation/test splits."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/preprocessed_anime_data.csv"),
        help="Input CSV with ID, Title, ImageUrl, and genre columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/series_split_outputs"),
        help="Directory where generated files are written.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Target row ratio for the training split.",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.1,
        help="Target row ratio for the validation split.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Target row ratio for the test split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for split search.",
    )
    parser.add_argument(
        "--split-trials",
        type=int,
        default=2000,
        help="Number of randomized group split candidates to evaluate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of AniList IDs fetched per GraphQL request. AniList Page perPage max is 50.",
    )
    parser.add_argument(
        "--request-interval",
        type=float,
        default=2.2,
        help="Minimum seconds between AniList requests.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum retries per AniList request batch.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Ignore existing relation cache and fetch all IDs again.",
    )
    parser.add_argument(
        "--relation-types",
        default=",".join(DEFAULT_GROUPING_RELATION_TYPES),
        help=(
            "Comma-separated AniList relationType values used for grouping. "
            "Use ALL to group by every anime relation in the cache."
        ),
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    missing = [col for col in ["ID", "Title", "ImageUrl", *GENRE_COLS] if col not in rows[0]]
    if missing:
        raise ValueError(f"input is missing required columns: {missing}")
    return rows


def write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")


def chunks(values: list[int], size: int) -> list[list[int]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def retry_after_seconds(response: requests.Response) -> float | None:
    retry_after = response.headers.get("Retry-After")
    if retry_after is None:
        return None
    try:
        return float(retry_after)
    except ValueError:
        return None


def fetch_relation_batch(
    session: requests.Session,
    ids: list[int],
    max_retries: int,
) -> list[dict[str, Any]]:
    for attempt in range(1, max_retries + 1):
        response = session.post(
            ANILIST_URL,
            json={"query": RELATION_QUERY, "variables": {"ids": ids}},
            timeout=30,
        )

        if response.status_code == 429:
            wait_seconds = retry_after_seconds(response) or 60.0
            print(f"rate limited: waiting {wait_seconds:.1f}s")
            time.sleep(wait_seconds)
            continue

        try:
            body = response.json()
        except ValueError as exc:
            if attempt == max_retries:
                raise RuntimeError(f"AniList returned non-JSON response: {response.status_code}") from exc
            time.sleep(2**attempt)
            continue

        if response.status_code >= 400 or body.get("errors"):
            if attempt == max_retries:
                raise RuntimeError(
                    f"AniList API error: status={response.status_code}, errors={body.get('errors')}"
                )
            wait_seconds = retry_after_seconds(response) or float(2**attempt)
            print(f"AniList API error, retrying in {wait_seconds:.1f}s")
            time.sleep(wait_seconds)
            continue

        return body.get("data", {}).get("Page", {}).get("media", []) or []

    raise RuntimeError("unreachable retry state")


def normalize_media_relations(media: dict[str, Any]) -> list[dict[str, Any]]:
    edges = media.get("relations", {}).get("edges") or []
    normalized = []
    for edge in edges:
        node = edge.get("node") or {}
        if node.get("type") != "ANIME":
            continue
        related_id = node.get("id")
        if related_id is None:
            continue
        normalized.append(
            {
                "relation_type": edge.get("relationType") or "UNKNOWN",
                "related_id": int(related_id),
            }
        )
    return normalized


def fetch_relations(
    ids: list[int],
    cache_path: Path,
    batch_size: int,
    request_interval: float,
    max_retries: int,
    refresh_cache: bool,
) -> dict[str, list[dict[str, Any]]]:
    cache: dict[str, list[dict[str, Any]]] = {} if refresh_cache else load_json(cache_path, {})
    missing_ids = [anime_id for anime_id in ids if str(anime_id) not in cache]

    if not missing_ids:
        print(f"relation cache is complete: {cache_path}")
        return cache

    print(f"fetching AniList relations: {len(missing_ids)} IDs")
    batches = chunks(missing_ids, min(batch_size, 50))
    last_request_at = 0.0

    with requests.Session() as session:
        for batch_index, batch in enumerate(batches, start=1):
            elapsed = time.perf_counter() - last_request_at
            if elapsed < request_interval:
                time.sleep(request_interval - elapsed)

            media_list = fetch_relation_batch(session, batch, max_retries)
            last_request_at = time.perf_counter()

            seen_ids = set()
            for media in media_list:
                anime_id = int(media["id"])
                seen_ids.add(anime_id)
                cache[str(anime_id)] = normalize_media_relations(media)

            for anime_id in batch:
                if anime_id not in seen_ids:
                    cache[str(anime_id)] = []

            write_json(cache_path, cache)
            print(f"  batch {batch_index}/{len(batches)} cached ({len(cache)}/{len(ids)})")

    return cache


def build_series_groups(
    rows: list[dict[str, str]],
    relations: dict[str, list[dict[str, Any]]],
    allowed_relation_types: set[str] | None,
) -> tuple[dict[int, str], list[dict[str, Any]]]:
    ids = [int(row["ID"]) for row in rows]
    id_set = set(ids)
    union_find = UnionFind(ids)
    relation_edges = []

    for source_id_text, related_items in relations.items():
        source_id = int(source_id_text)
        if source_id not in id_set:
            continue
        for item in related_items:
            target_id = int(item["related_id"])
            relation_type = item["relation_type"]
            target_in_dataset = target_id in id_set
            used_for_grouping = (
                target_in_dataset
                and (allowed_relation_types is None or relation_type in allowed_relation_types)
            )
            relation_edges.append(
                {
                    "source_id": source_id,
                    "target_id": target_id,
                    "relation_type": relation_type,
                    "target_in_dataset": target_in_dataset,
                    "used_for_grouping": used_for_grouping,
                }
            )
            if used_for_grouping:
                union_find.union(source_id, target_id)

    component_members: dict[int, list[int]] = defaultdict(list)
    for anime_id in ids:
        component_members[union_find.find(anime_id)].append(anime_id)

    root_to_group = {
        root: f"series_{min(members)}"
        for root, members in component_members.items()
    }
    id_to_group = {
        anime_id: root_to_group[union_find.find(anime_id)]
        for anime_id in ids
    }
    return id_to_group, relation_edges


def group_rows(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["SeriesGroup"]].append(row)
    return groups


def genre_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    return {genre: sum(int(row[genre]) for row in rows) for genre in GENRE_COLS}


def split_score(
    splits: dict[str, list[dict[str, str]]],
    total_rows: int,
    total_genres: dict[str, int],
    ratios: dict[str, float],
) -> float:
    size_score = 0.0
    genre_score = 0.0

    for split_name, rows in splits.items():
        actual_ratio = len(rows) / total_rows
        size_score += abs(actual_ratio - ratios[split_name])

        split_genres = genre_counts(rows)
        for genre in GENRE_COLS:
            if total_genres[genre] == 0:
                continue
            actual_genre_ratio = split_genres[genre] / total_genres[genre]
            genre_score += abs(actual_genre_ratio - ratios[split_name])

    return size_score * 5.0 + genre_score


def make_candidate_split(
    groups: dict[str, list[dict[str, str]]],
    group_names: list[str],
    validation_target: int,
    test_target: int,
) -> dict[str, list[dict[str, str]]]:
    splits = {"train": [], "validation": [], "test": []}

    for group_name in group_names:
        group = groups[group_name]
        if len(splits["validation"]) < validation_target:
            split_name = "validation"
        elif len(splits["test"]) < test_target:
            split_name = "test"
        else:
            split_name = "train"
        splits[split_name].extend(group)

    return splits


def search_group_split(
    rows: list[dict[str, str]],
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
    split_trials: int,
) -> dict[str, list[dict[str, str]]]:
    ratio_sum = train_ratio + validation_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-9:
        raise ValueError(f"split ratios must sum to 1.0, got {ratio_sum}")

    groups = group_rows(rows)
    group_names = list(groups)
    total_rows = len(rows)
    validation_target = round(total_rows * validation_ratio)
    test_target = round(total_rows * test_ratio)
    total_genres = genre_counts(rows)
    ratios = {"train": train_ratio, "validation": validation_ratio, "test": test_ratio}

    best_split = None
    best_score = float("inf")

    rng = random.Random(seed)
    for _ in range(split_trials):
        candidate_names = group_names[:]
        rng.shuffle(candidate_names)
        candidate = make_candidate_split(groups, candidate_names, validation_target, test_target)
        candidate_score = split_score(candidate, total_rows, total_genres, ratios)
        if candidate_score < best_score:
            best_split = candidate
            best_score = candidate_score

    if best_split is None:
        raise RuntimeError("failed to build a split")
    return best_split


def assert_no_group_leakage(splits: dict[str, list[dict[str, str]]]) -> None:
    split_groups = {
        split_name: {row["SeriesGroup"] for row in rows}
        for split_name, rows in splits.items()
    }
    pairs = [("train", "validation"), ("train", "test"), ("validation", "test")]
    for left, right in pairs:
        overlap = split_groups[left] & split_groups[right]
        if overlap:
            examples = sorted(overlap)[:10]
            raise AssertionError(f"group leakage between {left} and {right}: {examples}")


def split_summary(
    rows: list[dict[str, str]],
    splits: dict[str, list[dict[str, str]]],
    relation_edges: list[dict[str, Any]],
) -> dict[str, Any]:
    groups = group_rows(rows)
    group_sizes = sorted(
        (
            {
                "SeriesGroup": group_name,
                "rows": len(group),
                "ids": [int(row["ID"]) for row in group],
                "titles": [row["Title"] for row in group[:8]],
            }
            for group_name, group in groups.items()
        ),
        key=lambda item: item["rows"],
        reverse=True,
    )

    total_genres = genre_counts(rows)
    split_items = {}
    for split_name, split_rows in splits.items():
        split_genres = genre_counts(split_rows)
        split_items[split_name] = {
            "rows": len(split_rows),
            "row_ratio": len(split_rows) / len(rows),
            "series_groups": len({row["SeriesGroup"] for row in split_rows}),
            "genre_counts": split_genres,
            "genre_ratio_of_total": {
                genre: (split_genres[genre] / total_genres[genre] if total_genres[genre] else 0.0)
                for genre in GENRE_COLS
            },
        }

    return {
        "input_rows": len(rows),
        "series_groups": len(groups),
        "multi_item_series_groups": sum(1 for group in groups.values() if len(group) > 1),
        "largest_series_groups": group_sizes[:20],
        "relation_edges_total": len(relation_edges),
        "relation_edges_inside_dataset": sum(1 for edge in relation_edges if edge["target_in_dataset"]),
        "relation_edges_used_for_grouping": sum(1 for edge in relation_edges if edge["used_for_grouping"]),
        "splits": split_items,
        "leakage_check": "passed",
    }


def write_summary_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# AniList Relations Group Split Summary",
        "",
        f"- Input rows: {summary['input_rows']}",
        f"- Series groups: {summary['series_groups']}",
        f"- Multi-item series groups: {summary['multi_item_series_groups']}",
        f"- Relation edges total: {summary['relation_edges_total']}",
        f"- Relation edges inside dataset: {summary['relation_edges_inside_dataset']}",
        f"- Relation edges used for grouping: {summary['relation_edges_used_for_grouping']}",
        f"- Leakage check: {summary['leakage_check']}",
        "",
        "## Split Sizes",
        "",
        "| split | rows | row ratio | series groups |",
        "| --- | ---: | ---: | ---: |",
    ]

    for split_name in ["train", "validation", "test"]:
        item = summary["splits"][split_name]
        lines.append(
            f"| {split_name} | {item['rows']} | {item['row_ratio']:.4f} | {item['series_groups']} |"
        )

    lines.extend(
        [
            "",
            "## Largest Series Groups",
            "",
            "| SeriesGroup | rows | example titles |",
            "| --- | ---: | --- |",
        ]
    )
    for group in summary["largest_series_groups"][:10]:
        titles = " / ".join(group["titles"])
        lines.append(f"| {group['SeriesGroup']} | {group['rows']} | {titles} |")

    lines.extend(
        [
            "",
            "## Genre Ratio Of Total",
            "",
            "Each value means what fraction of all rows with that genre ended up in the split.",
            "",
            "| genre | train | validation | test |",
            "| --- | ---: | ---: | ---: |",
        ]
    )

    for genre in GENRE_COLS:
        values = [
            summary["splits"][split_name]["genre_ratio_of_total"][genre]
            for split_name in ["train", "validation", "test"]
        ]
        lines.append(f"| {genre} | {values[0]:.4f} | {values[1]:.4f} | {values[2]:.4f} |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(args.input)
    ids = [int(row["ID"]) for row in rows]

    cache_path = args.output_dir / "anilist_relations_cache.json"
    relations = fetch_relations(
        ids=ids,
        cache_path=cache_path,
        batch_size=args.batch_size,
        request_interval=args.request_interval,
        max_retries=args.max_retries,
        refresh_cache=args.refresh_cache,
    )

    relation_types_text = args.relation_types.strip()
    allowed_relation_types = (
        None
        if relation_types_text.upper() == "ALL"
        else {value.strip().upper() for value in relation_types_text.split(",") if value.strip()}
    )
    if allowed_relation_types == set():
        raise ValueError("--relation-types must not be empty")

    id_to_group, relation_edges = build_series_groups(rows, relations, allowed_relation_types)
    for row in rows:
        row["SeriesGroup"] = id_to_group[int(row["ID"])]

    splits = search_group_split(
        rows=rows,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        split_trials=args.split_trials,
    )
    assert_no_group_leakage(splits)

    with_group_fieldnames = ["ID", "Title", "ImageUrl", *GENRE_COLS, "SeriesGroup"]
    training_fieldnames = ["ID", "ImageUrl", *GENRE_COLS, "SeriesGroup"]
    relation_fieldnames = [
        "source_id",
        "target_id",
        "relation_type",
        "target_in_dataset",
        "used_for_grouping",
    ]

    write_rows(args.output_dir / "preprocessed_with_series_group.csv", rows, with_group_fieldnames)
    write_rows(args.output_dir / "training_data_grouped.csv", splits["train"], training_fieldnames)
    write_rows(args.output_dir / "validation_data_grouped.csv", splits["validation"], training_fieldnames)
    write_rows(args.output_dir / "test_data_grouped.csv", splits["test"], training_fieldnames)
    write_rows(args.output_dir / "anilist_relation_edges.csv", relation_edges, relation_fieldnames)

    summary = split_summary(rows, splits, relation_edges)
    write_json(args.output_dir / "split_summary.json", summary)
    write_summary_markdown(args.output_dir / "split_summary.md", summary)

    print(f"done: generated files in {args.output_dir}")
    print(
        "split rows: "
        f"train={len(splits['train'])}, "
        f"validation={len(splits['validation'])}, "
        f"test={len(splits['test'])}"
    )


if __name__ == "__main__":
    main()
