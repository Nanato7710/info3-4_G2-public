from __future__ import annotations

import argparse
import json
import os
import re
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_ANALYSIS_DIR = EXPERIMENT_DIR / "analysis"
DEFAULT_METRICS_PATH = EXPERIMENT_DIR / "outputs" / "metrics.csv"
DEFAULT_CONFIG_PATH = EXPERIMENT_DIR / "config.yaml"
DEFAULT_OUTPUT_PATH = EXPERIMENT_DIR / "README.md"
PRIMARY_METRIC = "mAP"
PRIMARY_SPLIT = "validation"
STANDARD_METHOD = "model_threshold_0.5"


METRIC_LABELS = {
    "macro_f1": "Macro F1",
    "samples_f1": "Samples F1",
    "hamming_loss": "Hamming Loss",
    "mAP": "mAP",
    "predicted_labels_per_item": "予測ジャンル数/作品",
    "train_loss": "Train Loss",
    "val_loss": "Val Loss",
    "epoch": "Epoch",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "ap": "AP",
    "support": "件数",
    "predicted_positive": "陽性予測数",
    "threshold": "しきい値",
}

METHOD_LABELS = {
    "model_threshold_0.5": "0.5 固定",
    "always_none": "always none",
    "always_top_2_train_genres": "always top 2 train genres",
    "always_top_3_train_genres": "always top 3 train genres",
    "bernoulli_by_train_prevalence_mean_100": "Bernoulli by train prevalence",
    "train_prevalence": "train prevalence",
}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def fmt_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        if abs(value) >= 100:
            return f"{value:.3f}"
        return f"{value:.4f}"
    return str(value)


def markdown_table(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    rename: dict[str, str] | None = None,
    max_rows: int | None = None,
) -> str:
    if df.empty:
        return "_該当するデータが見つかりませんでした。_"

    table = df.copy()
    if columns is not None:
        existing_columns = [col for col in columns if col in table.columns]
        table = table[existing_columns]
    if max_rows is not None:
        table = table.head(max_rows)
    if rename:
        table = table.rename(columns=rename)

    lines = []
    headers = list(table.columns)
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(fmt_value(row[col]) for col in table.columns) + " |")
    return "\n".join(lines)


def with_method_labels(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "method" not in df.columns:
        return df
    table = df.copy()
    table["method"] = table["method"].map(lambda value: METHOD_LABELS.get(value, value))
    return table


def rel_path(path: Path, output_path: Path) -> str:
    return os.path.relpath(path, start=output_path.parent)


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def experiment_base_name(name: str) -> str:
    patterns = [
        r"([_-])seed\d+$",
        r"([_-])s\d+$",
        r"([_-])\d+$",
    ]
    for pattern in patterns:
        stripped = re.sub(pattern, "", name)
        if stripped != name:
            return stripped
    return name


def experiment_seed(experiment_dir: Path) -> Any:
    config = read_yaml(experiment_dir / "config.yaml")
    seed = config.get("seed", "")
    if seed != "":
        return seed

    match = re.search(r"(?:seed|s)?(\d+)$", experiment_dir.name)
    if match:
        return int(match.group(1))
    return ""


def row_for(df: pd.DataFrame, **conditions: str) -> pd.Series | None:
    if df.empty:
        return None
    mask = pd.Series(True, index=df.index)
    for key, value in conditions.items():
        if key not in df.columns:
            return None
        mask &= df[key] == value
    matched = df[mask]
    if matched.empty:
        return None
    return matched.iloc[0]


def metric_delta(new_row: pd.Series | None, old_row: pd.Series | None, metric: str) -> str:
    if new_row is None or old_row is None or metric not in new_row or metric not in old_row:
        return ""
    return fmt_value(float(new_row[metric]) - float(old_row[metric]))


def signed_delta(new_value: float | None, base_value: float | None) -> str:
    if new_value is None or base_value is None:
        return ""
    delta = new_value - base_value
    sign = "+" if delta >= 0 else ""
    return f"{sign}{fmt_value(delta)}"


def metric_value(row: pd.Series | None, metric: str) -> float | None:
    if row is None or metric not in row or pd.isna(row[metric]):
        return None
    return float(row[metric])


def resolve_experiment_output_dir(config_path: Path) -> Path:
    config = read_yaml(config_path)
    output_dir = Path(str(config.get("output_dir", "outputs")))
    if output_dir.is_absolute():
        return output_dir
    return config_path.resolve().parent / output_dir


def discover_metrics_paths(metrics_path: Path, config_path: Path) -> list[Path]:
    if metrics_path.exists():
        return [metrics_path]

    output_dir = resolve_experiment_output_dir(config_path)
    single_seed_path = output_dir / metrics_path.name
    if single_seed_path.exists():
        return [single_seed_path]
    return sorted(
        path
        for path in output_dir.glob(f"seed_*/{metrics_path.name}")
        if path.is_file()
    )


def collect_epoch_logs(metrics_path: Path, config_path: Path) -> pd.DataFrame:
    paths = discover_metrics_paths(metrics_path, config_path)
    if len(paths) == 1 and not paths[0].parent.name.startswith("seed_"):
        metrics = read_csv(paths[0])
        if not metrics.empty and "seed" not in metrics.columns:
            config = read_yaml(config_path)
            metrics.insert(0, "seed", config.get("seed", ""))
        return metrics

    frames = []
    for path in paths:
        metrics = read_csv(path)
        if metrics.empty:
            continue
        seed_text = path.parent.name.removeprefix("seed_")
        try:
            seed: Any = int(seed_text)
        except ValueError:
            seed = seed_text
        if "seed" not in metrics.columns:
            metrics.insert(0, "seed", seed)
        frames.append(metrics)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summarize_overall(overall_df: pd.DataFrame) -> list[str]:
    fixed = row_for(overall_df, split=PRIMARY_SPLIT, method=STANDARD_METHOD)
    bullets: list[str] = []
    if fixed is not None:
        bullets.append(
            "主評価指標の validation mAP は "
            f"{fmt_value(fixed[PRIMARY_METRIC])} です"
            "（標準比較: validation split, 0.5 固定しきい値）。"
        )
        bullets.append(
            "補助指標は "
            f"Macro F1={fmt_value(fixed['macro_f1'])}, "
            f"Samples F1={fmt_value(fixed['samples_f1'])}, "
            f"Hamming Loss={fmt_value(fixed['hamming_loss'])} です。"
        )
        bullets.append("閾値最適化は行っていません。実験比較の主指標は、閾値に依存しない validation mAP です。")
    return bullets


def comparison_targets(config: dict[str, Any], args: argparse.Namespace) -> tuple[str | None, list[str]]:
    comparison = config.get("comparison") or {}
    primary = args.compare_to if args.compare_to is not None else comparison.get("primary")
    references = args.reference if args.reference is not None else comparison.get("references", [])
    if isinstance(references, str):
        references = [references]

    unique_references = []
    for name in references:
        if name and name != primary and name not in unique_references:
            unique_references.append(name)
    return primary or None, unique_references


def select_comparison_row(overall: pd.DataFrame) -> pd.Series | None:
    row = row_for(overall, split=PRIMARY_SPLIT, method=STANDARD_METHOD)
    if row is not None:
        return row
    if "split" not in overall.columns:
        return None
    validation_rows = overall[overall["split"] == PRIMARY_SPLIT]
    if validation_rows.empty or PRIMARY_METRIC not in validation_rows.columns:
        return None
    # A comparison experiment must expose one canonical validation row. Picking
    # the best row here would silently optimize the comparator after the fact.
    return validation_rows.iloc[0]


def collect_experiment_comparison(
    current_experiment: str,
    current_overall: pd.DataFrame,
    primary: str | None,
    references: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    rows = []
    missing = []
    targets = [(current_experiment, "今回")]
    if primary and primary != current_experiment:
        targets.append((primary, "主比較"))
    targets.extend((name, "参考") for name in references if name != current_experiment)

    current_map = None
    for experiment_name, role in targets:
        if experiment_name == current_experiment:
            overall = current_overall
        else:
            path = PROJECT_ROOT / "experiments" / experiment_name / "analysis" / "overall_model_metrics.csv"
            overall = read_csv(path)
        row = select_comparison_row(overall)
        if row is None:
            missing.append(experiment_name)
            continue
        map_value = metric_value(row, PRIMARY_METRIC)
        if role == "今回":
            current_map = map_value
        rows.append(
            {
                "実験": experiment_name,
                "役割": role,
                "method": row.get("method", ""),
                "validation mAP": map_value,
                "mAP 標準偏差": row.get("mAP_std", ""),
                "Macro F1": row.get("macro_f1", ""),
                "Samples F1": row.get("samples_f1", ""),
                "Hamming Loss": row.get("hamming_loss", ""),
                "予測ジャンル数/作品": row.get("predicted_labels_per_item", ""),
            }
        )

    comparison = with_method_labels(pd.DataFrame(rows))
    if not comparison.empty:
        comparison["今回との差"] = comparison["validation mAP"].map(
            lambda value: "" if pd.isna(value) or current_map is None else signed_delta(current_map, float(value))
        )
        comparison.loc[comparison["役割"] == "今回", "今回との差"] = "-"
    return comparison, missing


def collect_seed_runs(analysis_dir: Path, group_name: str, config_path: Path) -> pd.DataFrame:
    seed_metrics_path = analysis_dir / "seed_overall_model_metrics.csv"
    if seed_metrics_path.exists():
        seed_metrics = read_csv(seed_metrics_path)
        if not seed_metrics.empty:
            training_summary_path = resolve_experiment_output_dir(config_path) / "seed_training_summary.csv"
            training_summary = read_csv(training_summary_path)
            rows = []
            for _, row in seed_metrics.iterrows():
                seed = row.get("seed", "")
                training_row = row_for(training_summary, seed=seed) if not training_summary.empty else None
                rows.append(
                    {
                        "実験": EXPERIMENT_DIR.name,
                        "seed": seed,
                        "best epoch": training_row.get("best_epoch", "") if training_row is not None else "",
                        "epochs ran": training_row.get("epochs_ran", "") if training_row is not None else "",
                        "early stopped": training_row.get("stopped_early", "") if training_row is not None else "",
                        "validation mAP": row.get("mAP", ""),
                        "Macro F1": row.get("macro_f1", ""),
                        "Samples F1": row.get("samples_f1", ""),
                        "Hamming Loss": row.get("hamming_loss", ""),
                        "予測ジャンル数/作品": row.get("predicted_labels_per_item", ""),
                    }
                )
            return pd.DataFrame(rows).sort_values(["seed", "実験"]).reset_index(drop=True)

    rows = []
    for experiment_dir in sorted((PROJECT_ROOT / "experiments").iterdir()):
        if not experiment_dir.is_dir():
            continue
        if experiment_base_name(experiment_dir.name) != group_name:
            continue

        overall_path = experiment_dir / "analysis" / "overall_model_metrics.csv"
        if not overall_path.exists():
            continue
        overall = read_csv(overall_path)
        row = row_for(overall, split=PRIMARY_SPLIT, method=STANDARD_METHOD)
        if row is None:
            continue

        rows.append(
            {
                "実験": experiment_dir.name,
                "seed": experiment_seed(experiment_dir),
                "best epoch": "",
                "epochs ran": "",
                "early stopped": "",
                "validation mAP": row.get("mAP", ""),
                "Macro F1": row.get("macro_f1", ""),
                "Samples F1": row.get("samples_f1", ""),
                "Hamming Loss": row.get("hamming_loss", ""),
                "予測ジャンル数/作品": row.get("predicted_labels_per_item", ""),
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["seed", "実験"]).reset_index(drop=True)


def summarize_seed_runs(seed_runs: pd.DataFrame) -> pd.DataFrame:
    if seed_runs.empty:
        return pd.DataFrame()

    metric_columns = ["validation mAP", "Macro F1", "Samples F1", "Hamming Loss", "予測ジャンル数/作品"]
    rows = []
    for metric in metric_columns:
        values = pd.to_numeric(seed_runs[metric], errors="coerce").dropna()
        if values.empty:
            continue
        rows.append(
            {
                "指標": metric,
                "runs": int(values.size),
                "平均": values.mean(),
                "標準偏差": values.std(ddof=1) if values.size > 1 else 0.0,
                "最小": values.min(),
                "最大": values.max(),
            }
        )
    return pd.DataFrame(rows)


def build_genre_diff(analysis_dir: Path, compare_to: str | None) -> pd.DataFrame:
    if not compare_to:
        return pd.DataFrame()
    current_path = analysis_dir / "genre_metrics_validation_threshold_0.5.csv"
    base_path = PROJECT_ROOT / "experiments" / compare_to / "analysis" / "genre_metrics_validation_threshold_0.5.csv"
    current = read_csv(current_path)
    baseline = read_csv(base_path)
    if current.empty or baseline.empty:
        return pd.DataFrame()

    merged = baseline.merge(current, on="genre", suffixes=("_base", "_current"))
    if merged.empty:
        return pd.DataFrame()

    merged["AP差分"] = merged["ap_current"] - merged["ap_base"]
    merged["F1差分"] = merged["f1_current"] - merged["f1_base"]
    merged["Recall差分"] = merged["recall_current"] - merged["recall_base"]
    merged["Precision差分"] = merged["precision_current"] - merged["precision_base"]
    return merged.sort_values("AP差分", ascending=False)


def comparison_summary(comparison_df: pd.DataFrame, primary: str | None, missing: list[str]) -> list[str]:
    if comparison_df.empty:
        return ["この実験の validation 比較結果が見つかりませんでした。"]

    current = comparison_df[comparison_df["役割"] == "今回"]
    if current.empty:
        return ["この実験の比較行が見つかりませんでした。"]

    current_row = current.iloc[0]
    bullets = [
        f"この実験の validation mAP は {fmt_value(current_row['validation mAP'])} です。",
    ]
    if primary:
        primary_rows = comparison_df[comparison_df["実験"] == primary]
        if not primary_rows.empty:
            primary_row = primary_rows.iloc[0]
            bullets.append(
                f"主比較 `{primary}` の validation mAP は {fmt_value(primary_row['validation mAP'])} で、"
                f"今回との差は {primary_row['今回との差']} です。"
            )
    if missing:
        bullets.append(f"比較結果を読めなかった実験: {', '.join(f'`{name}`' for name in missing)}。")
    return bullets


def seed_summary_text(seed_runs: pd.DataFrame, seed_summary: pd.DataFrame) -> list[str]:
    if seed_runs.empty:
        return ["同じ実験グループの seed 別分析結果が見つかりませんでした。"]
    if len(seed_runs) == 1:
        return ["同じ実験グループで分析済みの seed は 1 件です。複数 seed を追加すると平均と標準偏差で安定性を確認できます。"]

    map_row = row_for(seed_summary, 指標="validation mAP")
    if map_row is None:
        return [f"同じ実験グループで {len(seed_runs)} 件の seed 結果を集計しました。"]
    return [
        f"同じ実験グループで {len(seed_runs)} 件の seed 結果を集計しました。",
        f"validation mAP は平均 {fmt_value(map_row['平均'])}、標準偏差 {fmt_value(map_row['標準偏差'])} です。",
    ]


def best_epoch_rows(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame()

    groups: list[tuple[Any, pd.DataFrame]]
    if "seed" in metrics_df.columns:
        groups = list(metrics_df.groupby("seed", sort=True))
    else:
        groups = [("", metrics_df)]

    rows = []
    for seed, group in groups:
        candidates: list[tuple[str, pd.Series]] = []
        if "epoch" in group.columns:
            candidates.append(("最終", group.iloc[-1]))
        if "val_loss" in group.columns:
            candidates.append(("Val Loss 最小", group.loc[group["val_loss"].idxmin()]))
        for metric in ["mAP", "macro_f1", "samples_f1"]:
            if metric in group.columns:
                candidates.append((f"{METRIC_LABELS[metric]} 最大", group.loc[group[metric].idxmax()]))

        seen = set()
        for label, row in candidates:
            key = (label, int(row["epoch"]) if "epoch" in row else len(rows))
            if key in seen:
                continue
            seen.add(key)
            item = {"seed": seed, "観点": label}
            for col in ["epoch", "train_loss", "val_loss", "macro_f1", "samples_f1", "hamming_loss", "mAP"]:
                if col in row:
                    item[METRIC_LABELS.get(col, col)] = row[col]
            rows.append(item)
    return pd.DataFrame(rows)


def build_report(args: argparse.Namespace) -> str:
    analysis_dir = Path(args.analysis_dir)
    metrics_path = Path(args.metrics)
    config_path = Path(args.config)
    output_path = Path(args.output)

    config = read_yaml(config_path)
    summary = read_json(analysis_dir / "analysis_summary.json")
    metrics_df = collect_epoch_logs(metrics_path, config_path)
    overall_df = read_csv(analysis_dir / "overall_model_metrics.csv")
    fixed_genre_df = read_csv(analysis_dir / "genre_metrics_validation_threshold_0.5.csv")

    experiment_name = EXPERIMENT_DIR.name
    seed_group = args.seed_group or experiment_base_name(experiment_name)
    primary_comparison, reference_comparisons = comparison_targets(config, args)
    learning_curve = analysis_dir / "learning_curves.png"
    output_files = summary.get("outputs") or (sorted(path.name for path in analysis_dir.glob("*")) if analysis_dir.exists() else [])

    config_rows = pd.DataFrame(
        [{"項目": key, "値": "" if value is None else value} for key, value in config.items()]
    )
    source_entries = [
        {
            "ファイル": display_path(config_path),
            "用途": "実験設定",
        }
    ]
    metrics_paths = discover_metrics_paths(metrics_path, config_path)
    for path in metrics_paths:
        seed = path.parent.name.removeprefix("seed_") if path.parent.name.startswith("seed_") else None
        purpose = f"epoch ごとの学習ログ (seed={seed})" if seed else "epoch ごとの学習ログ"
        source_entries.append({"ファイル": display_path(path), "用途": purpose})
    if not metrics_paths:
        source_entries.append({"ファイル": "未生成", "用途": "epoch ごとの学習ログ"})
    source_entries.extend(
        [
            {
                "ファイル": display_path(analysis_dir / "overall_model_metrics.csv"),
                "用途": "validation の全体指標",
            },
            {
                "ファイル": display_path(analysis_dir / "genre_metrics_validation_threshold_0.5.csv"),
                "用途": "validation のジャンル別指標",
            },
        ]
    )
    source_rows = pd.DataFrame(source_entries)

    experiment_comparison, missing_comparisons = collect_experiment_comparison(
        experiment_name,
        overall_df,
        primary_comparison,
        reference_comparisons,
    )
    seed_runs = collect_seed_runs(analysis_dir, seed_group, config_path)
    seed_summary = summarize_seed_runs(seed_runs)
    genre_diff = build_genre_diff(analysis_dir, primary_comparison)
    overall_display = with_method_labels(overall_df).rename(columns=METRIC_LABELS)
    best_epochs = best_epoch_rows(metrics_df)

    best_ap = fixed_genre_df.sort_values("ap", ascending=False).head(8) if not fixed_genre_df.empty else pd.DataFrame()
    weak_f1 = (
        fixed_genre_df.sort_values(["f1", "support"], ascending=[True, True]).head(8)
        if not fixed_genre_df.empty
        else pd.DataFrame()
    )

    lines: list[str] = []
    lines.append(f"# 実験レポート: {experiment_name}")
    lines.append("")
    lines.append(f"作成日: {date.today().isoformat()}")
    lines.append("")
    lines.append("## 1. 共有用サマリ")
    lines.append("")
    lines.append("### 1.1 この実験の位置づけ")
    lines.append("")
    lines.append("- 何を改善しようとしたか: TODO")
    lines.append("- ベースラインまたは直前実験から変えたこと: TODO")
    lines.append("- 主評価指標 mAP の結果をどう判断するか: TODO")
    lines.append("- 何がダメだったか / まだ残っている問題: TODO")
    lines.append("")
    lines.append("### 1.2 自動要約")
    lines.append("")
    bullets = summarize_overall(overall_df)
    if summary.get("checkpoint_path"):
        bullets.insert(0, f"評価に使った checkpoint は `{summary['checkpoint_path']}` です。")
    bullets.extend(comparison_summary(experiment_comparison, primary_comparison, missing_comparisons))
    bullets.extend(seed_summary_text(seed_runs, seed_summary))
    if not bullets:
        bullets.append("自動要約に必要な全体指標ファイルが見つかりませんでした。")
    lines.extend(f"- {bullet}" for bullet in bullets)
    lines.append("")
    lines.append("### 1.3 採用判断")
    lines.append("")
    lines.append("- 採用判断: TODO（採用 / 条件付き採用 / 不採用 / 保留）")
    lines.append("- 判断理由: TODO")
    lines.append("- 次に試すこと: TODO")
    lines.append("")
    lines.append("## 2. 他実験との比較")
    lines.append("")
    lines.append("`config.yaml` で明示した主比較と参考実験だけを、validation mAP を中心に比較します。test split は最終モデル選定後まで使いません。")
    lines.append("")
    lines.append(
        markdown_table(
            experiment_comparison,
            columns=[
                "実験",
                "役割",
                "method",
                "validation mAP",
                "mAP 標準偏差",
                "今回との差",
                "Macro F1",
                "Samples F1",
                "Hamming Loss",
                "予測ジャンル数/作品",
            ],
        )
    )
    lines.append("")
    lines.append("### 2.1 複数 seed 集計")
    lines.append("")
    lines.append(f"seed 集計グループ: `{seed_group}`")
    lines.append("")
    lines.append(markdown_table(seed_summary))
    lines.append("")
    lines.append("#### seed 別結果")
    lines.append("")
    lines.append(markdown_table(seed_runs))
    lines.append("")
    lines.append("## 3. 実験の目的と変更")
    lines.append("")
    lines.append("### 3.1 背景")
    lines.append("")
    lines.append("TODO: ベースラインまたは前回実験にどの問題があったかを書く。")
    lines.append("")
    lines.append("### 3.2 仮説")
    lines.append("")
    lines.append("TODO: なぜ今回の変更で mAP が改善すると考えたかを書く。")
    lines.append("")
    lines.append("### 3.3 検証した変更")
    lines.append("")
    lines.append("| 種類 | 内容 | mAP 改善につながると考えた理由 |")
    lines.append("|---|---|---|")
    lines.append("| モデル / loss / augmentation / threshold など | TODO | TODO |")
    lines.append("")
    lines.append("### 3.4 比較条件")
    lines.append("")
    lines.append(f"- 主比較: `{primary_comparison}`" if primary_comparison else "- 主比較: なし")
    references_text = ", ".join(f"`{name}`" for name in reference_comparisons) or "なし"
    lines.append(f"- 参考実験: {references_text}")
    lines.append("- 変えたもの: TODO")
    lines.append("- 変えていないもの: TODO")
    lines.append("- 主評価指標: validation mAP")
    lines.append("- 補助指標: Macro F1, Samples F1, Hamming Loss, ジャンル別 AP/F1")
    lines.append("- test split: 最終モデル選定後まで使用しない")
    lines.append("")
    lines.append("### 3.5 主な設定")
    lines.append("")
    lines.append(markdown_table(config_rows))
    lines.append("")
    lines.append("### 3.6 再現コマンド")
    lines.append("")
    lines.append("```bash")
    lines.append(f"uv run python {EXPERIMENT_DIR.relative_to(PROJECT_ROOT)}/run_exp.py")
    lines.append(f"uv run python {EXPERIMENT_DIR.relative_to(PROJECT_ROOT)}/analyze.py")
    lines.append(f"uv run python {EXPERIMENT_DIR.relative_to(PROJECT_ROOT)}/make_report.py")
    lines.append("```")
    lines.append("")
    lines.append("## 4. 学習ログ")
    lines.append("")
    lines.append("### 4.1 代表 epoch")
    lines.append("")
    lines.append(markdown_table(best_epochs))
    lines.append("")
    if learning_curve.exists():
        lines.append("### 4.2 学習曲線")
        lines.append("")
        lines.append(f"![Learning curves]({rel_path(learning_curve, output_path)})")
        lines.append("")
    lines.append("### 4.3 読み取りメモ")
    lines.append("")
    lines.append("- Train Loss と Val Loss の差が開く場合は、過学習を疑う。")
    lines.append("- 主評価指標は mAP。mAP 最大 epoch と最終 epoch の差を見る。")
    lines.append("- F1 はしきい値で 0/1 にした後の補助指標。mAP が改善していても F1 が悪い場合は threshold 設計を疑う。")
    lines.append("- Hamming Loss は低いほど良いが、何も予測しないモデルでも低く見える場合がある。")
    lines.append("")
    lines.append("## 5. 全体評価")
    lines.append("")
    lines.append(markdown_table(overall_display))
    lines.append("")
    lines.append("### 5.1 mAP 中心の読み取り")
    lines.append("")
    lines.append("- validation mAP が比較対象より上がったか: TODO")
    lines.append("- validation mAP の改善幅は、偶然や seed 差より十分大きそうか: TODO")
    lines.append("- mAP は上がったが補助指標が悪化した場合、その悪化を許容できるか: TODO")
    lines.append("")
    lines.append("## 6. ジャンル別結果")
    lines.append("")
    if primary_comparison:
        lines.append(f"### 6.1 主比較 `{primary_comparison}` とのジャンル別 AP 差分")
    else:
        lines.append("### 6.1 主比較とのジャンル別 AP 差分")
    lines.append("")
    if not primary_comparison:
        lines.append("主比較が設定されていないため、ジャンル別差分は生成しません。")
        lines.append("")
    lines.append(
        markdown_table(
            genre_diff,
            columns=[
                "genre",
                "support_current",
                "ap_base",
                "ap_current",
                "AP差分",
                "f1_base",
                "f1_current",
                "F1差分",
                "recall_base",
                "recall_current",
                "Recall差分",
                "precision_base",
                "precision_current",
                "Precision差分",
            ],
            rename={
                "support_current": "件数",
                "ap_base": "比較AP",
                "ap_current": "現AP",
                "f1_base": "比較F1",
                "f1_current": "現F1",
                "recall_base": "比較Recall",
                "recall_current": "現Recall",
                "precision_base": "比較Precision",
                "precision_current": "現Precision",
            },
            max_rows=30,
        )
    )
    lines.append("")
    lines.append("### 6.2 AP が高いジャンル")
    lines.append("")
    lines.append(
        markdown_table(
            best_ap.rename(columns=METRIC_LABELS),
            columns=["genre", "件数", "陽性予測数", "Precision", "Recall", "F1", "AP"],
        )
    )
    lines.append("")
    lines.append("### 6.3 F1 が低いジャンル")
    lines.append("")
    lines.append(
        markdown_table(
            weak_f1.rename(columns=METRIC_LABELS),
            columns=["genre", "件数", "陽性予測数", "Precision", "Recall", "F1", "AP"],
        )
    )
    lines.append("")
    lines.append("### 6.4 考察の観点")
    lines.append("")
    lines.append("- AP が高いジャンルは、モデルが順位付けできているジャンル。mAP 改善に寄与している可能性が高い。")
    lines.append("- AP が低いジャンルは、特徴量・データ量・ラベルの曖昧さなどを疑う。")
    lines.append("- AP は高いのに F1 が低いジャンルは、しきい値調整で改善する可能性がある。")
    lines.append("- Precision が低いジャンルは、関係ない作品にもそのジャンルを付けすぎている。")
    lines.append("- Recall が低いジャンルは、本当はそのジャンルの作品を見逃している。")
    lines.append("- 件数が少ないジャンルは、少数の正解/不正解で指標が大きく動く。")
    lines.append("")
    lines.append("## 7. 人が書く考察")
    lines.append("")
    lines.append("### 7.1 何を改善しようとして、改善できたか")
    lines.append("")
    lines.append("TODO")
    lines.append("")
    lines.append("### 7.2 何がダメだったか / 想定と違ったか")
    lines.append("")
    lines.append("TODO")
    lines.append("")
    lines.append("### 7.3 原因仮説")
    lines.append("")
    lines.append("| 仮説ID | 観察した結果 | 原因仮説 | 次の確認方法 |")
    lines.append("|---|---|---|---|")
    lines.append("| H1 | TODO | TODO | TODO |")
    lines.append("| H2 | TODO | TODO | TODO |")
    lines.append("")
    lines.append("### 7.4 他メンバーに共有したい注意点")
    lines.append("")
    lines.append("TODO")
    lines.append("")
    lines.append("### 7.5 次に試すこと")
    lines.append("")
    lines.append("TODO")
    lines.append("")
    lines.append("## 8. 生成元ファイル")
    lines.append("")
    lines.append(markdown_table(source_rows))
    lines.append("")
    lines.append("### 8.1 analysis ディレクトリ内のファイル")
    lines.append("")
    if output_files:
        lines.extend(f"- `{name}`" for name in output_files)
    else:
        lines.append("- analysis ディレクトリが見つかりませんでした。")
    lines.append("")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a Markdown experiment report from analysis outputs.")
    parser.add_argument("--analysis-dir", default=DEFAULT_ANALYSIS_DIR, type=Path)
    parser.add_argument("--metrics", default=DEFAULT_METRICS_PATH, type=Path)
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, type=Path)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, type=Path)
    parser.add_argument("--compare-to", default=None, help="Override comparison.primary from config.yaml.")
    parser.add_argument(
        "--reference",
        action="append",
        default=None,
        help="Override comparison.references. Repeat this option to select multiple reference experiments.",
    )
    parser.add_argument(
        "--seed-group",
        default=None,
        help="Experiment group name used for multi-seed aggregation. Defaults to current experiment name with a seed suffix removed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    if output_path.exists():
        print(f"Warning: Overwriting existing report at {output_path}")
        print("If you continue, the existing file will be replaced.")
        select = input("Enter 'y' to continue, or 'n' to cancel: ").strip().lower()
        if select != "y":
            print("Operation cancelled. No changes made.")
            return
    report = build_report(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
