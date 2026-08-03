from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

OUTPUTS_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = OUTPUTS_ROOT.parent
if str(OUTPUTS_ROOT) not in sys.path:
    sys.path.insert(0, str(OUTPUTS_ROOT))

from artifact_common import (
    GENRE_NAMES,
    backup_run_directory,
    calculate_genre_ap,
    calculate_metrics,
    current_environment,
    git_value,
    load_split,
    load_yaml,
    materialize_file,
    save_predictions,
    sha256_file,
    utc_now,
    validate_prediction_arrays,
    write_json,
    write_yaml,
)

MODELS = ("baseline", "final-tri-model")
SEEDS = (42, 43, 44)
METRIC_COLUMNS = ("mAP", "macro_f1", "samples_f1", "hamming_loss")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and materialize the six saved experiment runs."
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_model_builder(model_id: str):
    path = OUTPUTS_ROOT / model_id / "model.py"
    spec = importlib.util.spec_from_file_location(
        f"frozen_{model_id.replace('-', '_')}_model",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_model


def legacy_prediction_arrays(
    legacy_path: Path,
    validation_frame: pd.DataFrame,
) -> dict[str, np.ndarray]:
    with np.load(legacy_path, allow_pickle=False) as archive:
        expected = {"val_ids", "y_val", "val_logits", "val_probs"}
        if set(archive.files) != expected:
            raise ValueError(f"{legacy_path} has unexpected keys: {archive.files}")
        legacy = {key: archive[key] for key in archive.files}
    ids = legacy["val_ids"]
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"{legacy_path} contains duplicate IDs")
    position = {str(value): index for index, value in enumerate(ids)}
    expected_ids = validation_frame["ID"].to_numpy()
    if set(position) != set(map(str, expected_ids)):
        raise ValueError(f"{legacy_path} IDs do not match validation CSV")
    indices = np.asarray([position[str(value)] for value in expected_ids])
    targets = legacy["y_val"][indices].astype(np.uint8)
    csv_targets = validation_frame[GENRE_NAMES].to_numpy(dtype=np.uint8)
    if not np.array_equal(targets, csv_targets):
        raise ValueError(f"{legacy_path} targets do not match validation CSV")
    arrays = {
        "ids": expected_ids,
        "series_groups": np.asarray(
            validation_frame["SeriesGroup"].astype(str).tolist(),
            dtype=str,
        ),
        "genre_names": np.asarray(GENRE_NAMES),
        "targets": targets,
        "logits": legacy["val_logits"][indices].astype(np.float32),
        "scores": legacy["val_probs"][indices].astype(np.float32),
    }
    validate_prediction_arrays(arrays, expected_rows=1121)
    return arrays


def matching_history_epoch(
    history: pd.DataFrame,
    prediction_metrics: dict[str, float],
) -> int:
    matches = np.ones(len(history), dtype=bool)
    for column in METRIC_COLUMNS:
        matches &= np.isclose(
            history[column].to_numpy(dtype=float),
            prediction_metrics[column],
            atol=1e-5,
            rtol=0,
        )
    matched_rows = history.loc[matches]
    if len(matched_rows) != 1:
        raise ValueError(
            f"expected one history row matching predictions, got {len(matched_rows)}"
        )
    return int(matched_rows.iloc[0]["epoch"])


def enhanced_history(
    history: pd.DataFrame,
    learning_rate: float,
) -> pd.DataFrame:
    required = {
        "epoch",
        "train_loss",
        "val_loss",
        "mAP",
        "macro_f1",
        "samples_f1",
        "hamming_loss",
    }
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"history is missing columns: {sorted(missing)}")
    result = history[list(required)].copy()
    result["learning_rate"] = float(learning_rate)
    result["epoch_seconds"] = np.nan
    return result[
        [
            "epoch",
            "train_loss",
            "val_loss",
            "mAP",
            "macro_f1",
            "samples_f1",
            "hamming_loss",
            "learning_rate",
            "epoch_seconds",
        ]
    ]


def validate_checkpoint(model_id: str, config: dict[str, Any], path: Path) -> None:
    evaluation_config = dict(config)
    evaluation_config["pretrained"] = False
    model = load_model_builder(model_id)(evaluation_config)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    if model_id == "baseline" and tuple(state["backbone.fc.weight"].shape) != (19, 512):
        raise ValueError("baseline checkpoint output head is not 19 x 512")
    if model_id == "final-tri-model" and tuple(state["head.weight"].shape) != (
        19,
        1024,
    ):
        raise ValueError("final checkpoint output head is not 19 x 1024")


def verify_config_correspondence(model_id: str, submitted: dict[str, Any]) -> None:
    source = load_yaml(PROJECT_ROOT / "experiments" / model_id / "config.yaml")
    exact_keys = (
        "epochs",
        "batch_size",
        "num_workers",
        "image_size",
        "compile",
        "early_stopping",
    )
    for key in exact_keys:
        if source[key] != submitted[key]:
            raise ValueError(f"{model_id} config mismatch for {key}")
    if float(source["learning_rate"]) != float(submitted["learning_rate"]):
        raise ValueError(f"{model_id} config mismatch for learning_rate")
    if model_id == "final-tri-model":
        expected = {"gamma_pos": 0.1, "gamma_neg": 1.0, "clip": 0.05}
        actual = submitted["criterion"]
        for key, value in expected.items():
            if float(actual[key]) != value:
                raise ValueError(f"final criterion mismatch for {key}")


def prepare_run(
    model_id: str,
    seed: int,
    validation_frame: pd.DataFrame,
    force: bool,
) -> dict[str, Any]:
    source_run = (
        PROJECT_ROOT / "experiments" / model_id / "outputs" / f"seed_{seed}"
    )
    destination = OUTPUTS_ROOT / model_id / "runs" / f"seed_{seed}"
    if (destination / "COMPLETED").exists():
        if not force:
            validation_path = destination / "validation_predictions.npz"
            try:
                with np.load(validation_path, allow_pickle=False) as archive:
                    existing = {key: archive[key] for key in archive.files}
                validate_prediction_arrays(existing, expected_rows=1121)
            except ValueError:
                repaired = legacy_prediction_arrays(
                    source_run / "analysis" / "model_predictions.npz",
                    validation_frame,
                )
                save_predictions(validation_path, repaired)
                completed = json.loads(
                    (destination / "COMPLETED").read_text(encoding="utf-8")
                )
                completed["validation_predictions_sha256"] = sha256_file(
                    validation_path
                )
                completed["schema_repaired_at"] = utc_now()
                write_json(destination / "COMPLETED", completed)
            metadata = json.loads(
                (destination / "run-metadata.json").read_text(encoding="utf-8")
            )
            metadata["code_snapshot"]["sha256"] = code_snapshot_hash(model_id)
            write_json(destination / "run-metadata.json", metadata)
            return {
                "model_id": model_id,
                "seed": seed,
                "best_epoch": metadata["best_epoch"],
                "validation_mAP": metadata["best_validation_mAP"],
                "checkpoint_sha256": metadata["checkpoint_sha256"],
                "status": "already_complete",
            }
        backup_run_directory(destination)
    elif destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"{destination} is non-empty")
    destination.mkdir(parents=True, exist_ok=True)

    config_path = OUTPUTS_ROOT / model_id / "config.yaml"
    config = load_yaml(config_path)
    verify_config_correspondence(model_id, config)
    resolved_config = dict(config)
    resolved_config["seed"] = seed
    write_yaml(destination / "resolved-config.yaml", resolved_config)

    checkpoint_source = source_run / "best_model.pth"
    history_source = source_run / "metrics.csv"
    legacy_predictions = source_run / "analysis" / "model_predictions.npz"
    for path in (checkpoint_source, history_source, legacy_predictions):
        if not path.is_file():
            raise FileNotFoundError(path)
    validate_checkpoint(model_id, config, checkpoint_source)

    arrays = legacy_prediction_arrays(legacy_predictions, validation_frame)
    prediction_metrics = calculate_metrics(arrays["targets"], arrays["scores"])
    history = pd.read_csv(history_source)
    best_epoch = matching_history_epoch(history, prediction_metrics)
    materialize_file(checkpoint_source, destination / "best_model.pth")
    enhanced_history(history, float(config["learning_rate"])).to_csv(
        destination / "metrics.csv",
        index=False,
    )
    save_predictions(destination / "validation_predictions.npz", arrays)
    validation_metrics = {
        **prediction_metrics,
        "threshold": 0.5,
        "row_count": len(arrays["ids"]),
        "genre_count": len(GENRE_NAMES),
        "genre_metrics": calculate_genre_ap(arrays["targets"], arrays["scores"]),
        "source": "converted from saved per-run validation prediction",
    }
    write_json(destination / "validation_metrics.json", validation_metrics)
    write_json(
        destination / "environment.json",
        {
            "purpose": "historical training environment",
            "python": None,
            "pytorch": None,
            "torchvision": None,
            "timm": None,
            "unavailable_reason": (
                "the saved run did not record package versions; current versions "
                "are stored separately as artifact_preparation_environment"
            ),
            "artifact_preparation_environment": current_environment(),
        },
    )
    checkpoint_hash = sha256_file(destination / "best_model.pth")
    metadata = {
        "model_id": model_id,
        "seed": seed,
        "started_at": None,
        "ended_at": None,
        "duration_seconds": None,
        "unavailable_timing_reason": "the saved run did not record wall-clock timestamps",
        "best_epoch": best_epoch,
        "best_validation_mAP": prediction_metrics["mAP"],
        "early_stopping": {
            "monitor": "mAP",
            "mode": "max",
            "min_delta": float(config["early_stopping"]["min_delta"]),
            "stopped_early": len(history) < int(config["epochs"]),
            "stop_reason": (
                "saved history ended before configured epochs after patience was exhausted"
                if len(history) < int(config["epochs"])
                else "configured epochs completed"
            ),
        },
        "versions": {
            "python": None,
            "pytorch": None,
            "torchvision": None,
            "timm": None,
            "unavailable_reason": "not recorded by the historical training run",
        },
        "git_commit": None,
        "dirty_worktree": None,
        "historical_git_reason": "not recorded by the historical training run",
        "code_snapshot": {
            "path": f"outputs/{model_id}",
            "sha256": code_snapshot_hash(model_id),
            "correspondence_evidence": [
                "checkpoint loads strictly into the submitted model",
                "submitted config matches the saved experiment config",
                "saved validation predictions match exactly one history epoch",
            ],
            "limitation": "the exact historical Git commit was not recorded",
        },
        "resolved_config_sha256": sha256_file(
            destination / "resolved-config.yaml"
        ),
        "checkpoint_sha256": checkpoint_hash,
    }
    write_json(destination / "run-metadata.json", metadata)
    write_json(
        destination / "COMPLETED",
        {
            "completed_at": utc_now(),
            "integrity_checked": True,
            "test_evaluation_complete": False,
            "checkpoint_sha256": checkpoint_hash,
            "validation_predictions_sha256": sha256_file(
                destination / "validation_predictions.npz"
            ),
        },
    )
    return {
        "model_id": model_id,
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_mAP": prediction_metrics["mAP"],
        "checkpoint_sha256": checkpoint_hash,
    }


def code_snapshot_hash(model_id: str) -> str:
    import hashlib

    digest = hashlib.sha256()
    paths = [
        OUTPUTS_ROOT / "artifact_common.py",
        *sorted((OUTPUTS_ROOT / model_id).glob("*.py")),
        OUTPUTS_ROOT / model_id / "config.yaml",
    ]
    for path in paths:
        digest.update(path.relative_to(PROJECT_ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    validation_frame = load_split("validation")
    summaries = []
    for model_id in MODELS:
        for seed in SEEDS:
            summary = prepare_run(model_id, seed, validation_frame, args.force)
            summaries.append(summary)
            print(json.dumps(summary, ensure_ascii=False))
    write_json(
        OUTPUTS_ROOT / "evaluation" / "artifact-preparation-summary.json",
        {
            "created_at": utc_now(),
            "git_commit_at_preparation": git_value("rev-parse", "HEAD"),
            "runs": summaries,
        },
    )


if __name__ == "__main__":
    main()
