from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

OUTPUTS_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = OUTPUTS_ROOT.parent
if str(OUTPUTS_ROOT) not in sys.path:
    sys.path.insert(0, str(OUTPUTS_ROOT))

from artifact_common import (
    EXPECTED_SPLIT_ROWS,
    EXPECTED_TEST_SERIES_GROUPS,
    calculate_genre_ap,
    calculate_metrics,
    load_split,
    load_yaml,
    make_loader,
    predict_model,
    resolve_device,
    save_predictions,
    sha256_file,
    utc_now,
    validate_prediction_arrays,
    write_json,
)

MODELS = ("baseline", "final-tri-model")
SEEDS = (42, 43, 44)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate frozen checkpoints.")
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        required=True,
    )
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def verify_freeze_manifest() -> dict:
    manifest_path = OUTPUTS_ROOT / "freeze-manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("outputs/freeze-manifest.json is required for test")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mismatches = []
    for record in manifest["frozen_inputs"]:
        path = PROJECT_ROOT / record["path"]
        if not path.is_file():
            mismatches.append(f"missing: {record['path']}")
        elif sha256_file(path) != record["sha256"]:
            mismatches.append(f"hash mismatch: {record['path']}")
    if mismatches:
        raise ValueError("freeze manifest validation failed:\n" + "\n".join(mismatches))
    return manifest


def load_model(model_id: str, config: dict):
    path = OUTPUTS_ROOT / model_id / "model.py"
    spec = importlib.util.spec_from_file_location(
        f"evaluation_{model_id.replace('-', '_')}_model",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    evaluation_config = dict(config)
    evaluation_config["pretrained"] = False
    return module.build_model(evaluation_config)


def backup_file(path: Path) -> None:
    if not path.exists():
        return
    backup = path.with_name(f"{path.name}.backup-{utc_now().replace(':', '')}")
    path.rename(backup)
    print(f"--force: moved {path} to {backup}")


def evaluate_one(
    model_id: str,
    seed: int,
    split: str,
    device_name: str,
    batch_size_override: int | None,
    num_workers_override: int | None,
    force: bool,
) -> None:
    run_dir = OUTPUTS_ROOT / model_id / "runs" / f"seed_{seed}"
    prediction_path = run_dir / f"{split}_predictions.npz"
    metrics_path = run_dir / f"{split}_metrics.json"
    if prediction_path.exists() or metrics_path.exists():
        if not force:
            raise FileExistsError(
                f"{model_id} seed {seed} {split} already exists; pass --force"
            )
        backup_file(prediction_path)
        backup_file(metrics_path)
    config = load_yaml(run_dir / "resolved-config.yaml")
    frame = load_split(split)
    loader = make_loader(
        frame,
        image_size=int(config["image_size"]),
        training=False,
        batch_size=(
            int(batch_size_override)
            if batch_size_override is not None
            else int(config["batch_size"])
        ),
        num_workers=(
            int(num_workers_override)
            if num_workers_override is not None
            else int(config["num_workers"])
        ),
    )
    device = resolve_device(device_name)
    model = load_model(model_id, config).to(device)
    state = torch.load(
        run_dir / "best_model.pth",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state, strict=True)
    predictions = predict_model(model, loader, device)
    expected_groups = EXPECTED_TEST_SERIES_GROUPS if split == "test" else None
    validate_prediction_arrays(
        predictions,
        expected_rows=EXPECTED_SPLIT_ROWS[split],
        expected_series_groups=expected_groups,
    )
    save_predictions(prediction_path, predictions)
    metrics = {
        **calculate_metrics(predictions["targets"], predictions["scores"]),
        "threshold": 0.5,
        "row_count": len(predictions["ids"]),
        "genre_count": len(predictions["genre_names"]),
        "series_group_count": len(set(predictions["series_groups"].tolist())),
        "genre_metrics": calculate_genre_ap(
            predictions["targets"], predictions["scores"]
        ),
        "model_id": model_id,
        "seed": seed,
        "split": split,
        "created_at": utc_now(),
        "checkpoint_sha256": sha256_file(run_dir / "best_model.pth"),
        "prediction_sha256": sha256_file(prediction_path),
    }
    write_json(metrics_path, metrics)
    completed = json.loads((run_dir / "COMPLETED").read_text(encoding="utf-8"))
    completed[f"{split}_predictions_sha256"] = sha256_file(prediction_path)
    if split == "test":
        completed["test_evaluation_complete"] = True
    write_json(run_dir / "COMPLETED", completed)
    print(
        f"{model_id} seed {seed} {split}: "
        f"mAP={metrics['mAP']:.6f}, rows={metrics['row_count']}"
    )


def main() -> None:
    args = parse_args()
    if args.split == "test":
        verify_freeze_manifest()
    models = (args.model,) if args.model else MODELS
    seeds = (args.seed,) if args.seed is not None else SEEDS
    for model_id in models:
        for seed in seeds:
            evaluate_one(
                model_id,
                seed,
                args.split,
                args.device,
                args.batch_size,
                args.num_workers,
                args.force,
            )


if __name__ == "__main__":
    main()
