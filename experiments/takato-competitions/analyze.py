"""Calculate average precision for each genre with an exp_run checkpoint."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import average_precision_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.append(str(EXPERIMENT_DIR))

from exp_run import (
    DEFAULT_CONFIG,
    AnimeDataset,
    CTranModel,
    GENRE_COLS,
    build_transforms,
    resolve_device,
)


def parse_args() -> argparse.Namespace:
    default_checkpoint = (
        EXPERIMENT_DIR
        / DEFAULT_CONFIG["output_dir"]
        / DEFAULT_CONFIG["best_model_name"]
    )
    parser = argparse.ArgumentParser(
        description="Calculate validation AP for each genre using an exp_run model."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=default_checkpoint,
        help="Path to a checkpoint saved by exp_run.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to <checkpoint stem>_genre_ap.csv.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device override such as cpu, cuda, or mps.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold used for Precision and Recall.",
    )
    return parser.parse_args()


def load_validation_dataframe() -> pd.DataFrame:
    path = (
        PROJECT_ROOT
        / "data"
        / "series_split_outputs"
        / "validation_data_grouped.csv"
    )
    if not path.exists():
        raise FileNotFoundError(f"Validation data was not found: {path}")
    return pd.read_csv(path)


def load_model(
    checkpoint: dict,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[CTranModel, dict, int | None, float | None]:
    if "state_dict" not in checkpoint:
        raise ValueError(
            f"Checkpoint does not contain an exp_run state_dict: {checkpoint_path}"
        )

    config = {**DEFAULT_CONFIG, **checkpoint.get("config", {})}
    model = CTranModel(
        num_labels=len(GENRE_COLS),
        layers=int(config["layers"]),
        heads=int(config["heads"]),
        dropout=float(config["dropout"]),
        backbone_weights=None,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, config, checkpoint.get("epoch"), checkpoint.get("mAP")


def predict_validation(
    model: CTranModel,
    dataframe: pd.DataFrame,
    config: dict,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, validation_transform = build_transforms(
        int(config["scale_size"]),
        int(config["crop_size"]),
    )
    loader = DataLoader(
        AnimeDataset(dataframe, validation_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    all_probabilities = []
    all_targets = []
    with torch.inference_mode():
        for images, targets in tqdm(loader, desc="Calculating genre AP"):
            images = images.to(device)
            label_mask = torch.full(
                (images.size(0), len(GENRE_COLS)),
                -1.0,
                device=device,
            )
            logits = model(images, label_mask)
            all_probabilities.append(torch.sigmoid(logits).cpu())
            all_targets.append(targets.cpu())

    return torch.cat(all_probabilities), torch.cat(all_targets)


def calculate_genre_ap(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
) -> pd.DataFrame:
    probabilities_np = probabilities.numpy()
    targets_np = targets.numpy().astype(int)
    predictions_np = (probabilities_np >= threshold).astype(int)
    rows = []

    for index, genre in enumerate(GENRE_COLS):
        support = int(targets_np[:, index].sum())
        predicted_positive = int(predictions_np[:, index].sum())
        ap = (
            float(
                average_precision_score(
                    targets_np[:, index],
                    probabilities_np[:, index],
                )
            )
            if support > 0
            else float("nan")
        )
        precision = float(
            precision_score(
                targets_np[:, index],
                predictions_np[:, index],
                zero_division=0,
            )
        )
        recall = float(
            recall_score(
                targets_np[:, index],
                predictions_np[:, index],
                zero_division=0,
            )
        )
        rows.append(
            {
                "genre": genre,
                "support": support,
                "predicted_positive": predicted_positive,
                "AP": ap,
                "Precision": precision,
                "Recall": recall,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint was not found: {checkpoint_path}")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("threshold must be in the interval [0, 1]")
    output_override = args.output.resolve() if args.output is not None else None

    # dataset_utils uses project-relative image paths.
    os.chdir(PROJECT_ROOT)

    checkpoint_metadata = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    checkpoint_config = {
        **DEFAULT_CONFIG,
        **checkpoint_metadata.get("config", {}),
    }
    device = resolve_device(args.device or str(checkpoint_config["device"]))
    model, config, epoch, checkpoint_map = load_model(
        checkpoint_metadata,
        checkpoint_path,
        device,
    )

    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else int(config["batch_size"])
    )
    num_workers = (
        args.num_workers
        if args.num_workers is not None
        else int(config["num_workers"])
    )
    validation_df = load_validation_dataframe()
    probabilities, targets = predict_validation(
        model,
        validation_df,
        config,
        device,
        batch_size,
        num_workers,
    )
    genre_ap = calculate_genre_ap(probabilities, targets, args.threshold)

    output_path = (
        output_override
        if output_override is not None
        else checkpoint_path.with_name(f"{checkpoint_path.stem}_genre_ap.csv")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    genre_ap.to_csv(output_path, index=False)

    valid_ap = genre_ap["AP"].dropna()
    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"Epoch: {epoch} | checkpoint mAP: {checkpoint_map}")
    print(f"Device: {device} | validation samples: {len(validation_df)}")
    print(f"Precision/Recall threshold: {args.threshold}")
    print()
    print(
        genre_ap.to_string(
            index=False,
            formatters={
                "AP": lambda value: f"{value:.6f}",
                "Precision": lambda value: f"{value:.6f}",
                "Recall": lambda value: f"{value:.6f}",
            },
        )
    )
    print(f"\nValidation mAP: {valid_ap.mean():.6f}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
