import argparse
import random
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.preprocessing.dataset_utils import GENRE_COLS, load_dataset, load_image

from criterion import build_criterion
from evaluate import evaluate_model
from model import ExperimentModel
from optimizer import build_optimizer
from train import train_one_epoch


class AnimeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        anime_id = row["ID"]
        image = load_image(self.df, anime_id)
        if image is None:
            raise ValueError(f"ID {anime_id} の画像が取得できませんでした。")

        if self.transform:
            image = self.transform(image)

        labels = row[GENRE_COLS].values.astype("float32")
        return image, torch.tensor(labels)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an experiment template.")
    parser.add_argument(
        "--config",
        type=Path,
        default=EXPERIMENT_DIR / "config.yaml",
        help="Path to the YAML config file.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_output_dir(config_path: Path, output_dir: str) -> Path:
    path = Path(output_dir)
    if path.is_absolute():
        return path
    return config_path.resolve().parent / path


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_name)


def build_transform(image_size: int):
    """データ拡張（Train）と検証用（Val）のトランスフォームペアを返す"""
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),  # 左右反転
            transforms.RandomAffine(
                degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)
            ),  # 微小な回転・拡大縮小
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2
            ),  # 明るさや色のゆらぎ
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return train_transform, val_transform



def maybe_limit_dataframe(df: pd.DataFrame, max_samples: int | None) -> pd.DataFrame:
    if max_samples is None:
        return df
    return df.head(max_samples).copy()


def save_checkpoint(model: torch.nn.Module, path: Path) -> None:
    raw_model = getattr(model, "_orig_mod", model)
    state = {key: value.cpu() for key, value in raw_model.state_dict().items()}
    torch.save(state, path)


def configured_seeds(config: dict) -> list[int]:
    seeds = config.get("seeds")
    if seeds:
        return [int(seed) for seed in seeds]
    return [int(config["seed"])]


def metric_improved(value: float, best: float, mode: str, min_delta: float) -> bool:
    if mode == "min":
        return value < best - min_delta
    return value > best + min_delta


def initial_best(mode: str) -> float:
    if mode == "min":
        return float("inf")
    return float("-inf")


def should_stop_early(epoch: int, epochs_without_improvement: int, early_stopping: dict) -> bool:
    if not bool(early_stopping.get("enabled", False)):
        return False
    if epoch < int(early_stopping.get("min_epochs", 0)):
        return False
    return epochs_without_improvement >= int(early_stopping.get("patience", 0))


def output_dir_for_seed(config_path: Path, config: dict, seed: int, multi_seed: bool) -> Path:
    base_output_dir = resolve_output_dir(config_path, str(config["output_dir"]))
    if multi_seed:
        return base_output_dir / f"seed_{seed}"
    return base_output_dir


def run_single_seed(config_path: Path, base_config: dict, seed: int, multi_seed: bool) -> dict:
    config = dict(base_config)
    config["seed"] = int(seed)

    set_seed(int(config["seed"]))
    device = resolve_device(str(config["device"]))
    output_dir = output_dir_for_seed(config_path, config, seed, multi_seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Seed {seed} ===")
    print(f"Using device: {device}")
    print("Loading data...")
    train_df, val_df, _ = load_dataset()
    train_df = maybe_limit_dataframe(train_df, config.get("max_train_samples"))
    val_df = maybe_limit_dataframe(val_df, config.get("max_val_samples"))

    # train用（データ拡張あり）とval用（データ拡張なし）を分けて受け取る
    train_transform, val_transform = build_transform(int(config["image_size"]))
    train_dataset = AnimeDataset(train_df, transform=train_transform)  # データ拡張あり
    val_dataset = AnimeDataset(val_df, transform=val_transform)        # データ拡張なし

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
    )

    print("Initializing model...")
    model = ExperimentModel(num_classes=len(GENRE_COLS)).to(device)
    if bool(config["compile"]) and device.type != "mps":
        model = torch.compile(model)

    criterion = build_criterion(config)
    optimizer = build_optimizer(model, float(config["learning_rate"]))

    early_stopping = config.get("early_stopping") or {}
    monitor = str(early_stopping.get("monitor", "mAP"))
    mode = str(early_stopping.get("mode", "max"))
    min_delta = float(early_stopping.get("min_delta", 0.0))
    best_monitor_value = initial_best(mode)
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_early = False
    history = []

    print("Starting training loop...")
    for epoch in range(1, int(config["epochs"]) + 1):
        print(f"\n--- Epoch {epoch}/{config['epochs']} ---")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate_model(model, val_loader, criterion, device)

        row = {
            "epoch": epoch,
            "seed": int(seed),
            "train_loss": float(train_loss),
            "val_loss": val_metrics["val_loss"],
            "macro_f1": val_metrics["macro_f1"],
            "samples_f1": val_metrics["samples_f1"],
            "hamming_loss": val_metrics["hamming_loss"],
            "mAP": val_metrics["mAP"],
        }
        history.append(row)

        print(
            f"Train Loss: {row['train_loss']:.4f} | "
            f"Val Loss: {row['val_loss']:.4f}"
        )
        print(
            f"Macro F1: {row['macro_f1']:.4f} | "
            f"Samples F1: {row['samples_f1']:.4f} | "
            f"Hamming Loss: {row['hamming_loss']:.4f} | "
            f"mAP: {row['mAP']:.4f}"
        )

        monitor_value = float(row[monitor])
        if metric_improved(monitor_value, best_monitor_value, mode, min_delta):
            best_monitor_value = monitor_value
            best_epoch = epoch
            epochs_without_improvement = 0
            checkpoint_path = output_dir / str(config["best_model_name"])
            save_checkpoint(model, checkpoint_path)
            print(f"Saved best model to {checkpoint_path} ({monitor}={monitor_value:.4f})")
        else:
            epochs_without_improvement += 1

        if should_stop_early(epoch, epochs_without_improvement, early_stopping):
            stopped_early = True
            print(
                f"Early stopping at epoch {epoch}: no {monitor} improvement "
                f"for {epochs_without_improvement} epochs. Best epoch: {best_epoch}."
            )
            break

    metrics_path = output_dir / str(config["metrics_name"])
    pd.DataFrame(history).to_csv(metrics_path, index=False)
    print(f"Saved metrics to {metrics_path}")

    return {
        "seed": int(seed),
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
        "best_epoch": int(best_epoch),
        "best_monitor": monitor,
        "best_monitor_value": float(best_monitor_value),
        "epochs_ran": int(len(history)),
        "stopped_early": bool(stopped_early),
    }


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    seeds = configured_seeds(config)
    multi_seed = len(seeds) > 1

    summaries = []
    for seed in seeds:
        summaries.append(run_single_seed(config_path, config, seed, multi_seed))

    output_dir = resolve_output_dir(config_path, str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "seed_training_summary.csv"
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    print(f"Saved seed training summary to {summary_path}")


if __name__ == "__main__":
    main()
