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


def build_transform(
    image_size: int,
    use_random_horizontal_flip: bool = True,
    use_random_rotation: bool = True,
    use_color_jitter: bool = True,
):
    transform_list = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]

    if use_random_horizontal_flip:
        transform_list.append(transforms.RandomHorizontalFlip())
    if use_random_rotation:
        transform_list.append(transforms.RandomRotation(degrees = 50))
    if use_color_jitter:
        transform_list.append(
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
        )

    return transforms.Compose(transform_list)


def maybe_limit_dataframe(df: pd.DataFrame, max_samples: int | None) -> pd.DataFrame:
    if max_samples is None:
        return df
    return df.head(max_samples).copy()


def save_checkpoint(model: torch.nn.Module, path: Path) -> None:
    raw_model = getattr(model, "_orig_mod", model)
    state = {key: value.cpu() for key, value in raw_model.state_dict().items()}
    torch.save(state, path)


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)

    set_seed(int(config["seed"]))
    device = resolve_device(str(config["device"]))
    output_dir = resolve_output_dir(config_path, str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print("Loading data...")
    train_df, val_df, _ = load_dataset()
    train_df = maybe_limit_dataframe(train_df, config.get("max_train_samples"))
    val_df = maybe_limit_dataframe(val_df, config.get("max_val_samples"))

    transform = build_transform(
        int(config["image_size"]),
        use_random_horizontal_flip=bool(config.get("use_random_horizontal_flip", True)),
        use_random_rotation=bool(config.get("use_random_rotation", True)),
        use_color_jitter=bool(config.get("use_color_jitter", True)),
    )

    train_dataset = AnimeDataset(train_df, transform=transform)
    val_dataset = AnimeDataset(val_df, transform=transform)

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

    criterion = build_criterion()
    optimizer = build_optimizer(model, float(config["learning_rate"]))

    best_mAP = 0.0
    history = []

    print("Starting training loop...")
    for epoch in range(1, int(config["epochs"]) + 1):
        print(f"\n--- Epoch {epoch}/{config['epochs']} ---")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate_model(model, val_loader, criterion, device)

        row = {
            "epoch": epoch,
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

        if row["mAP"] > best_mAP:
            best_mAP = row["mAP"]
            checkpoint_path = output_dir / str(config["best_model_name"])
            save_checkpoint(model, checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")

    metrics_path = output_dir / str(config["metrics_name"])
    pd.DataFrame(history).to_csv(metrics_path, index=False)
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
