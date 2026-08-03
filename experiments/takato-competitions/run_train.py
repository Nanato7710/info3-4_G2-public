"""Train the takato experiment with the kazusa-baseline training recipe.

Running this file without options uses the same three seeds and hyperparameters
as ``experiments/kazusa-baseline``.  A YAML file can be supplied with
``--config`` to override those defaults.
"""

import argparse
import math
import random
import sys
from contextlib import nullcontext
from pathlib import Path

import pandas as pd
import timm
import torch
import yaml
from sklearn.metrics import average_precision_score, f1_score, hamming_loss
from torch import nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from tqdm import tqdm
import numpy as np
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.preprocessing.dataset_utils import GENRE_COLS, load_dataset, load_image


# These values intentionally match experiments/kazusa-baseline/config.yaml.
DEFAULT_CONFIG = {
    "seed": 42,
    "seeds": [42],
    "device": "auto",
    "epochs": 200,
    "early_stopping": {
        "enabled": True,
        "monitor": "mAP",
        "mode": "max",
        "patience": 10,
        "min_delta": 0.0001,
        "min_epochs": 10,
    },
    "batch_size": 32,
    "gradient_accumulation_steps": 4,
    "learning_rate": 0.0001,
    "ema_decay": 0.9997,
    "num_workers": 2,
    "image_size": 640,
    "compile": False,
    "use_amp": True,
    "max_train_samples": None,
    "max_val_samples": None,
    "output_dir": "outputs",
    "best_model_name": "bcew_best_model.pth",
    "metrics_name": "bcew_metrics.csv",
}


class AnimeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = load_image(self.df, row["ID"])
        if image is None:
            raise ValueError(f"ID {row['ID']} の画像が取得できませんでした。")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(row[GENRE_COLS].values.astype("float32"))


class ExperimentModel(nn.Module):
    def __init__(self, num_classes: int = 19):
        super().__init__()
        self.backbone = timm.create_model(
            "tresnet_v2_l.miil_in21k", num_classes=0, pretrained=True
        )
        self.head = nn.Linear(self.backbone.num_features, num_classes, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


class AsymmetricLossOptimized(nn.Module):
    """ASL used by kazusa-baseline (from the authors' official implementation)."""

    def __init__(
        self,
        gamma_neg: float = 4,
        gamma_pos: float = 0,
        clip: float = 0.05,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        anti_targets = 1 - targets
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos
        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        loss = targets * torch.log(xs_pos.clamp(min=self.eps))
        loss += anti_targets * torch.log(xs_neg.clamp(min=self.eps))
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            xs_pos = xs_pos * targets
            xs_neg = xs_neg * anti_targets
            asymmetric_weight = torch.pow(
                1 - xs_pos - xs_neg,
                self.gamma_pos * targets + self.gamma_neg * anti_targets,
            )
            loss *= asymmetric_weight
        return -loss.sum()


def build_transforms(image_size: int):
    train_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((image_size, image_size), interpolation=v2.InterpolationMode.BILINEAR),
        v2.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0.0),
        v2.RandAugment(interpolation=v2.InterpolationMode.BILINEAR),
        v2.ToDtype(torch.float32),
    ])
    val_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((image_size, image_size), interpolation=v2.InterpolationMode.BILINEAR),
        v2.ToDtype(torch.float32),
    ])
    return train_transform, val_transform


def calculate_metrics_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    predictions = (logits > 0).int().numpy()
    logits_np = logits.numpy()
    targets_np = targets.numpy()
    valid_classes = targets_np.sum(axis=0) > 0
    mean_average_precision = (
        average_precision_score(targets_np[:, valid_classes], logits_np[:, valid_classes], average="macro")
        if valid_classes.any()
        else 0.0
    )
    return {
        "macro_f1": float(f1_score(targets_np, predictions, average="macro", zero_division=0)),
        "samples_f1": float(f1_score(targets_np, predictions, average="samples", zero_division=0)),
        "hamming_loss": float(hamming_loss(targets_np, predictions)),
        "mAP": float(mean_average_precision),
    }


def train_one_epoch(model, ema_model, dataloader, optimizer, scheduler, criterion, device, accumulation_steps, scaler):
    model.train()
    running_loss = 0.0
    num_batches = len(dataloader)
    amp_enabled = scaler.is_enabled()
    optimizer.zero_grad(set_to_none=True)

    for batch_index, (inputs, targets) in enumerate(tqdm(dataloader, desc="Training", leave=False)):
        inputs, targets = inputs.to(device), targets.float().to(device)
        with torch.autocast(device_type=device.type) if amp_enabled else nullcontext():
            loss = criterion(model(inputs), targets)
        running_loss += loss.item() * inputs.size(0)

        group_start = (batch_index // accumulation_steps) * accumulation_steps
        group_size = min(accumulation_steps, num_batches - group_start)
        if amp_enabled:
            scaler.scale(loss / group_size).backward()
        else:
            (loss / group_size).backward()

        if (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == num_batches:
            if amp_enabled:
                scale_before_step = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                stepped = scaler.get_scale() >= scale_before_step
            else:
                optimizer.step()
                stepped = True
            if stepped:
                scheduler.step()
                ema_model.update_parameters(model)
            optimizer.zero_grad(set_to_none=True)
    return running_loss / len(dataloader.dataset)


@torch.inference_mode()
def evaluate_model(model, dataloader, criterion, device, use_amp: bool) -> dict[str, float]:
    model.eval()
    running_loss = 0.0
    all_logits, all_targets = [], []
    for inputs, targets in tqdm(dataloader, desc="Evaluating", leave=False):
        inputs, targets = inputs.to(device), targets.float().to(device)
        with torch.autocast(device_type=device.type) if use_amp else nullcontext():
            logits = model(inputs)
            loss = criterion(logits, targets)
        running_loss += loss.item() * inputs.size(0)
        all_logits.append(logits.float().cpu())
        all_targets.append(targets.cpu())
    metrics = calculate_metrics_from_logits(torch.cat(all_logits), torch.cat(all_targets))
    metrics["val_loss"] = running_loss / len(dataloader.dataset)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train with the kazusa-baseline recipe.")
    parser.add_argument("--config", type=Path, help="Optional YAML settings overriding the baseline defaults.")
    return parser.parse_args()


def load_config(path: Path | None) -> dict:
    config = DEFAULT_CONFIG.copy()
    config["early_stopping"] = DEFAULT_CONFIG["early_stopping"].copy()
    if path is not None:
        with path.open(encoding="utf-8") as file:
            overrides = yaml.safe_load(file) or {}
        config.update(overrides)
        config["early_stopping"] = {
            **DEFAULT_CONFIG["early_stopping"],
            **(overrides.get("early_stopping") or {}),
        }
    return config


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_criterion() -> nn.Module:
    # 1. デバイスの決定
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available()  else "cpu")

    # 2. 提供された訓練データのパスを設定
    # (リポジトリのルートからでも、実験フォルダ内からでも動くように自動フォールバック付き)
    csv_rel_path = "data/series_split_outputs/training_data_grouped.csv"
    if os.path.exists(csv_rel_path):
        csv_path = csv_rel_path
    else:
        csv_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../", csv_rel_path)
        )

    # 3. スクリーンショット通りの正確なジャンル列リスト（全19種類）
    genre_cols = [
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

    # 4. データの読み込みと pos_weight の自動計算
    print(f"📦 重み計算のためにデータを読み込んでいます: {csv_path}")
    train_df = pd.read_csv(csv_path)
    labels = train_df[genre_cols].values

    # 陰性数(0の数) / 陽性数(1の数) で各ジャンルの重みを算出
    raw_weight = (1 - labels).sum(axis=0) / (labels.sum(axis=0) + 1e-5)
    pos_weight = np.sqrt(raw_weight)
    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32).to(
        device
    )

    return nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config_dir = args.config.resolve().parent if args.config else EXPERIMENT_DIR
    output_root = Path(config["output_dir"])
    if not output_root.is_absolute():
        output_root = config_dir / output_root
    seeds = [int(seed) for seed in config.get("seeds") or [config["seed"]]]
    summaries = []

    for seed in seeds:
        set_seed(seed)
        device = resolve_device(str(config["device"]))
        output_dir = output_root / f"seed_{seed}" if len(seeds) > 1 else output_root
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Seed {seed} ===\nUsing device: {device}\nLoading data...")
        train_df, val_df, _ = load_dataset()
        if config["max_train_samples"] is not None:
            train_df = train_df.head(int(config["max_train_samples"])).copy()
        if config["max_val_samples"] is not None:
            val_df = val_df.head(int(config["max_val_samples"])).copy()
        train_transform, val_transform = build_transforms(int(config["image_size"]))
        loader_args = {
            "batch_size": int(config["batch_size"]),
            "num_workers": int(config["num_workers"]),
            "pin_memory": device.type == "cuda",
            "persistent_workers": int(config["num_workers"]) > 0,
        }
        train_loader = DataLoader(AnimeDataset(train_df, train_transform), shuffle=True, **loader_args)
        val_loader = DataLoader(AnimeDataset(val_df, val_transform), shuffle=False, **loader_args)

        print("Initializing model...")
        model = ExperimentModel(len(GENRE_COLS)).to(device)
        ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(float(config["ema_decay"])), use_buffers=True, device=device)
        if config["compile"] and device.type != "mps":
            model = torch.compile(model)
        criterion = build_criterion()
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]))
        accumulation_steps = int(config["gradient_accumulation_steps"])
        if accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be at least 1")
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(config["learning_rate"]),
            steps_per_epoch=math.ceil(len(train_loader) / accumulation_steps),
            epochs=int(config["epochs"]),
            pct_start=0.2,
        )
        amp_enabled = bool(config["use_amp"]) and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        early = config["early_stopping"]
        monitor, mode = str(early["monitor"]), str(early["mode"])
        best_value = float("inf") if mode == "min" else float("-inf")
        best_epoch = epochs_without_improvement = 0
        history = []

        for epoch in range(1, int(config["epochs"]) + 1):
            print(f"\n--- Epoch {epoch}/{config['epochs']} ---")
            train_loss = train_one_epoch(model, ema_model, train_loader, optimizer, scheduler, criterion, device, accumulation_steps, scaler)
            metrics = evaluate_model(ema_model, val_loader, criterion, device, amp_enabled)
            row = {"epoch": epoch, "seed": seed, "train_loss": train_loss, **metrics}
            history.append(row)
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {metrics['val_loss']:.4f}")
            print(f"Macro F1: {metrics['macro_f1']:.4f} | Samples F1: {metrics['samples_f1']:.4f} | Hamming Loss: {metrics['hamming_loss']:.4f} | mAP: {metrics['mAP']:.4f}")
            improved = metrics[monitor] < best_value - float(early["min_delta"]) if mode == "min" else metrics[monitor] > best_value + float(early["min_delta"])
            if improved:
                best_value, best_epoch, epochs_without_improvement = metrics[monitor], epoch, 0
                raw_ema = getattr(ema_model.module, "_orig_mod", ema_model.module)
                torch.save({key: value.cpu() for key, value in raw_ema.state_dict().items()}, output_dir / config["best_model_name"])
                print(f"Saved best model ({monitor}={best_value:.4f})")
            else:
                epochs_without_improvement += 1
            if early["enabled"] and epoch >= int(early["min_epochs"]) and epochs_without_improvement >= int(early["patience"]):
                print(f"Early stopping at epoch {epoch}; best epoch: {best_epoch}.")
                break

        pd.DataFrame(history).to_csv(output_dir / config["metrics_name"], index=False)
        summaries.append({"seed": seed, "output_dir": str(output_dir.relative_to(PROJECT_ROOT)), "best_epoch": best_epoch, "best_monitor": monitor, "best_monitor_value": best_value, "epochs_ran": len(history), "stopped_early": len(history) < int(config["epochs"])})

    output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(output_root / "seed_training_summary.csv", index=False)


if __name__ == "__main__":
    main()
