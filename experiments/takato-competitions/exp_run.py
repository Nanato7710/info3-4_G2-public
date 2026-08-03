"""Train C-Tran on the anime multi-label dataset.

The model follows QData/C-Tran: an ImageNet-pretrained ResNet-101 produces
spatial feature tokens, which are jointly encoded with learned label and label
state embeddings by a Transformer.  Label Mask Training (LMT) is enabled by
default, as in the repository's published COCO/VOC training commands.
"""

import argparse
import random
import sys
from contextlib import nullcontext
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as functional
import yaml
from sklearn.metrics import average_precision_score, f1_score, hamming_loss
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.preprocessing.dataset_utils import GENRE_COLS, load_dataset, load_image


DEFAULT_CONFIG = {
    "seed": 42,
    "device": "auto",
    "epochs": 100,
    "batch_size": 32,
    "gradient_accumulation_steps": 1,
    "learning_rate": 0.00001,
    "weight_decay": 0.0004,
    "num_workers": 2,
    "scale_size": 640,
    "crop_size": 576,
    "layers": 3,
    "heads": 4,
    "dropout": 0.1,
    "use_label_mask_training": True,
    "max_known_label_fraction": 0.75,
    "compile": False,
    "use_amp": True,
    "max_train_samples": None,
    "max_val_samples": None,
    "output_dir": "outputs",
    "best_model_name": "seed42_best_model.pth",
    "metrics_name": "seed42_metrics.csv",
    "early_stopping": {
        "monitor": "mAP",
        "mode": "max",
        "patience": 5,
        "min_delta": 0.0001,
        "min_epochs": 10,
    },
}


class AnimeDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform: transforms.Compose):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.dataframe.iloc[index]
        image = load_image(self.dataframe, row["ID"])
        if image is None:
            raise ValueError(f"ID {row['ID']} の画像が取得できませんでした。")
        labels = torch.tensor(row[GENRE_COLS].values.astype("float32"))
        return self.transform(image), labels


def initialize_module(module: nn.Module) -> None:
    """Match the initialization used by QData/C-Tran for non-backbone layers."""
    if isinstance(module, (nn.Linear, nn.Embedding)):
        bound = 1.0 / (module.weight.size(1) ** 0.5)
        module.weight.data.uniform_(-bound, bound)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.uniform_(-bound, bound)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


class CTranEncoderLayer(nn.Module):
    """The post-norm attention block in models/transformer_layers.py."""

    def __init__(self, hidden_size: int, heads: int, dropout: float):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            hidden_size, heads, dropout=dropout, batch_first=True
        )
        # The official SelfAttnLayer sets dim_feedforward to d_model, not 4*d_model.
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        attended, _ = self.self_attention(tokens, tokens, tokens, need_weights=False)
        tokens = self.norm1(tokens + self.dropout1(attended))
        feed_forward = self.linear2(self.dropout(functional.relu(self.linear1(tokens))))
        return self.norm2(tokens + self.dropout2(feed_forward))


class CTranModel(nn.Module):
    """C-Tran with the paper/repository's ResNet-101 feature extractor."""

    hidden_size = 2048

    def __init__(
        self,
        num_labels: int,
        layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
        backbone_weights: models.ResNet101_Weights | None = (
            models.ResNet101_Weights.IMAGENET1K_V1
        ),
    ):
        super().__init__()
        backbone = models.resnet101(weights=backbone_weights)
        # C-Tran takes the layer4 feature map before average pooling and the FC layer.
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.label_embedding = nn.Embedding(num_labels, self.hidden_size)
        # Index 0 is the "unknown" state; 1 and 2 are negative/positive.
        self.known_label_embedding = nn.Embedding(3, self.hidden_size, padding_idx=0)
        self.input_norm = nn.LayerNorm(self.hidden_size)
        self.encoder_layers = nn.ModuleList(
            [CTranEncoderLayer(self.hidden_size, heads, dropout) for _ in range(layers)]
        )
        self.output_linear = nn.Linear(self.hidden_size, num_labels)

        self.label_embedding.apply(initialize_module)
        self.known_label_embedding.apply(initialize_module)
        self.input_norm.apply(initialize_module)
        self.encoder_layers.apply(initialize_module)
        self.output_linear.apply(initialize_module)

    def forward(self, images: torch.Tensor, label_mask: torch.Tensor) -> torch.Tensor:
        batch_size = images.size(0)
        image_features = self.backbone(images)
        image_tokens = image_features.flatten(2).transpose(1, 2)

        label_ids = torch.arange(label_mask.size(1), device=images.device)
        label_tokens = self.label_embedding(label_ids).unsqueeze(0).expand(batch_size, -1, -1)
        # C-Tran's custom_replace(mask, 0, 1, 2): unknown=-1, negative=0, positive=1.
        state_indices = torch.where(label_mask < 0, 0, label_mask.long() + 1)
        label_tokens = label_tokens + self.known_label_embedding(state_indices)

        tokens = self.input_norm(torch.cat((image_tokens, label_tokens), dim=1))
        for layer in self.encoder_layers:
            tokens = layer(tokens)
        final_label_tokens = tokens[:, -label_mask.size(1) :, :]
        all_label_scores = self.output_linear(final_label_tokens)
        return all_label_scores.diagonal(dim1=1, dim2=2)


def build_transforms(scale_size: int, crop_size: int):
    """Use the C-Tran repository's COCO/VOC preprocessing recipe."""
    normalization = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_transform = transforms.Compose([
        transforms.Resize((scale_size, scale_size)),
        transforms.RandomChoice([
            transforms.RandomCrop(640),
            transforms.RandomCrop(576),
            transforms.RandomCrop(512),
            transforms.RandomCrop(384),
            transforms.RandomCrop(320),
        ]),
        transforms.Resize((crop_size, crop_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalization,
    ])
    validation_transform = transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        # transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalization,
    ])
    return train_transform, validation_transform


def create_label_mask(
    targets: torch.Tensor,
    use_lmt: bool,
    max_known_label_fraction: float,
    training: bool,
) -> torch.Tensor:
    """Create C-Tran's {-1 unknown, 0 negative, 1 positive} label-state input."""
    if not training or not use_lmt:
        return torch.full_like(targets, -1.0)

    mask = targets.clone()
    max_known = int(targets.size(1) * max_known_label_fraction)
    for row in range(targets.size(0)):
        number_known = int(torch.randint(0, max_known + 1, ()).item())
        unknown_indices = torch.randperm(targets.size(1))[number_known:]
        mask[row, unknown_indices] = -1
    return mask


def calculate_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    predictions = (logits > 0).int().numpy()
    logits_np = logits.numpy()
    targets_np = targets.numpy()
    valid_classes = targets_np.sum(axis=0) > 0
    m_ap = (
        average_precision_score(targets_np[:, valid_classes], logits_np[:, valid_classes], average="macro")
        if valid_classes.any()
        else 0.0
    )
    return {
        "macro_f1": float(f1_score(targets_np, predictions, average="macro", zero_division=0)),
        "samples_f1": float(f1_score(targets_np, predictions, average="samples", zero_division=0)),
        "hamming_loss": float(hamming_loss(targets_np, predictions)),
        "mAP": float(m_ap),
    }


def build_criterion(train_df: pd.DataFrame, device: torch.device) -> nn.BCEWithLogitsLoss:
    """Build the square-root class-balanced loss once before training starts."""
    labels = torch.tensor(train_df[GENRE_COLS].to_numpy(dtype="float32"))
    positive_counts = labels.sum(dim=0)
    negative_counts = labels.size(0) - positive_counts
    pos_weight = torch.sqrt(negative_counts / positive_counts.clamp_min(1e-5)).to(device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.BCEWithLogitsLoss,
    device: torch.device,
    use_lmt: bool,
    max_known_label_fraction: float,
    accumulation_steps: int,
    scaler: torch.amp.GradScaler,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    training = optimizer is not None
    model.train(training)
    amp_enabled = scaler.is_enabled()
    if training:
        optimizer.zero_grad(set_to_none=True)

    running_loss = 0.0
    all_logits, all_targets = [], []
    batches = len(dataloader)
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch_index, (images, targets) in enumerate(tqdm(dataloader, desc="Training" if training else "Evaluating", leave=False)):
            images, targets = images.to(device), targets.float().to(device)
            label_mask = create_label_mask(targets, use_lmt, max_known_label_fraction, training).to(device)
            unknown_labels = label_mask < 0
            with torch.autocast(device_type=device.type) if amp_enabled else nullcontext():
                logits = model(images, label_mask)
                elementwise_loss = criterion(logits, targets)
                # Official C-Tran uses only masked (unknown) labels for LMT, without mean reduction.
                loss = elementwise_loss[unknown_labels].sum()

            if training:
                group_start = (batch_index // accumulation_steps) * accumulation_steps
                group_size = min(accumulation_steps, batches - group_start)
                if amp_enabled:
                    scaler.scale(loss / group_size).backward()
                else:
                    (loss / group_size).backward()
                if (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == batches:
                    if amp_enabled:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()
            all_logits.append(logits.float().cpu())
            all_targets.append(targets.cpu())

    return running_loss / len(dataloader.dataset), torch.cat(all_logits), torch.cat(all_targets)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train C-Tran on the anime dataset.")
    parser.add_argument(
        "--config", type=Path, default=EXPERIMENT_DIR / "config_ctran.yaml",
        help="Path to a C-Tran YAML configuration file.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    if not path.exists():
        return DEFAULT_CONFIG.copy()

    with path.open(encoding="utf-8") as file:
        return {**DEFAULT_CONFIG, **(yaml.safe_load(file) or {})}


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    set_seed(int(config["seed"]))
    device = resolve_device(str(config["device"]))
    output_dir = Path(config["output_dir"])
    if not output_dir.is_absolute():
        output_dir = config_path.parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, validation_df, _ = load_dataset()
    if config["max_train_samples"] is not None:
        train_df = train_df.head(int(config["max_train_samples"])).copy()
    if config["max_val_samples"] is not None:
        validation_df = validation_df.head(int(config["max_val_samples"])).copy()
    train_transform, validation_transform = build_transforms(int(config["scale_size"]), int(config["crop_size"]))
    loader_options = {
        "batch_size": int(config["batch_size"]),
        "num_workers": int(config["num_workers"]),
        "pin_memory": device.type == "cuda",
        "persistent_workers": int(config["num_workers"]) > 0,
    }
    train_loader = DataLoader(AnimeDataset(train_df, train_transform), shuffle=True, **loader_options)
    validation_loader = DataLoader(AnimeDataset(validation_df, validation_transform), shuffle=False, **loader_options)
    criterion = build_criterion(train_df, device)

    model = CTranModel(len(GENRE_COLS), int(config["layers"]), int(config["heads"]), float(config["dropout"])).to(device)
    if config["compile"] and device.type != "mps":
        model = torch.compile(model)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        betas=(0.9, 0.999),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5
    )
    accumulation_steps = int(config["gradient_accumulation_steps"])
    if accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1")
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config["use_amp"]) and device.type == "cuda")

    early_stopping = config["early_stopping"]
    monitor = str(early_stopping["monitor"])
    mode = str(early_stopping["mode"])
    best_monitor_value = float("inf") if mode == "min" else float("-inf")
    best_map, best_epoch, epochs_without_improvement, history = float("-inf"), 0, 0, []
    print(f"Using device: {device}; C-Tran with {config['layers']} layers and {config['heads']} heads.")
    for epoch in range(1, int(config["epochs"]) + 1):
        print(f"\n--- Epoch {epoch}/{config['epochs']} ---")
        train_loss, _, _ = run_epoch(model, train_loader, optimizer, criterion, device, bool(config["use_label_mask_training"]), float(config["max_known_label_fraction"]), accumulation_steps, scaler)
        validation_loss, validation_logits, validation_targets = run_epoch(model, validation_loader, None, criterion, device, False, 0.0, accumulation_steps, scaler)
        metrics = calculate_metrics(validation_logits, validation_targets)
        scheduler.step(validation_loss)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": validation_loss, "learning_rate": optimizer.param_groups[0]["lr"], **metrics}
        history.append(row)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {validation_loss:.4f} | mAP: {metrics['mAP']:.4f}")

        if metrics["mAP"] > best_map:
            best_map, best_epoch = metrics["mAP"], epoch
            raw_model = getattr(model, "_orig_mod", model)
            torch.save({"epoch": epoch, "mAP": best_map, "state_dict": raw_model.state_dict(), "config": config}, output_dir / config["best_model_name"])
            print(f"Saved best model at epoch {epoch} (mAP={best_map:.4f}).")
        pd.DataFrame(history).to_csv(output_dir / config["metrics_name"], index=False)

        monitor_value = float(row[monitor])
        improved = (
            monitor_value < best_monitor_value - float(early_stopping["min_delta"])
            if mode == "min"
            else monitor_value > best_monitor_value + float(early_stopping["min_delta"])
        )
        if improved:
            best_monitor_value = monitor_value
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if (
            epoch >= int(early_stopping["min_epochs"])
            and epochs_without_improvement >= int(early_stopping["patience"])
        ):
            print(
                f"Early stopping at epoch {epoch}: no {monitor} improvement "
                f"for {epochs_without_improvement} epochs."
            )
            break

    print(f"Finished. Best validation mAP: {best_map:.4f} at epoch {best_epoch}.")


if __name__ == "__main__":
    main()
