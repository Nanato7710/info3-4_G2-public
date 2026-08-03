import json
import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)
sys.path.append(PROJECT_ROOT)

from src.preprocessing.dataset_utils import GENRE_COLS, load_dataset, load_image

from diagnostics import (
    assert_trainable_parameters,
    check_dataset_overlap,
    print_dataset_summary,
    print_trainable_parameter_summary,
)
from evaluate import evaluate_model
from model import ExperimentModel
from optimizer import build_optimizer
from synclr_teacher import SynCLRTeacher
from train import train_one_epoch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def clean_label_columns(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    ジャンル列のNaNや文字列を安全に処理する。
    マルチラベル分類なので、欠損値は0として扱う。
    """

    df = df.copy()

    missing_cols = [col for col in GENRE_COLS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"{name} に存在しないジャンル列があります: {missing_cols}")

    for col in GENRE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[GENRE_COLS] = df[GENRE_COLS].fillna(0)
    df[GENRE_COLS] = (df[GENRE_COLS] > 0).astype("float32")

    nan_count = df[GENRE_COLS].isna().sum().sum()
    print(f"{name} label NaN count after clean: {nan_count}")

    if nan_count != 0:
        raise ValueError(f"{name} のラベル列にまだNaNが残っています。")

    return df


class AnimeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        anime_id = row["ID"]

        image = load_image(self.df, anime_id)

        if image is None:
            raise ValueError(f"ID: {anime_id} の画像が取得できませんでした。")

        if self.transform is not None:
            image = self.transform(image)

        labels = row[GENRE_COLS].fillna(0).values.astype("float32")
        labels = np.nan_to_num(labels, nan=0.0, posinf=1.0, neginf=0.0)
        labels = (labels > 0).astype("float32")

        return image, torch.tensor(labels, dtype=torch.float32)


def build_transforms(image_size: int = 224, use_augmentation: bool = True):
    if use_augmentation:
        train_transform = transforms.Compose(
            [
                transforms.Resize((image_size + 32, image_size + 32)),
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.85, 1.0),
                    ratio=(0.90, 1.10),
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.08,
                    contrast=0.08,
                    saturation=0.08,
                    hue=0.01,
                ),
                transforms.RandomRotation(degrees=5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
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


def save_training_plots(metrics_csv_path: str, graph_dir: str) -> None:
    if not os.path.exists(metrics_csv_path):
        return

    df = pd.read_csv(metrics_csv_path)

    if len(df) == 0:
        return

    os.makedirs(graph_dir, exist_ok=True)

    plt.figure()
    plt.plot(df["Epoch"], df["Train_Loss"], label="Train Loss")
    plt.plot(df["Epoch"], df["Val_Loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train / Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(graph_dir, "loss_curve.png"), dpi=200, bbox_inches="tight")
    plt.close()

    if {"Train_Genre_Loss", "Train_Distill_Loss"}.issubset(df.columns):
        plt.figure()
        plt.plot(df["Epoch"], df["Train_Genre_Loss"], label="Genre Loss")
        plt.plot(df["Epoch"], df["Train_Distill_Loss"], label="Distill Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Genre / SynCLR Distillation Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(
            os.path.join(graph_dir, "distill_loss_curve.png"),
            dpi=200,
            bbox_inches="tight",
        )
        plt.close()

    plt.figure()
    plt.plot(df["Epoch"], df["mAP"], label="mAP")
    plt.xlabel("Epoch")
    plt.ylabel("mAP")
    plt.title("Validation mAP")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(graph_dir, "map_curve.png"), dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(df["Epoch"], df["Macro_F1"], label="Macro F1")
    plt.plot(df["Epoch"], df["Samples_F1"], label="Samples F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.title("Validation F1")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(graph_dir, "f1_curve.png"), dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(df["Epoch"], df["Hamming_Loss"], label="Hamming Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Hamming Loss")
    plt.title("Validation Hamming Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        os.path.join(graph_dir, "hamming_loss_curve.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()


def save_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def save_checkpoint(model: nn.Module, path: str) -> None:
    raw_model = getattr(model, "_orig_mod", model)
    state = {key: value.cpu() for key, value in raw_model.state_dict().items()}
    torch.save(state, path)


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)

    if value is None:
        return default

    value = value.strip().lower()

    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False

    raise ValueError(
        f"{name} must be one of 1/0, true/false, yes/no, or on/off."
    )


def main():
    set_seed(42)

    current_dir = Path(__file__).resolve().parent
    default_synclr_checkpoint = current_dir / "checkpoints" / "synclr_vit_b16.pth"
    synclr_checkpoint_path = os.environ.get(
        "SYNCLR_CHECKPOINT_PATH",
        str(default_synclr_checkpoint),
    )
    synclr_checkpoint_exists = Path(synclr_checkpoint_path).exists()

    model_name = "convnext_tiny"
    loss_name = "BCEWithLogitsLoss"
    use_synclr_distillation = parse_bool_env(
        "USE_SYNCLR_DISTILLATION",
        synclr_checkpoint_exists,
    )
    pretrained = (
        "ImageNet DEFAULT + SynCLR teacher distillation"
        if use_synclr_distillation
        else "ImageNet DEFAULT"
    )

    num_epochs = 50
    batch_size = 48
    image_size = 224
    use_augmentation = True

    head_lr = 3e-4
    backbone_lr = 1e-5
    weight_decay = 1e-4
    patience = 10
    distill_alpha = float(os.environ.get("DISTILL_ALPHA", "0.1"))
    if not use_synclr_distillation:
        distill_alpha = 0.0

    default_experiment_name = (
        "convnext_tiny_bce_last_cnblock_synclr_teacher"
        if use_synclr_distillation
        else "convnext_tiny_bce_last_cnblock"
    )
    experiment_name = os.environ.get(
        "EXPERIMENT_NAME",
        default_experiment_name,
    )
    synclr_teacher_model = os.environ.get(
        "SYNCLR_TEACHER_MODEL",
        "vit_base_patch16_224",
    )

    save_dir = current_dir / "outputs" / experiment_name
    graph_dir = current_dir / "graph" / experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)
    save_dir = str(save_dir)
    graph_dir = str(graph_dir)

    config = {
        "model_name": model_name,
        "pretrained": pretrained,
        "loss": loss_name,
        "trainable_part": "last_cnblock_and_classifier",
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "image_size": image_size,
        "use_augmentation": use_augmentation,
        "head_lr": head_lr,
        "backbone_lr": backbone_lr,
        "weight_decay": weight_decay,
        "patience": patience,
        "distillation": (
            "synclr_teacher"
            if use_synclr_distillation
            else "none"
        ),
        "use_synclr_distillation": use_synclr_distillation,
        "distill_alpha": distill_alpha,
        "synclr_teacher_model": synclr_teacher_model,
        "synclr_checkpoint_path": synclr_checkpoint_path,
        "seed": 42,
        "best_metric": "validation_mAP",
        "save_dir": save_dir,
        "graph_dir": graph_dir,
    }

    pd.DataFrame([config]).to_csv(
        os.path.join(save_dir, "config.csv"),
        index=False,
    )
    save_json(os.path.join(save_dir, "config.json"), config)

    device = resolve_device()

    print(f"Using device: {device}")
    print("Model: ConvNeXt-Tiny")
    print(f"Image size: {image_size}")
    print(f"Augmentation: {use_augmentation}")
    print("Trainable: last CNBlock and classifier")
    print("Loss: BCEWithLogitsLoss")
    print(
        "Distillation: "
        + ("SynCLR teacher" if use_synclr_distillation else "none")
    )
    print(f"Distill alpha: {distill_alpha}")
    if use_synclr_distillation:
        print(f"SynCLR checkpoint: {synclr_checkpoint_path}")
    print(f"Save dir: {save_dir}")
    print(f"Graph dir: {graph_dir}")

    print("Loading data...")
    train_df, val_df, test_df = load_dataset()

    train_df = clean_label_columns(train_df, "Train")
    val_df = clean_label_columns(val_df, "Validation")
    test_df = clean_label_columns(test_df, "Test")

    check_dataset_overlap(train_df, val_df, test_df)
    print_dataset_summary(train_df, val_df, test_df, GENRE_COLS)

    train_transform, val_transform = build_transforms(
        image_size=image_size,
        use_augmentation=use_augmentation,
    )

    train_dataset = AnimeDataset(train_df, transform=train_transform)
    val_dataset = AnimeDataset(val_df, transform=val_transform)

    num_workers = 4 if device.type == "cuda" else 0
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    print("Initializing model...")
    model = ExperimentModel(num_classes=len(GENRE_COLS)).to(device)
    raw_model = model

    teacher_model = None
    if use_synclr_distillation:
        teacher_model = SynCLRTeacher(
            checkpoint_path=synclr_checkpoint_path,
            model_name=synclr_teacher_model,
            device=device,
        )

        if raw_model.feature_dim != teacher_model.feature_dim:
            raise ValueError(
                "Student and teacher feature dimensions do not match: "
                f"student={raw_model.feature_dim}, "
                f"teacher={teacher_model.feature_dim}"
            )

    if device.type == "cuda":
        model = torch.compile(model)
        raw_model = getattr(model, "_orig_mod", model)

    print_trainable_parameter_summary(raw_model)
    assert_trainable_parameters(
        raw_model,
        allowed_prefixes=(
            "backbone.features.7.2.",
            "backbone.classifier.",
        ),
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = build_optimizer(
        model=raw_model,
        head_lr=head_lr,
        backbone_lr=backbone_lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6,
    )

    best_map = -1.0
    best_epoch = 0
    best_val_loss = None
    best_macro_f1 = None
    best_samples_f1 = None
    best_hamming_loss = None
    no_improve_count = 0
    metrics_history = []
    metrics_csv_path = os.path.join(save_dir, "metrics.csv")

    print("Starting training loop...")

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            grad_clip=1.0,
            teacher_model=teacher_model,
            distill_alpha=distill_alpha,
        )
        train_loss = train_metrics["loss"]
        train_genre_loss = train_metrics["genre_loss"]
        train_distill_loss = train_metrics["distill_loss"]

        val_loss, macro_f1, samples_f1, h_loss, m_ap = evaluate_model(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            threshold=0.5,
        )

        scheduler.step()
        current_lr_backbone = optimizer.param_groups[0]["lr"]
        current_lr_head = optimizer.param_groups[1]["lr"]

        print(f"Train Loss : {train_loss:.4f} | Val Loss : {val_loss:.4f}")
        print(
            f"Genre Loss : {train_genre_loss:.4f} | "
            f"Distill Loss: {train_distill_loss:.4f}"
        )
        print(
            f"Macro F1   : {macro_f1:.4f} | "
            f"Samples F1: {samples_f1:.4f} | "
            f"Hamming Loss: {h_loss:.4f} | "
            f"mAP: {m_ap:.4f}"
        )
        print(f"LR backbone: {current_lr_backbone:.8f}")
        print(f"LR head    : {current_lr_head:.8f}")

        metrics_history.append(
            {
                "Epoch": epoch + 1,
                "Train_Loss": train_loss,
                "Train_Genre_Loss": train_genre_loss,
                "Train_Distill_Loss": train_distill_loss,
                "Val_Loss": val_loss,
                "Macro_F1": macro_f1,
                "Samples_F1": samples_f1,
                "Hamming_Loss": h_loss,
                "mAP": m_ap,
                "LR_backbone": current_lr_backbone,
                "LR_head": current_lr_head,
                "Distill_Alpha": distill_alpha,
            }
        )

        pd.DataFrame(metrics_history).to_csv(metrics_csv_path, index=False)
        save_training_plots(metrics_csv_path, graph_dir)

        if m_ap > best_map:
            print(f">>> mAP improved ({best_map:.4f} -> {m_ap:.4f}). Saving best model...")

            best_map = m_ap
            best_epoch = epoch + 1
            best_val_loss = val_loss
            best_macro_f1 = macro_f1
            best_samples_f1 = samples_f1
            best_hamming_loss = h_loss
            no_improve_count = 0

            save_checkpoint(model, os.path.join(save_dir, "best_model.pth"))

            best_info = {
                "best_epoch": best_epoch,
                "best_mAP": best_map,
                "best_val_loss": best_val_loss,
                "best_macro_f1": best_macro_f1,
                "best_samples_f1": best_samples_f1,
                "best_hamming_loss": best_hamming_loss,
            }

            pd.DataFrame([best_info]).to_csv(
                os.path.join(save_dir, "summary.csv"),
                index=False,
            )
            save_json(os.path.join(save_dir, "summary.json"), best_info)

        else:
            no_improve_count += 1
            print(f"No improvement count: {no_improve_count}/{patience}")

        if no_improve_count >= patience:
            print("Early stopping triggered.")
            break

    save_training_plots(metrics_csv_path, graph_dir)

    final_summary = {
        "best_epoch": best_epoch,
        "best_mAP": best_map,
        "best_val_loss": best_val_loss,
        "best_macro_f1": best_macro_f1,
        "best_samples_f1": best_samples_f1,
        "best_hamming_loss": best_hamming_loss,
        "finished_epoch": len(metrics_history),
        "save_dir": save_dir,
        "graph_dir": graph_dir,
    }

    pd.DataFrame([final_summary]).to_csv(
        os.path.join(save_dir, "summary.csv"),
        index=False,
    )
    save_json(os.path.join(save_dir, "summary.json"), final_summary)

    print("\nTraining complete.")
    print(f"Best mAP: {best_map:.4f}")
    print(f"Best epoch: {best_epoch}")
    print(f"Saved to: {save_dir}")
    print(f"Graphs saved to: {graph_dir}")


if __name__ == "__main__":
    main()
