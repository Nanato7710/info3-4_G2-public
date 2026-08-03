from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from sklearn.metrics import average_precision_score, f1_score, hamming_loss
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_ROOT = Path(__file__).resolve().parent
GENRE_NAMES = [
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
EXPECTED_SPLIT_ROWS = {"validation": 1121, "test": 1121}
EXPECTED_TEST_SERIES_GROUPS = 677
PREDICTION_KEYS = (
    "ids",
    "series_groups",
    "genre_names",
    "targets",
    "logits",
    "scores",
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def write_yaml(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(value, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def split_csv_path(split: str) -> Path:
    if split not in {"training", "validation", "test"}:
        raise ValueError(f"unsupported split: {split}")
    return PROJECT_ROOT / "data" / "series_split_outputs" / f"{split}_data_grouped.csv"


def load_split(split: str, max_samples: int | None = None) -> pd.DataFrame:
    path = split_csv_path(split)
    frame = pd.read_csv(path)
    required = {"ID", "ImageUrl", "SeriesGroup", *GENRE_NAMES}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if frame["ID"].duplicated().any():
        raise ValueError(f"{path} contains duplicate IDs")
    if frame[list(required)].isna().any().any():
        raise ValueError(f"{path} contains missing required values")
    frame = frame.sort_values("ID", kind="stable").reset_index(drop=True)
    if max_samples is not None:
        frame = frame.head(int(max_samples)).copy()
    return frame


def resolve_image_dir() -> Path:
    override = os.environ.get("INFO3_IMAGE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return PROJECT_ROOT / "data" / "images"


def load_image(row: pd.Series) -> Image.Image:
    image_dir = resolve_image_dir()
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / f"{row['ID']}.jpg"
    if not image_path.exists():
        with urllib.request.urlopen(str(row["ImageUrl"]), timeout=20) as response:
            payload = response.read()
        temporary = image_path.with_suffix(".jpg.part")
        temporary.write_bytes(payload)
        temporary.replace(image_path)
    with Image.open(image_path) as image:
        return image.convert("RGB")


def build_transform(image_size: int, training: bool) -> transforms.Compose:
    operations: list[Any] = [transforms.Resize((image_size, image_size))]
    if training:
        operations.append(transforms.RandomHorizontalFlip(p=0.5))
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return transforms.Compose(operations)


class AnimeDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_size: int, training: bool):
        self.frame = frame.reset_index(drop=True)
        self.transform = build_transform(image_size, training)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, Any, str]:
        row = self.frame.iloc[index]
        image = self.transform(load_image(row))
        targets = torch.tensor(row[GENRE_NAMES].to_numpy(dtype=np.float32))
        return image, targets, row["ID"], str(row["SeriesGroup"])


def make_loader(
    frame: pd.DataFrame,
    *,
    image_size: int,
    training: bool,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        AnimeDataset(frame, image_size=image_size, training=training),
        batch_size=batch_size,
        shuffle=training,
        num_workers=num_workers,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name != "auto":
        return torch.device(device_name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def calculate_map(targets: np.ndarray, scores: np.ndarray) -> float:
    present = targets.sum(axis=0) > 0
    mean_ap = (
        average_precision_score(
            targets[:, present],
            scores[:, present],
            average="macro",
        )
        if present.any()
        else 0.0
    )
    return float(mean_ap)


def calculate_metrics(
    targets: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    predictions = scores >= threshold
    return {
        "mAP": calculate_map(targets, scores),
        "macro_f1": float(
            f1_score(targets, predictions, average="macro", zero_division=0)
        ),
        "samples_f1": float(
            f1_score(targets, predictions, average="samples", zero_division=0)
        ),
        "hamming_loss": float(hamming_loss(targets, predictions)),
    }


def calculate_genre_ap(targets: np.ndarray, scores: np.ndarray) -> list[dict[str, Any]]:
    result = []
    for index, genre_name in enumerate(GENRE_NAMES):
        positives = int(targets[:, index].sum())
        result.append(
            {
                "genre": genre_name,
                "positive_count": positives,
                "average_precision": (
                    float(average_precision_score(targets[:, index], scores[:, index]))
                    if positives
                    else None
                ),
            }
        )
    return result


def predict_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    all_ids: list[np.ndarray] = []
    all_groups: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_logits: list[np.ndarray] = []
    with torch.no_grad():
        for inputs, targets, ids, groups in loader:
            logits = model(inputs.to(device)).detach().cpu().float()
            all_ids.append(np.asarray(ids))
            all_groups.append(np.asarray(groups, dtype=str))
            all_targets.append(targets.numpy())
            all_logits.append(logits.numpy())
    logits = np.concatenate(all_logits).astype(np.float32)
    scores = torch.sigmoid(torch.from_numpy(logits)).numpy().astype(np.float32)
    return {
        "ids": np.concatenate(all_ids),
        "series_groups": np.concatenate(all_groups).astype(str),
        "genre_names": np.asarray(GENRE_NAMES),
        "targets": np.concatenate(all_targets).astype(np.uint8),
        "logits": logits,
        "scores": scores,
    }


def save_predictions(path: Path, arrays: dict[str, np.ndarray]) -> None:
    validate_prediction_arrays(arrays)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_predictions(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    validate_prediction_arrays(arrays)
    return arrays


def validate_prediction_arrays(
    arrays: dict[str, np.ndarray],
    *,
    expected_rows: int | None = None,
    expected_series_groups: int | None = None,
) -> None:
    if tuple(arrays) != PREDICTION_KEYS:
        raise ValueError(
            f"prediction keys must be {PREDICTION_KEYS}, got {tuple(arrays)}"
        )
    row_count = len(arrays["ids"])
    if expected_rows is not None and row_count != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, got {row_count}")
    if len(np.unique(arrays["ids"])) != row_count:
        raise ValueError("prediction IDs are not unique")
    if arrays["series_groups"].shape != (row_count,):
        raise ValueError("series_groups has an invalid shape")
    if arrays["series_groups"].dtype.kind not in {"U", "S"}:
        raise ValueError("series_groups must use a non-object string dtype")
    if arrays["genre_names"].tolist() != GENRE_NAMES:
        raise ValueError("genre_names or their order do not match")
    expected_matrix_shape = (row_count, len(GENRE_NAMES))
    for key in ("targets", "logits", "scores"):
        if arrays[key].shape != expected_matrix_shape:
            raise ValueError(f"{key} has shape {arrays[key].shape}")
    if arrays["targets"].dtype not in (np.dtype("uint8"), np.dtype("bool")):
        raise ValueError("targets must use uint8 or bool")
    if not np.isfinite(arrays["logits"]).all():
        raise ValueError("logits contain non-finite values")
    if not np.isfinite(arrays["scores"]).all():
        raise ValueError("scores contain non-finite values")
    if not ((arrays["scores"] >= 0) & (arrays["scores"] <= 1)).all():
        raise ValueError("scores are outside [0, 1]")
    expected_scores = torch.sigmoid(torch.from_numpy(arrays["logits"])).numpy()
    if not np.allclose(arrays["scores"], expected_scores, atol=1e-6, rtol=1e-6):
        raise ValueError("scores do not match sigmoid(logits)")
    if expected_series_groups is not None:
        actual_groups = len(np.unique(arrays["series_groups"]))
        if actual_groups != expected_series_groups:
            raise ValueError(
                f"expected {expected_series_groups} SeriesGroups, got {actual_groups}"
            )


def git_value(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def current_environment() -> dict[str, Any]:
    try:
        import timm
        import torchvision
    except ImportError:
        timm = None
        torchvision = None
    return {
        "captured_at": utc_now(),
        "purpose": "artifact preparation or execution environment",
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "torchvision": getattr(torchvision, "__version__", None),
        "timm": getattr(timm, "__version__", None),
        "platform": platform.platform(),
        "device_requested": None,
        "image_cache": str(resolve_image_dir()),
    }


def backup_run_directory(run_dir: Path) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup = run_dir.with_name(f"{run_dir.name}.backup-{stamp}")
    counter = 1
    while backup.exists():
        backup = run_dir.with_name(f"{run_dir.name}.backup-{stamp}-{counter}")
        counter += 1
    run_dir.rename(backup)
    print(f"--force: moved existing run {run_dir} to {backup}")
    return backup


def save_checkpoint(model: torch.nn.Module, path: Path) -> None:
    raw_model = getattr(model, "_orig_mod", model)
    state = {key: value.detach().cpu() for key, value in raw_model.state_dict().items()}
    torch.save(state, path)


def _code_snapshot_hash(model_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(
        [*model_dir.glob("*.py"), model_dir / "config.yaml", OUTPUTS_ROOT / "artifact_common.py"]
    ):
        if path.exists():
            digest.update(path.relative_to(PROJECT_ROOT).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def run_training(
    *,
    model_id: str,
    config_path: Path,
    seed: int,
    force: bool,
    smoke: bool,
    build_model: Callable[[dict[str, Any]], torch.nn.Module],
    build_criterion: Callable[[dict[str, Any]], torch.nn.Module],
    build_optimizer: Callable[
        [torch.nn.Module, dict[str, Any]], torch.optim.Optimizer
    ],
    train_one_epoch: Callable[..., float],
    evaluate_model: Callable[..., dict[str, Any]],
) -> Path:
    started_at = utc_now()
    start = time.monotonic()
    config = load_yaml(config_path)
    config["seed"] = int(seed)
    if smoke:
        config.update(
            {
                "epochs": 1,
                "batch_size": 2,
                "num_workers": 0,
                "max_train_samples": 2,
                "max_val_samples": 2,
                "compile": False,
                "device": "cpu",
                "pretrained": False,
            }
        )
        run_dir = OUTPUTS_ROOT / "smoke" / model_id / "runs" / f"seed_{seed}"
    else:
        output_root = Path(str(config.get("output_dir", "runs")))
        if not output_root.is_absolute():
            output_root = config_path.resolve().parent / output_root
        run_dir = output_root / f"seed_{seed}"
    if (run_dir / "COMPLETED").exists() and not force:
        raise FileExistsError(
            f"{run_dir} is complete; pass --force to preserve it as a backup and rerun"
        )
    if run_dir.exists():
        if force:
            backup_run_directory(run_dir)
        elif any(run_dir.iterdir()):
            raise FileExistsError(f"{run_dir} is non-empty; pass --force to back it up")
    run_dir.mkdir(parents=True, exist_ok=True)
    write_yaml(run_dir / "resolved-config.yaml", config)

    set_seed(seed)
    device = resolve_device(str(config.get("device", "auto")))
    train_frame = load_split("training", config.get("max_train_samples"))
    validation_frame = load_split("validation", config.get("max_val_samples"))
    train_loader = make_loader(
        train_frame,
        image_size=int(config["image_size"]),
        training=True,
        batch_size=int(config["batch_size"]),
        num_workers=int(config["num_workers"]),
    )
    validation_loader = make_loader(
        validation_frame,
        image_size=int(config["image_size"]),
        training=False,
        batch_size=int(config["batch_size"]),
        num_workers=int(config["num_workers"]),
    )
    model = build_model(config).to(device)
    if bool(config.get("compile")) and device.type not in {"mps", "cpu"}:
        model = torch.compile(model)
    criterion = build_criterion(config)
    optimizer = build_optimizer(model, config)
    early = config.get("early_stopping") or {}
    best_value = float("-inf")
    best_epoch = 0
    without_improvement = 0
    stopped_early = False
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(config["epochs"]) + 1):
        epoch_start = time.monotonic()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        validation = evaluate_model(
            model,
            validation_loader,
            criterion,
            device,
            collect_predictions=False,
        )
        monitor = float(validation[str(early.get("monitor", "mAP"))])
        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(validation["val_loss"]),
            "mAP": float(validation["mAP"]),
            "macro_f1": float(validation["macro_f1"]),
            "samples_f1": float(validation["samples_f1"]),
            "hamming_loss": float(validation["hamming_loss"]),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "epoch_seconds": float(time.monotonic() - epoch_start),
        }
        history.append(row)
        min_delta = float(early.get("min_delta", 0))
        if monitor > best_value + min_delta:
            best_value = monitor
            best_epoch = epoch
            without_improvement = 0
            save_checkpoint(model, run_dir / "best_model.pth")
        else:
            without_improvement += 1
        if (
            early.get("enabled")
            and epoch >= int(early.get("min_epochs", 0))
            and without_improvement >= int(early.get("patience", 0))
        ):
            stopped_early = True
            break
    pd.DataFrame(history).to_csv(run_dir / "metrics.csv", index=False)

    checkpoint = torch.load(
        run_dir / "best_model.pth",
        map_location=device,
        weights_only=True,
    )
    raw_model = getattr(model, "_orig_mod", model)
    raw_model.load_state_dict(checkpoint, strict=True)
    predictions = predict_model(raw_model, validation_loader, device)
    save_predictions(run_dir / "validation_predictions.npz", predictions)
    validation_metrics = {
        **calculate_metrics(predictions["targets"], predictions["scores"]),
        "threshold": 0.5,
        "row_count": len(predictions["ids"]),
        "genre_count": len(GENRE_NAMES),
    }
    write_json(run_dir / "validation_metrics.json", validation_metrics)

    ended_at = utc_now()
    environment = current_environment()
    environment["device_requested"] = str(device)
    write_json(run_dir / "environment.json", environment)
    resolved_hash = sha256_file(run_dir / "resolved-config.yaml")
    checkpoint_hash = sha256_file(run_dir / "best_model.pth")
    metadata = {
        "model_id": model_id,
        "seed": int(seed),
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_seconds": float(time.monotonic() - start),
        "best_epoch": int(best_epoch),
        "best_validation_mAP": float(validation_metrics["mAP"]),
        "early_stopping": {
            "monitor": str(early.get("monitor", "mAP")),
            "min_delta": float(early.get("min_delta", 0)),
            "stopped_early": stopped_early,
            "stop_reason": (
                "patience exhausted" if stopped_early else "configured epochs completed"
            ),
        },
        "versions": {
            "python": environment["python"],
            "pytorch": environment["pytorch"],
            "torchvision": environment["torchvision"],
            "timm": environment["timm"],
        },
        "git_commit": git_value("rev-parse", "HEAD"),
        "dirty_worktree": bool(git_value("status", "--porcelain")),
        "code_snapshot_sha256": _code_snapshot_hash(config_path.parent),
        "resolved_config_sha256": resolved_hash,
        "checkpoint_sha256": checkpoint_hash,
        "historical_metadata_note": None,
    }
    write_json(run_dir / "run-metadata.json", metadata)
    write_json(
        run_dir / "COMPLETED",
        {
            "completed_at": ended_at,
            "integrity_checked": True,
            "checkpoint_sha256": checkpoint_hash,
            "validation_predictions_sha256": sha256_file(
                run_dir / "validation_predictions.npz"
            ),
        },
    )
    return run_dir


def materialize_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    if temporary.exists():
        temporary.unlink()
    shutil.copy2(source, temporary)
    temporary.replace(destination)
