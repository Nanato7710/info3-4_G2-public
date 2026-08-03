from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from model import build_model

SUPPORTED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
IMAGE_SIZE = 384
MODEL_ID = "final-tri-model"
MODEL_SEED = 44


class AppConfigurationError(RuntimeError):
    pass


class UserInputError(ValueError):
    pass


@dataclass(frozen=True)
class ThresholdConfig:
    model_id: str
    seed: int
    checkpoint_sha256: str
    validation_predictions_sha256: str
    genre_names: tuple[str, ...]
    thresholds: tuple[float, ...]
    validation_f1: tuple[float, ...]
    validation_macro_f1: float


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_genre_names(path: Path) -> tuple[str, ...]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AppConfigurationError(f"ジャンル設定を読み込めません: {path}") from exc
    if not isinstance(value, list) or len(value) != 19:
        raise AppConfigurationError("genres.jsonには19個のジャンルが必要です")
    names = tuple(str(item) for item in value)
    if any(not name for name in names) or len(set(names)) != len(names):
        raise AppConfigurationError("ジャンル名は空でない一意な値にしてください")
    return names


def load_threshold_config(path: Path, genre_names: Sequence[str]) -> ThresholdConfig:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AppConfigurationError(f"閾値設定を読み込めません: {path}") from exc

    configured_names = tuple(value.get("genre_names", []))
    expected_names = tuple(genre_names)
    if configured_names != expected_names:
        raise AppConfigurationError("threshold.jsonとgenres.jsonのジャンル順が一致しません")

    threshold_map = value.get("thresholds")
    f1_map = value.get("validation_f1")
    if not isinstance(threshold_map, dict) or not isinstance(f1_map, dict):
        raise AppConfigurationError("thresholdsとvalidation_f1はジャンル別のobjectが必要です")
    if tuple(threshold_map) != expected_names or tuple(f1_map) != expected_names:
        raise AppConfigurationError("ジャンル別閾値またはF1のキーと順序が不正です")

    thresholds = tuple(float(threshold_map[name]) for name in expected_names)
    validation_f1 = tuple(float(f1_map[name]) for name in expected_names)
    if not all(np.isfinite(item) and 0.0 <= item <= 1.0 for item in thresholds):
        raise AppConfigurationError("ジャンル別閾値は0以上1以下の有限値にしてください")
    if not all(np.isfinite(item) and 0.0 <= item <= 1.0 for item in validation_f1):
        raise AppConfigurationError("ジャンル別validation F1が不正です")

    model_id = str(value.get("model_id", ""))
    seed = int(value.get("seed", -1))
    source_split = str(value.get("source_split", ""))
    objective = str(value.get("objective", ""))
    search = value.get("search")
    checkpoint_sha256 = str(value.get("checkpoint_sha256", ""))
    validation_sha256 = str(value.get("validation_predictions_sha256", ""))
    macro_f1 = float(value.get("validation_macro_f1", float("nan")))
    if model_id != MODEL_ID or seed != MODEL_SEED:
        raise AppConfigurationError("閾値設定はfinal-tri-modelのseed 44に対応していません")
    if source_split != "validation" or objective != "per_genre_binary_f1":
        raise AppConfigurationError("閾値設定はvalidationのジャンル別F1に基づく必要があります")
    if not isinstance(search, dict) or search != {
        "minimum": 0.0,
        "maximum": 1.0,
        "step": 0.01,
        "tie_break": "highest_threshold",
    }:
        raise AppConfigurationError("閾値の探索条件が正式仕様と一致しません")
    if len(checkpoint_sha256) != 64 or len(validation_sha256) != 64:
        raise AppConfigurationError("閾値設定のSHA-256が不正です")
    if not np.isfinite(macro_f1) or not 0.0 <= macro_f1 <= 1.0:
        raise AppConfigurationError("validation Macro F1が不正です")

    return ThresholdConfig(
        model_id=model_id,
        seed=seed,
        checkpoint_sha256=checkpoint_sha256,
        validation_predictions_sha256=validation_sha256,
        genre_names=expected_names,
        thresholds=thresholds,
        validation_f1=validation_f1,
        validation_macro_f1=macro_f1,
    )


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_preprocess() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def preprocess_image(image: Image.Image | str | Path) -> torch.Tensor:
    if image is None:
        raise UserInputError("画像を一枚選択してください")
    if isinstance(image, (str, Path)):
        image_path = Path(image)
        if image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            raise UserInputError("JPEG、PNG、WebPの画像を選択してください")
        try:
            with Image.open(image_path) as opened:
                loaded = opened.copy()
                loaded.format = opened.format
            image = loaded
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            raise UserInputError("画像を読み込めません") from exc
    if not isinstance(image, Image.Image):
        raise UserInputError("画像を読み込めません")
    image_format = image.format.upper() if image.format else None
    if image_format and image_format not in SUPPORTED_IMAGE_FORMATS:
        raise UserInputError("JPEG、PNG、WebPの画像を選択してください")
    try:
        rgb_image = image.convert("RGB")
        tensor = build_preprocess()(rgb_image)
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        raise UserInputError("画像を読み込めません") from exc
    if tensor.shape != (3, IMAGE_SIZE, IMAGE_SIZE):
        raise UserInputError("画像の前処理結果が不正です")
    return tensor


def validate_scores(scores: np.ndarray, genre_count: int) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float32)
    if values.shape != (genre_count,):
        raise RuntimeError(f"推論出力shapeが不正です: {values.shape}")
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise RuntimeError("推論scoreは0以上1以下の有限値である必要があります")
    return values


def format_results(
    scores: Sequence[float],
    thresholds: Sequence[float],
    genre_names: Sequence[str],
) -> tuple[list[list[object]], list[list[object]], list[list[object]]]:
    values = validate_scores(np.asarray(scores), len(genre_names))
    threshold_values = np.asarray(thresholds, dtype=np.float64)
    if threshold_values.shape != values.shape:
        raise ValueError("scoreと閾値の個数が一致しません")
    if not np.isfinite(threshold_values).all() or (
        (threshold_values < 0) | (threshold_values > 1)
    ).any():
        raise ValueError("閾値は0以上1以下の有限値にしてください")

    order = np.argsort(-values, kind="stable")
    rows = []
    for rank, index in enumerate(order, start=1):
        selected = bool(values[index] >= threshold_values[index])
        rows.append(
            [
                rank,
                str(genre_names[index]),
                round(float(values[index]), 6),
                round(float(threshold_values[index]), 2),
                "候補" if selected else "",
            ]
        )
    return rows[:5], [row for row in rows if row[4]], rows


class InferenceEngine:
    def __init__(
        self,
        checkpoint_path: Path,
        genre_names: Sequence[str],
        threshold_config: ThresholdConfig,
        device: torch.device | None = None,
    ):
        self.checkpoint_path = checkpoint_path
        self.genre_names = tuple(genre_names)
        self.threshold_config = threshold_config
        if not checkpoint_path.is_file():
            raise AppConfigurationError(f"checkpointがありません: {checkpoint_path}")
        actual_sha256 = sha256_file(checkpoint_path)
        if actual_sha256 != threshold_config.checkpoint_sha256:
            raise AppConfigurationError("checkpointのSHA-256がthreshold.jsonと一致しません")
        self.device = device or select_device()
        self.model = build_model(num_classes=len(self.genre_names))
        try:
            state_dict = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            self.model.load_state_dict(state_dict, strict=True)
        except (OSError, RuntimeError, ValueError) as exc:
            raise AppConfigurationError(
                "モデル定義とcheckpointに互換性がありません"
            ) from exc
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image: Image.Image | str | Path) -> np.ndarray:
        inputs = preprocess_image(image).unsqueeze(0).to(self.device)
        try:
            with torch.inference_mode():
                logits = self.model(inputs)
                scores = torch.sigmoid(logits).detach().cpu().numpy()
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise RuntimeError(
                    "推論中にメモリが不足しました。CPUでの起動を試してください"
                ) from exc
            raise
        if scores.shape != (1, len(self.genre_names)):
            raise RuntimeError(f"推論出力shapeが不正です: {scores.shape}")
        return validate_scores(scores[0], len(self.genre_names))
