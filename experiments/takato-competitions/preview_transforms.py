"""Save a visual preview of the image preprocessing in exp_run.py.

Example:
    uv run python experiments/takato-competitions/preview_transforms.py --id 1
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from exp_run import build_transforms, load_config
from src.preprocessing.dataset_utils import load_dataset, load_image


IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGE_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """Undo ImageNet normalization for plotting."""
    image = (tensor.cpu() * IMAGE_STD + IMAGE_MEAN).clamp(0, 1)
    return image.permute(1, 2, 0).numpy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview C-Tran image preprocessing.")
    parser.add_argument("--config", type=Path, default=EXPERIMENT_DIR / "config_ctran.yaml")
    parser.add_argument("--id", type=int, help="Anime ID to preview. Defaults to the first training sample.")
    parser.add_argument("--samples", type=int, default=4, help="Number of random training augmentations to show.")
    parser.add_argument("--output", type=Path, default=EXPERIMENT_DIR / "outputs" / "transform_preview.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples < 1:
        raise ValueError("--samples must be at least 1")

    config = load_config(args.config.resolve())
    train_df, _, _ = load_dataset()
    if args.id is None:
        row = train_df.iloc[0]
    else:
        selected = train_df.loc[train_df["ID"] == args.id]
        if selected.empty:
            raise ValueError(f"Training dataに ID {args.id} はありません。")
        row = selected.iloc[0]

    image = load_image(train_df, row["ID"])
    if image is None:
        raise ValueError(f"ID {row['ID']} の画像が取得できませんでした。")
    train_transform, validation_transform = build_transforms(
        int(config["scale_size"]), int(config["crop_size"])
    )

    total_images = args.samples + 2
    figure, axes = plt.subplots(1, total_images, figsize=(4 * total_images, 4))
    axes[0].imshow(image)
    axes[0].set_title(f"Original\nID {row['ID']}")
    axes[0].axis("off")

    for index in range(args.samples):
        axes[index + 1].imshow(tensor_to_image(train_transform(image.copy())))
        axes[index + 1].set_title(f"Train augmentation {index + 1}")
        axes[index + 1].axis("off")

    axes[-1].imshow(tensor_to_image(validation_transform(image.copy())))
    axes[-1].set_title("Validation preprocessing")
    axes[-1].axis("off")

    figure.suptitle(
        f"C-Tran preprocessing: resize {config['scale_size']} -> crop {config['crop_size']}",
        fontsize=14,
    )
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved transform preview to {args.output}")


if __name__ == "__main__":
    main()
