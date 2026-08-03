import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from metrics import calculate_multilabel_metrics


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
):
    model.eval()

    total_loss = 0.0
    all_labels = []
    all_probs = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        images = torch.nan_to_num(
            images,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )
        labels = torch.nan_to_num(
            labels,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )
        labels = labels.clamp(0.0, 1.0)

        logits = model(images)
        logits = torch.nan_to_num(
            logits,
            nan=0.0,
            posinf=20.0,
            neginf=-20.0,
        )

        loss = criterion(logits, labels)
        probs = torch.sigmoid(logits)

        total_loss += loss.item() * images.size(0)

        all_labels.append(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)

    y_true = np.concatenate(all_labels, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)
    metrics = calculate_multilabel_metrics(
        y_true,
        y_prob,
        threshold=threshold,
    )

    return (
        avg_loss,
        metrics["macro_f1"],
        metrics["samples_f1"],
        metrics["hamming_loss"],
        metrics["mAP"],
    )
