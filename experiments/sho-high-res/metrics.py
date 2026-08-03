import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, hamming_loss


def calculate_metrics_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, float]:
    preds = (logits > 0).int()

    preds_np = preds.cpu().numpy()
    logits_np = logits.cpu().numpy()
    targets_np = targets.cpu().numpy()

    # ロジットを確率(0.0~1.0)に変換
    probs_np = 1.0 / (1.0 + np.exp(-logits_np))

    macro_f1 = f1_score(targets_np, preds_np, average="macro", zero_division=0)
    samples_f1 = f1_score(targets_np, preds_np, average="samples", zero_division=0)
    h_loss = hamming_loss(targets_np, preds_np)

    valid_class_indices = targets_np.sum(axis=0) > 0
    if valid_class_indices.any():
        mean_average_precision = average_precision_score(
            targets_np[:, valid_class_indices],
            logits_np[:, valid_class_indices],
            average="macro",
        )
    else:
        mean_average_precision = 0.0

    return {
        "macro_f1": float(macro_f1),
        "samples_f1": float(samples_f1),
        "hamming_loss": float(h_loss),
        "mAP": float(mean_average_precision),
        "_raw_probs": probs_np,     # PR曲線用
        "_raw_targets": targets_np, # PR曲線用
    }
