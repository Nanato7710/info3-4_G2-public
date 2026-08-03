import numpy as np
from sklearn.metrics import average_precision_score, f1_score, hamming_loss


def sanitize_binary_targets(y_true: np.ndarray) -> np.ndarray:
    y_true = np.nan_to_num(
        y_true,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
    return (y_true > 0).astype(np.int32)


def sanitize_probabilities(y_prob: np.ndarray) -> np.ndarray:
    y_prob = np.nan_to_num(
        y_prob,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
    return np.clip(y_prob, 0.0, 1.0)


def calculate_multilabel_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    y_true = sanitize_binary_targets(y_true)
    y_prob = sanitize_probabilities(y_prob)
    y_pred = (y_prob >= threshold).astype(np.int32)

    try:
        m_ap = average_precision_score(
            y_true,
            y_prob,
            average="macro",
        )
    except ValueError:
        m_ap = 0.0

    return {
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "samples_f1": f1_score(
            y_true,
            y_pred,
            average="samples",
            zero_division=0,
        ),
        "hamming_loss": hamming_loss(y_true, y_pred),
        "mAP": float(m_ap),
    }
