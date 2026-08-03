import numpy as np
import torch

from artifact_common import calculate_metrics


def evaluate_model(
    model,
    dataloader,
    criterion,
    device,
    *,
    collect_predictions: bool = False,
) -> dict:
    model.eval()
    total_loss = 0.0
    logits_batches = []
    target_batches = []
    with torch.no_grad():
        for inputs, targets, _, _ in dataloader:
            targets = targets.float().to(device)
            logits = model(inputs.to(device))
            total_loss += float(criterion(logits, targets).item()) * len(inputs)
            logits_batches.append(logits.detach().cpu().float().numpy())
            target_batches.append(targets.detach().cpu().numpy())
    logits = np.concatenate(logits_batches)
    targets = np.concatenate(target_batches)
    scores = torch.sigmoid(torch.from_numpy(logits)).numpy()
    result = {
        **calculate_metrics(targets, scores),
        "val_loss": total_loss / len(dataloader.dataset),
    }
    if collect_predictions:
        result["logits"] = logits
        result["targets"] = targets
    return result
