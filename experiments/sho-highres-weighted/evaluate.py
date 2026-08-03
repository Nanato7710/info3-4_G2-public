import torch
from tqdm import tqdm

from metrics import calculate_metrics_from_logits


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    running_loss = 0.0
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs = inputs.to(device)
            targets = targets.float().to(device)

            logits = model(inputs)
            loss = criterion(logits, targets)

            running_loss += loss.item() * inputs.size(0)
            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())

    val_loss = running_loss / len(dataloader.dataset)
    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    metrics = calculate_metrics_from_logits(logits, targets)
    metrics["val_loss"] = float(val_loss)
    metrics["_raw_probs"] = torch.sigmoid(logits).numpy()
    metrics["_raw_targets"] = targets.numpy()
    return metrics
