from contextlib import nullcontext

import torch
from tqdm import tqdm

from metrics import calculate_metrics_from_logits


@torch.inference_mode()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    use_amp: bool = False,
) -> dict[str, float]:
    model.eval()
    running_loss = 0.0
    all_logits = []
    all_targets = []

    for inputs, targets in tqdm(dataloader, desc="Evaluating", leave=False):
        inputs = inputs.to(device)
        targets = targets.float().to(device)

        autocast_context = (
            torch.autocast(device_type=device.type) if use_amp else nullcontext()
        )
        with autocast_context:
            logits = model(inputs)
            loss = criterion(logits, targets)

        running_loss += loss.item() * inputs.size(0)
        all_logits.append(logits.float().cpu())
        all_targets.append(targets.cpu())

    val_loss = running_loss / len(dataloader.dataset)
    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    metrics = calculate_metrics_from_logits(logits, targets)
    metrics["val_loss"] = float(val_loss)
    return metrics
