from contextlib import nullcontext

import torch
from torch.optim.swa_utils import AveragedModel
from tqdm import tqdm


def train_one_epoch(
    model: torch.nn.Module,
    ema_model: AveragedModel,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    criterion: torch.nn.Module,
    device: torch.device,
    gradient_accumulation_steps: int = 1,
    scaler: torch.amp.GradScaler | None = None,
) -> float:
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1")

    model.train()
    running_loss = 0.0
    num_batches = len(dataloader)
    amp_enabled = scaler is not None and scaler.is_enabled()
    optimizer.zero_grad(set_to_none=True)

    for batch_index, (inputs, targets) in enumerate(
        tqdm(dataloader, desc="Training", leave=False)
    ):
        inputs = inputs.to(device)
        targets = targets.float().to(device)

        autocast_context = (
            torch.autocast(device_type=device.type) if amp_enabled else nullcontext()
        )
        with autocast_context:
            logits = model(inputs)
            loss = criterion(logits, targets)
        running_loss += loss.item() * inputs.size(0)

        accumulation_group_start = (
            batch_index // gradient_accumulation_steps
        ) * gradient_accumulation_steps
        accumulation_group_size = min(
            gradient_accumulation_steps,
            num_batches - accumulation_group_start,
        )
        normalized_loss = loss / accumulation_group_size
        if amp_enabled:
            scaler.scale(normalized_loss).backward()
        else:
            normalized_loss.backward()

        is_accumulation_boundary = (
            (batch_index + 1) % gradient_accumulation_steps == 0
            or batch_index + 1 == num_batches
        )
        if is_accumulation_boundary:
            optimizer_step_completed = True
            if amp_enabled:
                scale_before_step = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer_step_completed = scaler.get_scale() >= scale_before_step
            else:
                optimizer.step()

            if optimizer_step_completed:
                scheduler.step()
                ema_model.update_parameters(model)
            optimizer.zero_grad(set_to_none=True)

    return running_loss / len(dataloader.dataset)
