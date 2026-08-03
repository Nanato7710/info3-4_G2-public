import torch
from tqdm import tqdm


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0

    for inputs, targets in tqdm(dataloader, desc="Training", leave=False):
        inputs = inputs.to(device)
        targets = targets.float().to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    return running_loss / len(dataloader.dataset)
