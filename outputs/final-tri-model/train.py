import torch


def train_one_epoch(model, dataloader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    for inputs, targets, _, _ in dataloader:
        inputs = inputs.to(device)
        targets = targets.float().to(device)
        optimizer.zero_grad()
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * len(inputs)
    return total_loss / len(dataloader.dataset)
