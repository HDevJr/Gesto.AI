from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, object]:
    is_training = optimizer is not None
    model.train(mode=is_training)

    running_loss = 0.0
    correct = 0
    total = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            logits = model(x)
            loss = criterion(logits, y)

            if is_training:
                loss.backward()
                optimizer.step()

        batch_size = y.size(0)
        running_loss += loss.item() * batch_size

        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += batch_size

        y_true.extend(y.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())

    avg_loss = running_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "y_true": y_true,
        "y_pred": y_pred,
    }
