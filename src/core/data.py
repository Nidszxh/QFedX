"""Unified DataLoader factory and evaluation function for consistent usage across modules."""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


def create_dataloader(
    X: Tensor | np.ndarray,
    y: Tensor | np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True,
    *,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    if not isinstance(X, Tensor):
        X = torch.as_tensor(X, dtype=torch.float32)
    if not isinstance(y, Tensor):
        y = torch.as_tensor(y, dtype=torch.long)

    dataset = TensorDataset(X, y)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    *,
    criterion: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
) -> tuple[float, float]:
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)
        outputs = model(data)
        loss = criterion(outputs, target)
        total_loss += loss.item()
        num_batches += 1
        correct += (outputs.argmax(1) == target).sum().item()
        total += target.size(0)

    avg_loss = total_loss / max(num_batches, 1)
    accuracy = min(correct / max(total, 1), 1.0)
    return avg_loss, accuracy
