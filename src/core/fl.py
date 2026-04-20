"""Unified federated averaging and FL configuration."""

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class FederatedLearningConfig:
    num_rounds: int = 5
    local_epochs: int = 3
    batch_size: int = 16
    client_fraction: float = 1.0
    classical_lr: float = 1e-3
    quantum_lr: float = 5e-4
    weight_decay: float = 1e-4
    grad_clip: Optional[float] = 1.0
    num_clients: int = 4
    dp_enabled: bool = False
    dp_clip_norm: float = 1.0
    dp_noise_multiplier: float = 1.0
    dp_delta: float = 1e-5
    secure_aggregation: bool = False
    early_stop_patience: int = 0
    early_stop_delta: float = 1e-4
    checkpoint_dir: str = ""
    checkpoint_interval: int = 5

    def to_dict(self) -> dict:
        return asdict(self)


def federated_averaging(
    client_updates: list[tuple[dict[str, torch.Tensor], int, float]],
    global_params_template: dict[str, torch.Tensor],
    *,
    device: torch.device = torch.device('cpu'),
    wrap_angles: bool = False,
) -> tuple[dict[str, torch.Tensor], float]:
    if not client_updates:
        raise ValueError("No client updates to aggregate")

    total_samples = sum(n for _, n, _ in client_updates)
    if total_samples == 0:
        raise ValueError("Total samples is zero")

    aggregated: dict[str, torch.Tensor] = {}
    for key, tensor in global_params_template.items():
        if tensor.dtype == torch.long or 'num_batches_tracked' in key:
            aggregated[key] = tensor.clone().to(device)
        else:
            aggregated[key] = torch.zeros_like(tensor, device=device)

    weighted_loss = 0.0
    for params, n_samples, loss in client_updates:
        weight = n_samples / total_samples
        weighted_loss += weight * loss
        for key in aggregated.keys():
            if aggregated[key].dtype != torch.long and 'num_batches_tracked' not in key:
                aggregated[key] += params[key].to(device) * weight

    if wrap_angles and 'q_weights' in aggregated:
        aggregated['q_weights'] = torch.atan2(
            torch.sin(aggregated['q_weights']),
            torch.cos(aggregated['q_weights'])
        )

    return aggregated, weighted_loss


def state_dict_to_ndarrays(state_dict: dict[str, torch.Tensor]) -> list[np.ndarray]:
    return [state_dict[k].detach().cpu().numpy() for k in sorted(state_dict)]


def ndarrays_to_state_dict(
    ndarrays: list[np.ndarray],
    template: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    keys = sorted(template)
    return {k: torch.from_numpy(a) for k, a in zip(keys, ndarrays)}
