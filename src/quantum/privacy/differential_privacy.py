"""Differential privacy mechanisms for secure federated learning."""

import numpy as np
import torch


def clip_gradients(gradients: dict[str, torch.Tensor], clip_norm: float = 1.0) -> dict[str, torch.Tensor]:
    clipped = {}
    global_norm = 0.0
    for key, grad in gradients.items():
        if grad is not None:
            global_norm += grad.norm().item() ** 2
    global_norm = np.sqrt(global_norm)
    scale = min(1.0, clip_norm / (global_norm + 1e-8))
    for key, grad in gradients.items():
        if grad is not None:
            clipped[key] = grad * scale
    return clipped, global_norm


def add_gaussian_noise(gradients: dict[str, torch.Tensor], noise_scale: float) -> dict[str, torch.Tensor]:
    noisy = {}
    for key, grad in gradients.items():
        if grad is not None:
            noise = torch.normal(mean=0.0, std=noise_scale, size=grad.shape, device=grad.device)
            noisy[key] = grad + noise
    return noisy


class DifferentialPrivacy:
    def __init__(self, clip_norm: float = 1.0, noise_multiplier: float = 1.0,
                 secure_mode: bool = True):
        self.clip_norm = clip_norm
        self.noise_multiplier = noise_multiplier
        self.secure_mode = secure_mode

    def apply(self, params: dict[str, torch.Tensor],
              batch_size: int, sample_rate: float) -> tuple[dict[str, torch.Tensor], float]:
        grads = {}
        for key, param in params.items():
            if param.grad is not None:
                grads[key] = param.grad.detach().clone()

        if not grads:
            return params, 0.0

        clipped_grads, global_norm = clip_gradients(grads, self.clip_norm)

        noise_scale = self.noise_multiplier * self.clip_norm
        if batch_size > 0:
            noise_scale = noise_scale / batch_size

        noisy_grads = add_gaussian_noise(clipped_grads, noise_scale)

        for key in grads:
            if params[key].grad is not None:
                params[key].grad = noisy_grads[key]

        return params, global_norm

    def state_dict(self) -> dict:
        return {"clip_norm": self.clip_norm, "noise_multiplier": self.noise_multiplier}

    def load_state_dict(self, state_dict: dict):
        self.clip_norm = state_dict["clip_norm"]
        self.noise_multiplier = state_dict["noise_multiplier"]
