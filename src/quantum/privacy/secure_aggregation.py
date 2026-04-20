"""Secure aggregation protocols for privacy-preserving federated learning."""

import torch


class SecureAggregator:
    def __init__(self, num_clients: int, modulus: float = 1e8):
        self.num_clients = num_clients
        self.modulus = modulus
        self.mask_pairs: dict[tuple[int, int], torch.Tensor] = {}

    def _pair_key(self, i: int, j: int) -> tuple[int, int]:
        return (min(i, j), max(i, j))

    def _deterministic_mask(self, client_i: int, client_j: int, param_shape: tuple, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        key = self._pair_key(client_i, client_j)
        seed_val = hash(key) % (2**31)
        rng_state = torch.random.get_rng_state()
        torch.manual_seed(seed_val)
        mask = torch.empty(param_shape, dtype=dtype, device=device).uniform_(-self.modulus, self.modulus)
        torch.random.set_rng_state(rng_state)
        return mask

    def generate_masks_for_client(self, client_id: int, param_shapes: dict[str, torch.Size],
                                   dtype: torch.dtype = torch.float32,
                                   device: torch.device = torch.device('cpu')) -> dict[str, torch.Tensor]:
        masks = {}
        for key, shape in param_shapes.items():
            mask_total = torch.zeros(shape, dtype=dtype, device=device)
            for other_id in range(self.num_clients):
                if other_id == client_id:
                    continue
                pair_mask = self._deterministic_mask(client_id, other_id, shape, dtype, device)
                if other_id < client_id:
                    mask_total = mask_total + pair_mask
                else:
                    mask_total = mask_total - pair_mask
            masks[key] = mask_total
        return masks

    def mask_params(self, params: dict[str, torch.Tensor],
                    masks: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        masked = {}
        for key in params:
            masked[key] = params[key] + masks[key]
        return masked

    def unmask_params(self, masked_params: dict[str, torch.Tensor],
                       all_client_masks: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        if not all_client_masks:
            return masked_params
        aggregated = {}
        mask_sum = {key: torch.zeros_like(masked_params[key]) for key in masked_params}
        for client_masks in all_client_masks:
            for key in client_masks:
                mask_sum[key] = mask_sum[key] + client_masks[key]
        for key in masked_params:
            aggregated[key] = masked_params[key] - mask_sum[key]
        return aggregated

    def state_dict(self) -> dict:
        return {"num_clients": self.num_clients, "modulus": self.modulus}

    def load_state_dict(self, state_dict: dict):
        self.num_clients = state_dict["num_clients"]
        self.modulus = state_dict.get("modulus", 1e8)
