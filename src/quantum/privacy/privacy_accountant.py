"""Privacy accounting for tracking cumulative privacy loss."""
import math
from typing import Optional


class PrivacyAccountant:
    def __init__(self, delta: float = 1e-5):
        self.delta = delta
        self.eps_cumulative = 0.0
        self.q_cumulative = 0.0

    def compute_eps_gaussian(self, q: float, sigma: float, steps: int, delta: Optional[float] = None) -> float:
        if delta is None:
            delta = self.delta
        if sigma <= 0 or q <= 0:
            return float('inf')
        if q >= 1:
            eps = math.sqrt(2 * math.log(1.25 / delta)) / sigma
            return eps * math.sqrt(steps)
        else:
            sampling_prob = q
            noise_scale = sigma / sampling_prob
            eps = math.sqrt(2 * math.log(1.25 / delta)) / noise_scale
            return eps * math.sqrt(steps)

    def step(self, q: float, sigma: float) -> float:
        eps_step = self.compute_eps_gaussian(q, sigma, 1)
        self.eps_cumulative += eps_step
        self.q_cumulative += q
        return eps_step

    def get_privacy_spent(self) -> float:
        return self.eps_cumulative

    def reset(self):
        self.eps_cumulative = 0.0
        self.q_cumulative = 0.0

    def state_dict(self) -> dict:
        return {"delta": self.delta, "eps_cumulative": self.eps_cumulative,
                "q_cumulative": self.q_cumulative}

    def load_state_dict(self, state_dict: dict):
        self.delta = state_dict["delta"]
        self.eps_cumulative = state_dict["eps_cumulative"]
        self.q_cumulative = state_dict.get("q_cumulative", 0.0)
