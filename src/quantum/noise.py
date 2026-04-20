"""Noise models and noisy quantum circuit for simulating quantum hardware imperfections."""

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Optional

import pennylane as qml
import torch

from core.defs import NoiseType
from core.quantum import apply_encoding, apply_variational_layer, build_entanglement_pairs

# ── Noise Config & Model ─────────────────────────────────────────────────

@dataclass
class NoiseConfig:
    noise_type: NoiseType = NoiseType.NONE
    depolarizing_p: float = 0.001
    amplitude_gamma: float = 0.001
    readout_flip_prob: float = 0.01
    shots: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def __repr__(self) -> str:
        return (f"NoiseConfig(type={self.noise_type.value}, "
                f"depolarizing_p={self.depolarizing_p}, "
                f"amplitude_gamma={self.amplitude_gamma})")


class NoiseModel:
    def __init__(self, config: NoiseConfig):
        self.config = config
        self.noise_type = config.noise_type

    def apply_to_device(self, device: object, n_qubits: int) -> object:
        if self.noise_type == NoiseType.NONE:
            return device
        return qml.device('default.mixed', wires=n_qubits, shots=self.config.shots)

    def apply_readout_error(self, probabilities: torch.Tensor) -> torch.Tensor:
        if self.config.readout_flip_prob <= 0:
            return probabilities
        flip = self.config.readout_flip_prob
        n_classes = probabilities.shape[-1]
        confusion = torch.ones(n_classes, n_classes) * (flip / (n_classes - 1))
        confusion.fill_diagonal_(1.0 - flip)
        return probabilities @ confusion.T

    def state_dict(self) -> dict:
        return {
            "noise_type": self.noise_type.value,
            "depolarizing_p": self.config.depolarizing_p,
            "amplitude_gamma": self.config.amplitude_gamma,
            "readout_flip_prob": self.config.readout_flip_prob,
            "shots": self.config.shots,
        }

    def load_state_dict(self, state_dict: dict):
        raw = state_dict.get("noise_type", "none")
        noise_type = NoiseType(raw) if isinstance(raw, str) else raw
        self.config = NoiseConfig(
            noise_type=noise_type,
            depolarizing_p=state_dict.get("depolarizing_p", 0.001),
            amplitude_gamma=state_dict.get("amplitude_gamma", 0.001),
            readout_flip_prob=state_dict.get("readout_flip_prob", 0.01),
            shots=state_dict.get("shots", None),
        )
        self.noise_type = self.config.noise_type


def noisy_quantum_device(noise_config: Optional[NoiseConfig], n_qubits: int) -> object:
    if noise_config is None or noise_config.noise_type == NoiseType.NONE:
        return qml.device('default.qubit', wires=n_qubits)
    return qml.device('default.mixed', wires=n_qubits, shots=noise_config.shots)


# ── Noisy Quantum Circuit ────────────────────────────────────────────────

class NoisyQuantumCircuit:
    def __init__(self, n_qubits: int, n_layers: int, n_readout: int,
                 encoding: str, entanglement: str,
                 noise_config: Optional[NoiseConfig] = None,
                 diff_method: str = 'backprop'):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_readout = n_readout
        self.encoding = encoding
        self.entanglement = entanglement
        self.noise_config = noise_config
        self.diff_method = diff_method if noise_config is None or noise_config.noise_type == NoiseType.NONE else 'parameter-shift'

        self.dev = noisy_quantum_device(noise_config, n_qubits)
        self.entanglement_pairs = build_entanglement_pairs(n_qubits, entanglement)
        self.qnode = self._build_qnode()

    def _build_qnode(self) -> Callable:
        noise_ops = []
        if self.noise_config is not None and self.noise_config.noise_type != NoiseType.NONE:
            nm = NoiseModel(self.noise_config)
            noise_ops = getattr(nm, f'_build_{self.noise_config.noise_type.value}_model', lambda n: [])(self.n_qubits)

        wires = range(self.n_qubits)

        @qml.qnode(self.dev, interface='torch', diff_method=self.diff_method)
        def circuit(inputs: torch.Tensor, weights: torch.Tensor):
            apply_encoding(inputs, wires, self.encoding)
            if noise_ops:
                for op in noise_ops:
                    qml.apply(op)
            for layer_idx in range(self.n_layers):
                apply_variational_layer(weights[layer_idx], wires, self.entanglement_pairs)
                if noise_ops and layer_idx < self.n_layers - 1:
                    for op in noise_ops:
                        qml.apply(op)
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_readout)]
        return circuit

    def forward(self, inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        result = self.qnode(inputs, weights)
        if isinstance(result, (list, tuple)):
            result = torch.stack([torch.as_tensor(r, dtype=torch.float32) for r in result])
        else:
            result = torch.as_tensor(result, dtype=torch.float32)
        return result
