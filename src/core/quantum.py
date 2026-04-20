"""Quantum circuit building blocks — encoding strategies and entanglement topologies."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pennylane as qml
import torch

from core.defs import DIVISION_EPSILON, EncodingType, EntanglementPairs, EntanglementTopology

if TYPE_CHECKING:
    from collections.abc import Sequence


# ── Encoding ──────────────────────────────────────────────────────────────

def apply_angle_encoding(inputs: torch.Tensor, wires: Sequence[int]) -> None:
    angles = (inputs + 1.0) * (np.pi / 2.0)
    qml.AngleEmbedding(angles, wires=wires, rotation="Y")


def apply_amplitude_encoding(inputs: torch.Tensor, wires: Sequence[int]) -> None:
    norm = torch.norm(inputs) + DIVISION_EPSILON
    amplitudes = inputs / norm
    qml.AmplitudeEmbedding(amplitudes, wires=wires, normalize=True, pad_with=0.0)


ENCODING_DISPATCH = {
    "angle": apply_angle_encoding,
    "amplitude": apply_amplitude_encoding,
}


def apply_encoding(inputs: torch.Tensor, wires: Sequence[int], encoding: EncodingType) -> None:
    encoder = ENCODING_DISPATCH.get(encoding)
    if encoder is None:
        raise ValueError(f"Unknown encoding type: {encoding!r}")
    encoder(inputs, wires)


def apply_variational_layer(params: torch.Tensor, wires: Sequence[int], entanglement_pairs: list[tuple[int, int]]) -> None:
    for i in wires:
        qml.RY(params[i, 0], wires=i)
        qml.RZ(params[i, 1], wires=i)
    for ctrl, tgt in entanglement_pairs:
        qml.CNOT(wires=[ctrl, tgt])


# ── Entanglement ──────────────────────────────────────────────────────────

def _build_linear_pairs(num_qubits: int) -> EntanglementPairs:
    return [(i, i + 1) for i in range(num_qubits - 1)]


def _build_circular_pairs(num_qubits: int) -> EntanglementPairs:
    return [(i, (i + 1) % num_qubits) for i in range(num_qubits)]


def _build_full_pairs(num_qubits: int) -> EntanglementPairs:
    return [(i, j) for i in range(num_qubits) for j in range(i + 1, num_qubits)]


_ENTANGLEMENT_DISPATCH: dict[str, callable] = {
    "linear": _build_linear_pairs,
    "circular": _build_circular_pairs,
    "full": _build_full_pairs,
}


def build_entanglement_pairs(
    num_qubits: int,
    topology: EntanglementTopology,
) -> EntanglementPairs:
    builder = _ENTANGLEMENT_DISPATCH.get(topology)
    if builder is None:
        raise ValueError(f"Unknown entanglement topology: {topology!r}")
    return builder(num_qubits)
