from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import numpy as np

"""
Amplitude encoding module.
Takes a downsampled image vector and encodes it as amplitudes on n qubits.
Uses Qiskit's initialize to build a state |psi> = sum_i x_i |i>.
"""

def normalize_for_amplitude(vec: np.ndarray) -> np.ndarray:
    """Normalize a real vector so that sum(|x|^2) = 1 for amplitude encoding.
    If the vector is all zeros, returns a uniform state.
    """
    v = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(v)
    if norm == 0:
        # fallback to uniform superposition
        v = np.ones_like(v) / np.sqrt(v.size)
    else:
        v = v / norm
    return v


def amplitude_encode(vector: np.ndarray) -> QuantumCircuit:
    """Return a QuantumCircuit with the amplitude-initialized state for the input vector.
    vector length must equal 2^n for some n (pad or resize beforehand).
    """
    v = np.asarray(vector, dtype=float)
    # ensure length is power of two
    L = v.size
    if not (L != 0 and ((L & (L - 1)) == 0)):
        raise ValueError("Vector length must be a power of 2 for amplitude encoding. Got {}".format(L))
    n_qubits = int(np.log2(L))

    # normalize to unit norm
    state = normalize_for_amplitude(v)

    qc = QuantumCircuit(n_qubits, name="AmplitudeEncode")
    qc.initialize(state, qc.qubits)
    return qc


def get_statevector_from_circuit(qc: QuantumCircuit):
    sv = Statevector.from_instruction(qc)
    return sv