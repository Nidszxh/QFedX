from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import numpy as np

"""
Amplitude encoding module.
Takes a downsampled image vector and encodes it as amplitudes on n qubits.
Uses Qiskit's initialize to build a state |psi> = sum_i x_i |i>.
"""

def normalize_for_amplitude(vec: np.ndarray) -> np.ndarray:
    """Normalize a (possibly complex) vector so that sum(|x|^2) = 1.
    If the vector is all zeros, returns the uniform state of the same length.
    """
    v = np.asarray(vec, dtype=np.complex128)
    norm = np.linalg.norm(v)
    if norm == 0:
        N = v.size
        return np.ones(N, dtype=np.complex128) / np.sqrt(N)
    return v / norm

def pad_to_pow2(v: np.ndarray) -> np.ndarray:
    """
    Pad the vector with zeros so its length is the nearest power of 2.
    Example: [1,2,3] -> [1,2,3,0]
    """
    v = np.asarray(v)
    L = v.size
    if L == 0:
        raise ValueError("Input vector must have non-zero length.")
    next_pow2 = 1 << (L - 1).bit_length()  # nearest power of two >= L
    if L != next_pow2:
        padded = np.zeros(next_pow2, dtype=v.dtype)
        padded[:L] = v
        return padded
    return v

def amplitude_encode(vector: np.ndarray) -> QuantumCircuit:
    """
    Return a QuantumCircuit with amplitude encoding.
    Automatically pads input to nearest 2^n length.
    Output circuit has n qubits, where n = ceil(log2(len(vector))).
    """
    v = np.asarray(vector)
    if v.ndim != 1:
        raise ValueError("amplitude_encode expects a 1D vector.")
    v_padded = pad_to_pow2(v)
    state = normalize_for_amplitude(v_padded)
    n_qubits = int(np.log2(state.size))

    qc = QuantumCircuit(n_qubits, name="AmplitudeEncode")
    qc.initialize(state, qc.qubits)
    return qc

def get_statevector_from_circuit(qc: QuantumCircuit) -> Statevector:
    sv = Statevector.from_instruction(qc)
    return sv