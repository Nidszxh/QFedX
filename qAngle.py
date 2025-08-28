from qiskit import QuantumCircuit
import numpy as np

"""
Angle (rotation) encoding module.
Maps features -> rotation angles on qubits (Rx, Ry, or Rz). We use Ry here.
"""

def pool_to_n_features(vec: np.ndarray, n_features: int) -> np.ndarray:
    """Pool a 1D vector into n_features by averaging contiguous chunks."""
    v = np.asarray(vec, dtype=float)
    L = v.size
    if n_features >= L:
        # pad with zeros
        out = np.zeros(n_features)
        out[:L] = v
        return out
    chunk_size = L // n_features
    out = np.zeros(n_features)
    for i in range(n_features):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < n_features - 1 else L
        out[i] = v[start:end].mean()
    return out


def angle_encode(features: np.ndarray, n_qubits: int = None, basis='ry') -> QuantumCircuit:
    """Encode a feature vector into single-qubit rotations."""
    f = np.asarray(features, dtype=float)
    if n_qubits is None:
        n_qubits = f.size
    if f.size != n_qubits:
        f = pool_to_n_features(f, n_qubits)

    # normalize features to [0,1] based on max for stable angle mapping
    if f.max() == f.min():
        normed = np.zeros_like(f)
    else:
        normed = (f - f.min()) / (f.max() - f.min())

    angles = normed * np.pi  # map to [0, pi]

    qc = QuantumCircuit(n_qubits, name=f"AngleEncode_{basis.upper()}")
    for i, ang in enumerate(angles):
        if basis.lower() == 'rx':
            qc.rx(ang, i)
        elif basis.lower() == 'rz':
            qc.rz(ang, i)
        else:
            qc.ry(ang, i)
    return qc
