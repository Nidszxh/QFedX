from qiskit import QuantumCircuit
import numpy as np

"""
Angle (rotation) encoding module.
Maps features -> rotation angles on qubits (Rx, Ry, or Rz). We use Ry here.
"""

def pool_to_n_features(vec: np.ndarray, n_features: int) -> np.ndarray:
    # Pool a 1D vector into n_features by averaging contiguous chunks.
    v = np.asarray(vec, dtype=float)
    L = v.size
    if n_features <= 0:
        raise ValueError("n_features must be > 0")
    if n_features >= L:
        out = np.zeros(n_features, dtype=float)
        out[:L] = v
        return out
    chunk_size = L // n_features
    out = np.zeros(n_features, dtype=float)
    for i in range(n_features):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < n_features - 1 else L
        out[i] = v[start:end].mean()
    return out

def angle_encode(features: np.ndarray, n_qubits: int = None, basis: str = 'ry', angle_range: float = np.pi) -> QuantumCircuit:
    """Encode features as single-qubit rotations. Maps normalized features -> [0, angle_range].
    - basis: 'rx', 'ry', or 'rz'
    - angle_range: maximum rotation (e.g. np.pi or 2*np.pi)
    """
    f = np.asarray(features, dtype=float)
    if f.ndim != 1:
        raise ValueError("angle_encode expects a 1D feature vector.")
    if n_qubits is None:
        n_qubits = f.size
    if n_qubits <= 0:
        raise ValueError("n_qubits must be > 0")
    if f.size != n_qubits:
        f = pool_to_n_features(f, n_qubits)

    # normalize to [0,1]
    if np.isclose(f.max(), f.min()):
        normed = np.zeros_like(f)
    else:
        normed = (f - f.min()) / (f.max() - f.min())

    angles = normed * angle_range

    qc = QuantumCircuit(n_qubits, name=f"AngleEncode_{basis.upper()}")
    for i, ang in enumerate(angles):
        if basis.lower() == 'rx':
            qc.rx(ang, i)
        elif basis.lower() == 'rz':
            qc.rz(ang, i)
        else:
            qc.ry(ang, i)
    return qc