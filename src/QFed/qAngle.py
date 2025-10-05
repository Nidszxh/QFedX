"""
Angle (Rotation) Encoding for Quantum Machine Learning

Encodes classical features as rotation angles on qubits using Rx, Ry, or Rz gates.
For n features on n qubits: |0⟩⊗ⁿ → R(θ₁)|0⟩ ⊗ R(θ₂)|0⟩ ⊗ ... ⊗ R(θₙ)|0⟩
"""

from qiskit import QuantumCircuit
import numpy as np
from typing import Optional, Literal

def pool_features(vec: np.ndarray, n_features: int,
                    method: str = 'mean') -> np.ndarray:
    """
    Pool or pad a vector to match target feature count.
    
    When input length != n_features:
    - If longer: pools consecutive chunks using specified method
    - If shorter: pads with zeros
    
    Args:
        vec: Input feature vector
        n_features: Target number of features
        method: Pooling method ('mean', 'max', 'median')
    
    Returns:
        Vector of length n_features
    
    Raises:
        ValueError: If n_features <= 0 or method is invalid
    
    Examples:
        >>> pool_features([1, 2, 3, 4], 2, 'mean')
        array([1.5, 3.5])
    """
    v = np.asarray(vec, dtype=np.float64)
    L = v.size
    
    if n_features <= 0:
        raise ValueError("n_features must be positive")
    
    # Case 1: Padding needed
    if n_features >= L:
        out = np.zeros(n_features, dtype=np.float64)
        out[:L] = v
        return out
    
    # Case 2: Pooling needed
    chunk_size = L / n_features  # Use float division for better distribution
    out = np.zeros(n_features, dtype=np.float64)
    
    pooling_fn = {
        'mean': np.mean,
        'max': np.max,
        'median': np.median
    }.get(method.lower())
    
    if pooling_fn is None:
        raise ValueError(f"Invalid pooling method: {method}. Use 'mean', 'max', or 'median'")
    
    for i in range(n_features):
        start = int(i * chunk_size)
        end = int((i + 1) * chunk_size) if i < n_features - 1 else L
        out[i] = pooling_fn(v[start:end])
    
    return out

def normalize_features(features: np.ndarray, feature_range: tuple = (0, 1),
    epsilon: float = 1e-10) -> np.ndarray:
    """
    Normalize features to specified range using min-max scaling.
    Handles constant features (all same value) by returning midpoint.
    
    Args:
        features: Input feature vector
        feature_range: Target range as (min, max)
        epsilon: Tolerance for detecting constant features
    
    Returns:
        Normalized features in specified range
    
    Examples:
        >>> normalize_features([1, 2, 3], (0, 1))
        array([0. , 0.5, 1. ])
    """
    f = np.asarray(features, dtype=np.float64)
    f_min, f_max = f.min(), f.max()
    
    # Handle constant features
    if f_max - f_min < epsilon:
        midpoint = (feature_range[0] + feature_range[1]) / 2
        return np.full_like(f, midpoint)
    
    # Min-max normalization
    normalized = (f - f_min) / (f_max - f_min)
    
    # Scale to target range
    target_min, target_max = feature_range
    return normalized * (target_max - target_min) + target_min

# Angle Encoding construction
def angle_encode(features: np.ndarray, n_qubits: Optional[int] = None, basis: Literal['rx', 'ry', 'rz'] = 'ry',
                    angle_range: float = np.pi, pooling_method: str = 'mean', name: Optional[str] = None) -> QuantumCircuit:
    """
    Create quantum circuit with angle encoding of classical features.
    
    Encodes features as single-qubit rotations:
        |ψ⟩ = R(θ₁) ⊗ R(θ₂) ⊗ ... ⊗ R(θₙ) |0⟩⊗ⁿ
    where θᵢ ∈ [0, angle_range] is derived from features[i]
    
    Args:
        features: Classical feature vector (1D array)
        n_qubits: Number of qubits (if None, uses len(features))
        basis: Rotation gate type - 'rx', 'ry', or 'rz'
        angle_range: Maximum rotation angle (typically π or 2π)
        pooling_method: Method for feature reduction ('mean', 'max', 'median')
        name: Custom circuit name (default: auto-generated)
    
    Returns:
        QuantumCircuit with n_qubits containing angle-encoded features
    
    Raises:
        ValueError: If features not 1D, n_qubits invalid, or basis invalid
    
    Examples:
        >>> features = [0.2, 0.5, 0.8]
        >>> qc = angle_encode(features, basis='ry')
        >>> qc.num_qubits
        3
    """
    f = np.asarray(features, dtype=np.float64)
    
    # Validation
    if f.ndim != 1:
        raise ValueError(f"Expected 1D feature vector, got shape {f.shape}")
    if f.size == 0:
        raise ValueError("Feature vector cannot be empty")
    
    # Determine qubit count
    if n_qubits is None:
        n_qubits = f.size
    elif n_qubits <= 0:
        raise ValueError("n_qubits must be positive")
    
    # Validate basis
    basis_lower = basis.lower()
    if basis_lower not in ['rx', 'ry', 'rz']:
        raise ValueError(f"Invalid basis '{basis}'. Use 'rx', 'ry', or 'rz'")
    
    # Pool/pad features to match qubit count
    if f.size != n_qubits:
        f = pool_features(f, n_qubits, method=pooling_method)
    
    # Normalize to [0, angle_range]
    angles = normalize_features(f, feature_range=(0, angle_range))
    
    # Build circuit
    if name is None:
        name = f"AngleEncode_{basis_lower.upper()}"
    
    qc = QuantumCircuit(n_qubits, name=name)
    
    # Apply rotations
    rotation_gate = {
        'rx': qc.rx,
        'ry': qc.ry,
        'rz': qc.rz
    }[basis_lower]
    
    for i, angle in enumerate(angles):
        rotation_gate(angle, i)
    
    return qc

# Verify the encoding methods
def get_rotation_angles(features: np.ndarray, n_qubits: Optional[int] = None, 
                        angle_range: float = np.pi, pooling_method: str = 'mean') -> np.ndarray:
    """
    Get the rotation angles that would be applied (without creating circuit).
    
    Useful for debugging or analyzing feature encoding.
    
    Args:
        features: Input feature vector
        n_qubits: Target number of qubits
        angle_range: Maximum rotation angle
        pooling_method: Method for feature reduction
    
    Returns:
        Array of rotation angles
    
    Examples:
        >>> features = [0, 0.5, 1.0]
        >>> angles = get_rotation_angles(features, angle_range=np.pi)
        >>> np.allclose(angles, [0, np.pi/2, np.pi])
        True
    """
    f = np.asarray(features, dtype=np.float64)
    
    if n_qubits is None:
        n_qubits = f.size
    
    if f.size != n_qubits:
        f = pool_features(f, n_qubits, method=pooling_method)
    
    return normalize_features(f, feature_range=(0, angle_range))

def compare_bases(features: np.ndarray, n_qubits: int = 4) -> dict:
    """
    Compare encoding across different rotation bases.
    
    Args:
        features: Input feature vector
        n_qubits: Number of qubits
    
    Returns:
        Dictionary with circuit depths and gate counts for each basis
    
    Examples:
        >>> features = np.random.rand(4)
        >>> stats = compare_bases(features)
        >>> stats['rx']['depth']
        1
    """
    results = {}
    
    for basis in ['rx', 'ry', 'rz']:
        qc = angle_encode(features, n_qubits=n_qubits, basis=basis)
        results[basis] = {
            'depth': qc.depth(),
            'num_gates': sum(qc.count_ops().values()),
            'qubits': qc.num_qubits
        }
    
    return results

# Example Usage and Testing
if __name__ == "__main__":
    print("Angle Encoding Module - Examples\n")
    
    # Example 1: Basic Ry encoding
    print("Example 1: Basic Ry encoding (3 features → 3 qubits)")
    features1 = [0.2, 0.5, 0.8]
    qc1 = angle_encode(features1, basis='ry')
    angles1 = get_rotation_angles(features1)
    print(f"  Input features: {features1}")
    print(f"  Rotation angles: {angles1}")
    print(f"  Circuit depth: {qc1.depth()}")
    print(f"  Qubits: {qc1.num_qubits}\n")
    
    # Example 2: Feature pooling
    print("Example 2: Feature pooling (8 features → 4 qubits)")
    features2 = np.random.rand(8)
    qc2 = angle_encode(features2, n_qubits=4, basis='ry')
    print(f"  Input dimension: {len(features2)}")
    print(f"  Output qubits: {qc2.num_qubits}")
    print(f"  Pooling: 8 → 4 via averaging\n")
    
    # Example 3: Different rotation bases
    print("Example 3: Comparing rotation bases")
    features3 = np.random.rand(4)
    stats = compare_bases(features3, n_qubits=4)
    for basis, info in stats.items():
        print(f"  {basis.upper()}: depth={info['depth']}, gates={info['num_gates']}")
    print()
    
    # Example 4: Constant features (edge case)
    print("Example 4: Constant features (all same value)")
    features4 = [5.0, 5.0, 5.0, 5.0]
    qc4 = angle_encode(features4, basis='ry', angle_range=np.pi)
    angles4 = get_rotation_angles(features4, angle_range=np.pi)
    print(f"  Input features: {features4}")
    print(f"  Rotation angles: {angles4}")
    print(f"  All angles = π/2 (midpoint): {np.allclose(angles4, np.pi/2)}\n")
    
    # Example 5: PCA features with different angle ranges
    print("Example 5: PCA features with 2π range")
    pca_features = [-0.5, 0.0, 0.5, 1.0]  # Typical PCA-scaled values
    qc5 = angle_encode(pca_features, basis='ry', angle_range=2*np.pi)
    angles5 = get_rotation_angles(pca_features, angle_range=2*np.pi)
    print(f"  PCA features: {pca_features}")
    print(f"  Angle range: [0, 2π]")
    print(f"  Rotation angles: {angles5}\n")
    
    # Example 6: Custom pooling methods
    print("Example 6: Different pooling methods")
    features6 = [1, 2, 3, 4, 5, 6]
    for method in ['mean', 'max', 'median']:
        pooled = pool_features(features6, 3, method=method)
        print(f"  {method.capitalize():>6} pooling: {features6} → {pooled}")
    print()
    
    print("✓ All examples completed successfully")