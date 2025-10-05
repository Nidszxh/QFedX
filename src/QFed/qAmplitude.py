"""
Amplitude Encoding for Quantum Machine Learning
Encodes classical data vectors as quantum state amplitudes on n qubits.
For a vector x of length 2^n, creates |ψ⟩ = Σᵢ xᵢ|i⟩ where ||x||₂ = 1.
"""

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import numpy as np
from typing import Union

def normalize_for_amplitude(vec: np.ndarray) -> np.ndarray:
    """
    Normalize vector to unit L2 norm for valid quantum state.
    
    For quantum amplitude encoding, we require Σᵢ |xᵢ|² = 1.
    If input is all zeros, returns uniform superposition.
    
    Args:
        vec: Input vector (real or complex)
    
    Returns:
        Normalized vector with unit L2 norm
    
    Examples:
        >>> normalize_for_amplitude([3, 4])
        array([0.6, 0.8])
    """
    v = np.asarray(vec, dtype=np.complex128)
    norm = np.linalg.norm(v)
    
    if norm < 1e-12:  # Handle zero vector case
        N = v.size
        return np.ones(N, dtype=np.complex128) / np.sqrt(N)
    
    return v / norm

def pad_to_pow2(vec: np.ndarray) -> np.ndarray:
    """
    Pad vector with zeros to nearest power of 2 length.
    
    Quantum amplitude encoding requires 2^n amplitudes for n qubits.
    This function ensures the input meets this requirement.
    
    Args:
        vec: Input vector of any length > 0
    
    Returns:
        Zero-padded vector with length = 2^n
    
    Raises:
        ValueError: If input vector is empty
    
    Examples:
        >>> pad_to_pow2([1, 2, 3])
        array([1, 2, 3, 0])
    """
    v = np.asarray(vec)
    L = v.size
    
    if L == 0:
        raise ValueError("Input vector must have non-zero length")
    
    # Calculate next power of 2: 2^⌈log₂(L)⌉
    next_pow2 = 1 << (L - 1).bit_length()
    
    if L == next_pow2:
        return v
    
    padded = np.zeros(next_pow2, dtype=v.dtype)
    padded[:L] = v
    return padded

def amplitude_encode(vector: Union[np.ndarray, list], normalize: bool = True,
                    name: str = "AmpEncode") -> QuantumCircuit:
    """
    Create quantum circuit with amplitude encoding of classical data.
    
    Encodes a classical vector into quantum amplitudes:
        |ψ⟩ = Σᵢ xᵢ|i⟩ where xᵢ = vector[i] (normalized)
    
    The encoding requires n = ⌈log₂(len(vector))⌉ qubits.
    Vector is automatically padded to length 2^n if needed.
    
    Args:
        vector: Classical data vector (1D array)
        normalize: Whether to normalize before encoding (default: True)
        name: Name for the quantum circuit
    
    Returns:
        QuantumCircuit with n qubits containing amplitude-encoded state
    
    Raises:
        ValueError: If input is not 1D or is empty
    
    Examples:
        >>> vec = [1, 2, 3]  # Will be normalized and padded
        >>> qc = amplitude_encode(vec)
        >>> qc.num_qubits
        2
    """
    v = np.asarray(vector)
    if v.ndim != 1:
        raise ValueError(f"Expected 1D vector, got shape {v.shape}")
    if v.size == 0:
        raise ValueError("Input vector cannot be empty")
    
    # Preprocess
    v_padded = pad_to_pow2(v)
    
    if normalize:
        state = normalize_for_amplitude(v_padded)
    else:
        state = v_padded.astype(np.complex128)
    
    # Build circuit
    n_qubits = int(np.log2(state.size))
    qc = QuantumCircuit(n_qubits, name=name)
    qc.initialize(state, qc.qubits)
    return qc

def get_statevector(circuit: QuantumCircuit) -> Statevector:
    """
    Extract statevector from a quantum circuit.
    
    Args:
        circuit: Quantum circuit to simulate
    
    Returns:
        Statevector object representing the quantum state
    
    Examples:
        >>> qc = amplitude_encode([1, 0, 0, 0])
        >>> sv = get_statevector(qc)
        >>> np.abs(sv.data[0])
        1.0
    """
    return Statevector.from_instruction(circuit)

# Verify encoding correctness
def verify_encoding(
    original_vector: np.ndarray,
    encoded_circuit: QuantumCircuit,
    tolerance: float = 1e-6
) -> bool:
    """
    Verify that amplitude encoding correctly represents the input vector.
    
    Args:
        original_vector: Original classical vector
        encoded_circuit: Circuit from amplitude_encode()
        tolerance: Maximum allowed difference
    
    Returns:
        True if encoding is correct within tolerance
    
    Examples:
        >>> vec = [0.6, 0.8]
        >>> qc = amplitude_encode(vec)
        >>> verify_encoding(vec, qc)
        True
    """
    # Get encoded state
    sv = get_statevector(encoded_circuit)
    encoded_amps = sv.data
    
    # Prepare expected state
    v = np.asarray(original_vector)
    v_padded = pad_to_pow2(v)
    expected_amps = normalize_for_amplitude(v_padded)
    
    # Compare
    return np.allclose(encoded_amps, expected_amps, atol=tolerance)

def get_required_qubits(vector_length: int) -> int:
    """
    Calculate number of qubits needed for amplitude encoding.
    
    Args:
        vector_length: Length of classical data vector
    
    Returns:
        Number of qubits required (⌈log₂(length)⌉)
    
    Examples:
        >>> get_required_qubits(4)
        2
        >>> get_required_qubits(5)
        3
    """
    if vector_length <= 0:
        raise ValueError("Vector length must be positive")
    return (vector_length - 1).bit_length()

# Example usage and tests
if __name__ == "__main__":
    print("Amplitude Encoding Module - Examples\n")
    
    # Example 1: Simple 2D vector
    print("Example 1: Encoding [0.6, 0.8] (normalized)")
    vec1 = [0.6, 0.8]
    qc1 = amplitude_encode(vec1)
    print(f"  Input length: {len(vec1)}")
    print(f"  Required qubits: {qc1.num_qubits}")
    print(f"  Verification: {verify_encoding(vec1, qc1)}")
    
    sv1 = get_statevector(qc1)
    print(f"  Encoded amplitudes: {np.abs(sv1.data)}\n")
    
    # Example 2: Non-power-of-2 length
    print("Example 2: Encoding [1, 2, 3] (requires padding)")
    vec2 = [1, 2, 3]
    qc2 = amplitude_encode(vec2)
    print(f"  Input length: {len(vec2)}")
    print(f"  Padded length: {2**qc2.num_qubits}")
    print(f"  Required qubits: {qc2.num_qubits}")
    print(f"  Verification: {verify_encoding(vec2, qc2)}")
    
    sv2 = get_statevector(qc2)
    print(f"  Encoded amplitudes: {np.abs(sv2.data)}")
    print(f"  Normalized: {np.allclose(np.sum(np.abs(sv2.data)**2), 1.0)}\n")
    
    # Example 3: PCA-reduced MNIST features
    print("Example 3: PCA-reduced MNIST features")
    pca_features = np.random.randn(4)  # 4 PCA components
    qc3 = amplitude_encode(pca_features, name="MNIST_PCA")
    print(f"  PCA components: {len(pca_features)}")
    print(f"  Required qubits: {qc3.num_qubits}")
    print(f"  Circuit depth: {qc3.depth()}")
    print(f"  Circuit name: {qc3.name}\n")
    
    # Example 4: Zero vector handling
    print("Example 4: Zero vector (edge case)")
    vec4 = [0, 0, 0, 0]
    qc4 = amplitude_encode(vec4)
    sv4 = get_statevector(qc4)
    print(f"  Input: {vec4}")
    print(f"  Encoded as uniform: {np.allclose(np.abs(sv4.data), 0.5)}")
    print(f"  Amplitudes: {np.abs(sv4.data)}\n")
    
    print("✓ All examples completed successfully")