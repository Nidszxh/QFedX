"""Quantum ML components: QNN, QFL, noise models, privacy, and visualization modules."""
from quantum.qfl import QuantumFederatedLearning
from quantum.qnn import (
    SPSA,
    ClassicalPreprocessor,
    QuantumNeuralNetwork,
    QuantumNeuralNetworkConfig,
    QuantumNeuralNetworkTrainer,
    load_iris_data,
    verify_gradients,
)

try:
    from quantum.noise import NoiseConfig, NoiseModel, NoisyQuantumCircuit
    __all__ = [
        "ClassicalPreprocessor",
        "QuantumNeuralNetwork",
        "QuantumNeuralNetworkConfig",
        "QuantumNeuralNetworkTrainer",
        "SPSA",
        "load_iris_data",
        "verify_gradients",
        "QuantumFederatedLearning",
        "NoiseConfig",
        "NoiseModel",
        "NoisyQuantumCircuit",
    ]
except ImportError:
    __all__ = [
        "ClassicalPreprocessor",
        "QuantumNeuralNetwork",
        "QuantumNeuralNetworkConfig",
        "QuantumNeuralNetworkTrainer",
        "SPSA",
        "load_iris_data",
        "verify_gradients",
        "QuantumFederatedLearning",
    ]
