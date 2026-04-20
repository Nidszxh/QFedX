import pytest
import torch

from quantum.qnn import ClassicalPreprocessor, QuantumNeuralNetwork, QuantumNeuralNetworkConfig


class TestClassicalPreprocessor:
    def test_output_shape(self):
        pp = ClassicalPreprocessor(16, 4, 'angle')
        x = torch.randn(8, 16)
        out = pp(x)
        assert out.shape == (8, 4)

    def test_angle_encoding_output_range(self):
        pp = ClassicalPreprocessor(16, 4, 'angle')
        x = torch.randn(8, 16)
        out = pp(x)
        assert out.min() >= -1.0
        assert out.max() <= 1.0


class TestQuantumNeuralNetwork:
    def test_output_shape_angle(self):
        config = QuantumNeuralNetworkConfig(n_qubits=4, n_layers=2, n_readout=4, n_classes=3,
                           n_features=16, encoding='angle')
        model = QuantumNeuralNetwork(config)
        x = torch.randn(8, 16)
        out = model(x)
        assert out.shape == (8, 3)

    def test_output_shape_amplitude(self):
        config = QuantumNeuralNetworkConfig(n_qubits=2, n_layers=1, n_readout=2, n_classes=3,
                           n_features=8, encoding='amplitude')
        model = QuantumNeuralNetwork(config)
        x = torch.randn(8, 8)
        out = model(x)
        assert out.shape == (8, 3)

    def test_gradient_flow(self):
        config = QuantumNeuralNetworkConfig(n_qubits=4, n_layers=2, n_readout=4, n_classes=3,
                           n_features=16, encoding='angle')
        model = QuantumNeuralNetwork(config)
        x = torch.randn(4, 16)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert model.q_weights.grad is not None
        assert model.q_weights.grad.norm().item() > 0
        assert model.classifier[1].weight.grad is not None

    def test_batched_fallback(self):
        config = QuantumNeuralNetworkConfig(n_qubits=4, n_layers=2, n_readout=4, n_classes=3,
                           n_features=16, encoding='angle')
        model = QuantumNeuralNetwork(config)
        model.quantum_circuit._is_batched = False
        x = torch.randn(8, 16)
        out = model(x)
        assert out.shape == (8, 3)

    def test_single_sample(self):
        config = QuantumNeuralNetworkConfig(n_qubits=4, n_layers=2, n_readout=4, n_classes=3,
                           n_features=16, encoding='angle')
        model = QuantumNeuralNetwork(config).eval()
        x = torch.randn(1, 16)
        out = model(x)
        assert out.shape == (1, 3)

    def test_classical_params(self):
        config = QuantumNeuralNetworkConfig(n_qubits=4, n_layers=2, n_readout=4, n_classes=3,
                           n_features=16, encoding='angle')
        model = QuantumNeuralNetwork(config)
        params = model.get_classical_params()
        assert len(params) > 0

    def test_quantum_params(self):
        config = QuantumNeuralNetworkConfig(n_qubits=4, n_layers=2, n_readout=4, n_classes=3,
                           n_features=16, encoding='angle')
        model = QuantumNeuralNetwork(config)
        params = model.get_quantum_params()
        assert len(params) == 1
        assert params[0].shape == (2, 4, 2)


class TestConfigValidation:
    def test_n_readout_default(self):
        config = QuantumNeuralNetworkConfig(n_qubits=6, n_readout=None)
        assert config.n_readout == 6

    def test_n_readout_error(self):
        with pytest.raises(ValueError, match="n_readout"):
            QuantumNeuralNetworkConfig(n_qubits=4, n_readout=5)

    def test_backprop_shots(self):
        config = QuantumNeuralNetworkConfig(shots=100)
        assert config.diff_method == 'parameter-shift'

    def test_noise_changes_diff_method(self):
        config = QuantumNeuralNetworkConfig(noise_type='depolarizing', shots=None)
        assert config.diff_method == 'parameter-shift'

    def test_to_dict(self):
        config = QuantumNeuralNetworkConfig(n_qubits=8, n_layers=4)
        d = config.to_dict()
        assert d['n_qubits'] == 8
        assert d['encoding'] == 'amplitude'


