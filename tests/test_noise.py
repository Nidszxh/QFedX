import torch

from core.defs import NoiseType
from quantum.noise import NoiseConfig, NoiseModel


class TestNoiseConfig:
    def test_default_values(self):
        cfg = NoiseConfig()
        assert cfg.noise_type == NoiseType.NONE
        assert cfg.depolarizing_p == 0.001
        assert cfg.amplitude_gamma == 0.001
        assert cfg.readout_flip_prob == 0.01

    def test_to_dict(self):
        cfg = NoiseConfig(noise_type=NoiseType.DEPOLARIZING, depolarizing_p=0.01)
        d = cfg.to_dict()
        assert d['noise_type'] == NoiseType.DEPOLARIZING
        assert d['depolarizing_p'] == 0.01


class TestNoiseModel:
    def test_no_noise(self):
        cfg = NoiseConfig(noise_type=NoiseType.NONE)
        nm = NoiseModel(cfg)
        assert nm.noise_type == NoiseType.NONE

    def test_apply_readout_error_no_flip(self):
        cfg = NoiseConfig(readout_flip_prob=0.0)
        nm = NoiseModel(cfg)
        probs = torch.tensor([[0.8, 0.2], [0.1, 0.9]])
        result = nm.apply_readout_error(probs)
        assert torch.allclose(result, probs)

    def test_apply_readout_error_with_flip(self):
        cfg = NoiseConfig(readout_flip_prob=0.1)
        nm = NoiseModel(cfg)
        probs = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        result = nm.apply_readout_error(probs)
        assert result[0, 0] < 1.0
        assert result[0, 1] > 0.0
        assert abs(result.sum().item() - 1.0) < 1e-6

    def test_state_dict(self):
        cfg = NoiseConfig(noise_type=NoiseType.DEPOLARIZING, depolarizing_p=0.01)
        nm = NoiseModel(cfg)
        sd = nm.state_dict()
        assert sd["noise_type"] == NoiseType.DEPOLARIZING

    def test_load_state_dict(self):
        nm = NoiseModel(NoiseConfig())
        nm.load_state_dict({"noise_type": NoiseType.AMPLITUDE_DAMPING, "amplitude_gamma": 0.005,
                           "depolarizing_p": 0.0, "readout_flip_prob": 0.0, "shots": None})
        assert nm.noise_type == NoiseType.AMPLITUDE_DAMPING
        assert nm.config.amplitude_gamma == 0.005

    def test_apply_to_device_noiseless(self):
        cfg = NoiseConfig(noise_type=NoiseType.NONE, depolarizing_p=0.0, amplitude_gamma=0.0)
        nm = NoiseModel(cfg)
        import pennylane as qml
        dev = qml.device('default.qubit', wires=2)
        result = nm.apply_to_device(dev, 2)
        assert result is dev

    def test_apply_to_device_depolarizing(self):
        cfg = NoiseConfig(noise_type=NoiseType.DEPOLARIZING, depolarizing_p=0.01)
        nm = NoiseModel(cfg)
        import pennylane as qml
        dev = qml.device('default.qubit', wires=2)
        result = nm.apply_to_device(dev, 2)
        assert result is not dev

    def test_noise_model_creates_mixed_device(self):
        import pennylane as qml
        cfg = NoiseConfig(noise_type=NoiseType.DEPOLARIZING, depolarizing_p=0.01)
        nm = NoiseModel(cfg)
        dev = qml.device('default.qubit', wires=2)
        result = nm.apply_to_device(dev, 2)
        assert hasattr(result, 'name')
        assert result.name == 'default.mixed'

    def test_noise_model_builders(self):
        cfg = NoiseConfig(noise_type=NoiseType.COMBINED, depolarizing_p=0.01, amplitude_gamma=0.01)
        nm = NoiseModel(cfg)
        import pennylane as qml
        dev = qml.device('default.qubit', wires=2)
        result = nm.apply_to_device(dev, 2)
        assert result is not None
