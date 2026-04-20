
import pytest
import torch

from core.fl import federated_averaging
from core.utils import set_seed
from quantum.qfl import QuantumFederatedLearning
from quantum.qnn import QuantumNeuralNetworkConfig


class TestFederatedAveraging:
    def test_basic_averaging(self):
        global_params = {
            "w": torch.tensor([1.0, 2.0, 3.0]),
            "b": torch.tensor([0.0]),
        }
        updates = [
            ({"w": torch.tensor([1.0, 1.0, 1.0]), "b": torch.tensor([0.1])}, 10, 0.5),
            ({"w": torch.tensor([3.0, 3.0, 3.0]), "b": torch.tensor([-0.1])}, 20, 0.3),
        ]
        aggregated, avg_loss = federated_averaging(updates, global_params)
        assert "w" in aggregated
        assert "b" in aggregated
        expected_w = (1/3)*1.0 + (2/3)*3.0  # ~2.333
        assert abs(aggregated["w"][0].item() - expected_w) < 1e-4
        assert abs(avg_loss - (10/30*0.5 + 20/30*0.3)) < 1e-4

    def test_integer_params_preserved(self):
        global_params = {
            "w": torch.tensor([1.0]),
            "num_batches_tracked": torch.tensor(5, dtype=torch.long),
        }
        updates = [
            ({"w": torch.tensor([2.0]), "num_batches_tracked": torch.tensor(5, dtype=torch.long)}, 10, 0.0),
        ]
        aggregated, _ = federated_averaging(updates, global_params)
        assert aggregated["num_batches_tracked"].item() == 5

    def test_empty_updates(self):
        global_params = {"w": torch.tensor([1.0])}
        with pytest.raises(ValueError, match="No client updates"):
            federated_averaging([], global_params)

    def test_wrap_angles(self):
        global_params = {"q_weights": torch.tensor([3.5, -3.5])}
        updates = [
            ({"q_weights": torch.tensor([3.5, -3.5])}, 10, 0.0),
        ]
        aggregated, _ = federated_averaging(updates, global_params, wrap_angles=True)
        assert -torch.pi <= aggregated["q_weights"].min() <= torch.pi


class TestQuantumFederatedLearning:
    def test_initialization(self):
        config = QuantumNeuralNetworkConfig(n_qubits=2, n_layers=1, n_readout=2, n_classes=3,
                           n_features=4, encoding='angle')
        fl_config = {
            'num_rounds': 2, 'local_epochs': 1, 'batch_size': 4,
            'classical_lr': 1e-3, 'quantum_lr': 5e-4,
            'client_fraction': 1.0,
            'dp_enabled': False, 'secure_aggregation': False,
        }
        qfl = QuantumFederatedLearning(config, fl_config, device='cpu')
        assert qfl.global_model is not None

    def test_dp_initialization(self):
        config = QuantumNeuralNetworkConfig(n_qubits=2, n_layers=1, n_readout=2, n_classes=3,
                           n_features=4, encoding='angle')
        fl_config = {
            'num_rounds': 2, 'local_epochs': 1, 'batch_size': 4,
            'classical_lr': 1e-3, 'quantum_lr': 5e-4,
            'num_clients': 3, 'client_fraction': 1.0,
            'dp_enabled': True, 'dp_delta': 1e-5,
            'secure_aggregation': False,
        }
        qfl = QuantumFederatedLearning(config, fl_config, device='cpu')
        assert qfl.privacy_accountant is not None

    def test_secure_agg_initialization(self):
        config = QuantumNeuralNetworkConfig(n_qubits=2, n_layers=1, n_readout=2, n_classes=3,
                           n_features=4, encoding='angle')
        fl_config = {
            'num_rounds': 2, 'local_epochs': 1, 'batch_size': 4,
            'classical_lr': 1e-3, 'quantum_lr': 5e-4,
            'num_clients': 3, 'client_fraction': 1.0,
            'dp_enabled': False,
            'secure_aggregation': True,
        }
        qfl = QuantumFederatedLearning(config, fl_config, device='cpu')
        assert qfl.secure_aggregator is not None

    def test_save_results(self, tmp_path):
        config = QuantumNeuralNetworkConfig(n_qubits=2, n_layers=1, n_readout=2, n_classes=3,
                           n_features=4, encoding='angle')
        fl_config = {
            'num_rounds': 2, 'local_epochs': 1, 'batch_size': 4,
            'classical_lr': 1e-3, 'quantum_lr': 5e-4,
            'client_fraction': 1.0,
            'dp_enabled': False, 'secure_aggregation': False,
        }
        qfl = QuantumFederatedLearning(config, fl_config, device='cpu')
        qfl.save_results(save_dir=str(tmp_path))
        pt_files = list(tmp_path.glob("*.pt"))
        csv_files = list(tmp_path.glob("*.csv"))
        assert len(pt_files) >= 1
        assert len(csv_files) >= 1

    def test_evaluate_global_basic(self):
        config = QuantumNeuralNetworkConfig(n_qubits=2, n_layers=1, n_readout=2, n_classes=3,
                           n_features=4, encoding='angle')
        fl_config = {
            'num_rounds': 2, 'local_epochs': 1, 'batch_size': 4,
            'classical_lr': 1e-3, 'quantum_lr': 5e-4,
            'client_fraction': 1.0,
            'dp_enabled': False, 'secure_aggregation': False,
        }
        qfl = QuantumFederatedLearning(config, fl_config, device='cpu')
        test_data = (torch.randn(8, 4), torch.randint(0, 3, (8,)))
        loss, acc = qfl.evaluate_global(test_data)
        assert 0.0 <= acc <= 1.0
        assert loss >= 0

    def test_set_seed(self):
        import random
        set_seed(42)
        a = random.random()
        set_seed(42)
        b = random.random()
        assert a == b
