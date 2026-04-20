"""Quantum Federated Learning orchestration for QFedX."""
import csv
import os
import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from core.data import evaluate_model
from core.defs import DEFAULT_ARTIFACTS_DIR, DEFAULT_MLFLOW_URI, DEFAULT_VISUALIZATION_ROOT, NoiseType
from core.fl import FederatedLearningConfig, federated_averaging
from core.utils import get_logger
from data.preprocess import preprocess_mnist
from quantum.qnn import QuantumNeuralNetwork, QuantumNeuralNetworkConfig, QuantumNeuralNetworkTrainer

logger = get_logger(__name__)

try:
    from quantum.privacy.differential_privacy import DifferentialPrivacy
    from quantum.privacy.privacy_accountant import PrivacyAccountant
    from quantum.privacy.secure_aggregation import SecureAggregator
    PRIVACY_AVAILABLE = True
except ImportError:
    logger.warning("Privacy modules not available — DP and secure aggregation disabled")
    PRIVACY_AVAILABLE = False
    DifferentialPrivacy = None
    PrivacyAccountant = None
    SecureAggregator = None

os.environ.setdefault("MLFLOW_TRACKING_URI", DEFAULT_MLFLOW_URI)
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    logger.warning("mlflow not available — metrics will not be tracked")
    MLFLOW_AVAILABLE = False


def _dict_to_fl_config(d: dict[str, Any]) -> FederatedLearningConfig:
    return FederatedLearningConfig(
        num_rounds=d.get("num_rounds", 5),
        local_epochs=d.get("local_epochs", 3),
        batch_size=d.get("batch_size", 16),
        client_fraction=d.get("client_fraction", 1.0),
        classical_lr=d.get("classical_lr", 1e-3),
        quantum_lr=d.get("quantum_lr", 5e-4),
        weight_decay=d.get("weight_decay", 1e-4),
        grad_clip=d.get("grad_clip", 1.0),
        num_clients=d.get("num_clients", 4),
        dp_enabled=d.get("dp_enabled", False),
        dp_clip_norm=d.get("dp_clip_norm", 1.0),
        dp_noise_multiplier=d.get("dp_noise_multiplier", 1.0),
        dp_delta=d.get("dp_delta", 1e-5),
        secure_aggregation=d.get("secure_aggregation", False),
        early_stop_patience=d.get("early_stop_patience", 0),
        early_stop_delta=d.get("early_stop_delta", 1e-4),
        checkpoint_dir=d.get("checkpoint_dir", ""),
        checkpoint_interval=d.get("checkpoint_interval", 5),
    )


class QuantumFederatedLearning:
    def __init__(self, config: QuantumNeuralNetworkConfig, fl_config: dict, device: Optional[str] = None):
        self.config = config
        self.fl_config = _dict_to_fl_config(fl_config) if not isinstance(fl_config, FederatedLearningConfig) else fl_config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.global_model = QuantumNeuralNetwork(config).to(self.device)
        self.test_accuracies: list[float] = []
        self.test_losses: list[float] = []
        self.train_losses: list[float] = []
        self.client_losses_history: list[list[float]] = []
        self.privacy_accountant: Optional[Any] = None
        self.secure_aggregator: Optional[Any] = None
        self.privacy_eps_history: list[float] = []
        self.dp_global_norms: list[float] = []
        if PRIVACY_AVAILABLE and self.fl_config.dp_enabled:
            self.privacy_accountant = PrivacyAccountant(delta=self.fl_config.dp_delta)
        if PRIVACY_AVAILABLE and self.fl_config.secure_aggregation:
            self.secure_aggregator = SecureAggregator(num_clients=self.fl_config.num_clients)
        self._best_acc = 0.0
        self._patience_counter = 0
        self._best_state_dict = None
        self._print_initialization_info()

    def _print_initialization_info(self):
        logger.info("QUANTUM FEDERATED LEARNING SYSTEM")
        logger.info("Device: %s", self.device)
        logger.info("Quantum Configuration:")
        logger.info("  Qubits: %d", self.config.n_qubits)
        logger.info("  Layers: %d", self.config.n_layers)
        logger.info("  Encoding: %s", self.config.encoding)
        logger.info("  Entanglement: %s", self.config.entanglement)
        logger.info("  Noise: %s", self.config.noise_type)
        fc = self.fl_config
        logger.info("Privacy:")
        logger.info("  Differential Privacy: %s", 'Enabled' if fc.dp_enabled else 'Disabled')
        if fc.dp_enabled:
            logger.info("    Clip Norm: %s", fc.dp_clip_norm)
            logger.info("    Noise Multiplier: %s", fc.dp_noise_multiplier)
            logger.info("    Delta: %s", fc.dp_delta)
        logger.info("  Secure Aggregation: %s", 'Enabled' if fc.secure_aggregation else 'Disabled')
        logger.info("Federated Learning Configuration:")
        logger.info("  Rounds: %d", fc.num_rounds)
        logger.info("  Local epochs: %d", fc.local_epochs)
        logger.info("  Batch size: %d", fc.batch_size)
        logger.info("  Client fraction: %.2f", fc.client_fraction)
        total_params = sum(p.numel() for p in self.global_model.parameters())
        q_params = sum(p.numel() for p in self.global_model.get_quantum_params())
        c_params = sum(p.numel() for p in self.global_model.get_classical_params())
        logger.info("Model Architecture:")
        logger.info("  Total parameters: %d", total_params)
        logger.info("  Quantum parameters: %d (%.1f%%)", q_params, 100*q_params/total_params)
        logger.info("  Classical parameters: %d (%.1f%%)", c_params, 100*c_params/total_params)

    def _build_local_config(self) -> QuantumNeuralNetworkConfig:
        fc = self.fl_config
        return QuantumNeuralNetworkConfig(
            n_qubits=self.config.n_qubits, n_features=self.config.n_features,
            n_classes=self.config.n_classes, encoding=self.config.encoding,
            n_layers=self.config.n_layers, n_readout=self.config.n_readout,
            entanglement=self.config.entanglement,
            batch_size=fc.batch_size, epochs=fc.local_epochs,
            classical_lr=fc.classical_lr, quantum_lr=fc.quantum_lr,
            weight_decay=fc.weight_decay, grad_clip=fc.grad_clip or 0.0,
            device_name=self.config.device_name,
            diff_method=self.config.diff_method, shots=self.config.shots,
            noise_type=self.config.noise_type,
            depolarizing_p=self.config.depolarizing_p,
            amplitude_gamma=self.config.amplitude_gamma,
            readout_flip_prob=self.config.readout_flip_prob,
            dp_enabled=fc.dp_enabled, dp_clip_norm=fc.dp_clip_norm,
            dp_noise_multiplier=fc.dp_noise_multiplier,
        )

    def train_local_client(
        self, client_data: tuple[torch.Tensor, torch.Tensor], client_id: int
    ) -> tuple[dict, int, float]:
        X_client, y_client = client_data
        X_client = torch.as_tensor(X_client, dtype=torch.float32).to(self.device)
        y_client = torch.as_tensor(y_client, dtype=torch.long).to(self.device)

        local_model = QuantumNeuralNetwork(self.config).to(self.device)
        local_model.load_state_dict(self.global_model.state_dict())

        train_loader = DataLoader(
            TensorDataset(X_client, y_client),
            batch_size=self.fl_config.batch_size, shuffle=True, num_workers=0,
            pin_memory=(self.device == 'cuda'),
        )

        local_config = self._build_local_config()

        try:
            trainer = QuantumNeuralNetworkTrainer(local_model, local_config, self.device)
            for epoch in range(local_config.epochs):
                epoch_loss, _ = trainer.train_epoch(train_loader)
                if isinstance(epoch_loss, torch.Tensor) and (torch.isnan(epoch_loss) or torch.isinf(epoch_loss)):
                    logger.warning("NaN/Inf loss at epoch %d client %d", epoch, client_id)
                    return self.global_model.state_dict(), len(X_client), float('inf')
                if self.fl_config.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(local_model.parameters(), self.fl_config.grad_clip)

            avg_loss = trainer.train_losses[-1] if trainer.train_losses else float('inf')
            return local_model.state_dict(), len(X_client), avg_loss
        except Exception as e:
            logger.warning("Client %d training failed: %s", client_id, e)
            return self.global_model.state_dict(), len(X_client), float('inf')


    def federated_round(self, client_data: list[tuple[torch.Tensor, torch.Tensor]], round_num: int) -> float:
        num_clients = len(client_data)
        num_selected = max(1, int(self.fl_config.client_fraction * num_clients))
        selected_clients = random.sample(range(num_clients), num_selected)

        logger.info("Round %d: Selected %d/%d clients %s", round_num, num_selected, num_clients, selected_clients)

        client_updates = []
        round_losses = []

        for client_id in selected_clients:
            state_dict, num_samples, train_loss = self.train_local_client(
                client_data[client_id], client_id
            )
            client_updates.append((state_dict, num_samples, train_loss))
            round_losses.append(train_loss)
            logger.info("  Client %d: Loss = %.4f, Samples = %d", client_id, train_loss, num_samples)

        self.client_losses_history.append(round_losses)

        if not client_updates:
            logger.warning("  No successful client updates")
            return float('inf')

        use_sa = self.secure_aggregator is not None and self.fl_config.secure_aggregation
        if use_sa and PRIVACY_AVAILABLE:
            logger.info("  Applying Secure Aggregation...")
            param_shapes = {k: v.shape for k, v in self.global_model.state_dict().items()}
            all_masked_updates = []
            for client_id, (state_dict, n_samples, loss) in enumerate(client_updates):
                masks = self.secure_aggregator.generate_masks_for_client(
                    client_id, param_shapes, dtype=torch.float32,
                    device=torch.device(self.device),
                )
                masked_update = self.secure_aggregator.mask_params(state_dict, masks)
                all_masked_updates.append((masked_update, n_samples, loss, masks))

            aggregated_params, avg_train_loss = federated_averaging(
                [(m, n, loss_val) for m, n, loss_val, _ in all_masked_updates],
                self.global_model.state_dict(),
                device=torch.device(self.device), wrap_angles=True,
            )
            client_masks = [m for _, _, _, m in all_masked_updates]
            aggregated_params = self.secure_aggregator.unmask_params(aggregated_params, client_masks)
        else:
            try:
                aggregated_params, avg_train_loss = federated_averaging(
                    client_updates, self.global_model.state_dict(),
                    device=torch.device(self.device), wrap_angles=True,
                )
            except Exception as e:
                logger.error("  Aggregation failed: %s", e)
                return float('inf')

        self.global_model.load_state_dict(aggregated_params)

        if self.privacy_accountant is not None and PRIVACY_AVAILABLE:
            q = num_selected / max(num_clients, 1)
            sigma = self.fl_config.dp_noise_multiplier
            self.privacy_accountant.step(q, sigma)
            eps = self.privacy_accountant.get_privacy_spent()
            self.privacy_eps_history.append(eps)
            logger.info("  Privacy budget (ε): %.4f", eps)
            if MLFLOW_AVAILABLE:
                mlflow.log_metric("privacy_epsilon", eps, step=round_num)

        logger.info("  Aggregated train loss: %.4f", avg_train_loss)
        return avg_train_loss

    @torch.no_grad()
    def evaluate_global(self, test_data: tuple[torch.Tensor, torch.Tensor]) -> tuple[float, float]:
        X_test, y_test = test_data
        X_test = torch.as_tensor(X_test, dtype=torch.float32)
        y_test = torch.as_tensor(y_test, dtype=torch.long)

        test_loader = DataLoader(
            TensorDataset(X_test, y_test),
            batch_size=self.fl_config.batch_size * 2,
            shuffle=False, num_workers=0,
            pin_memory=(self.device == 'cuda'),
        )
        return evaluate_model(
            self.global_model, test_loader,
            criterion=nn.CrossEntropyLoss(),
            device=torch.device(self.device),
        )

    def train(self, client_data: list[tuple[torch.Tensor, torch.Tensor]], test_data: tuple[torch.Tensor, torch.Tensor]) -> dict:
        num_rounds = self.fl_config.num_rounds

        logger.info("Starting Quantum Federated Learning Training")
        if self.config.noise_type != NoiseType.NONE:
            logger.info("  Noise model: %s", self.config.noise_type.value)
            logger.info("    Depolarizing p=%s, Amplitude γ=%s", self.config.depolarizing_p, self.config.amplitude_gamma)

        fc = self.fl_config
        logger.info("Round 0: Initial Evaluation")
        initial_acc, initial_loss = self.evaluate_global(test_data)
        self.test_accuracies.append(initial_acc)
        self.test_losses.append(initial_loss)
        self.train_losses.append(0.0)
        self._best_acc = initial_acc
        self._best_state_dict = self.global_model.state_dict()
        logger.info("  Test Accuracy: %.4f", initial_acc)
        logger.info("  Test Loss: %.4f", initial_loss)
        if fc.early_stop_patience > 0:
            logger.info("Early stopping: patience=%d, delta=%s", fc.early_stop_patience, fc.early_stop_delta)
        if fc.checkpoint_dir:
            logger.info("Checkpoint dir: %s", fc.checkpoint_dir)

        for round_num in range(1, num_rounds + 1):
            avg_train_loss = self.federated_round(client_data, round_num)
            test_acc, test_loss = self.evaluate_global(test_data)
            self.test_accuracies.append(test_acc)
            self.test_losses.append(test_loss)
            self.train_losses.append(avg_train_loss)

            if test_acc > self._best_acc - fc.early_stop_delta:
                if test_acc > self._best_acc:
                    self._best_acc = test_acc
                    self._best_state_dict = self.global_model.state_dict()
                self._patience_counter = 0
            else:
                self._patience_counter += 1

            if MLFLOW_AVAILABLE:
                metrics = {"test_acc": test_acc, "test_loss": test_loss, "train_loss": avg_train_loss}
                if self.privacy_accountant is not None:
                    metrics["privacy_epsilon"] = self.privacy_accountant.get_privacy_spent()
                mlflow.log_metrics(metrics, step=round_num)

            if fc.checkpoint_interval > 0 and round_num % fc.checkpoint_interval == 0:
                self.save_checkpoint(round_num)

            if round_num % 5 == 0 or round_num == num_rounds:
                logger.info("\n%s", '='*70)
                logger.info("Round %d Summary:", round_num)
                logger.info("  Test Accuracy: %.4f", test_acc)
                logger.info("  Test Loss: %.4f", test_loss)
                logger.info("  Train Loss: %.4f", avg_train_loss)
                logger.info("%s", '='*70)

            if fc.early_stop_patience > 0 and self._patience_counter >= fc.early_stop_patience:
                logger.info("Early stopping triggered at round %d (test acc did not improve for %d rounds)",
                            round_num, self._patience_counter)
                break

        if self._best_state_dict is not None:
            self.global_model.load_state_dict(self._best_state_dict)
            self.save_checkpoint(round_num, is_best=True)

        self._print_final_summary()
        best_acc = max(self.test_accuracies[1:]) if len(self.test_accuracies) > 1 else 0.0

        return {
            'model': self.global_model,
            'test_accuracies': self.test_accuracies,
            'test_losses': self.test_losses,
            'train_losses': self.train_losses,
            'client_losses': self.client_losses_history,
            'final_accuracy': self.test_accuracies[-1],
            'best_accuracy': best_acc,
        }

    def _print_final_summary(self):
        """
        Print the final summary of the training process, including the best test accuracy and final loss.
        """
        logger.info("TRAINING COMPLETE")
        final_acc = self.test_accuracies[-1]
        best_acc = max(self.test_accuracies[1:]) if len(self.test_accuracies) > 1 else final_acc
        final_loss = self.test_losses[-1]
        logger.info("Final Test Accuracy: %.4f", final_acc)
        logger.info("Best Test Accuracy:  %.4f", best_acc)
        logger.info("Final Test Loss:     %.4f", final_loss)

    def save_checkpoint(self, round_num: int, is_best: bool = False) -> str:
        save_dir = Path(self.fl_config.checkpoint_dir) if self.fl_config.checkpoint_dir else Path("./checkpoints")
        save_dir.mkdir(parents=True, exist_ok=True)
        filename = "best_model.pt" if is_best else f"checkpoint_round_{round_num}.pt"
        path = save_dir / filename
        torch.save({
            'round': round_num,
            'model_state_dict': self.global_model.state_dict(),
            'config': self.config.to_dict(),
            'fl_config': self.fl_config,
            'test_accuracies': self.test_accuracies,
            'test_losses': self.test_losses,
            'train_losses': self.train_losses,
            'client_losses': self.client_losses_history,
            'privacy_eps_history': self.privacy_eps_history,
            'best_acc': self._best_acc,
        }, path)
        logger.info("Checkpoint saved: %s", path)
        return str(path)

    @classmethod
    def load_checkpoint(cls, path: str, device: Optional[str] = None) -> "QuantumFederatedLearning":
        ckpt = torch.load(path, map_location=torch.device(device or 'cpu'))
        config = QuantumNeuralNetworkConfig(**ckpt['config'])
        fl_config = ckpt['fl_config']
        instance = cls(config, fl_config, device)
        instance.global_model.load_state_dict(ckpt['model_state_dict'])
        instance.test_accuracies = ckpt.get('test_accuracies', [])
        instance.test_losses = ckpt.get('test_losses', [])
        instance.train_losses = ckpt.get('train_losses', [])
        instance.client_losses_history = ckpt.get('client_losses', [])
        instance.privacy_eps_history = ckpt.get('privacy_eps_history', [])
        instance._best_acc = ckpt.get('best_acc', 0.0)
        logger.info("Checkpoint loaded from round %d: %s", ckpt.get('round', 0), path)
        return instance

    def save_results(self, save_dir: str = ""):
        save_dir = save_dir or str(DEFAULT_ARTIFACTS_DIR)
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        checkpoint = {
            'model_state_dict': self.global_model.state_dict(),
            'config': self.config.to_dict(),
            'fl_config': self.fl_config,
            'test_accuracies': self.test_accuracies,
            'test_losses': self.test_losses,
            'train_losses': self.train_losses,
            'client_losses': self.client_losses_history,
            'best_accuracy': max(self.test_accuracies[1:]) if len(self.test_accuracies) > 1 else 0.0,
        }
        if self.privacy_accountant is not None:
            checkpoint['privacy_eps_history'] = self.privacy_eps_history
            checkpoint['final_privacy_epsilon'] = self.privacy_accountant.get_privacy_spent()
        if self.config.noise_type != NoiseType.NONE:
            checkpoint['noise_config'] = {
                'noise_type': self.config.noise_type.value,
                'depolarizing_p': self.config.depolarizing_p,
                'amplitude_gamma': self.config.amplitude_gamma,
                'readout_flip_prob': self.config.readout_flip_prob
            }
        model_path = Path(save_dir) / "quantum_federated_model.pt"
        torch.save(checkpoint, model_path)
        logger.info("Model saved: %s", model_path)

        csv_path = Path(save_dir) / "qfl_metrics.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            headers = ["Round", "Test_Accuracy", "Test_Loss", "Train_Loss"]
            if self.privacy_accountant is not None:
                headers.append("Privacy_Epsilon")
            writer.writerow(headers)
            for i in range(len(self.test_accuracies)):
                row = [
                    i, f"{self.test_accuracies[i]:.6f}",
                    f"{self.test_losses[i]:.6f}",
                    f"{self.train_losses[i]:.6f}"
                ]
                if self.privacy_accountant is not None and i > 0 and i-1 < len(self.privacy_eps_history):
                    row.append(f"{self.privacy_eps_history[i-1]:.4f}")
                writer.writerow(row)
        logger.info("Metrics saved: %s", csv_path)


def _parse_qfl_args():
    import argparse
    p = argparse.ArgumentParser(description="QFedX: Quantum Federated Learning")
    p.add_argument('--noise', type=str, default='none', choices=['none', 'depolarizing', 'amplitude_damping', 'readout', 'combined'])
    p.add_argument('--depolarizing-p', type=float, default=0.001)
    p.add_argument('--amplitude-gamma', type=float, default=0.001)
    p.add_argument('--readout-flip', type=float, default=0.01)
    p.add_argument('--dp-enable', action='store_true')
    p.add_argument('--dp-clip', type=float, default=1.0)
    p.add_argument('--dp-noise', type=float, default=1.0)
    p.add_argument('--dp-delta', type=float, default=1e-5)
    p.add_argument('--secure-agg', action='store_true')
    p.add_argument('--rounds', type=int, default=5)
    p.add_argument('--local-epochs', type=int, default=3)
    p.add_argument('--qubits', type=int, default=4)
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--clients', type=int, default=4)
    return p.parse_args()


def _load_or_synthesize_data(qnn_config, data_config):
    try:
        result = preprocess_mnist(**data_config, generate_plots=False)
        if result is None:
            raise ValueError("Preprocessing returned None")
        train_data, val_data, test_data, client_data = result
        qnn_config.n_features = client_data[0][0].shape[1]
        logger.info(" Data loaded: %d clients, %d features, %d test samples",
                    len(client_data), qnn_config.n_features, len(test_data[0]))
        return client_data, test_data
    except Exception as e:
        logger.error("Data loading failed: %s", e)
        logger.warning("Using synthetic data")
        rng = np.random.default_rng(42)
        n_clients = 4
        client_data = [(torch.from_numpy(rng.standard_normal((250, qnn_config.n_features)).astype(np.float32)),
                        torch.from_numpy(rng.integers(0, qnn_config.n_classes, 250))) for _ in range(n_clients)]
        test_data = (torch.from_numpy(rng.standard_normal((200, qnn_config.n_features)).astype(np.float32)),
                     torch.from_numpy(rng.integers(0, qnn_config.n_classes, 200)))
        return client_data, test_data


def _generate_qfl_visualizations(model, results, test_data, client_data, qnn_config, device):
    logger.info("Generating QFL visualizations...")
    try:
        from quantum.plots_qfl import generate_all_qfl_plots
        model.eval()
        X_test, y_test = test_data
        X_test_t = torch.as_tensor(X_test, dtype=torch.float32).to(device)
        with torch.no_grad():
            _, y_pred = torch.max(model(X_test_t), 1)
        y_pred_np = y_pred.cpu().numpy()
        y_test_np = y_test.numpy() if isinstance(y_test, torch.Tensor) else y_test
        plots = generate_all_qfl_plots(
            results=results, client_data=client_data, y_test=y_test_np,
            y_pred=y_pred_np, class_names=[f'Digit {i}' for i in range(qnn_config.n_classes)],
            save_dir=str(DEFAULT_VISUALIZATION_ROOT / 'qfl'),
        )
        logger.info("Generated %d QFL plots", len(plots))
    except ImportError as e:
        logger.warning("Visualization module not available: %s", e)
    except Exception as e:
        logger.error("Visualization error: %s", e)
        import traceback
        traceback.print_exc()


def _run_comparative_analysis(results, qfl, client_data, qnn_config, fl_config):
    cfl_metrics_path = DEFAULT_ARTIFACTS_DIR / 'metrics.csv'
    if not cfl_metrics_path.exists():
        logger.warning("Classical FL results not found at '%s' — skipping comparison", cfl_metrics_path)
        return
    logger.info("Found Classical FL results, loading for comparison...")
    import csv
    cfl_results = {'test_accuracies': [], 'test_losses': [], 'train_losses': []}
    with open(cfl_metrics_path, 'r') as f:
        for row in csv.DictReader(f):
            cfl_results['test_accuracies'].append(float(row['Test_Accuracy']))
            cfl_results['test_losses'].append(float(row['Test_Loss']))
            cfl_results['train_losses'].append(float(row['Train_Loss']))
    logger.info("Loaded %d rounds of Classical FL data", len(cfl_results['test_accuracies']))
    try:
        from quantum.plots_comparative_analysis import generate_all_comparative_plots, plot_3d_performance_surface
        comp_plots = generate_all_comparative_plots(
            qfl_results=results, cfl_results=cfl_results, qfl_model=qfl.global_model,
            client_data=client_data, qnn_config=qnn_config.to_dict(),
            qfl_config=fl_config, save_dir=str(DEFAULT_VISUALIZATION_ROOT / 'comparative'),
        )
        logger.info("Generated %d comparative plots", len(comp_plots))
        try:
            sp = plot_3d_performance_surface(results, cfl_results, str(DEFAULT_VISUALIZATION_ROOT / 'comparative'))
            comp_plots['3d_surface'] = sp
        except Exception as e:
            logger.error("3D surface plot failed: %s", e)
        final_q = results['test_accuracies'][-1]
        final_c = cfl_results['test_accuracies'][-1]
        best_q = max(results['test_accuracies'][1:]) if len(results['test_accuracies']) > 1 else final_q
        best_c = max(cfl_results['test_accuracies'][1:]) if len(cfl_results['test_accuracies']) > 1 else final_c
        logger.info("Comparative Summary: QFL final=%.4f best=%.4f | CFL final=%.4f best=%.4f",
                    final_q, best_q, final_c, best_c)
    except ImportError as e:
        logger.warning("Comparative visualization module not available: %s", e)
    except Exception as e:
        logger.error("Comparative analysis failed: %s", e)
        import traceback
        traceback.print_exc()


def main():
    args = _parse_qfl_args()

    qnn_config = QuantumNeuralNetworkConfig(
        n_qubits=args.qubits, n_features=4, n_classes=3,
        encoding='amplitude', n_layers=args.layers, entanglement='circular',
        batch_size=16, classical_lr=1e-3, quantum_lr=5e-4, grad_clip=1.0,
        noise_type=args.noise, depolarizing_p=args.depolarizing_p,
        amplitude_gamma=args.amplitude_gamma, readout_flip_prob=args.readout_flip,
        dp_enabled=args.dp_enable, dp_clip_norm=args.dp_clip, dp_noise_multiplier=args.dp_noise,
    )

    fl_config = {
        'num_rounds': args.rounds, 'local_epochs': args.local_epochs, 'batch_size': 16,
        'classical_lr': 1e-3, 'quantum_lr': 5e-4,
        'client_fraction': 0.75, 'grad_clip': 1.0,
        'dp_enabled': args.dp_enable, 'dp_clip_norm': args.dp_clip,
        'dp_noise_multiplier': args.dp_noise, 'dp_delta': args.dp_delta,
        'secure_aggregation': args.secure_agg, 'num_clients': args.clients,
    }

    data_config = {
        'raw_folder': "./dataset/raw", 'processed_folder': "./dataset/processed",
        'digits': (0, 1, 2), 'val_split': 0.1,
        'num_clients': args.clients, 'partition_type': 'iid',
        'alpha': 0.5, 'apply_pca': True, 'pca_components': qnn_config.n_features,
    }

    logger.info("QUANTUM FEDERATED LEARNING WITH PENNYLANE")

    client_data, test_data = _load_or_synthesize_data(qnn_config, data_config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    qfl = QuantumFederatedLearning(qnn_config, fl_config, device)

    try:
        results = qfl.train(client_data, test_data)
        qfl.save_results()
        _generate_qfl_visualizations(qfl.global_model, results, test_data, client_data, qnn_config, device)
        _run_comparative_analysis(results, qfl, client_data, qnn_config, fl_config)
        logger.info("QUANTUM FEDERATED LEARNING COMPLETED SUCCESSFULLY")
        return results
    except Exception as e:
        logger.error("Training failed: %s", e)
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()
