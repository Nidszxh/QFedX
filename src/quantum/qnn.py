"""Quantum Neural Network components for QFedX."""

import os
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from core.data import create_dataloader, evaluate_model
from core.defs import (
    DEFAULT_MLFLOW_URI,
    DEFAULT_MODEL_SAVE_PATH,
    DEFAULT_VISUALIZATION_ROOT,
    DIVISION_EPSILON,
    EncodingType,
    EntanglementTopology,
    NoiseType,
    OptimizerType,
)
from core.quantum import apply_encoding, apply_variational_layer, build_entanglement_pairs
from core.utils import get_logger
from data.preprocess import preprocess_mnist

logger = get_logger(__name__)

os.environ.setdefault("MLFLOW_TRACKING_URI", DEFAULT_MLFLOW_URI)
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    logger.warning("mlflow not available — metrics will not be tracked")
    MLFLOW_AVAILABLE = False

try:
    from sklearn.datasets import load_iris
    from sklearn.preprocessing import StandardScaler
    IRIS_AVAILABLE = True
except ImportError:
    logger.warning("sklearn.datasets not available — Iris dataset disabled")
    IRIS_AVAILABLE = False

try:
    from quantum.noise import NoiseConfig, NoisyQuantumCircuit
    NOISE_AVAILABLE = True
except ImportError:
    logger.warning("Quantum noise module not available — noise models disabled")
    NOISE_AVAILABLE = False

try:
    from quantum.privacy.differential_privacy import DifferentialPrivacy
    PRIVACY_AVAILABLE = True
except ImportError:
    logger.warning("DifferentialPrivacy module not available — DP disabled")
    PRIVACY_AVAILABLE = False

torch.set_default_dtype(torch.float32)

@dataclass
class QuantumNeuralNetworkConfig:
    """Configuration for Quantum Neural Network."""
    n_qubits: int = 4
    n_layers: int = 2
    n_readout: Optional[int] = None
    encoding: EncodingType = EncodingType.AMPLITUDE
    entanglement: EntanglementTopology = EntanglementTopology.CIRCULAR
    n_features: int = 4
    n_classes: int = 3
    batch_size: int = 16
    epochs: int = 5
    classical_lr: float = 1e-3
    quantum_lr: float = 5e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    device_name: str = 'default.qubit'
    diff_method: str = 'backprop'
    shots: Optional[int] = None
    optimizer_type: OptimizerType = OptimizerType.ADAMW
    spsa_a: float = 0.01
    spsa_c: float = 0.1
    wrap_angles: bool = True
    noise_type: NoiseType = NoiseType.NONE
    depolarizing_p: float = 0.001
    amplitude_gamma: float = 0.001
    readout_flip_prob: float = 0.01
    dp_enabled: bool = False
    dp_clip_norm: float = 1.0
    dp_noise_multiplier: float = 1.0
    early_stop_patience: int = 0
    early_stop_delta: float = 1e-4
    checkpoint_dir: str = ""

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> "QuantumNeuralNetworkConfig":
        return cls(
            n_qubits=cfg.get("n_qubits", 4),
            n_layers=cfg.get("n_layers", 2),
            encoding=cfg.get("encoding", "amplitude"),
            entanglement=cfg.get("entanglement", "circular"),
            n_features=cfg.get("n_features", 4),
            n_classes=cfg.get("n_classes", 3),
            batch_size=cfg.get("batch_size", 16),
            epochs=cfg.get("epochs", 5),
            classical_lr=cfg.get("classical_lr", 1e-3),
            quantum_lr=cfg.get("quantum_lr", 5e-4),
            weight_decay=cfg.get("weight_decay", 1e-4),
            grad_clip=cfg.get("grad_clip", 1.0),
            device_name=cfg.get("device_name", "default.qubit"),
            diff_method=cfg.get("diff_method", "backprop"),
            shots=cfg.get("shots", None),
            optimizer_type=cfg.get("optimizer_type", "adamw"),
            noise_type=cfg.get("noise_type", "none"),
            depolarizing_p=cfg.get("depolarizing_p", 0.001),
            amplitude_gamma=cfg.get("amplitude_gamma", 0.001),
            readout_flip_prob=cfg.get("readout_flip_prob", 0.01),
            spsa_a=cfg.get("spsa_a", 0.01),
            spsa_c=cfg.get("spsa_c", 0.1),
            wrap_angles=cfg.get("wrap_angles", True),
            dp_enabled=cfg.get("dp_enabled", False),
            early_stop_patience=cfg.get("early_stop_patience", 0),
            early_stop_delta=cfg.get("early_stop_delta", 1e-4),
            checkpoint_dir=cfg.get("checkpoint_dir", ""),
        )

    def __post_init__(self):
        if self.n_readout is None:
            self.n_readout = self.n_qubits
        if self.n_readout > self.n_qubits:
            raise ValueError(f"n_readout ({self.n_readout}) > n_qubits ({self.n_qubits})")
        if self.shots is not None and self.diff_method == 'backprop':
            self.diff_method = 'parameter-shift'
        if self.noise_type != NoiseType.NONE:
            self.diff_method = 'parameter-shift'

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def get_noise_config(self):
        if not NOISE_AVAILABLE or self.noise_type == NoiseType.NONE:
            return None
        return NoiseConfig(
            noise_type=self.noise_type.value,
            depolarizing_p=self.depolarizing_p,
            amplitude_gamma=self.amplitude_gamma,
            readout_flip_prob=self.readout_flip_prob,
            shots=self.shots
        )


class ClassicalPreprocessor(nn.Module):
    """Classical preprocessing layer with batch normalization."""

    def __init__(self, n_features: int, target_size: int, encoding: str):
        super().__init__()
        self.encoding = encoding
        self.network = nn.Sequential(
            nn.Linear(n_features, target_size),
            nn.BatchNorm1d(target_size),
            nn.Tanh()
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class QuantumCircuit:
    def __init__(self, config: QuantumNeuralNetworkConfig):
        self.config = config
        self.n_qubits = config.n_qubits
        self.n_layers = config.n_layers
        self.n_readout = config.n_readout
        self.encoding = config.encoding
        self.entanglement = config.entanglement
        self._is_batched = False

        use_noise = NOISE_AVAILABLE and config.noise_type != NoiseType.NONE
        if use_noise:
            noise_cfg = config.get_noise_config()
            self.noisy_circuit = NoisyQuantumCircuit(
                n_qubits=config.n_qubits, n_layers=config.n_layers,
                n_readout=config.n_readout, encoding=config.encoding,
                entanglement=config.entanglement, noise_config=noise_cfg,
                diff_method='parameter-shift'
            )
            self.dev = self.noisy_circuit.dev
            self.qnode = self.noisy_circuit.qnode
        else:
            self.dev = qml.device(config.device_name, wires=self.n_qubits, shots=config.shots)
            self.entanglement_pairs = build_entanglement_pairs(self.n_qubits, self.entanglement)
            self.qnode = self._build_qnode()

    def _build_qnode(self) -> Callable:
        wires = range(self.n_qubits)

        @qml.qnode(self.dev, interface='torch', diff_method=self.config.diff_method)
        def circuit(inputs: torch.Tensor, weights: torch.Tensor):
            apply_encoding(inputs, wires, self.encoding)
            for layer_idx in range(self.n_layers):
                apply_variational_layer(weights[layer_idx], wires, self.entanglement_pairs)
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_readout)]

        try:
            self.batch_qnode = qml.transforms.batch_input(circuit, argnum=0)
            self._is_batched = True
        except Exception as e:
            logger.debug("Batch input transform not supported, using per-sample execution: %s", e)
            self.batch_qnode = circuit
            self._is_batched = False
        return circuit

    def forward(self, inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        is_batched = inputs.dim() > 1 and self._is_batched
        if is_batched:
            result = self.batch_qnode(inputs, weights)
        else:
            result = self.qnode(inputs, weights)
        if isinstance(result, (list, tuple)):
            result = torch.stack([torch.as_tensor(r, dtype=torch.float32) for r in result])
        else:
            result = torch.as_tensor(result, dtype=torch.float32)
        if is_batched and result.dim() == 2:
            result = result.permute(1, 0)
        elif is_batched and result.dim() > 2:
            result = result.permute(*range(result.dim() - 1, -1, -1))
        return result


class QuantumNeuralNetwork(nn.Module):
    """Hybrid Quantum-Classical Neural Network."""

    def __init__(self, config: QuantumNeuralNetworkConfig):
        super().__init__()
        self.config = config
        if config.encoding == 'angle':
            quantum_input_size = config.n_qubits
        else:
            quantum_input_size = 2 ** config.n_qubits
        self.preprocessor = ClassicalPreprocessor(config.n_features, quantum_input_size, config.encoding)
        self.quantum_circuit = QuantumCircuit(config)
        self.q_weights = nn.Parameter(0.01 * torch.randn(config.n_layers, config.n_qubits, 2))
        self.classifier = nn.Sequential(nn.Dropout(0.1), nn.Linear(config.n_readout, config.n_classes))
        nn.init.xavier_uniform_(self.classifier[1].weight, gain=0.5)
        nn.init.zeros_(self.classifier[1].bias)

    def _forward_per_sample(self, preprocessed: torch.Tensor, device: torch.device) -> torch.Tensor:
        batch_size = preprocessed.shape[0]
        q_outputs = []
        for i in range(batch_size):
            sample = preprocessed[i]
            try:
                q_out = self.quantum_circuit.forward(sample.cpu(), self.q_weights.cpu())
                q_out = q_out.to(device).float()
                q_outputs.append(q_out)
            except Exception as e:
                logger.warning(f"Quantum circuit failed for sample {i}: {e}")
                q_outputs.append(torch.zeros(self.config.n_readout, dtype=torch.float32, device=device))
        return torch.stack(q_outputs, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        preprocessed = self.preprocessor(x)
        if self.quantum_circuit._is_batched:
            try:
                q_batch = self.quantum_circuit.forward(preprocessed.detach(), self.q_weights)
                q_batch = q_batch.to(device).float()
            except Exception as e:
                logger.warning(f"Batched quantum circuit failed ({e}), falling back to per-sample")
                q_batch = self._forward_per_sample(preprocessed, device)
        else:
            q_batch = self._forward_per_sample(preprocessed, device)
        logits = self.classifier(q_batch)
        if NOISE_AVAILABLE and self.config.noise_type != NoiseType.NONE and self.config.readout_flip_prob > 0:
            from quantum.noise import NoiseConfig, NoiseModel
            nm = NoiseModel(NoiseConfig(noise_type='readout', readout_flip_prob=self.config.readout_flip_prob))
            probs = torch.softmax(logits, dim=-1)
            probs = nm.apply_readout_error(probs)
            logits = torch.log(probs + DIVISION_EPSILON)
        return logits

    def get_quantum_params(self) -> list[nn.Parameter]:
        return [self.q_weights]

    def get_classical_params(self) -> list[nn.Parameter]:
        return list(self.preprocessor.parameters()) + list(self.classifier.parameters())


class SPSA:
    """Simultaneous Perturbation Stochastic Approximation.

    Gradient-free optimization requiring only 2 loss evaluations per step
    regardless of parameter count. Useful for quantum circuits where
    parameter-shift gradient computation scales linearly with parameters.
    """

    def __init__(
        self,
        params: list[torch.nn.Parameter],
        a: float = 0.01,
        c: float = 0.1,
        alpha: float = 0.602,
        gamma: float = 0.101,
        A: Optional[float] = None,
        wrap_angles: bool = True,
    ):
        self.params = params
        self.a = a
        self.c = c
        self.alpha = alpha
        self.gamma = gamma
        self.A = A if A is not None else 0.0
        self.wrap_angles = wrap_angles
        self.k = 0

    def step(self, closure: Callable[[], torch.Tensor]) -> torch.Tensor:
        self.k += 1
        ak = self.a / (self.k + self.A) ** self.alpha
        ck = self.c / self.k ** self.gamma

        orig_params = [p.data.clone() for p in self.params]
        delta = [
            (torch.randint(0, 2, p.shape, device=p.device, dtype=p.dtype) * 2 - 1)
            for p in self.params
        ]

        with torch.no_grad():
            for p, d in zip(self.params, delta):
                p.data.add_(ck * d)
                if self.wrap_angles:
                    p.data.copy_(torch.atan2(torch.sin(p.data), torch.cos(p.data)))
        loss_plus = closure()

        with torch.no_grad():
            for p, d in zip(self.params, delta):
                p.data.sub_(2 * ck * d)
                if self.wrap_angles:
                    p.data.copy_(torch.atan2(torch.sin(p.data), torch.cos(p.data)))
        loss_minus = closure()

        with torch.no_grad():
            for p, orig in zip(self.params, orig_params):
                p.data.copy_(orig)

        with torch.no_grad():
            for p, d in zip(self.params, delta):
                grad_estimate = (loss_plus - loss_minus) / (2 * ck * d)
                p.data.sub_(ak * grad_estimate)
                if self.wrap_angles:
                    p.data.copy_(torch.atan2(torch.sin(p.data), torch.cos(p.data)))

        return (loss_plus + loss_minus) / 2.0

    def state_dict(self) -> dict:
        return {"k": self.k, "a": self.a, "c": self.c}

    def load_state_dict(self, state_dict: dict):
        self.k = state_dict["k"]
        self.a = state_dict["a"]
        self.c = state_dict["c"]


class QuantumNeuralNetworkTrainer:
    def __init__(self, model: QuantumNeuralNetwork, config: QuantumNeuralNetworkConfig, device: str = 'cpu'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.spsa_optimizer = None

        if config.optimizer_type == 'spsa':
            self.optimizer = optim.AdamW([
                {"params": model.get_classical_params(), "lr": config.classical_lr, "weight_decay": config.weight_decay},
            ]) if model.get_classical_params() else None
            self.spsa_optimizer = SPSA(
                model.get_quantum_params(),
                a=config.spsa_a, c=config.spsa_c,
                wrap_angles=config.wrap_angles,
            )
        else:
            self.optimizer = optim.AdamW([
                {"params": model.get_classical_params(), "lr": config.classical_lr, "weight_decay": config.weight_decay},
                {"params": model.get_quantum_params(), "lr": config.quantum_lr, "weight_decay": 0}
            ])
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2) if self.optimizer else None
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.best_state_dict = None
        self.lr_history = []
        self.clip_norms_history = []
        self.dp_engine = None
        self._patience_counter = 0
        if PRIVACY_AVAILABLE and config.dp_enabled:
            self.dp_engine = DifferentialPrivacy(
                clip_norm=config.dp_clip_norm,
                noise_multiplier=config.dp_noise_multiplier
            )

    def train_epoch(self, train_loader: DataLoader) -> tuple[float, float]:
        if self.config.optimizer_type == 'spsa':
            return self._train_epoch_spsa(train_loader)
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            try:
                outputs = self.model(data)
                loss = self.criterion(outputs, target)
                if torch.isnan(loss):
                    logger.warning(f"NaN loss in batch {batch_idx}, skipping...")
                    continue
                loss.backward()
                if self.config.grad_clip > 0 and self.dp_engine is None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                if self.dp_engine is not None:
                    named_params = {k: v for k, v in self.model.named_parameters() if v.grad is not None}
                    if named_params:
                        _, clip_norm = self.dp_engine.apply(
                            named_params, data.size(0),
                            sample_rate=data.size(0) / max(len(train_loader.dataset), 1)
                        )
                        self.clip_norms_history.append(clip_norm)
                self.optimizer.step()
                total_loss += loss.item()
                num_batches += 1
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == target).sum().item()
                total += target.size(0)
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
                continue
        avg_loss = total_loss / max(num_batches, 1)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy

    def _train_epoch_spsa(self, train_loader: DataLoader) -> tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            def closure():
                outputs = self.model(data)
                return self.criterion(outputs, target)

            if self.optimizer:
                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.criterion(outputs, target)
                loss.backward()
                if self.config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()
            else:
                loss = self.spsa_optimizer.step(closure)

            if torch.isnan(loss):
                logger.warning(f"NaN loss in batch {batch_idx}, skipping...")
                continue

            total_loss += loss.item() if hasattr(loss, 'item') else loss
            num_batches += 1
            with torch.no_grad():
                outputs = self.model(data)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == target).sum().item()
                total += target.size(0)

        avg_loss = total_loss / max(num_batches, 1)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> tuple[float, float]:
        return evaluate_model(self.model, loader, criterion=self.criterion, device=self.device)

    def save_checkpoint(self, filename: str = "best_model.pt") -> str:
        save_dir = Path(self.config.checkpoint_dir) if self.config.checkpoint_dir else Path("./checkpoints")
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config.to_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_acc': self.best_val_acc,
        }, path)
        return str(path)

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=torch.device(self.device))
        self.model.load_state_dict(ckpt['model_state_dict'])
        if self.optimizer and ckpt.get('optimizer_state_dict'):
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.train_losses = ckpt.get('train_losses', [])
        self.val_losses = ckpt.get('val_losses', [])
        self.best_val_acc = ckpt.get('best_val_acc', 0.0)
        logger.info("Checkpoint loaded: %s", path)

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
                verbose: bool = True) -> dict:
        if verbose:
            logger.info("Quantum Neural Network Training")
            logger.info(f"Device: {self.device}")
            logger.info(f"Total Parameters: {sum(p.numel() for p in self.model.parameters())}")
            logger.info(f"Quantum Parameters: {self.model.q_weights.numel()}")
            logger.info(f"Classical Parameters: {sum(p.numel() for p in self.model.get_classical_params())}")
            if self.config.early_stop_patience > 0:
                logger.info(f"Early stopping: patience={self.config.early_stop_patience}, delta={self.config.early_stop_delta}")

        for epoch in range(self.config.epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)

            if self.optimizer:
                current_lr = self.optimizer.param_groups[0]['lr']
                self.lr_history.append(current_lr)

            val_loss, val_acc = 0.0, 0.0
            if val_loader is not None:
                val_loss, val_acc = self.evaluate(val_loader)
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_acc)
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_state_dict = self.model.state_dict()
                if val_loss < self.best_val_loss - self.config.early_stop_delta:
                    self.best_val_loss = val_loss
                    self._patience_counter = 0
                else:
                    self._patience_counter += 1

            if MLFLOW_AVAILABLE:
                mlflow.log_metrics({"train_loss": train_loss, "train_acc": train_acc,
                                    "val_loss": val_loss, "val_acc": val_acc}, step=epoch)

            if self.scheduler:
                self.scheduler.step()

            if verbose and ((epoch + 1) % 2 == 0 or epoch == 0):
                lr = self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.0
                if val_loader is not None:
                    logger.info(f"Epoch {epoch+1:3d}/{self.config.epochs} | "
                          f"Loss: {train_loss:.4f} | "
                          f"Train Acc: {train_acc:.4f} | "
                          f"Val Acc: {val_acc:.4f} | "
                          f"LR: {lr:.6f}")
                else:
                    logger.info(f"Epoch {epoch+1:3d}/{self.config.epochs} | "
                          f"Loss: {train_loss:.4f} | "
                          f"Train Acc: {train_acc:.4f} | "
                          f"LR: {lr:.6f}")

            if self.config.early_stop_patience > 0 and self._patience_counter >= self.config.early_stop_patience:
                logger.info("Early stopping triggered at epoch %d (val loss did not improve for %d epochs)",
                            epoch + 1, self._patience_counter)
                break

        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
            if self.config.checkpoint_dir:
                path = self.save_checkpoint()
                logger.info("Best model checkpoint saved: %s", path)

        if verbose:
            logger.info("=" * 70)
            logger.info("Training Complete!")
            if self.best_val_acc > 0:
                logger.info(f"Best Validation Accuracy: {self.best_val_acc:.4f}")
            logger.info("=" * 70)

        return {
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'best_val_acc': self.best_val_acc,
            'lr_history': self.lr_history
        }


def load_iris_data(config: QuantumNeuralNetworkConfig, val_split: float = 0.2, test_split: float = 0.2):
    """Load Iris dataset, split into train/val/test, return DataLoaders."""
    if not IRIS_AVAILABLE:
        raise ImportError("scikit-learn required for Iris dataset")

    iris = load_iris()
    X, y = iris.data.astype(np.float32), iris.target.astype(np.int64)

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_split / (1 - test_split),
        stratify=y_train, random_state=42
    )

    train_loader = create_dataloader(X_train, y_train, batch_size=config.batch_size, shuffle=True)
    val_loader = create_dataloader(X_val, y_val, batch_size=config.batch_size, shuffle=False)
    test_loader = create_dataloader(X_test, y_test, batch_size=config.batch_size, shuffle=False)

    logger.info(f"Iris: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test samples")
    return train_loader, val_loader, test_loader


def verify_gradients(config: Optional[QuantumNeuralNetworkConfig] = None, tol: float = 0.05):
    """Compare gradients from backprop vs parameter-shift on a small batch."""
    if config is None:
        config = QuantumNeuralNetworkConfig(n_qubits=2, n_layers=1, n_features=2, n_classes=2,
                           batch_size=4, epochs=1, encoding='angle',
                           device_name='default.qubit', diff_method='backprop')

    X = torch.randn(config.batch_size, config.n_features)
    y = torch.randint(0, config.n_classes, (config.batch_size,))

    torch.manual_seed(42)
    # Backprop
    model_bp = QuantumNeuralNetwork(config)
    loss_bp = nn.CrossEntropyLoss()(model_bp(X), y)
    loss_bp.backward()
    grad_bp = model_bp.q_weights.grad.detach().clone()

    # Parameter-shift with identical weights
    ps_config = QuantumNeuralNetworkConfig(**config.to_dict())
    ps_config.diff_method = 'parameter-shift'
    torch.manual_seed(42)
    model_ps = QuantumNeuralNetwork(ps_config)
    # Copy ALL weights to ensure identical starting point
    model_ps.load_state_dict(model_bp.state_dict())
    loss_ps = nn.CrossEntropyLoss()(model_ps(X), y)
    loss_ps.backward()
    grad_ps = model_ps.q_weights.grad.detach().clone()

    # Compare
    diff = (grad_bp - grad_ps).abs().max().item()
    cosine = nn.functional.cosine_similarity(grad_bp.flatten().unsqueeze(0),
                                              grad_ps.flatten().unsqueeze(0)).item()
    passed = diff < tol
    logger.info("Gradient verification (backprop vs parameter-shift):")
    logger.info(f"  Max absolute difference: {diff:.6f}")
    logger.info(f"  Cosine similarity:       {cosine:.6f}")
    logger.info(f"  {'PASSED' if passed else 'FAILED'} (tol={tol})")
    return passed, diff, cosine


def _generate_predictions(
    model: QuantumNeuralNetwork,
    loader: DataLoader,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    y_pred_list, y_scores_list, y_true_list = [], [], []
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            outputs = model(data)
            y_scores_list.append(torch.softmax(outputs, dim=1).cpu().numpy())
            y_pred_list.append(torch.max(outputs, 1)[1].cpu().numpy())
            y_true_list.append(target.cpu().numpy())
    return np.concatenate(y_pred_list), np.concatenate(y_scores_list), np.concatenate(y_true_list)


def main():
    config = QuantumNeuralNetworkConfig(
        n_qubits=4, n_layers=2, encoding='amplitude', entanglement='circular',
        n_features=4, n_classes=3, batch_size=16, epochs=5,
        classical_lr=1e-3, quantum_lr=5e-4, weight_decay=1e-4, grad_clip=1.0,
        device_name='default.qubit', diff_method='backprop', shots=None
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger.info("Quantum Neural Network for Federated Learning")
    logger.info("Framework: PennyLane + PyTorch")
    logger.info(f"Device: {device}")
    for k, v in config.to_dict().items():
        logger.info(f"  {k:20s}: {v}")

    # Load data
    try:
        train_set, val_set, test_set, _ = preprocess_mnist(
            raw_folder="./dataset/raw",
            processed_folder="./dataset/processed",
            digits=(0, 1, 2), val_split=0.1, num_clients=4,
            partition_type='iid', apply_pca=True,
            pca_components=config.n_features, generate_plots=False
        )
        X_train, y_train = train_set
        X_val, y_val = val_set
        X_test, y_test = test_set
        config.n_features = X_train.shape[1]
        logger.info(f"Data loaded: {X_train.shape[0]} train, {X_val.shape[0]} val samples")
        logger.info(f"Feature dimension: {config.n_features}")
    except Exception as e:
        logger.warning(f"Using dummy data due to error: {e}")
        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((1000, config.n_features))
        y_train = rng.integers(0, config.n_classes, 1000)
        X_val = rng.standard_normal((200, config.n_features))
        y_val = rng.integers(0, config.n_classes, 200)
        X_test = rng.standard_normal((200, config.n_features))
        y_test = rng.integers(0, config.n_classes, 200)

    train_loader = create_dataloader(X_train, y_train, batch_size=config.batch_size, shuffle=True)
    val_loader = create_dataloader(X_val, y_val, batch_size=config.batch_size, shuffle=False)
    test_loader = create_dataloader(X_test, y_test, batch_size=config.batch_size, shuffle=False)

    # Initialize model and trainer
    model = QuantumNeuralNetwork(config)
    trainer = QuantumNeuralNetworkTrainer(model, config, device)

    # Train
    history = trainer.train(train_loader, val_loader)

    # Evaluate on test set
    test_loss, test_acc = trainer.evaluate(test_loader)
    logger.info(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

    # Predictions for visualization
    y_pred, y_scores, y_test_np = _generate_predictions(model, test_loader, device)
    X_test_np = X_test.numpy() if isinstance(X_test, torch.Tensor) else X_test

    # Generate visualizations
    logger.info("Generating QNN visualizations...")
    try:
        from quantum.plots_qnn import generate_all_qnn_plots

        saved_plots = generate_all_qnn_plots(
            history=history, model=model,
            X_test=X_test_np, y_test=y_test_np,
            y_pred=y_pred, y_scores=y_scores,
            class_names=[f'Digit {i}' for i in range(config.n_classes)],
            save_dir=str(DEFAULT_VISUALIZATION_ROOT / 'qnn')
        )
        logger.info(f"Generated {len(saved_plots)} QNN visualization plots:")
        for name, path in saved_plots.items():
            logger.info(f"  - {name}: {path}")
    except ImportError as e:
        logger.warning(f"Visualization module not available: {e}")
    except Exception as e:
        logger.error(f"Visualization error: {e}")
        import traceback
        logger.debug(traceback.format_exc())

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.to_dict(),
        'history': history,
        'test_accuracy': test_acc,
        'test_loss': test_loss,
    }, DEFAULT_MODEL_SAVE_PATH)
    logger.info(f"Model saved to '{DEFAULT_MODEL_SAVE_PATH}'")

    return model, history


def demo_iris():
    """Train QNN on Iris dataset as a standalone verification."""
    logger.info("=" * 70)
    logger.info("IRIS DATASET DEMO")
    logger.info("=" * 70)
    config = QuantumNeuralNetworkConfig(
        n_qubits=4, n_layers=2, encoding='angle', entanglement='circular',
        n_features=4, n_classes=3, batch_size=8, epochs=20,
        classical_lr=1e-3, quantum_lr=5e-4, device_name='default.qubit',
        diff_method='backprop', shots=None
    )
    device = 'cpu'
    train_loader, val_loader, test_loader = load_iris_data(config)
    model = QuantumNeuralNetwork(config)
    trainer = QuantumNeuralNetworkTrainer(model, config, device)
    history = trainer.train(train_loader, val_loader)
    test_loss, test_acc = trainer.evaluate(test_loader)
    logger.info(f"Iris Test Accuracy: {test_acc:.4f} (random baseline: {1/config.n_classes:.4f})")
    if test_acc > 1.0 / config.n_classes:
        logger.info("Iris verification PASSED: accuracy exceeds random baseline")
    else:
        logger.warning("Iris verification FAILED: accuracy at or below random baseline")
    return history, test_acc


if __name__ == "__main__":
    # Phase 1 verification: gradient comparison
    logger.info("=" * 70)
    logger.info("PHASE 1: GRADIENT VERIFICATION")
    logger.info("=" * 70)
    verify_gradients()

    # Phase 1 verification: Iris training
    demo_iris()

    # Full MNIST-PCA training
    model, history = main()
