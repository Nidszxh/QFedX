import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
from typing import Optional, Literal, Tuple, List, Dict
from dataclasses import dataclass, asdict
from torch.utils.data import DataLoader, TensorDataset, Dataset

# GPU optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# CONFIGURATION

@dataclass
class NoiseConfig:
    depolarizing_p: float = 0.0
    amplitude_damping_gamma: float = 0.0
    readout_error_p: float = 0.0

@dataclass
class QNNConfig:
    # Quantum architecture
    n_qubits: int = 4
    n_layers: int = 2
    encoding: Literal['amplitude', 'angle'] = 'angle'
    entanglement: Literal['linear', 'circular', 'full', 'pyramid'] = 'pyramid'
    measurement: Literal['single_z', 'multi_basis', 'parity'] = 'multi_basis'
    
    # Model architecture
    n_features: int = 4
    n_classes: int = 3
    batch_size: int = 32
    epochs: int = 10
    
    # Optimization
    classical_lr: float = 1e-3
    quantum_lr: float = 5e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    
    # Quantum device
    diff_method: str = 'adjoint'
    shots: Optional[int] = None
    
    # Performance
    use_gpu_cache: bool = True
    parallel_execution: bool = False
    
    # Reproducibility
    seed: int = 42
    
    # Federated Learning
    client_id: Optional[int] = None
    
    # Noise mapping
    noise_config: Optional[NoiseConfig] = None
    
    def __post_init__(self):
        if self.noise_config is None:
            self.noise_config = NoiseConfig()
        # Determine number of measurements
        if self.measurement == 'single_z':
            self.n_readout = self.n_qubits
        elif self.measurement == 'multi_basis':
            # Z on all + X,Y on first 2 qubits
            self.n_readout = self.n_qubits + 4
        elif self.measurement == 'parity':
            # Z on all + pairwise correlations
            self.n_readout = self.n_qubits + (self.n_qubits - 1)
        else:
            self.n_readout = self.n_qubits
    
    def to_dict(self) -> Dict:
        return asdict(self)


def set_seeds(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Make CUDA operations deterministic
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class GPUDataset(Dataset):
    """Dataset that caches data on GPU for faster training."""
    def __init__(self, X: torch.Tensor, y: torch.Tensor, device: str):
        if torch.cuda.is_available() and device == 'cuda':
            self.X = X.to(device)
            self.y = y.to(device)
        else:
            self.X, self.y = X, y
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# QUANTUM CIRCUIT

class QuantumCircuit:  
    def __init__(self, config: QNNConfig, device: str):
        self.config = config
        self.n_qubits = config.n_qubits
        self.n_layers = config.n_layers
        self.encoding = config.encoding
        self.measurement = config.measurement
        
        # Initialize device with GPU support
        self.dev = self._init_device(device)
        self.entangle_pairs = self._build_entanglement()
        self.qnode = self._build_circuit()
        
        print(f"    Quantum circuit initialized:")
        print(f"      Encoding: {self.encoding}")
        print(f"      Entanglement: {config.entanglement} ({len(self.entangle_pairs)} CNOTs)")
        print(f"      Measurements: {self.measurement} ({config.n_readout} outputs)")
    
    def _init_device(self, device: str):
        """Auto-select best available quantum device."""
        device_priority = ['lightning.gpu', 'lightning.qubit', 'default.qubit']
        
        for dev_name in device_priority:
            # Skip GPU device if CUDA not available
            if dev_name == 'lightning.gpu' and device != 'cuda':
                continue
            
            try:
                dev = qml.device(dev_name, wires=self.n_qubits, shots=self.config.shots)
                print(f"    ✓ Quantum device: {dev_name}")
                return dev
            except:
                continue
        
        raise RuntimeError("No quantum device available")
    
    def _build_entanglement(self) -> List[Tuple[int, int]]:
        """
        Build entanglement structure with barren plateau mitigation.
        
        Pyramid entanglement is best for deep circuits (n_layers >= 2).
        """
        ent = self.config.entanglement
        n = self.n_qubits
        
        if ent == 'linear':
            return [(i, i+1) for i in range(n-1)]
        
        elif ent == 'circular':
            return [(i, (i+1) % n) for i in range(n)]
        
        elif ent == 'full':
            return [(i, j) for i in range(n) for j in range(i+1, n)]
        
        elif ent == 'pyramid':
            # Pyramid: alternating patterns across layers
            # Proven to mitigate barren plateaus
            pairs = []
            for layer in range(self.n_layers):
                if layer % 2 == 0:
                    # Even layers: nearest-neighbor
                    pairs.extend([(i, i+1) for i in range(0, n-1, 2)])
                else:
                    # Odd layers: offset + long-range
                    pairs.extend([(i, i+1) for i in range(1, n-1, 2)])
                    if n > 2:
                        pairs.append((0, n-1))
            return list(set(pairs))  # Remove duplicates
        
        else:
            raise ValueError(f"Unknown entanglement: {ent}")
    
    def _build_circuit(self):
        """Build differentiable quantum circuit."""
        @qml.qnode(self.dev, interface='torch', diff_method=self.config.diff_method)
        def circuit(inputs: torch.Tensor, weights: torch.Tensor):
            """
            Quantum circuit supporting batch processing.
            
            Args:
                inputs: (batch_size, n_features) or (n_features,)
                weights: (n_layers, n_qubits, 3) quantum parameters
            
            Returns:
                Measurement outcomes
            """
            # Handle batch dimension
            is_batch = inputs.ndim == 2
            if not is_batch:
                inputs = inputs.unsqueeze(0)
            
            batch_size = inputs.shape[0]
            
            # ========== ENCODING ==========
            if self.encoding == 'angle':
                # Map [-1, 1] → [0, π] for angle encoding
                angles = (inputs + 1.0) * (np.pi / 2.0)
                
                # Batch-compatible angle encoding
                for i in range(self.n_qubits):
                    if batch_size == 1:
                        qml.RY(angles[0, i], wires=i)
                    else:
                        # Note: PennyLane batching works for angle encoding
                        qml.RY(angles[:, i], wires=i)
            
            elif self.encoding == 'amplitude':
                # Amplitude encoding (no native batching support)
                if batch_size > 1:
                    raise NotImplementedError(
                        "Amplitude encoding doesn't support native batching. "
                        "Use angle encoding for better performance."
                    )
                qml.AmplitudeEmbedding(
                    inputs[0],
                    wires=range(self.n_qubits),
                    normalize=True,
                    pad_with=0.0
                )
            
            else:
                raise ValueError(f"Unknown encoding: {self.encoding}")
            
            # ========== VARIATIONAL LAYERS ==========
            for layer in range(self.n_layers):
                # Single-qubit rotations (full SU(2))
                for i in range(self.n_qubits):
                    qml.RX(weights[layer, i, 0], wires=i)
                    qml.RY(weights[layer, i, 1], wires=i)
                    qml.RZ(weights[layer, i, 2], wires=i)
                
                # Entanglement layer
                for ctrl, tgt in self.entangle_pairs:
                    qml.CNOT(wires=[ctrl, tgt])
                
                # Apply noise channels if enabled
                if self.config.noise_config.depolarizing_p > 0:
                    for i in range(self.n_qubits):
                        qml.DepolarizingChannel(self.config.noise_config.depolarizing_p, wires=i)
                
                if self.config.noise_config.amplitude_damping_gamma > 0:
                    for i in range(self.n_qubits):
                        qml.AmplitudeDamping(self.config.noise_config.amplitude_damping_gamma, wires=i)
            
            # ========== MEASUREMENT ==========
            # Apply Readout Error (BitFlip) before expval if enabled
            if self.config.noise_config.readout_error_p > 0:
                for i in range(self.n_qubits):
                    qml.BitFlip(self.config.noise_config.readout_error_p, wires=i)
                    
            measurements = []
            
            if self.measurement == 'single_z':
                # Standard Z-basis measurement
                for i in range(self.n_qubits):
                    measurements.append(qml.expval(qml.PauliZ(i)))
            
            elif self.measurement == 'multi_basis':
                # Measure in multiple bases (richer information)
                # Z-basis on all qubits
                for i in range(self.n_qubits):
                    measurements.append(qml.expval(qml.PauliZ(i)))
                
                # X-basis on first 2 qubits
                for i in range(min(2, self.n_qubits)):
                    measurements.append(qml.expval(qml.PauliX(i)))
                
                # Y-basis on first 2 qubits
                for i in range(min(2, self.n_qubits)):
                    measurements.append(qml.expval(qml.PauliY(i)))
            
            elif self.measurement == 'parity':
                # Single-qubit + pairwise correlations
                for i in range(self.n_qubits):
                    measurements.append(qml.expval(qml.PauliZ(i)))
                
                # 2-qubit correlations (captures entanglement)
                for i in range(self.n_qubits - 1):
                    measurements.append(qml.expval(qml.PauliZ(i) @ qml.PauliZ(i+1)))
            
            else:
                raise ValueError(f"Unknown measurement: {self.measurement}")
            
            # Return measurements
            if not is_batch:
                return torch.stack([torch.as_tensor(m[0], dtype=torch.float32) for m in measurements])
            
            return measurements
        
        return circuit
    
    def forward(self, inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with automatic batch handling.
        
        Args:
            inputs: (batch_size, n_features) or (n_features,)
            weights: (n_layers, n_qubits, 3)
        
        Returns:
            (batch_size, n_readout) or (n_readout,) measurement outcomes
        """
        if self.encoding == 'amplitude' and inputs.ndim == 2:
            # Amplitude encoding: sequential processing required
            batch_size = inputs.shape[0]
            device = inputs.device
            outputs = torch.zeros(batch_size, self.config.n_readout, 
                                dtype=torch.float32, device=device)
            
            for i in range(batch_size):
                result = self.qnode(inputs[i], weights)
                if isinstance(result, list):
                    result = torch.stack([torch.as_tensor(r, dtype=torch.float32) for r in result])
                outputs[i] = result.to(device)
            
            return outputs
        
        else:
            # Angle encoding or single sample
            result = self.qnode(inputs, weights)
            
            if isinstance(result, list):
                return torch.stack([torch.as_tensor(r, dtype=torch.float32) for r in result])
            return result


# HYBRID QUANTUM-CLASSICAL MODEL

class QNN(nn.Module):
    """
    Hybrid Quantum-Classical Neural Network.
    
    Architecture:
    1. Classical preprocessor (feature transformation)
    2. Quantum circuit (variational quantum layer)
    3. Classical classifier (final readout)
    """
    
    def __init__(self, config: QNNConfig, device: str):
        super().__init__()
        self.config = config
        self.device = device
        
        # Determine quantum input size
        if config.encoding == 'amplitude':
            q_input_size = 2 ** config.n_qubits
        else:  # angle
            q_input_size = config.n_qubits
        
        # Classical preprocessor
        hidden = max(q_input_size, config.n_features) // 2
        self.preprocessor = nn.Sequential(
            nn.Linear(config.n_features, hidden),
            nn.LayerNorm(hidden),  # LayerNorm instead of BatchNorm
            nn.Tanh(),
            nn.Linear(hidden, q_input_size),
            nn.LayerNorm(q_input_size),
            nn.Tanh()
        )
        self._init_classical_weights(self.preprocessor)
        
        # Quantum circuit
        self.quantum = QuantumCircuit(config, device)
        self.q_weights = self._init_quantum_weights(config.n_layers, config.n_qubits)
        
        # Classical classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(config.n_readout, config.n_classes)
        )
        self._init_classical_weights(self.classifier)
    
    def _init_classical_weights(self, module):
        """Initialize classical layers with Xavier initialization."""
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _init_quantum_weights(self, n_layers: int, n_qubits: int) -> nn.Parameter:
        """
        Initialize quantum weights with layer-wise scaling.
        
        Deeper layers get smaller initialization to maintain gradient flow.
        Reference: arXiv:2108.13023
        """
        weights = torch.zeros(n_layers, n_qubits, 3)
        
        for layer in range(n_layers):
            # Decrease variance with depth
            scale = 0.01 / np.sqrt(layer + 1)
            weights[layer] = torch.randn(n_qubits, 3) * scale
        
        return nn.Parameter(weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through hybrid model.
        
        Args:
            x: (batch_size, n_features) input features
        
        Returns:
            (batch_size, n_classes) logits
        """
        # Classical preprocessing
        x = self.preprocessor(x).float()
        
        # Quantum processing
        x = self.quantum.forward(x, self.q_weights)
        
        # Classical classification
        return self.classifier(x)
    
    def get_param_groups(self) -> List[Dict]:
        """
        Get parameter groups for optimizer with separate learning rates.
        
        Returns:
            List of parameter groups for optimizer
        """
        classical_params = list(self.preprocessor.parameters()) + \
                          list(self.classifier.parameters())
        
        return [
            {
                'params': classical_params,
                'lr': self.config.classical_lr,
                'weight_decay': self.config.weight_decay
            },
            {
                'params': [self.q_weights],
                'lr': self.config.quantum_lr,
                'weight_decay': 0  # No weight decay for quantum params
            }
        ]


# TRAINER

class Trainer:
    """
    Trainer with gradient monitoring and barren plateau detection.
    """
    
    def __init__(self, model: QNN, config: QNNConfig, device: str):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer with separate learning rates
        self.optimizer = optim.AdamW(
            model.get_param_groups(),
            fused=(torch.cuda.is_available() and device == 'cuda')
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10
        )
        
        # Metrics tracking
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        self.best_val_acc = 0.0
        
        # Gradient monitoring
        self.gradient_variance = []
    
    def check_barren_plateau(self) -> bool:
        """
        Check if training is in a barren plateau.
        
        Returns:
            True if barren plateau detected
        """
        if self.model.q_weights.grad is None:
            return False
        
        grad_var = self.model.q_weights.grad.var().item()
        self.gradient_variance.append(grad_var)
        
        if grad_var < 1e-6:
            return True
        
        return False
    
    def train_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        """Train one epoch."""
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        
        for data, target in loader:
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            outputs = self.model(data)
            loss = self.criterion(outputs, target)
            
            if torch.isnan(loss):
                continue
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.grad_clip
            )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            correct += (outputs.argmax(1) == target).sum().item()
            total += target.size(0)
        
        return total_loss / len(loader), correct / total
    
    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Tuple[float, float]:
        """Evaluate model."""
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        
        for data, target in loader:
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            
            outputs = self.model(data)
            loss = self.criterion(outputs, target)
            
            total_loss += loss.item()
            correct += (outputs.argmax(1) == target).sum().item()
            total += target.size(0)
        
        return total_loss / len(loader), correct / total
    
    def train(self, train_loader: DataLoader, 
             val_loader: Optional[DataLoader] = None) -> Dict:
        """Main training loop."""
        print(f"\n{'='*80}")
        print(f"Training QNN on {self.device}")
        print(f"  Total params: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Quantum params: {self.model.q_weights.numel():,}")
        print(f"{'='*80}\n")
        
        for epoch in range(self.config.epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            # Check for barren plateau
            if self.check_barren_plateau():
                print(f"  ⚠️  Barren plateau detected at epoch {epoch+1}")
            
            if val_loader:
                val_loss, val_acc = self.evaluate(val_loader)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                self.best_val_acc = max(self.best_val_acc, val_acc)
                
                print(f"Epoch {epoch+1:3d}/{self.config.epochs} | "
                      f"Train: {train_acc:.4f} | Val: {val_acc:.4f} | "
                      f"Loss: {train_loss:.4f}")
            else:
                print(f"Epoch {epoch+1:3d}/{self.config.epochs} | "
                      f"Train Acc: {train_acc:.4f} | Loss: {train_loss:.4f}")
            
            self.scheduler.step()
        
        print(f"\n{'='*80}")
        print(f"Training Complete! Best Val Acc: {self.best_val_acc:.4f}")
        print(f"{'='*80}\n")
        
        return self.history


# FEDERATED UTILITIES

def fedavg_weights(
    client_updates: List[Tuple[Dict[str, torch.Tensor], int, float]],
    client_samples: List[int]
) -> Tuple[Dict[str, torch.Tensor], float]:
    """
    FedAvg aggregation with quantum weight clamping.
    
    Args:
        client_updates: List of (state_dict, n_samples, loss)
        client_samples: List of sample counts
    
    Returns:
        (aggregated_state_dict, weighted_avg_loss)
    """
    if not client_updates:
        raise ValueError("No client updates to aggregate")
    
    total_samples = sum(client_samples)
    if total_samples == 0:
        raise ValueError("Total samples is zero")
    
    # Initialize aggregated state
    template = client_updates[0][0]
    aggregated = {k: torch.zeros_like(v) for k, v in template.items()}
    
    weighted_loss = 0.0
    
    # Weighted averaging
    for (state_dict, _, loss), n_samples in zip(client_updates, client_samples):
        weight = n_samples / total_samples
        weighted_loss += weight * loss
        
        for k in aggregated:
            aggregated[k] += state_dict[k] * weight
    
    # Wrap quantum weights to [-pi, pi] to maintain periodicity and prevent drift
    for k in aggregated:
        if 'q_weights' in k:
            aggregated[k] = torch.remainder(aggregated[k] + torch.pi, 2 * torch.pi) - torch.pi
    
    return aggregated, weighted_loss


# MAIN

def main():
    """Example usage."""
    config = QNNConfig(
        n_qubits=4,
        n_layers=2,
        encoding='angle',
        entanglement='pyramid',
        measurement='multi_basis',
        n_features=4,
        n_classes=3,
        batch_size=32,
        epochs=10,
        seed=42
    )
    
    set_seeds(config.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model
    model = QNN(config, device)
    
    # Create synthetic data
    X_train = torch.randn(1000, config.n_features)
    y_train = torch.randint(0, config.n_classes, (1000,))
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=config.batch_size,
        shuffle=True
    )
    
    # Train
    trainer = Trainer(model, config, device)
    history = trainer.train(train_loader)
    
    print(f"Final training accuracy: {history['train_acc'][-1]:.4f}")
    
    return model, history

if __name__ == "__main__":
    model, history = main()