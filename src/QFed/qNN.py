import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
from typing import Optional, Literal, Tuple, List, Dict, Callable
from dataclasses import dataclass, asdict
from torch.utils.data import DataLoader, TensorDataset

# Note: qAngle and qAmplitude modules provide encoding utilities
# The preprocessor handles dimension transformation
# The quantum circuit handles final range mapping for gates
from data.preprocess import preprocess_mnist

torch.set_default_dtype(torch.float32)

@dataclass
class QNNConfig:
    # Configuration for Quantum Neural Network.
    # Quantum circuit parameters
    n_qubits: int = 4
    n_layers: int = 2
    n_readout: Optional[int] = None
    encoding: Literal['amplitude', 'angle'] = 'amplitude'
    entanglement: Literal['linear', 'circular', 'full'] = 'circular'
    
    # Data parameters
    n_features: int = 4
    n_classes: int = 3
    
    # Training parameters
    batch_size: int = 16
    epochs: int = 1     # 20 for real training
    classical_lr: float = 1e-3
    quantum_lr: float = 5e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    
    # Device parameters
    device_name: str = 'default.qubit'
    diff_method: str = 'backprop'  # 'backprop' or 'parameter-shift'
    shots: Optional[int] = None
    
    def __post_init__(self):
        if self.n_readout is None:
            self.n_readout = self.n_qubits
        if self.n_readout > self.n_qubits:
            raise ValueError(f"n_readout ({self.n_readout}) > n_qubits ({self.n_qubits})")
        
        # Auto-select diff_method based on shots
        if self.shots is not None and self.diff_method == 'backprop':
            self.diff_method = 'parameter-shift'
    
    def to_dict(self) -> Dict:
        return asdict(self)


class ClassicalPreprocessor(nn.Module):
    """
    Classical preprocessing layer with batch normalization.
    
    Transforms input features to appropriate dimension for quantum encoding.
    Uses batch normalization for stable training and Tanh activation for bounded outputs suitable for quantum circuits.
    """
    
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
        # Xavier initialization for better gradient flow.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch_size, n_features)
        
        Returns:
            Preprocessed tensor in [-1, 1] range (encoding-agnostic)
        """
        # Output is in [-1, 1] due to Tanh
        # Let encoding layer handle range mapping
        return self.network(x)


class QuantumCircuit:
    """
    Variational Quantum Circuit (VQC) for quantum machine learning.
    
    Implements a parameterized quantum circuit with:
    - Data encoding layer (amplitude or angle)
    - Variational layers with single-qubit rotations
    - Entangling gates (CNOT)
    - Measurement in computational basis
    """
    
    def __init__(self, config: QNNConfig):
        self.config = config
        self.n_qubits = config.n_qubits
        self.n_layers = config.n_layers
        self.n_readout = config.n_readout
        self.encoding = config.encoding
        self.entanglement = config.entanglement
        
        # Initialize quantum device
        self.dev = qml.device(
            config.device_name,
            wires=self.n_qubits,
            shots=config.shots
        )
        
        # Pre-compute entanglement structure for efficiency
        self.entanglement_pairs = self._build_entanglement_structure()
        
        # Build QNode
        self.qnode = self._build_qnode()
    
    def _build_entanglement_structure(self) -> List[Tuple[int, int]]:
        # Pre-compute wire pairs for entanglement.
        if self.entanglement == 'linear':
            return [(i, i + 1) for i in range(self.n_qubits - 1)]
        elif self.entanglement == 'circular':
            return [(i, (i + 1) % self.n_qubits) for i in range(self.n_qubits)]
        elif self.entanglement == 'full':
            return [(i, j) for i in range(self.n_qubits) 
                    for j in range(i + 1, self.n_qubits)]
        else:
            raise ValueError(f"Unknown entanglement: {self.entanglement}")
    
    def _encoding_layer(self, inputs: torch.Tensor):
        """
        Data encoding layer that properly integrates with our custom encoding modules.
        
        Args:
            inputs: Preprocessed tensor in [-1, 1] range
        """
        if self.encoding == 'angle':
            # For angle encoding: map [-1, 1] → [0, π]
            # Each feature becomes a rotation angle for one qubit
            angles = (inputs + 1.0) * (np.pi / 2.0)
            
            for i in range(self.n_qubits):
                qml.RY(angles[i], wires=i)
        
        else: 
            # For amplitude encoding: normalize to unit vector
            # The preprocessor outputs 2^n_qubits values in [-1, 1]
            
            # L2 normalization for valid quantum state
            norm = torch.norm(inputs) + 1e-8
            amplitudes = inputs / norm

            # PennyLane's AmplitudeEmbedding expects the state to sum to 1 in absolute square
            qml.AmplitudeEmbedding(
                amplitudes,
                wires=range(self.n_qubits),
                normalize=True,
                pad_with=0.0
            )
    
    def _variational_layer(self, params: torch.Tensor):
        # Single variational layer with rotations and entanglement.
        # params shape: (n_qubits, 2) for RY and RZ
        
        # Single-qubit rotations
        for i in range(self.n_qubits):
            qml.RY(params[i, 0], wires=i)
            qml.RZ(params[i, 1], wires=i)
        
        # Entangling gates
        for ctrl, tgt in self.entanglement_pairs:
            qml.CNOT(wires=[ctrl, tgt])
    
    def _build_qnode(self) -> Callable:
        # Build the quantum node with proper interface.
        
        @qml.qnode(self.dev, interface='torch', diff_method=self.config.diff_method)
        def circuit(inputs: torch.Tensor, weights: torch.Tensor):
            """
            Quantum circuit execution.
            
            Args:
                inputs: Encoded input features
                weights: Variational parameters (n_layers, n_qubits, 2)
            
            Returns:
                Expectation values of Pauli-Z for readout qubits
            """
            # Encoding
            self._encoding_layer(inputs)
            
            # Variational layers
            for layer_idx in range(self.n_layers):
                self._variational_layer(weights[layer_idx])
            
            # Measurement
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_readout)]
        
        return circuit
    
    def forward(self, inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Execute quantum circuit (single sample).
        
        Args:
            inputs: Single input vector
            weights: Variational parameters
        
        Returns:
            Measurement outcomes (n_readout,)
        """
        result = self.qnode(inputs, weights)
        
        # Ensure tensor format
        if isinstance(result, (list, tuple)):
            result = torch.stack([torch.as_tensor(r, dtype=torch.float32) for r in result])
        else:
            result = torch.as_tensor(result, dtype=torch.float32)
        
        return result

# Hybrid Quantum Neural Network

class QuantumNeuralNetwork(nn.Module):
    """
    Hybrid Quantum-Classical Neural Network.
    
    Architecture:
        1. Classical Preprocessor (features → quantum input)
        2. Quantum Variational Circuit (quantum processing)
        3. Classical Classifier (quantum output → class logits)
    """
    
    def __init__(self, config: QNNConfig):
        super().__init__()
        self.config = config
        
        # Determine quantum input size
        if config.encoding == 'angle':
            quantum_input_size = config.n_qubits
        else:  # amplitude
            quantum_input_size = 2 ** config.n_qubits
        
        # 1. Classical preprocessor
        self.preprocessor = ClassicalPreprocessor(
            config.n_features,
            quantum_input_size,
            config.encoding
        )
        
        # 2. Quantum circuit
        self.quantum_circuit = QuantumCircuit(config)
        
        # 3. Quantum parameters (trainable)
        n_params = config.n_layers * config.n_qubits * 2
        self.q_weights = nn.Parameter(
            0.01 * torch.randn(config.n_layers, config.n_qubits, 2)
        )
        
        # 4. Classical classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(config.n_readout, config.n_classes)
        )
        
        # Initialize classifier
        nn.init.xavier_uniform_(self.classifier[1].weight, gain=0.5)
        nn.init.zeros_(self.classifier[1].bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through hybrid network.
        
        Args:
            x: Input batch (batch_size, n_features)
        
        Returns:
            Class logits (batch_size, n_classes)
        """
        batch_size = x.shape[0]
        device = x.device
        
        # 1. Classical preprocessing
        preprocessed = self.preprocessor(x)
        
        # 2. Quantum processing (batch loop - PennyLane limitation)
        q_outputs = []
        
        for i in range(batch_size):
            sample = preprocessed[i]
            
            try:
                q_out = self.quantum_circuit.forward(sample, self.q_weights)
                
                # Ensure proper device and dtype
                if q_out.device != device:
                    q_out = q_out.to(device)
                if q_out.dtype != torch.float32:
                    q_out = q_out.float()
                
                q_outputs.append(q_out)
            
            except Exception as e:
                print(f"Warning: Quantum circuit failed for sample {i}: {e}")
                # Fallback to zero output
                q_outputs.append(torch.zeros(
                    self.config.n_readout,
                    dtype=torch.float32,
                    device=device
                ))
        
        # Stack batch
        q_batch = torch.stack(q_outputs, dim=0)  # (batch_size, n_readout)
        
        # 3. Classical classification
        logits = self.classifier(q_batch)
        
        return logits
    
    def get_quantum_params(self) -> List[nn.Parameter]:
        # Get quantum circuit parameters.
        return [self.q_weights]
    
    def get_classical_params(self) -> List[nn.Parameter]:
        # Get classical network parameters.
        return (list(self.preprocessor.parameters()) + 
                list(self.classifier.parameters()))

# Training Utilities

class QNNTrainer:
    # Trainer for Quantum Neural Network with best practices.
    
    def __init__(
        self,
        model: QuantumNeuralNetwork,
        config: QNNConfig,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer with separate learning rates
        self.optimizer = optim.AdamW([
            {
                "params": model.get_classical_params(),
                "lr": config.classical_lr,
                "weight_decay": config.weight_decay
            },
            {
                "params": model.get_quantum_params(),
                "lr": config.quantum_lr,
                "weight_decay": 0  # No regularization for quantum params
            }
        ])
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2
        )
        
        # Metrics tracking
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_val_acc = 0.0
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        # Train for one epoch.
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            try:
                # Forward pass
                outputs = self.model(data)
                loss = self.criterion(outputs, target)
                
                # Check for NaN
                if torch.isnan(loss):
                    print(f"Warning: NaN loss in batch {batch_idx}, skipping...")
                    continue
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.grad_clip
                    )
                
                self.optimizer.step()
                
                # Metrics
                total_loss += loss.item()
                num_batches += 1
                
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == target).sum().item()
                total += target.size(0)
            
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / max(num_batches, 1)
        accuracy = correct / max(total, 1)
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Tuple[float, float]:
        # Evaluate on a dataset.
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0
        
        for data, target in loader:
            data, target = data.to(self.device), target.to(self.device)
            
            try:
                outputs = self.model(data)
                loss = self.criterion(outputs, target)
                
                total_loss += loss.item()
                num_batches += 1
                
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == target).sum().item()
                total += target.size(0)
            
            except Exception as e:
                print(f"Warning: Evaluation error: {e}")
                continue
        
        avg_loss = total_loss / max(num_batches, 1)
        accuracy = correct / max(total, 1)
        
        return avg_loss, accuracy
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
                verbose: bool = True) -> Dict:
        """
        Full training loop.
        
        Returns:
            Dictionary with training history
        """
        if verbose:
            print("\nQuantum Neural Network Training")
            print(f"Device: {self.device}")
            print(f"Total Parameters: {sum(p.numel() for p in self.model.parameters())}")
            print(f"Quantum Parameters: {self.model.q_weights.numel()}")
            print(f"Classical Parameters: {sum(p.numel() for p in self.model.get_classical_params())} \n")
        
        for epoch in range(self.config.epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # Validation
            if val_loader is not None:
                val_loss, val_acc = self.evaluate(val_loader)
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_acc)
                
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
            
            # Update learning rate
            self.scheduler.step()
            
            # Logging
            if verbose and ((epoch + 1) % 2 == 0 or epoch == 0):
                lr = self.optimizer.param_groups[0]['lr']
                if val_loader is not None:
                    print(f"Epoch {epoch+1:3d}/{self.config.epochs} | "
                          f"Loss: {train_loss:.4f} | "
                          f"Train Acc: {train_acc:.4f} | "
                          f"Val Acc: {val_acc:.4f} | "
                          f"LR: {lr:.6f}")
                else:
                    print(f"Epoch {epoch+1:3d}/{self.config.epochs} | "
                          f"Loss: {train_loss:.4f} | "
                          f"Train Acc: {train_acc:.4f} | "
                          f"LR: {lr:.6f}")
        
        if verbose:
            print("\n" + "=" * 70)
            print("Training Complete!")
            if self.best_val_acc > 0:
                print(f"Best Validation Accuracy: {self.best_val_acc:.4f}")
            print("=" * 70 + "\n")
        
        return {
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'best_val_acc': self.best_val_acc
        }

def main():
        # Configuration
    config = QNNConfig(
        # Quantum parameters
        n_qubits=4,
        n_layers=2,
        encoding='amplitude',  # 'amplitude' or 'angle'
        entanglement='circular',  # 'linear', 'circular', 'full'
        
        # Data parameters
        n_features=4,  # Will be updated from data
        n_classes=3,
        
        # Training parameters
        batch_size=16,
        epochs=1,  # 20 for real training
        classical_lr=1e-3,
        quantum_lr=5e-4,
        weight_decay=1e-4,
        grad_clip=1.0,
        
        # Device parameters
        device_name='default.qubit',
        diff_method='backprop',
        shots=None
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Quantum Neural Network for Federated Learning")
    print(f"Framework: PennyLane + PyTorch")
    print(f"Device: {device}")
    print(f"Configuration:")
    for k, v in config.to_dict().items():
        print(f"  {k:20s}: {v}")
    
    # Load data
    try:
        train_set, val_set, test_set, _ = preprocess_mnist(
            raw_folder="./dataset/raw",
            processed_folder="./dataset/processed",
            digits=(0, 1, 2),
            val_split=0.1,
            num_clients=4,
            partition_type='iid',
            apply_pca=True,
            pca_components=config.n_features,
            generate_plots=False
        )
        
        X_train, y_train = train_set
        X_val, y_val = val_set
        
        config.n_features = X_train.shape[1]
        print(f"\nData loaded: {X_train.shape[0]} train, {X_val.shape[0]} val samples")
        print(f"Feature dimension: {config.n_features}\n")
    
    except Exception as e:
        print(f"Using dummy data due to error: {e}\n")
        np.random.seed(42)
        X_train = torch.randn(1000, config.n_features)
        y_train = torch.randint(0, config.n_classes, (1000,))
        X_val = torch.randn(200, config.n_features)
        y_val = torch.randint(0, config.n_classes, (200,))
    
    # Create data loaders
    train_loader = DataLoader(
        TensorDataset(
            torch.as_tensor(X_train, dtype=torch.float32),
            torch.as_tensor(y_train, dtype=torch.long)
        ),
        batch_size=config.batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        TensorDataset(
            torch.as_tensor(X_val, dtype=torch.float32),
            torch.as_tensor(y_val, dtype=torch.long)
        ),
        batch_size=config.batch_size,
        shuffle=False
    )
    
    # Initialize model
    model = QuantumNeuralNetwork(config)
    
    # Initialize trainer
    trainer = QNNTrainer(model, config, device)
    
    # Train
    history = trainer.train(train_loader, val_loader)
    
    # Save model
    save_dict = {
        'model_state_dict': model.state_dict(),
        'config': config.to_dict(),
        'history': history
    }
    
    save_path = './dataset/processed/qNN_pennylane_model.pt'
    torch.save(save_dict, save_path)
    print(f"Model saved to '{save_path}'")
    
    return model, history

if __name__ == "__main__":
    model, history = main()