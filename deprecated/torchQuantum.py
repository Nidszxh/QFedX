import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  

import torch
import torch.nn as nn
import torch.optim as optim
import torchquantum as tq
import numpy as np
from typing import Optional, Literal
from torch.utils.data import DataLoader, TensorDataset

# Import the data                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          preprocessing and encoding modules
from qAngle import angle_encode, pool_to_n_features
from qAmplitude import amplitude_encode, normalize_for_amplitude, pad_to_pow2
from data.preprocess import preprocess_mnist

torch.set_default_dtype(torch.float32)

class TorchQuantumQNN(nn.Module):
    """
    Quantum Neural Network (QNN) using TorchQuantum with hybrid architecture:
    - Classical preprocessor
    - Quantum variational layers
    - Classical head for output
    """
    
    def __init__(self, 
                 n_qubits: int = 4,
                 n_features: int = 4, 
                 n_classes: int = 3, 
                 encoding: Literal['amplitude', 'angle'] = 'angle',
                 n_layers: int = 2,
                 n_readout: Optional[int] = None,
                 entanglement: Literal['linear', 'circular', 'full'] = 'linear'):

        super().__init__()
        
        self.n_qubits = n_qubits
        self.n_features = n_features
        self.n_classes = n_classes
        self.encoding = encoding
        self.n_layers = n_layers
        self.n_readout = n_readout or n_qubits
        self.entanglement = entanglement
        
        # Validate inputs
        if self.n_readout > n_qubits:
            raise ValueError(f"n_readout ({self.n_readout}) cannot be greater than n_qubits ({n_qubits})")
        
        # Initialize quantum device
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        
        # HYBRID ARCHITECTURE COMPONENTS
        
        # 1. Classical preprocessor (adapted for custom encoding modules)
        if encoding == 'angle':
            # For angle encoding, we might need different preprocessing
            # depending on what your qAngle module expects
            self.pre_fc = nn.Sequential(
                nn.Linear(n_features, n_qubits),
                nn.Tanh()  # Keep in [-1,1] for angle encoding
            )
            self.quantum_input_size = n_qubits
        else:  # amplitude encoding
            # For amplitude encoding, prepare data for your qAmplitude module
            # The module will handle padding to power of 2, so we can be flexible here
            target_size = min(2 ** n_qubits, n_features * 2)  # Reasonable intermediate size
            self.pre_fc = nn.Sequential(
                nn.Linear(n_features, target_size),
                nn.Tanh()
            )
            self.quantum_input_size = target_size
        
        # 2. Quantum circuit layers
        self.q_layers = nn.ModuleList()
        
        # Initial encoding layer (configured for custom modules)
        if encoding == 'angle':
            # For angle encoding, we'll use the custom qAngle module
            # The TorchQuantum encoder will be used as fallback
            self.encoder = tq.GeneralEncoder([
                {'input_idx': [i], 'func': 'ry', 'wires': [i]} 
                for i in range(n_qubits)
            ])
        else:
            # For amplitude encoding, we rely on custom qAmplitude module
            # No TorchQuantum encoder needed as we'll set states directly
            self.encoder = None
        
        # Variational layers
        for _ in range(n_layers):
            layer = nn.ModuleList([
                # Single-qubit rotations for each qubit
                nn.ModuleList([
                    tq.RY(has_params=True, trainable=True),
                    tq.RZ(has_params=True, trainable=True)
                ]) for _ in range(n_qubits)
            ])
            self.q_layers.append(layer)
        
        # Entangling gates (fixed structure, no parameters)
        self.entangling_layers = nn.ModuleList()
        for _ in range(n_layers):
            entangling_ops = nn.ModuleList()
            if entanglement == 'linear':
                for i in range(n_qubits - 1):
                    entangling_ops.append(tq.CNOT())
            elif entanglement == 'circular':
                for i in range(n_qubits):
                    entangling_ops.append(tq.CNOT())
            elif entanglement == 'full':
                for i in range(n_qubits):
                    for j in range(i + 1, n_qubits):
                        entangling_ops.append(tq.CNOT())
            self.entangling_layers.append(entangling_ops)
        
        # 3. Classical head (output layer)
        self.classical_head = nn.Linear(self.n_readout, n_classes)
        
        # Initialize classical layers with small weights
        self._init_classical_weights()
        
    def _init_classical_weights(self):
        """Initialize classical layer weights"""
        for module in [self.pre_fc, self.classical_head]:
            if isinstance(module, nn.Sequential):
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight, gain=0.1)
                        if layer.bias is not None:
                            nn.init.constant_(layer.bias, 0.01)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)

    def _apply_entanglement(self, layer_idx):
        """Apply entangling gates for a specific layer"""
        entangling_ops = self.entangling_layers[layer_idx]
        
        if self.entanglement == 'linear':
            for i, cnot in enumerate(entangling_ops):
                cnot(self.q_device, wires=[i, i + 1])
        elif self.entanglement == 'circular':
            for i, cnot in enumerate(entangling_ops):
                cnot(self.q_device, wires=[i, (i + 1) % self.n_qubits])
        elif self.entanglement == 'full':
            idx = 0
            for i in range(self.n_qubits):
                for j in range(i + 1, self.n_qubits):
                    entangling_ops[idx](self.q_device, wires=[i, j])
                    idx += 1

    def quantum_forward(self, x):
        """Forward pass through quantum circuit using custom encoding modules"""
        batch_size = x.shape[0]
        device = x.device
        
        # Initialize quantum device for batch processing
        self.q_device.reset_states(batch_size)
        
        if self.encoding == 'angle':
            # Use custom angle encoding from qAngle module
            try:
                # Apply angle encoding using your custom module
                encoded_data = angle_encode(x)  # Your custom angle encoding
                
                # If angle_encode returns data that needs pooling to fit n_qubits
                if encoded_data.shape[1] != self.n_qubits:
                    encoded_data = pool_to_n_features(encoded_data, self.n_qubits)
                
                # Ensure proper device and dtype
                if isinstance(encoded_data, np.ndarray):
                    encoded_data = torch.tensor(encoded_data, dtype=torch.float32, device=device)
                else:
                    encoded_data = encoded_data.to(device=device, dtype=torch.float32)
                
                # Apply the encoded angles to quantum device
                self.encoder(self.q_device, encoded_data)
                
            except Exception as e:
                print(f"Warning: Custom angle encoding failed ({e}), using fallback")
                # Fallback to simple encoding
                self.encoder(self.q_device, x)
                
        else:  # amplitude encoding
            # Use custom amplitude encoding from qAmplitude module
            try:
                # Process each sample in batch
                encoded_states = []
                for b in range(batch_size):
                    sample = x[b]
                    
                    # Apply your custom amplitude encoding pipeline
                    # 1. Normalize for amplitude encoding
                    normalized_sample = normalize_for_amplitude(sample.cpu().numpy() if sample.is_cuda else sample.numpy())
                    
                    # 2. Pad to power of 2 if needed
                    padded_sample = pad_to_pow2(normalized_sample)
                    
                    # 3. Apply amplitude encoding
                    encoded_sample = amplitude_encode(padded_sample, self.n_qubits)
                    
                    # Convert back to tensor if needed
                    if isinstance(encoded_sample, np.ndarray):
                        encoded_sample = torch.tensor(encoded_sample, dtype=torch.float32, device=device)
                    
                    encoded_states.append(encoded_sample)
                
                # Apply amplitude encoding by setting quantum states
                for b, amplitudes in enumerate(encoded_states):
                    # Ensure proper normalization
                    amplitudes = amplitudes / (torch.norm(amplitudes) + 1e-8)
                    
                    # Set the quantum state for this batch element
                    # Note: This assumes TorchQuantum supports batch state setting
                    if hasattr(self.q_device, 'set_states_batch'):
                        # If batch setting is supported
                        if b == 0:  # Initialize batch states
                            batch_states = amplitudes.unsqueeze(0)
                        else:
                            batch_states = torch.cat([batch_states, amplitudes.unsqueeze(0)], dim=0)
                    else:
                        # Set state individually (fallback)
                        self.q_device.set_states(amplitudes.unsqueeze(0), wires=list(range(self.n_qubits)))
                
                # If we accumulated batch states, set them all at once
                if 'batch_states' in locals():
                    try:
                        self.q_device.set_states_batch(batch_states, wires=list(range(self.n_qubits)))
                    except:
                        # Fallback to individual setting
                        for b, state in enumerate(batch_states):
                            self.q_device.set_states(state.unsqueeze(0), wires=list(range(self.n_qubits)))
                    
            except Exception as e:
                print(f"Warning: Custom amplitude encoding failed ({e}), using fallback")
                # Fallback to simple normalization
                for b in range(batch_size):
                    input_sample = x[b]
                    input_normalized = input_sample / (torch.norm(input_sample) + 1e-8)
                    
                    # Pad or truncate to match 2^n_qubits
                    target_size = 2 ** self.n_qubits
                    if input_normalized.shape[0] > target_size:
                        input_normalized = input_normalized[:target_size]
                    elif input_normalized.shape[0] < target_size:
                        padding = torch.zeros(target_size - input_normalized.shape[0], device=device)
                        input_normalized = torch.cat([input_normalized, padding])
                    
                    self.q_device.set_states(input_normalized.unsqueeze(0), wires=list(range(self.n_qubits)))
        
        # Apply variational layers
        for layer_idx in range(self.n_layers):
            # Single-qubit rotations
            for qubit_idx in range(self.n_qubits):
                ry_gate, rz_gate = self.q_layers[layer_idx][qubit_idx]
                ry_gate(self.q_device, wires=qubit_idx)
                rz_gate(self.q_device, wires=qubit_idx)
            
            # Entangling gates
            self._apply_entanglement(layer_idx)
        
        # Measurement: expectation values of Pauli-Z on readout qubits
        measurements = []
        for i in range(self.n_readout):
            exp_val = tq.expval(self.q_device, i, 'z')
            measurements.append(exp_val)
        
        return torch.stack(measurements, dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        device = x.device
        
        # 1. Classical preprocessing
        pre_processed = self.pre_fc(x)
        
        if self.encoding == 'angle':
            # Scale to appropriate range for angles (0 to π)
            quantum_input = torch.abs(pre_processed) * np.pi
        else:
            # For amplitude encoding, keep in reasonable range
            quantum_input = pre_processed
        
        # 2. Quantum processing
        try:
            q_out = self.quantum_forward(quantum_input)
            
            # Ensure q_out has correct device and dtype
            if q_out.device != device:
                q_out = q_out.to(device)
            if q_out.dtype != torch.float32:
                q_out = q_out.float()
                
        except Exception as e:
            print(f"Warning: Quantum circuit failed: {e}")
            # Fallback: use zero output with proper device and dtype
            q_out = torch.zeros(batch_size, self.n_readout, 
                              dtype=torch.float32, device=device)
        
        # 3. Classical head
        logits = self.classical_head(q_out)
        return logits


def train_torchquantum_qnn(model, train_loader, val_loader=None, 
                          epochs=10, classical_lr=1e-3, quantum_lr=5e-4, 
                          device='cpu', print_every=2, grad_clip=1.0):
    """
    Training function with separate learning rates for classical and quantum parameters.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Separate optimizers for classical and quantum parameters
    classical_params = list(model.pre_fc.parameters()) + list(model.classical_head.parameters())
    
    # Get quantum parameters from TorchQuantum layers
    quantum_params = []
    for layer in model.q_layers:
        for qubit_gates in layer:
            for gate in qubit_gates:
                quantum_params.extend(gate.parameters())
    
    optimizer = optim.Adam([
        {"params": classical_params, "lr": classical_lr},
        {"params": quantum_params, "lr": quantum_lr}
    ])
    
    # Optional scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)

    train_losses = []
    val_accuracies = []
    
    print(f"Starting training for {epochs} epochs...")
    print(f"Classical LR: {classical_lr}, Quantum LR: {quantum_lr}")
    print(f"Device: {device}")
    print(f"Using TorchQuantum backend")
    print("-" * 50)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        correct_predictions = 0
        total_predictions = 0
    
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            try:
                # Forward pass
                outputs = model(data)
                loss = criterion(outputs, target)
                
                # Check for NaN loss
                if torch.isnan(loss):
                    print(f"Warning: NaN loss detected in batch {batch_idx}, skipping...")
                    continue
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping for stability
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Calculate training accuracy
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == target).sum().item()
                total_predictions += target.size(0)
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        if num_batches == 0:
            print(f"Warning: No successful batches in epoch {epoch}")
            continue
            
        avg_loss = total_loss / num_batches
        train_acc = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        train_losses.append(avg_loss)
        
        # Validation
        val_acc = 0.0
        if val_loader is not None:
            val_acc = evaluate_model(model, val_loader, device)
            val_accuracies.append(val_acc)
            scheduler.step(-val_acc)  # Negative because we want to maximize accuracy
        else:
            scheduler.step(avg_loss)  # Use training loss if no validation
        
        # Print progress
        if (epoch + 1) % print_every == 0 or epoch == 0:
            if val_loader is not None:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
            else:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}")
            
            # Print parameter info for debugging
            total_q_params = sum(p.numel() for p in quantum_params)
            print(f"  Quantum parameters count: {total_q_params}")
    
    return train_losses, val_accuracies


def evaluate_model(model, data_loader, device='cpu'):
    """Evaluate model on given data loader."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            try:
                outputs = model(data)
                _, predicted = torch.max(outputs, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
            except Exception as e:
                print(f"Error in evaluation batch: {e}")
                continue
    
    return correct / total if total > 0 else 0.0


def main():
    """Main function with TorchQuantum implementation."""
    
    # Configuration
    config = {
        'n_qubits': 4,
        'n_features': 16,  # Will be adjusted based on PCA
        'n_classes': 3,
        'encoding': 'angle',  # TorchQuantum works better with angle encoding
        'n_layers': 2,
        'entanglement': 'circular',
        'batch_size': 16,
        'epochs': 20,
        'classical_lr': 1e-3,
        'quantum_lr': 5e-4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }
    
    print("Quantum Neural Network Training (TorchQuantum)")
    print(f"Config: {config}")
    print("-" * 50)
    
    # Data preprocessing with fallback
    try:
        train_set, val_set, test_set, client_data = preprocess_mnist(
            raw_folder="./dataset/raw",
            processed_folder="./dataset/processed",
            digits=(0, 1, 2),
            val_split=0.1,
            num_clients=4,
            partition_type='iid',
            alpha=0.5,
            apply_pca=True,
            pca_components=config['n_features']
        )
        
        X_train, y_train = train_set
        X_val, y_val = val_set
        
        # Update n_features based on actual data
        config['n_features'] = X_train.shape[1]
        print(f"Successfully loaded MNIST data with {config['n_features']} features")
        
    except Exception as e:
        print(f"Using dummy data due to preprocessing error: {e}")
        # Generate more realistic dummy data
        np.random.seed(42)
        X_train = torch.tensor(np.random.randn(1000, config['n_features']), dtype=torch.float32)
        y_train = torch.tensor(np.random.randint(0, config['n_classes'], 1000), dtype=torch.long)
        X_val = torch.tensor(np.random.randn(200, config['n_features']), dtype=torch.float32)
        y_val = torch.tensor(np.random.randint(0, config['n_classes'], 200), dtype=torch.long)
    
    # Convert to proper torch tensors
    if not isinstance(X_train, torch.Tensor):
        X_train = torch.tensor(X_train, dtype=torch.float32)
    if not isinstance(y_train, torch.Tensor):
        y_train = torch.tensor(y_train, dtype=torch.long)
    if not isinstance(X_val, torch.Tensor):
        X_val = torch.tensor(X_val, dtype=torch.float32)
    if not isinstance(y_val, torch.Tensor):
        y_val = torch.tensor(y_val, dtype=torch.long)
    
    print(f"Data shapes: Train {X_train.shape}, Val {X_val.shape}")
    print(f"Class distribution: {torch.bincount(y_train)}")
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    # Initialize QNN
    try:
        model = TorchQuantumQNN(
            n_qubits=config['n_qubits'],
            n_features=config['n_features'],
            n_classes=config['n_classes'],
            encoding=config['encoding'],
            n_layers=config['n_layers'],
            entanglement=config['entanglement']
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        
        # Count quantum parameters
        quantum_params = []
        for layer in model.q_layers:
            for qubit_gates in layer:
                for gate in qubit_gates:
                    quantum_params.extend(gate.parameters())
        quantum_param_count = sum(p.numel() for p in quantum_params)
        classical_param_count = total_params - quantum_param_count
        
        print(f"Model initialized successfully:")
        print(f"  Total parameters: {total_params}")
        print(f"  Quantum parameters: {quantum_param_count}")
        print(f"  Classical parameters: {classical_param_count}")
        print(f"  Using TorchQuantum backend")
        print()
        
    except Exception as e:
        print(f"Error initializing model: {e}")
        return None, None, None
    
    # Train the model
    try:
        train_losses, val_accuracies = train_torchquantum_qnn(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config['epochs'],
            classical_lr=config['classical_lr'],
            quantum_lr=config['quantum_lr'],
            device=config['device']
        )
        
        print(f"\nTraining completed!")
        if train_losses:
            print(f"Final training loss: {train_losses[-1]:.4f}")
        if val_accuracies:
            print(f"Final validation accuracy: {val_accuracies[-1]:.4f}")
            print(f"Best validation accuracy: {max(val_accuracies):.4f}")

        # Save model with complete metadata
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'train_losses': train_losses,
            'val_accuracies': val_accuracies,
            'model_class': 'TorchQuantumQNN'
        }, 'torchquantum_qnn_model.pt')

        print("Model saved as 'torchquantum_qnn_model.pt'")
        
    except Exception as e:
        print(f"Error during training: {e}")
        return model, None, None
    
    return model, train_losses, val_accuracies


if __name__ == "__main__":
    model, losses, accuracies = main()