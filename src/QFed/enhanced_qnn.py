import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  

import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
import numpy as np
from typing import Optional, Literal
from torch.utils.data import DataLoader, TensorDataset

from qAngle import angle_encode, pool_to_n_features
from qAmplitude import amplitude_encode, normalize_for_amplitude, pad_to_pow2
from data.preprocess import preprocess_mnist

torch.set_default_dtype(torch.float32)

class EnhancedQNN(nn.Module):
    """
    Enhanced QNN with hybrid architecture following best practices:
    - Classical preprocessor
    - Quantum variational layers
    - Classical head for output
    """
    
    def __init__(self, 
                 n_qubits: int = 4,
                 n_features: int = 4, 
                 n_classes: int = 3, 
                 encoding: Literal['amplitude', 'angle'] = 'amplitude',
                 n_layers: int = 2,
                 n_readout: Optional[int] = None,
                 entanglement: Literal['linear', 'circular', 'full'] = 'linear',
                 device_name: str = 'default.qubit',
                 shots: Optional[int] = None):

        super().__init__()
        
        self.n_qubits = n_qubits
        self.n_features = n_features
        self.n_classes = n_classes
        self.encoding = encoding
        self.n_layers = n_layers
        self.n_readout = n_readout or n_qubits
        self.entanglement = entanglement
        self.shots = shots
        
        # Initialize PennyLane device
        self.dev = qml.device("lightning.gpu", wires=n_qubits, shots=shots)
        
        # HYBRID ARCHITECTURE COMPONENTS
        
        # 1. Classical preprocessor (small front-end)
        if encoding == 'angle':
            self.pre_fc = nn.Sequential(
                nn.Linear(n_features, n_qubits),
                nn.Tanh()  # Keep in [-1,1] for angle encoding
            )
        else:  # amplitude encoding
            required_size = 2 ** n_qubits
            self.pre_fc = nn.Sequential(
                nn.Linear(n_features, required_size),
                nn.Tanh()
            )
        
        # 2. Quantum parameters (single tensor for efficiency)
        n_qparams = n_layers * n_qubits * 2  # RY + RZ per qubit per layer
        self.q_params = nn.Parameter(0.01 * torch.randn(n_qparams, dtype=torch.float32))
        
        # 3. Classical head (output layer)
        self.classical_head = nn.Linear(self.n_readout, n_classes)
        
        # Initialize classical layers with small weights
        self._init_classical_weights()
        
        # Build QNode with proper batching support
        self._build_qnode()
        
    def _init_classical_weights(self):
      modules = [self.pre_fc, self.classical_head]
      for module in modules:
            if isinstance(module, nn.Sequential):
               # Iterate over sub-layers
               for layer in module:
                  if isinstance(layer, nn.Linear):
                      nn.init.xavier_uniform_(layer.weight, gain=0.1)
                      if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0.01)
            elif isinstance(module, nn.Linear):
            # Single linear layer
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)


    
    def _build_qnode(self):
        """Build the quantum circuit with proper batching."""
        
        # Choose diff_method based on shots
        diff_method = 'parameter-shift'
        
        @qml.qnode(self.dev, interface='torch', diff_method=diff_method)
        def circuit(inputs, qparams):
            # inputs: shape (batch, n_qubits) for angle or (batch, 2^n_qubits) for amplitude
            # qparams: 1D tensor with quantum parameters
            
            if self.encoding == 'angle':
                # Angle encoding
                for i in range(self.n_qubits):
                    qml.RY(inputs[i], wires=i)
            else:
                # Amplitude encoding
                qml.AmplitudeEmbedding(inputs, wires=range(self.n_qubits), normalize=True)
            
            # Variational layers
            params = qparams.reshape(self.n_layers, self.n_qubits, 2)
            for layer in range(self.n_layers):
                # Single-qubit rotations
                for qubit in range(self.n_qubits):
                    qml.RY(params[layer, qubit, 0], wires=qubit)
                    qml.RZ(params[layer, qubit, 1], wires=qubit)
                
                # Entangling gates
                self._add_entanglement()
            
            # Readout: return expectations for readout qubits
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_readout)]
        
        # Try to use batch transform for better performance
        try:
            self.circuit = qml.batch_transform(circuit)[0]
        except (AttributeError, TypeError):
            # Fallback to regular circuit if batch_transform not available
            self.circuit = circuit
    
    def _add_entanglement(self):
        """Add entangling gates based on entanglement strategy."""
        if self.entanglement == 'linear':
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        elif self.entanglement == 'circular':
            for i in range(self.n_qubits):
                qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
        elif self.entanglement == 'full':
            for i in range(self.n_qubits):
                for j in range(i + 1, self.n_qubits):
                    qml.CNOT(wires=[i, j])

    def forward(self, x):
        batch_size = x.shape[0]

        # 1. Classical preprocessing
        pre_processed = self.pre_fc(x)

        if self.encoding == 'angle':
            angles = (pre_processed + 1.0) * (np.pi / 2.0)
            quantum_input = angles
        else:
            quantum_input = pre_processed

    # 2. Quantum processing
        q_outputs = []

        for i in range(batch_size):
            single_input = quantum_input[i]
            single_out = self.circuit(single_input, self.q_params)

            # Keep gradient flow
            single_out = torch.tensor(single_out, dtype=torch.float32, device=x.device)

            # Ensure 1D
            if single_out.dim() == 0:
              single_out = single_out.unsqueeze(0)

            q_outputs.append(single_out)  # ✅ inside the loop

        q_out = torch.stack(q_outputs, dim=0)  # shape (batch_size, n_readout)

        # 3. Classical head
        logits = self.classical_head(q_out)
        return logits


def train_enhanced_qnn(model, train_loader, val_loader=None, 
                      epochs=10, classical_lr=1e-3, quantum_lr=5e-4, 
                      device='cpu', print_every=2, grad_clip=1.0):
    """
    Training function with separate learning rates for classical and quantum parameters.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Separate optimizers for classical and quantum parameters
    classical_params = list(model.pre_fc.parameters()) + list(model.classical_head.parameters())
    quantum_params = [model.q_params]
    
    optimizer = optim.Adam([
        {"params": classical_params, "lr": classical_lr},
        {"params": quantum_params, "lr": quantum_lr}
    ])
    
    # Optional scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    train_losses = []
    val_accuracies = []
    
    print(f"Starting training for {epochs} epochs...")
    print(f"Classical LR: {classical_lr}, Quantum LR: {quantum_lr}")
    print("-" * 50)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
    
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            try:
                # Forward pass
                outputs = model(data)
                loss = criterion(outputs, target)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping for stability
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        if num_batches == 0:
            print(f"Warning: No successful batches in epoch {epoch}")
            continue
            
        avg_loss = total_loss / num_batches
        train_losses.append(avg_loss)
        
        # Validation
        val_acc = 0.0
        if val_loader is not None:
            val_acc = evaluate_model(model, val_loader, device)
            val_accuracies.append(val_acc)
            scheduler.step(val_acc)  # Use validation accuracy for scheduling
        else:
            scheduler.step(avg_loss)  # Use training loss if no validation
        
        # Print progress
        if (epoch + 1) % print_every == 0 or epoch == 0:
            if val_loader is not None:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}")
            else:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
            
            # Print parameter norms for debugging
            q_norm = torch.norm(model.q_params).item()
            print(f"  Quantum params norm: {q_norm:.4f}")
    
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
    """Main function with improved configuration and error handling."""
    
    # Configuration
    config = {
        'n_qubits': 4,
        'n_features': 16,  # Will be adjusted based on PCA
        'n_classes': 3,
        'encoding': 'amplitude',  # or 'amplitude'
        'n_layers': 2,
        'entanglement': 'circular',
        'batch_size': 16,
        'epochs': 1,
        'classical_lr': 1e-3,
        'quantum_lr': 5e-4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'shots': None,  # Use None for development, 256-1024 for production
    }
    
    print("Enhanced Quantum Neural Network Training")
    print(f"Config: {config}")
    print("-" * 50)
    
    # Data preprocessing
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
        
    except Exception as e:
        print(f"Using dummy data due to preprocessing error: {e}")
        X_train = torch.randn(200, config['n_features'])
        y_train = torch.randint(0, config['n_classes'], (200,))
        X_val = torch.randn(50, config['n_features'])
        y_val = torch.randint(0, config['n_classes'], (50,))
    
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
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    # Initialize QNN
    model = EnhancedQNN(
        n_qubits=config['n_qubits'],
        n_features=config['n_features'],
        n_classes=config['n_classes'],
        encoding=config['encoding'],
        n_layers=config['n_layers'],
        entanglement=config['entanglement'],
        shots=config['shots']
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    quantum_params = model.q_params.numel()
    classical_params = total_params - quantum_params
    
    print(f"Model initialized:")
    print(f"  Total parameters: {total_params}")
    print(f"  Quantum parameters: {quantum_params}")
    print(f"  Classical parameters: {classical_params}")
    print()
    
    # Train the model
    train_losses, val_accuracies = train_enhanced_qnn(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['epochs'],
        classical_lr=config['classical_lr'],
        quantum_lr=config['quantum_lr'],
        device=config['device']
    )
    
    print(f"\nTraining completed!")
    print(f"Final training loss: {train_losses[-1]:.4f}")
    if val_accuracies:
        print(f"Final validation accuracy: {val_accuracies[-1]:.4f}")

    # Save model with complete metadata
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'model_class': 'EnhancedQNN'
    }, 'enhanced_qnn_model.pt')

    print("Model saved as 'enhanced_qnn_model.pt'")
    
    return model, train_losses, val_accuracies


if __name__ == "__main__":
    main()