"""
Quantum Federated Learning (QFL) Implementation

Integrates quantum neural networks with federated learning for distributed quantum machine learning across multiple clients.

Architecture:
    Global Server (Quantum Model)
    ├── Client 1 (Local QNN Training)
    ├── Client 2 (Local QNN Training)
    ├── Client 3 (Local QNN Training)
    └── Client N (Local QNN Training)
    
    -> Federated Averaging (Quantum + Classical)
    
    -> Updated Global Model
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import os
import numpy as np
import torch
import torch.nn as nn
import random
from typing import List, Tuple, Dict, Optional
import csv
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# Import our optimized modules
from qNN import QuantumNeuralNetwork, QNNConfig, QNNTrainer
from data.preprocess import preprocess_mnist


def set_qfl_seeds(seed: int = 42):
    # Set all random seeds for reproducible quantum federated learning.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_qfl_seeds(42)

# Federated Averaging for Quantum Models

def quantum_federated_averaging(client_updates: List[Tuple[Dict, int, float]], global_params_template: Dict[str, torch.Tensor],
                                device: str = 'cpu') -> Tuple[Dict[str, torch.Tensor], float]:
    """
    FedAvg aggregation adapted for quantum-classical hybrid models.
    
    Handles both quantum variational parameters and classical neural network
    parameters with proper weighted averaging based on client data sizes.
    
    Args:
        client_updates: List of (state_dict, num_samples, loss) tuples
        global_params_template: Template state dict from global model
        device: Device for aggregation computation
    
    Returns:
        (aggregated_params, weighted_avg_loss)
    """
    
    if not client_updates:
        raise ValueError("No client updates to aggregate")
    
    total_samples = sum(n for _, n, _ in client_updates)
    if total_samples == 0:
        raise ValueError("Total samples is zero")
    
    # Initialize aggregated parameters
    aggregated = {}
    for key, tensor in global_params_template.items():
        if tensor.dtype == torch.long or 'num_batches_tracked' in key:
            aggregated[key] = tensor.clone().to(device)
        else:
            aggregated[key] = torch.zeros_like(tensor, device=device)
    
    # Weighted averaging
    weighted_loss = 0.0
    for params, n_samples, loss in client_updates:
        weight = n_samples / total_samples
        weighted_loss += weight * loss
        
        for key in aggregated.keys():
            if aggregated[key].dtype != torch.long and 'num_batches_tracked' not in key:
                aggregated[key] += params[key].to(device) * weight
    
    return aggregated, weighted_loss

# Quantum Federated Learning Orchestrator

class QuantumFederatedLearning:
    """
    Orchestrator for Quantum Federated Learning with PennyLane QNNs.
    
    Manages the complete federated learning workflow:
    - Global model initialization
    - Client selection and local training
    - Federated parameter aggregation
    - Global evaluation and metrics tracking
    """
    
    def __init__(
        self,
        config: QNNConfig,
        fl_config: Dict,
        device: Optional[str] = None
    ):
        self.config = config
        self.fl_config = fl_config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize global quantum model
        self.global_model = QuantumNeuralNetwork(config).to(self.device)
        
        # Metrics tracking
        self.test_accuracies = []
        self.test_losses = []
        self.train_losses = []
        self.client_losses_history = []
        
        # Print initialization info
        self._print_initialization_info()
    
    def _print_initialization_info(self):
        # Print comprehensive initialization information.
        print("\nQUANTUM FEDERATED LEARNING SYSTEM")
        print(f"Device: {self.device}")
        print(f"\nQuantum Configuration:")
        print(f"  Qubits: {self.config.n_qubits}")
        print(f"  Layers: {self.config.n_layers}")
        print(f"  Encoding: {self.config.encoding}")
        print(f"  Entanglement: {self.config.entanglement}")
        
        print(f"\nFederated Learning Configuration:")
        for key, val in self.fl_config.items():
            print(f"  {key}: {val}")
        
        # Parameter counts
        total_params = sum(p.numel() for p in self.global_model.parameters())
        q_params = sum(p.numel() for p in self.global_model.get_quantum_params())
        c_params = sum(p.numel() for p in self.global_model.get_classical_params())
        
        print(f"\nModel Architecture:")
        print(f"  Total parameters: {total_params}")
        print(f"  Quantum parameters: {q_params} ({100*q_params/total_params:.1f}%)")
        print(f"  Classical parameters: {c_params} ({100*c_params/total_params:.1f}%)\n")
    
    def train_local_client(
        self,
        client_data: Tuple[torch.Tensor, torch.Tensor],
        client_id: int
    ) -> Tuple[Dict, int, float]:
        """
        Train a local quantum model on client data using QNNTrainer.
        
        Args:
            client_data: (X, y) tuple for this client
            client_id: Client identifier for logging
        
        Returns:
            (updated_state_dict, num_samples, avg_loss)
        """
        X_client, y_client = client_data
        
        # Ensure proper types
        X_client = torch.as_tensor(X_client, dtype=torch.float32)
        y_client = torch.as_tensor(y_client, dtype=torch.long)
        
        # Create local model (copy from global)
        local_model = QuantumNeuralNetwork(self.config).to(self.device)
        local_model.load_state_dict(self.global_model.state_dict())
        
        # Create data loader
        train_loader = DataLoader(
            TensorDataset(X_client, y_client),
            batch_size=self.fl_config.get('batch_size', 16),
            shuffle=True,
            num_workers=0,
            pin_memory=(self.device == 'cuda')
        )
        
        # Create temporary config for local training
        local_config = QNNConfig(
            n_qubits=self.config.n_qubits,
            n_features=self.config.n_features,
            n_classes=self.config.n_classes,
            encoding=self.config.encoding,
            n_layers=self.config.n_layers,
            n_readout=self.config.n_readout,
            entanglement=self.config.entanglement,
            batch_size=self.fl_config.get('batch_size', 16),
            epochs=self.fl_config.get('local_epochs', 3),
            classical_lr=self.fl_config.get('classical_lr', 1e-3),
            quantum_lr=self.fl_config.get('quantum_lr', 5e-4),
            weight_decay=self.fl_config.get('weight_decay', 1e-4),
            grad_clip=self.fl_config.get('grad_clip', 1.0),
            device_name=self.config.device_name,
            diff_method=self.config.diff_method,
            shots=self.config.shots
        )
        
        # Use QNNTrainer for local training
        try:
            trainer = QNNTrainer(local_model, local_config, self.device)
            
            # Train for local epochs (no validation during FL)
            for epoch in range(local_config.epochs):
                epoch_loss, _ = trainer.train_epoch(train_loader)
            
            # Get average training loss from last epoch
            avg_loss = trainer.train_losses[-1] if trainer.train_losses else float('inf')
            
            return local_model.state_dict(), len(X_client), avg_loss
        
        except Exception as e:
            print(f"  Warning: Client {client_id} training failed: {e}")
            # Return global model state as fallback
            return self.global_model.state_dict(), len(X_client), float('inf')
    
    def federated_round(self, client_data: List[Tuple[torch.Tensor, torch.Tensor]],
                        round_num: int) -> float:
        """
        Execute one round of federated learning.
        
        Steps:
        1. Sample clients
        2. Distribute global model
        3. Local training
        4. Aggregate updates
        5. Update global model
        """
        num_clients = len(client_data)
        client_fraction = self.fl_config.get('client_fraction', 1.0)
        
        # Client sampling
        num_selected = max(1, int(client_fraction * num_clients))
        selected_clients = random.sample(range(num_clients), num_selected)
        
        print(f"\nRound {round_num}: Selected {num_selected}/{num_clients} clients {selected_clients}")
        
        # Local training on selected clients
        client_updates = []
        round_losses = []
        
        for client_id in selected_clients:
            state_dict, num_samples, train_loss = self.train_local_client(
                client_data[client_id],
                client_id
            )
            
            client_updates.append((state_dict, num_samples, train_loss))
            round_losses.append(train_loss)
            
            print(f"  Client {client_id}: Loss = {train_loss:.4f}, Samples = {num_samples}")
        
        # Store client losses
        self.client_losses_history.append(round_losses)
        
        # Federated aggregation
        if not client_updates:
            print("  ⚠️  No successful client updates")
            return float('inf')
        
        try:
            aggregated_params, avg_train_loss = quantum_federated_averaging(
                client_updates,
                self.global_model.state_dict(),
                self.device
            )
            self.global_model.load_state_dict(aggregated_params)
            
            print(f"  ✓ Aggregated train loss: {avg_train_loss:.4f}")
            return avg_train_loss
        
        except Exception as e:
            print(f"  ⚠️  Aggregation failed: {e}")
            return float('inf')
    
    @torch.no_grad()
    def evaluate_global(self, test_data: Tuple[torch.Tensor, torch.Tensor]
                        ) -> Tuple[float, float]:
        """
        Evaluate global model on test set.
        
        Returns:
            (accuracy, loss)
        """
        self.global_model.eval()
        
        X_test, y_test = test_data
        X_test = torch.as_tensor(X_test, dtype=torch.float32)
        y_test = torch.as_tensor(y_test, dtype=torch.long)
        
        test_loader = DataLoader(
            TensorDataset(X_test, y_test),
            batch_size=self.fl_config.get('batch_size', 16) * 2,
            shuffle=False,
            num_workers=0,
            pin_memory=(self.device == 'cuda')
        )
        
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for data, target in test_loader:
            data, target = data.to(self.device), target.to(self.device)
            
            try:
                outputs = self.global_model(data)
                loss = criterion(outputs, target)
                
                total_loss += loss.item() * data.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == target).sum().item()
                total += target.size(0)
            
            except Exception as e:
                print(f"  ⚠️  Evaluation error: {e}")
                continue
        
        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / total if total > 0 else float('inf')
        
        return accuracy, avg_loss
    
    def train(self, client_data: List[Tuple[torch.Tensor, torch.Tensor]],
                test_data: Tuple[torch.Tensor, torch.Tensor]) -> Dict:
        """
        Main federated training loop.
        
        Returns:
            Dictionary with complete training history
        """
        num_rounds = self.fl_config.get('num_rounds', 1) # 20 for real training
        
        print("Starting Quantum Federated Learning Training")
        
        # Initial evaluation
        print("\nRound 0: Initial Evaluation")
        initial_acc, initial_loss = self.evaluate_global(test_data)
        self.test_accuracies.append(initial_acc)
        self.test_losses.append(initial_loss)
        self.train_losses.append(0.0)
        
        print(f"  Test Accuracy: {initial_acc:.4f}")
        print(f"  Test Loss: {initial_loss:.4f}")
        
        # Federated training rounds
        for round_num in range(1, num_rounds + 1):
            # Execute round
            avg_train_loss = self.federated_round(client_data, round_num)
            
            # Global evaluation
            test_acc, test_loss = self.evaluate_global(test_data)
            
            # Store metrics
            self.test_accuracies.append(test_acc)
            self.test_losses.append(test_loss)
            self.train_losses.append(avg_train_loss)
            
            # Progress logging
            if round_num % 5 == 0 or round_num == num_rounds:
                print(f"\n{'='*70}")
                print(f"Round {round_num} Summary:")
                print(f"  Test Accuracy: {test_acc:.4f}")
                print(f"  Test Loss: {test_loss:.4f}")
                print(f"  Train Loss: {avg_train_loss:.4f}")
                print(f"{'='*70}")
        
        # Final summary
        self._print_final_summary()
        
        return {
            'model': self.global_model,
            'test_accuracies': self.test_accuracies,
            'test_losses': self.test_losses,
            'train_losses': self.train_losses,
            'client_losses': self.client_losses_history,
            'final_accuracy': self.test_accuracies[-1],
            'best_accuracy': max(self.test_accuracies[1:]) if len(self.test_accuracies) > 1 else 0.0
        }
    
    def _print_final_summary(self):
        # Print final training summary.
        print("\nTRAINING COMPLETE")
        
        final_acc = self.test_accuracies[-1]
        best_acc = max(self.test_accuracies[1:]) if len(self.test_accuracies) > 1 else final_acc
        final_loss = self.test_losses[-1]
        
        print(f"Final Test Accuracy: {final_acc:.4f}")
        print(f"Best Test Accuracy:  {best_acc:.4f}")
        print(f"Final Test Loss:     {final_loss:.4f}\n")
    
    def save_results(self, save_dir: str = "./artifacts"):
        """Save model checkpoint and metrics."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save checkpoint
        checkpoint = {
            'model_state_dict': self.global_model.state_dict(),
            'config': self.config.to_dict(),
            'fl_config': self.fl_config,
            'test_accuracies': self.test_accuracies,
            'test_losses': self.test_losses,
            'train_losses': self.train_losses,
            'client_losses': self.client_losses_history,
            'best_accuracy': max(self.test_accuracies[1:]) if len(self.test_accuracies) > 1 else 0.0
        }
        
        model_path = Path(save_dir) / "quantum_federated_model.pt"
        torch.save(checkpoint, model_path)
        print(f"💾 Model saved: {model_path}")
        
        # Save metrics CSV
        csv_path = Path(save_dir) / "qfl_metrics.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Round", "Test_Accuracy", "Test_Loss", "Train_Loss"])
            
            for i in range(len(self.test_accuracies)):
                writer.writerow([
                    i,
                    f"{self.test_accuracies[i]:.6f}",
                    f"{self.test_losses[i]:.6f}",
                    f"{self.train_losses[i]:.6f}"
                ])
        
        print(f"📊 Metrics saved: {csv_path}")
    
    def plot_results(self, save_dir: str = "./artifacts"):
        """Generate comprehensive training visualization."""
        os.makedirs(save_dir, exist_ok=True)
        
        fig = plt.figure(figsize=(18, 5))
        
        # Plot 1: Test Accuracy
        ax1 = plt.subplot(1, 3, 1)
        rounds = range(len(self.test_accuracies))
        ax1.plot(rounds, self.test_accuracies, 'o-', color='#2ecc71', 
                linewidth=2, markersize=6, label='Test Accuracy')
        
        if len(self.test_accuracies) > 1:
            best_acc = max(self.test_accuracies[1:])
            ax1.axhline(y=best_acc, color='r', linestyle='--', 
                       alpha=0.5, label=f'Best: {best_acc:.4f}')
        
        ax1.set_xlabel("Round", fontsize=12)
        ax1.set_ylabel("Accuracy", fontsize=12)
        ax1.set_title("Test Accuracy", fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim([0, 1.05])
        
        # Plot 2: Loss Curves
        ax2 = plt.subplot(1, 3, 2)
        ax2.plot(rounds, self.test_losses, 's-', color='#e74c3c', 
                linewidth=2, markersize=6, label='Test Loss')
        ax2.plot(rounds, self.train_losses, '^-', color='#3498db', 
                linewidth=2, markersize=6, label='Train Loss')
        ax2.set_xlabel("Round", fontsize=12)
        ax2.set_ylabel("Loss", fontsize=12)
        ax2.set_title("Loss Curves", fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot 3: Client Loss Distribution
        ax3 = plt.subplot(1, 3, 3)
        if self.client_losses_history:
            valid_losses = [losses for losses in self.client_losses_history 
                          if losses and all(l != float('inf') for l in losses)]
            
            if valid_losses:
                bp = ax3.boxplot(valid_losses, patch_artist=True)
                for patch in bp['boxes']:
                    patch.set_facecolor('#9b59b6')
                    patch.set_alpha(0.6)
                
                ax3.set_xlabel("Round", fontsize=12)
                ax3.set_ylabel("Client Loss", fontsize=12)
                ax3.set_title("Client Loss Distribution", fontsize=14, fontweight='bold')
                ax3.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plot_path = Path(save_dir) / "qfl_results.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Results plot saved: {plot_path}")

# Main Execution

def main():
    """Main entry point for Quantum Federated Learning."""
    
    # Quantum model configuration
    qnn_config = QNNConfig(
        n_qubits=4,
        n_features=4,  # Will be updated from data
        n_classes=3,
        encoding='amplitude',  # 'amplitude' or 'angle'
        n_layers=2,
        entanglement='circular',
        batch_size=16,
        classical_lr=1e-3,
        quantum_lr=5e-4,
        grad_clip=1.0
    )
    
    # Federated learning configuration
    fl_config = {
        'num_rounds': 1,  # 20 for real training
        'local_epochs': 3,
        'batch_size': 16,
        'classical_lr': 1e-3,
        'quantum_lr': 5e-4,
        'client_fraction': 0.75,
        'grad_clip': 1.0
    }
    
    # Data configuration
    data_config = {
        'raw_folder': "./dataset/raw",
        'processed_folder': "./dataset/processed",
        'digits': (0, 1, 2),
        'val_split': 0.1,
        'num_clients': 4,
        'partition_type': 'iid',  # 'iid' or 'non_iid'
        'alpha': 0.5,
        'apply_pca': True,
        'pca_components': qnn_config.n_features
    }
    
    print("\nQUANTUM FEDERATED LEARNING WITH PENNYLANE")
    
    # Load data
    try:
        result = preprocess_mnist(**data_config, generate_plots=False)
        
        if result is None:
            raise ValueError("Preprocessing returned None")
        
        train_data, val_data, test_data, client_data = result
        
        # Update feature dimension
        qnn_config.n_features = client_data[0][0].shape[1]
        
        print(f"\n Data loaded successfully:")
        print(f"\n  Clients: {len(client_data)}")
        print(f"  Features: {qnn_config.n_features}")
        print(f"\n  Test samples: {len(test_data[0])}")
        
        for i, (X, y) in enumerate(client_data):
            y_np = y.numpy() if isinstance(y, torch.Tensor) else y
            print(f"  Client {i}: {len(X)} samples, "
                  f"classes {np.bincount(y_np)}")
    
    except Exception as e:
        print(f"\n⚠️  Data loading failed: {e}")
        print("Using synthetic data for demonstration\n")
        
        # Synthetic data
        np.random.seed(42)
        n_clients = 4
        n_samples = 250
        
        client_data = []
        for _ in range(n_clients):
            X = torch.randn(n_samples, qnn_config.n_features)
            y = torch.randint(0, qnn_config.n_classes, (n_samples,))
            client_data.append((X, y))
        
        X_test = torch.randn(200, qnn_config.n_features)
        y_test = torch.randint(0, qnn_config.n_classes, (200,))
        test_data = (X_test, y_test)
    
    # Initialize QFL
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    qfl = QuantumFederatedLearning(qnn_config, fl_config, device)
    
    # Train
    try:
        results = qfl.train(client_data, test_data)
        
        # Save results
        qfl.save_results()
        qfl.plot_results()
        
        print("\nQUANTUM FEDERATED LEARNING COMPLETED SUCCESSFULLY\n")
        
        return results

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()