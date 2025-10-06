import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from typing import List, Tuple, Dict
import csv
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# Import your existing modules
from deprecated.hybridqnn import EnhancedQNN, train_enhanced_qnn, evaluate_model as evaluate_qnn_model
from deprecated.classical_fl import federated_averaging, set_seeds, evaluate_model as evaluate_classical_model
from data.preprocess import preprocess_mnist

def set_qfl_seeds(seed=42):
    """Set seeds for reproducible quantum federated learning"""
    set_seeds(seed)  # Use the existing seed function from classical FL

set_qfl_seeds(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class QuantumFederatedLearning:
    """
    Quantum Federated Learning orchestrator that combines:
    - EnhancedQNN from your quantum module
    - Federated averaging from your classical FL module
    - Custom quantum client training logic
    """
    
    def __init__(self, qnn_config: Dict, fl_config: Dict):
        self.qnn_config = qnn_config
        self.fl_config = fl_config
        self.device = device
        
        # Initialize global quantum model
        self.global_model = EnhancedQNN(**qnn_config).to(device)
        
        # Metrics tracking
        self.test_accuracies = []
        self.test_losses = []
        self.train_losses = []
        
        print(f"Quantum Federated Learning Orchestrator Initialized:")
        print(f"- Quantum Model: {qnn_config}")
        print(f"- FL Config: {fl_config}")
        print(f"- Device: {device}")
        print("-" * 60)
    
    def quantum_client_update(self, client_data: Tuple[torch.Tensor, torch.Tensor], 
                            client_id: int) -> Tuple[Dict, int, float]:
        """
        Quantum client update using the existing EnhancedQNN training logic
        Adapts the train_enhanced_qnn function for federated learning
        """
        X_client, y_client = client_data
        
        # Convert to proper format
        if not isinstance(X_client, torch.Tensor):
            X_client = torch.tensor(X_client, dtype=torch.float32)
        if not isinstance(y_client, torch.Tensor):
            y_client = torch.tensor(y_client, dtype=torch.long)
            
        # Create local model copy
        local_model = EnhancedQNN(**self.qnn_config).to(device)
        local_model.load_state_dict(self.global_model.state_dict())
        
        # Create data loader
        dataset = TensorDataset(X_client, y_client)
        train_loader = DataLoader(
            dataset, 
            batch_size=self.fl_config.get('batch_size', 16), 
            shuffle=True
        )
        
        # Train using adapted quantum training function
        try:
            train_losses, _ = self._train_quantum_client(
                model=local_model,
                train_loader=train_loader,
                epochs=self.fl_config.get('local_epochs', 3),
                classical_lr=self.fl_config.get('classical_lr', 1e-3),
                quantum_lr=self.fl_config.get('quantum_lr', 5e-4),
                client_id=client_id
            )
            
            avg_loss = np.mean(train_losses) if train_losses else float('inf')
            return local_model.state_dict(), len(X_client), avg_loss
            
        except Exception as e:
            print(f"Error training quantum client {client_id}: {e}")
            return self.global_model.state_dict(), len(X_client), float('inf')
    
    def _train_quantum_client(self, model, train_loader, epochs, classical_lr, quantum_lr, client_id):
        """
        Adapted training function for quantum clients in federated setting
        Based on your train_enhanced_qnn function but simplified for FL
        """
        model.train()
        criterion = nn.CrossEntropyLoss()
        
        # Separate optimizers for classical and quantum parameters
        classical_params = list(model.pre_fc.parameters()) + list(model.classical_head.parameters())
        quantum_params = [model.q_params]
        
        optimizer = optim.Adam([
            {"params": classical_params, "lr": classical_lr},
            {"params": quantum_params, "lr": quantum_lr}
        ])
        
        train_losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                
                try:
                    outputs = model(data)
                    loss = criterion(outputs, target)
                    
                    if torch.isnan(loss):
                        print(f"Warning: NaN loss in client {client_id}, batch {batch_idx}")
                        continue
                    
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    print(f"Warning: Error in client {client_id}, batch {batch_idx}: {e}")
                    continue
            
            if num_batches > 0:
                train_losses.append(epoch_loss / num_batches)
            else:
                train_losses.append(float('inf'))
        
        return train_losses, None
    
    def federated_round(self, client_data: List[Tuple[torch.Tensor, torch.Tensor]], 
                       round_num: int) -> float:
        """
        Execute one round of quantum federated learning
        """
        num_clients = len(client_data)
        client_fraction = self.fl_config.get('client_fraction', 1.0)
        
        # Sample clients
        num_selected = max(1, int(client_fraction * num_clients))
        selected_clients = random.sample(range(num_clients), num_selected)
        
        print(f"Round {round_num}: Selected {num_selected}/{num_clients} quantum clients: {selected_clients}")
        
        # Collect client updates
        client_updates = []
        for client_id in selected_clients:
            params, num_samples, training_loss = self.quantum_client_update(
                client_data[client_id], client_id
            )
            client_updates.append((params, num_samples, training_loss))
        
        # Federated averaging using existing function
        if client_updates:
            try:
                aggregated_params, avg_train_loss = federated_averaging(
                    client_updates, device, self.global_model.state_dict()
                )
                self.global_model.load_state_dict(aggregated_params)
                return avg_train_loss
            except Exception as e:
                print(f"Error in federated averaging for round {round_num}: {e}")
                return float('inf')
        else:
            print(f"No successful client updates in round {round_num}")
            return float('inf')
    
    def evaluate_global_model(self, test_data: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[float, float]:
        """
        Evaluate global quantum model using existing evaluation function
        """
        X_test, y_test = test_data
        
        # Convert to proper format if needed
        if not isinstance(X_test, torch.Tensor):
            X_test = torch.tensor(X_test, dtype=torch.float32)
        if not isinstance(y_test, torch.Tensor):
            y_test = torch.tensor(y_test, dtype=torch.long)
        
        # Create test data loader
        test_dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        # Use existing evaluation function adapted for quantum model
        return evaluate_qnn_model(self.global_model, test_loader, device)
    
    def train(self, client_data: List[Tuple[torch.Tensor, torch.Tensor]], 
             test_data: Tuple[torch.Tensor, torch.Tensor]) -> Dict:
        """
        Main training loop for Quantum Federated Learning
        """
        num_rounds = self.fl_config.get('num_rounds', 20)
        
        # Initial evaluation
        initial_acc, initial_loss = self.evaluate_global_model(test_data)
        self.test_accuracies.append(initial_acc)
        self.test_losses.append(initial_loss)
        self.train_losses.append(0.0)
        
        print(f"Round 0: Test Accuracy = {initial_acc:.4f}, Test Loss = {initial_loss:.4f}")
        
        # Training rounds
        for round_num in range(1, num_rounds + 1):
            # Execute federated round
            avg_train_loss = self.federated_round(client_data, round_num)
            
            # Evaluate global model
            test_acc, test_loss = self.evaluate_global_model(test_data)
            
            # Store metrics
            self.test_accuracies.append(test_acc)
            self.test_losses.append(test_loss)
            self.train_losses.append(avg_train_loss)
            
            # Print progress
            if round_num % 5 == 0 or round_num == num_rounds:
                print(f"Round {round_num}: Test Acc = {test_acc:.4f}, Test Loss = {test_loss:.4f}, Train Loss = {avg_train_loss:.4f}")
        
        # Final results
        final_acc = self.test_accuracies[-1]
        final_loss = self.test_losses[-1]
        best_acc = max(self.test_accuracies[1:]) if len(self.test_accuracies) > 1 else final_acc
        
        print(f"\nQuantum Federated Learning Results:")
        print(f"- Final Test Accuracy: {final_acc:.4f}")
        print(f"- Final Test Loss: {final_loss:.4f}")
        print(f"- Best Test Accuracy: {best_acc:.4f}")
        
        return {
            'model': self.global_model,
            'test_accuracies': self.test_accuracies,
            'test_losses': self.test_losses,
            'train_losses': self.train_losses
        }
    
    def save_results(self, save_dir="artifacts"):
        """Save model and metrics"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model
        torch.save({
            'model_state_dict': self.global_model.state_dict(),
            'qnn_config': self.qnn_config,
            'fl_config': self.fl_config,
            'test_accuracies': self.test_accuracies,
            'test_losses': self.test_losses,
            'train_losses': self.train_losses
        }, f"{save_dir}/quantum_federated_model.pt")
        
        # Save metrics CSV
        with open(f"{save_dir}/qfl_metrics.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Round", "Test_Accuracy", "Test_Loss", "Train_Loss"])
            for i, (acc, test_loss, train_loss) in enumerate(zip(
                self.test_accuracies, self.test_losses, self.train_losses
            )):
                writer.writerow([i, acc, test_loss, train_loss])
    
    def plot_results(self, save_dir="artifacts"):
        """Plot and save training results"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy plot
        ax1.plot(range(len(self.test_accuracies)), self.test_accuracies, 
                marker='o', label='Test Accuracy', color='green', linewidth=2)
        ax1.set_xlabel("Round")
        ax1.set_ylabel("Accuracy")
        ax1.set_title("Quantum Federated Learning - Test Accuracy")
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Loss plot
        ax2.plot(range(len(self.test_losses)), self.test_losses, 
                marker='s', label='Test Loss', color='red', linewidth=2)
        ax2.plot(range(len(self.train_losses)), self.train_losses, 
                marker='^', label='Train Loss', color='blue', linewidth=2)
        ax2.set_xlabel("Round")
        ax2.set_ylabel("Loss")
        ax2.set_title("Quantum Federated Learning - Loss Curves")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/qfl_results.png", dpi=150, bbox_inches='tight')
        print(f"Results plot saved to {save_dir}/qfl_results.png")
        plt.show()


def main():
    """
    Main function demonstrating Quantum Federated Learning
    """
    # Configuration
    qnn_config = {
        'n_qubits': 4,
        'n_features': 16,  # Will be updated based on PCA
        'n_classes': 3,
        'encoding': 'amplitude',
        'n_layers': 2,
        'entanglement': 'circular',
        'shots': None
    }
    
    fl_config = {
        'num_rounds': 15,
        'local_epochs': 3,
        'batch_size': 16,
        'classical_lr': 1e-3,
        'quantum_lr': 5e-4,
        'client_fraction': 0.75
    }
    
    data_config = {
        'raw_folder': "./dataset/raw",
        'processed_folder': "./dataset/processed",
        'digits': (0, 1, 2),
        'val_split': 0.1,
        'num_clients': 4,
        'partition_type': 'iid',
        'alpha': 0.5
    }
    
    print("🚀 Quantum Federated Learning with Modular Architecture")
    print("=" * 70)
    print(f"QNN Config: {qnn_config}")
    print(f"FL Config: {fl_config}")
    print(f"Data Config: {data_config}")
    print()
    
    # Data preprocessing
    try:
        result = preprocess_mnist(
            raw_folder=data_config['raw_folder'],
            processed_folder=data_config['processed_folder'],
            digits=data_config['digits'],
            val_split=data_config['val_split'],
            num_clients=data_config['num_clients'],
            partition_type=data_config['partition_type'],
            alpha=data_config['alpha'],
            apply_pca=True,
            pca_components=qnn_config['n_features']
        )
        
        if result is None:
            raise ValueError("Data preprocessing returned None")
        
        train_data, val_data, test_data, client_data = result
        
        # Update feature size based on actual data
        actual_features = client_data[0][0].shape[1]
        qnn_config['n_features'] = actual_features
        print(f"✅ Data loaded successfully. Updated n_features to {actual_features}")
        
    except Exception as e:
        print(f"⚠️  Data preprocessing failed: {e}")
        print("🔄 Using synthetic data for demonstration...")
        
        # Generate synthetic data
        np.random.seed(42)
        n_samples_per_client = 200
        n_features = qnn_config['n_features']
        n_classes = qnn_config['n_classes']
        
        client_data = []
        for i in range(data_config['num_clients']):
            X = torch.randn(n_samples_per_client, n_features) * 0.5
            y = torch.randint(0, n_classes, (n_samples_per_client,))
            client_data.append((X, y))
        
        # Test data
        X_test = torch.randn(150, n_features) * 0.5
        y_test = torch.randint(0, n_classes, (150,))
        test_data = (X_test, y_test)
        
        print(f"✅ Synthetic data generated: {len(client_data)} clients, {n_samples_per_client} samples each")
    
    # Initialize Quantum Federated Learning
    qfl = QuantumFederatedLearning(qnn_config, fl_config)
    
    # Train
    try:
        results = qfl.train(client_data, test_data)
        
        # Save and plot results
        qfl.save_results()
        qfl.plot_results()
        
        print("\n🎉 Quantum Federated Learning completed successfully!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()