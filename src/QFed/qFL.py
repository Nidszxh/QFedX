# Quantum Federated Learning (QFL) Implementation

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

from qNN import QuantumNeuralNetwork, QNNConfig, QNNTrainer
from data.preprocess import preprocess_mnist


def set_qfl_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_qfl_seeds(42)


def quantum_federated_averaging(client_updates: List[Tuple[Dict, int, float]], 
                                global_params_template: Dict[str, torch.Tensor],
                                device: str = 'cpu') -> Tuple[Dict[str, torch.Tensor], float]:
    """FedAvg aggregation adapted for quantum-classical hybrid models."""
    if not client_updates:
        raise ValueError("No client updates to aggregate")
    
    total_samples = sum(n for _, n, _ in client_updates)
    if total_samples == 0:
        raise ValueError("Total samples is zero")
    
    aggregated = {}
    for key, tensor in global_params_template.items():
        if tensor.dtype == torch.long or 'num_batches_tracked' in key:
            aggregated[key] = tensor.clone().to(device)
        else:
            aggregated[key] = torch.zeros_like(tensor, device=device)
    
    weighted_loss = 0.0
    for params, n_samples, loss in client_updates:
        weight = n_samples / total_samples
        weighted_loss += weight * loss
        for key in aggregated.keys():
            if aggregated[key].dtype != torch.long and 'num_batches_tracked' not in key:
                aggregated[key] += params[key].to(device) * weight
    
    return aggregated, weighted_loss


class QuantumFederatedLearning:
    """Orchestrator for Quantum Federated Learning with PennyLane QNNs."""
    
    def __init__(self, config: QNNConfig, fl_config: Dict, device: Optional[str] = None):
        self.config = config
        self.fl_config = fl_config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.global_model = QuantumNeuralNetwork(config).to(self.device)
        self.test_accuracies = []
        self.test_losses = []
        self.train_losses = []
        self.client_losses_history = []
        self._print_initialization_info()
    
    def _print_initialization_info(self):
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
        total_params = sum(p.numel() for p in self.global_model.parameters())
        q_params = sum(p.numel() for p in self.global_model.get_quantum_params())
        c_params = sum(p.numel() for p in self.global_model.get_classical_params())
        print(f"\nModel Architecture:")
        print(f"  Total parameters: {total_params}")
        print(f"  Quantum parameters: {q_params} ({100*q_params/total_params:.1f}%)")
        print(f"  Classical parameters: {c_params} ({100*c_params/total_params:.1f}%)\n")
    
    def train_local_client(self, client_data: Tuple[torch.Tensor, torch.Tensor],
                          client_id: int) -> Tuple[Dict, int, float]:
        X_client, y_client = client_data
        X_client = torch.as_tensor(X_client, dtype=torch.float32)
        y_client = torch.as_tensor(y_client, dtype=torch.long)
        
        local_model = QuantumNeuralNetwork(self.config).to(self.device)
        local_model.load_state_dict(self.global_model.state_dict())
        
        train_loader = DataLoader(
            TensorDataset(X_client, y_client),
            batch_size=self.fl_config.get('batch_size', 16),
            shuffle=True, num_workers=0,
            pin_memory=(self.device == 'cuda')
        )
        
        local_config = QNNConfig(
            n_qubits=self.config.n_qubits, n_features=self.config.n_features,
            n_classes=self.config.n_classes, encoding=self.config.encoding,
            n_layers=self.config.n_layers, n_readout=self.config.n_readout,
            entanglement=self.config.entanglement,
            batch_size=self.fl_config.get('batch_size', 16),
            epochs=self.fl_config.get('local_epochs', 3),
            classical_lr=self.fl_config.get('classical_lr', 1e-3),
            quantum_lr=self.fl_config.get('quantum_lr', 5e-4),
            weight_decay=self.fl_config.get('weight_decay', 1e-4),
            grad_clip=self.fl_config.get('grad_clip', 1.0),
            device_name=self.config.device_name,
            diff_method=self.config.diff_method, shots=self.config.shots
        )
        
        try:
            trainer = QNNTrainer(local_model, local_config, self.device)
            for epoch in range(local_config.epochs):
                epoch_loss, _ = trainer.train_epoch(train_loader)
            avg_loss = trainer.train_losses[-1] if trainer.train_losses else float('inf')
            return local_model.state_dict(), len(X_client), avg_loss
        except Exception as e:
            print(f"  Warning: Client {client_id} training failed: {e}")
            return self.global_model.state_dict(), len(X_client), float('inf')
    
    def federated_round(self, client_data: List[Tuple[torch.Tensor, torch.Tensor]],
                        round_num: int) -> float:
        num_clients = len(client_data)
        client_fraction = self.fl_config.get('client_fraction', 1.0)
        num_selected = max(1, int(client_fraction * num_clients))
        selected_clients = random.sample(range(num_clients), num_selected)
        
        print(f"\nRound {round_num}: Selected {num_selected}/{num_clients} clients {selected_clients}")
        
        client_updates = []
        round_losses = []
        
        for client_id in selected_clients:
            state_dict, num_samples, train_loss = self.train_local_client(
                client_data[client_id], client_id
            )
            client_updates.append((state_dict, num_samples, train_loss))
            round_losses.append(train_loss)
            print(f"  Client {client_id}: Loss = {train_loss:.4f}, Samples = {num_samples}")
        
        self.client_losses_history.append(round_losses)
        
        if not client_updates:
            print("  No successful client updates")
            return float('inf')
        
        try:
            aggregated_params, avg_train_loss = quantum_federated_averaging(
                client_updates, self.global_model.state_dict(), self.device
            )
            self.global_model.load_state_dict(aggregated_params)
            print(f"  ✓ Aggregated train loss: {avg_train_loss:.4f}")
            return avg_train_loss
        except Exception as e:
            print(f"  Aggregation failed: {e}")
            return float('inf')
    
    @torch.no_grad()
    def evaluate_global(self, test_data: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[float, float]:
        self.global_model.eval()
        X_test, y_test = test_data
        X_test = torch.as_tensor(X_test, dtype=torch.float32)
        y_test = torch.as_tensor(y_test, dtype=torch.long)
        
        test_loader = DataLoader(
            TensorDataset(X_test, y_test),
            batch_size=self.fl_config.get('batch_size', 16) * 2,
            shuffle=False, num_workers=0,
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
                print(f"  Evaluation error: {e}")
                continue
        
        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / total if total > 0 else float('inf')
        return accuracy, avg_loss
    
    def train(self, client_data: List[Tuple[torch.Tensor, torch.Tensor]],
                test_data: Tuple[torch.Tensor, torch.Tensor]) -> Dict:
        num_rounds = self.fl_config.get('num_rounds', 5)
        
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
            avg_train_loss = self.federated_round(client_data, round_num)
            test_acc, test_loss = self.evaluate_global(test_data)
            self.test_accuracies.append(test_acc)
            self.test_losses.append(test_loss)
            self.train_losses.append(avg_train_loss)
            
            if round_num % 5 == 0 or round_num == num_rounds:
                print(f"\n{'='*70}")
                print(f"Round {round_num} Summary:")
                print(f"  Test Accuracy: {test_acc:.4f}")
                print(f"  Test Loss: {test_loss:.4f}")
                print(f"  Train Loss: {avg_train_loss:.4f}")
                print(f"{'='*70}")
        
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
        print("\nTRAINING COMPLETE")
        final_acc = self.test_accuracies[-1]
        best_acc = max(self.test_accuracies[1:]) if len(self.test_accuracies) > 1 else final_acc
        final_loss = self.test_losses[-1]
        print(f"Final Test Accuracy: {final_acc:.4f}")
        print(f"Best Test Accuracy:  {best_acc:.4f}")
        print(f"Final Test Loss:     {final_loss:.4f}\n")
    
    def save_results(self, save_dir: str = "./artifacts"):
        os.makedirs(save_dir, exist_ok=True)
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
        
        csv_path = Path(save_dir) / "qfl_metrics.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Round", "Test_Accuracy", "Test_Loss", "Train_Loss"])
            for i in range(len(self.test_accuracies)):
                writer.writerow([
                    i, f"{self.test_accuracies[i]:.6f}",
                    f"{self.test_losses[i]:.6f}",
                    f"{self.train_losses[i]:.6f}"
                ])
        print(f"📊 Metrics saved: {csv_path}")


def main():
    qnn_config = QNNConfig(
        n_qubits=4, n_features=4, n_classes=3,
        encoding='amplitude', n_layers=2, entanglement='circular',
        batch_size=16, classical_lr=1e-3, quantum_lr=5e-4, grad_clip=1.0
    )
    
    fl_config = {
        'num_rounds': 5, 'local_epochs': 3, 'batch_size': 16,
        'classical_lr': 1e-3, 'quantum_lr': 5e-4,
        'client_fraction': 0.75, 'grad_clip': 1.0
    }
    
    data_config = {
        'raw_folder': "./dataset/raw",
        'processed_folder': "./dataset/processed",
        'digits': (0, 1, 2), 'val_split': 0.1,
        'num_clients': 4, 'partition_type': 'iid',
        'alpha': 0.5, 'apply_pca': True,
        'pca_components': qnn_config.n_features
    }
    
    print("\nQUANTUM FEDERATED LEARNING WITH PENNYLANE")
    
    # Load data
    try:
        result = preprocess_mnist(**data_config, generate_plots=False)
        if result is None:
            raise ValueError("Preprocessing returned None")
        train_data, val_data, test_data, client_data = result
        qnn_config.n_features = client_data[0][0].shape[1]
        
        print(f"\n Data loaded successfully:")
        print(f"\n  Clients: {len(client_data)}")
        print(f"  Features: {qnn_config.n_features}")
        print(f"\n  Test samples: {len(test_data[0])}")
        for i, (X, y) in enumerate(client_data):
            y_np = y.numpy() if isinstance(y, torch.Tensor) else y
            print(f"  Client {i}: {len(X)} samples, classes {np.bincount(y_np)}")
    
    except Exception as e:
        print(f"\nData loading failed: {e}")
        print("Using synthetic data for demonstration\n")
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
        qfl.save_results()
        
        # Generate QFL-specific visualizations
        print("\nGenerating QFL visualizations...")
        try:
            from viz_qFL import generate_all_qfl_plots
            
            # Get predictions for confusion matrix
            qfl.global_model.eval()
            X_test, y_test = test_data
            X_test_tensor = torch.as_tensor(X_test, dtype=torch.float32).to(device)
            with torch.no_grad():
                outputs = qfl.global_model(X_test_tensor)
                _, y_pred = torch.max(outputs, 1)
            y_pred_np = y_pred.cpu().numpy()
            y_test_np = y_test.numpy() if isinstance(y_test, torch.Tensor) else y_test
            
            saved_plots = generate_all_qfl_plots(
                results=results,
                client_data=client_data,
                y_test=y_test_np,
                y_pred=y_pred_np,
                class_names=[f'Digit {i}' for i in range(qnn_config.n_classes)],
                save_dir='./visualizations/qfl'
            )
            
            print(f"\nGenerated {len(saved_plots)} QFL visualization plots:")
            for name, path in saved_plots.items():
                print(f"  - {name}: {path}")
        
        except ImportError as e:
            print(f"Visualization module not available: {e}")
        except Exception as e:
            print(f"Visualization error: {e}")
            import traceback
            traceback.print_exc()
        

        print("\nSTARTING COMPARATIVE ANALYSIS")
        
        # Check if classical FL results exist
        cfl_metrics_path = Path('./artifacts/metrics.csv')
        
        if cfl_metrics_path.exists():
            print("\n✓ Found Classical FL results, loading for comparison...")
            
            # Load classical FL metrics
            import csv
            cfl_results = {'test_accuracies': [], 'test_losses': [], 'train_losses': []}
            
            with open(cfl_metrics_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cfl_results['test_accuracies'].append(float(row['Test_Accuracy']))
                    cfl_results['test_losses'].append(float(row['Test_Loss']))
                    cfl_results['train_losses'].append(float(row['Train_Loss']))
            
            print(f"  Loaded {len(cfl_results['test_accuracies'])} rounds of Classical FL data")
            
            # Generate comparative analysis
            try:
                from viz_comparative_analysis import generate_all_comparative_plots, plot_3d_performance_surface
                
                comparative_plots = generate_all_comparative_plots(
                    qfl_results=results,
                    cfl_results=cfl_results,
                    qfl_model=qfl.global_model,
                    client_data=client_data,
                    qnn_config=qnn_config.to_dict(),
                    qfl_config=fl_config,
                    save_dir='./visualizations/comparative'
                )
                
                print(f"\n✓ Generated {len(comparative_plots)} comparative analysis plots")
                
                # Bonus: 3D surface plot
                try:
                    surface_plot = plot_3d_performance_surface(
                        results, cfl_results, './visualizations/comparative'
                    )
                    comparative_plots['3d_surface'] = surface_plot
                    print("  ✓ 3D Performance Surface")
                except Exception as e:
                    print(f"  ✗ 3D surface plot failed: {e}")
                
                # Print summary
                print("\n" + "="*70)
                print("COMPARATIVE ANALYSIS SUMMARY")
                print("="*70)
                
                final_q_acc = results['test_accuracies'][-1]
                final_c_acc = cfl_results['test_accuracies'][-1]
                improvement = (final_q_acc - final_c_acc) * 100
                
                print(f"\nFinal Accuracies:")
                print(f"  Quantum FL:   {final_q_acc:.4f}")
                print(f"  Classical FL: {final_c_acc:.4f}")
                print(f"  Improvement:  {improvement:+.2f}%")
                
                best_q = max(results['test_accuracies'][1:]) if len(results['test_accuracies']) > 1 else final_q_acc
                best_c = max(cfl_results['test_accuracies'][1:]) if len(cfl_results['test_accuracies']) > 1 else final_c_acc
                
                print(f"\nBest Accuracies:")
                print(f"  Quantum FL:   {best_q:.4f}")
                print(f"  Classical FL: {best_c:.4f}")
                print(f"  Improvement:  {(best_q - best_c)*100:+.2f}%")
                
                print(f"\nAll comparative plots saved to: ./visualizations/comparative/")
                print("="*70 + "\n")
            
            except ImportError as e:
                print(f"\n✗ Comparative visualization module not available: {e}")
                print("  Please ensure viz_comparative_analysis.py is in the same directory")
            except Exception as e:
                print(f"\n✗ Comparative analysis failed: {e}")
                import traceback
                traceback.print_exc()
        
        else:
            print("\n⚠ Classical FL results not found at './artifacts/metrics.csv'")
            print("  Run cFL.py first to generate classical results for comparison")
            print("  Comparative analysis will be skipped.")
        
        print("\nQUANTUM FEDERATED LEARNING COMPLETED SUCCESSFULLY\n")
        return results

    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()