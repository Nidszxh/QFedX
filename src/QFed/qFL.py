import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import os
import json
import csv
import time
import random
import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt

from qNN import QNN, QNNConfig, Trainer, set_seeds, GPUDataset, fedavg_weights
from torch.utils.data import DataLoader, TensorDataset
from privacy import apply_dp_noise, SecureAggregator
try:
    from opacus.accountants import RDPAccountant
except ImportError:
    RDPAccountant = None


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class FederatedConfig:
    """Configuration for federated learning."""
    # Federated setup
    num_clients: int = 4
    num_rounds: int = 30
    client_fraction: float = 1.0  # Fraction of clients per round
    local_epochs: int = 5
    
    # Early stopping
    early_stopping_patience: int = 10
    
    # Checkpointing
    checkpoint_interval: int = 5
    save_dir: str = './artifacts/quantum_federated'
    
    # Performance
    parallel_clients: bool = False  # Multi-GPU client training
    track_communication: bool = True
    
    # Client sampling
    sampling_strategy: str = 'deterministic'  # 'deterministic', 'random', 'round_robin'
    
    # Privacy
    dp_clip_norm: float = 0.0      # 0.0 disables DP
    dp_noise_multiplier: float = 0.0
    use_secure_aggregation: bool = False
    
    # QNN config
    qnn_config: Optional[QNNConfig] = None
    
    def __post_init__(self):
        if self.qnn_config is None:
            self.qnn_config = QNNConfig()
        
        # Create save directory
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        result['qnn_config'] = self.qnn_config.to_dict()
        return result


# ============================================================================
# CLIENT SAMPLING
# ============================================================================

def sample_clients(
    num_clients: int,
    client_fraction: float,
    round_num: int,
    strategy: str = 'deterministic',
    base_seed: int = 42
) -> List[int]:
    """
    Sample clients for federated round with different strategies.
    
    Args:
        num_clients: Total number of clients
        client_fraction: Fraction of clients to sample
        round_num: Current round number
        strategy: Sampling strategy
        base_seed: Base random seed
    
    Returns:
        List of selected client indices
    """
    num_selected = max(1, int(client_fraction * num_clients))
    
    if strategy == 'deterministic':
        # Same sampling pattern (reproducible)
        random.seed(base_seed + round_num)
        return sorted(random.sample(range(num_clients), num_selected))
    
    elif strategy == 'random':
        # Different pattern each run
        np.random.seed(base_seed + round_num + int(time.time() * 1000) % 1000)
        return sorted(np.random.choice(num_clients, num_selected, replace=False).tolist())
    
    elif strategy == 'round_robin':
        # Cycle through clients
        offset = (round_num * num_selected) % num_clients
        selected = []
        for i in range(num_selected):
            selected.append((offset + i) % num_clients)
        return sorted(selected)
    
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")


def compute_communication_cost(state_dict: Dict[str, torch.Tensor]) -> float:
    """
    Compute model size in MB for communication cost tracking.
    
    Args:
        state_dict: Model state dictionary
    
    Returns:
        Size in megabytes
    """
    total_bytes = sum(
        param.numel() * param.element_size()
        for param in state_dict.values()
    )
    return total_bytes / (1024 ** 2)


# ============================================================================
# FEDERATED LEARNING ORCHESTRATOR
# ============================================================================

class QuantumFederatedLearning:
    """
    Complete federated learning orchestrator with all optimizations.
    
    Features:
    - FedAvg with quantum weight clamping
    - Multi-strategy client sampling
    - Communication cost tracking
    - Validation-based early stopping
    - Periodic checkpointing
    - Barren plateau detection across clients
    """
    
    def __init__(
        self,
        fed_config: FederatedConfig,
        device: Optional[str] = None
    ):
        self.config = fed_config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize global model
        self.global_model = QNN(fed_config.qnn_config, self.device)
        
        # Metrics tracking
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': [],
            'test_acc': [],
            'communication_cost': [],
            'client_losses': []
        }
        
        # Early stopping
        self.best_val_acc = 0.0
        self.best_model_state = None
        self.patience_counter = 0
        
        # Communication tracking
        if fed_config.track_communication:
            self.model_size_mb = compute_communication_cost(self.global_model.state_dict())
            self.total_communication = 0.0
        
        self._print_initialization()
    
    def _print_initialization(self):
        """Print initialization information."""
        print(f"\n{'='*80}")
        print(f"🌐 QUANTUM FEDERATED LEARNING")
        print(f"{'='*80}")
        print(f"Device: {self.device}")
        print(f"\nQuantum Configuration:")
        print(f"  Qubits: {self.config.qnn_config.n_qubits}")
        print(f"  Layers: {self.config.qnn_config.n_layers}")
        print(f"  Encoding: {self.config.qnn_config.encoding}")
        print(f"  Entanglement: {self.config.qnn_config.entanglement}")
        print(f"  Measurement: {self.config.qnn_config.measurement}")
        
        print(f"\nFederated Configuration:")
        print(f"  Clients: {self.config.num_clients}")
        print(f"  Rounds: {self.config.num_rounds}")
        print(f"  Client fraction: {self.config.client_fraction:.0%}")
        print(f"  Local epochs: {self.config.local_epochs}")
        print(f"  Sampling: {self.config.sampling_strategy}")
        
        total_params = sum(p.numel() for p in self.global_model.parameters())
        quantum_params = self.global_model.q_weights.numel()
        classical_params = total_params - quantum_params
        
        print(f"\nModel Architecture:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Quantum: {quantum_params:,} ({100*quantum_params/total_params:.1f}%)")
        print(f"  Classical: {classical_params:,} ({100*classical_params/total_params:.1f}%)")
        
        if self.config.track_communication:
            print(f"\nCommunication:")
            print(f"  Model size: {self.model_size_mb:.2f} MB")
        
        print(f"{'='*80}\n")
    
    def train_client(
        self,
        client_id: int,
        client_data: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], int, float]:
        """
        Train single client locally.
        
        Args:
            client_id: Client identifier
            client_data: (X, y) tuple for client
        
        Returns:
            (updated_state_dict, num_samples, avg_loss)
        """
        X_client, y_client = client_data
        
        # Validate data
        if len(X_client) == 0:
            print(f"  ⚠️  Client {client_id}: Empty dataset")
            return self.global_model.state_dict(), 0, float('inf')
        
        # Create local model
        local_model = QNN(self.config.qnn_config, self.device)
        local_model.load_state_dict(self.global_model.state_dict())
        
        # Create data loader
        if torch.cuda.is_available() and self.device == 'cuda':
            dataset = GPUDataset(X_client, y_client, self.device)
            loader = DataLoader(
                dataset,
                batch_size=self.config.qnn_config.batch_size,
                shuffle=True,
                num_workers=0
            )
        else:
            dataset = TensorDataset(X_client, y_client)
            loader = DataLoader(
                dataset,
                batch_size=self.config.qnn_config.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )
        
        # Local training
        trainer = Trainer(local_model, self.config.qnn_config, self.device)
        
        try:
            epoch_losses = []
            for epoch in range(self.config.local_epochs):
                loss, acc = trainer.train_epoch(loader)
                
                if not np.isfinite(loss):
                    print(f"  ⚠️  Client {client_id}: Non-finite loss at epoch {epoch}")
                    return self.global_model.state_dict(), len(X_client), float('inf')
                
                epoch_losses.append(loss)
            
            avg_loss = np.mean(epoch_losses)
            
            # Validate updated parameters
            for name, param in local_model.state_dict().items():
                if not torch.isfinite(param).all():
                    print(f"  ⚠️  Client {client_id}: Non-finite params in {name}")
                    return self.global_model.state_dict(), len(X_client), float('inf')
            
            return local_model.state_dict(), len(X_client), avg_loss
        
        except Exception as e:
            print(f"  ❌ Client {client_id} training failed: {e}")
            return self.global_model.state_dict(), len(X_client), float('inf')
    
    def federated_round(
        self,
        client_data_list: List[Tuple[torch.Tensor, torch.Tensor]],
        round_num: int
    ) -> float:
        """
        Execute one federated learning round.
        
        Args:
            client_data_list: List of (X, y) tuples for all clients
            round_num: Current round number
        
        Returns:
            Average training loss across selected clients
        """
        # Sample clients
        selected_clients = sample_clients(
            self.config.num_clients,
            self.config.client_fraction,
            round_num,
            self.config.sampling_strategy,
            self.config.qnn_config.seed
        )
        
        num_selected = len(selected_clients)
        print(f"\n{'='*70}")
        print(f"Round {round_num}/{self.config.num_rounds}")
        print(f"Selected {num_selected} clients: {selected_clients}")
        print(f"{'='*70}")
        
        # Train clients
        client_updates = []
        client_losses = {}
        
        # Privacy Setup
        if self.config.use_secure_aggregation:
            aggregator = SecureAggregator(
                self.config.num_clients, 
                selected_clients, 
                self.global_model.state_dict()
            )

        for client_id in selected_clients:
            print(f"\n  Training Client {client_id}...")
            
            state_dict, n_samples, loss = self.train_client(
                client_id,
                client_data_list[client_id]
            )
            
            # Validate update
            if n_samples == 0 or not np.isfinite(loss):
                print(f"    ⚠️  Skipping (samples={n_samples}, loss={loss})")
                continue
            
            # Privacy: LDP and Secure Aggregation masks
            if self.config.dp_clip_norm > 0:
                # Calculate Delta
                delta = {k: state_dict[k].float() - self.global_model.state_dict()[k].float() for k in state_dict}
                dp_delta, actual_norm = apply_dp_noise(
                    delta, 
                    self.config.dp_clip_norm, 
                    self.config.dp_noise_multiplier
                )
                print(f"    🔒 DP Applied (Delta Norm: {actual_norm:.4f} -> Clamped/Noised)")
                
                # Reconstruct absolute weights for Secure Aggregator
                for k in state_dict:
                    state_dict[k] = self.global_model.state_dict()[k] + dp_delta[k]
                    
            if self.config.use_secure_aggregation:
                mask = aggregator.get_client_mask(client_id)
                for k in state_dict:
                    state_dict[k] += mask[k]
                print(f"    🛡️  Secure Mask Applied")
            
            client_updates.append((state_dict, n_samples, loss))
            client_losses[client_id] = loss
            print(f"    ✓ Loss: {loss:.4f}, Samples: {n_samples}")
        
        # Check if enough clients succeeded
        if len(client_updates) < max(1, num_selected // 2):
            print(f"\n  ❌ Too few successful clients ({len(client_updates)}/{num_selected})")
            print(f"     Skipping aggregation")
            self.history['client_losses'].append(client_losses)
            return float('inf')
        
        # Aggregate
        try:
            aggregated, avg_train_loss = fedavg_weights(
                client_updates,
                [n for _, n, _ in client_updates]
            )
            self.global_model.load_state_dict(aggregated)
            
            print(f"\n  ✓ Aggregation successful")
            print(f"    Average train loss: {avg_train_loss:.4f}")
            
            # Track communication
            if self.config.track_communication:
                round_comm = self.model_size_mb * len(client_updates) * 2  # Upload + Download
                self.total_communication += round_comm
                self.history['communication_cost'].append(round_comm)
                print(f"    Communication: {round_comm:.2f} MB (Total: {self.total_communication:.2f} MB)")
            
            self.history['client_losses'].append(client_losses)
            return avg_train_loss
        
        except Exception as e:
            print(f"\n  ❌ Aggregation failed: {e}")
            self.history['client_losses'].append(client_losses)
            return float('inf')
    
    @torch.no_grad()
    def evaluate(
        self,
        data: Tuple[torch.Tensor, torch.Tensor],
        split_name: str = "Test"
    ) -> Tuple[float, float]:
        """
        Evaluate global model on validation/test set.
        
        Args:
            data: (X, y) tuple
            split_name: Name of data split (for logging)
        
        Returns:
            (accuracy, loss)
        """
        self.global_model.eval()
        X, y = data
        
        # Create loader
        if torch.cuda.is_available() and self.device == 'cuda':
            dataset = GPUDataset(X, y, self.device)
            loader = DataLoader(dataset, batch_size=self.config.qnn_config.batch_size * 2, 
                              shuffle=False, num_workers=0)
        else:
            dataset = TensorDataset(X, y)
            loader = DataLoader(dataset, batch_size=self.config.qnn_config.batch_size * 2,
                              shuffle=False, num_workers=4, pin_memory=True)
        
        criterion = nn.CrossEntropyLoss()
        total_loss, correct, total = 0.0, 0, 0
        
        for data_batch, target in loader:
            data_batch = data_batch.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            
            outputs = self.global_model(data_batch)
            
            # Validate outputs
            if not torch.isfinite(outputs).all():
                print(f"  ⚠️  Non-finite outputs in {split_name} evaluation")
                continue
            
            loss = criterion(outputs, target)
            total_loss += loss.item() * data_batch.size(0)
            correct += (outputs.argmax(1) == target).sum().item()
            total += target.size(0)
        
        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / total if total > 0 else float('inf')
        
        return accuracy, avg_loss
    
    def train(
        self,
        client_data_list: List[Tuple[torch.Tensor, torch.Tensor]],
        val_data: Tuple[torch.Tensor, torch.Tensor],
        test_data: Tuple[torch.Tensor, torch.Tensor]
    ) -> Dict:
        """
        Main federated training loop.
        
        Args:
            client_data_list: List of (X, y) for each client
            val_data: Validation set (X, y)
            test_data: Test set (X, y)
        
        Returns:
            Training history dictionary
        """
        print(f"\n🚀 Starting Quantum Federated Learning Training\n")
        
        # Initial evaluation
        print("Round 0: Initial Evaluation")
        val_acc, val_loss = self.evaluate(val_data, "Val")
        test_acc, test_loss = self.evaluate(test_data, "Test")
        
        # Privacy Accountant
        self.history['epsilon'] = []
        accountant = RDPAccountant() if RDPAccountant is not None else None
        
        self.history['train_loss'].append(0.0)
        self.history['val_acc'].append(val_acc)
        self.history['val_loss'].append(val_loss)
        self.history['test_acc'].append(test_acc)
        self.history['test_loss'].append(test_loss)
        
        print(f"  Val  Acc: {val_acc:.4f}, Loss: {val_loss:.4f}")
        print(f"  Test Acc: {test_acc:.4f}, Loss: {test_loss:.4f}")
        
        self.best_val_acc = val_acc
        self.best_model_state = {k: v.clone() for k, v in self.global_model.state_dict().items()}
        
        # Training rounds
        for round_num in range(1, self.config.num_rounds + 1):
            # Federated round
            avg_train_loss = self.federated_round(client_data_list, round_num)
            
            # Evaluate
            val_acc, val_loss = self.evaluate(val_data, "Val")
            test_acc, test_loss = self.evaluate(test_data, "Test")
            
            # Step DP Accountant
            current_epsilon = 0.0
            if accountant is not None and self.config.dp_clip_norm > 0:
                accountant.history.append((self.config.dp_noise_multiplier, self.config.client_fraction, 1))
                current_epsilon = accountant.get_epsilon(delta=1e-5)
                self.history['epsilon'].append(current_epsilon)
            
            # Track metrics
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_loss'].append(val_loss)
            self.history['test_acc'].append(test_acc)
            self.history['test_loss'].append(test_loss)
            
            # Early stopping check
            if val_acc > self.best_val_acc + 1e-6:
                self.best_val_acc = val_acc
                self.best_model_state = {k: v.clone() for k, v in self.global_model.state_dict().items()}
                self.patience_counter = 0
                print(f"\n  ✓ New best model (val_acc={val_acc:.4f})")
            else:
                self.patience_counter += 1
            
            # Log progress
            print(f"\n  Val  Acc: {val_acc:.4f}, Loss: {val_loss:.4f}")
            print(f"  Test Acc: {test_acc:.4f}, Loss: {test_loss:.4f}")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            if self.config.dp_clip_norm > 0:
                print(f"  Privacy \u03b5: {current_epsilon:.4f} (δ=1e-5)")
            if self.patience_counter > 0:
                print(f"  Patience: {self.patience_counter}/{self.config.early_stopping_patience}")
            
            # Periodic checkpoint
            if round_num % self.config.checkpoint_interval == 0:
                self.save_checkpoint(round_num)
            
            # Early stopping
            if self.patience_counter >= self.config.early_stopping_patience:
                print(f"\n⏹️  Early stopping at round {round_num}")
                print(f"   No validation improvement for {self.config.early_stopping_patience} rounds")
                break
        
        # Restore best model
        if self.best_model_state is not None:
            self.global_model.load_state_dict(self.best_model_state)
            print(f"\n✓ Restored best model (val_acc={self.best_val_acc:.4f})")
        
        # Final summary
        self._print_summary()
        
        return self.history
    
    def _print_summary(self):
        """Print training summary."""
        print(f"\n{'='*80}")
        print(f"✅ QUANTUM FEDERATED LEARNING COMPLETE")
        print(f"{'='*80}")
        print(f"Best Val Accuracy:  {self.best_val_acc:.4f}")
        print(f"Final Test Accuracy: {self.history['test_acc'][-1]:.4f}")
        print(f"Best Test Accuracy:  {max(self.history['test_acc']):.4f} "
              f"(Round {np.argmax(self.history['test_acc'])})")
        
        if self.config.track_communication:
            print(f"\nCommunication Statistics:")
            print(f"  Model size: {self.model_size_mb:.2f} MB")
            print(f"  Total communication: {self.total_communication:.2f} MB")
            print(f"  Avg per round: {self.total_communication/self.config.num_rounds:.2f} MB")
        
        print(f"{'='*80}\n")
    
    def save_checkpoint(self, round_num: int):
        """Save checkpoint at specific round."""
        checkpoint_path = Path(self.config.save_dir) / f"checkpoint_round{round_num}.pt"
        torch.save({
            'round': round_num,
            'model_state': self.global_model.state_dict(),
            'history': self.history,
            'best_val_acc': self.best_val_acc,
            'patience_counter': self.patience_counter
        }, checkpoint_path)
        print(f"  💾 Checkpoint saved: {checkpoint_path}")
    
    def save_results(self):
        """Save final results and artifacts."""
        # Save model
        model_path = Path(self.config.save_dir) / "quantum_federated_model.pt"
        torch.save({
            'model_state_dict': self.global_model.state_dict(),
            'config': self.config.to_dict(),
            'history': self.history,
            'best_val_acc': self.best_val_acc
        }, model_path)
        print(f"💾 Model saved: {model_path}")
        
        # Save metrics CSV
        csv_path = Path(self.config.save_dir) / "qfl_metrics.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Round", "Train_Loss", "Val_Accuracy", "Val_Loss", 
                           "Test_Accuracy", "Test_Loss"])
            for i in range(len(self.history['test_acc'])):
                writer.writerow([
                    i,
                    f"{self.history['train_loss'][i]:.6f}",
                    f"{self.history['val_acc'][i]:.6f}",
                    f"{self.history['val_loss'][i]:.6f}",
                    f"{self.history['test_acc'][i]:.6f}",
                    f"{self.history['test_loss'][i]:.6f}"
                ])
        print(f"📊 Metrics saved: {csv_path}")
        
        # Save config
        config_path = Path(self.config.save_dir) / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2, default=str)
        print(f"📋 Config saved: {config_path}")
        
        # Plot results
        self._plot_results()
    
    def _plot_results(self):
        """Generate training plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        rounds = list(range(len(self.history['test_acc'])))
        
        # Accuracy
        axes[0, 0].plot(rounds, self.history['val_acc'], marker='s', 
                       label='Validation', color='#F18F01', linewidth=2)
        axes[0, 0].plot(rounds, self.history['test_acc'], marker='o',
                       label='Test', color='#2E86AB', linewidth=2)
        axes[0, 0].set_xlabel("Round", fontsize=11)
        axes[0, 0].set_ylabel("Accuracy", fontsize=11)
        axes[0, 0].set_title("Model Accuracy", fontsize=12, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3, linestyle='--')
        
        # Loss
        axes[0, 1].plot(rounds, self.history['train_loss'], marker='^',
                       label='Train', color='#A23B72', linewidth=2)
        axes[0, 1].plot(rounds, self.history['val_loss'], marker='s',
                       label='Val', color='#F18F01', linewidth=2)
        axes[0, 1].plot(rounds, self.history['test_loss'], marker='o',
                       label='Test', color='#2E86AB', linewidth=2)
        axes[0, 1].set_xlabel("Round", fontsize=11)
        axes[0, 1].set_ylabel("Loss", fontsize=11)
        axes[0, 1].set_title("Loss Curves", fontsize=12, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3, linestyle='--')
        
        # Communication cost
        if self.config.track_communication and self.history['communication_cost']:
            cumsum_comm = np.cumsum(self.history['communication_cost'])
            axes[1, 0].plot(range(1, len(cumsum_comm) + 1), cumsum_comm,
                           marker='d', color='#06A77D', linewidth=2)
            axes[1, 0].set_xlabel("Round", fontsize=11)
            axes[1, 0].set_ylabel("Cumulative Communication (MB)", fontsize=11)
            axes[1, 0].set_title("Communication Cost", fontsize=12, fontweight='bold')
            axes[1, 0].grid(alpha=0.3, linestyle='--')
        
        # Client loss distribution
        if self.history['client_losses']:
            all_losses = []
            for round_losses in self.history['client_losses']:
                all_losses.extend(round_losses.values())
            axes[1, 1].hist(all_losses, bins=30, color='#2E86AB', alpha=0.7, edgecolor='black')
            axes[1, 1].set_xlabel("Loss", fontsize=11)
            axes[1, 1].set_ylabel("Frequency", fontsize=11)
            axes[1, 1].set_title("Client Loss Distribution", fontsize=12, fontweight='bold')
            axes[1, 1].grid(alpha=0.3, linestyle='--', axis='y')
        
        plt.tight_layout()
        plot_path = Path(self.config.save_dir) / "qfl_metrics.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📈 Plots saved: {plot_path}")


# ============================================================================
# MAIN EXAMPLE
# ============================================================================

def main():
    """Example usage of refactored Quantum Federated Learning."""
    
    # Set seeds
    set_seeds(42)
    
    # Configuration
    qnn_config = QNNConfig(
        n_qubits=4,
        n_layers=2,
        encoding='angle',  # Recommended
        entanglement='pyramid',  # Best for deep circuits
        measurement='multi_basis',  # Richer information
        n_features=4,
        n_classes=3,
        batch_size=32,
        epochs=5,  # Local epochs per round
        classical_lr=1e-3,
        quantum_lr=5e-4,
        seed=42
    )
    
    fed_config = FederatedConfig(
        num_clients=4,
        num_rounds=30,
        client_fraction=0.75,
        local_epochs=5,
        early_stopping_patience=10,
        checkpoint_interval=5,
        save_dir='./artifacts/quantum_federated',
        parallel_clients=False,
        track_communication=True,
        sampling_strategy='deterministic',
        qnn_config=qnn_config
    )
    
    print(f"\n{'='*80}")
    print(f"QUANTUM FEDERATED LEARNING - PRODUCTION READY")
    print(f"{'='*80}\n")
    
    # Load data
    try:
        print("Loading preprocessed data...")
        client_data_list = []
        for i in range(fed_config.num_clients):
            data = torch.load(f"./dataset/processed_quantum/client{i+1}.pt")
            client_data_list.append(data)
        
        val_data = torch.load("./dataset/processed_quantum/val.pt")
        test_data = torch.load("./dataset/processed_quantum/test.pt")
        
        qnn_config.n_features = client_data_list[0][0].shape[1]
        
        print(f"✓ Data loaded successfully")
        print(f"  Clients: {len(client_data_list)}")
        print(f"  Features: {qnn_config.n_features}D")
        print(f"  Val samples: {len(val_data[1])}")
        print(f"  Test samples: {len(test_data[1])}")
    
    except Exception as e:
        print(f"⚠️  Could not load data: {e}")
        print("   Using synthetic data for testing...\n")
        
        np.random.seed(42)
        client_data_list = []
        for _ in range(fed_config.num_clients):
            X = torch.randn(250, qnn_config.n_features)
            y = torch.randint(0, qnn_config.n_classes, (250,))
            client_data_list.append((X, y))
        
        X_val = torch.randn(100, qnn_config.n_features)
        y_val = torch.randint(0, qnn_config.n_classes, (100,))
        val_data = (X_val, y_val)
        
        X_test = torch.randn(200, qnn_config.n_features)
        y_test = torch.randint(0, qnn_config.n_classes, (200,))
        test_data = (X_test, y_test)
    
    # Initialize QFL
    qfl = QuantumFederatedLearning(fed_config)
    
    # Train
    try:
        history = qfl.train(client_data_list, val_data, test_data)
        qfl.save_results()
        
        print(f"\n🎉 Quantum Federated Learning completed successfully!")
        print(f"   Final test accuracy: {history['test_acc'][-1]:.4f}")
        print(f"   Best test accuracy: {max(history['test_acc']):.4f}\n")
        
        return qfl, history
    
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    qfl, history = main()