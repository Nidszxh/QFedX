"""
Classical Federated Learning with CNN
Optimized for correctness, performance, and reproducibility
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import csv
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional

from data.preprocess import preprocess_mnist_classical

# Import visualization utilities
try:
    from viz_cFL import generate_all_cfl_visualizations
    VISUALIZATIONS_AVAILABLE = True
except ImportError:
    VISUALIZATIONS_AVAILABLE = False


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def set_seeds(seed: int = 42):
    """Comprehensive seeding for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # DataLoader worker seeding
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    return seed_worker


seed_worker = set_seeds(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Device: {device}")


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class TinyCNN(nn.Module):
    """
    Lightweight CNN for MNIST with Batch Normalization.
    Architecture: Conv(16) -> Pool -> Conv(32) -> Pool -> FC(64) -> FC(num_classes)
    
    Parameters: ~82K (suitable for 3-class classification)
    """
    def __init__(self, num_classes: int = 3):
        super(TinyCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        x = x.view(-1, 32 * 7 * 7)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


# ============================================================================
# DATA VALIDATION
# ============================================================================

def verify_data_format(
    client_data: List[Tuple[torch.Tensor, torch.Tensor]], 
    expected_dim: int = 784, 
    expected_range: Tuple[float, float] = (0, 1)
) -> None:
    """Verify data format matches CNN requirements."""
    print("\n🔍 Verifying data format...")
    
    for i, (X, y) in enumerate(client_data):
        # Check dimensions
        if X.shape[1] != expected_dim:
            raise ValueError(
                f"❌ Client {i}: Expected {expected_dim}D, got {X.shape[1]}D.\n"
                f"   Use preprocess_mnist_classical() for CNN models."
            )
        
        # Check range
        min_val, max_val = X.min().item(), X.max().item()
        exp_min, exp_max = expected_range
        
        if min_val < exp_min - 0.01 or max_val > exp_max + 0.01:
            raise ValueError(
                f"❌ Client {i}: Data range [{min_val:.3f}, {max_val:.3f}] "
                f"outside expected {expected_range}"
            )
        
        print(f"   Client {i+1}: ✓ {X.shape[1]}D, range [{min_val:.3f}, {max_val:.3f}]")
    
    print("   ✅ All clients validated!\n")


# ============================================================================
# CLIENT TRAINING
# ============================================================================

class ClientTrainer:
    """
    Persistent client trainer with reusable DataLoader.
    Optimized to avoid repeated DataLoader creation overhead.
    """
    def __init__(
        self, 
        client_data: Tuple[torch.Tensor, torch.Tensor],
        batch_size: int = 32,
        device: torch.device = torch.device('cpu')
    ):
        self.X, self.y = client_data
        self.device = device
        self.batch_size = batch_size
        self.n_samples = len(self.X)
        
        # Create DataLoader once (reused across rounds)
        self.dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(self.X, self.y),
            batch_size=batch_size,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(42),
            pin_memory=(device.type == 'cuda')
        )
    
    def train(
        self, 
        model_params: Dict[str, torch.Tensor],
        epochs: int = 5,
        lr: float = 0.01
    ) -> Tuple[Dict[str, torch.Tensor], int, float]:
        """
        Perform local training.
        
        Returns:
            (updated_params, num_samples, avg_training_loss)
        """
        model = TinyCNN().to(self.device)
        model.load_state_dict(model_params)
        model.train()
        
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        epoch_losses = []
        for _ in range(epochs):
            batch_losses = []
            for batch_x, batch_y in self.dataloader:
                # Reshape to images and move to device
                batch_x = batch_x.view(-1, 1, 28, 28).to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)
                
                optimizer.zero_grad()
                loss = criterion(model(batch_x), batch_y)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())
            
            epoch_losses.append(np.mean(batch_losses))
        
        return model.state_dict(), self.n_samples, np.mean(epoch_losses)


# ============================================================================
# FEDERATED AGGREGATION
# ============================================================================

def federated_averaging(
    client_updates: List[Tuple[Dict[str, torch.Tensor], int, float]], 
    global_params_template: Dict[str, torch.Tensor]
) -> Tuple[Dict[str, torch.Tensor], float]:
    """
    Aggregate client updates using weighted averaging (FedAvg).
    
    FIXED: Proper BatchNorm handling - skips running statistics.
    
    Returns:
        (aggregated_params, weighted_avg_loss)
    """
    if not client_updates:
        raise ValueError("No client updates to aggregate")
    
    total_samples = sum(n for _, n, _ in client_updates)
    if total_samples == 0:
        raise ValueError("Total samples is zero")
    
    # Initialize aggregated parameters
    aggregated = {
        key: torch.zeros_like(tensor, device=device)
        for key, tensor in global_params_template.items()
    }
    
    weighted_loss = 0.0
    
    for params, n_samples, loss in client_updates:
        weight = n_samples / total_samples
        weighted_loss += weight * loss
        
        for key in aggregated.keys():
            # Skip BatchNorm running statistics (not trainable)
            if any(skip in key for skip in ['running_mean', 'running_var', 'num_batches_tracked']):
                # Keep global statistics (don't aggregate)
                aggregated[key] = global_params_template[key].clone()
            else:
                # Aggregate trainable parameters
                aggregated[key] += params[key].to(device) * weight
    
    return aggregated, weighted_loss


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(
    model: nn.Module, 
    test_data: Tuple[torch.Tensor, torch.Tensor],
    batch_size: int = 256
) -> Tuple[float, float]:
    """
    Evaluate model on test/validation set.
    
    Returns:
        (accuracy, avg_loss)
    """
    model.eval()
    X_test, y_test = test_data
    
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(device.type == 'cuda')
    )
    
    criterion = nn.CrossEntropyLoss()
    correct, total = 0, 0
    losses = []
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.view(-1, 1, 28, 28).to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            
            outputs = model(batch_x)
            losses.append(criterion(outputs, batch_y).item())
            correct += (outputs.argmax(1) == batch_y).sum().item()
            total += batch_y.size(0)
    
    return correct / total, np.mean(losses)


# ============================================================================
# FEDERATED LEARNING LOOP
# ============================================================================

def federated_learning(
    client_data: List[Tuple[torch.Tensor, torch.Tensor]],
    val_data: Tuple[torch.Tensor, torch.Tensor],
    test_data: Tuple[torch.Tensor, torch.Tensor],
    num_rounds: int = 30,
    local_epochs: int = 5,
    learning_rate: float = 0.01,
    batch_size: int = 32,
    num_classes: int = 3,
    client_fraction: float = 1.0,
    early_stopping_patience: int = 10,
    save_dir: str = "./artifacts/classical",
    config: Optional[Dict] = None
) -> Dict:
    """
    Run federated learning with FedAvg algorithm.
    
    NEW FEATURES:
    - Validation set evaluation
    - Early stopping based on validation accuracy
    - Fixed BatchNorm aggregation
    - Optimized client training (persistent DataLoaders)
    
    Args:
        client_data: List of (X, y) tuples for each client
        val_data: Validation set (X, y) - NEW!
        test_data: Test set (X, y)
        num_rounds: Number of federated rounds
        local_epochs: Local training epochs per client
        learning_rate: Client SGD learning rate
        batch_size: Client training batch size
        num_classes: Number of output classes
        client_fraction: Fraction of clients to sample per round
        early_stopping_patience: Stop if no val improvement for N rounds - NEW!
        save_dir: Directory for saving artifacts
        config: Configuration dictionary
    
    Returns:
        Dictionary with model and training metrics
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize global model
    global_model = TinyCNN(num_classes=num_classes).to(device)
    num_clients = len(client_data)
    
    # Initialize client trainers (reusable DataLoaders)
    print("\n🔧 Initializing client trainers...")
    client_trainers = [
        ClientTrainer(data, batch_size, device) 
        for data in client_data
    ]
    print(f"   ✓ {num_clients} client trainers ready")
    
    # Metrics tracking
    test_accuracies, test_losses = [], []
    val_accuracies, val_losses = [], []
    train_losses = []
    client_losses_per_round = []
    
    # Early stopping
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    
    print("\n" + "="*70)
    print("🚀 Federated Learning Training")
    print("="*70)
    print(f"   Total clients: {num_clients}")
    print(f"   Participation rate: {client_fraction:.0%}")
    print(f"   Rounds: {num_rounds}")
    print(f"   Local epochs: {local_epochs}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Early stopping patience: {early_stopping_patience}")
    print("="*70 + "\n")
    
    # Initial evaluation
    init_val_acc, init_val_loss = evaluate_model(global_model, val_data)
    init_test_acc, init_test_loss = evaluate_model(global_model, test_data)
    
    val_accuracies.append(init_val_acc)
    val_losses.append(init_val_loss)
    test_accuracies.append(init_test_acc)
    test_losses.append(init_test_loss)
    train_losses.append(0.0)
    
    print(f"Round  0 [Initial]:")
    print(f"   Val  Acc = {init_val_acc:.4f}, Val  Loss = {init_val_loss:.4f}")
    print(f"   Test Acc = {init_test_acc:.4f}, Test Loss = {init_test_loss:.4f}\n")
    
    # Training loop
    for round_num in range(1, num_rounds + 1):
        # Sample clients (deterministic per round)
        random.seed(42 + round_num)
        num_selected = max(1, int(client_fraction * num_clients))
        selected = random.sample(range(num_clients), num_selected)
        
        # Client updates
        client_updates = []
        round_client_losses = [0.0] * num_clients
        
        for cid in selected:
            try:
                params, n_samples, loss = client_trainers[cid].train(
                    global_model.state_dict(),
                    epochs=local_epochs,
                    lr=learning_rate
                )
                client_updates.append((params, n_samples, loss))
                round_client_losses[cid] = loss
            except Exception as e:
                print(f"   ⚠️  Client {cid} failed: {e}")
                continue
        
        # Check if enough clients succeeded
        if len(client_updates) < max(1, num_selected // 2):
            print(f"   ⚠️  Too many client failures ({len(client_updates)}/{num_selected})")
            print(f"      Skipping round {round_num}")
            continue
        
        client_losses_per_round.append([round_client_losses[cid] for cid in selected])
        
        # Aggregate
        aggregated, avg_train_loss = federated_averaging(
            client_updates,
            global_model.state_dict()
        )
        global_model.load_state_dict(aggregated)
        
        # Evaluate on validation set
        val_acc, val_loss = evaluate_model(global_model, val_data)
        val_accuracies.append(val_acc)
        val_losses.append(val_loss)
        
        # Evaluate on test set
        test_acc, test_loss = evaluate_model(global_model, test_data)
        test_accuracies.append(test_acc)
        test_losses.append(test_loss)
        train_losses.append(avg_train_loss)
        
        # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = global_model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Log progress
        if round_num % 5 == 0 or round_num == num_rounds:
            print(f"Round {round_num:>3}:")
            print(f"   Val  Acc = {val_acc:.4f}, Val  Loss = {val_loss:.4f}")
            print(f"   Test Acc = {test_acc:.4f}, Test Loss = {test_loss:.4f}")
            print(f"   Train Loss = {avg_train_loss:.4f}")
            if patience_counter > 0:
                print(f"   Patience: {patience_counter}/{early_stopping_patience}")
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\n⏹️  Early stopping at round {round_num}")
            print(f"   No validation improvement for {early_stopping_patience} rounds")
            print(f"   Restoring best model (val_acc={best_val_acc:.4f})")
            global_model.load_state_dict(best_model_state)
            break
    
    # Restore best model if not already done
    if best_model_state is not None and patience_counter < early_stopping_patience:
        global_model.load_state_dict(best_model_state)
    
    # Final results
    print("\n" + "="*70)
    print("✅ Training Complete!")
    print("="*70)
    print(f"   Final Test Accuracy: {test_accuracies[-1]:.4f}")
    print(f"   Final Test Loss: {test_losses[-1]:.4f}")
    print(f"   Best Val Accuracy: {best_val_acc:.4f}")
    print(f"   Best Test Accuracy: {max(test_accuracies):.4f} (Round {np.argmax(test_accuracies)})")
    print("="*70 + "\n")
    
    # Save artifacts
    torch.save(global_model.state_dict(), Path(save_dir) / "global_model.pt")
    print(f"💾 Model saved: {save_dir}/global_model.pt")
    
    # Save metrics CSV
    with open(Path(save_dir) / "metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Round", "Val_Accuracy", "Val_Loss", "Test_Accuracy", "Test_Loss", "Train_Loss"])
        for i in range(len(test_accuracies)):
            writer.writerow([
                i,
                f"{val_accuracies[i]:.6f}",
                f"{val_losses[i]:.6f}",
                f"{test_accuracies[i]:.6f}",
                f"{test_losses[i]:.6f}",
                f"{train_losses[i]:.6f}"
            ])
    print(f"📊 Metrics saved: {save_dir}/metrics.csv")
    
    # Plot metrics
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    rounds = list(range(len(test_accuracies)))
    
    # Accuracy
    axes[0].plot(rounds, val_accuracies, marker='s', linewidth=2, markersize=4, 
                 label='Validation', color='#F18F01')
    axes[0].plot(rounds, test_accuracies, marker='o', linewidth=2, markersize=4, 
                 label='Test', color='#2E86AB')
    axes[0].set_xlabel("Round", fontsize=11)
    axes[0].set_ylabel("Accuracy", fontsize=11)
    axes[0].set_title("Model Accuracy", fontsize=12, fontweight='bold')
    axes[0].legend(loc='lower right')
    axes[0].grid(alpha=0.3, linestyle='--')
    axes[0].set_ylim([0, 1.05])
    
    # Loss curves
    axes[1].plot(rounds, val_losses, marker='s', linewidth=2, markersize=4,
                 label='Val Loss', color='#F18F01')
    axes[1].plot(rounds, test_losses, marker='o', linewidth=2, markersize=4, 
                 label='Test Loss', color='#2E86AB')
    axes[1].plot(rounds, train_losses, marker='^', linewidth=2, markersize=4,
                 label='Train Loss', color='#A23B72')
    axes[1].set_xlabel("Round", fontsize=11)
    axes[1].set_ylabel("Loss", fontsize=11)
    axes[1].set_title("Loss Curves", fontsize=12, fontweight='bold')
    axes[1].legend(loc='upper right')
    axes[1].grid(alpha=0.3, linestyle='--')
    
    # Generalization gap
    gen_gap = [test_acc - val_acc for test_acc, val_acc in zip(test_accuracies, val_accuracies)]
    axes[2].plot(rounds, gen_gap, marker='d', linewidth=2, markersize=4, color='#06A77D')
    axes[2].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[2].set_xlabel("Round", fontsize=11)
    axes[2].set_ylabel("Test Acc - Val Acc", fontsize=11)
    axes[2].set_title("Generalization Gap", fontsize=12, fontweight='bold')
    axes[2].grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plot_path = Path(save_dir) / "fedavg_metrics.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"📈 Plots saved: {plot_path}")
    
    # Prepare metrics dictionary
    metrics = {
        'model': global_model,
        'val_accuracies': val_accuracies,
        'val_losses': val_losses,
        'test_accuracies': test_accuracies,
        'test_losses': test_losses,
        'train_losses': train_losses,
        'client_losses_per_round': client_losses_per_round,
        'best_val_acc': best_val_acc
    }
    
    # Generate advanced visualizations
    if VISUALIZATIONS_AVAILABLE:
        try:
            class_names = [f"Digit {d}" for d in config.get('digits', range(num_classes))] if config else None
            
            generate_all_cfl_visualizations(
                metrics=metrics,
                config=config if config else {},
                client_data=client_data,
                test_data=test_data,
                global_model=global_model,
                device=device,
                client_losses_per_round=client_losses_per_round,
                class_names=class_names,
                save_dir="./results/cfl"
            )
        except Exception as e:
            print(f"\n⚠️  Warning: Could not generate advanced visualizations: {e}")
    
    return metrics


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point for classical federated learning."""
    config = {
        'raw_folder': "./dataset/raw",
        'digits': (0, 1, 2),
        'val_split': 0.1,
        'num_clients': 4,
        'partition_type': 'iid',
        'alpha': 0.5,
        'num_rounds': 30,
        'local_epochs': 5,
        'learning_rate': 0.01,
        'batch_size': 32,
        'client_fraction': 0.75,
        'early_stopping_patience': 10,
        'save_dir': './artifacts/classical',
        'seed': 42
    }
    
    print("\n" + "="*70)
    print("📊 Classical Federated Learning on MNIST")
    print("="*70)
    print("\n📋 Configuration:")
    for k, v in config.items():
        print(f"  {k:25s}: {v}")
    print()
    
    # Preprocessing
    try:
        print("="*70)
        result = preprocess_mnist_classical(
            raw_folder=config['raw_folder'],
            digits=config['digits'],
            val_split=config['val_split'],
            num_clients=config['num_clients'],
            partition_type=config['partition_type'],
            alpha=config['alpha'],
            seed=config['seed'],
            generate_plots=False
        )
        print("="*70)
        
        if result is None:
            print("\n❌ Data preprocessing failed.")
            return
        
        train_data, val_data, test_data, client_data = result
        
        # Verify data format
        verify_data_format(client_data, expected_dim=784, expected_range=(0, 1))
        
        print(f"✅ Data loaded successfully!")
        print(f"   Clients: {len(client_data)}")
        print(f"   Val samples: {len(val_data[1])}")
        print(f"   Test samples: {len(test_data[1])}\n")
        
    except Exception as e:
        print(f"\n❌ Preprocessing error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Run federated learning
    try:
        results = federated_learning(
            client_data=client_data,
            val_data=val_data,
            test_data=test_data,
            num_rounds=config['num_rounds'],
            local_epochs=config['local_epochs'],
            learning_rate=config['learning_rate'],
            batch_size=config['batch_size'],
            num_classes=len(config['digits']),
            client_fraction=config['client_fraction'],
            early_stopping_patience=config['early_stopping_patience'],
            save_dir=config['save_dir'],
            config=config
        )
        
        print(f"🎉 Final test accuracy: {results['test_accuracies'][-1]:.4f}")
        print(f"🏆 Best val accuracy: {results['best_val_acc']:.4f}\n")
        
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()