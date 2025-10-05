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

from data.preprocess import preprocess_mnist

# Configuration and Device Setup
def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seeds(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# CNN Model Architecture
class TinyCNN(nn.Module):
    """
    Lightweight CNN for MNIST with Batch Normalization.
    Architecture: Conv(16) -> Pool -> Conv(32) -> Pool -> FC(64) -> FC(num_classes)
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

# Federated Learning Core Functions
def client_update(model_params: Dict[str, torch.Tensor], client_data: Tuple[torch.Tensor, torch.Tensor], 
                    epochs: int = 5, lr: float = 0.01, batch_size: int = 32) -> Tuple[Dict[str, torch.Tensor], int, float]:
    """
    Perform local training on a client.
    Returns: 
        (updated_params, num_samples, avg_training_loss)
    """
    model = TinyCNN().to(device)
    model.load_state_dict(model_params)
    model.train()
    
    X_client, y_client = client_data
    X_client = X_client.view(-1, 1, 28, 28).to(device)
    y_client = y_client.to(device)
    
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_client, y_client),
        batch_size=batch_size,
        shuffle=True
    )
    
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    epoch_losses = []
    for _ in range(epochs):
        batch_losses = []
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        epoch_losses.append(np.mean(batch_losses))
    
    return model.state_dict(), len(X_client), np.mean(epoch_losses)

def federated_averaging(client_updates: List[Tuple[Dict, int, float]], 
                        global_params_template: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], float]:
    """
    Aggregate client updates using weighted averaging (FedAvg).
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
        key: (tensor.clone().to(device) if tensor.dtype == torch.long 
            else torch.zeros_like(tensor, device=device))
        for key, tensor in global_params_template.items()
    }
    
    weighted_loss = 0.0
    for params, n_samples, loss in client_updates:
        weight = n_samples / total_samples
        weighted_loss += weight * loss
        
        for key in aggregated.keys():
            if aggregated[key].dtype != torch.long:  # Skip BatchNorm counters
                aggregated[key] += params[key].to(device) * weight
    
    return aggregated, weighted_loss

def evaluate_model(model: nn.Module, test_data: Tuple[torch.Tensor, torch.Tensor],
                    batch_size: int = 256) -> Tuple[float, float]:
    """
    Evaluate model on test set.
    Returns:
        (accuracy, avg_loss)
    """
    model.eval()
    X_test, y_test = test_data
    X_test = X_test.view(-1, 1, 28, 28).to(device)
    y_test = y_test.to(device)
    
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False
    )
    
    criterion = nn.CrossEntropyLoss()
    correct, total = 0, 0
    losses = []
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            outputs = model(batch_x)
            losses.append(criterion(outputs, batch_y).item())
            correct += (outputs.argmax(1) == batch_y).sum().item()
            total += batch_y.size(0)
    
    return correct / total, np.mean(losses)

# Client Pool Manager (Optimization)

class ClientPool:
    """Manages pre-instantiated client models to avoid repeated allocation."""
    def __init__(self, num_clients: int, num_classes: int = 3):
        self.clients = [TinyCNN(num_classes).to(device) for _ in range(num_clients)]
    
    def get_client(self, client_id: int) -> nn.Module:
        return self.clients[client_id]

# Main Federated Learning Training Loop

def federated_learning(client_data: List[Tuple[torch.Tensor, torch.Tensor]], test_data: Tuple[torch.Tensor, torch.Tensor],
                        num_rounds: int = 30, local_epochs: int = 5, learning_rate: float = 0.01,
                        batch_size: int = 32, num_classes: int = 3, client_fraction: float = 1.0,
                        save_dir: str = "./artifacts") -> Dict:
    """
    Run federated learning with FedAvg algorithm.
    
    Args:
        client_data: List of (X, y) tuples for each client
        test_data: Global test set (X, y)
        num_rounds: Number of federated rounds
        local_epochs: Local training epochs per client
        learning_rate: Client SGD learning rate
        batch_size: Client training batch size
        num_classes: Number of output classes
        client_fraction: Fraction of clients to sample per round
        save_dir: Directory for saving artifacts
    
    Returns:
        Dictionary with model and training metrics
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize
    global_model = TinyCNN(num_classes=num_classes).to(device)
    client_pool = ClientPool(len(client_data), num_classes)
    num_clients = len(client_data)
    
    # Metrics tracking
    test_accuracies, test_losses, train_losses = [], [], []
    
    print("Federated Learning Training")
    print(f"   Total clients: {num_clients}")
    print(f"   Participation rate: {client_fraction:.0%}")
    print(f"   Rounds: {num_rounds}")
    print(f"   Local epochs: {local_epochs}")
    print(f"   Learning rate: {learning_rate}"+ "\n")
    
    # Initial evaluation
    init_acc, init_loss = evaluate_model(global_model, test_data)
    test_accuracies.append(init_acc)
    test_losses.append(init_loss)
    train_losses.append(0.0)
    print(f"Round  0 [Initial]: Test Acc = {init_acc:.4f}, Test Loss = {init_loss:.4f}")
    
    # Training loop
    for round_num in range(1, num_rounds + 1):
        # Sample clients
        num_selected = max(1, int(client_fraction * num_clients))
        selected = random.sample(range(num_clients), num_selected)
        
        # Client updates
        client_updates = []
        for cid in selected:
            params, n_samples, loss = client_update(
                global_model.state_dict(),
                client_data[cid],
                epochs=local_epochs,
                lr=learning_rate,
                batch_size=batch_size
            )
            client_updates.append((params, n_samples, loss))
        
        # Aggregate
        aggregated, avg_train_loss = federated_averaging(
            client_updates,
            global_model.state_dict()
        )
        global_model.load_state_dict(aggregated)
        
        # Evaluate
        test_acc, test_loss = evaluate_model(global_model, test_data)
        test_accuracies.append(test_acc)
        test_losses.append(test_loss)
        train_losses.append(avg_train_loss)
        
        # Log progress
        if round_num % 5 == 0 or round_num == num_rounds:
            print(f"Round {round_num:>3}: Selected {num_selected} clients | "
                  f"Test Acc = {test_acc:.4f} | Test Loss = {test_loss:.4f} | "
                  f"Train Loss = {avg_train_loss:.4f}")
    
    # Final results
    print("\nTraining Complete!")
    print(f"   Final Test Accuracy: {test_accuracies[-1]:.4f}")
    print(f"   Final Test Loss: {test_losses[-1]:.4f}")

    # Save artifacts
    torch.save(global_model.state_dict(), Path(save_dir) / "global_model.pt")
    print(f"💾 Model saved: {save_dir}/global_model.pt")
    
    # Save metrics CSV
    with open(Path(save_dir) / "metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Round", "Test_Accuracy", "Test_Loss", "Train_Loss"])
        for i, (acc, tl, tr) in enumerate(zip(test_accuracies, test_losses, train_losses)):
            writer.writerow([i, f"{acc:.6f}", f"{tl:.6f}", f"{tr:.6f}"])
    print(f"📊 Metrics saved: {save_dir}/metrics.csv")
    
    # Plot metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    rounds = list(range(len(test_accuracies)))
    
    # Accuracy
    ax1.plot(rounds, test_accuracies, marker='o', linewidth=2, markersize=4, color='#2E86AB')
    ax1.set_xlabel("Round", fontsize=11)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.set_title("Test Accuracy", fontsize=12, fontweight='bold')
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.set_ylim([0, 1.05])
    
    # Loss
    ax2.plot(rounds, test_losses, marker='s', linewidth=2, markersize=4, 
             label='Test Loss', color='#A23B72')
    ax2.plot(rounds, train_losses, marker='^', linewidth=2, markersize=4,
             label='Train Loss', color='#F18F01')
    ax2.set_xlabel("Round", fontsize=11)
    ax2.set_ylabel("Loss", fontsize=11)
    ax2.set_title("Loss Curves", fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plot_path = Path(save_dir) / "fedavg_metrics.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"📈 Plots saved: {plot_path}\n")
    
    return {
        'model': global_model,
        'test_accuracies': test_accuracies,
        'test_losses': test_losses,
        'train_losses': train_losses
    }

def main():
    """Main entry point for classical federated learning."""
    config = {
        'raw_folder': "./dataset/raw",
        'processed_folder': "./dataset/processed",
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
        'save_dir': './artifacts'
    }
    
    print("\nClassical Federated Learning on MNIST")
    print("Configuration:")
    for k, v in config.items():
        print(f"  {k:20s}: {v}")
    
    # Preprocessing
    try:
        result = preprocess_mnist(
            raw_folder=config['raw_folder'],
            processed_folder=config['processed_folder'],
            digits=config['digits'],
            val_split=config['val_split'],
            num_clients=config['num_clients'],
            partition_type=config['partition_type'],
            alpha=config['alpha'],
            generate_plots=True  # Disable for cleaner output
        )
        
        if result is None:
            print(" Data preprocessing failed. Check MNIST files in raw folder.")
            print("   Expected: train-images.idx3-ubyte, train-labels.idx1-ubyte,")
            print("            t10k-images.idx3-ubyte, t10k-labels.idx1-ubyte")
            return
        
        train_data, val_data, test_data, client_data = result
        print(f"Data loaded: {len(client_data)} clients, {len(test_data[1])} test samples\n")
        
    except Exception as e:
        print(f" Preprocessing error: {e}")
        return
    
    # Run federated learning
    try:
        results = federated_learning(
            client_data=client_data,
            test_data=test_data,
            num_rounds=config['num_rounds'],
            local_epochs=config['local_epochs'],
            learning_rate=config['learning_rate'],
            batch_size=config['batch_size'],
            num_classes=len(config['digits']),
            client_fraction=config['client_fraction'],
            save_dir=config['save_dir']
        )
        
        print(f"Final accuracy: {results['test_accuracies'][-1]:.4f}\n")
        
    except Exception as e:
        print(f" Training error: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()