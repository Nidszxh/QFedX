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
from typing import List, Tuple, Dict
import csv
import matplotlib.pyplot as plt

# Import preprocessing functions
from data.preprocess import preprocess_mnist

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seeds(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TinyCNN(nn.Module):
    # eCNN with Batch Normalization for better training stability
    def __init__(self, num_classes=3):
        super(TinyCNN, self).__init__()
        # Convolutional layers with batch normalization
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # First conv block with batch norm
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        
        # Second conv block with batch norm
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        
        # Flatten and fully connected layers
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

def client_update(model_params: Dict, client_data: Tuple[torch.Tensor, torch.Tensor], 
                 epochs: int = 5, lr: float = 0.01, batch_size: int = 32) -> Tuple[Dict, int, float]:
    # Enhanced client update that returns training loss as well
    model = TinyCNN().to(device)
    model.load_state_dict(model_params)
    model.train()
    
    X_client, y_client = client_data
    X_client = X_client.to(dtype=torch.float32).view(-1, 1, 28, 28).to(device)
    y_client = y_client.to(dtype=torch.long).to(device)

    
    dataset = torch.utils.data.TensorDataset(X_client, y_client)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    epoch_losses = []
    for epoch in range(epochs):
        batch_losses = []
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        epoch_losses.append(np.mean(batch_losses))
    
    avg_training_loss = np.mean(epoch_losses)
    return model.state_dict(), len(X_client), avg_training_loss

def federated_averaging(client_updates, device, global_params_template):
    if len(client_updates) == 0:
        raise ValueError("No client updates to aggregate")

    total_samples = sum(num_samples for _, num_samples, _ in client_updates)
    if total_samples == 0:
        raise ValueError("Total number of samples is zero")

    aggregated_params = {}
    for key, tensor in global_params_template.items():
        if tensor.dtype == torch.long:  
            # counters like BatchNorm.num_batches_tracked
            aggregated_params[key] = tensor.clone().to(device)
        else:
            aggregated_params[key] = torch.zeros_like(tensor, dtype=torch.float32, device=device)

    weighted_loss = 0.0
    for params, num_samples, training_loss in client_updates:
        weight = num_samples / total_samples
        weighted_loss += weight * training_loss
        for key in aggregated_params.keys():
            client_tensor = params[key].to(device)
            if aggregated_params[key].dtype == torch.long:
                # just copy (don’t try to average)
                aggregated_params[key] = client_tensor.clone()
            else:
                aggregated_params[key] += client_tensor.float() * weight

    return aggregated_params, weighted_loss

def evaluate_model(model: nn.Module, test_data: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[float, float]:
    # Enhanced evaluation that returns both accuracy and loss
    model.eval()
    X_test, y_test = test_data
    X_test = X_test.to(dtype=torch.float32).view(-1, 1, 28, 28).to(device)
    y_test = y_test.to(dtype=torch.long).to(device)

    dataset = torch.utils.data.TensorDataset(X_test, y_test)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    losses = []
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            losses.append(loss.item())
            
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    
    accuracy = correct / total
    avg_loss = np.mean(losses)
    return accuracy, avg_loss

class ClientPool:
    # Manages client model instances to avoid repeated instantiation
    def __init__(self, num_clients: int, num_classes: int = 3):
        self.clients = [TinyCNN(num_classes).to(device) for _ in range(num_clients)]
        
    def get_client(self, client_id: int):
        return self.clients[client_id]

def federated_learning(client_data: List[Tuple[torch.Tensor, torch.Tensor]], 
                      test_data: Tuple[torch.Tensor, torch.Tensor],
                      num_rounds: int = 30, 
                      local_epochs: int = 5,
                      learning_rate: float = 0.01,
                      batch_size: int = 32,
                      num_classes: int = 3,
                      client_fraction: float = 1.0) -> Dict:
    # Enhanced federated learning with client sampling and detailed metrics
    os.makedirs("artifacts", exist_ok=True)

    # Initialize global model and client pool for efficiency
    global_model = TinyCNN(num_classes=num_classes).to(device)
    client_pool = ClientPool(len(client_data), num_classes)
    
    # Track metrics over rounds
    test_accuracies = []
    test_losses = []
    train_losses = []
    num_clients = len(client_data)
    
    print(f"Enhanced Federated Learning:")
    print(f"- {num_clients} total clients")
    print(f"- Client participation rate: {client_fraction:.1%}")
    print(f"- Training for {num_rounds} rounds")
    print("-" * 60)
    
    # Initial evaluation
    initial_acc, initial_loss = evaluate_model(global_model, test_data)
    test_accuracies.append(initial_acc)
    test_losses.append(initial_loss)
    train_losses.append(0.0)  # No training loss for initial round
    print(f"Round 0: Test Accuracy = {initial_acc:.4f}, Test Loss = {initial_loss:.4f}")
    
    # Training loop with client sampling
    for round_num in range(num_rounds):
        # Sample clients for this round
        num_selected = max(1, int(client_fraction * num_clients))
        selected_clients = random.sample(range(num_clients), num_selected)
        
        print(f"Round {round_num + 1}: Selected {num_selected}/{num_clients} clients: {selected_clients}")
        
        client_updates = []
        for client_id in selected_clients:
            # Use pooled client model for efficiency
            client_model = client_pool.get_client(client_id)
            client_model.load_state_dict(global_model.state_dict())
            
            params, num_samples, training_loss = client_update(
                global_model.state_dict(),
                client_data[client_id], 
                epochs=local_epochs,
                lr=learning_rate,
                batch_size=batch_size
            )
            client_updates.append((params, num_samples, training_loss))

        # Aggregate updates
        aggregated_params, avg_train_loss = federated_averaging(client_updates, device, global_model.state_dict())
        global_model.load_state_dict(aggregated_params)
        
        # Evaluate
        test_acc, test_loss = evaluate_model(global_model, test_data)
        test_accuracies.append(test_acc)
        test_losses.append(test_loss)
        train_losses.append(avg_train_loss)
        
        # Print progress every 5 rounds or final round
        if (round_num + 1) % 5 == 0 or round_num == num_rounds - 1:
            print(f"Round {round_num + 1}: Test Acc = {test_acc:.4f}, Test Loss = {test_loss:.4f}, Train Loss = {avg_train_loss:.4f}")

    print(f"\nFinal Results:")
    print(f"- Test Accuracy: {test_accuracies[-1]:.4f}")
    print(f"- Test Loss: {test_losses[-1]:.4f}")

    # Save global model
    torch.save(global_model.state_dict(), "artifacts/enhanced_global_model.pt")
    
    # Save comprehensive metrics
    with open("artifacts/enhanced_round_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Round", "Test_Accuracy", "Test_Loss", "Train_Loss"])
        for i, (acc, test_loss, train_loss) in enumerate(zip(test_accuracies, test_losses, train_losses)):
            writer.writerow([i, acc, test_loss, train_loss])

    # Enhanced plotting with dual metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    ax1.plot(range(len(test_accuracies)), test_accuracies, marker='o', label='Test Accuracy')
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("FedAvg Test Accuracy")
    ax1.grid(True)
    ax1.legend()
    
    # Loss plot
    ax2.plot(range(len(test_losses)), test_losses, marker='s', label='Test Loss', color='red')
    ax2.plot(range(len(train_losses)), train_losses, marker='^', label='Train Loss', color='blue')
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Loss")
    ax2.set_title("FedAvg Loss Curves")
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig("artifacts/enhanced_fedavg_metrics.png", dpi=150)
    print("Enhanced metrics plot saved to artifacts/enhanced_fedavg_metrics.png")
    plt.close(fig)

    return {
        'model': global_model, 
        'test_accuracies': test_accuracies,
        'test_losses': test_losses,
        'train_losses': train_losses
    }

def main():
    config = {
        'raw_folder': "./dataset/raw",
        'processed_folder': "./dataset/processed",
        'digits': (0, 1, 2),
        'val_split': 0.1,
        'num_clients': 4,
        'partition_type': 'iid',  # or 'non_iid'
        'alpha': 0.5,
        'num_rounds': 30,
        'local_epochs': 5,
        'learning_rate': 0.01,
        'batch_size': 32,
        'client_fraction': 0.75  # New parameter: fraction of clients participating per round
    }
    
    print("Enhanced Federated Learning on MNIST")
    print(f"Configuration: {config}")
  
    # Data preprocessing
    try:
        result = preprocess_mnist(
            raw_folder=config['raw_folder'],
            processed_folder=config['processed_folder'],
            digits=config['digits'],
            val_split=config['val_split'],
            num_clients=config['num_clients'],
            partition_type=config['partition_type'],
            alpha=config['alpha']
        )
        
        # Error handling for preprocessing failure
        if result is None:
            print("Error: Data preprocessing failed. Please check that MNIST data files exist in the raw folder.")
            print("Expected files: train-images.idx3-ubyte, train-labels.idx1-ubyte, t10k-images.idx3-ubyte, t10k-labels.idx1-ubyte")
            return
        
        train_data, val_data, test_data, client_data = result
        
    except Exception as e:
        print(f"Error during data preprocessing: {e}")
        return
    
    # Run enhanced federated learning
    try:
        results = federated_learning(
            client_data=client_data,
            test_data=test_data,
            num_rounds=config['num_rounds'],
            local_epochs=config['local_epochs'],
            learning_rate=config['learning_rate'],
            batch_size=config['batch_size'],
            num_classes=len(config['digits']),
            client_fraction=config['client_fraction']  # New parameter
        )
        
        print(f"Enhanced training completed successfully!")
        print(f"Final test accuracy: {results['test_accuracies'][-1]:.4f}")
        print(f"Final test loss: {results['test_losses'][-1]:.4f}")
        
    except Exception as e:
        print(f"Error during federated learning training: {e}")
        return

if __name__ == "__main__":
    main()