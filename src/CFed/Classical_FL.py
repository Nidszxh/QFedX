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
from data.Preprocess import preprocess_mnist

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seeds(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TinyCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(TinyCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

def client_update(model_params: Dict, client_data: Tuple[torch.Tensor, torch.Tensor], 
                 epochs: int = 5, lr: float = 0.01, batch_size: int = 32) -> Tuple[Dict, int]:
    model = TinyCNN().to(device)
    model.load_state_dict(model_params)
    model.train()
    
    X_client, y_client = client_data
    X_client = torch.tensor(X_client, dtype=torch.float32).view(-1, 1, 28, 28).to(device)
    y_client = torch.tensor(y_client, dtype=torch.long).to(device)
    
    dataset = torch.utils.data.TensorDataset(X_client, y_client)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    for _ in range(epochs):
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    return model.state_dict(), len(X_client)

def federated_averaging(client_updates: List[Tuple[Dict, int]]) -> Dict:
    # Aggregate client updates using weighted averaging
    total_samples = sum(num_samples for _, num_samples in client_updates)
    
    aggregated_params = {}
    first_params = client_updates[0][0]
    
    for key in first_params.keys():
        aggregated_params[key] = torch.zeros_like(first_params[key])
    
    for params, num_samples in client_updates:
        weight = num_samples / total_samples
        for key in aggregated_params.keys():
            aggregated_params[key] += weight * params[key].to(device)

    return aggregated_params

def evaluate_model(model: nn.Module, test_data: Tuple[torch.Tensor, torch.Tensor]) -> float:
    # Evaluate model and return accuracy
    model.eval()
    X_test, y_test = test_data
    X_test = torch.tensor(X_test, dtype=torch.float32).view(-1, 1, 28, 28).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)
    
    dataset = torch.utils.data.TensorDataset(X_test, y_test)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            outputs = model(batch_x)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    
    return correct / total

def federated_learning(client_data: List[Tuple[torch.Tensor, torch.Tensor]], 
                      test_data: Tuple[torch.Tensor, torch.Tensor],
                      num_rounds: int = 30, 
                      local_epochs: int = 5,
                      learning_rate: float = 0.01,
                      batch_size: int = 32,
                      num_classes: int = 3) -> Dict:
    import os
    os.makedirs("artifacts", exist_ok=True)

    # Initialize global model
    global_model = TinyCNN(num_classes=num_classes).to(device)
    
    # Track accuracy over rounds
    test_accuracies = []
    num_clients = len(client_data)
    
    print(f"Starting Federated Learning with {num_clients} clients for {num_rounds} rounds")
    print("-" * 60)
    
    # Initial evaluation
    initial_acc = evaluate_model(global_model, test_data)
    test_accuracies.append(initial_acc)
    print(f"Round 0: Test Accuracy = {initial_acc:.4f}")
    
    # Training loop
    for round_num in range(num_rounds):
        client_updates = []
        for client_id in range(num_clients):
            params, num_samples = client_update(
                global_model.state_dict(),
                client_data[client_id], 
                epochs=local_epochs,
                lr=learning_rate,
                batch_size=batch_size
            )
            client_updates.append((params, num_samples))
        
        # Aggregate updates
        aggregated_params = federated_averaging(client_updates)
        global_model.load_state_dict(aggregated_params)
        
        # Evaluate
        test_acc = evaluate_model(global_model, test_data)
        test_accuracies.append(test_acc)
        
        # Print progress every 5 rounds
        if (round_num + 1) % 5 == 0:
            print(f"Round {round_num + 1}: Test Accuracy = {test_acc:.4f}")

    print(f"\nFinal Test Accuracy: {test_accuracies[-1]:.4f}")

    # Save global model
    torch.save(global_model.state_dict(), "artifacts/global_model.pt")
    
    # Save per-round accuracies
    with open("artifacts/round_accuracies.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Round", "Accuracy"])
        for i, acc in enumerate(test_accuracies):
            writer.writerow([i, acc])

    # Plot accuracy per round
    plt.figure(figsize=(8,5))
    plt.plot(range(len(test_accuracies)), test_accuracies, marker='o')
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title("FedAvg Per-Round Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("artifacts/fedavg_accuracy_plot.png", dpi=150)
    print("FedAvg accuracy plot saved to artifacts/fedavg_accuracy_plot.png")
    plt.show()

    return {'model': global_model, 'accuracies': test_accuracies}

def main():
    """Main function"""
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
        'batch_size': 32
    }
    
    print("Classical Federated Learning on MNIST")
  
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
    
    # Run federated learning
    try:
        results = federated_learning(
            client_data=client_data,
            test_data=test_data,
            num_rounds=config['num_rounds'],
            local_epochs=config['local_epochs'],
            learning_rate=config['learning_rate'],
            batch_size=config['batch_size'],
            num_classes=len(config['digits'])
        )
        
        print(f"Training completed successfully with final accuracy: {results['accuracies'][-1]:.4f}")
        
    except Exception as e:
        print(f"Error during federated learning training: {e}")
        return

if __name__ == "__main__":
    main()