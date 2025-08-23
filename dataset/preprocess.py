import os
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split


def read_idx_images(filename: str) -> np.ndarray:
    with open(filename, 'rb') as f:
        f.read(16)
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 28, 28)


def read_idx_labels(filename: str) -> np.ndarray:
    with open(filename, 'rb') as f:
        f.read(8)
        return np.frombuffer(f.read(), dtype=np.uint8)


def create_iid_partition(X_data, y_data, num_clients):
    # Create IID partitions of the dataset for federated learning
    indices = list(range(len(X_data)))
    random.shuffle(indices)
    
    num_items = len(X_data) // num_clients
    client_data = []
    
    for i in range(num_clients):
        start_idx = i * num_items
        end_idx = len(X_data) if i == num_clients - 1 else (i + 1) * num_items
        client_indices = indices[start_idx:end_idx]
        client_data.append((X_data[client_indices], y_data[client_indices]))
    
    return client_data


def create_non_iid_partition(X_data, y_data, num_clients, alpha=0.5):
    # Create non-IID partitions using Dirichlet distribution
    num_classes = len(np.unique(y_data))
    
    # Group data by class
    class_indices = defaultdict(list)
    for idx, label in enumerate(y_data):
        class_indices[label].append(idx)
    
    client_indices = [[] for _ in range(num_clients)]
    
    for class_label in range(num_classes):
        class_data = class_indices[class_label]
        random.shuffle(class_data)
        
        # Sample proportions from Dirichlet distribution
        proportions = np.random.dirichlet([alpha] * num_clients)
        
        # Distribute data according to proportions
        start_idx = 0
        for client_id in range(num_clients):
            end_idx = start_idx + int(len(class_data) * proportions[client_id])
            if client_id == num_clients - 1:  # Last client gets remaining data
                end_idx = len(class_data)
            client_indices[client_id].extend(class_data[start_idx:end_idx])
            start_idx = end_idx
    
    # Create client datasets
    return [(X_data[indices], y_data[indices]) for indices in client_indices]


def visualize_client_data(client_data, save_path="./results/client_samples.png", samples_per_client=5):
    # Visualize sample images from each client
    num_clients = len(client_data)
    fig, axes = plt.subplots(num_clients, samples_per_client, figsize=(12, 2.5*num_clients))
    
    # Handle single client or single sample cases
    axes = np.atleast_2d(axes)
    if num_clients == 1:
        axes = axes[None, :]

    for i, (X_client, y_client) in enumerate(client_data):
        for j in range(samples_per_client):
            ax = axes[i, j]
            if j < len(X_client):
                image = X_client[j, 0] if X_client.ndim == 4 else X_client[j]
                ax.imshow(image, cmap='gray')
                ax.set_title(f'Label: {y_client[j]}')
            ax.axis('off')
    
    plt.suptitle("Sample Images from Each Client", fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nClient sample visualization saved to {save_path}")


def plot_class_distribution(client_data, save_path="./results/class_distribution.png"):
    # Plot class distribution across clients
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Get unique classes from all client data
    all_classes = set()
    for _, y_client in client_data:
        all_classes.update(y_client)
    classes = sorted(all_classes)
    
    # Create distribution matrix
    distributions = []
    for _, y_client in client_data:
        unique, counts = np.unique(y_client, return_counts=True)
        dist = {class_id: 0 for class_id in classes}
        for class_id, count in zip(unique, counts):
            dist[class_id] = count
        distributions.append([dist[c] for c in classes])
    
    # Create stacked bar chart
    distributions = np.array(distributions).T  # Shape: (num_classes, num_clients)
    client_names = [f'Client {i+1}' for i in range(len(client_data))]
    colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(client_data))
    
    for i, class_id in enumerate(classes):
        ax.bar(client_names, distributions[i], bottom=bottom, 
               label=f'Digit {class_id}', color=colors[i], alpha=0.8)
        bottom += distributions[i]
    
    ax.set_xlabel('Clients')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Class Distribution Across Clients')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Class distribution plot saved to {save_path}")


def preprocess_mnist(raw_folder: str, processed_folder: str, digits=(0,1,2), 
                    val_split=0.1, num_clients=4, partition_type='iid', alpha=0.5):
    """
    Preprocess MNIST dataset for federated learning
    
    Args:
        raw_folder: Path to raw MNIST .idx files
        processed_folder: Path to save processed datasets
        digits: Tuple of digits to include (default: (0,1,2))
        val_split: Validation split ratio
        num_clients: Number of federated learning clients
        partition_type: 'iid' or 'non_iid'
        alpha: Dirichlet parameter for non-IID partitioning
    """
    # Create directories
    Path(processed_folder).mkdir(parents=True, exist_ok=True)
    Path("./results").mkdir(parents=True, exist_ok=True)
        
    # Load raw data
    try:
        files = {
            'train_images': "train-images.idx3-ubyte",
            'train_labels': "train-labels.idx1-ubyte", 
            'test_images': "t10k-images.idx3-ubyte",
            'test_labels': "t10k-labels.idx1-ubyte"
        }
        
        X_train = read_idx_images(os.path.join(raw_folder, files['train_images']))
        y_train = read_idx_labels(os.path.join(raw_folder, files['train_labels']))
        X_test = read_idx_images(os.path.join(raw_folder, files['test_images']))
        y_test = read_idx_labels(os.path.join(raw_folder, files['test_labels']))
        
        print(f"\nRaw data loaded: Train {X_train.shape}, Test {X_test.shape}")
    except FileNotFoundError as e:
        print(f"Error loading raw data: {e}")
        return None
    
    # Filter digits and normalize in one step
    print(f"Filter digits {digits} and Normalize")
    train_mask = np.isin(y_train, digits)
    test_mask = np.isin(y_test, digits)
    
    X_train = (X_train[train_mask].astype(np.float32) / 255.0)[:, None, :, :]
    y_train = y_train[train_mask]
    X_test = (X_test[test_mask].astype(np.float32) / 255.0)[:, None, :, :]
    y_test = y_test[test_mask]
    
    print(f"\nAfter processing: Train {X_train.shape}, Test {X_test.shape}")
    
    # Create validation split and convert to tensors
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_split, stratify=y_train, random_state=42
    )
    
    # Convert to tensors and save
    datasets = {
        'train': (torch.tensor(X_train), torch.tensor(y_train, dtype=torch.long)),
        'val': (torch.tensor(X_val), torch.tensor(y_val, dtype=torch.long)),
        'test': (torch.tensor(X_test), torch.tensor(y_test, dtype=torch.long))
    }
    
    for name, data in datasets.items():
        torch.save(data, os.path.join(processed_folder, f"{name}.pt"))
    
    print(f"Datasets saved to {processed_folder}")
    
    # Print statistics
    splits_data = [("Train", y_train), ("Val", y_val), ("Test", y_test)]
    print("\nClass distribution:")
    for split_name, y_split in splits_data:
        unique, counts = np.unique(y_split, return_counts=True)
        dist_str = ", ".join([f"Digit {u}: {c}" for u, c in zip(unique, counts)])
        print(f"  {split_name}: {dist_str}")
    
    # Create client partitions
    print(f"\nCreating {partition_type.upper()} client partitions:")
    partition_func = create_iid_partition if partition_type == 'iid' else create_non_iid_partition
    client_data = partition_func(X_train, y_train, num_clients, alpha) if partition_type == 'non_iid' else partition_func(X_train, y_train, num_clients)
    
    # Print client statistics and create visualizations
    print("\nClient data distribution:")
    for i, (X_client, y_client) in enumerate(client_data):
        unique, counts = np.unique(y_client, return_counts=True)
        dist_str = ", ".join([f"Digit {u}: {c}" for u, c in zip(unique, counts)])
        print(f"  Client {i+1}: {X_client.shape[0]} samples ({dist_str})")
    
    visualize_client_data(client_data)
    plot_class_distribution(client_data)
    
    print(f"\nPreprocessing completed! Files saved in: {processed_folder}")
    
    return datasets['train'], datasets['val'], datasets['test'], client_data


if __name__ == "__main__":
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    # Run preprocessing
    preprocess_mnist(
        raw_folder="./dataset/raw",
        processed_folder="./dataset/processed",
        digits=(0, 1, 2),
        val_split=0.1,
        num_clients=4,
        partition_type='iid',  # or 'non_iid'
        alpha=0.5
    )