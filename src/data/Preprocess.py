import os
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import joblib

def read_idx_images(filename: str) -> np.ndarray:
    with open(filename, 'rb') as f:
        f.read(16)
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 28, 28)

def read_idx_labels(filename: str) -> np.ndarray:
    with open(filename, 'rb') as f:
        f.read(8)
        return np.frombuffer(f.read(), dtype=np.uint8)

def create_iid_partition(X_data, y_data, num_clients):
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
    num_classes = len(np.unique(y_data))
    
    class_indices = defaultdict(list)
    for idx, label in enumerate(y_data):
        class_indices[label].append(idx)
    
    client_indices = [[] for _ in range(num_clients)]
    
    for class_label in range(num_classes):
        class_data = class_indices[class_label]
        random.shuffle(class_data)
        
        proportions = np.random.dirichlet([alpha] * num_clients)
        
        start_idx = 0
        for client_id in range(num_clients):
            end_idx = start_idx + int(len(class_data) * proportions[client_id])
            if client_id == num_clients - 1:
                end_idx = len(class_data)
            client_indices[client_id].extend(class_data[start_idx:end_idx])
            start_idx = end_idx
    
    return [(X_data[indices], y_data[indices]) for indices in client_indices]

# Unified partition API
def create_partition(X, y, num_clients, alpha=None):
    if alpha is None:
        return create_iid_partition(X, y, num_clients)
    else:
        return create_non_iid_partition(X, y, num_clients, alpha)

# Visualization fix for empty subplots
def visualize_client_data(client_data, save_path=None, samples_per_client=5, is_pca=False):
    if is_pca:
        # Scatter plot for PCA features
        num_clients = len(client_data)
        fig, axes = plt.subplots(1, num_clients, figsize=(num_clients * 4, 4))

        if num_clients == 1:
            axes = [axes]

        for i, (X, y) in enumerate(client_data):
            ax = axes[i]
            if X.shape[1] >= 2:
                ax.scatter(X[:, 0], X[:, 1], c=y, cmap='tab10', s=10, alpha=0.7)
                ax.set_title(f"Client {i} PCA Features")
                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")
            else:
                ax.text(0.5, 0.5, "PCA Dim < 2", ha='center', va='center')
                ax.axis('off')

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        plt.close(fig)

    else:
        # Original MNIST image visualization
        num_clients = len(client_data)
        fig, axes = plt.subplots(num_clients, samples_per_client, figsize=(samples_per_client * 2, num_clients * 2))

        if num_clients == 1:
            axes = [axes]

        for i, (X, y) in enumerate(client_data):
            for j in range(samples_per_client):
                ax = axes[i][j] if num_clients > 1 else axes[j]
                if j < len(X):
                    image = X[j].reshape(28, 28)
                    ax.imshow(image, cmap='gray')
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title(f"Label: {y[j]}", fontsize=8)
                else:
                    ax.axis('off')
            axes[i][0].set_ylabel(f"Client {i}", fontsize=10)

        plt.suptitle("Sample Images from Each Client", fontsize=14)
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        plt.close(fig)

def plot_class_distribution(client_data, save_path="./results/class_distribution.png"):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    all_classes = set()
    for _, y_client in client_data:
        all_classes.update(y_client)
    classes = sorted(all_classes)
    
    distributions = []
    for _, y_client in client_data:
        unique, counts = np.unique(y_client, return_counts=True)
        dist = {class_id: 0 for class_id in classes}
        for class_id, count in zip(unique, counts):
            dist[class_id] = count
        distributions.append([dist[c] for c in classes])
    
    distributions = np.array(distributions).T
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
                     val_split=0.1, num_clients=4, partition_type='iid', alpha=0.5,
                     apply_pca=False, pca_components=4):
    Path(processed_folder).mkdir(parents=True, exist_ok=True)
    Path("./results").mkdir(parents=True, exist_ok=True)
    Path("./artifacts").mkdir(parents=True, exist_ok=True)
        
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
    
    print(f"Filter digits {digits} and Normalize")
    train_mask = np.isin(y_train, digits)
    test_mask = np.isin(y_test, digits)
    
    X_train = (X_train[train_mask].astype(np.float32) / 255.0)[:, None, :, :]
    y_train = y_train[train_mask]
    X_test = (X_test[test_mask].astype(np.float32) / 255.0)[:, None, :, :]
    y_test = y_test[test_mask]
    
    print(f"\nAfter processing: Train {X_train.shape}, Test {X_test.shape}")
    
    # Flatten for PCA & Scaling
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_test_flat = X_test.reshape(len(X_test), -1)

    # Create validation split
    X_train_flat, X_val_flat, y_train, y_val = train_test_split(
        X_train_flat, y_train, test_size=val_split, stratify=y_train, random_state=42
    )

    # Optional PCA
    if apply_pca:
        print(f"\nApplying PCA with {pca_components} components...")
        pca = PCA(n_components=pca_components)
        X_train_flat = pca.fit_transform(X_train_flat)
        X_val_flat = pca.transform(X_val_flat)
        X_test_flat = pca.transform(X_test_flat)
        joblib.dump(pca, "./artifacts/pca_k.pkl")
        print("PCA model saved to artifacts/pca_k.pkl")

    # Scaling for Quantum Encoding
    print("\nApplying MinMax scaling to [-1, 1] for quantum encoding...")
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_flat = scaler.fit_transform(X_train_flat)
    X_val_flat = scaler.transform(X_val_flat)
    X_test_flat = scaler.transform(X_test_flat)
    joblib.dump(scaler, "./artifacts/scaler.pkl")
    print("Scaler saved to artifacts/scaler.pkl")

    # Convert back to tensors
    datasets = {
        'train': (torch.tensor(X_train_flat, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
        'val': (torch.tensor(X_val_flat, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)),
        'test': (torch.tensor(X_test_flat, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    }

    for name, data in datasets.items():
        torch.save(data, os.path.join(processed_folder, f"{name}.pt"))
    print(f"Datasets saved to {processed_folder}")

    # Stats
    splits_data = [("Train", y_train), ("Val", y_val), ("Test", y_test)]
    print("\nClass distribution:")
    for split_name, y_split in splits_data:
        unique, counts = np.unique(y_split, return_counts=True)
        dist_str = ", ".join([f"Digit {u}: {c}" for u, c in zip(unique, counts)])
        print(f"  {split_name}: {dist_str}")

    # Partitioning
    client_data = create_partition(
        X_train_flat, y_train, num_clients, alpha if partition_type == 'non_iid' else None
    )

    print("\nClient data distribution:")
    for i, (X_client, y_client) in enumerate(client_data):
        unique, counts = np.unique(y_client, return_counts=True)
        dist_str = ", ".join([f"Digit {u}: {c}" for u, c in zip(unique, counts)])
        print(f"  Client {i+1}: {X_client.shape[0]} samples ({dist_str})")

        # Save each client partition
        client_file = os.path.join(processed_folder, f"client{i+1}.pt")
        torch.save((torch.tensor(X_client, dtype=torch.float32), torch.tensor(y_client, dtype=torch.long)), client_file)
        print(f"Client {i+1} data saved to {client_file}")

    visualize_client_data(
    client_data,
    save_path=os.path.join("results", "client_data_visualization.png"),
    is_pca=(pca_components is not None)
    )

    plot_class_distribution(client_data)
    
    print(f"\nPreprocessing completed! Files saved in: {processed_folder}")
    
    return datasets['train'], datasets['val'], datasets['test'], client_data

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    preprocess_mnist(
        raw_folder="./dataset/raw",
        processed_folder="./dataset/processed",
        digits=(0, 1, 2),
        val_split=0.1,
        num_clients=4,
        partition_type='iid',  # or 'non_iid'
        alpha=0.5,
        apply_pca=True,  # Enable PCA
        pca_components=4
    )
