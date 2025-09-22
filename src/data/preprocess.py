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
import json

def read_idx_images(filename: str) -> np.ndarray:
    with open(filename, 'rb') as f:
        f.read(16)  
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 28, 28)

def read_idx_labels(filename: str) -> np.ndarray:
    with open(filename, 'rb') as f:
        f.read(8)  
        return np.frombuffer(f.read(), dtype=np.uint8)

def create_iid_partition(X_data, y_data, num_clients, rng):
    # Create IID partition of data across clients.
    indices = np.arange(len(X_data))
    rng.shuffle(indices)
    
    num_items = len(X_data) // num_clients
    client_data = []
    
    for i in range(num_clients):
        start_idx = i * num_items
        end_idx = len(X_data) if i == num_clients - 1 else (i + 1) * num_items
        client_indices = indices[start_idx:end_idx]
        client_data.append((X_data[client_indices], y_data[client_indices]))
    
    return client_data

def create_non_iid_partition(X_data, y_data, num_clients, alpha, rng):
    # Create non-IID partition using Dirichlet distribution.
    num_classes = len(np.unique(y_data))
    
    class_indices = defaultdict(list)
    for idx, label in enumerate(y_data):
        class_indices[label].append(idx)
    
    client_indices = [np.zeros(len(y_data), dtype=int) for _ in range(num_clients)]
    
    for class_label in range(num_classes):
        class_data = np.array(class_indices[class_label])
        rng.shuffle(class_data)
        
        proportions = rng.dirichlet([alpha] * num_clients)
        
        start_idx = 0
        for client_id in range(num_clients):
            end_idx = start_idx + int(len(class_data) * proportions[client_id])
            if client_id == num_clients - 1:
                end_idx = len(class_data)
            client_indices[client_id].extend(class_data[start_idx:end_idx])
            start_idx = end_idx
    
    return [(X_data[indices], y_data[indices]) for indices in client_indices]

def create_partition(X, y, num_clients, alpha=None, seed=42):
    # Unified partition API with proper RNG handling.
    rng = np.random.default_rng(seed)
    
    if alpha is None:
        return create_iid_partition(X, y, num_clients, rng) 
    else:
        return create_non_iid_partition(X, y, num_clients, alpha, rng)
    
def visualize_client_data(client_data, save_path=None, samples_per_client=5, is_pca=False):
    # Visualize client data with proper handling for both PCA and image data
    num_clients = len(client_data)
    max_clients_plot = min(num_clients, 8)
    
    if is_pca:
        # Scatter plot for PCA features
        fig, axes = plt.subplots(1, max_clients_plot, figsize=(max_clients_plot * 4, 4))
        
        # Handle single client case properly
        if max_clients_plot == 1:
            axes = [axes]

        for i in range(max_clients_plot):
            X, y = client_data[i]
            ax = axes[i]
            if X.shape[1] >= 2:
                scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='tab10', s=10, alpha=0.7)
                ax.set_title(f"Client {i+1} PCA Features")
                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")
                plt.colorbar(scatter, ax=ax)
            else:
                ax.text(0.5, 0.5, "PCA Dim < 2", ha='center', va='center')
                ax.set_title(f"Client {i+1}")
                ax.axis('off')

        plt.suptitle(f"PCA Features Distribution (showing {max_clients_plot}/{num_clients} clients)", fontsize=14)
        plt.tight_layout()

    else:
        # Original MNIST image visualization
        fig, axes = plt.subplots(max_clients_plot, samples_per_client, 
                               figsize=(samples_per_client * 2, max_clients_plot * 2))

        # Critical fix: properly handle axes for different cases
        axes = np.atleast_2d(axes)
        if axes.shape[0] == 1 and max_clients_plot > 1:
            axes = axes.T  # Transpose if needed
        elif max_clients_plot == 1 and samples_per_client == 1:
            axes = axes.reshape(1, 1)
        elif max_clients_plot == 1:
            axes = axes.reshape(1, -1)

        for i in range(max_clients_plot):
            X, y = client_data[i]
            for j in range(samples_per_client):
                ax = axes[i, j]
                if j < len(X):
                    # Handle different input shapes properly
                    image = X[j]
                    if image.ndim == 1:  # Flattened
                        image = image.reshape(28, 28)
                    elif image.ndim == 3 and image.shape[0] == 1:  # (1, 28, 28)
                        image = image[0]
                    # else assume it's already (28, 28)
                    
                    ax.imshow(image, cmap='gray')
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title(f"Label: {y[j]}", fontsize=8)
                else:
                    ax.axis('off')
                    
            # Add client label to first column
            axes[i, 0].set_ylabel(f"Client {i+1}", fontsize=10)

        plt.suptitle(f"Sample Images from Clients (showing {max_clients_plot}/{num_clients})", fontsize=14)
        plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
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
    plt.close(fig)
    print(f"Class distribution plot saved to {save_path}")

def preprocess_mnist(raw_folder: str, processed_folder: str, digits=(0,1,2), 
                     val_split=0.1, num_clients=4, partition_type='iid', alpha=0.5,
                     apply_pca=False, pca_components=4, seed=42):

    # Set random seeds consistently  
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Create directories
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
        
        X_train = read_idx_images(Path(raw_folder) / files['train_images'])
        y_train = read_idx_labels(Path(raw_folder) / files['train_labels'])
        X_test = read_idx_images(Path(raw_folder) / files['test_images'])
        y_test = read_idx_labels(Path(raw_folder) / files['test_labels'])
        
        print(f"\nRaw data loaded: Train {X_train.shape}, Test {X_test.shape}")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading raw data: {e}")
        return None
    
    print(f"Filtering digits {digits} and normalizing...")
    train_mask = np.isin(y_train, digits)
    test_mask = np.isin(y_test, digits)

    # FIXED: Keeping original images (28, 28) for partitioning
    X_train_orig = X_train[train_mask].astype(np.float32) / 255.0
    y_train = y_train[train_mask]
    X_test_orig = X_test[test_mask].astype(np.float32) / 255.0  
    y_test = y_test[test_mask]
    
    print(f"After filtering: Train {X_train_orig.shape}, Test {X_test_orig.shape}")
    
    # CRITICAL: Partition on original images (not flattened) for visualization
    print(f"\nCreating {partition_type} partition across {num_clients} clients...")
    client_data_orig = create_partition(
        X_train_orig, y_train, num_clients, 
        alpha if partition_type == 'non_iid' else None, seed
    )
    
    print("Client data distribution (original images):")
    for i, (X_client, y_client) in enumerate(client_data_orig):
        unique, counts = np.unique(y_client, return_counts=True)
        dist_str = ", ".join([f"Digit {u}: {c}" for u, c in zip(unique, counts)])
        print(f"  Client {i+1}: {X_client.shape} ({dist_str})")

    # Now process for quantum ML pipeline (flatten → PCA → scale)
    X_train_flat = X_train_orig.reshape(len(X_train_orig), -1)
    X_test_flat = X_test_orig.reshape(len(X_test_orig), -1)

    # Create validation split
    X_train_flat, X_val_flat, y_train_split, y_val = train_test_split(
        X_train_flat, y_train, test_size=val_split, stratify=y_train, random_state=seed
    )

    # Validate PCA components
    if apply_pca:
        if pca_components > X_train_flat.shape[1]:
            raise ValueError(f"PCA components ({pca_components}) cannot exceed feature dimensions ({X_train_flat.shape[1]})")
        
        print(f"\nApplying PCA: {X_train_flat.shape[1]} → {pca_components} components for quantum qubits")
        pca = PCA(n_components=pca_components)
        X_train_flat = pca.fit_transform(X_train_flat)
        X_val_flat = pca.transform(X_val_flat)
        X_test_flat = pca.transform(X_test_flat)
        
        joblib.dump(pca, Path("./artifacts") / "pca_k.pkl")
        print(f"PCA model saved (explained variance ratio: {pca.explained_variance_ratio_})")

    # Scale for quantum encoding
    print("\nApplying MinMax scaling to [-1, 1] for quantum encoding...")
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_flat = scaler.fit_transform(X_train_flat)
    X_val_flat = scaler.transform(X_val_flat)
    X_test_flat = scaler.transform(X_test_flat)
    
    joblib.dump(scaler, Path("./artifacts") / "scaler.pkl")

    # Save global datasets
    datasets = {
        'train': (torch.tensor(X_train_flat, dtype=torch.float32), torch.tensor(y_train_split, dtype=torch.long)),
        'val': (torch.tensor(X_val_flat, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)),
        'test': (torch.tensor(X_test_flat, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    }

    for name, data in datasets.items():
        torch.save(data, Path(processed_folder) / f"{name}.pt")

    # Process client partitions with same transformations
    client_data_processed = []
    print("\nProcessing client partitions with PCA/scaling...")
    
    for i, (X_client_orig, y_client) in enumerate(client_data_orig):
        # Apply same pipeline: flatten → PCA → scale
        X_client_flat = X_client_orig.reshape(len(X_client_orig), -1)
        
        if apply_pca:
            X_client_flat = pca.transform(X_client_flat)
        
        X_client_flat = scaler.transform(X_client_flat)

        # --- NEW: convert to PyTorch tensors here so code can call .to(device) ---
        X_client_tensor = torch.tensor(X_client_flat, dtype=torch.float32)
        y_client_tensor = torch.tensor(y_client, dtype=torch.long)

        client_data_processed.append((X_client_tensor, y_client_tensor))

        torch.save((X_client_tensor, y_client_tensor),
                   Path(processed_folder) / f"client{i+1}.pt")

    # Print statistics
    print(f"\nGlobal datasets saved to {processed_folder}")
    splits_data = [("Train", y_train_split), ("Val", y_val), ("Test", y_test)]
    print("Global class distribution:")
    for split_name, y_split in splits_data:
        unique, counts = np.unique(y_split, return_counts=True)
        dist_str = ", ".join([f"Digit {u}: {c}" for u, c in zip(unique, counts)])
        print(f"  {split_name}: {dist_str}")

    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Always visualize original images (meaningful)
    visualize_client_data(
        client_data_orig,
        save_path=Path("results") / "client_data_images.png",
        is_pca=False  # FIXED: Use False for original images
    )
    
    # Visualize PCA features only if PCA was applied and has enough dimensions
    if apply_pca and pca_components >= 2:
        visualize_client_data(
            client_data_processed,
            save_path=Path("results") / "client_data_pca.png", 
            is_pca=apply_pca  # FIXED: Use the actual boolean, not a condition
        )

    plot_class_distribution(client_data_orig)
    
    # Save metadata
    metadata = {
        'digits': digits,
        'num_clients': num_clients,
        'partition_type': partition_type,
        'alpha': alpha if partition_type == 'non_iid' else None,
        'apply_pca': apply_pca,
        'pca_components': pca_components if apply_pca else None,
        'val_split': val_split,
        'seed': seed,
        'feature_dim_final': X_train_flat.shape[1],
        'total_samples': {'train': len(y_train_split), 'val': len(y_val), 'test': len(y_test)}
    }
    
    with open(Path("artifacts") / "preprocessing_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nQuantum FL preprocessing completed!")
    print(f"Original: 784D → Final: {X_train_flat.shape[1]}D ({'PCA+Scaling' if apply_pca else 'Scaling only'})")
    print(f"Files saved in: {processed_folder}")
    print(f"Artifacts saved in: ./artifacts/")
    
    return datasets['train'], datasets['val'], datasets['test'], client_data_processed

if __name__ == "__main__":
    preprocess_mnist(
        raw_folder="./dataset/raw",
        processed_folder="./dataset/processed",
        digits=(0, 1, 2),
        val_split=0.1,
        num_clients=4,
        partition_type='iid',  # or 'non_iid'
        alpha=0.5,         # Lower values = more non-IID (e.g., 0.1)
        apply_pca=True,   
        pca_components=4,  # Number of qubits for feature encoding
        seed=42
    )