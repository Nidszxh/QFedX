import numpy as np
import torch
import random
import matplotlib.pyplot as plt

import json
import joblib
from pathlib import Path
from typing import Tuple, List, Optional, Union

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import MinMaxScaler

# Import visualization utilities
try:
    from viz_preprocess import generate_all_preprocessing_visualizations
    VISUALIZATIONS_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: viz_preprocess.py not found. Visualizations will be skipped.")
    VISUALIZATIONS_AVAILABLE = False

def read_idx_images(filename: str) -> np.ndarray:
    """Memory-efficient IDX image reader with proper header parsing."""
    with open(filename, 'rb') as f:
        magic, num_images, rows, cols = np.frombuffer(f.read(16), dtype='>i4')
        if magic != 2051:
            raise ValueError(f"Invalid magic number {magic} for image file")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(num_images, rows, cols)

def read_idx_labels(filename: str) -> np.ndarray:
    """Memory-efficient IDX label reader with proper header parsing."""
    with open(filename, 'rb') as f:
        magic, num_labels = np.frombuffer(f.read(8), dtype='>i4')
        if magic != 2049:
            raise ValueError(f"Invalid magic number {magic} for label file")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data

# Partitioning Logic (IID and Non-IID via Dirichlet)
def create_iid_partition(indices: np.ndarray, num_clients: int, rng: np.random.Generator) -> List[np.ndarray]:
    """Create IID partition using indices (zero-copy until final assignment)."""
    rng.shuffle(indices)
    return np.array_split(indices, num_clients)  # Cleaner than manual slicing

def create_non_iid_partition(y_data: np.ndarray, num_clients: int, alpha: float, rng: np.random.Generator) -> List[np.ndarray]:
    """Optimized non-IID partition using Dirichlet distribution."""
    num_classes = len(np.unique(y_data))
    label_indices = [np.where(y_data == i)[0] for i in range(num_classes)]
    client_indices = [[] for _ in range(num_clients)]
    
    for indices in label_indices:
        rng.shuffle(indices)
        proportions = rng.dirichlet([alpha] * num_clients)
        splits = np.insert(np.cumsum(proportions), 0, 0) * len(indices)
        splits = np.round(splits).astype(int)
        splits[-1] = len(indices)  # Guarantee all samples used
        
        for cid in range(num_clients):
            client_indices[cid].extend(indices[splits[cid]:splits[cid+1]])
    
    return [np.array(idx, dtype=np.int64) for idx in client_indices]

def create_partition(y_data: np.ndarray, num_clients: int, alpha: Optional[float] = None, seed: int = 42) -> List[np.ndarray]:
    """Unified partition API: IID if alpha=None, else Dirichlet non-IID."""
    rng = np.random.default_rng(seed)
    indices = np.arange(len(y_data))
    
    return (create_iid_partition(indices, num_clients, rng) if alpha is None 
            else create_non_iid_partition(y_data, num_clients, alpha, rng))

# Visualization Functions
def visualize_client_data(client_data: List[Tuple], save_path: Optional[str] = None, 
                          samples_per_client: int = 5, is_pca: bool = False):
    """Visualize client data: PCA scatter or image grid."""
    num_clients = len(client_data)
    max_clients_plot = min(num_clients, 8)
    
    if is_pca:
        fig, axes = plt.subplots(1, max_clients_plot, figsize=(max_clients_plot * 4, 4))
        axes = np.atleast_1d(axes)

        for i in range(max_clients_plot):
            X, y = client_data[i]
            ax = axes[i]
            if X.shape[1] >= 2:
                scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='tab10', s=10, alpha=0.7)
                ax.set_title(f"Client {i+1} PCA")
                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")
                plt.colorbar(scatter, ax=ax)
            else:
                ax.text(0.5, 0.5, "Insufficient PCA dims", ha='center', va='center', fontsize=10)
                ax.set_title(f"Client {i+1}")
                ax.axis('off')

        plt.suptitle(f"PCA Feature Distribution ({max_clients_plot}/{num_clients} clients)", fontsize=14)
    
    else:
        fig, axes = plt.subplots(max_clients_plot, samples_per_client, 
                                figsize=(samples_per_client * 2, max_clients_plot * 2))
        axes = np.atleast_2d(axes).reshape(max_clients_plot, samples_per_client)

        for i in range(max_clients_plot):
            X, y = client_data[i]
            for j in range(samples_per_client):
                ax = axes[i, j]
                if j < len(X):
                    img = X[j].reshape(28, 28) if X[j].ndim == 1 else X[j].squeeze()
                    ax.imshow(img, cmap='gray')
                    ax.set_title(f"Label: {y[j]}", fontsize=8)
                    ax.axis('off')
                else:
                    ax.axis('off')
                    
            axes[i, 0].set_ylabel(f"Client {i+1}", fontsize=10, rotation=0, labelpad=30)

        plt.suptitle(f"Sample Images ({max_clients_plot}/{num_clients} clients)", fontsize=14)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Visualization saved: {save_path}")
    plt.close(fig)

def plot_class_distribution(client_data: List[Tuple], save_path: str = "./results/class_distribution.png"):
    """Plot stacked bar chart of class distribution across clients."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    all_classes = sorted(set().union(*[set(y.tolist() if isinstance(y, torch.Tensor) else y) 
                                        for _, y in client_data]))
    
    distributions = []
    for _, y_client in client_data:
        y_np = y_client.numpy() if isinstance(y_client, torch.Tensor) else y_client
        counts = {cls: 0 for cls in all_classes}
        unique, class_counts = np.unique(y_np, return_counts=True)
        counts.update(dict(zip(unique, class_counts)))
        distributions.append([counts[c] for c in all_classes])
    
    distributions = np.array(distributions).T
    colors = plt.cm.Set3(np.linspace(0, 1, len(all_classes)))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(client_data))
    
    for i, cls in enumerate(all_classes):
        ax.bar(range(len(client_data)), distributions[i], bottom=bottom, 
               label=f'Digit {cls}', color=colors[i], alpha=0.85)
        bottom += distributions[i]
    
    ax.set_xticks(range(len(client_data)))
    ax.set_xticklabels([f'Client {i+1}' for i in range(len(client_data))])
    ax.set_xlabel('Clients', fontsize=11)
    ax.set_ylabel('Sample Count', fontsize=11)
    ax.set_title('Class Distribution Across Clients', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"📊 Class distribution plot saved: {save_path}")

# Main Preprocessing Pipeline
def preprocess_mnist(
    raw_folder: str,
    processed_folder: str,
    digits: Tuple[int, ...] = (0, 1, 2),
    val_split: float = 0.1,
    num_clients: int = 4,
    partition_type: str = 'iid',
    alpha: float = 0.5,
    apply_pca: bool = False,
    pca_components: int = 4,
    seed: int = 42,
    generate_plots: bool = True,
    use_incremental_pca: bool = False,
    pca_batch_size: int = 1000
) -> Tuple:
    """
    Quantum Federated Learning MNIST preprocessing pipeline.
    
    Args:
        raw_folder: Path to raw MNIST IDX files
        processed_folder: Output path for processed data
        digits: Tuple of digit classes to include
        val_split: Validation set fraction
        num_clients: Number of federated clients
        partition_type: 'iid' or 'non_iid'
        alpha: Dirichlet concentration (lower = more non-IID)
        apply_pca: Whether to apply PCA dimensionality reduction
        pca_components: Number of PCA components
        seed: Random seed for reproducibility
        generate_plots: Enable visualization generation
        use_incremental_pca: Use IncrementalPCA for large datasets
        pca_batch_size: Batch size for incremental PCA
    
    Returns:
        (train_data, val_data, test_data, client_data_list)
    """
    
    # Reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Directory setup
    for path in [processed_folder, "./results", "./artifacts"]:
        Path(path).mkdir(parents=True, exist_ok=True)
    
    # 1. Load Raw Data    
    print("\n" + "="*70)
    print("Quantum Federated Learning - MNIST Preprocessing")
    print("="*70)
    
    try:
        file_map = {
            'train_images': "train-images.idx3-ubyte",
            'train_labels': "train-labels.idx1-ubyte",
            'test_images': "t10k-images.idx3-ubyte",
            'test_labels': "t10k-labels.idx1-ubyte"
        }
        
        X_train = read_idx_images(Path(raw_folder) / file_map['train_images'])
        y_train = read_idx_labels(Path(raw_folder) / file_map['train_labels'])
        X_test = read_idx_images(Path(raw_folder) / file_map['test_images'])
        y_test = read_idx_labels(Path(raw_folder) / file_map['test_labels'])
        
        print(f"\n✅ Raw data loaded: Train {X_train.shape}, Test {X_test.shape}")
        
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ Error loading raw data: {e}")
        return None
    
    # 2. Filter Digits  
    print(f"\n🔍 Filtering digits {digits}...")
    train_mask = np.isin(y_train, digits)
    test_mask = np.isin(y_test, digits)
    
    X_train_filt = X_train[train_mask]
    y_train_filt = y_train[train_mask]
    X_test_filt = X_test[test_mask]
    y_test_filt = y_test[test_mask]
    
    print(f"   After filtering: Train {X_train_filt.shape}, Test {X_test_filt.shape}")
    
    # 3. Create Partitions 
    print(f"\n📊 Creating {partition_type.upper()} partition for {num_clients} clients...")
    client_indices = create_partition(
        y_train_filt, num_clients,
        alpha if partition_type == 'non_iid' else None, seed
    )
    
    # Validate partitions
    for i, idx in enumerate(client_indices):
        if len(idx) == 0:
            raise ValueError(f"❌ Client {i+1} received no data!")
        if len(np.unique(y_train_filt[idx])) == 1:
            print(f"⚠️  Warning: Client {i+1} has only one class")
    
    print("\n   Client data distribution:")
    for i, idx in enumerate(client_indices):
        y_cli = y_train_filt[idx]
        unique, counts = np.unique(y_cli, return_counts=True)
        dist = ", ".join([f"Digit {u}: {c}" for u, c in zip(unique, counts)])
        print(f"     Client {i+1}: {len(idx):>5} samples  [{dist}]")
    
    # 4. Train/Val Split (Before Transformation)  
    print(f"\n✂️  Splitting train/val ({1-val_split:.0%}/{val_split:.0%})...")
    train_idx, val_idx = train_test_split(
        np.arange(len(y_train_filt)),
        test_size=val_split,
        stratify=y_train_filt,
        random_state=seed
    )
    
    X_train_split = X_train_filt[train_idx]
    y_train_split = y_train_filt[train_idx]
    X_val_split = X_train_filt[val_idx]
    y_val = y_train_filt[val_idx]
    
    # 5. Normalize and Flatten  
    print("\n🔢 Normalizing and flattening to [0,1]...")
    X_train_flat = (X_train_split / 255.0).astype(np.float32).reshape(len(X_train_split), -1)
    X_val_flat = (X_val_split / 255.0).astype(np.float32).reshape(len(X_val_split), -1)
    X_test_flat = (X_test_filt / 255.0).astype(np.float32).reshape(len(X_test_filt), -1)
    
    # Store data before scaling for visualization
    data_before_scaling = X_train_flat.copy()
    
    # 6. Apply PCA (Optional) 
    pca_model = None
    if apply_pca:
        if pca_components > X_train_flat.shape[1]:
            raise ValueError(f"PCA components ({pca_components}) > features ({X_train_flat.shape[1]})")
        
        print(f"\n🧬 Applying PCA: {X_train_flat.shape[1]}D → {pca_components}D")
        
        if use_incremental_pca and len(X_train_flat) > 10000:
            print(f"   Using Incremental PCA (batch_size={pca_batch_size})")
            pca_model = IncrementalPCA(n_components=pca_components, batch_size=pca_batch_size)
            for i in range(0, len(X_train_flat), pca_batch_size):
                pca_model.partial_fit(X_train_flat[i:i+pca_batch_size])
        else:
            pca_model = PCA(n_components=pca_components)
            pca_model.fit(X_train_flat)
        
        X_train_flat = pca_model.transform(X_train_flat)
        X_val_flat = pca_model.transform(X_val_flat)
        X_test_flat = pca_model.transform(X_test_flat)
        
        joblib.dump(pca_model, Path("./artifacts") / "pca_model.pkl")
        print(f"   Explained variance: {pca_model.explained_variance_ratio_.sum():.4f}")
        
        # Update before scaling data for PCA case
        data_before_scaling = X_train_flat.copy()
    
    # 7. Scale to [-1, 1] (Quantum Encoding Range)
    print("\n⚡ Scaling to [-1, 1] for quantum encoding...")
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_flat = scaler.fit_transform(X_train_flat)
    X_val_flat = scaler.transform(X_val_flat)
    X_test_flat = scaler.transform(X_test_flat)
    
    joblib.dump(scaler, Path("./artifacts") / "scaler.pkl")
    
    # Store data after scaling for visualization
    data_after_scaling = X_train_flat.copy()
    
    # 8. Save Global Datasets
    print(f"\n💾 Saving global datasets to {processed_folder}...")
    datasets = {
        'train': (torch.tensor(X_train_flat, dtype=torch.float32),
                  torch.tensor(y_train_split, dtype=torch.long)),
        'val': (torch.tensor(X_val_flat, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.long)),
        'test': (torch.tensor(X_test_flat, dtype=torch.float32),
                 torch.tensor(y_test_filt, dtype=torch.long))
    }
    
    for name, data in datasets.items():
        torch.save(data, Path(processed_folder) / f"{name}.pt")
    
    # 9. Process Client Data (Transform Once, Slice Many)
    print("\n🌐 Processing client partitions...")
    
    # Transform ALL filtered training data once
    X_all_norm = (X_train_filt / 255.0).astype(np.float32).reshape(len(X_train_filt), -1)
    if apply_pca:
        X_all_norm = pca_model.transform(X_all_norm)
    X_all_norm = scaler.transform(X_all_norm)
    
    # Slice for each client
    client_data_processed = []
    client_data_orig = []
    
    for i, idx in enumerate(client_indices):
        X_cli = torch.tensor(X_all_norm[idx], dtype=torch.float32)
        y_cli = torch.tensor(y_train_filt[idx], dtype=torch.long)
        
        client_data_processed.append((X_cli, y_cli))
        torch.save((X_cli, y_cli), Path(processed_folder) / f"client{i+1}.pt")
        
        # For visualization (original images)
        X_cli_orig = (X_train_filt[idx] / 255.0).astype(np.float32)
        client_data_orig.append((X_cli_orig, y_train_filt[idx]))
        print(f"   Client {i+1}: {X_cli.shape}, Labels: {len(torch.unique(y_cli))} classes")
        
    # 10. Print Statistics
    print("\n📈 Global class distribution:")
    for name, labels in [("Train", y_train_split), ("Val", y_val), ("Test", y_test_filt)]:
        unique, counts = np.unique(labels, return_counts=True)
        dist = ", ".join([f"Digit {u}: {c}" for u, c in zip(unique, counts)])
        print(f"   {name:>5}: {dist}")
    
    # 11. Generate Basic Visualizations (Original)
    if generate_plots:
        print("\n📊 Generating basic visualizations...")
        
        visualize_client_data(
            client_data_orig,
            save_path="./results/client_data_images.png",
            is_pca=False
        )
        
        if apply_pca and pca_components >= 2:
            visualize_client_data(
                client_data_processed,
                save_path="./results/client_data_pca.png",
                is_pca=True
            )
        
        plot_class_distribution(client_data_orig)
    
    # 12. Save Metadata   
    metadata = {
        'digits': list(digits),
        'num_clients': num_clients,
        'partition_type': partition_type,
        'alpha': alpha if partition_type == 'non_iid' else None,
        'apply_pca': apply_pca,
        'pca_components': pca_components if apply_pca else None,
        'use_incremental_pca': use_incremental_pca if apply_pca else None,
        'val_split': val_split,
        'seed': seed,
        'feature_dim': X_train_flat.shape[1],
        'samples': {
            'train': len(y_train_split),
            'val': len(y_val),
            'test': len(y_test_filt)
        }
    }
    
    with open(Path("./artifacts") / "preprocessing_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # 13. Generate Advanced Visualizations (from viz_preprocess.py)
    if generate_plots and VISUALIZATIONS_AVAILABLE:
        try:
            generate_all_preprocessing_visualizations(
                pca_model=pca_model if apply_pca else None,
                data_before_scaling=data_before_scaling,
                data_after_scaling=data_after_scaling,
                client_indices=client_indices,
                client_data=client_data_processed,
                y_train=y_train_split,
                y_val=y_val,
                metadata=metadata,
                num_classes=len(digits),
                save_dir="./results/preprocessing"
            )
        except Exception as e:
            print(f"\n⚠️  Warning: Could not generate advanced visualizations: {e}")
    
    print("\n" + "="*70)
    print("✅ Quantum FL Preprocessing Complete!")
    print("="*70)
    print("\n📋 Summary:")
    print(f"   Dimensionality: 784D → {X_train_flat.shape[1]}D")
    print(f"   Method: {'PCA + MinMax Scaling' if apply_pca else 'MinMax Scaling Only'}")
    print(f"   Processed data: {processed_folder}")
    print(f"   Artifacts: ./artifacts/")
    print(f"   Visualizations: ./results/preprocessing/")
    print("="*70 + "\n")
    
    return datasets['train'], datasets['val'], datasets['test'], client_data_processed

if __name__ == "__main__":
    preprocess_mnist(
        raw_folder="./dataset/raw",
        processed_folder="./dataset/processed",
        digits=(0, 1, 2),
        val_split=0.1,
        num_clients=4,
        partition_type='iid',  # 'iid' or 'non_iid'
        alpha=0.5,             # Lower = more non-IID (e.g., 0.1)
        apply_pca=True,
        pca_components=4,
        seed=42,
        generate_plots=True,   # Disable for HPC batch jobs
        use_incremental_pca=False,
        pca_batch_size=1000
    )