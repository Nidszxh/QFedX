import numpy as np
import torch
import random
import platform
import hashlib
import matplotlib.pyplot as plt

import json
import joblib
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import os

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import MinMaxScaler

# Import visualization utilities
try:
    from viz_preprocess import generate_all_preprocessing_visualizations
    VISUALIZATIONS_AVAILABLE = True
except ImportError:
    print("Warning: viz_preprocess.py not found. Advanced visualizations will be skipped.")
    VISUALIZATIONS_AVAILABLE = False


# DATA I/O - Memory-Efficient Binary Reading

def read_idx_images(filename: str) -> np.ndarray:
    with open(filename, 'rb') as f:
        magic, num_images, rows, cols = np.frombuffer(f.read(16), dtype='>i4')
        if magic != 2051:
            raise ValueError(f"Invalid magic number {magic} for image file (expected 2051)")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(num_images, rows, cols)

def read_idx_labels(filename: str) -> np.ndarray:
    with open(filename, 'rb') as f:
        magic, num_labels = np.frombuffer(f.read(8), dtype='>i4')
        if magic != 2049:
            raise ValueError(f"Invalid magic number {magic} for label file (expected 2049)")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data


# PARTITIONING - IID and Non-IID via Dirichlet

def create_iid_partition(indices: np.ndarray, num_clients: int, rng: np.random.Generator) -> List[np.ndarray]:
    rng.shuffle(indices)
    return np.array_split(indices, num_clients)

def create_non_iid_partition(y_data: np.ndarray, num_clients: int, alpha: float, 
    rng: np.random.Generator) -> List[np.ndarray]:
    """
    Create non-IID partition using Dirichlet distribution.
    
    Lower alpha values create more heterogeneous (non-IID) distributions.
    Typical values: alpha ∈ [0.1, 10.0]
    - alpha < 1.0: High heterogeneity (recommended: 0.1-0.5 for research)
    - alpha = 1.0: Moderate heterogeneity
    - alpha > 1.0: Approaching IID
    """
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

def create_partition(y_data: np.ndarray, num_clients: int, alpha: Optional[float] = None, 
    seed: int = 42) -> List[np.ndarray]:
    """ Unified partition API: IID if alpha=None, else Dirichlet non-IID. """
    rng = np.random.default_rng(seed)
    indices = np.arange(len(y_data))
    
    return (create_iid_partition(indices, num_clients, rng) if alpha is None 
            else create_non_iid_partition(y_data, num_clients, alpha, rng))


# VALIDATION - Data Quality Checks

def validate_client_partitions(client_indices: List[np.ndarray], y_data: np.ndarray, num_classes: int,
    partition_type: str) -> None:
    for i, idx in enumerate(client_indices):
        # Check for empty clients
        if len(idx) == 0:
            raise ValueError(f"❌ Client {i+1} received no data samples!")
        
        # Check for single-class clients (causes gradient errors)
        unique_classes = np.unique(y_data[idx])
        if len(unique_classes) == 1:
            raise ValueError(
                f"❌ Client {i+1} has only one class (label {unique_classes[0]}). "
                f"This will cause training instability.\n"
                f"   Solutions: Reduce alpha (non-IID) or increase samples per client."
            )
        
        # Warn about severe imbalance
        if partition_type == 'non_iid':
            class_counts = np.bincount(y_data[idx], minlength=num_classes)
            max_ratio = class_counts.max() / (class_counts[class_counts > 0].mean())
            if max_ratio > 10.0:
                print(f"   ⚠️  Client {i+1}: Extreme imbalance (ratio {max_ratio:.1f}:1)")
                print(f"       Consider using weighted loss or client sampling strategies")


def compute_data_hash(X: np.ndarray, y: np.ndarray) -> str:
    # Compute deterministic hash of processed data for reproducibility verification.
    combined = np.concatenate([X.flatten(), y.flatten()])
    return hashlib.sha256(combined.tobytes()).hexdigest()


# PCA OPTIMIZATION

def compute_optimal_pca_batch_size(n_samples: int, n_features: int, n_components: int) -> int:
    # Compute batch size balancing memory constraints and convergence.
    # Memory limit (1GB for batch, leave 50% overhead)
    max_batch_memory = 1e9
    max_batch_size = int(max_batch_memory / (n_features * 8))
    
    # Convergence constraint (rule of thumb: ≥10x components)
    min_batch_size = max(10 * n_components, 100)
    
    # Adaptive: larger batches for larger datasets
    adaptive_batch = max(n_samples // 50, min_batch_size)
    
    return min(max_batch_size, max(adaptive_batch, min_batch_size))

def apply_pca_transform(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray, n_components: int, use_incremental: bool = False,
    batch_size: Optional[int] = None, variance_threshold: float = 0.80) -> Tuple[np.ndarray, np.ndarray, np.ndarray, PCA]:

    if n_components > X_train.shape[1]:
        raise ValueError(
            f"PCA components ({n_components}) exceeds feature dimension ({X_train.shape[1]})"
        )
    
    print(f"\n🔄 Applying PCA: {X_train.shape[1]}D → {n_components}D")
    
    # Select PCA implementation
    if use_incremental and len(X_train) > 10000:
        if batch_size is None:
            batch_size = compute_optimal_pca_batch_size(
                len(X_train), X_train.shape[1], n_components
            )
        
        print(f"   Using Incremental PCA (batch_size={batch_size})")
        pca_model = IncrementalPCA(n_components=n_components, batch_size=batch_size)
        
        for i in range(0, len(X_train), batch_size):
            pca_model.partial_fit(X_train[i:i+batch_size])
    else:
        pca_model = PCA(n_components=n_components)
        pca_model.fit(X_train)
    
    # Validate explained variance
    explained_var = pca_model.explained_variance_ratio_.sum()
    print(f"   Explained variance: {explained_var:.4f}")
    
    if explained_var < variance_threshold:
        print(f"   ⚠️  Warning: Low variance retained ({explained_var:.2%})")
        
        # Suggest optimal components
        cumsum_var = np.cumsum(pca_model.explained_variance_ratio_)
        optimal_k = np.argmax(cumsum_var >= 0.90) + 1
        print(f"   Recommendation: Use {optimal_k} components for 90% variance")
    
    # Transform all datasets
    X_train_pca = pca_model.transform(X_train)
    X_val_pca = pca_model.transform(X_val)
    X_test_pca = pca_model.transform(X_test)
    
    return X_train_pca, X_val_pca, X_test_pca, pca_model


# VISUALIZATION

def visualize_client_data(client_data: List[Tuple], save_path: Optional[str] = None, samples_per_client: int = 5, is_pca: bool = False):
    # Visualize client data: PCA scatter or image grid.
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
                ax.text(0.5, 0.5, "Insufficient PCA dims", ha='center', va='center')
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
        print(f"   📊 Visualization saved: {save_path}")
    plt.close(fig)


def plot_class_distribution(client_data: List[Tuple], save_path: str = "./results/class_distribution.png"):

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    all_classes = sorted(set().union(*[
        set(y.tolist() if isinstance(y, torch.Tensor) else y) 
        for _, y in client_data
    ]))
    
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
    print(f"   📊 Class distribution plot saved: {save_path}")


# PARALLEL I/O

def save_client_data_parallel(client_data: List[Tuple[torch.Tensor, torch.Tensor]], 
    save_dir: Path, num_workers: Optional[int] = None) -> None:

    def save_single_client(args):
        client_id, X_cli, y_cli = args
        output_path = save_dir / f"client{client_id}.pt"
        torch.save((X_cli, y_cli), output_path)
        return client_id, output_path
    
    if num_workers is None:
        num_workers = min(os.cpu_count() or 4, len(client_data))
    
    save_args = [(i+1, X, y) for i, (X, y) in enumerate(client_data)]
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(save_single_client, save_args))
    
    for client_id, path in results:
        print(f"   Client {client_id}: {path}")


# MAIN PREPROCESSING PIPELINE

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
    pca_batch_size: Optional[int] = None,
    encoding_range: Tuple[float, float] = (-1, 1),
    variance_threshold: float = 0.80,
    parallel_io: bool = True
) -> Tuple[Tuple, Tuple, Tuple, List[Tuple]]:
    """
    Quantum Federated Learning MNIST preprocessing pipeline.
    
    Args:
        raw_folder: Path to raw MNIST IDX files
        processed_folder: Output path for processed data
        digits: Tuple of digit classes to include
        val_split: Validation set fraction (from training data)
        num_clients: Number of federated clients
        partition_type: 'iid' or 'non_iid'
        alpha: Dirichlet concentration (lower = more non-IID, typical: 0.1-0.5)
        apply_pca: Whether to apply PCA dimensionality reduction
        pca_components: Number of PCA components
        seed: Random seed for reproducibility
        generate_plots: Enable visualization generation
        use_incremental_pca: Use IncrementalPCA for large datasets
        pca_batch_size: Batch size for incremental PCA (auto if None)
        encoding_range: Feature scaling range (default: [-1, 1] for quantum)
        variance_threshold: Minimum PCA explained variance
        parallel_io: Use parallel I/O for client data saving
    
    Returns:
        (train_data, val_data, test_data, client_data_list); Each is (X_tensor, y_tensor) tuple
    """
    
    # SETUP - Reproducibility & Directories
    print("🌟 Quantum Federated Learning - MNIST Preprocessing")
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    for path in [processed_folder, "./results", "./artifacts"]:
        Path(path).mkdir(parents=True, exist_ok=True)

    # STEP 1: LOAD RAW DATA
    print("\n📂 Loading raw MNIST data...")
    
    try:
        file_map = {
            'train_images': "train-images.idx3-ubyte",
            'train_labels': "train-labels.idx1-ubyte",
            'test_images': "t10k-images.idx3-ubyte",
            'test_labels': "t10k-labels.idx1-ubyte"
        }
        
        X_train_raw = read_idx_images(Path(raw_folder) / file_map['train_images'])
        y_train_raw = read_idx_labels(Path(raw_folder) / file_map['train_labels'])
        X_test_raw = read_idx_images(Path(raw_folder) / file_map['test_images'])
        y_test_raw = read_idx_labels(Path(raw_folder) / file_map['test_labels'])
        
        print(f"   ✓ Training: {X_train_raw.shape}")
        print(f"   ✓ Test: {X_test_raw.shape}")
        
    except (FileNotFoundError, ValueError) as e:
        print(f"\n❌ Error loading raw data: {e}")
        print(f"   Expected files in {raw_folder}:")
        for fname in file_map.values():
            print(f"     - {fname}")
        return None
    
    # STEP 2: FILTER DIGITS
    print(f"\n🔍 Filtering digits: {digits}")
    
    train_mask = np.isin(y_train_raw, digits)
    test_mask = np.isin(y_test_raw, digits)
    
    X_train_filt = X_train_raw[train_mask]
    y_train_filt = y_train_raw[train_mask]
    X_test_filt = X_test_raw[test_mask]
    y_test_filt = y_test_raw[test_mask]
    
    print(f"   Training: {X_train_raw.shape[0]:,} → {X_train_filt.shape[0]:,} samples")
    print(f"   Test: {X_test_raw.shape[0]:,} → {X_test_filt.shape[0]:,} samples")
    
    # STEP 3: TRAIN/VAL SPLIT (BEFORE TRANSFORMATIONS - CRITICAL!)
    print(f"\n✂️  Splitting train/val ({1-val_split:.0%}/{val_split:.0%})...")
    
    train_idx, val_idx = train_test_split(
        np.arange(len(y_train_filt)),
        test_size=val_split,
        stratify=y_train_filt,
        random_state=seed
    )
    
    X_train = X_train_filt[train_idx]
    y_train = y_train_filt[train_idx]
    X_val = X_train_filt[val_idx]
    y_val = y_train_filt[val_idx]
    
    print(f"   Training: {len(y_train):,} samples")
    print(f"   Validation: {len(y_val):,} samples")
    print(f"   Test: {len(y_test_filt):,} samples")
    
    # STEP 4: NORMALIZE AND FLATTEN
    print("\n🔢 Normalizing to [0, 1] and flattening...")
    
    X_train_flat = (X_train / 255.0).astype(np.float32).reshape(len(X_train), -1)
    X_val_flat = (X_val / 255.0).astype(np.float32).reshape(len(X_val), -1)
    X_test_flat = (X_test_filt / 255.0).astype(np.float32).reshape(len(X_test_filt), -1)
    
    print(f"   Feature dimension: {X_train_flat.shape[1]}D")
    
    data_before_scaling = X_train_flat.copy()
    
    # STEP 5: APPLY PCA (OPTIONAL - FIT ON TRAIN ONLY)
    pca_model = None
    if apply_pca:
        X_train_flat, X_val_flat, X_test_flat, pca_model = apply_pca_transform(
            X_train_flat, X_val_flat, X_test_flat,
            n_components=pca_components,
            use_incremental=use_incremental_pca,
            batch_size=pca_batch_size,
            variance_threshold=variance_threshold
        )
        
        joblib.dump(pca_model, Path("./artifacts") / "pca_model.pkl")
        data_before_scaling = X_train_flat.copy()
    
    # STEP 6: SCALE TO QUANTUM ENCODING RANGE
    print(f"\n⚛️  Scaling to {encoding_range} for quantum encoding...")
    
    scaler = MinMaxScaler(feature_range=encoding_range)
    X_train_flat = scaler.fit_transform(X_train_flat)  # Fit on TRAIN only
    X_val_flat = scaler.transform(X_val_flat)
    X_test_flat = scaler.transform(X_test_flat)
    
    joblib.dump(scaler, Path("./artifacts") / "scaler.pkl")
    
    data_after_scaling = X_train_flat.copy()
    
    print(f"   Train range: [{X_train_flat.min():.3f}, {X_train_flat.max():.3f}]")
    print(f"   Val range: [{X_val_flat.min():.3f}, {X_val_flat.max():.3f}]")
    print(f"   Test range: [{X_test_flat.min():.3f}, {X_test_flat.max():.3f}]")
    
    # STEP 7: CREATE CLIENT PARTITIONS (TRAIN DATA ONLY)
    print(f"\n🌐 Creating {partition_type.upper()} partition for {num_clients} clients...")
    
    client_indices = create_partition(
        y_train,  # Only training labels (validation held out)
        num_clients,
        alpha if partition_type == 'non_iid' else None,
        seed
    )
    
    # Validate partitions
    validate_client_partitions(client_indices, y_train, len(digits), partition_type)
    
    print("\n   Client data distribution:")
    for i, idx in enumerate(client_indices):
        y_cli = y_train[idx]
        unique, counts = np.unique(y_cli, return_counts=True)
        dist = ", ".join([f"Digit {u}: {c}" for u, c in zip(unique, counts)])
        print(f"     Client {i+1}: {len(idx):>5} samples  [{dist}]")
    
    # STEP 8: CONVERT TO TENSORS (GLOBAL DATASETS)
    print(f"\n💾 Saving global datasets to {processed_folder}...")
    
    train_dataset = (
        torch.tensor(X_train_flat, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    val_dataset = (
        torch.tensor(X_val_flat, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )
    test_dataset = (
        torch.tensor(X_test_flat, dtype=torch.float32),
        torch.tensor(y_test_filt, dtype=torch.long)
    )
    
    torch.save(train_dataset, Path(processed_folder) / "train.pt")
    torch.save(val_dataset, Path(processed_folder) / "val.pt")
    torch.save(test_dataset, Path(processed_folder) / "test.pt")
    
    print(f"   ✓ train.pt: {train_dataset[0].shape}")
    print(f"   ✓ val.pt: {val_dataset[0].shape}")
    print(f"   ✓ test.pt: {test_dataset[0].shape}")
    
    # STEP 9: CREATE CLIENT DATASETS (ZERO-COPY SLICING)
    print(f"\n👥 Processing client datasets...")
    
    # Convert training data to tensor once
    X_train_tensor = torch.tensor(X_train_flat, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    
    # Slice for each client (zero-copy operation)
    client_data_processed = []
    for i, idx in enumerate(client_indices):
        X_cli = X_train_tensor[idx]
        y_cli = y_train_tensor[idx]
        client_data_processed.append((X_cli, y_cli))
        
        print(f"   Client {i+1}: {X_cli.shape}, Classes: {len(torch.unique(y_cli))}")
    
    # Save client data (parallel I/O if enabled)
    if parallel_io and num_clients > 2:
        print(f"\n💾 Saving client data (parallel I/O)...")
        save_client_data_parallel(
            client_data_processed, 
            Path(processed_folder),
            num_workers=min(os.cpu_count() or 4, num_clients)
        )
    else:
        print(f"\n💾 Saving client data...")
        for i, (X_cli, y_cli) in enumerate(client_data_processed):
            path = Path(processed_folder) / f"client{i+1}.pt"
            torch.save((X_cli, y_cli), path)
            print(f"   Client {i+1}: {path}")
    
    # STEP 10: STATISTICS & VISUALIZATION
    print("\n📈 Global class distribution:")
    for name, labels in [("Train", y_train), ("Val", y_val), ("Test", y_test_filt)]:
        unique, counts = np.unique(labels, return_counts=True)
        dist = ", ".join([f"Digit {u}: {c}" for u, c in zip(unique, counts)])
        print(f"   {name:>5}: {dist}")
    
    # Generate visualizations
    if generate_plots:
        print("\n📊 Generating visualizations...")
        
        # Prepare data for visualization (original images)
        client_data_orig = []
        max_viz_samples = 500  # Limit for memory efficiency
        
        for idx in client_indices:
            sample_idx = idx[:min(len(idx), max_viz_samples)]
            X_cli_orig = (X_train[sample_idx] / 255.0).astype(np.float32)
            y_cli_orig = y_train[sample_idx]
            client_data_orig.append((X_cli_orig, y_cli_orig))
        
        # Image grid visualization
        visualize_client_data(
            client_data_orig,
            save_path="./results/client_data_images.png",
            is_pca=False
        )
        
        # PCA scatter plot (if PCA applied)
        if apply_pca and pca_components >= 2:
            visualize_client_data(
                client_data_processed,
                save_path="./results/client_data_pca.png",
                is_pca=True
            )
        
        # Class distribution bar chart
        plot_class_distribution(client_data_orig)
    
    # STEP 11: SAVE METADATA & CHECKSUMS
    print("\n📋 Saving metadata and checksums...")
    
    metadata = {
        # Configuration
        'digits': list(digits),
        'num_clients': num_clients,
        'partition_type': partition_type,
        'alpha': alpha if partition_type == 'non_iid' else None,
        'val_split': val_split,
        'seed': seed,
        
        # Dimensionality
        'original_features': 784,
        'pca_applied': apply_pca,
        'pca_components': pca_components if apply_pca else None,
        'final_features': X_train_flat.shape[1],
        
        # PCA details
        'use_incremental_pca': use_incremental_pca if apply_pca else None,
        'pca_explained_variance': pca_model.explained_variance_ratio_.sum() if apply_pca else None,
        'pca_component_variances': pca_model.explained_variance_ratio_.tolist() if apply_pca else None,
        
        # Scaling
        'encoding_range': list(encoding_range),
        'scaler_min': scaler.data_min_.tolist(),
        'scaler_max': scaler.data_max_.tolist(),
        
        # Sample counts
        'samples': {
            'train': len(y_train),
            'val': len(y_val),
            'test': len(y_test_filt),
            'clients': [len(idx) for idx in client_indices]
        },
        
        # Client class distributions
        'client_class_counts': [
            {int(cls): int(cnt) for cls, cnt in zip(*np.unique(y_train[idx], return_counts=True))}
            for idx in client_indices
        ],
        
        # Reproducibility
        'package_versions': {
            'python': platform.python_version(),
            'numpy': np.__version__,
            'torch': torch.__version__,
            'sklearn': __import__('sklearn').__version__
        },
        
        # Data checksums (for verification)
        'data_hashes': {
            'train': compute_data_hash(X_train_flat, y_train),
            'val': compute_data_hash(X_val_flat, y_val),
            'test': compute_data_hash(X_test_flat, y_test_filt)
        }
    }
    
    metadata_path = Path("./artifacts") / "preprocessing_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   ✓ Metadata saved: {metadata_path}")
    
    # STEP 12: ADVANCED VISUALIZATIONS (OPTIONAL)
    if generate_plots and VISUALIZATIONS_AVAILABLE:
        try:
            print("\n📊 Generating advanced visualizations...")
            generate_all_preprocessing_visualizations(
                pca_model=pca_model if apply_pca else None,
                data_before_scaling=data_before_scaling,
                data_after_scaling=data_after_scaling,
                client_indices=client_indices,
                client_data=client_data_processed,
                y_train=y_train,
                y_val=y_val,
                metadata=metadata,
                num_classes=len(digits),
                save_dir="./results/preprocessing"
            )
        except Exception as e:
            print(f"   ⚠️  Warning: Could not generate advanced visualizations: {e}")
    
    # COMPLETION SUMMARY
    print("\n✅ Quantum FL Preprocessing Complete!")
    print(f"\n📊 Summary:")
    print(f"   Dimensionality: 784D → {X_train_flat.shape[1]}D")
    print(f"   Method: {'PCA + MinMax Scaling' if apply_pca else 'MinMax Scaling Only'}")
    print(f"   Encoding range: {encoding_range}")
    print(f"   Processed data: {processed_folder}/")
    print(f"   Artifacts: ./artifacts/")
    print(f"   Visualizations: ./results/")
    
    print(f"\n📁 Output files:")
    print(f"   Global datasets: train.pt, val.pt, test.pt")
    print(f"   Client datasets: client1.pt ... client{num_clients}.pt")
    print(f"   Transformers: pca_model.pkl, scaler.pkl")
    print(f"   Metadata: preprocessing_metadata.json")
    
    print(f"\n🔐 Data integrity:")
    print(f"   Train hash: {metadata['data_hashes']['train'][:16]}...")
    print(f"   Val hash: {metadata['data_hashes']['val'][:16]}...")
    print(f"   Test hash: {metadata['data_hashes']['test'][:16]}...")
    
    if partition_type == 'non_iid':
        print(f"\n⚠️  Non-IID partitioning (alpha={alpha}):")
        print(f"   Lower alpha = more heterogeneous client distributions")
        print(f"   Consider using weighted aggregation or FedProx for stability")
        
    return train_dataset, val_dataset, test_dataset, client_data_processed
    
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
        use_incremental_pca=True,
        pca_batch_size=1000,
        encoding_range=(-1, 1),  # Suitable for quantum encoding
        variance_threshold=0.80,
        parallel_io=True
    )