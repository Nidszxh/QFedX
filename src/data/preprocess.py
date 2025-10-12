import numpy as np
import torch
import random
import platform
import hashlib
import json
import joblib
from pathlib import Path
from typing import Tuple, List, Optional
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
    VISUALIZATIONS_AVAILABLE = False

"""
Dual-Mode MNIST Preprocessing for Federated Learning
- preprocess_mnist_classical(): For CNN-based classical FL (784D, [0,1])
- preprocess_mnist_quantum(): For quantum VQC (PCA to 4D, [-1,1]) 
"""

# DATA I/O

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


# PARTITIONING

def create_iid_partition(indices: np.ndarray, num_clients: int, rng: np.random.Generator) -> List[np.ndarray]:
    rng.shuffle(indices)
    return np.array_split(indices, num_clients)

def create_non_iid_partition(y_data: np.ndarray, num_clients: int, alpha: float, rng: np.random.Generator) -> List[np.ndarray]:
    num_classes = len(np.unique(y_data))
    label_indices = [np.where(y_data == i)[0] for i in range(num_classes)]
    client_indices = [[] for _ in range(num_clients)]
    
    for indices in label_indices:
        rng.shuffle(indices)
        proportions = rng.dirichlet([alpha] * num_clients)
        splits = np.insert(np.cumsum(proportions), 0, 0) * len(indices)
        splits = np.round(splits).astype(int)
        splits[-1] = len(indices)
        
        for cid in range(num_clients):
            client_indices[cid].extend(indices[splits[cid]:splits[cid+1]])
    
    return [np.array(idx, dtype=np.int64) for idx in client_indices]

def create_partition(y_data: np.ndarray, num_clients: int, alpha: Optional[float] = None, seed: int = 42) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(y_data))
    return (create_iid_partition(indices, num_clients, rng) if alpha is None 
            else create_non_iid_partition(y_data, num_clients, alpha, rng))

# VALIDATION

def validate_client_partitions(client_indices: List[np.ndarray], y_data: np.ndarray, num_classes: int, partition_type: str) -> None:
    for i, idx in enumerate(client_indices):
        if len(idx) == 0:
            raise ValueError(f"❌ Client {i+1} received no data samples!")
        
        unique_classes = np.unique(y_data[idx])
        if len(unique_classes) == 1:
            raise ValueError(
                f"❌ Client {i+1} has only one class (label {unique_classes[0]}). "
                f"This will cause training instability.\n"
                f"   Solutions: Reduce alpha (non-IID) or increase samples per client."
            )
        
        if partition_type == 'non_iid':
            class_counts = np.bincount(y_data[idx], minlength=num_classes)
            max_ratio = class_counts.max() / (class_counts[class_counts > 0].mean())
            if max_ratio > 10.0:
                print(f"   ⚠️  Client {i+1}: Extreme imbalance (ratio {max_ratio:.1f}:1)")

def compute_data_hash(X: np.ndarray, y: np.ndarray) -> str:
    combined = np.concatenate([X.flatten(), y.flatten()])
    return hashlib.sha256(combined.tobytes()).hexdigest()

# PCA

def compute_optimal_pca_batch_size(n_samples: int, n_features: int, n_components: int) -> int:  
    max_batch_memory = 1e9
    max_batch_size = int(max_batch_memory / (n_features * 8))
    min_batch_size = max(10 * n_components, 100)
    adaptive_batch = max(n_samples // 50, min_batch_size)
    return min(max_batch_size, max(adaptive_batch, min_batch_size))

def apply_pca_transform(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray, n_components: int, use_incremental: bool = False, 
                        batch_size: Optional[int] = None, variance_threshold: float = 0.80) -> Tuple[np.ndarray, np.ndarray, np.ndarray, PCA]:
    if n_components > X_train.shape[1]:
        raise ValueError(
            f"PCA components ({n_components}) exceeds feature dimension ({X_train.shape[1]})"
        )
    
    print(f"\n🔄 Applying PCA: {X_train.shape[1]}D → {n_components}D")
    
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
    
    explained_var = pca_model.explained_variance_ratio_.sum()
    print(f"   Explained variance: {explained_var:.4f}")
    
    if explained_var < variance_threshold:
        print(f"   ⚠️  Warning: Low variance retained ({explained_var:.2%})")
        cumsum_var = np.cumsum(pca_model.explained_variance_ratio_)
        optimal_k = np.argmax(cumsum_var >= 0.90) + 1
        print(f"   Recommendation: Use {optimal_k} components for 90% variance")
    
    X_train_pca = pca_model.transform(X_train)
    X_val_pca = pca_model.transform(X_val)
    X_test_pca = pca_model.transform(X_test)
    
    return X_train_pca, X_val_pca, X_test_pca, pca_model


# PARALLEL I/O

def save_client_data_parallel(client_data: List[Tuple[torch.Tensor, torch.Tensor]], save_dir: Path, num_workers: Optional[int] = None) -> None:
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


# CORE PREPROCESSING (Internal)

def _preprocess_mnist_core(
    raw_folder: str,
    processed_folder: str,
    digits: Tuple[int, ...],
    val_split: float,
    num_clients: int,
    partition_type: str,
    alpha: Optional[float],
    apply_pca: bool,
    pca_components: Optional[int],
    encoding_range: Tuple[float, float],
    seed: int,
    generate_plots: bool,
    use_incremental_pca: bool,
    pca_batch_size: Optional[int],
    variance_threshold: float,
    parallel_io: bool,
    model_type: str
) -> Tuple[Tuple, Tuple, Tuple, List[Tuple]]:
    
    # Setup
    print(f"🌟 {model_type.upper()} Federated Learning - MNIST Preprocessing")
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    for path in [processed_folder, "./results", "./artifacts"]:
        Path(path).mkdir(parents=True, exist_ok=True)

    # Load raw data
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
        return None
    
    # Filter digits
    print(f"\n🔍 Filtering digits: {digits}")
    
    train_mask = np.isin(y_train_raw, digits)
    test_mask = np.isin(y_test_raw, digits)
    
    X_train_filt = X_train_raw[train_mask]
    y_train_filt = y_train_raw[train_mask]
    X_test_filt = X_test_raw[test_mask]
    y_test_filt = y_test_raw[test_mask]
    
    print(f"   Training: {X_train_raw.shape[0]:,} → {X_train_filt.shape[0]:,} samples")
    print(f"   Test: {X_test_raw.shape[0]:,} → {X_test_filt.shape[0]:,} samples")
    
    # Train/val split
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
    
    # Normalize and flatten
    print("\n🔢 Normalizing to [0, 1] and flattening...")
    
    X_train_flat = (X_train / 255.0).astype(np.float32).reshape(len(X_train), -1)
    X_val_flat = (X_val / 255.0).astype(np.float32).reshape(len(X_val), -1)
    X_test_flat = (X_test_filt / 255.0).astype(np.float32).reshape(len(X_test_filt), -1)
    
    print(f"   Feature dimension: {X_train_flat.shape[1]}D")
    
    data_before_scaling = X_train_flat.copy()
    
    # Apply PCA (optional)
    pca_model = None
    if apply_pca:
        X_train_flat, X_val_flat, X_test_flat, pca_model = apply_pca_transform(
            X_train_flat, X_val_flat, X_test_flat,
            n_components=pca_components,
            use_incremental=use_incremental_pca,
            batch_size=pca_batch_size,
            variance_threshold=variance_threshold
        )
        
        joblib.dump(pca_model, Path("./artifacts") / f"pca_model_{model_type}.pkl")
        data_before_scaling = X_train_flat.copy()
    
    # Scale to encoding range
    print(f"\n⚛️  Scaling to {encoding_range}...")
    
    scaler = MinMaxScaler(feature_range=encoding_range)
    X_train_flat = scaler.fit_transform(X_train_flat)
    X_val_flat = scaler.transform(X_val_flat)
    X_test_flat = scaler.transform(X_test_flat)
    
    joblib.dump(scaler, Path("./artifacts") / f"scaler_{model_type}.pkl")
    
    data_after_scaling = X_train_flat.copy()
    
    print(f"   Train range: [{X_train_flat.min():.3f}, {X_train_flat.max():.3f}]")
    print(f"   Val range: [{X_val_flat.min():.3f}, {X_val_flat.max():.3f}]")
    print(f"   Test range: [{X_test_flat.min():.3f}, {X_test_flat.max():.3f}]")
    
    # Create client partitions
    print(f"\n🌐 Creating {partition_type.upper()} partition for {num_clients} clients...")

    client_indices = create_partition(y_train, num_clients, 
                                    alpha if partition_type == 'non_iid' else None, seed)

    validate_client_partitions(client_indices, y_train, len(digits), partition_type)
    
    print("\n   Client data distribution:")
    for i, idx in enumerate(client_indices):
        y_cli = y_train[idx]
        unique, counts = np.unique(y_cli, return_counts=True)
        dist = ", ".join([f"Digit {u}: {c}" for u, c in zip(unique, counts)])
        print(f"     Client {i+1}: {len(idx):>5} samples  [{dist}]")
    
    # Convert to tensors
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
    
    # Create client datasets
    print(f"\n👥 Processing client datasets...")
    
    X_train_tensor = torch.tensor(X_train_flat, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    
    client_data_processed = []
    for i, idx in enumerate(client_indices):
        X_cli = X_train_tensor[idx]
        y_cli = y_train_tensor[idx]
        client_data_processed.append((X_cli, y_cli))
        
        print(f"   Client {i+1}: {X_cli.shape}, Classes: {len(torch.unique(y_cli))}")
    
    # Save client data
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
    
    # Statistics
    print("\n📈 Global class distribution:")
    for name, labels in [("Train", y_train), ("Val", y_val), ("Test", y_test_filt)]:
        unique, counts = np.unique(labels, return_counts=True)
        dist = ", ".join([f"Digit {u}: {c}" for u, c in zip(unique, counts)])
        print(f"   {name:>5}: {dist}")
    
    # Metadata
    print("\n📋 Saving metadata...")
    
    metadata = {
        'model_type': model_type,
        'digits': list(digits),
        'num_clients': num_clients,
        'partition_type': partition_type,
        'alpha': float(alpha) if partition_type == 'non_iid' and alpha is not None else None,
        'val_split': float(val_split),
        'seed': int(seed),
        'original_features': 784,
        'pca_applied': bool(apply_pca),
        'pca_components': int(pca_components) if apply_pca else None,
        'final_features': int(X_train_flat.shape[1]),
        'pca_explained_variance': float(pca_model.explained_variance_ratio_.sum()) if apply_pca else None,
        'encoding_range': [float(encoding_range[0]), float(encoding_range[1])],
        'samples': {
            'train': int(len(y_train)),
            'val': int(len(y_val)),
            'test': int(len(y_test_filt)),
            'clients': [int(len(idx)) for idx in client_indices]
        },
        'package_versions': {
            'python': platform.python_version(),
            'numpy': np.__version__,
            'torch': torch.__version__,
        },
        'data_hashes': {
            'train': compute_data_hash(X_train_flat, y_train),
            'val': compute_data_hash(X_val_flat, y_val),
            'test': compute_data_hash(X_test_flat, y_test_filt)
        }
    }
    
    metadata_path = Path("./artifacts") / f"preprocessing_metadata_{model_type}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   ✓ Metadata saved: {metadata_path}")
    
    # Advanced visualizations
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
                save_dir=f"./results/preprocessing_{model_type}"
            )
        except Exception as e:
            print(f"   ⚠️  Warning: Could not generate advanced visualizations: {e}")
    
    # Summary
    print("\n✅ Preprocessing Complete!")
    print(f"\n📊 Summary:")
    print(f"   Model type: {model_type.upper()}")
    print(f"   Dimensionality: 784D → {X_train_flat.shape[1]}D")
    print(f"   Encoding range: {encoding_range}")
    print(f"   Processed data: {processed_folder}/")
    
    if partition_type == 'non_iid':
        print(f"\n⚠️  Non-IID partitioning (alpha={alpha})")
    
    print()
    return train_dataset, val_dataset, test_dataset, client_data_processed


# PUBLIC API: CLASSICAL FEDERATED LEARNING

def preprocess_mnist_classical(
    raw_folder: str = "./dataset/raw",
    processed_folder: str = "./dataset/processed_classical",
    digits: Tuple[int, ...] = (0, 1, 2),
    val_split: float = 0.1,
    num_clients: int = 4,
    partition_type: str = 'iid',
    alpha: float = 0.5,
    seed: int = 42,
    generate_plots: bool = True,
    parallel_io: bool = True
) -> Tuple[Tuple, Tuple, Tuple, List[Tuple]]:
    """
    Preprocess MNIST for Classical Federated Learning (CNN).
    
    Configuration:
    - NO PCA (keeps full 784D image data)
    - Scaling to [0, 1] (standard for CNNs)
    
    Returns:
        (train_data, val_data, test_data, client_data_list)
        Each X has shape (n_samples, 784) with range [0, 1]
    """
    return _preprocess_mnist_core(
        raw_folder=raw_folder,
        processed_folder=processed_folder,
        digits=digits,
        val_split=val_split,
        num_clients=num_clients,
        partition_type=partition_type,
        alpha=alpha if partition_type == 'non_iid' else None,
        apply_pca=False,
        pca_components=None,
        encoding_range=(0, 1),
        seed=seed,
        generate_plots=generate_plots,
        use_incremental_pca=False,
        pca_batch_size=None,
        variance_threshold=0.80,
        parallel_io=parallel_io,
        model_type='classical'
    )


# PUBLIC API: QUANTUM FEDERATED LEARNING

def preprocess_mnist_quantum(
    raw_folder: str = "./dataset/raw",
    processed_folder: str = "./dataset/processed_quantum",
    digits: Tuple[int, ...] = (0, 1, 2),
    val_split: float = 0.1,
    num_clients: int = 4,
    partition_type: str = 'iid',
    alpha: float = 0.5,
    pca_components: int = 4,
    seed: int = 42,
    generate_plots: bool = True,
    use_incremental_pca: bool = False,
    pca_batch_size: Optional[int] = None,
    variance_threshold: float = 0.80,
    parallel_io: bool = True
) -> Tuple[Tuple, Tuple, Tuple, List[Tuple]]:
    """
    Preprocess MNIST for Quantum Federated Learning (VQC).
    
    Configuration:
    - PCA dimensionality reduction (784D → pca_components D)
    - Scaling to [-1, 1] (for quantum angle encoding)
    
    Returns:
        (train_data, val_data, test_data, client_data_list)
        Each X has shape (n_samples, pca_components) with range [-1, 1]
    """
    return _preprocess_mnist_core(
        raw_folder=raw_folder,
        processed_folder=processed_folder,
        digits=digits,
        val_split=val_split,
        num_clients=num_clients,
        partition_type=partition_type,
        alpha=alpha if partition_type == 'non_iid' else None,
        apply_pca=True,
        pca_components=pca_components,
        encoding_range=(-1, 1),
        seed=seed,
        generate_plots=generate_plots,
        use_incremental_pca=use_incremental_pca,
        pca_batch_size=pca_batch_size,
        variance_threshold=variance_threshold,
        parallel_io=parallel_io,
        model_type='quantum'
    )


# MAIN

if __name__ == "__main__":
    print("\n" + "="*70)
    print("MNIST Preprocessing - Dual Mode")
    print("="*70)
    print("\nRunning both classical and quantum preprocessing...")
    
    # Classical preprocessing
    print("\n" + "-"*70)
    print("CLASSICAL MODE")
    print("-"*70)
    
    train_c, val_c, test_c, clients_c = preprocess_mnist_classical(
        raw_folder="./dataset/raw",
        digits=(0, 1, 2),
        num_clients=4,
        partition_type='iid',
        alpha=0.5,
        seed=42,
        generate_plots=False
    )
    
    print(f"✅ Classical preprocessing complete!")
    print(f"   Feature dimension: {clients_c[0][0].shape[1]}D")
    print(f"   Data range: [{clients_c[0][0].min():.3f}, {clients_c[0][0].max():.3f}]")
    
    # Quantum preprocessing
    print("\n" + "-"*70)
    print("QUANTUM MODE")
    print("-"*70)
    
    train_q, val_q, test_q, clients_q = preprocess_mnist_quantum(
        raw_folder="./dataset/raw",
        digits=(0, 1, 2),
        num_clients=4,
        partition_type='iid',
        alpha=0.5,
        pca_components=4,
        seed=42,
        generate_plots=False
    )
    
    print(f"✅ Quantum preprocessing complete!")
    print(f"   Feature dimension: {clients_q[0][0].shape[1]}D")
    print(f"   Data range: [{clients_q[0][0].min():.3f}, {clients_q[0][0].max():.3f}]")
    
    print("\n🎉 Both preprocessing modes completed successfully!")
    print(f"\nOutput directories:")
    print(f"   Classical: ./dataset/processed_classical/")
    print(f"   Quantum:   ./dataset/processed_quantum/")
    print()