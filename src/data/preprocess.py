import json
from pathlib import Path
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from core.utils import get_logger, set_seed

logger = get_logger(__name__)

try:
    from data.plots_preprocess import generate_all_preprocessing_visualizations
    VISUALIZATIONS_AVAILABLE = True
except ImportError:
    logger.warning("Preprocessing visualization module not available")
    VISUALIZATIONS_AVAILABLE = False

def read_idx_images(filename: str) -> np.ndarray:
    # Memory-efficient IDX image reader with proper header parsing.
    with open(filename, 'rb') as f:
        magic, num_images, rows, cols = np.frombuffer(f.read(16), dtype='>i4')
        if magic != 2051:
            raise ValueError(f"Invalid magic number {magic} for image file")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(num_images, rows, cols)

def read_idx_labels(filename: str) -> np.ndarray:
    # Memory-efficient IDX label reader with proper header parsing.
    with open(filename, 'rb') as f:
        magic, num_labels = np.frombuffer(f.read(8), dtype='>i4')
        if magic != 2049:
            raise ValueError(f"Invalid magic number {magic} for label file")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data

# Partitioning Logic (IID and Non-IID via Dirichlet)
def create_iid_partition(indices: np.ndarray, num_clients: int, rng: np.random.Generator) -> list[np.ndarray]:
    # Create IID partition using indices (zero-copy until final assignment)."""
    rng.shuffle(indices)
    return np.array_split(indices, num_clients)  # Cleaner than manual slicing

def create_non_iid_partition(y_data: np.ndarray, num_clients: int, alpha: float, rng: np.random.Generator) -> list[np.ndarray]:
    # Optimized non-IID partition using Dirichlet distribution.
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

def create_partition(y_data: np.ndarray, num_clients: int, alpha: Optional[float] = None, seed: int = 42) -> list[np.ndarray]:
    # Unified partition API: IID if alpha=None, else Dirichlet non-IID.
    rng = np.random.default_rng(seed)
    indices = np.arange(len(y_data))

    return (create_iid_partition(indices, num_clients, rng) if alpha is None
            else create_non_iid_partition(y_data, num_clients, alpha, rng))

# Visualization Functions
def visualize_client_data(client_data: list[tuple], save_path: Optional[str] = None,
                          samples_per_client: int = 5, is_pca: bool = False):
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
        logger.info(f"Visualization saved: {save_path}")
    plt.close(fig)

def plot_class_distribution(client_data: list[tuple], save_path: str = "./results/class_distribution.png"):
    # Plot stacked bar chart of class distribution across clients.
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
    logger.info(f"Class distribution plot saved: {save_path}")

# ── Internal helpers for preprocess_mnist ────────────────────────────

def _load_raw_mnist(raw_folder: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    file_map = {
        'train_images': "train-images.idx3-ubyte",
        'train_labels': "train-labels.idx1-ubyte",
        'test_images': "t10k-images.idx3-ubyte",
        'test_labels': "t10k-labels.idx1-ubyte",
    }
    X_train = read_idx_images(Path(raw_folder) / file_map['train_images'])
    y_train = read_idx_labels(Path(raw_folder) / file_map['train_labels'])
    X_test = read_idx_images(Path(raw_folder) / file_map['test_images'])
    y_test = read_idx_labels(Path(raw_folder) / file_map['test_labels'])
    logger.info(f"Raw data loaded: Train {X_train.shape}, Test {X_test.shape}")
    return X_train, y_train, X_test, y_test


def _filter_digits(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, digits: list) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_mask = np.isin(y_train, digits)
    test_mask = np.isin(y_test, digits)
    X_train_f = X_train[train_mask]
    y_train_f = y_train[train_mask]
    X_test_f = X_test[test_mask]
    y_test_f = y_test[test_mask]
    logger.info(f"After filtering digits {digits}: Train {X_train_f.shape}, Test {X_test_f.shape}")
    return X_train_f, y_train_f, X_test_f, y_test_f


def _train_val_split(X_full: np.ndarray, y_full: np.ndarray, val_split: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_idx, val_idx = train_test_split(
        np.arange(len(y_full)), test_size=val_split,
        stratify=y_full, random_state=seed,
    )
    return X_full[train_idx], y_full[train_idx], X_full[val_idx], y_full[val_idx]


def _normalize_and_flatten(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_train_flat = (X_train / 255.0).astype(np.float32).reshape(len(X_train), -1)
    X_val_flat = (X_val / 255.0).astype(np.float32).reshape(len(X_val), -1)
    X_test_flat = (X_test / 255.0).astype(np.float32).reshape(len(X_test), -1)
    return X_train_flat, X_val_flat, X_test_flat


def _apply_pca_if_needed(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray, pca_components: int, use_incremental_pca: bool, pca_batch_size: int) -> tuple:
    if pca_components > X_train.shape[1]:
        raise ValueError(f"PCA components ({pca_components}) > features ({X_train.shape[1]})")
    logger.info(f"Applying PCA: {X_train.shape[1]}D → {pca_components}D")

    if use_incremental_pca and len(X_train) > 10000:
        logger.info(f"Using Incremental PCA (batch_size={pca_batch_size})")
        model = IncrementalPCA(n_components=pca_components, batch_size=pca_batch_size)
        for i in range(0, len(X_train), pca_batch_size):
            model.partial_fit(X_train[i:i + pca_batch_size])
    else:
        model = PCA(n_components=pca_components)
        model.fit(X_train)

    X_train_t = model.transform(X_train)
    X_val_t = model.transform(X_val)
    X_test_t = model.transform(X_test)
    joblib.dump(model, Path("./artifacts") / "pca_model.pkl")
    logger.info(f"Explained variance: {model.explained_variance_ratio_.sum():.4f}")
    return model, X_train_t, X_val_t, X_test_t


def _scale_data(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray) -> tuple:
    logger.info("Scaling to [-1, 1] for quantum encoding...")
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    joblib.dump(scaler, Path("./artifacts") / "scaler.pkl")
    return scaler, X_train_s, X_val_s, X_test_s


def _save_global_datasets(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, processed_folder: str) -> dict:
    datasets = {
        'train': (torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
        'val': (torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)),
        'test': (torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)),
    }
    for name, data in datasets.items():
        torch.save(data, Path(processed_folder) / f"{name}.pt")
    return datasets


def _process_clients(X_train_filt: np.ndarray, y_train_filt: np.ndarray, client_indices: list, apply_pca: bool, pca_model, scaler, processed_folder: str) -> tuple:
    X_all_norm = (X_train_filt / 255.0).astype(np.float32).reshape(len(X_train_filt), -1)
    if apply_pca:
        X_all_norm = pca_model.transform(X_all_norm)
    X_all_norm = scaler.transform(X_all_norm)

    processed, orig = [], []
    for i, idx in enumerate(client_indices):
        X_cli = torch.tensor(X_all_norm[idx], dtype=torch.float32)
        y_cli = torch.tensor(y_train_filt[idx], dtype=torch.long)
        processed.append((X_cli, y_cli))
        torch.save((X_cli, y_cli), Path(processed_folder) / f"client{i+1}.pt")
        X_cli_orig = (X_train_filt[idx] / 255.0).astype(np.float32)
        orig.append((X_cli_orig, y_train_filt[idx]))
        logger.info(f"Client {i+1}: {X_cli.shape}, Labels: {len(torch.unique(y_cli))} classes")
    return processed, orig


def _save_metadata(digits, num_clients, partition_type, alpha, apply_pca,
                   pca_components, use_incremental_pca, val_split, seed, feature_dim, splits):
    metadata = {
        'digits': list(digits), 'num_clients': num_clients,
        'partition_type': partition_type, 'alpha': alpha if partition_type == 'non_iid' else None,
        'apply_pca': apply_pca, 'pca_components': pca_components if apply_pca else None,
        'use_incremental_pca': use_incremental_pca if apply_pca else None,
        'val_split': val_split, 'seed': seed, 'feature_dim': feature_dim,
        'samples': {'train': splits[0], 'val': splits[1], 'test': splits[2]},
    }
    with open(Path("./artifacts") / "preprocessing_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    return metadata


# Main Preprocessing Pipeline
def preprocess_mnist(
    raw_folder: str,
    processed_folder: str,
    digits: tuple[int, ...] = (0, 1, 2),
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
) -> tuple:
    set_seed(seed)
    for path in [processed_folder, "./results", "./artifacts"]:
        Path(path).mkdir(parents=True, exist_ok=True)

    logger.info("Quantum Federated Learning - MNIST Preprocessing")

    try:
        X_train_raw, y_train_raw, X_test_raw, y_test_raw = _load_raw_mnist(raw_folder)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Error loading raw data: {e}")
        return None

    X_train_filt, y_train_filt, X_test_filt, y_test_filt = _filter_digits(
        X_train_raw, y_train_raw, X_test_raw, y_test_raw, digits
    )

    logger.info(f"Creating {partition_type.upper()} partition for {num_clients} clients...")
    client_indices = create_partition(
        y_train_filt, num_clients,
        alpha if partition_type == 'non_iid' else None, seed,
    )
    for i, idx in enumerate(client_indices):
        if len(idx) == 0:
            raise ValueError(f"Client {i+1} received no data!")
        if len(np.unique(y_train_filt[idx])) == 1:
            logger.warning(f"Client {i+1} has only one class")
    for i, idx in enumerate(client_indices):
        y_cli = y_train_filt[idx]
        unique, counts = np.unique(y_cli, return_counts=True)
        logger.info(f"Client {i+1}: {len(idx):>5} samples  [{', '.join(f'Digit {u}: {c}' for u, c in zip(unique, counts))}]")

    X_train, y_train, X_val, y_val = _train_val_split(X_train_filt, y_train_filt, val_split, seed)
    X_train_flat, X_val_flat, X_test_flat = _normalize_and_flatten(X_train, X_val, X_test_filt)

    data_before_scaling = X_train_flat.copy()
    pca_model = None
    if apply_pca:
        pca_model, X_train_flat, X_val_flat, X_test_flat = _apply_pca_if_needed(
            X_train_flat, X_val_flat, X_test_flat, pca_components, use_incremental_pca, pca_batch_size,
        )
        data_before_scaling = X_train_flat.copy()

    scaler, X_train_flat, X_val_flat, X_test_flat = _scale_data(X_train_flat, X_val_flat, X_test_flat)
    data_after_scaling = X_train_flat.copy()

    datasets = _save_global_datasets(X_train_flat, y_train, X_val_flat, y_val, X_test_flat, y_test_filt, processed_folder)

    client_data_processed, client_data_orig = _process_clients(
        X_train_filt, y_train_filt, client_indices, apply_pca, pca_model, scaler, processed_folder,
    )

    logger.info("Global class distribution:")
    for name, labels in [("Train", y_train), ("Val", y_val), ("Test", y_test_filt)]:
        unique, counts = np.unique(labels, return_counts=True)
        logger.info(f"{name:>5}: {', '.join(f'Digit {u}: {c}' for u, c in zip(unique, counts))}")

    if generate_plots:
        logger.info("Generating basic visualizations...")
        visualize_client_data(client_data_orig, save_path="./results/client_data_images.png", is_pca=False)
        if apply_pca and pca_components >= 2:
            visualize_client_data(client_data_processed, save_path="./results/client_data_pca.png", is_pca=True)
        plot_class_distribution(client_data_orig)

    metadata = _save_metadata(
        digits, num_clients, partition_type, alpha, apply_pca,
        pca_components, use_incremental_pca, val_split, seed,
        X_train_flat.shape[1], (len(y_train), len(y_val), len(y_test_filt)),
    )

    if generate_plots and VISUALIZATIONS_AVAILABLE:
        try:
            generate_all_preprocessing_visualizations(
                pca_model=pca_model if apply_pca else None,
                data_before_scaling=data_before_scaling,
                data_after_scaling=data_after_scaling,
                client_indices=client_indices,
                client_data=client_data_processed,
                y_train=y_train, y_val=y_val,
                metadata=metadata, num_classes=len(digits),
                save_dir="./results/preprocessing",
            )
        except Exception as e:
            logger.warning(f"Could not generate advanced visualizations: {e}")

    logger.info("Preprocessing Complete! %dD → %dD", 784, X_train_flat.shape[1])
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
