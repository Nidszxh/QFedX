import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import torch
from scipy.spatial.distance import cosine
from scipy.stats import entropy

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def create_viz_folder(base_path: str = "./results/preprocessing") -> Path:
    """Create and return visualization folder path."""
    folder = Path(base_path)
    folder.mkdir(parents=True, exist_ok=True)
    return folder

# PCA ANALYSIS PLOTS
def plot_pca_variance(pca_model, save_path: str):
    """Plot explained variance ratio and cumulative variance for PCA components."""
    variance_ratio = pca_model.explained_variance_ratio_
    cumulative_variance = np.cumsum(variance_ratio)
    n_components = len(variance_ratio)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Individual variance bar plot
    ax1.bar(range(1, n_components + 1), variance_ratio, alpha=0.8, 
            color='steelblue', edgecolor='black')
    ax1.set_xlabel('Principal Component', fontsize=12)
    ax1.set_ylabel('Explained Variance Ratio', fontsize=12)
    ax1.set_title('PCA: Variance Explained per Component', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_xticks(range(1, n_components + 1))
    
    for i, v in enumerate(variance_ratio):
        ax1.text(i + 1, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)
    
    # Cumulative variance line plot
    ax2.plot(range(1, n_components + 1), cumulative_variance, marker='o', 
             linewidth=2.5, markersize=8, color='darkgreen')
    ax2.axhline(y=0.95, color='red', linestyle='--', linewidth=1.5, label='95% Threshold')
    ax2.fill_between(range(1, n_components + 1), cumulative_variance, alpha=0.2, color='green')
    ax2.set_xlabel('Number of Components', fontsize=12)
    ax2.set_ylabel('Cumulative Explained Variance', fontsize=12)
    ax2.set_title('PCA: Cumulative Variance Explained', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=10)
    ax2.set_xticks(range(1, n_components + 1))
    ax2.set_ylim([0, 1.05])
    
    total_var_text = f'Total: {cumulative_variance[-1]:.4f}'
    ax2.text(n_components, cumulative_variance[-1], total_var_text, 
             fontsize=10, ha='right', va='bottom', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   ✓ PCA variance plot saved")


def plot_pca_components_heatmap(pca_model, save_path: str, top_features: int = 50):
    """Plot heatmap of PCA component loadings (feature contributions)."""
    components = pca_model.components_
    n_components, n_features = components.shape
    
    # Select top features based on absolute contribution
    feature_importance = np.abs(components).sum(axis=0)
    top_indices = np.argsort(feature_importance)[-top_features:]
    
    fig, ax = plt.subplots(figsize=(12, max(6, n_components * 0.8)))
    
    im = ax.imshow(components[:, top_indices], cmap='RdBu_r', aspect='auto', 
                   vmin=-np.abs(components).max(), vmax=np.abs(components).max())
    
    ax.set_xlabel(f'Top {top_features} Original Features (by importance)', fontsize=11)
    ax.set_ylabel('Principal Components', fontsize=11)
    ax.set_title('PCA Component Loadings (Feature Contributions)', fontsize=13, fontweight='bold')
    ax.set_yticks(range(n_components))
    ax.set_yticklabels([f'PC{i+1}' for i in range(n_components)])
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Loading Strength', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   ✓ PCA components heatmap saved")


def plot_pca_2d_projection(data_after_pca, y_train, save_path: str, max_samples: int = 3000):
    """2D scatter plot of first two PCA components colored by class label."""
    if data_after_pca.shape[1] < 2:
        print(f"   ⚠️  Skipping 2D PCA projection (only {data_after_pca.shape[1]} components)")
        return
    
    # Sample for faster plotting
    if len(data_after_pca) > max_samples:
        indices = np.random.choice(len(data_after_pca), max_samples, replace=False)
        data_sample = data_after_pca[indices]
        y_sample = y_train[indices]
    else:
        data_sample = data_after_pca
        y_sample = y_train
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    unique_classes = np.unique(y_sample)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))
    
    for i, cls in enumerate(unique_classes):
        mask = y_sample == cls
        ax.scatter(data_sample[mask, 0], data_sample[mask, 1], 
                  c=[colors[i]], label=f'Digit {cls}', alpha=0.6, s=20, edgecolors='k', linewidth=0.3)
    
    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.set_title('PCA 2D Projection (Colored by Class)', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   ✓ PCA 2D projection saved")

# FEATURE SCALING VERIFICATION
def plot_scaling_verification(data_before: np.ndarray, data_after: np.ndarray,
                              save_path: str, sample_size: int = 5000):
    """Compare feature distributions before and after scaling."""
    # Sample data for faster plotting
    if len(data_before) > sample_size:
        indices = np.random.choice(len(data_before), sample_size, replace=False)
        data_before = data_before[indices]
        data_after = data_after[indices]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    before_flat = data_before.flatten()
    after_flat = data_after.flatten()
    
    # Histogram before scaling
    axes[0, 0].hist(before_flat, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_xlabel('Feature Value', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].set_title('Before Scaling (Post-PCA Normalized)', fontsize=12, fontweight='bold')
    axes[0, 0].axvline(before_flat.mean(), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {before_flat.mean():.3f}')
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Histogram after scaling
    axes[0, 1].hist(after_flat, bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].set_xlabel('Feature Value', fontsize=11)
    axes[0, 1].set_ylabel('Frequency', fontsize=11)
    axes[0, 1].set_title('After Scaling (Quantum/Classical Ready)', fontsize=12, fontweight='bold')
    axes[0, 1].axvline(after_flat.mean(), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {after_flat.mean():.3f}')
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Box plot per feature
    n_features_show = min(10, data_before.shape[1])
    axes[1, 0].boxplot([data_before[:, i] for i in range(n_features_show)], 
                       labels=[f'F{i+1}' for i in range(n_features_show)])
    axes[1, 0].set_xlabel('Feature Index', fontsize=11)
    axes[1, 0].set_ylabel('Value', fontsize=11)
    axes[1, 0].set_title(f'Before Scaling: Feature Distribution (First {n_features_show})', 
                        fontsize=11, fontweight='bold')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    axes[1, 1].boxplot([data_after[:, i] for i in range(n_features_show)], 
                       labels=[f'F{i+1}' for i in range(n_features_show)])
    axes[1, 1].set_xlabel('Feature Index', fontsize=11)
    axes[1, 1].set_ylabel('Value', fontsize=11)
    axes[1, 1].set_title(f'After Scaling: Feature Distribution (First {n_features_show})', 
                        fontsize=11, fontweight='bold')
    
    # Add range lines for quantum mode
    if data_after.min() < 0:
        axes[1, 1].axhline(y=-1, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Range')
        axes[1, 1].axhline(y=1, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   ✓ Scaling verification plot saved")

# CLIENT DATA VISUALIZATION
def visualize_client_sample_images(client_data: List[Tuple], y_train: np.ndarray,
                                   client_indices: List[np.ndarray], X_train_original: np.ndarray,
                                   save_path: str, samples_per_client: int = 5,
                                   max_clients_show: int = 8):
    """Visualize sample images from each client (for classical mode with original 28x28 images)."""
    num_clients = len(client_data)
    max_clients_plot = min(num_clients, max_clients_show)
    
    fig, axes = plt.subplots(max_clients_plot, samples_per_client, 
                            figsize=(samples_per_client * 2, max_clients_plot * 2))
    axes = np.atleast_2d(axes).reshape(max_clients_plot, samples_per_client)

    for i in range(max_clients_plot):
        idx = client_indices[i]
        y_cli = y_train[idx]
        
        for j in range(samples_per_client):
            ax = axes[i, j]
            if j < len(idx):
                img = X_train_original[idx[j]].reshape(28, 28)
                ax.imshow(img, cmap='gray')
                ax.set_title(f"Label: {y_cli[j]}", fontsize=8)
                ax.axis('off')
            else:
                ax.axis('off')
                
        axes[i, 0].set_ylabel(f"Client {i+1}", fontsize=10, rotation=0, labelpad=30)

    plt.suptitle(f"Sample Images from Clients ({max_clients_plot}/{num_clients} shown)", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   ✓ Client sample images saved")


def visualize_client_pca_distribution(client_data: List[Tuple], save_path: str,
                                      max_clients_show: int = 8):
    """Visualize PCA feature distribution for each client (scatter plots)."""
    num_clients = len(client_data)
    max_clients_plot = min(num_clients, max_clients_show)
    
    # Check if we have at least 2 PCA components
    if client_data[0][0].shape[1] < 2:
        print(f"   ⚠️  Skipping PCA distribution plot (only {client_data[0][0].shape[1]} components)")
        return
    
    fig, axes = plt.subplots(1, max_clients_plot, figsize=(max_clients_plot * 4, 4))
    axes = np.atleast_1d(axes)

    for i in range(max_clients_plot):
        X, y = client_data[i]
        X_np = X.numpy() if isinstance(X, torch.Tensor) else X
        y_np = y.numpy() if isinstance(y, torch.Tensor) else y
        
        ax = axes[i]
        scatter = ax.scatter(X_np[:, 0], X_np[:, 1], c=y_np, cmap='tab10', s=15, alpha=0.7, edgecolors='k', linewidth=0.3)
        ax.set_title(f"Client {i+1}", fontsize=11, fontweight='bold')
        ax.set_xlabel("PC1", fontsize=10)
        ax.set_ylabel("PC2", fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Class')

    plt.suptitle(f"PCA Feature Distribution per Client ({max_clients_plot}/{num_clients} shown)", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   ✓ Client PCA distribution saved")


def plot_class_distribution_stacked(client_data: List[Tuple], save_path: str):
    """Stacked bar chart showing class distribution across all clients."""
    all_classes = sorted(set().union(*[
        set((y.numpy() if isinstance(y, torch.Tensor) else y).tolist()) 
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
    
    fig, ax = plt.subplots(figsize=(12, 7))
    bottom = np.zeros(len(client_data))
    
    for i, cls in enumerate(all_classes):
        bars = ax.bar(range(len(client_data)), distributions[i], bottom=bottom, 
                     label=f'Digit {cls}', color=colors[i], alpha=0.85, edgecolor='black', linewidth=0.5)
        bottom += distributions[i]
        
        # Add count labels on bars
        for j, bar in enumerate(bars):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_y() + height/2.,
                       f'{int(height)}', ha='center', va='center', fontsize=8, fontweight='bold')
    
    ax.set_xticks(range(len(client_data)))
    ax.set_xticklabels([f'Client {i+1}' for i in range(len(client_data))])
    ax.set_xlabel('Clients', fontsize=12)
    ax.set_ylabel('Sample Count', fontsize=12)
    ax.set_title('Class Distribution Across Clients (Stacked)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   ✓ Class distribution plot saved")

# PARTITIONING ANALYSIS
def plot_client_sample_counts(client_indices: List[np.ndarray], save_path: str):
    """Bar plot showing number of samples per client."""
    counts = [len(idx) for idx in client_indices]
    num_clients = len(counts)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(range(1, num_clients + 1), counts, alpha=0.8, 
                  color='skyblue', edgecolor='black', linewidth=1.5)
    
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Client ID', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title('Sample Distribution Across Clients', fontsize=13, fontweight='bold')
    ax.set_xticks(range(1, num_clients + 1))
    ax.set_xticklabels([f'Client {i}' for i in range(1, num_clients + 1)])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    mean_count = np.mean(counts)
    ax.axhline(y=mean_count, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_count:.1f}')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   ✓ Client sample counts plot saved")


def plot_client_similarity_matrix(client_data: List[Tuple[torch.Tensor, torch.Tensor]], 
                                  save_path: str):
    """Heatmap showing cosine similarity between client feature distributions."""
    num_clients = len(client_data)
    
    # Compute mean feature vector for each client
    client_means = []
    for X, _ in client_data:
        X_np = X.numpy() if isinstance(X, torch.Tensor) else X
        client_means.append(X_np.mean(axis=0))
    
    # Compute pairwise cosine similarity
    similarity_matrix = np.zeros((num_clients, num_clients))
    for i in range(num_clients):
        for j in range(num_clients):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                similarity_matrix[i, j] = 1 - cosine(client_means[i], client_means[j])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(similarity_matrix, cmap='YlGnBu', vmin=0, vmax=1, aspect='auto')
    
    # Add text annotations
    for i in range(num_clients):
        for j in range(num_clients):
            text = ax.text(j, i, f'{similarity_matrix[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=10, fontweight='bold')
    
    ax.set_xticks(range(num_clients))
    ax.set_yticks(range(num_clients))
    ax.set_xticklabels([f'Client {i+1}' for i in range(num_clients)])
    ax.set_yticklabels([f'Client {i+1}' for i in range(num_clients)])
    ax.set_xlabel('Client ID', fontsize=12)
    ax.set_ylabel('Client ID', fontsize=12)
    ax.set_title('Client Similarity Matrix (Cosine Similarity)', fontsize=13, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Cosine Similarity', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   ✓ Client similarity matrix saved")


def plot_kl_divergence_heatmap(client_data: List[Tuple[torch.Tensor, torch.Tensor]], 
                               num_classes: int, save_path: str):
    """KL divergence heatmap showing label distribution differences between clients."""
    num_clients = len(client_data)
    
    # Compute label distribution for each client
    client_distributions = []
    for _, y in client_data:
        y_np = y.numpy() if isinstance(y, torch.Tensor) else y
        
        # Create probability distribution with smoothing
        counts = np.bincount(y_np, minlength=num_classes)
        dist = (counts + 1e-10) / (counts.sum() + num_classes * 1e-10)
        client_distributions.append(dist)
    
    # Compute pairwise KL divergence
    kl_matrix = np.zeros((num_clients, num_clients))
    for i in range(num_clients):
        for j in range(num_clients):
            if i == j:
                kl_matrix[i, j] = 0.0
            else:
                kl_matrix[i, j] = entropy(client_distributions[i], client_distributions[j])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(kl_matrix, cmap='Reds', aspect='auto')
    
    # Add text annotations
    for i in range(num_clients):
        for j in range(num_clients):
            text = ax.text(j, i, f'{kl_matrix[i, j]:.3f}',
                          ha="center", va="center", 
                          color="white" if kl_matrix[i, j] > kl_matrix.max()/2 else "black",
                          fontsize=10, fontweight='bold')
    
    ax.set_xticks(range(num_clients))
    ax.set_yticks(range(num_clients))
    ax.set_xticklabels([f'Client {i+1}' for i in range(num_clients)])
    ax.set_yticklabels([f'Client {i+1}' for i in range(num_clients)])
    ax.set_xlabel('Client ID', fontsize=12)
    ax.set_ylabel('Client ID', fontsize=12)
    ax.set_title('Client KL Divergence (Label Distribution)', fontsize=13, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('KL Divergence', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   ✓ KL divergence heatmap saved")

# TRAIN/VAL SPLIT ANALYSIS
def plot_train_val_split_comparison(y_train: np.ndarray, y_val: np.ndarray, save_path: str):
    """Side-by-side bar chart comparing train and validation label distributions."""
    all_classes = sorted(set(y_train) | set(y_val))
    
    train_counts = [np.sum(y_train == cls) for cls in all_classes]
    val_counts = [np.sum(y_val == cls) for cls in all_classes]
    
    train_pct = np.array(train_counts) / len(y_train) * 100
    val_pct = np.array(val_counts) / len(y_val) * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    x = np.arange(len(all_classes))
    width = 0.35
    
    # Absolute counts
    bars1 = ax1.bar(x - width/2, train_counts, width, label='Train', 
                    alpha=0.8, color='steelblue', edgecolor='black')
    bars2 = ax1.bar(x + width/2, val_counts, width, label='Validation', 
                    alpha=0.8, color='coral', edgecolor='black')
    
    ax1.set_xlabel('Class Label', fontsize=12)
    ax1.set_ylabel('Sample Count', fontsize=12)
    ax1.set_title('Train vs Validation: Absolute Counts', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Digit {cls}' for cls in all_classes])
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Percentage distribution
    bars3 = ax2.bar(x - width/2, train_pct, width, label='Train', 
                    alpha=0.8, color='steelblue', edgecolor='black')
    bars4 = ax2.bar(x + width/2, val_pct, width, label='Validation', 
                    alpha=0.8, color='coral', edgecolor='black')
    
    ax2.set_xlabel('Class Label', fontsize=12)
    ax2.set_ylabel('Percentage (%)', fontsize=12)
    ax2.set_title('Train vs Validation: Percentage Distribution', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Digit {cls}' for cls in all_classes])
    ax2.legend(fontsize=11)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   ✓ Train/Val split comparison saved")

# METADATA SUMMARY
def plot_dataset_summary(metadata: Dict, save_path: str):
    """Create dataset summary visualization with donut chart."""
    samples = metadata.get('samples', {})
    train_count = samples.get('train', 0)
    val_count = samples.get('val', 0)
    test_count = samples.get('test', 0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Donut chart
    sizes = [train_count, val_count, test_count]
    labels = ['Train', 'Validation', 'Test']
    colors = ['#FF9999', '#66B2FF', '#99FF99']
    explode = (0.05, 0.05, 0.05)
    
    wedges, texts, autotexts = ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
                                        autopct='%1.1f%%', startangle=90, pctdistance=0.85,
                                        textprops={'fontsize': 11, 'fontweight': 'bold'})
    
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    ax1.add_artist(centre_circle)
    ax1.set_title('Dataset Split Distribution', fontsize=13, fontweight='bold', pad=20)
    
    # Metadata text summary
    ax2.axis('off')
    
    model_type = metadata.get('model_type', 'N/A').upper()
    pca_info = f"{metadata.get('pca_components', 'N/A')}" if metadata.get('pca_applied') else "No"
    explained_var = metadata.get('pca_explained_variance')
    var_text = f" ({explained_var:.4f})" if explained_var else ""
    
    summary_text = f"""
    📊 PREPROCESSING SUMMARY
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    Model Type: {model_type}
    Encoding Range: {metadata.get('encoding_range', 'N/A')}
    
    Dataset Configuration:
    • Digits: {metadata.get('digits', 'N/A')}
    • Original Features: {metadata.get('original_features', 'N/A')}D
    • Final Features: {metadata.get('final_features', 'N/A')}D
    • PCA Applied: {pca_info}{var_text}
    
    Federated Setup:
    • Number of Clients: {metadata.get('num_clients', 'N/A')}
    • Partition Type: {metadata.get('partition_type', 'N/A').upper()}
    • Alpha (Dirichlet): {metadata.get('alpha', 'N/A')}
    
    Sample Counts:
    • Train: {train_count:,}
    • Validation: {val_count:,}
    • Test: {test_count:,}
    • Total: {train_count + val_count + test_count:,}
    
    Other Settings:
    • Validation Split: {metadata.get('val_split', 'N/A')}
    • Random Seed: {metadata.get('seed', 'N/A')}
    """
    
    ax2.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', 
             facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   ✓ Dataset summary saved")

# MAIN ORCHESTRATION FUNCTION
def generate_all_preprocessing_visualizations(
    pca_model=None,
    data_before_scaling=None,
    data_after_scaling=None,
    client_indices=None,
    client_data=None,
    y_train=None,
    y_val=None,
    metadata=None,
    num_classes=None,
    save_dir: str = "./results/preprocessing",
    X_train_original=None
):
    """
    Generate all preprocessing visualizations in one call.
    
    Args:
        pca_model: Fitted PCA model (optional)
        data_before_scaling: Data before scaling (optional)
        data_after_scaling: Data after scaling (optional)
        client_indices: List of client index arrays (optional)
        client_data: List of (X, y) tuples for clients (optional)
        y_train: Training labels (optional)
        y_val: Validation labels (optional)
        metadata: Preprocessing metadata dict (optional)
        num_classes: Number of classes (optional)
        save_dir: Base directory for saving plots
        X_train_original: Original 28x28 images for visualization (optional)
    """
    print("\n" + "="*70)
    print("  GENERATING PREPROCESSING VISUALIZATIONS")
    print("="*70)
    
    viz_folder = create_viz_folder(save_dir)
    print(f"\n📁 Visualization folder: {viz_folder}")
    
    # PCA visualizations
    if pca_model is not None:
        print("\n🔬 PCA Analysis:")
        plot_pca_variance(pca_model, f"{save_dir}/pca_variance.png")
        plot_pca_components_heatmap(pca_model, f"{save_dir}/pca_components_heatmap.png")
        
        if data_after_scaling is not None and y_train is not None:
            plot_pca_2d_projection(data_after_scaling, y_train, 
                                  f"{save_dir}/pca_2d_projection.png")
    
    # Scaling verification
    if data_before_scaling is not None and data_after_scaling is not None:
        print("\n⚖️  Scaling Analysis:")
        plot_scaling_verification(data_before_scaling, data_after_scaling, 
                                 f"{save_dir}/scaling_verification.png")
    
    # Client data visualizations
    if client_data is not None and client_indices is not None:
        print("\n👥 Client Data Analysis:")
        
        # Sample images (for classical mode or if original images provided)
        if X_train_original is not None and y_train is not None:
            visualize_client_sample_images(
                client_data, y_train, client_indices, X_train_original,
                f"{save_dir}/client_sample_images.png"
            )
        
        # PCA distribution (for quantum mode)
        if pca_model is not None:
            visualize_client_pca_distribution(client_data, 
                                             f"{save_dir}/client_pca_distribution.png")
        
        # Class distribution
        plot_class_distribution_stacked(client_data, 
                                       f"{save_dir}/class_distribution_stacked.png")
    
    # Partitioning analysis
    if client_indices is not None:
        print("\n🌐 Partitioning Analysis:")
        plot_client_sample_counts(client_indices, f"{save_dir}/client_sample_counts.png")
    
    if client_data is not None:
        print("\n📊 Client Similarity Analysis:")
        plot_client_similarity_matrix(client_data, f"{save_dir}/client_similarity_matrix.png")
        
        if num_classes is not None:
            plot_kl_divergence_heatmap(client_data, num_classes, 
                                      f"{save_dir}/kl_divergence_heatmap.png")
    
    # Train/Val split verification
    if y_train is not None and y_val is not None:
        print("\n✂️  Train/Val Split Analysis:")
        plot_train_val_split_comparison(y_train, y_val, 
                                       f"{save_dir}/train_val_split_comparison.png")
    
    # Dataset summary
    if metadata is not None:
        print("\n📋 Metadata Summary:")
        plot_dataset_summary(metadata, f"{save_dir}/dataset_summary.png")
    
    print("\n" + "="*70)
    print("✅ ALL PREPROCESSING VISUALIZATIONS COMPLETE")
    print(f"📂 Saved to: {save_dir}/")
    print("="*70 + "\n")