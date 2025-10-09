"""
Visualization utilities for MNIST preprocessing pipeline.
Generates plots for PCA analysis, feature scaling, partitioning, and data distribution.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import torch
from scipy.spatial.distance import cosine
from scipy.stats import entropy


def create_viz_folder(base_path: str = "./results/preprocessing") -> Path:
    """Create and return visualization folder path."""
    folder = Path(base_path)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


# PCA Analysis Plots

def plot_pca_variance(pca_model, save_path: Optional[str] = None):
    """
    Plot explained variance ratio and cumulative variance for PCA components.
    
    Args:
        pca_model: Fitted PCA or IncrementalPCA object
        save_path: Path to save the figure (default: ./results/preprocessing/pca_variance.png)
    """
    if save_path is None:
        save_path = create_viz_folder() / "pca_variance.png"
    
    variance_ratio = pca_model.explained_variance_ratio_
    cumulative_variance = np.cumsum(variance_ratio)
    n_components = len(variance_ratio)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Individual variance bar plot
    ax1.bar(range(1, n_components + 1), variance_ratio, alpha=0.8, color='steelblue', edgecolor='black')
    ax1.set_xlabel('Principal Component', fontsize=12)
    ax1.set_ylabel('Explained Variance Ratio', fontsize=12)
    ax1.set_title('PCA: Variance Explained per Component', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_xticks(range(1, n_components + 1))
    
    # Add value labels on bars
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
    
    # Add text annotation for total variance
    total_var_text = f'Total: {cumulative_variance[-1]:.4f}'
    ax2.text(n_components, cumulative_variance[-1], total_var_text, 
             fontsize=10, ha='right', va='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"📊 PCA variance plot saved: {save_path}")


def plot_pca_components_heatmap(pca_model, save_path: Optional[str] = None, 
                                top_features: int = 50):
    """
    Plot heatmap of PCA component loadings (feature contributions).
    
    Args:
        pca_model: Fitted PCA model
        save_path: Path to save the figure
        top_features: Number of top contributing features to display
    """
    if save_path is None:
        save_path = create_viz_folder() / "pca_components_heatmap.png"
    
    components = pca_model.components_
    n_components, n_features = components.shape
    
    # Select top features based on absolute contribution across all components
    feature_importance = np.abs(components).sum(axis=0)
    top_indices = np.argsort(feature_importance)[-top_features:]
    
    fig, ax = plt.subplots(figsize=(12, max(6, n_components * 0.8)))
    
    # Plot heatmap
    im = ax.imshow(components[:, top_indices], cmap='RdBu_r', aspect='auto', 
                   vmin=-np.abs(components).max(), vmax=np.abs(components).max())
    
    ax.set_xlabel(f'Top {top_features} Original Features (by importance)', fontsize=11)
    ax.set_ylabel('Principal Components', fontsize=11)
    ax.set_title('PCA Component Loadings (Feature Contributions)', fontsize=13, fontweight='bold')
    ax.set_yticks(range(n_components))
    ax.set_yticklabels([f'PC{i+1}' for i in range(n_components)])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Loading Strength', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"📊 PCA components heatmap saved: {save_path}")


#  Feature Scaling Verification 

def plot_scaling_verification(data_before: np.ndarray, data_after: np.ndarray,
                              save_path: Optional[str] = None, sample_size: int = 5000):
    """
    Compare feature distributions before and after scaling.
    
    Args:
        data_before: Data before scaling (normalized to [0,1])
        data_after: Data after MinMaxScaler to [-1,1]
        save_path: Path to save the figure
        sample_size: Number of samples to use for visualization
    """
    if save_path is None:
        save_path = create_viz_folder() / "scaling_verification.png"
    
    # Sample data for faster plotting
    if len(data_before) > sample_size:
        indices = np.random.choice(len(data_before), sample_size, replace=False)
        data_before = data_before[indices]
        data_after = data_after[indices]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Flatten for histogram
    before_flat = data_before.flatten()
    after_flat = data_after.flatten()
    
    # Histogram before scaling
    axes[0, 0].hist(before_flat, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_xlabel('Pixel Value', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].set_title('Before Scaling: [0, 1] Range', fontsize=12, fontweight='bold')
    axes[0, 0].axvline(before_flat.mean(), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {before_flat.mean():.3f}')
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Histogram after scaling
    axes[0, 1].hist(after_flat, bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].set_xlabel('Pixel Value', fontsize=11)
    axes[0, 1].set_ylabel('Frequency', fontsize=11)
    axes[0, 1].set_title('After Scaling: [-1, 1] Range (Quantum-Ready)', fontsize=12, fontweight='bold')
    axes[0, 1].axvline(after_flat.mean(), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {after_flat.mean():.3f}')
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Box plot per feature (show first 10 features)
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
    axes[1, 1].axhline(y=-1, color='red', linestyle='--', linewidth=1, alpha=0.5)
    axes[1, 1].axhline(y=1, color='red', linestyle='--', linewidth=1, alpha=0.5)
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"📊 Scaling verification plot saved: {save_path}")


#  Partitioning Distribution 

def plot_client_sample_counts(client_indices: List[np.ndarray], 
                              save_path: Optional[str] = None):
    """
    Bar plot showing number of samples per client.
    
    Args:
        client_indices: List of index arrays for each client
        save_path: Path to save the figure
    """
    if save_path is None:
        save_path = create_viz_folder() / "client_sample_counts.png"
    
    counts = [len(idx) for idx in client_indices]
    num_clients = len(counts)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(range(1, num_clients + 1), counts, alpha=0.8, 
                  color='skyblue', edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
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
    
    # Add mean line
    mean_count = np.mean(counts)
    ax.axhline(y=mean_count, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_count:.1f}')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"📊 Client sample counts plot saved: {save_path}")


def plot_client_similarity_matrix(client_data: List[Tuple[torch.Tensor, torch.Tensor]], 
                                  save_path: Optional[str] = None):
    """
    Heatmap showing cosine similarity between client feature distributions.
    
    Args:
        client_data: List of (X, y) tuples for each client
        save_path: Path to save the figure
    """
    if save_path is None:
        save_path = create_viz_folder() / "client_similarity_matrix.png"
    
    num_clients = len(client_data)
    
    # Compute mean feature vector for each client
    client_means = []
    for X, _ in client_data:
        if isinstance(X, torch.Tensor):
            X_np = X.numpy()
        else:
            X_np = X
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
                          ha="center", va="center", color="black", fontsize=10)
    
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
    print(f"📊 Client similarity matrix saved: {save_path}")


def plot_kl_divergence_heatmap(client_data: List[Tuple[torch.Tensor, torch.Tensor]], 
                               num_classes: int, save_path: Optional[str] = None):
    """
    KL divergence heatmap showing label distribution differences between clients.
    
    Args:
        client_data: List of (X, y) tuples for each client
        num_classes: Number of classes in the dataset
        save_path: Path to save the figure
    """
    if save_path is None:
        save_path = create_viz_folder() / "kl_divergence_heatmap.png"
    
    num_clients = len(client_data)
    
    # Compute label distribution for each client
    client_distributions = []
    for _, y in client_data:
        if isinstance(y, torch.Tensor):
            y_np = y.numpy()
        else:
            y_np = y
        
        # Create probability distribution
        counts = np.bincount(y_np, minlength=num_classes)
        dist = (counts + 1e-10) / (counts.sum() + num_classes * 1e-10)  # Add smoothing
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
                          fontsize=10)
    
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
    print(f"📊 KL divergence heatmap saved: {save_path}")


#  Validation Split Verification 

def plot_train_val_split_comparison(y_train: np.ndarray, y_val: np.ndarray, 
                                    save_path: Optional[str] = None):
    """
    Side-by-side bar chart comparing train and validation label distributions.
    
    Args:
        y_train: Training labels
        y_val: Validation labels
        save_path: Path to save the figure
    """
    if save_path is None:
        save_path = create_viz_folder() / "train_val_split_comparison.png"
    
    # Get unique classes
    all_classes = sorted(set(y_train) | set(y_val))
    
    # Count occurrences
    train_counts = [np.sum(y_train == cls) for cls in all_classes]
    val_counts = [np.sum(y_val == cls) for cls in all_classes]
    
    # Calculate percentages
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
    print(f"📊 Train/Val split comparison saved: {save_path}")


#  Metadata Summary 

def plot_dataset_summary(metadata: Dict, save_path: Optional[str] = None):
    """
    Create dataset size summary visualization with donut chart.
    
    Args:
        metadata: Preprocessing metadata dictionary
        save_path: Path to save the figure
    """
    if save_path is None:
        save_path = create_viz_folder() / "dataset_summary.png"
    
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
    
    # Draw circle for donut effect
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    ax1.add_artist(centre_circle)
    ax1.set_title('Dataset Split Distribution', fontsize=13, fontweight='bold', pad=20)
    
    # Metadata text summary
    ax2.axis('off')
    summary_text = f"""
    📊 PREPROCESSING SUMMARY
    
    Dataset Configuration:
    • Digits: {metadata.get('digits', 'N/A')}
    • Feature Dimension: {metadata.get('feature_dim', 'N/A')}
    • PCA Applied: {metadata.get('apply_pca', False)}
    • PCA Components: {metadata.get('pca_components', 'N/A')}
    
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
    
    ax2.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', 
             facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"📊 Dataset summary saved: {save_path}")


# Main Orchestration Function 

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
    save_dir: str = "./results/preprocessing"
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
    """
    print("\n" )
    print("🎨 GENERATING PREPROCESSING VISUALIZATIONS")
    
    create_viz_folder(save_dir)
    
    # PCA visualizations
    if pca_model is not None:
        print("\n📊 Generating PCA analysis plots...")
        plot_pca_variance(pca_model, f"{save_dir}/pca_variance.png")
        plot_pca_components_heatmap(pca_model, f"{save_dir}/pca_components_heatmap.png")
    
    # Scaling verification
    if data_before_scaling is not None and data_after_scaling is not None:
        print("\n📊 Generating scaling verification plots...")
        plot_scaling_verification(data_before_scaling, data_after_scaling, 
                                 f"{save_dir}/scaling_verification.png")
    
    # Partitioning visualizations
    if client_indices is not None:
        print("\n📊 Generating partitioning plots...")
        plot_client_sample_counts(client_indices, f"{save_dir}/client_sample_counts.png")
    
    if client_data is not None:
        print("\n📊 Generating client similarity plots...")
        plot_client_similarity_matrix(client_data, f"{save_dir}/client_similarity_matrix.png")
        
        if num_classes is not None:
            plot_kl_divergence_heatmap(client_data, num_classes, 
                                      f"{save_dir}/kl_divergence_heatmap.png")
    
    # Train/Val split verification
    if y_train is not None and y_val is not None:
        print("\n📊 Generating train/val comparison...")
        plot_train_val_split_comparison(y_train, y_val, 
                                       f"{save_dir}/train_val_split_comparison.png")
    
    # Dataset summary
    if metadata is not None:
        print("\n📊 Generating dataset summary...")
        plot_dataset_summary(metadata, f"{save_dir}/dataset_summary.png")
    
    print("\n" )
    print("✅ ALL PREPROCESSING VISUALIZATIONS COMPLETE")
    print(f" Saved to: {save_dir}")