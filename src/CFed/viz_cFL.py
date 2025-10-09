"""
Visualization utilities for Classical Federated Learning (CFL).
Generates plots for client performance, training dynamics, model evaluation, and interpretability.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd


def create_viz_folder(base_path: str = "./results/cfl") -> Path:
    """Create and return visualization folder path."""
    folder = Path(base_path)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


# Per-Client Performance 
def plot_client_training_losses(client_losses_per_round: List[List[float]], 
                                save_path: Optional[str] = None):
    """
    Plot per-client training losses across federated rounds.
    
    Args:
        client_losses_per_round: List of lists, where each inner list contains
                                 client losses for that round
        save_path: Path to save the figure
    """
    if save_path is None:
        save_path = create_viz_folder() / "client_training_losses.png"
    
    num_rounds = len(client_losses_per_round)
    
    # Transpose to get per-client history
    client_losses_dict = {}
    for round_idx, losses in enumerate(client_losses_per_round):
        for client_idx, loss in enumerate(losses):
            if client_idx not in client_losses_dict:
                client_losses_dict[client_idx] = []
            client_losses_dict[client_idx].append((round_idx, loss))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Line plot
    colors = plt.cm.tab10(np.linspace(0, 1, len(client_losses_dict)))
    for client_idx, color in zip(sorted(client_losses_dict.keys()), colors):
        rounds, losses = zip(*client_losses_dict[client_idx])
        ax1.plot(rounds, losses, marker='o', linewidth=2, markersize=4, 
                label=f'Client {client_idx+1}', color=color, alpha=0.8)
    
    ax1.set_xlabel('Federated Round', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Per-Client Training Loss Over Rounds', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(alpha=0.3, linestyle='--')
    
    # Boxplot
    data_for_boxplot = [client_losses_dict[i] for i in sorted(client_losses_dict.keys())]
    data_for_boxplot = [[loss for _, loss in client_data] for client_data in data_for_boxplot]
    
    bp = ax2.boxplot(data_for_boxplot, labels=[f'C{i+1}' for i in range(len(data_for_boxplot))],
                     patch_artist=True, showmeans=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax2.set_xlabel('Client ID', fontsize=12)
    ax2.set_ylabel('Training Loss', fontsize=12)
    ax2.set_title('Loss Distribution per Client', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"📊 Client training losses plot saved: {save_path}")


def plot_client_final_accuracies(client_data: List[Tuple[torch.Tensor, torch.Tensor]],
                                 global_model: nn.Module,
                                 device: torch.device,
                                 save_path: Optional[str] = None):
    """
    Bar plot of final accuracy for each client using the global model.
    
    Args:
        client_data: List of (X, y) tuples for each client
        global_model: Trained global model
        device: torch device
        save_path: Path to save the figure
    """
    if save_path is None:
        save_path = create_viz_folder() / "client_final_accuracies.png"
    
    global_model.eval()
    client_accuracies = []
    client_sample_counts = []
    
    with torch.no_grad():
        for X_client, y_client in client_data:
            X_client = X_client.view(-1, 1, 28, 28).to(device)
            y_client = y_client.to(device)
            
            outputs = global_model(X_client)
            predictions = outputs.argmax(1)
            accuracy = (predictions == y_client).float().mean().item()
            
            client_accuracies.append(accuracy)
            client_sample_counts.append(len(y_client))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    client_ids = [f'Client {i+1}\n(n={n})' for i, n in enumerate(client_sample_counts)]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(client_accuracies)))
    
    bars = ax.bar(client_ids, client_accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, acc in zip(bars, client_accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add mean line
    mean_acc = np.mean(client_accuracies)
    ax.axhline(y=mean_acc, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_acc:.3f}')
    
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Final Client Accuracies (Global Model)', fontsize=13, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"📊 Client final accuracies plot saved: {save_path}")


# ==================== Client Data Distribution ====================

def plot_client_label_distribution_heatmap(client_data: List[Tuple[torch.Tensor, torch.Tensor]],
                                          num_classes: int,
                                          save_path: Optional[str] = None):
    """
    Heatmap showing label distribution across clients.
    
    Args:
        client_data: List of (X, y) tuples for each client
        num_classes: Number of classes
        save_path: Path to save the figure
    """
    if save_path is None:
        save_path = create_viz_folder() / "client_label_distribution_heatmap.png"
    
    num_clients = len(client_data)
    label_counts = np.zeros((num_clients, num_classes))
    
    for i, (_, y_client) in enumerate(client_data):
        if isinstance(y_client, torch.Tensor):
            y_np = y_client.numpy()
        else:
            y_np = y_client
        
        for label in range(num_classes):
            label_counts[i, label] = np.sum(y_np == label)
    
    fig, ax = plt.subplots(figsize=(10, max(6, num_clients * 0.8)))
    
    im = sns.heatmap(label_counts, annot=True, fmt='.0f', cmap='YlGnBu', 
                     cbar_kws={'label': 'Sample Count'}, ax=ax)
    
    ax.set_xlabel('Class Label', fontsize=12)
    ax.set_ylabel('Client ID', fontsize=12)
    ax.set_title('Client-wise Label Distribution', fontsize=13, fontweight='bold')
    ax.set_xticklabels([f'Class {i}' for i in range(num_classes)])
    ax.set_yticklabels([f'Client {i+1}' for i in range(num_clients)], rotation=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"📊 Client label distribution heatmap saved: {save_path}")


# ==================== Training Dynamics ====================

def plot_training_dynamics(metrics: Dict, save_path: Optional[str] = None):
    """
    Comprehensive training dynamics visualization.
    
    Args:
        metrics: Dictionary containing 'test_accuracies', 'test_losses', 'train_losses'
        save_path: Path to save the figure
    """
    if save_path is None:
        save_path = create_viz_folder() / "training_dynamics.png"
    
    test_acc = metrics['test_accuracies']
    test_loss = metrics['test_losses']
    train_loss = metrics['train_losses']
    rounds = list(range(len(test_acc)))
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy curve
    axes[0, 0].plot(rounds, test_acc, marker='o', linewidth=2.5, markersize=5, 
                    color='#2E86AB', label='Test Accuracy')
    axes[0, 0].fill_between(rounds, test_acc, alpha=0.2, color='#2E86AB')
    axes[0, 0].set_xlabel('Federated Round', fontsize=11)
    axes[0, 0].set_ylabel('Accuracy', fontsize=11)
    axes[0, 0].set_title('Test Accuracy Evolution', fontsize=12, fontweight='bold')
    axes[0, 0].grid(alpha=0.3, linestyle='--')
    axes[0, 0].set_ylim([0, 1.05])
    axes[0, 0].legend(fontsize=10)
    
    # Loss curves
    axes[0, 1].plot(rounds, test_loss, marker='s', linewidth=2, markersize=4,
                    label='Test Loss', color='#A23B72')
    axes[0, 1].plot(rounds, train_loss, marker='^', linewidth=2, markersize=4,
                    label='Train Loss', color='#F18F01')
    axes[0, 1].set_xlabel('Federated Round', fontsize=11)
    axes[0, 1].set_ylabel('Loss', fontsize=11)
    axes[0, 1].set_title('Loss Curves', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(alpha=0.3, linestyle='--')
    
    # Accuracy improvement per round
    acc_improvement = np.diff([0] + test_acc)
    axes[1, 0].bar(rounds[1:], acc_improvement[1:], alpha=0.7, color='teal', edgecolor='black')
    axes[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=1)
    axes[1, 0].set_xlabel('Federated Round', fontsize=11)
    axes[1, 0].set_ylabel('Accuracy Δ', fontsize=11)
    axes[1, 0].set_title('Accuracy Improvement per Round', fontsize=12, fontweight='bold')
    axes[1, 0].grid(axis='y', alpha=0.3, linestyle='--')
    
    # Loss reduction per round
    loss_reduction = -np.diff([train_loss[0]] + train_loss)
    axes[1, 1].bar(rounds[1:], loss_reduction[1:], alpha=0.7, color='coral', edgecolor='black')
    axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=1)
    axes[1, 1].set_xlabel('Federated Round', fontsize=11)
    axes[1, 1].set_ylabel('Loss Reduction', fontsize=11)
    axes[1, 1].set_title('Training Loss Reduction per Round', fontsize=12, fontweight='bold')
    axes[1, 1].grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"📊 Training dynamics plot saved: {save_path}")


# ==================== Model Evaluation ====================

def plot_confusion_matrix(model: nn.Module, test_data: Tuple[torch.Tensor, torch.Tensor],
                         device: torch.device, class_names: List[str] = None,
                         save_path: Optional[str] = None):
    """
    Generate and plot confusion matrix for the final model.
    
    Args:
        model: Trained model
        test_data: Test dataset (X, y)
        device: torch device
        class_names: List of class names for labels
        save_path: Path to save the figure
    """
    if save_path is None:
        save_path = create_viz_folder() / "confusion_matrix.png"
    
    model.eval()
    X_test, y_test = test_data
    X_test = X_test.view(-1, 1, 28, 28).to(device)
    y_test = y_test.to(device)
    
    with torch.no_grad():
        outputs = model(X_test)
        predictions = outputs.argmax(1).cpu().numpy()
    
    y_true = y_test.cpu().numpy()
    cm = confusion_matrix(y_true, predictions)
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(cm.shape[0])]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', 
                     xticklabels=class_names, yticklabels=class_names,
                     cbar_kws={'label': 'Count'}, ax=ax)
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix - Global Model', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"📊 Confusion matrix saved: {save_path}")
    
    # Generate classification report
    report = classification_report(y_true, predictions, target_names=class_names)
    report_path = Path(save_path).parent / "classification_report.txt"
    with open(report_path, 'w') as f:
        f.write("Classification Report\n")
        f.write("=" * 50 + "\n")
        f.write(report)
    print(f"📄 Classification report saved: {report_path}")


# ==================== Metric Correlation ====================

def plot_metric_correlation(metrics: Dict, save_path: Optional[str] = None):
    """
    Correlation heatmap between training metrics.
    
    Args:
        metrics: Dictionary containing training metrics
        save_path: Path to save the figure
    """
    if save_path is None:
        save_path = create_viz_folder() / "metric_correlation.png"
    
    df = pd.DataFrame({
        'Test Accuracy': metrics['test_accuracies'][1:],  # Skip initial
        'Test Loss': metrics['test_losses'][1:],
        'Train Loss': metrics['train_losses'][1:]
    })
    
    corr_matrix = df.corr()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                center=0, vmin=-1, vmax=1, square=True,
                cbar_kws={'label': 'Correlation'}, ax=ax)
    
    ax.set_title('Metric Correlation Heatmap', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"📊 Metric correlation heatmap saved: {save_path}")


# ==================== Feature Visualization ====================

def plot_conv_feature_maps(model: nn.Module, sample_input: torch.Tensor,
                          device: torch.device, save_path: Optional[str] = None):
    """
    Visualize feature maps from the first convolutional layer.
    
    Args:
        model: Trained CNN model
        sample_input: Input image tensor (1, 28, 28)
        device: torch device
        save_path: Path to save the figure
    """
    if save_path is None:
        save_path = create_viz_folder() / "conv_feature_maps.png"
    
    model.eval()
    
    # Hook to capture first conv layer output
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    model.conv1.register_forward_hook(get_activation('conv1'))
    
    # Forward pass
    sample_input = sample_input.unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        _ = model(sample_input)
    
    feature_maps = activation['conv1'].squeeze(0).cpu().numpy()  # (num_filters, H, W)
    num_filters = feature_maps.shape[0]
    
    # Plot grid
    n_cols = 8
    n_rows = (num_filters + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 2))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for i in range(num_filters):
        axes[i].imshow(feature_maps[i], cmap='viridis')
        axes[i].set_title(f'Filter {i+1}', fontsize=8)
        axes[i].axis('off')
    
    # Hide extra subplots
    for i in range(num_filters, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Convolutional Layer 1 Feature Maps', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"📊 Feature maps visualization saved: {save_path}")


# ==================== Embedding Visualization ====================

def plot_embeddings(model: nn.Module, test_data: Tuple[torch.Tensor, torch.Tensor],
                   device: torch.device, method: str = 'tsne',
                   save_path: Optional[str] = None):
    """
    Visualize learned embeddings using t-SNE or PCA.
    
    Args:
        model: Trained model
        test_data: Test dataset (X, y)
        device: torch device
        method: 'tsne' or 'pca'
        save_path: Path to save the figure
    """
    if save_path is None:
        save_path = create_viz_folder() / f"embeddings_{method}.png"
    
    model.eval()
    X_test, y_test = test_data
    X_test = X_test.view(-1, 1, 28, 28).to(device)
    
    # Extract embeddings (before final FC layer)
    embeddings = []
    def get_embedding(module, input, output):
        embeddings.append(output.detach().cpu())
    
    hook = model.fc1.register_forward_hook(get_embedding)
    
    with torch.no_grad():
        _ = model(X_test)
    
    hook.remove()
    
    embeddings = torch.cat(embeddings, dim=0).numpy()
    labels = y_test.cpu().numpy()
    
    # Dimensionality reduction
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        title = 't-SNE Visualization of Learned Embeddings'
    else:
        reducer = PCA(n_components=2)
        title = 'PCA Visualization of Learned Embeddings'
    
    embeddings_2d = reducer.fit_transform(embeddings)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        mask = labels == label
        ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                  c=[color], label=f'Class {label}', alpha=0.6, s=30)
    
    ax.set_xlabel('Component 1', fontsize=12)
    ax.set_ylabel('Component 2', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"📊 Embeddings visualization saved: {save_path}")


# ==================== Summary Metrics Table ====================

def create_summary_table(metrics: Dict, config: Dict, save_path: Optional[str] = None):
    """
    Create a summary table of training metrics and configuration.
    
    Args:
        metrics: Dictionary of training metrics
        config: Dictionary of configuration parameters
        save_path: Path to save the table
    """
    if save_path is None:
        save_path = create_viz_folder() / "summary_metrics.csv"
    
    summary_data = {
        'Metric': [
            'Final Test Accuracy',
            'Final Test Loss',
            'Final Train Loss',
            'Best Test Accuracy',
            'Convergence Round (Best Acc)',
            'Total Rounds',
            'Number of Clients',
            'Local Epochs',
            'Learning Rate',
            'Batch Size',
            'Partition Type'
        ],
        'Value': [
            f"{metrics['test_accuracies'][-1]:.4f}",
            f"{metrics['test_losses'][-1]:.4f}",
            f"{metrics['train_losses'][-1]:.4f}",
            f"{max(metrics['test_accuracies']):.4f}",
            f"{np.argmax(metrics['test_accuracies'])}",
            f"{len(metrics['test_accuracies']) - 1}",
            config.get('num_clients', 'N/A'),
            config.get('local_epochs', 'N/A'),
            config.get('learning_rate', 'N/A'),
            config.get('batch_size', 'N/A'),
            config.get('partition_type', 'N/A')
        ]
    }
    
    df = pd.DataFrame(summary_data)
    df.to_csv(save_path, index=False)
    print(f"📄 Summary table saved: {save_path}")
    
    # Also create a visual table
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values, colLabels=df.columns,
                    cellLoc='left', loc='center',
                    colWidths=[0.6, 0.4])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title('Training Summary', fontsize=14, fontweight='bold', pad=20)
    
    table_img_path = Path(save_path).parent / "summary_metrics_table.png"
    plt.savefig(table_img_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"📊 Summary table image saved: {table_img_path}")


# ==================== Main Orchestration Function ====================

def generate_all_cfl_visualizations(
    metrics: Dict,
    config: Dict,
    client_data: List[Tuple[torch.Tensor, torch.Tensor]],
    test_data: Tuple[torch.Tensor, torch.Tensor],
    global_model: nn.Module,
    device: torch.device,
    client_losses_per_round: Optional[List[List[float]]] = None,
    class_names: Optional[List[str]] = None,
    save_dir: str = "./results/cfl"
):
    """
    Generate all CFL visualizations in one call.
    
    Args:
        metrics: Dictionary with 'test_accuracies', 'test_losses', 'train_losses'
        config: Configuration dictionary
        client_data: List of (X, y) tuples for clients
        test_data: Test dataset (X, y)
        global_model: Trained global model
        device: torch device
        client_losses_per_round: Optional list of per-client losses per round
        class_names: Optional list of class names
        save_dir: Base directory for saving plots
    """
    print("\n" + "="*60)
    print("🎨 GENERATING CLASSICAL FL VISUALIZATIONS")
    print("="*60)
    
    create_viz_folder(save_dir)
    
    # Training dynamics
    print("\n📊 Generating training dynamics plots...")
    plot_training_dynamics(metrics, f"{save_dir}/training_dynamics.png")
    
    # Per-client performance
    if client_losses_per_round is not None and len(client_losses_per_round) > 0:
        print("\n📊 Generating per-client loss plots...")
        plot_client_training_losses(client_losses_per_round, 
                                    f"{save_dir}/client_training_losses.png")
    
    print("\n📊 Generating client final accuracies...")
    plot_client_final_accuracies(client_data, global_model, device,
                                 f"{save_dir}/client_final_accuracies.png")
    
    # Client data distribution
    print("\n📊 Generating client label distribution...")
    num_classes = len(torch.unique(test_data[1]))
    plot_client_label_distribution_heatmap(client_data, num_classes,
                                          f"{save_dir}/client_label_distribution_heatmap.png")
    
    # Model evaluation
    print("\n📊 Generating confusion matrix...")
    plot_confusion_matrix(global_model, test_data, device, class_names,
                         f"{save_dir}/confusion_matrix.png")
    
    # Metric correlation
    print("\n📊 Generating metric correlation heatmap...")
    plot_metric_correlation(metrics, f"{save_dir}/metric_correlation.png")
    
    # Feature maps (use first test sample)
    print("\n📊 Generating feature map visualization...")
    sample_input = test_data[0][0].view(1, 28, 28)
    plot_conv_feature_maps(global_model, sample_input, device,
                          f"{save_dir}/conv_feature_maps.png")
    
    # Embeddings
    print("\n📊 Generating embedding visualizations...")
    plot_embeddings(global_model, test_data, device, method='tsne',
                   save_path=f"{save_dir}/embeddings_tsne.png")
    plot_embeddings(global_model, test_data, device, method='pca',
                   save_path=f"{save_dir}/embeddings_pca.png")
    
    # Summary table
    print("\n📊 Generating summary metrics table...")
    create_summary_table(metrics, config, f"{save_dir}/summary_metrics.csv")
    
    print("\n" + "="*60)
    print("✅ ALL CLASSICAL FL VISUALIZATIONS COMPLETE")
    print(f"📁 Saved to: {save_dir}")
    print("="*60 + "\n")