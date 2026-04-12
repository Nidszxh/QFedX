
# Visualization utilities for Quantum Neural Network (QNN) Training

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def ensure_viz_dir(save_dir: str = "./visualizations/qnn") -> Path:

    viz_path = Path(save_dir)
    viz_path.mkdir(parents=True, exist_ok=True)
    return viz_path


def plot_training_curves(train_losses: List[float],
                         val_losses: List[float],
                         train_accs: List[float],
                         val_accs: List[float],
                         save_dir: str = "./visualizations/qnn",
                         filename: str = "training_curves.png") -> str:

    viz_path = ensure_viz_dir(save_dir)
    
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    ax1.plot(epochs, train_losses, 'o-', label='Train Loss', color='#3498db', linewidth=2, markersize=5)
    ax1.plot(epochs, val_losses, 's-', label='Val Loss', color='#e74c3c', linewidth=2, markersize=5)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training & Validation Loss', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax2.plot(epochs, train_accs, 'o-', label='Train Acc', color='#2ecc71', linewidth=2, markersize=5)
    ax2.plot(epochs, val_accs, 's-', label='Val Acc', color='#f39c12', linewidth=2, markersize=5)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('Training & Validation Accuracy', fontsize=13, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    
    # Add best validation accuracy marker
    if val_accs:
        best_epoch = np.argmax(val_accs) + 1
        best_acc = max(val_accs)
        ax2.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5)
        ax2.text(best_epoch, best_acc, f' Best: {best_acc:.4f}', 
                fontsize=9, va='bottom', ha='left', color='red')
    
    plt.suptitle('QNN Training Dynamics', fontsize=15, fontweight='bold', y=1.02)
    
    save_path = viz_path / filename
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return str(save_path)


def plot_learning_rate_schedule(lr_values: List[float],
                                save_dir: str = "./visualizations/qnn",
                                filename: str = "lr_schedule.png") -> str:

    viz_path = ensure_viz_dir(save_dir)
    
    epochs = range(1, len(lr_values) + 1)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, lr_values, 'o-', color='purple', linewidth=2, markersize=5)
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
    ax.set_title('Learning Rate Schedule (CosineAnnealingWarmRestarts)', 
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    save_path = viz_path / filename
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return str(save_path)


def plot_confusion_matrix(y_true: np.ndarray,
                         y_pred: np.ndarray,
                         class_names: Optional[List[str]] = None,
                         save_dir: str = "./visualizations/qnn",
                         filename: str = "confusion_matrix.png") -> str:

    viz_path = ensure_viz_dir(save_dir)
    
    cm = confusion_matrix(y_true, y_pred)
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(cm.shape[0])]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, ax=ax)
    
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    # Add accuracy annotation
    accuracy = np.trace(cm) / np.sum(cm)
    ax.text(0.5, -0.15, f'Overall Accuracy: {accuracy:.4f}', 
            ha='center', va='top', transform=ax.transAxes, 
            fontsize=11, fontweight='bold')
    
    save_path = viz_path / filename
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return str(save_path)


def plot_per_class_metrics(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          class_names: Optional[List[str]] = None,
                          save_dir: str = "./visualizations/qnn",
                          filename: str = "per_class_metrics.png") -> str:

    viz_path = ensure_viz_dir(save_dir)
    
    # Get classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    
    if class_names is None:
        class_names = [f'Class {i}' for i in sorted(set(y_true))]
    
    # Extract metrics
    metrics = ['precision', 'recall', 'f1-score']
    data = {metric: [] for metric in metrics}
    
    for i, class_name in enumerate(class_names):
        class_key = str(i)
        if class_key in report:
            for metric in metrics:
                data[metric].append(report[class_key][metric])
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(class_names))
    width = 0.25
    
    bars1 = ax.bar(x - width, data['precision'], width, label='Precision', 
                   color='#3498db', alpha=0.8)
    bars2 = ax.bar(x, data['recall'], width, label='Recall', 
                   color='#2ecc71', alpha=0.8)
    bars3 = ax.bar(x + width, data['f1-score'], width, label='F1-Score', 
                   color='#e74c3c', alpha=0.8)
    
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Classification Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    save_path = viz_path / filename
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return str(save_path)


def plot_roc_curves(y_true: np.ndarray,
                   y_scores: np.ndarray,
                   n_classes: int,
                   class_names: Optional[List[str]] = None,
                   save_dir: str = "./visualizations/qnn",
                   filename: str = "roc_curves.png") -> str:

    viz_path = ensure_viz_dir(save_dir)
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(n_classes)]
    
    # Binarize labels
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    # Compute ROC curve and AUC for each class
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.Set2(np.linspace(0, 1, n_classes))
    
    for i, color in zip(range(n_classes), colors):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color=color, lw=2,
               label=f'{class_names[i]} (AUC = {roc_auc:.3f})')
    
    # Plot diagonal
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curves (One-vs-Rest)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    
    save_path = viz_path / filename
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return str(save_path)


def plot_parameter_distribution(model,
                                save_dir: str = "./visualizations/qnn",
                                filename: str = "parameter_distribution.png") -> str:

    viz_path = ensure_viz_dir(save_dir)
    
    # Extract parameters
    q_params = []
    c_params = []
    
    for param in model.get_quantum_params():
        q_params.extend(param.detach().cpu().flatten().numpy())
    
    for param in model.get_classical_params():
        c_params.extend(param.detach().cpu().flatten().numpy())
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Quantum parameters
    axes[0].hist(q_params, bins=50, color='#9b59b6', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Parameter Value', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[0].set_title(f'Quantum Parameters (n={len(q_params)})', 
                     fontsize=13, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Classical parameters
    axes[1].hist(c_params, bins=50, color='#3498db', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Parameter Value', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[1].set_title(f'Classical Parameters (n={len(c_params)})', 
                     fontsize=13, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.suptitle('Parameter Distribution Analysis', fontsize=15, fontweight='bold', y=1.02)
    
    save_path = viz_path / filename
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return str(save_path)


def plot_quantum_circuit_depth(n_qubits_range: List[int],
                               n_layers_range: List[int],
                               entanglement_types: List[str],
                               save_dir: str = "./visualizations/qnn",
                               filename: str = "circuit_depth_analysis.png") -> str:

    viz_path = ensure_viz_dir(save_dir)
    
    fig, axes = plt.subplots(1, len(entanglement_types), figsize=(6*len(entanglement_types), 5))
    
    if len(entanglement_types) == 1:
        axes = [axes]
    
    for idx, entanglement in enumerate(entanglement_types):
        ax = axes[idx]
        
        # Create depth matrix
        depth_matrix = np.zeros((len(n_layers_range), len(n_qubits_range)))
        
        for i, n_layers in enumerate(n_layers_range):
            for j, n_qubits in enumerate(n_qubits_range):
                # Estimate depth (simplified calculation)
                gate_layers = n_layers * 2  # RY and RZ per layer
                if entanglement == 'linear':
                    cnot_layers = n_layers * (n_qubits - 1)
                elif entanglement == 'circular':
                    cnot_layers = n_layers * n_qubits
                else:  # full
                    cnot_layers = n_layers * (n_qubits * (n_qubits - 1) // 2)
                
                depth_matrix[i, j] = gate_layers + cnot_layers
        
        # Plot heatmap
        im = ax.imshow(depth_matrix, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(range(len(n_qubits_range)))
        ax.set_yticks(range(len(n_layers_range)))
        ax.set_xticklabels(n_qubits_range)
        ax.set_yticklabels(n_layers_range)
        ax.set_xlabel('Number of Qubits', fontsize=11, fontweight='bold')
        ax.set_ylabel('Number of Layers', fontsize=11, fontweight='bold')
        ax.set_title(f'{entanglement.capitalize()} Entanglement', 
                    fontsize=12, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Circuit Depth', rotation=270, labelpad=15)
        
        # Add text annotations
        for i in range(len(n_layers_range)):
            for j in range(len(n_qubits_range)):
                text = ax.text(j, i, f'{int(depth_matrix[i, j])}',
                             ha="center", va="center", color="black", fontsize=8)
    
    plt.suptitle('Circuit Depth Analysis', fontsize=15, fontweight='bold', y=1.02)
    
    save_path = viz_path / filename
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return str(save_path)


def plot_feature_space_tsne(X: np.ndarray,
                            y: np.ndarray,
                            class_names: Optional[List[str]] = None,
                            title: str = "t-SNE Feature Space",
                            save_dir: str = "./visualizations/qnn",
                            filename: str = "tsne_feature_space.png") -> str:

    viz_path = ensure_viz_dir(save_dir)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X)
    
    if class_names is None:
        class_names = [f'Class {i}' for i in sorted(set(y))]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(class_names)))
    
    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        mask = y == i
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                  c=[color], label=class_name, alpha=0.6, s=50, edgecolors='k')
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    save_path = viz_path / filename
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return str(save_path)


def plot_training_time_comparison(classical_times: List[float],
                                  quantum_times: List[float],
                                  save_dir: str = "./visualizations/qnn",
                                  filename: str = "time_comparison.png") -> str:

    viz_path = ensure_viz_dir(save_dir)
    
    epochs = range(1, len(classical_times) + 1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(epochs, classical_times, 'o-', label='Classical', 
           color='#3498db', linewidth=2, markersize=6)
    ax.plot(epochs, quantum_times, 's-', label='Quantum', 
           color='#9b59b6', linewidth=2, markersize=6)
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Add average time annotations
    avg_classical = np.mean(classical_times)
    avg_quantum = np.mean(quantum_times)
    
    ax.axhline(y=avg_classical, color='#3498db', linestyle='--', alpha=0.5)
    ax.axhline(y=avg_quantum, color='#9b59b6', linestyle='--', alpha=0.5)
    
    ax.text(0.02, 0.98, f'Avg Classical: {avg_classical:.2f}s\nAvg Quantum: {avg_quantum:.2f}s',
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    save_path = viz_path / filename
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return str(save_path)


def generate_all_qnn_plots(history: Dict,
                           model,
                           X_test: Optional[np.ndarray] = None,
                           y_test: Optional[np.ndarray] = None,
                           y_pred: Optional[np.ndarray] = None,
                           y_scores: Optional[np.ndarray] = None,
                           class_names: Optional[List[str]] = None,
                           save_dir: str = "./visualizations/qnn") -> Dict[str, str]:
    
    saved_plots = {}
    
    try:
        # Training curves
        if 'train_losses' in history and 'val_losses' in history:
            path = plot_training_curves(
                history['train_losses'],
                history['val_losses'],
                history['train_accuracies'],
                history['val_accuracies'],
                save_dir
            )
            saved_plots['training_curves'] = path
            print(f"Saved: {path}")
        
        # Parameter distribution
        path = plot_parameter_distribution(model, save_dir)
        saved_plots['parameter_distribution'] = path
        print(f"Saved: {path}")
        
        # Circuit depth analysis
        path = plot_quantum_circuit_depth(
            n_qubits_range=[2, 3, 4, 5, 6],
            n_layers_range=[1, 2, 3, 4],
            entanglement_types=['linear', 'circular', 'full'],
            save_dir=save_dir
        )
        saved_plots['circuit_depth'] = path
        print(f"Saved: {path}")
        
        # Evaluation plots (if test data provided)
        if y_test is not None and y_pred is not None:
            # Confusion matrix
            path = plot_confusion_matrix(y_test, y_pred, class_names, save_dir)
            saved_plots['confusion_matrix'] = path
            print(f"Saved: {path}")
            
            # Per-class metrics
            path = plot_per_class_metrics(y_test, y_pred, class_names, save_dir)
            saved_plots['per_class_metrics'] = path
            print(f"Saved: {path}")
            
            # ROC curves (if scores provided)
            if y_scores is not None:
                n_classes = len(set(y_test))
                path = plot_roc_curves(y_test, y_scores, n_classes, class_names, save_dir)
                saved_plots['roc_curves'] = path
                print(f"Saved: {path}")
        
        # Feature space visualization (if test data provided)
        if X_test is not None and y_test is not None:
            path = plot_feature_space_tsne(X_test, y_test, class_names, 
                                          "t-SNE: Test Feature Space", save_dir)
            saved_plots['tsne'] = path
            print(f"Saved: {path}")
        
    except Exception as e:
        print(f"Warning: Some plots could not be generated: {e}")
    
    return saved_plots