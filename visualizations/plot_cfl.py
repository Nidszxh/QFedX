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
from scipy.stats import entropy
import pandas as pd

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def create_viz_folder(base_path: str = "./results/cfl") -> Path:
    """Create and return visualization folder path."""
    folder = Path(base_path)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


# ============================================================================
# PER-CLIENT PERFORMANCE
# ============================================================================

def plot_client_training_losses(client_losses_per_round: List[List[float]], 
                                save_path: str):
    """
    Plot per-client training losses across federated rounds with statistical analysis.
    """
    num_rounds = len(client_losses_per_round)
    
    # Transpose to get per-client history
    client_losses_dict = {}
    for round_idx, losses in enumerate(client_losses_per_round):
        for client_idx, loss in enumerate(losses):
            if client_idx not in client_losses_dict:
                client_losses_dict[client_idx] = []
            client_losses_dict[client_idx].append((round_idx, loss))
    
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Main line plot
    ax1 = fig.add_subplot(gs[0, :2])
    colors = plt.cm.tab10(np.linspace(0, 1, len(client_losses_dict)))
    
    for client_idx, color in zip(sorted(client_losses_dict.keys()), colors):
        rounds, losses = zip(*client_losses_dict[client_idx])
        ax1.plot(rounds, losses, marker='o', linewidth=2, markersize=4, 
                label=f'Client {client_idx+1}', color=color, alpha=0.8)
    
    ax1.set_xlabel('Federated Round', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Per-Client Training Loss Evolution', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9, ncol=2)
    ax1.grid(alpha=0.3, linestyle='--')
    
    # Boxplot with outlier detection
    ax2 = fig.add_subplot(gs[0, 2])
    data_for_boxplot = [[loss for _, loss in client_losses_dict[i]] 
                        for i in sorted(client_losses_dict.keys())]
    
    bp = ax2.boxplot(data_for_boxplot, labels=[f'C{i+1}' for i in range(len(data_for_boxplot))],
                     patch_artist=True, showmeans=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax2.set_xlabel('Client ID', fontsize=11)
    ax2.set_ylabel('Training Loss', fontsize=11)
    ax2.set_title('Loss Distribution per Client', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Client ranking by average loss
    ax3 = fig.add_subplot(gs[1, :2])
    avg_losses = {cid: np.mean([loss for _, loss in data]) 
                  for cid, data in client_losses_dict.items()}
    std_losses = {cid: np.std([loss for _, loss in data]) 
                  for cid, data in client_losses_dict.items()}
    
    sorted_clients = sorted(avg_losses.items(), key=lambda x: x[1])
    client_ids = [f'Client {cid+1}' for cid, _ in sorted_clients]
    avg_vals = [avg for _, avg in sorted_clients]
    std_vals = [std_losses[cid] for cid, _ in sorted_clients]
    
    bars = ax3.barh(client_ids, avg_vals, xerr=std_vals, alpha=0.7, 
                    color=colors, edgecolor='black', linewidth=1)
    
    ax3.set_xlabel('Average Training Loss (± Std)', fontsize=11)
    ax3.set_title('Client Ranking by Average Loss', fontsize=12, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar, val in zip(bars, avg_vals):
        ax3.text(val, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
                ha='left', va='center', fontsize=9, fontweight='bold')
    
    # Statistical summary table
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis('off')
    
    summary_text = "📊 Statistical Summary\n" + "="*30 + "\n\n"
    all_losses = [loss for data in data_for_boxplot for loss in data]
    summary_text += f"Global Mean: {np.mean(all_losses):.4f}\n"
    summary_text += f"Global Std:  {np.std(all_losses):.4f}\n"
    summary_text += f"Min Loss:    {np.min(all_losses):.4f}\n"
    summary_text += f"Max Loss:    {np.max(all_losses):.4f}\n\n"
    summary_text += f"Best Client: Client {sorted_clients[0][0]+1}\n"
    summary_text += f"Worst Client: Client {sorted_clients[-1][0]+1}\n\n"
    
    # Identify problem clients (high variance or persistently high loss)
    problem_clients = [cid for cid, std in std_losses.items() if std > 1.5 * np.mean(list(std_losses.values()))]
    if problem_clients:
        summary_text += "⚠️ High Variance Clients:\n"
        summary_text += ", ".join([f"Client {c+1}" for c in problem_clients])
    
    ax4.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
             verticalalignment='center', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle('Client Training Loss Analysis', fontsize=15, fontweight='bold', y=0.98)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   ✓ Client training losses plot saved")


def plot_client_final_accuracies(client_data: List[Tuple[torch.Tensor, torch.Tensor]],
                                 global_model: nn.Module,
                                 device: torch.device,
                                 save_path: str):
    """
    Bar plot of final accuracy for each client with statistical analysis.
    """
    global_model.eval()
    client_accuracies = []
    client_sample_counts = []
    per_class_accuracies = []
    
    with torch.no_grad():
        for X_client, y_client in client_data:
            X_client = X_client.view(-1, 1, 28, 28).to(device)
            y_client = y_client.to(device)
            
            outputs = global_model(X_client)
            predictions = outputs.argmax(1)
            accuracy = (predictions == y_client).float().mean().item()
            
            client_accuracies.append(accuracy)
            client_sample_counts.append(len(y_client))
            
            # Per-class accuracy for this client
            unique_classes = torch.unique(y_client)
            class_acc = {}
            for cls in unique_classes:
                mask = y_client == cls
                class_acc[cls.item()] = (predictions[mask] == y_client[mask]).float().mean().item()
            per_class_accuracies.append(class_acc)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Main bar plot
    client_ids = [f'Client {i+1}\n(n={n})' for i, n in enumerate(client_sample_counts)]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(client_accuracies)))
    
    bars = ax1.bar(client_ids, client_accuracies, color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, acc in zip(bars, client_accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add mean line and confidence interval
    mean_acc = np.mean(client_accuracies)
    std_acc = np.std(client_accuracies)
    ax1.axhline(y=mean_acc, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_acc:.3f} ± {std_acc:.3f}')
    ax1.axhspan(mean_acc - std_acc, mean_acc + std_acc, alpha=0.2, color='red')
    
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Final Client Accuracies (Global Model)', fontsize=13, fontweight='bold')
    ax1.set_ylim([0, 1.05])
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Per-class accuracy heatmap
    num_clients = len(per_class_accuracies)
    all_classes = sorted(set().union(*[set(d.keys()) for d in per_class_accuracies]))
    
    heatmap_data = np.zeros((num_clients, len(all_classes)))
    for i, class_acc_dict in enumerate(per_class_accuracies):
        for j, cls in enumerate(all_classes):
            heatmap_data[i, j] = class_acc_dict.get(cls, 0)
    
    im = ax2.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    ax2.set_xticks(range(len(all_classes)))
    ax2.set_xticklabels([f'Class {c}' for c in all_classes])
    ax2.set_yticks(range(num_clients))
    ax2.set_yticklabels([f'Client {i+1}' for i in range(num_clients)])
    ax2.set_xlabel('Class Label', fontsize=11)
    ax2.set_ylabel('Client ID', fontsize=11)
    ax2.set_title('Per-Class Accuracy Matrix', fontsize=12, fontweight='bold')
    
    # Add text annotations
    for i in range(num_clients):
        for j in range(len(all_classes)):
            text = ax2.text(j, i, f'{heatmap_data[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im, ax=ax2, label='Accuracy')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   ✓ Client final accuracies plot saved")


def plot_client_participation_history(selected_clients_history: List[List[int]],
                                      num_clients: int,
                                      save_path: str):
    """
    Heatmap showing which clients participated in each round.
    """
    num_rounds = len(selected_clients_history)
    
    # Create participation matrix
    participation_matrix = np.zeros((num_rounds, num_clients))
    for round_idx, selected in enumerate(selected_clients_history):
        for client_id in selected:
            participation_matrix[round_idx, client_id] = 1
    
    # Calculate participation frequency
    participation_freq = participation_matrix.sum(axis=0)
    participation_pct = (participation_freq / num_rounds) * 100
    
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.3, wspace=0.3)
    
    # Main heatmap
    ax1 = fig.add_subplot(gs[0, :])
    im = ax1.imshow(participation_matrix.T, cmap='Greens', aspect='auto', interpolation='nearest')
    
    ax1.set_xlabel('Federated Round', fontsize=12)
    ax1.set_ylabel('Client ID', fontsize=12)
    ax1.set_title('Client Participation History', fontsize=13, fontweight='bold')
    ax1.set_yticks(range(num_clients))
    ax1.set_yticklabels([f'Client {i+1}' for i in range(num_clients)])
    
    plt.colorbar(im, ax=ax1, label='Participated', ticks=[0, 1])
    
    # Participation frequency bar chart
    ax2 = fig.add_subplot(gs[1, 0])
    colors = plt.cm.viridis(participation_pct / 100)
    bars = ax2.bar(range(num_clients), participation_freq, color=colors, 
                   alpha=0.8, edgecolor='black')
    
    for bar, freq, pct in zip(bars, participation_freq, participation_pct):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(freq)}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9)
    
    ax2.set_xlabel('Client ID', fontsize=11)
    ax2.set_ylabel('Total Participations', fontsize=11)
    ax2.set_title('Participation Frequency', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(num_clients))
    ax2.set_xticklabels([f'C{i+1}' for i in range(num_clients)])
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Fairness metrics
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    
    fairness_text = "⚖️ Fairness Metrics\n" + "="*30 + "\n\n"
    fairness_text += f"Expected (uniform): {num_rounds/num_clients:.1f}\n"
    fairness_text += f"Actual Mean: {participation_freq.mean():.1f}\n"
    fairness_text += f"Std Dev: {participation_freq.std():.2f}\n\n"
    
    # Gini coefficient for fairness
    sorted_freq = np.sort(participation_freq)
    n = len(sorted_freq)
    gini = (2 * np.sum((np.arange(n) + 1) * sorted_freq)) / (n * np.sum(sorted_freq)) - (n + 1) / n
    fairness_text += f"Gini Coefficient: {gini:.3f}\n"
    fairness_text += "(0 = perfect equality)\n\n"
    
    if gini < 0.1:
        fairness_text += "✅ Fair participation"
    elif gini < 0.3:
        fairness_text += "⚠️ Moderate imbalance"
    else:
        fairness_text += "❌ High imbalance"
    
    ax3.text(0.1, 0.5, fairness_text, fontsize=10, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   ✓ Client participation history saved")


# ============================================================================
# CLIENT HETEROGENEITY ANALYSIS
# ============================================================================

def plot_client_heterogeneity_analysis(client_data: List[Tuple[torch.Tensor, torch.Tensor]],
                                      client_losses_per_round: List[List[float]],
                                      save_path: str):
    """
    Comprehensive analysis of data heterogeneity and its impact on training.
    """
    num_clients = len(client_data)
    num_classes = len(torch.unique(client_data[0][1]))
    
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. Label distribution divergence (KL divergence heatmap)
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Compute label distributions
    label_distributions = []
    for _, y in client_data:
        y_np = y.numpy() if isinstance(y, torch.Tensor) else y
        counts = np.bincount(y_np, minlength=num_classes)
        dist = (counts + 1e-10) / (counts.sum() + num_classes * 1e-10)
        label_distributions.append(dist)
    
    # KL divergence matrix
    kl_matrix = np.zeros((num_clients, num_clients))
    for i in range(num_clients):
        for j in range(num_clients):
            if i == j:
                kl_matrix[i, j] = 0
            else:
                kl_matrix[i, j] = entropy(label_distributions[i], label_distributions[j])
    
    im1 = ax1.imshow(kl_matrix, cmap='Reds', aspect='auto')
    ax1.set_xticks(range(num_clients))
    ax1.set_yticks(range(num_clients))
    ax1.set_xticklabels([f'C{i+1}' for i in range(num_clients)])
    ax1.set_yticklabels([f'C{i+1}' for i in range(num_clients)])
    ax1.set_title('KL Divergence (Label Distribution)', fontsize=11, fontweight='bold')
    plt.colorbar(im1, ax=ax1)
    
    # 2. Gini coefficient per client (class imbalance)
    ax2 = fig.add_subplot(gs[0, 1])
    
    gini_coefficients = []
    for _, y in client_data:
        y_np = y.numpy() if isinstance(y, torch.Tensor) else y
        counts = np.bincount(y_np, minlength=num_classes)
        sorted_counts = np.sort(counts)
        n = len(sorted_counts)
        gini = (2 * np.sum((np.arange(n) + 1) * sorted_counts)) / (n * np.sum(sorted_counts)) - (n + 1) / n
        gini_coefficients.append(gini)
    
    colors = plt.cm.Oranges(np.array(gini_coefficients) / max(gini_coefficients))
    bars = ax2.bar(range(num_clients), gini_coefficients, color=colors, 
                   alpha=0.8, edgecolor='black')
    
    for bar, gini in zip(bars, gini_coefficients):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{gini:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax2.axhline(y=np.mean(gini_coefficients), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(gini_coefficients):.3f}')
    ax2.set_xlabel('Client ID', fontsize=11)
    ax2.set_ylabel('Gini Coefficient', fontsize=11)
    ax2.set_title('Class Imbalance per Client', fontsize=11, fontweight='bold')
    ax2.set_xticks(range(num_clients))
    ax2.set_xticklabels([f'C{i+1}' for i in range(num_clients)])
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 3. Sample size distribution
    ax3 = fig.add_subplot(gs[0, 2])
    
    sample_sizes = [len(y) for _, y in client_data]
    colors = plt.cm.Blues(np.array(sample_sizes) / max(sample_sizes))
    bars = ax3.bar(range(num_clients), sample_sizes, color=colors, 
                   alpha=0.8, edgecolor='black')
    
    for bar, size in zip(bars, sample_sizes):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{size}', ha='center', va='bottom', fontsize=9)
    
    ax3.axhline(y=np.mean(sample_sizes), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(sample_sizes):.1f}')
    ax3.set_xlabel('Client ID', fontsize=11)
    ax3.set_ylabel('Sample Count', fontsize=11)
    ax3.set_title('Data Size Distribution', fontsize=11, fontweight='bold')
    ax3.set_xticks(range(num_clients))
    ax3.set_xticklabels([f'C{i+1}' for i in range(num_clients)])
    ax3.legend(fontsize=9)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 4. Correlation: Heterogeneity vs Loss Variance
    ax4 = fig.add_subplot(gs[1, 0])
    
    # Average KL divergence per client (heterogeneity score)
    heterogeneity_scores = kl_matrix.mean(axis=1)
    
    # Loss variance per client
    loss_variances = []
    for client_idx in range(num_clients):
        client_losses = [round_losses[client_idx] for round_losses in client_losses_per_round 
                        if client_idx < len(round_losses)]
        loss_variances.append(np.var(client_losses) if client_losses else 0)
    
    scatter = ax4.scatter(heterogeneity_scores, loss_variances, s=100, 
                         c=range(num_clients), cmap='rainbow', alpha=0.7, edgecolors='black')
    
    # Add trend line
    z = np.polyfit(heterogeneity_scores, loss_variances, 1)
    p = np.poly1d(z)
    ax4.plot(heterogeneity_scores, p(heterogeneity_scores), "r--", linewidth=2, alpha=0.7)
    
    # Correlation coefficient
    corr = np.corrcoef(heterogeneity_scores, loss_variances)[0, 1]
    ax4.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax4.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    for i in range(num_clients):
        ax4.annotate(f'C{i+1}', (heterogeneity_scores[i], loss_variances[i]),
                    fontsize=8, ha='center')
    
    ax4.set_xlabel('Heterogeneity Score (Avg KL Divergence)', fontsize=11)
    ax4.set_ylabel('Loss Variance', fontsize=11)
    ax4.set_title('Heterogeneity vs Training Stability', fontsize=11, fontweight='bold')
    ax4.grid(alpha=0.3, linestyle='--')
    
    # 5. Client difficulty score
    ax5 = fig.add_subplot(gs[1, 1])
    
    # Difficulty = average loss rank across rounds
    difficulty_scores = []
    for client_idx in range(num_clients):
        ranks = []
        for round_losses in client_losses_per_round:
            if client_idx < len(round_losses):
                # Rank: higher loss = higher rank (more difficult)
                sorted_indices = np.argsort(round_losses)
                rank = np.where(sorted_indices == client_idx)[0][0]
                ranks.append(rank)
        difficulty_scores.append(np.mean(ranks) if ranks else 0)
    
    colors = plt.cm.RdYlGn_r(np.array(difficulty_scores) / max(difficulty_scores))
    bars = ax5.barh(range(num_clients), difficulty_scores, color=colors, 
                    alpha=0.8, edgecolor='black')
    
    for bar, score in zip(bars, difficulty_scores):
        width = bar.get_width()
        ax5.text(width, bar.get_y() + bar.get_height()/2.,
                f'{score:.2f}', ha='left', va='center', fontsize=9)
    
    ax5.set_yticks(range(num_clients))
    ax5.set_yticklabels([f'Client {i+1}' for i in range(num_clients)])
    ax5.set_xlabel('Difficulty Score (Avg Loss Rank)', fontsize=11)
    ax5.set_title('Client Difficulty Ranking', fontsize=11, fontweight='bold')
    ax5.grid(axis='x', alpha=0.3, linestyle='--')
    
    # 6. Summary statistics
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    summary_text = "📊 Heterogeneity Summary\n" + "="*35 + "\n\n"
    summary_text += f"Avg KL Divergence: {kl_matrix[np.triu_indices_from(kl_matrix, k=1)].mean():.4f}\n"
    summary_text += f"Avg Gini Coeff: {np.mean(gini_coefficients):.4f}\n"
    summary_text += f"Sample Size CV: {np.std(sample_sizes)/np.mean(sample_sizes):.4f}\n\n"
    
    summary_text += f"Heterogeneity-Loss Corr: {corr:.3f}\n\n"
    
    most_heterogeneous = np.argmax(heterogeneity_scores)
    most_difficult = np.argmax(difficulty_scores)
    
    summary_text += f"Most Heterogeneous:\n  Client {most_heterogeneous+1}\n\n"
    summary_text += f"Most Difficult:\n  Client {most_difficult+1}\n\n"
    
    if corr > 0.5:
        summary_text += "⚠️ High heterogeneity\n   impacts stability"
    else:
        summary_text += "✅ Heterogeneity\n   manageable"
    
    ax6.text(0.1, 0.5, summary_text, fontsize=9, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    plt.suptitle('Client Heterogeneity Analysis', fontsize=15, fontweight='bold', y=0.98)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   ✓ Client heterogeneity analysis saved")


# ============================================================================
# CONVERGENCE ANALYSIS
# ============================================================================

def plot_convergence_analysis(metrics: Dict, target_accuracy: float = 0.95,
                              save_path: str = None):
    """
    Comprehensive convergence analysis with rate estimation and smoothing.
    """
    if save_path is None:
        save_path = create_viz_folder() / "convergence_analysis.png"
    
    test_acc = np.array(metrics['test_accuracies'])
    val_acc = np.array(metrics.get('val_accuracies', test_acc))
    rounds = np.arange(len(test_acc))
    
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. Raw vs Smoothed accuracy
    ax1 = fig.add_subplot(gs[0, :2])
    
    # Moving average smoothing
    window = min(5, len(test_acc) // 4)
    if window > 1:
        smoothed_test = np.convolve(test_acc, np.ones(window)/window, mode='valid')
        smoothed_rounds = rounds[:len(smoothed_test)]
    else:
        smoothed_test = test_acc
        smoothed_rounds = rounds
    
    ax1.plot(rounds, test_acc, marker='o', linewidth=1.5, markersize=4, 
            alpha=0.5, label='Raw Test Accuracy', color='lightblue')
    ax1.plot(smoothed_rounds, smoothed_test, linewidth=3, 
            label=f'Smoothed (MA-{window})', color='darkblue')
    
    if 'val_accuracies' in metrics:
        ax1.plot(rounds, val_acc, marker='s', linewidth=1.5, markersize=4,
                alpha=0.6, label='Val Accuracy', color='orange')
    
    # Mark best model
    best_round = np.argmax(test_acc)
    ax1.axvline(x=best_round, color='red', linestyle='--', linewidth=2, alpha=0.7,
               label=f'Best Model (Round {best_round})')
    ax1.scatter([best_round], [test_acc[best_round]], color='red', s=200, 
               marker='*', zorder=5, edgecolors='black', linewidth=2)
    
    # Target accuracy line
    ax1.axhline(y=target_accuracy, color='green', linestyle=':', linewidth=2,
               label=f'Target ({target_accuracy:.2f})')
    
    ax1.set_xlabel('Federated Round', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Convergence Trajectory', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.set_ylim([0, 1.05])
    
    # 2. Convergence rate (accuracy gain per round)
    ax2 = fig.add_subplot(gs[0, 2])
    
    acc_gains = np.diff(test_acc)
    ax2.bar(rounds[1:], acc_gains, alpha=0.7, color='teal', edgecolor='black')
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax2.axhline(y=np.mean(acc_gains), color='orange', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(acc_gains):.4f}')
    
    ax2.set_xlabel('Round', fontsize=11)
    ax2.set_ylabel('Accuracy Δ', fontsize=11)
    ax2.set_title('Per-Round Improvement', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 3. Cumulative improvement
    ax3 = fig.add_subplot(gs[1, 0])
    
    cumulative_gain = test_acc - test_acc[0]
    ax3.fill_between(rounds, cumulative_gain, alpha=0.3, color='purple')
    ax3.plot(rounds, cumulative_gain, linewidth=2.5, color='purple', marker='o', markersize=4)
    
    ax3.set_xlabel('Round', fontsize=11)
    ax3.set_ylabel('Cumulative Accuracy Gain', fontsize=11)
    ax3.set_title('Cumulative Improvement', fontsize=12, fontweight='bold')
    ax3.grid(alpha=0.3, linestyle='--')
    
    total_gain = test_acc[-1] - test_acc[0]
    ax3.text(0.5, 0.95, f'Total Gain: {total_gain:.4f}', 
            transform=ax3.transAxes, fontsize=11, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # 4. Convergence speed analysis
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Detect convergence (when improvement drops below threshold)
    convergence_threshold = 0.001
    converged_mask = np.abs(acc_gains) < convergence_threshold
    
    if np.any(converged_mask):
        convergence_round = np.where(converged_mask)[0][0] + 1
        convergence_text = f"Converged at Round {convergence_round}"
        convergence_color = 'green'
    else:
        convergence_text = "Not yet converged"
        convergence_color = 'orange'
    
    # Moving standard deviation (stability metric)
    if len(test_acc) > 5:
        rolling_std = pd.Series(test_acc).rolling(window=5).std().fillna(0)
        ax4.plot(rounds, rolling_std, linewidth=2.5, color='brown', marker='o', markersize=4)
        ax4.fill_between(rounds, rolling_std, alpha=0.2, color='brown')
    
    ax4.set_xlabel('Round', fontsize=11)
    ax4.set_ylabel('Rolling Std Dev (window=5)', fontsize=11)
    ax4.set_title('Training Stability', fontsize=12, fontweight='bold')
    ax4.grid(alpha=0.3, linestyle='--')
    
    ax4.text(0.5, 0.95, convergence_text, transform=ax4.transAxes,
            fontsize=11, ha='center', va='top', color=convergence_color,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 5. Estimated rounds to target
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    summary_text = "📈 Convergence Metrics\n" + "="*35 + "\n\n"
    summary_text += f"Initial Acc: {test_acc[0]:.4f}\n"
    summary_text += f"Final Acc:   {test_acc[-1]:.4f}\n"
    summary_text += f"Best Acc:    {test_acc[best_round]:.4f}\n"
    summary_text += f"  (Round {best_round})\n\n"
    
    summary_text += f"Total Gain:  {total_gain:.4f}\n"
    summary_text += f"Avg Gain/Round: {np.mean(acc_gains):.4f}\n\n"
    
    # Estimate rounds to target (linear extrapolation)
    if test_acc[-1] < target_accuracy and np.mean(acc_gains) > 0:
        remaining = target_accuracy - test_acc[-1]
        estimated_rounds = int(remaining / np.mean(acc_gains))
        summary_text += f"Est. rounds to {target_accuracy:.2f}:\n"
        summary_text += f"  ~{estimated_rounds} more rounds\n\n"
    elif test_acc[-1] >= target_accuracy:
        reached_round = np.where(test_acc >= target_accuracy)[0][0]
        summary_text += f"✅ Target reached at\n   Round {reached_round}\n\n"
    else:
        summary_text += f"⚠️ Negative trend\n\n"
    
    # Convergence status
    if np.any(converged_mask):
        summary_text += f"✅ Converged\n   (Round {convergence_round})"
    else:
        summary_text += "⚠️ Still improving"
    
    ax5.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5))
    
    plt.suptitle('Convergence Analysis Dashboard', fontsize=15, fontweight='bold', y=0.98)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   ✓ Convergence analysis saved")


# ============================================================================
# OVERFITTING DETECTION
# ============================================================================

def plot_overfitting_analysis(metrics: Dict, save_path: str):
    """
    Multi-faceted overfitting detection and analysis.
    """
    train_losses = np.array(metrics['train_losses'])
    val_losses = np.array(metrics.get('val_losses', train_losses))
    test_losses = np.array(metrics['test_losses'])
    
    test_acc = np.array(metrics['test_accuracies'])
    val_acc = np.array(metrics.get('val_accuracies', test_acc))
    
    rounds = np.arange(len(test_acc))
    
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. Loss divergence
    ax1 = fig.add_subplot(gs[0, :2])
    
    ax1.plot(rounds, train_losses, marker='o', linewidth=2, markersize=4,
            label='Train Loss', color='blue', alpha=0.8)
    ax1.plot(rounds, val_losses, marker='s', linewidth=2, markersize=4,
            label='Val Loss', color='orange', alpha=0.8)
    ax1.plot(rounds, test_losses, marker='^', linewidth=2, markersize=4,
            label='Test Loss', color='green', alpha=0.8)
    
    ax1.set_xlabel('Federated Round', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss Curves (Train/Val/Test)', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=11)
    ax1.grid(alpha=0.3, linestyle='--')
    
    # Detect overfitting (when val/test loss starts increasing while train decreases)
    if len(rounds) > 5:
        train_trend = np.polyfit(rounds[-5:], train_losses[-5:], 1)[0]
        val_trend = np.polyfit(rounds[-5:], val_losses[-5:], 1)[0]
        
        if train_trend < 0 and val_trend > 0:
            ax1.text(0.5, 0.95, '⚠️ Overfitting Detected', transform=ax1.transAxes,
                    fontsize=12, ha='center', va='top', color='red',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    # 2. Generalization gap
    ax2 = fig.add_subplot(gs[0, 2])
    
    gen_gap_loss = test_losses - train_losses
    gen_gap_acc = test_acc - val_acc if 'val_accuracies' in metrics else np.zeros_like(test_acc)
    
    ax2_twin = ax2.twinx()
    
    line1 = ax2.plot(rounds, gen_gap_loss, marker='o', linewidth=2.5, markersize=5,
                     label='Loss Gap (Test-Train)', color='red', alpha=0.8)
    line2 = ax2_twin.plot(rounds, gen_gap_acc, marker='s', linewidth=2.5, markersize=5,
                          label='Acc Gap (Test-Val)', color='blue', alpha=0.8)
    
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2_twin.axhline(y=0, color='black', linestyle='--', linewidth=1)
    
    ax2.set_xlabel('Round', fontsize=11)
    ax2.set_ylabel('Loss Gap', fontsize=11, color='red')
    ax2_twin.set_ylabel('Accuracy Gap', fontsize=11, color='blue')
    ax2.set_title('Generalization Gap', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3, linestyle='--')
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper left', fontsize=9)
    
    # 3. Gap evolution rate
    ax3 = fig.add_subplot(gs[1, 0])
    
    gap_change = np.diff(gen_gap_loss)
    ax3.bar(rounds[1:], gap_change, alpha=0.7, 
           color=['red' if x > 0 else 'green' for x in gap_change],
           edgecolor='black')
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=2)
    
    ax3.set_xlabel('Round', fontsize=11)
    ax3.set_ylabel('Gap Change', fontsize=11)
    ax3.set_title('Gap Evolution (Δ per round)', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 4. Early warning indicator
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Calculate rolling correlation between train and val loss
    if len(rounds) > 10:
        window = 5
        correlations = []
        for i in range(window, len(rounds)):
            corr = np.corrcoef(train_losses[i-window:i], val_losses[i-window:i])[0, 1]
            correlations.append(corr)
        
        corr_rounds = rounds[window:]
        ax4.plot(corr_rounds, correlations, linewidth=2.5, color='purple', 
                marker='o', markersize=5)
        ax4.axhline(y=0.8, color='green', linestyle='--', linewidth=2,
                   label='Healthy (>0.8)', alpha=0.7)
        ax4.axhline(y=0.5, color='orange', linestyle='--', linewidth=2,
                   label='Warning (<0.5)', alpha=0.7)
        ax4.fill_between(corr_rounds, 0.8, 1.0, alpha=0.1, color='green')
        ax4.fill_between(corr_rounds, 0, 0.5, alpha=0.1, color='red')
    
    ax4.set_xlabel('Round', fontsize=11)
    ax4.set_ylabel('Train-Val Correlation', fontsize=11)
    ax4.set_title('Loss Correlation (Early Warning)', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.3, linestyle='--')
    ax4.set_ylim([-0.1, 1.1])
    
    # 5. Overfitting summary
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    summary_text = "🔍 Overfitting Analysis\n" + "="*35 + "\n\n"
    
    # Current generalization gap
    current_gap = gen_gap_loss[-1]
    summary_text += f"Current Loss Gap:\n  {current_gap:.4f}\n\n"
    
    # Gap trend
    if len(rounds) > 5:
        recent_trend = np.polyfit(rounds[-5:], gen_gap_loss[-5:], 1)[0]
        if recent_trend > 0.01:
            summary_text += "📈 Gap increasing\n   ⚠️ Risk: HIGH\n\n"
            risk_color = 'red'
        elif recent_trend < -0.01:
            summary_text += "📉 Gap decreasing\n   ✅ Risk: LOW\n\n"
            risk_color = 'green'
        else:
            summary_text += "➡️ Gap stable\n   ⚠️ Risk: MEDIUM\n\n"
            risk_color = 'orange'
    
    # Best model indicator
    best_gap_round = np.argmin(np.abs(gen_gap_loss))
    summary_text += f"Best Gen. Gap:\n  Round {best_gap_round}\n  ({gen_gap_loss[best_gap_round]:.4f})\n\n"
    
    # Correlation status
    if len(rounds) > 10 and len(correlations) > 0:
        recent_corr = correlations[-1]
        summary_text += f"Train-Val Corr:\n  {recent_corr:.3f}\n\n"
        if recent_corr > 0.8:
            summary_text += "✅ Losses aligned"
        elif recent_corr > 0.5:
            summary_text += "⚠️ Diverging slightly"
        else:
            summary_text += "❌ Significant divergence"
    
    ax5.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor=risk_color, alpha=0.2))
    
    plt.suptitle('Overfitting Detection Dashboard', fontsize=15, fontweight='bold', y=0.98)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   ✓ Overfitting analysis saved")


# ============================================================================
# PER-CLASS PERFORMANCE EVOLUTION
# ============================================================================

def plot_per_class_performance_evolution(model_checkpoints: List[Dict],
                                        test_data: Tuple[torch.Tensor, torch.Tensor],
                                        device: torch.device,
                                        class_names: List[str],
                                        checkpoint_rounds: List[int],
                                        save_path: str):
    """
    Track per-class accuracy evolution across checkpoints.
    """
    from collections import defaultdict
    
    X_test, y_test = test_data
    X_test = X_test.view(-1, 1, 28, 28).to(device)
    y_test = y_test.to(device)
    
    num_classes = len(class_names)
    class_accuracies = defaultdict(list)
    
    # Import model architecture (assuming TinyCNN)
    from torch import nn as nn_module
    
    for checkpoint in model_checkpoints:
        # Load checkpoint into temporary model
        temp_model = TinyCNN(num_classes=num_classes).to(device)
        temp_model.load_state_dict(checkpoint)
        temp_model.eval()
        
        with torch.no_grad():
            outputs = temp_model(X_test)
            predictions = outputs.argmax(1)
        
        # Calculate per-class accuracy
        for cls in range(num_classes):
            mask = y_test == cls
            if mask.sum() > 0:
                acc = (predictions[mask] == y_test[mask]).float().mean().item()
                class_accuracies[cls].append(acc)
            else:
                class_accuracies[cls].append(0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Line plot
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    for cls, color in enumerate(colors):
        ax1.plot(checkpoint_rounds, class_accuracies[cls], marker='o', 
                linewidth=2.5, markersize=6, label=class_names[cls], 
                color=color, alpha=0.8)
    
    ax1.set_xlabel('Federated Round', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Per-Class Accuracy Evolution', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.set_ylim([0, 1.05])
    
    # Heatmap
    heatmap_data = np.array([class_accuracies[cls] for cls in range(num_classes)])
    im = ax2.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    ax2.set_yticks(range(num_classes))
    ax2.set_yticklabels(class_names)
    ax2.set_xticks(range(len(checkpoint_rounds)))
    ax2.set_xticklabels([f'R{r}' for r in checkpoint_rounds])
    ax2.set_xlabel('Checkpoint Round', fontsize=11)
    ax2.set_ylabel('Class', fontsize=11)
    ax2.set_title('Per-Class Accuracy Heatmap', fontsize=12, fontweight='bold')
    
    plt.colorbar(im, ax=ax2, label='Accuracy')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   ✓ Per-class performance evolution saved")


# ============================================================================
# CLIENT DATA DISTRIBUTION
# ============================================================================

def plot_client_label_distribution_heatmap(client_data: List[Tuple[torch.Tensor, torch.Tensor]],
                                          num_classes: int,
                                          save_path: str):
    """
    Enhanced heatmap showing label distribution across clients with statistics.
    """
    num_clients = len(client_data)
    label_counts = np.zeros((num_clients, num_classes))
    
    for i, (_, y_client) in enumerate(client_data):
        y_np = y_client.numpy() if isinstance(y_client, torch.Tensor) else y_client
        for label in range(num_classes):
            label_counts[i, label] = np.sum(y_np == label)
    
    # Calculate percentages
    label_percentages = label_counts / label_counts.sum(axis=1, keepdims=True) * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(6, num_clients * 0.8)))
    
    # Absolute counts
    im1 = sns.heatmap(label_counts, annot=True, fmt='.0f', cmap='YlGnBu', 
                     cbar_kws={'label': 'Sample Count'}, ax=ax1)
    ax1.set_xlabel('Class Label', fontsize=12)
    ax1.set_ylabel('Client ID', fontsize=12)
    ax1.set_title('Client-wise Label Distribution (Counts)', fontsize=12, fontweight='bold')
    ax1.set_xticklabels([f'Class {i}' for i in range(num_classes)])
    ax1.set_yticklabels([f'Client {i+1}' for i in range(num_clients)], rotation=0)
    
    # Percentages
    im2 = sns.heatmap(label_percentages, annot=True, fmt='.1f', cmap='RdYlGn', 
                     cbar_kws={'label': 'Percentage (%)'}, ax=ax2, vmin=0, vmax=100)
    ax2.set_xlabel('Class Label', fontsize=12)
    ax2.set_ylabel('Client ID', fontsize=12)
    ax2.set_title('Client-wise Label Distribution (Percentages)', fontsize=12, fontweight='bold')
    ax2.set_xticklabels([f'Class {i}' for i in range(num_classes)])
    ax2.set_yticklabels([f'Client {i+1}' for i in range(num_clients)], rotation=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   ✓ Client label distribution heatmap saved")


# ============================================================================
# TRAINING DYNAMICS
# ============================================================================

def plot_training_dynamics(metrics: Dict, save_path: str):
    """
    Comprehensive training dynamics visualization with enhanced metrics.
    """
    test_acc = metrics['test_accuracies']
    val_acc = metrics.get('val_accuracies', test_acc)
    test_loss = metrics['test_losses']
    train_loss = metrics['train_losses']
    rounds = list(range(len(test_acc)))
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # 1. Accuracy curves with confidence bands
    ax1 = fig.add_subplot(gs[0, :2])
    
    ax1.plot(rounds, test_acc, marker='o', linewidth=2.5, markersize=5, 
            color='#2E86AB', label='Test Accuracy', alpha=0.9)
    ax1.fill_between(rounds, test_acc, alpha=0.2, color='#2E86AB')
    
    if 'val_accuracies' in metrics:
        ax1.plot(rounds, val_acc, marker='s', linewidth=2.5, markersize=5,
                color='#F18F01', label='Val Accuracy', alpha=0.9)
        ax1.fill_between(rounds, val_acc, alpha=0.2, color='#F18F01')
    
    # Mark best model
    best_idx = np.argmax(test_acc)
    ax1.scatter([best_idx], [test_acc[best_idx]], color='red', s=300, 
               marker='*', zorder=5, edgecolors='black', linewidth=2,
               label=f'Best ({test_acc[best_idx]:.4f})')
    
    ax1.set_xlabel('Federated Round', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Accuracy Evolution', fontsize=13, fontweight='bold')
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.set_ylim([0, 1.05])
    ax1.legend(fontsize=10, loc='lower right')
    
    # 2. Loss curves
    ax2 = fig.add_subplot(gs[0, 2])
    
    ax2.plot(rounds, test_loss, marker='s', linewidth=2, markersize=4,
            label='Test Loss', color='#A23B72', alpha=0.8)
    ax2.plot(rounds, train_loss, marker='^', linewidth=2, markersize=4,
            label='Train Loss', color='#06A77D', alpha=0.8)
    ax2.set_xlabel('Round', fontsize=11)
    ax2.set_ylabel('Loss', fontsize=11)
    ax2.set_title('Loss Curves', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3, linestyle='--')
    
    # 3. Accuracy improvement per round
    ax3 = fig.add_subplot(gs[1, 0])
    
    acc_improvement = np.diff([0] + test_acc)
    colors_imp = ['green' if x > 0 else 'red' for x in acc_improvement[1:]]
    ax3.bar(rounds[1:], acc_improvement[1:], alpha=0.7, color=colors_imp, edgecolor='black')
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=1.5)
    ax3.set_xlabel('Round', fontsize=11)
    ax3.set_ylabel('Accuracy Δ', fontsize=11)
    ax3.set_title('Per-Round Improvement', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 4. Loss reduction per round
    ax4 = fig.add_subplot(gs[1, 1])
    
    loss_reduction = -np.diff([train_loss[0]] + train_loss)
    colors_loss = ['green' if x > 0 else 'red' for x in loss_reduction[1:]]
    ax4.bar(rounds[1:], loss_reduction[1:], alpha=0.7, color=colors_loss, edgecolor='black')
    ax4.axhline(y=0, color='black', linestyle='--', linewidth=1.5)
    ax4.set_xlabel('Round', fontsize=11)
    ax4.set_ylabel('Loss Reduction', fontsize=11)
    ax4.set_title('Training Loss Reduction', fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 5. Learning curve (log scale)
    ax5 = fig.add_subplot(gs[1, 2])
    
    ax5.semilogy(rounds, train_loss, marker='o', linewidth=2, label='Train Loss', alpha=0.8)
    ax5.semilogy(rounds, test_loss, marker='s', linewidth=2, label='Test Loss', alpha=0.8)
    ax5.set_xlabel('Round', fontsize=11)
    ax5.set_ylabel('Loss (log scale)', fontsize=11)
    ax5.set_title('Learning Curve (Log Scale)', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(alpha=0.3, linestyle='--', which='both')
    
    # 6. Metrics summary table
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')
    
    summary_data = {
        'Metric': ['Initial Acc', 'Final Acc', 'Best Acc', 'Total Gain', 
                  'Avg Gain/Round', 'Final Train Loss', 'Final Test Loss'],
        'Value': [
            f"{test_acc[0]:.4f}",
            f"{test_acc[-1]:.4f}",
            f"{max(test_acc):.4f} (R{np.argmax(test_acc)})",
            f"{test_acc[-1] - test_acc[0]:.4f}",
            f"{np.mean(np.diff(test_acc)):.4f}",
            f"{train_loss[-1]:.4f}",
            f"{test_loss[-1]:.4f}"
        ]
    }
    
    df = pd.DataFrame(summary_data)
    table = ax6.table(cellText=df.values, colLabels=df.columns,
                     cellLoc='center', loc='center', colWidths=[0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_face_color('#f0f0f0')