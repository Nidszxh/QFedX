
# Visualization utilities for Quantum Federated Learning (QFL)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
import networkx as nx
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def ensure_viz_dir(save_dir: str = "./visualizations/qfl") -> Path:

    viz_path = Path(save_dir)
    viz_path.mkdir(parents=True, exist_ok=True)
    return viz_path


def plot_global_convergence(test_accuracies: List[float],
                            test_losses: List[float],
                            train_losses: List[float],
                            save_dir: str = "./visualizations/qfl",
                            filename: str = "global_convergence.png") -> str:

    viz_path = ensure_viz_dir(save_dir)
    
    rounds = range(len(test_accuracies))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy curve with smoothing
    ax1.plot(rounds, test_accuracies, 'o-', label='Test Accuracy', 
            color='#2ecc71', linewidth=2, markersize=6)
    
    # Add smoothed curve if enough data
    if len(test_accuracies) >= 3:
        smoothed = pd.Series(test_accuracies).rolling(window=3, center=True).mean()
        ax1.plot(rounds, smoothed, '--', label='Smoothed', 
                color='darkgreen', linewidth=2, alpha=0.7)
    
    # Mark best accuracy
    if len(test_accuracies) > 1:
        best_round = np.argmax(test_accuracies[1:]) + 1
        best_acc = max(test_accuracies[1:])
        ax1.plot(best_round, best_acc, '*', color='red', markersize=15, 
                label=f'Best: {best_acc:.4f}')
        ax1.axhline(y=best_acc, color='red', linestyle='--', alpha=0.3)
    
    ax1.set_xlabel('Federated Round', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Global Model Test Accuracy', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])
    
    # Loss curves
    ax2.plot(rounds, test_losses, 's-', label='Test Loss', 
            color='#e74c3c', linewidth=2, markersize=6)
    ax2.plot(rounds, train_losses, '^-', label='Train Loss', 
            color='#3498db', linewidth=2, markersize=6)
    
    ax2.set_xlabel('Federated Round', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Global Model Loss Curves', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Global Model Convergence Over Federated Rounds', 
                fontsize=15, fontweight='bold', y=1.02)
    
    save_path = viz_path / filename
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return str(save_path)


def plot_parameter_drift(param_drifts: List[float],
                        save_dir: str = "./visualizations/qfl",
                        filename: str = "parameter_drift.png") -> str:

    viz_path = ensure_viz_dir(save_dir)
    
    rounds = range(1, len(param_drifts) + 1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(rounds, param_drifts, 'o-', color='purple', linewidth=2, markersize=6)
    ax.fill_between(rounds, param_drifts, alpha=0.3, color='purple')
    
    # Add average line
    avg_drift = np.mean(param_drifts)
    ax.axhline(y=avg_drift, color='red', linestyle='--', alpha=0.6, 
              label=f'Average: {avg_drift:.4f}')
    
    ax.set_xlabel('Federated Round', fontsize=12, fontweight='bold')
    ax.set_ylabel('Parameter Drift (L2 Norm)', fontsize=12, fontweight='bold')
    ax.set_title('Global Model Parameter Drift per Round', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    save_path = viz_path / filename
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return str(save_path)


def plot_federated_system_graph(n_clients: int,
                                save_dir: str = "./visualizations/qfl",
                                filename: str = "federated_system.png") -> str:

    viz_path = ensure_viz_dir(save_dir)
    
    G = nx.DiGraph()
    
    # Add nodes
    G.add_node("Global Server", node_type='server')
    for i in range(n_clients):
        G.add_node(f"Client {i}", node_type='client')
    
    # Add edges (bidirectional communication)
    for i in range(n_clients):
        G.add_edge("Global Server", f"Client {i}", edge_type='distribute')
        G.add_edge(f"Client {i}", "Global Server", edge_type='aggregate')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Layout
    pos = {}
    pos["Global Server"] = (0.5, 0.9)
    
    # Arrange clients in a circle
    angle_step = 2 * np.pi / n_clients
    radius = 0.4
    for i in range(n_clients):
        angle = i * angle_step
        x = 0.5 + radius * np.cos(angle)
        y = 0.3 + radius * np.sin(angle)
        pos[f"Client {i}"] = (x, y)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, nodelist=["Global Server"], 
                          node_color='#e74c3c', node_size=3000, 
                          node_shape='s', ax=ax, label='Global Server')
    
    client_nodes = [f"Client {i}" for i in range(n_clients)]
    nx.draw_networkx_nodes(G, pos, nodelist=client_nodes,
                          node_color='#3498db', node_size=2000,
                          node_shape='o', ax=ax, label='Clients')
    
    # Draw edges
    distribute_edges = [(u, v) for u, v, d in G.edges(data=True) 
                       if d['edge_type'] == 'distribute']
    aggregate_edges = [(u, v) for u, v, d in G.edges(data=True) 
                      if d['edge_type'] == 'aggregate']
    
    nx.draw_networkx_edges(G, pos, edgelist=distribute_edges,
                          edge_color='green', arrows=True, 
                          arrowsize=20, width=2, alpha=0.6,
                          connectionstyle='arc3,rad=0.1', ax=ax)
    
    nx.draw_networkx_edges(G, pos, edgelist=aggregate_edges,
                          edge_color='orange', arrows=True,
                          arrowsize=20, width=2, alpha=0.6,
                          connectionstyle='arc3,rad=-0.1', ax=ax)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
    
    ax.set_title('Quantum Federated Learning System Architecture', 
                fontsize=14, fontweight='bold', pad=20)
    ax.axis('off')
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#e74c3c', 
              markersize=12, label='Global Server'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db',
              markersize=12, label='Clients'),
        Line2D([0], [0], color='green', linewidth=2, label='Model Distribution'),
        Line2D([0], [0], color='orange', linewidth=2, label='Update Aggregation')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    save_path = viz_path / filename
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return str(save_path)


def plot_quantum_classical_balance(model,
                                   save_dir: str = "./visualizations/qfl",
                                   filename: str = "quantum_classical_balance.png") -> str:

    viz_path = ensure_viz_dir(save_dir)
    
    q_params = sum(p.numel() for p in model.get_quantum_params())
    c_params = sum(p.numel() for p in model.get_classical_params())
    total_params = q_params + c_params
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pie chart
    sizes = [q_params, c_params]
    labels = [f'Quantum\n({q_params} params)', f'Classical\n({c_params} params)']
    colors = ['#9b59b6', '#3498db']
    explode = (0.05, 0.05)
    
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
           shadow=True, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax1.set_title('Parameter Distribution', fontsize=13, fontweight='bold')
    
    # Bar chart
    categories = ['Quantum', 'Classical']
    values = [q_params, c_params]
    bars = ax2.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
    
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{value}\n({100*value/total_params:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2.set_ylabel('Number of Parameters', fontsize=12, fontweight='bold')
    ax2.set_title('Parameter Count Comparison', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Quantum-Classical Parameter Balance', fontsize=15, fontweight='bold', y=1.00)
    
    save_path = viz_path / filename
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return str(save_path)


def plot_client_boxplot(client_losses: List[List[float]],
                        save_dir: str = "./visualizations/qfl",
                        filename: str = "client_loss_boxplot.png") -> str:

    viz_path = ensure_viz_dir(save_dir)
    
    # Filter valid losses
    valid_losses = [losses for losses in client_losses 
                   if losses and all(l != float('inf') for l in losses)]
    
    if not valid_losses:
        print("Warning: No valid client losses for boxplot")
        return ""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bp = ax.boxplot(valid_losses, patch_artist=True, 
                    labels=[f'R{i+1}' for i in range(len(valid_losses))])
    
    # Color boxes
    for patch in bp['boxes']:
        patch.set_facecolor('#9b59b6')
        patch.set_alpha(0.6)
    
    ax.set_xlabel('Federated Round', fontsize=12, fontweight='bold')
    ax.set_ylabel('Client Loss', fontsize=12, fontweight='bold')
    ax.set_title('Client Loss Distribution per Round', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    save_path = viz_path / filename
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return str(save_path)


def plot_federated_confusion_matrix(y_true: np.ndarray,
                                    y_pred: np.ndarray,
                                    class_names: Optional[List[str]] = None,
                                    save_dir: str = "./visualizations/qfl",
                                    filename: str = "global_confusion_matrix.png") -> str:

    viz_path = ensure_viz_dir(save_dir)
    
    cm = confusion_matrix(y_true, y_pred)
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(cm.shape[0])]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis',
               xticklabels=class_names, yticklabels=class_names,
               cbar_kws={'label': 'Count'}, ax=ax)
    
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Global Model Confusion Matrix', fontsize=14, fontweight='bold')
    
    # Add accuracy
    accuracy = np.trace(cm) / np.sum(cm)
    ax.text(0.5, -0.15, f'Global Accuracy: {accuracy:.4f}',
           ha='center', va='top', transform=ax.transAxes,
           fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    save_path = viz_path / filename
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return str(save_path)


def plot_round_comparison_dashboard(test_accuracies: List[float],
                                   test_losses: List[float],
                                   client_losses: List[List[float]],
                                   save_dir: str = "./visualizations/qfl",
                                   filename: str = "round_dashboard.png") -> str:

    viz_path = ensure_viz_dir(save_dir)
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    rounds = range(len(test_accuracies))
    
    # 1. Global Accuracy
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(rounds, test_accuracies, 'o-', color='#2ecc71', linewidth=2.5, markersize=7)
    if len(test_accuracies) > 1:
        best_acc = max(test_accuracies[1:])
        ax1.axhline(y=best_acc, color='red', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Round', fontweight='bold')
    ax1.set_ylabel('Accuracy', fontweight='bold')
    ax1.set_title('Global Test Accuracy', fontsize=12, fontweight='bold')
    ax1.grid(alpha=0.3)
    ax1.set_ylim([0, 1.05])
    
    # 2. Best Round Marker
    ax2 = fig.add_subplot(gs[0, 2])
    if len(test_accuracies) > 1:
        best_round = np.argmax(test_accuracies[1:]) + 1
        best_acc = test_accuracies[best_round]
        ax2.text(0.5, 0.6, f'Best Round', ha='center', fontsize=14, fontweight='bold')
        ax2.text(0.5, 0.4, f'{best_round}', ha='center', fontsize=40, fontweight='bold', color='red')
        ax2.text(0.5, 0.2, f'Acc: {best_acc:.4f}', ha='center', fontsize=12)
    ax2.axis('off')
    
    # 3. Loss Curves
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(rounds, test_losses, 's-', label='Test', color='#e74c3c', linewidth=2, markersize=6)
    ax3.set_xlabel('Round', fontweight='bold')
    ax3.set_ylabel('Loss', fontweight='bold')
    ax3.set_title('Global Loss Progression', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # 4. Client Loss Heatmap
    ax4 = fig.add_subplot(gs[2, :2])
    if client_losses:
        max_clients = max(len(losses) for losses in client_losses if losses)
        loss_matrix = []
        for round_losses in client_losses:
            if round_losses:
                padded = list(round_losses) + [np.nan] * (max_clients - len(round_losses))
                loss_matrix.append(padded)
        
        if loss_matrix:
            sns.heatmap(np.array(loss_matrix), cmap='YlOrRd', ax=ax4,
                       cbar_kws={'label': 'Loss'}, annot=True, fmt='.2f')
            ax4.set_xlabel('Client ID', fontweight='bold')
            ax4.set_ylabel('Round', fontweight='bold')
            ax4.set_title('Client Loss Heatmap', fontsize=12, fontweight='bold')
    
    # 5. Summary Stats
    ax5 = fig.add_subplot(gs[2, 2])
    final_acc = test_accuracies[-1]
    best_acc = max(test_accuracies[1:]) if len(test_accuracies) > 1 else final_acc
    final_loss = test_losses[-1]
    
    summary_text = f"Final Accuracy: {final_acc:.4f}\n"
    summary_text += f"Best Accuracy: {best_acc:.4f}\n"
    summary_text += f"Final Loss: {final_loss:.4f}\n"
    summary_text += f"Total Rounds: {len(test_accuracies) - 1}"
    
    ax5.text(0.5, 0.5, summary_text, ha='center', va='center',
            fontsize=11, fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax5.axis('off')
    
    plt.suptitle('Quantum Federated Learning - Round Comparison Dashboard',
                fontsize=16, fontweight='bold', y=0.98)
    
    save_path = viz_path / filename
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return str(save_path)


def generate_results_table(results: Dict,
                           save_dir: str = "./visualizations/qfl",
                           filename: str = "results_summary.csv") -> str:

    viz_path = ensure_viz_dir(save_dir)
    
    metrics = {
        'Metric': [],
        'Value': []
    }
    
    metrics['Metric'].append('Final Test Accuracy')
    metrics['Value'].append(f"{results.get('final_accuracy', 0):.4f}")
    
    metrics['Metric'].append('Best Test Accuracy')
    metrics['Value'].append(f"{results.get('best_accuracy', 0):.4f}")
    
    if 'test_losses' in results and results['test_losses']:
        metrics['Metric'].append('Final Test Loss')
        metrics['Value'].append(f"{results['test_losses'][-1]:.4f}")
    
    if 'train_losses' in results and results['train_losses']:
        avg_train_loss = np.mean([l for l in results['train_losses'] if l != 0 and l != float('inf')])
        metrics['Metric'].append('Avg Train Loss')
        metrics['Value'].append(f"{avg_train_loss:.4f}")
    
    if 'test_accuracies' in results:
        metrics['Metric'].append('Total Rounds')
        metrics['Value'].append(f"{len(results['test_accuracies']) - 1}")
    
    df = pd.DataFrame(metrics)
    
    save_path = viz_path / filename
    df.to_csv(save_path, index=False)
    
    return str(save_path)


def generate_all_qfl_plots(results: Dict,
                           client_data: Optional[List[Tuple]] = None,
                           y_test: Optional[np.ndarray] = None,
                           y_pred: Optional[np.ndarray] = None,
                           class_names: Optional[List[str]] = None,
                           save_dir: str = "./visualizations/qfl") -> Dict[str, str]:

    saved_plots = {}
    
    try:
        # Global convergence
        if 'test_accuracies' in results:
            path = plot_global_convergence(
                results['test_accuracies'],
                results['test_losses'],
                results['train_losses'],
                save_dir
            )
            saved_plots['global_convergence'] = path
            print(f"✓ Saved: {path}")
        
        # Client loss heatmap
        if 'client_losses' in results and results['client_losses']:
            path = plot_client_loss_heatmap(results['client_losses'], save_dir)
            saved_plots['client_loss_heatmap'] = path
            print(f"✓ Saved: {path}")
            
            # Client loss boxplot
            path = plot_client_boxplot(results['client_losses'], save_dir)
            if path:
                saved_plots['client_loss_boxplot'] = path
                print(f"✓ Saved: {path}")
        
        # Client contribution
        if client_data is not None:
            path = plot_client_contribution(client_data, save_dir)
            saved_plots['client_contribution'] = path
            print(f"✓ Saved: {path}")
            
            # Federated system graph
            path = plot_federated_system_graph(len(client_data), save_dir)
            saved_plots['federated_system'] = path
            print(f"✓ Saved: {path}")
        
        # Quantum-classical balance
        if 'model' in results:
            path = plot_quantum_classical_balance(results['model'], save_dir)
            saved_plots['quantum_classical_balance'] = path
            print(f"✓ Saved: {path}")
        
        # Dashboard
        if 'test_accuracies' in results:
            path = plot_round_comparison_dashboard(
                results['test_accuracies'],
                results['test_losses'],
                results.get('client_losses', []),
                save_dir
            )
            saved_plots['dashboard'] = path
            print(f"✓ Saved: {path}")
        
        # Confusion matrix
        if y_test is not None and y_pred is not None:
            path = plot_federated_confusion_matrix(y_test, y_pred, class_names, save_dir)
            saved_plots['confusion_matrix'] = path
            print(f"✓ Saved: {path}")
        
        # Results table
        path = generate_results_table(results, save_dir)
        saved_plots['results_table'] = path
        print(f"✓ Saved: {path}")
        
    except Exception as e:
        print(f"⚠ Warning: Some plots could not be generated: {e}")
        import traceback
        traceback.print_exc()
    
    return saved_plots


def plot_client_loss_heatmap(client_losses: List[List[float]],
                             save_dir: str = "./visualizations/qfl",
                             filename: str = "client_loss_heatmap.png") -> str:

    viz_path = ensure_viz_dir(save_dir)
    
    # Convert to array (handle variable client counts per round)
    max_clients = max(len(losses) for losses in client_losses if losses)
    loss_matrix = []
    
    for round_losses in client_losses:
        if round_losses:
            # Pad with NaN if fewer clients in this round
            padded = list(round_losses) + [np.nan] * (max_clients - len(round_losses))
            loss_matrix.append(padded)
    
    loss_matrix = np.array(loss_matrix)
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(client_losses) * 0.5)))
    
    # Create heatmap
    sns.heatmap(loss_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
               cbar_kws={'label': 'Loss'}, ax=ax, 
               xticklabels=[f'Client {i}' for i in range(max_clients)],
               yticklabels=[f'Round {i+1}' for i in range(len(client_losses))])
    
    ax.set_xlabel('Client ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('Federated Round', fontsize=12, fontweight='bold')
    ax.set_title('Client Loss Across FL Rounds', fontsize=14, fontweight='bold')
    
    save_path = viz_path / filename
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return str(save_path)


def plot_client_contribution(client_data: List[Tuple],
                             save_dir: str = "./visualizations/qfl",
                             filename: str = "client_contribution.png") -> str:

    viz_path = ensure_viz_dir(save_dir)
    
    n_clients = len(client_data)
    sample_counts = [len(X) for X, _ in client_data]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Sample distribution
    client_ids = [f'Client {i}' for i in range(n_clients)]
    bars = ax1.bar(client_ids, sample_counts, color='steelblue', alpha=0.7, edgecolor='black')
    
    for bar, count in zip(bars, sample_counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.set_xlabel('Client', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax1.set_title('Client Data Distribution', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Class distribution per client
    class_distributions = []
    for _, y in client_data:
        y_np = y.numpy() if hasattr(y, 'numpy') else y
        class_counts = np.bincount(y_np)
        class_distributions.append(class_counts)
    
    # Stack bars for class distribution
    class_distributions = np.array(class_distributions)
    n_classes = class_distributions.shape[1]
    
    x_pos = np.arange(n_clients)
    colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
    
    bottom = np.zeros(n_clients)
    for i in range(n_classes):
        ax2.bar(x_pos, class_distributions[:, i], bottom=bottom, 
               label=f'Class {i}', color=colors[i], alpha=0.8, edgecolor='black')
        bottom += class_distributions[:, i]
    
    ax2.set_xlabel('Client', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Sample Count', fontsize=12, fontweight='bold')
    ax2.set_title('Per-Client Class Distribution', fontsize=13, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(client_ids)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Client Contribution Analysis', fontsize=15, fontweight='bold', y=1.02)
    
    save_path = viz_path / filename
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    return str(save_path)