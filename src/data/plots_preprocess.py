from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from core.plot_utils import save_figure
from core.utils import get_logger

logger = get_logger(__name__)


def plot_pca_variance(pca_model: object, save_dir: str) -> None:
    if pca_model is None or not hasattr(pca_model, 'explained_variance_ratio_'):
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ev = pca_model.explained_variance_ratio_
    axes[0].bar(range(1, len(ev) + 1), ev)
    axes[0].set_xlabel('Component')
    axes[0].set_ylabel('Explained Variance Ratio')
    axes[0].set_title('Individual')
    axes[1].plot(range(1, len(ev) + 1), np.cumsum(ev), 'o-')
    axes[1].set_xlabel('Component')
    axes[1].set_ylabel('Cumulative')
    axes[1].set_title('Cumulative Explained Variance')
    plt.suptitle('PCA Variance Analysis')
    plt.tight_layout()
    save_figure(fig, save_dir, 'pca_variance.png', dpi=150)


def plot_feature_distribution(
    data_before_scaling: np.ndarray,
    data_after_scaling: np.ndarray,
    save_dir: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for col in [data_before_scaling[:, 0], data_before_scaling[:, 1]]:
        if len(col) > 0:
            axes[0].hist(col, bins=50, alpha=0.7)
    axes[0].set_title('Before Scaling')
    axes[0].set_xlabel('Feature Value')
    for col in [data_after_scaling[:, 0], data_after_scaling[:, 1]]:
        if len(col) > 0:
            axes[1].hist(col, bins=50, alpha=0.7)
    axes[1].set_title('After Scaling')
    axes[1].set_xlabel('Feature Value')
    plt.suptitle('Feature Distribution Before/After Scaling')
    plt.tight_layout()
    save_figure(fig, save_dir, 'feature_distribution.png', dpi=150)


def plot_client_data_shift(client_data: list, save_dir: str) -> None:
    num_clients = len(client_data)
    if num_clients == 0:
        return
    fig, axes = plt.subplots(1, min(num_clients, 4), figsize=(12, 3))
    if num_clients == 1:
        axes = [axes]
    for i in range(min(num_clients, 4)):
        X, y = client_data[i]
        X_np = X.numpy() if hasattr(X, 'numpy') else X
        if X_np.ndim == 2 and X_np.shape[1] >= 2:
            axes[i].scatter(X_np[:, 0], X_np[:, 1], c=y, cmap='tab10', s=8, alpha=0.6)
        axes[i].set_title(f'Client {i+1}')
        axes[i].set_xlabel('Feature 1')
    plt.suptitle('Client Feature Space Shift')
    plt.tight_layout()
    save_figure(fig, save_dir, 'client_data_shift.png', dpi=150)


def generate_all_preprocessing_visualizations(
    pca_model: object = None,
    data_before_scaling: Optional[np.ndarray] = None,
    data_after_scaling: Optional[np.ndarray] = None,
    client_indices: Optional[list[np.ndarray]] = None,
    client_data: Optional[list] = None,
    y_train: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    metadata: Optional[dict] = None,
    num_classes: int = 3,
    save_dir: str = "./results/preprocessing",
) -> None:
    plot_pca_variance(pca_model, save_dir)
    if data_before_scaling is not None and data_after_scaling is not None:
        plot_feature_distribution(data_before_scaling, data_after_scaling, save_dir)
    if client_data is not None:
        plot_client_data_shift(client_data, save_dir)
