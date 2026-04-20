"""Shared visualization utilities."""
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

DEFAULT_DPI = 300


def setup_plot_style(dpi: int = DEFAULT_DPI, font_size: int = 10, style: str = "whitegrid") -> None:
    sns.set_style(style)
    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams['savefig.dpi'] = dpi
    plt.rcParams['font.size'] = font_size


def save_figure(fig: plt.Figure, save_dir: str, filename: str, dpi: int = DEFAULT_DPI) -> str:
    save_path = Path(save_dir) / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)
