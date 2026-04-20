"""Main entry point for QFedX experiments using Hydra configuration."""
import itertools
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import mlflow
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from core.data import create_dataloader
from core.defs import DEFAULT_MLFLOW_URI
from core.utils import get_logger, set_seed
from data.preprocess import preprocess_mnist
from quantum.qfl import QuantumFederatedLearning
from quantum.qnn import (
    QuantumNeuralNetwork,
    QuantumNeuralNetworkConfig,
    QuantumNeuralNetworkTrainer,
    load_iris_data,
    verify_gradients,
)

logger = get_logger(__name__)

os.environ.setdefault("MLFLOW_TRACKING_URI", DEFAULT_MLFLOW_URI)


def _resolve_device(cfg: DictConfig) -> str:
    return 'cuda' if torch.cuda.is_available() and cfg.get('device', 'cpu') != 'cpu' else 'cpu'


def run_qnn(cfg: DictConfig) -> dict[str, Any]:
    qnn_cfg = QuantumNeuralNetworkConfig.from_hydra(cfg)
    device = _resolve_device(cfg)

    train_set, val_set, test_set, _ = preprocess_mnist(**cfg)
    X_train, y_train = train_set
    X_val, y_val = val_set
    X_test, y_test = test_set
    qnn_cfg.n_features = X_train.shape[1]

    train_loader = create_dataloader(X_train, y_train, batch_size=qnn_cfg.batch_size, shuffle=True)
    val_loader = create_dataloader(X_val, y_val, batch_size=qnn_cfg.batch_size, shuffle=False)
    test_loader = create_dataloader(X_test, y_test, batch_size=qnn_cfg.batch_size, shuffle=False)

    model = QuantumNeuralNetwork(qnn_cfg)
    trainer = QuantumNeuralNetworkTrainer(model, qnn_cfg, device)
    history = trainer.train(train_loader, val_loader)
    test_loss, test_acc = trainer.evaluate(test_loader)
    logger.info(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

    return {"history": history, "test_accuracy": test_acc}


def run_qfl(cfg: DictConfig) -> dict[str, Any]:
    qnn_cfg = QuantumNeuralNetworkConfig.from_hydra(cfg)
    device = _resolve_device(cfg)

    fl_cfg = {
        k: cfg.get(k) for k in (
            "num_rounds", "local_epochs", "batch_size",
            "classical_lr", "quantum_lr", "client_fraction", "grad_clip",
            "weight_decay",
            "dp_enabled", "dp_clip_norm", "dp_noise_multiplier", "dp_delta",
            "secure_aggregation", "num_clients",
            "early_stop_patience", "early_stop_delta",
            "checkpoint_dir", "checkpoint_interval",
        )
    }

    result = preprocess_mnist(**cfg)
    if result is None:
        raise ValueError("Preprocessing failed")
    _, _, test_data, client_data = result
    qnn_cfg.n_features = client_data[0][0].shape[1]

    qfl = QuantumFederatedLearning(qnn_cfg, fl_cfg, device)
    results = qfl.train(client_data, test_data)
    qfl.save_results()
    return results


def run_centralized(cfg: DictConfig) -> dict[str, Any]:
    qnn_cfg = QuantumNeuralNetworkConfig.from_hydra(cfg)
    device = _resolve_device(cfg)

    result = preprocess_mnist(**cfg)
    if result is None:
        raise ValueError("Preprocessing failed")
    train_set, val_set, test_set, client_data = result
    X_train_all = torch.cat([torch.as_tensor(X) for X, _ in client_data])
    y_train_all = torch.cat([torch.as_tensor(y) for _, y in client_data])
    qnn_cfg.n_features = X_train_all.shape[1]

    train_loader = create_dataloader(X_train_all, y_train_all, batch_size=qnn_cfg.batch_size, shuffle=True)
    val_loader = create_dataloader(*val_set, batch_size=qnn_cfg.batch_size, shuffle=False)
    test_loader = create_dataloader(*test_set, batch_size=qnn_cfg.batch_size, shuffle=False)

    model = QuantumNeuralNetwork(qnn_cfg)
    trainer = QuantumNeuralNetworkTrainer(model, qnn_cfg, device)
    history = trainer.train(train_loader, val_loader)
    test_loss, test_acc = trainer.evaluate(test_loader)
    logger.info(f"Centralized Test Accuracy: {test_acc:.4f}")
    return {"history": history, "test_accuracy": test_acc}


def run_iris(cfg: DictConfig) -> dict[str, Any]:
    qnn_cfg = QuantumNeuralNetworkConfig.from_hydra(cfg)
    train_loader, val_loader, test_loader = load_iris_data(qnn_cfg)
    model = QuantumNeuralNetwork(qnn_cfg)
    trainer = QuantumNeuralNetworkTrainer(model, qnn_cfg, 'cpu')
    history = trainer.train(train_loader, val_loader)
    test_loss, test_acc = trainer.evaluate(test_loader)
    logger.info(f"Iris Test Accuracy: {test_acc:.4f}")
    return {"history": history, "test_accuracy": test_acc}


def run_verify_gradients(cfg: DictConfig) -> dict[str, Any]:
    qnn_cfg = QuantumNeuralNetworkConfig.from_hydra(cfg)
    passed, diff, cosine = verify_gradients(qnn_cfg)
    return {"grad_diff": diff, "grad_cosine": cosine, "passed": passed}


def _flatten_grid(grid: dict) -> list:
    keys = list(grid.keys())
    values = list(grid.values())
    combos = list(itertools.product(*values))
    return [dict(zip(keys, combo)) for combo in combos]


def _run_single_experiment(params: dict, base_data_config: dict, seed: int = 42):
    from core.utils import set_seed as _set_seed
    _set_seed(seed)
    logger.info(f"\n{'='*70}")
    logger.info(f"Experiment: {json.dumps(params, indent=2)}")
    logger.info(f"{'='*70}")

    qnn_config = QuantumNeuralNetworkConfig(
        n_qubits=params.get('n_qubits', 4),
        n_layers=params.get('n_layers', 2),
        n_readout=None,
        encoding='amplitude',
        entanglement='circular',
        n_features=base_data_config.get('pca_components', 4),
        n_classes=3,
        batch_size=16,
        epochs=params.get('local_epochs', 3),
        classical_lr=1e-3,
        quantum_lr=5e-4,
        grad_clip=1.0,
        noise_type=params.get('noise_type', 'none'),
        depolarizing_p=params.get('depolarizing_p', 0.001),
        amplitude_gamma=params.get('amplitude_gamma', 0.001),
        readout_flip_prob=params.get('readout_flip_prob', 0.01),
        dp_enabled=params.get('dp_enabled', False),
        dp_clip_norm=params.get('dp_clip_norm', 1.0),
        dp_noise_multiplier=params.get('dp_noise_multiplier', 1.0),
    )

    fl_config = {
        'num_rounds': params.get('num_rounds', 5),
        'local_epochs': params.get('local_epochs', 3),
        'batch_size': 16,
        'classical_lr': 1e-3,
        'quantum_lr': 5e-4,
        'client_fraction': params.get('client_fraction', 0.75),
        'grad_clip': 1.0,
        'dp_enabled': params.get('dp_enabled', False),
        'dp_clip_norm': params.get('dp_clip_norm', 1.0),
        'dp_noise_multiplier': params.get('dp_noise_multiplier', 1.0),
        'dp_delta': 1e-5,
        'secure_aggregation': params.get('secure_aggregation', False),
        'num_clients': base_data_config.get('num_clients', 4),
    }

    data_config = dict(base_data_config)
    data_config['partition_type'] = params.get('partition_type', 'iid')
    data_config['alpha'] = 0.5 if data_config['partition_type'] == 'non_iid' else None

    try:
        result = preprocess_mnist(**data_config, generate_plots=False)
        if result is None:
            return None
        _, _, test_data, client_data = result
        qnn_config.n_features = client_data[0][0].shape[1]
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        return None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    qfl = QuantumFederatedLearning(qnn_config, fl_config, device)

    start_time = time.time()
    results = qfl.train(client_data, test_data)
    elapsed = time.time() - start_time

    if results:
        results['wall_time'] = elapsed
        results['params'] = params
        accs = results.get('test_accuracies', [])
        results['final_accuracy'] = accs[-1] if accs else 0.0
        results['best_accuracy'] = max(accs[1:]) if len(accs) > 1 else (accs[-1] if accs else 0.0)
        results['total_rounds'] = len(accs) - 1 if accs else 0
        if qfl.privacy_accountant is not None:
            results['privacy_epsilon'] = qfl.privacy_accountant.get_privacy_spent()

    qfl.save_results()
    return results


def run_experiment_grid(cfg: DictConfig) -> dict[str, Any]:
    grid = {
        'n_qubits': [2, 4],
        'n_layers': [1, 2],
        'client_fraction': [0.75],
        'partition_type': ['iid'],
        'noise_type': ['none', 'depolarizing'],
        'depolarizing_p': [0.001],
        'dp_enabled': [False, True],
        'dp_noise_multiplier': [1.0],
    }
    data_cfg = {k: cfg.get(k) for k in (
        "raw_folder", "processed_folder", "digits", "val_split",
        "num_clients", "partition_type", "alpha", "apply_pca",
        "pca_components", "generate_plots",
    )}

    experiments = _flatten_grid(grid)
    max_exp = cfg.get('max_exp', 5)
    if max_exp:
        experiments = experiments[:max_exp]

    logger.info(f"\nExperiment Grid: {len(experiments)} configurations")
    all_results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i, params in enumerate(experiments):
        run_name = f"exp_{i:03d}"
        logger.info(f"\n--- Run {run_name} ---")
        result = _run_single_experiment(params, data_cfg)
        if result:
            all_results.append(result)

    results_path = Path(f"./results/experiment_grid_{timestamp}.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    summary = []
    for r in all_results:
        summary.append({
            'params': r.get('params', {}),
            'final_accuracy': r.get('final_accuracy', 0),
            'best_accuracy': r.get('best_accuracy', 0),
            'wall_time': r.get('wall_time', 0),
            'privacy_epsilon': r.get('privacy_epsilon', None),
            'total_rounds': r.get('total_rounds', 0),
        })
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nResults saved to {results_path}")

    if summary:
        accuracies = [s['final_accuracy'] for s in summary if s['final_accuracy']]
        logger.info(f"\nGrid Summary ({len(summary)} runs):")
        logger.info(f"  Mean accuracy: {np.mean(accuracies):.4f}")
        logger.info(f"  Std accuracy:  {np.std(accuracies):.4f}")
        logger.info(f"  Max accuracy:  {np.max(accuracies):.4f}")
        logger.info(f"  Min accuracy:  {np.min(accuracies):.4f}")

    return {"grid_results": all_results}


def run_ablation(cfg: DictConfig) -> dict[str, Any]:
    logger.info("\n" + "="*70)
    logger.info("ABLATION STUDY")
    logger.info("="*70)

    base_data_config = {
        'raw_folder': "./dataset/raw",
        'processed_folder': "./dataset/processed",
        'digits': (0, 1, 2),
        'val_split': 0.1,
        'num_clients': 4,
        'partition_type': 'iid',
        'apply_pca': True,
        'pca_components': 4,
    }

    ablations = [
        {"name": "Baseline (4q, 2l, no noise, no DP)", "n_qubits": 4, "n_layers": 2,
         "noise_type": "none", "dp_enabled": False},
        {"name": "Depolarizing noise (p=0.001)", "n_qubits": 4, "n_layers": 2,
         "noise_type": "depolarizing", "depolarizing_p": 0.001, "dp_enabled": False},
        {"name": "Depolarizing noise (p=0.01)", "n_qubits": 4, "n_layers": 2,
         "noise_type": "depolarizing", "depolarizing_p": 0.01, "dp_enabled": False},
        {"name": "Amplitude damping (γ=0.001)", "n_qubits": 4, "n_layers": 2,
         "noise_type": "amplitude_damping", "amplitude_gamma": 0.001, "dp_enabled": False},
        {"name": "With DP (clip=1.0, noise=1.0)", "n_qubits": 4, "n_layers": 2,
         "noise_type": "none", "dp_enabled": True, "dp_clip_norm": 1.0, "dp_noise_multiplier": 1.0},
        {"name": "With DP (clip=0.1, noise=0.5)", "n_qubits": 4, "n_layers": 2,
         "noise_type": "none", "dp_enabled": True, "dp_clip_norm": 0.1, "dp_noise_multiplier": 0.5},
        {"name": "8 qubits", "n_qubits": 8, "n_layers": 2,
         "noise_type": "none", "dp_enabled": False},
        {"name": "3 layers", "n_qubits": 4, "n_layers": 3,
         "noise_type": "none", "dp_enabled": False},
        {"name": "Non-IID partition", "n_qubits": 4, "n_layers": 2,
         "noise_type": "none", "dp_enabled": False, "partition_type": "non_iid"},
        {"name": "Full noise + DP", "n_qubits": 4, "n_layers": 2,
         "noise_type": "combined", "depolarizing_p": 0.001, "amplitude_gamma": 0.001, "readout_flip_prob": 0.01,
         "dp_enabled": True, "dp_clip_norm": 1.0, "dp_noise_multiplier": 1.0},
    ]

    results_list = []
    for ablation in ablations:
        name = ablation.pop("name")
        logger.info(f"\n--- Ablation: {name} ---")
        result = _run_single_experiment(ablation, base_data_config)
        if result:
            result['ablation_name'] = name
            results_list.append(result)
        ablation['name'] = name

    logger.info("\n" + "="*70)
    logger.info("ABLATION STUDY SUMMARY")
    logger.info("="*70)
    for r in results_list:
        rname = r.get('ablation_name', 'Unknown')
        acc = r.get('final_accuracy', 0)
        best = r.get('best_accuracy', 0)
        eps = r.get('privacy_epsilon', None)
        eps_str = f", ε={eps:.2f}" if eps else ""
        logger.info(f"  {rname:35s}: final={acc:.4f}, best={best:.4f}{eps_str}")

    return {"ablation_results": results_list}


_MODE_DISPATCH: dict[str, tuple] = {
    "qnn": (run_qnn, lambda r, c: {"test_accuracy": r["test_accuracy"]}),
    "qfl": (run_qfl, lambda r, c: {"best_accuracy": r.get("best_accuracy", 0)}),
    "centralized": (run_centralized, lambda r, c: {"test_accuracy": r["test_accuracy"]}),
    "iris": (run_iris, lambda r, c: {"test_accuracy": r["test_accuracy"]}),
    "verify_gradients": (run_verify_gradients, lambda r, c: {"grad_diff": r["grad_diff"], "grad_cosine": r["grad_cosine"]}),
    "qfl_noise": (run_qfl, lambda r, c: {"best_accuracy": r.get("best_accuracy", 0), "_param:noise_type": c.get("noise_type", "none")}),
    "qfl_dp": (run_qfl, lambda r, c: {"best_accuracy": r.get("best_accuracy", 0), "privacy_epsilon": r.get("privacy_epsilon", 0)}),
    "experiment_grid": (run_experiment_grid, lambda r, c: {}),
    "ablation": (run_ablation, lambda r, c: {}),
}


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    logger.info("QFedX Experiment")
    logger.info(OmegaConf.to_yaml(cfg))

    tracking_uri = str(Path(cfg.get("mlflow_tracking_uri", DEFAULT_MLFLOW_URI)).resolve())
    mlflow.set_tracking_uri(tracking_uri)

    with mlflow.start_run(run_name=cfg.get("experiment_name", "default")):
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))

        entry = _MODE_DISPATCH.get(cfg.mode)
        if entry is None:
            raise ValueError(f"Unknown mode: {cfg.mode}")
        runner, log_fn = entry

        results = runner(cfg)
        for key, value in log_fn(results, cfg).items():
            if key.startswith("_param:"):
                mlflow.log_param(key.removeprefix("_param:"), value)
            else:
                mlflow.log_metric(key, value)

    logger.info("Done.")


if __name__ == "__main__":
    main()
