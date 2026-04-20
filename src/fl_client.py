"""
Flower client for Quantum Federated Learning (QFL).
Each client:
 - loads its local partition (from volume or synthetic fallback)
 - syncs parameters with the server
 - trains locally using QuantumNeuralNetworkTrainer
 - returns updated parameters and metrics
 - exposes Prometheus metrics (port configurable)
 - logs to TensorBoard (/workspace/logs/client-<id>)
"""

import os
import socket
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from flwr.client import NumPyClient, start_numpy_client
from prometheus_client import Gauge, start_http_server
from torch.utils.tensorboard import SummaryWriter

from core.data import create_dataloader
from core.defs import DEFAULT_CLIENT_PROM_PORT
from core.fl import ndarrays_to_state_dict, state_dict_to_ndarrays
from core.utils import get_logger
from quantum.qnn import QuantumNeuralNetwork, QuantumNeuralNetworkConfig, QuantumNeuralNetworkTrainer

logger = get_logger(__name__)

PROM_PORT: int = int(os.getenv("PROM_PORT", DEFAULT_CLIENT_PROM_PORT))
g_train_loss: Gauge = Gauge("qfl_client_train_loss", "Client training loss")
g_val_acc: Gauge = Gauge("qfl_client_val_acc", "Client validation accuracy")

# ===========================
# Client Definition
# ===========================
class QFLClient(NumPyClient):
    """Flower NumPyClient for Quantum Federated Learning."""

    def __init__(
        self,
        client_id: str,
        local_data: tuple[np.ndarray, np.ndarray],
        qcfg: QuantumNeuralNetworkConfig,
        fl_config: dict[str, Any],
        tb_logdir: str,
    ) -> None:
        self.client_id = client_id
        self.qcfg = qcfg
        self.fl_config = fl_config
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"

        # Model + Trainer
        self.model = QuantumNeuralNetwork(self.qcfg).to(self.device)
        self.trainer = QuantumNeuralNetworkTrainer(self.model, self.qcfg, device=self.device)

        # Data
        X_local, y_local = local_data
        self.X_local = torch.as_tensor(X_local, dtype=torch.float32)
        self.y_local = torch.as_tensor(y_local, dtype=torch.long)
        self.tb_writer = SummaryWriter(log_dir=tb_logdir)

        # Template for mapping weights
        self.template_state = self.model.state_dict()

    def close(self) -> None:
        if hasattr(self, 'tb_writer'):
            self.tb_writer.close()

    def __del__(self) -> None:
        self.close()

    # ---------------------------
    # FLWR interface methods
    # ---------------------------
    def get_parameters(self, config: dict[str, Any]) -> list[np.ndarray]:
        """Return local model weights."""
        return state_dict_to_ndarrays(self.model.state_dict())

    def fit(
        self,
        parameters: list[np.ndarray],
        config: dict[str, Any],
    ) -> tuple[list[np.ndarray], int, dict[str, float]]:
        """Train locally and return updated weights."""
        incoming_state = ndarrays_to_state_dict(parameters, self.template_state)
        self.model.load_state_dict(incoming_state)

        batch_size: int = self.fl_config.get("batch_size", 16)
        local_epochs: int = self.fl_config.get("local_epochs", 1)
        loader = create_dataloader(self.X_local, self.y_local, batch_size=batch_size, shuffle=True)

        last_loss: float = 0.0
        for e in range(local_epochs):
            loss, acc = self.trainer.train_epoch(loader)
            last_loss = loss
            # Log metrics
            self.tb_writer.add_scalar("train/loss", loss, e)
            self.tb_writer.add_scalar("train/acc", acc, e)
            g_train_loss.set(loss)
        self.tb_writer.flush()

        return state_dict_to_ndarrays(self.model.state_dict()), len(self.X_local), {"loss": float(last_loss)}

    def evaluate(
        self,
        parameters: list[np.ndarray],
        config: dict[str, Any],
    ) -> tuple[float, int, dict[str, float]]:
        """Evaluate global model on local data."""
        incoming_state = ndarrays_to_state_dict(parameters, self.template_state)
        self.model.load_state_dict(incoming_state)

        loader = create_dataloader(self.X_local, self.y_local,
                                   batch_size=self.qcfg.batch_size, shuffle=False)
        val_loss, val_acc = self.trainer.evaluate(loader)
        g_val_acc.set(val_acc)

        return float(val_loss), len(self.X_local), {"val_acc": float(val_acc)}

# ===========================
# Data Loading
# ===========================
def load_local_partition() -> tuple[np.ndarray, np.ndarray, QuantumNeuralNetworkConfig]:
    """Load local client partition (or synthetic fallback)."""
    partition_path: Optional[str] = os.getenv("LOCAL_PARTITION")
    qcfg = QuantumNeuralNetworkConfig()

    if partition_path and Path(partition_path).exists():
        d = torch.load(partition_path, weights_only=True)
        X = d["X"].numpy() if torch.is_tensor(d["X"]) else d["X"]
        y = d["y"].numpy() if torch.is_tensor(d["y"]) else d["y"]
        return X, y, qcfg
    else:
        seed = int(os.getenv("SEED", "42"))
        rng = np.random.default_rng(seed)
        n: int = int(os.getenv("LOCAL_SAMPLES", "200"))
        X = rng.standard_normal((n, qcfg.n_features)).astype(np.float32)
        y = rng.integers(0, qcfg.n_classes, size=(n,))
        return X, y, qcfg

# ===========================
# Main
# ===========================
def main() -> None:
    """Main entry point for the Flower client."""
    server_address: str = os.getenv("SERVER_ADDRESS", "localhost:8080")
    client_id: str = os.getenv("CLIENT_ID", socket.gethostname())
    tb_dir: str = os.getenv("TB_LOGDIR", f"/workspace/logs/{client_id}")
    Path(tb_dir).mkdir(parents=True, exist_ok=True)

    start_http_server(PROM_PORT)

    # Load data
    X_local, y_local, qcfg = load_local_partition()

    fl_config: dict[str, Any] = {
        "local_epochs": int(os.getenv("LOCAL_EPOCHS", "1")),
        "batch_size": int(os.getenv("BATCH_SIZE", "16")),
    }

    client = QFLClient(client_id, (X_local, y_local), qcfg, fl_config, tb_logdir=tb_dir)

    logger.info(f"[Client {client_id}] Connecting to server at {server_address}, prometheus_port={PROM_PORT}")

    # Modern API (non-deprecated)
    start_numpy_client(server_address=server_address, client=client)

if __name__ == "__main__":
    main()
