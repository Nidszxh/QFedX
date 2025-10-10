"""
Flower client for Quantum Federated Learning (QFL).
Each client:
 - loads its local partition (from volume or synthetic fallback)
 - syncs parameters with the server
 - trains locally using QNNTrainer
 - returns updated parameters and metrics
 - exposes Prometheus metrics (port configurable)
 - logs to TensorBoard (/workspace/logs/client-<id>)
"""

import os
import socket
import time
import json
import torch
import numpy as np
import flwr as fl
from flwr.client import NumPyClient, start_numpy_client
from prometheus_client import start_http_server, Gauge
from torch.utils.tensorboard import SummaryWriter
from typing import List, Dict, Any

# Import QNN modules
from QFed.qNN import QNNConfig, QuantumNeuralNetwork, QNNTrainer

# ===========================
# Prometheus Metrics Setup
# ===========================
PROM_PORT = int(os.getenv("PROM_PORT", "9103"))
start_http_server(PROM_PORT)
g_train_loss = Gauge("qfl_client_train_loss", "Client training loss")
g_val_acc = Gauge("qfl_client_val_acc", "Client validation accuracy")

# ===========================
# Helpers for parameter exchange
# ===========================
def state_dict_to_ndarrays(state: Dict[str, torch.Tensor]) -> List[np.ndarray]:
    return [v.detach().cpu().numpy() for v in state.values()]

def ndarrays_to_state_dict(arrs: List[np.ndarray], template_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    keys = list(template_state.keys())
    if len(arrs) != len(keys):
        raise ValueError(f"Parameter length mismatch: got {len(arrs)} vs expected {len(keys)}")
    return {k: torch.tensor(arr, dtype=template_state[k].dtype) for k, arr in zip(keys, arrs)}

# ===========================
# Client Definition
# ===========================
class QFLClient(NumPyClient):
    def __init__(self, client_id: str, local_data, qcfg: QNNConfig, fl_config: Dict[str, Any], tb_logdir: str):
        self.client_id = client_id
        self.qcfg = qcfg
        self.fl_config = fl_config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Model + Trainer
        self.model = QuantumNeuralNetwork(self.qcfg).to(self.device)
        self.trainer = QNNTrainer(self.model, self.qcfg, device=self.device)

        # Data
        X_local, y_local = local_data
        self.X_local = torch.as_tensor(X_local, dtype=torch.float32)
        self.y_local = torch.as_tensor(y_local, dtype=torch.long)
        self.tb_writer = SummaryWriter(log_dir=tb_logdir)

        # Template for mapping weights
        self.template_state = self.model.state_dict()

    # ---------------------------
    # FLWR interface methods
    # ---------------------------
    def get_parameters(self, config):
        """Return local model weights."""
        return state_dict_to_ndarrays(self.model.state_dict())

    def fit(self, parameters, config):
        """Train locally and return updated weights."""
        incoming_state = ndarrays_to_state_dict(parameters, self.template_state)
        self.model.load_state_dict(incoming_state)

        batch_size = self.fl_config.get("batch_size", 16)
        local_epochs = self.fl_config.get("local_epochs", 1)
        from torch.utils.data import DataLoader, TensorDataset
        loader = DataLoader(TensorDataset(self.X_local, self.y_local), batch_size=batch_size, shuffle=True)

        last_loss = None
        for e in range(local_epochs):
            loss, acc = self.trainer.train_epoch(loader)
            last_loss = loss
            # Log metrics
            self.tb_writer.add_scalar("train/loss", loss, e)
            self.tb_writer.add_scalar("train/acc", acc, e)
            g_train_loss.set(loss)
        self.tb_writer.flush()

        return state_dict_to_ndarrays(self.model.state_dict()), len(self.X_local), {"loss": float(last_loss)}

    def evaluate(self, parameters, config):
        """Evaluate global model on local data."""
        incoming_state = ndarrays_to_state_dict(parameters, self.template_state)
        self.model.load_state_dict(incoming_state)

        from torch.utils.data import DataLoader, TensorDataset
        loader = DataLoader(TensorDataset(self.X_local, self.y_local),
                            batch_size=self.qcfg.batch_size, shuffle=False)
        val_loss, val_acc = self.trainer.evaluate(loader)
        g_val_acc.set(val_acc)

        return float(val_loss), len(self.X_local), {"val_acc": float(val_acc)}

# ===========================
# Data Loading
# ===========================
def load_local_partition():
    """Load local client partition (or synthetic fallback)."""
    partition_path = os.getenv("LOCAL_PARTITION")
    qcfg = QNNConfig()

    if partition_path and os.path.exists(partition_path):
        d = torch.load(partition_path)
        X = d["X"].numpy() if torch.is_tensor(d["X"]) else d["X"]
        y = d["y"].numpy() if torch.is_tensor(d["y"]) else d["y"]
        return X, y, qcfg
    else:
        np.random.seed(int(os.getenv("SEED", "42")))
        n = int(os.getenv("LOCAL_SAMPLES", "200"))
        X = np.random.randn(n, qcfg.n_features).astype(np.float32)
        y = np.random.randint(0, qcfg.n_classes, size=(n,))
        return X, y, qcfg

# ===========================
# Main
# ===========================
def main():
    server_address = os.getenv("SERVER_ADDRESS", "localhost:8080")
    client_id = os.getenv("CLIENT_ID", socket.gethostname())
    tb_dir = os.getenv("TB_LOGDIR", f"/workspace/logs/{client_id}")
    os.makedirs(tb_dir, exist_ok=True)

    # Load data
    X_local, y_local, qcfg = load_local_partition()

    fl_config = {
        "local_epochs": int(os.getenv("LOCAL_EPOCHS", "1")),
        "batch_size": int(os.getenv("BATCH_SIZE", "16")),
    }

    client = QFLClient(client_id, (X_local, y_local), qcfg, fl_config, tb_logdir=tb_dir)

    print(f"[Client {client_id}] Connecting to server at {server_address}, prometheus_port={PROM_PORT}")

    # Modern API (non-deprecated)
    start_numpy_client(server_address=server_address, client=client)

if __name__ == "__main__":
    main()
