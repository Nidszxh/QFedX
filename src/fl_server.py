"""
Flower Server for Quantum Federated Learning (QFL)
- Aggregates client updates using weighted FedAvg
- Logs metrics to Prometheus
- Compatible with Flower v1.7+ / v2.x
"""

import os
from typing import Optional

import flwr as fl
import numpy as np
from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.typing import FitRes, Metrics
from flwr.server.client_proxy import ClientProxy
from prometheus_client import Gauge, start_http_server

from core.defs import DEFAULT_SERVER_PROM_PORT
from core.utils import get_logger

logger = get_logger(__name__)

# ===========================
# Prometheus Metrics
# ===========================
PROM_PORT: int = int(os.getenv("PROM_PORT", str(DEFAULT_SERVER_PROM_PORT)))
g_round: Gauge = Gauge("qfl_server_round", "Current training round")
g_avg_loss: Gauge = Gauge("qfl_server_avg_loss", "Average client loss")
g_avg_acc: Gauge = Gauge("qfl_server_avg_acc", "Average client accuracy")

# ===========================
# Custom FedAvg strategy
# ===========================
class QFLFedAvg(fl.server.strategy.FedAvg):
    """Custom FedAvg strategy with explicit aggregation and metrics."""

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[BaseException],
    ) -> Optional[tuple[Parameters, Metrics]]:
        if not results:
            logger.warning(f"[Server] No results in round {server_round}")
            return None, {}

        total_examples: int = sum(fit_res.num_examples for _, fit_res in results)
        if total_examples == 0:
            return None, {}

        # Convert parameters and do weighted average
        weighted_params: list[list[np.ndarray]] = []
        losses: list[float] = []
        accs: list[float] = []

        for _, fit_res in results:
            weights: float = fit_res.num_examples / total_examples
            ndarrays: list[np.ndarray] = parameters_to_ndarrays(fit_res.parameters)
            weighted_params.append([w * weights for w in ndarrays])

            if "loss" in fit_res.metrics:
                losses.append(fit_res.metrics["loss"])
            if "val_acc" in fit_res.metrics:
                accs.append(fit_res.metrics["val_acc"])

        # Elementwise weighted average
        avg_params: list[np.ndarray] = [
            sum(p[i] for p in weighted_params) for i in range(len(weighted_params[0]))
        ]
        aggregated_parameters: Parameters = ndarrays_to_parameters(avg_params)

        avg_loss: float = float(sum(losses) / len(losses)) if losses else 0.0
        avg_acc: float = float(sum(accs) / len(accs)) if accs else 0.0

        g_round.set(server_round)
        g_avg_loss.set(avg_loss)
        g_avg_acc.set(avg_acc)

        logger.info(f"[Round {server_round}] Aggregated avg_loss={avg_loss:.4f}, avg_acc={avg_acc:.4f}")

        # Return (parameters, metrics)
        return aggregated_parameters, {"avg_loss": avg_loss, "avg_acc": avg_acc}

# ===========================
# Server entrypoint
# ===========================
def main() -> None:
    """Main entry point for the Flower server."""
    server_address: str = os.getenv("SERVER_ADDRESS", "0.0.0.0:8080")
    num_rounds: int = int(os.getenv("NUM_ROUNDS", "3"))

    start_http_server(PROM_PORT)

    strategy = QFLFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=4,
        min_available_clients=4,
        evaluate_fn=None,
    )

    logger.info(f"[Server] Starting QFL server at {server_address} for {num_rounds} rounds")
    logger.info(f"[Server] Prometheus port: {PROM_PORT}")

    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
