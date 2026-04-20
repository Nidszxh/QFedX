"""Type aliases, constants, and enum types for the codebase."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

# ── Type aliases ──────────────────────────────────────────────────────────

EntanglementPairs = list[tuple[int, int]]


# ── Paths ─────────────────────────────────────────────────────────────────

DEFAULT_VISUALIZATION_ROOT = Path("./visualizations")
DEFAULT_DATASET_RAW = Path("./dataset/raw")
DEFAULT_DATASET_PROCESSED = Path("./dataset/processed")
DEFAULT_MODEL_SAVE_PATH = Path("./dataset/processed/qNN_pennylane_model.pt")
DEFAULT_ARTIFACTS_DIR = Path("./artifacts")

# MLflow
DEFAULT_MLFLOW_URI = "sqlite:///mlflow.db"

# Seeds
DEFAULT_SEED = 42

# Numerical stability
DIVISION_EPSILON = 1e-8

# Prometheus
DEFAULT_SERVER_PROM_PORT = 9102
DEFAULT_CLIENT_PROM_PORT = 9103


# ── Enums ─────────────────────────────────────────────────────────────────

class EncodingType(str, Enum):
    AMPLITUDE = "amplitude"
    ANGLE = "angle"


class EntanglementTopology(str, Enum):
    LINEAR = "linear"
    CIRCULAR = "circular"
    FULL = "full"


class OptimizerType(str, Enum):
    ADAMW = "adamw"
    SPSA = "spsa"


class DiffMethod(str, Enum):
    BACKPROP = "backprop"
    PARAMETER_SHIFT = "parameter-shift"
    BEST = "best"


class NoiseType(str, Enum):
    NONE = "none"
    DEPOLARIZING = "depolarizing"
    AMPLITUDE_DAMPING = "amplitude_damping"
    READOUT = "readout"
    COMBINED = "combined"


class RunMode(str, Enum):
    QNN = "qnn"
    QFL = "qfl"
    CENTRALIZED = "centralized"
    IRIS = "iris"
    GRID = "grid"


class DeviceType(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
