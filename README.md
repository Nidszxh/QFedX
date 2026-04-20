# QFedX

A research prototype on **Privacy-Preserving Quantum Federated Learning (QFL)** using **Variational Quantum Circuits (VQCs)**, simulated on **HPC resources**.

Built with **PennyLane** (quantum computing), **PyTorch** (classical deep learning), and **Flower** (federated learning).

## Project Structure

```
QFedX/
├── src/
│   ├── run.py                   # Main entry point (Hydra)
│   ├── fl_client.py             # Flower FL client
│   ├── fl_server.py             # Flower FL server
│   ├── core/                    # Core utilities and shared components
│   │   ├── defs.py              # Type aliases, constants, enums
│   │   ├── utils.py             # Logging + reproducibility
│   │   ├── data.py              # DataLoader factory + evaluation
│   │   ├── fl.py                # Federated averaging + FL config
│   │   ├── quantum.py           # Encoding + entanglement helpers
│   │   └── plot_utils.py        # Shared visualization helpers
│   ├── quantum/                 # Quantum ML components
│   │   ├── qnn.py               # Quantum Neural Network
│   │   ├── qfl.py               # Quantum Federated Learning
│   │   ├── noise.py             # Noise models for quantum hardware
│   │   ├── plots_qfl.py         # QFL visualization
│   │   ├── plots_qnn.py         # QNN visualization
│   │   ├── plots_comparative_analysis.py  # Comparative analysis plots
│   │   └── privacy/             # Differential privacy & secure aggregation
│   │       ├── differential_privacy.py
│   │       ├── privacy_accountant.py
│   │       └── secure_aggregation.py
│   ├── classical/               # Classical FL baseline
│   │   ├── cfl.py               # Classical Federated Learning
│   │   └── plots_cfl.py         # CFL visualization
│   └── data/                    # Data loading and preprocessing
│       ├── preprocess.py        # MNIST preprocessing pipeline
│       └── plots_preprocess.py  # Data visualization
├── config/
│   └── config.yaml              # Single flat Hydra configuration
├── tests/                       # Test suite
│   ├── conftest.py
│   ├── test_qnn.py
│   ├── test_qfl.py
│   ├── test_noise.py
│   ├── test_privacy.py
│   └── test_preprocess.py
├── slurm/                       # HPC/SLURM scripts
│   ├── hpc_experiment.sh
│   ├── ray_cluster.sh
│   ├── array_clients.sh
│   └── fl_server.sh
├── docs/                        # Documentation
│   ├── quickstart.md
│   ├── config_reference.md
│   └── flat_config_cli.md
├── pyproject.toml               # Package configuration
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Container definition
├── docker-compose.yml           # Multi-container orchestration
└── LICENSE                      # Apache 2.0
```

## Quick Start

### Installation

```bash
git clone <repo> && cd QFedX
python -m venv .quantum && source .quantum/bin/activate

# Install in editable mode (recommended for development)
pip install -e .

# Or install dependencies only
pip install -r requirements.txt
```

### Preprocess Data

```bash
python src/data/preprocess.py
```

Downloads MNIST from `dataset/raw/`, filters digits (0,1,2), applies PCA, and saves client partitions.

### Running Experiments

```bash
# Run with Hydra configuration
python src/run.py mode=qfl

# Run specific modes
python src/run.py mode=qnn          # QNN training only
python src/run.py mode=centralized  # Centralized (non-federated) training
python src/run.py mode=iris         # Iris dataset verification
python src/run.py mode=verify_gradients  # Gradient verification
python src/run.py mode=qfl_noise    # QFL with noise
python src/run.py mode=qfl_dp       # QFL with differential privacy
python src/run.py mode=experiment_grid  # Grid search
python src/run.py mode=ablation     # Ablation study
```

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Run server (distributed mode)
docker-compose run server

# Run clients
docker-compose run client1
docker-compose run client2
```

### HPC (SLURM)

```bash
sbatch slurm/hpc_experiment.sh
```

## Development

### Code Quality

```bash
# Run tests
PYTHONPATH=src python -m pytest tests/ -v

# Linting
ruff check src/

# Type checking (if configured)
mypy src/
```

### Project Conventions

- **Logging**: `from core.utils import get_logger`
- **Seeds**: `from core.utils import set_seed`
- **DataLoaders**: `from core.data import create_dataloader`
- **Types/Enums**: `from core.defs import EncodingType, NoiseType, ...`
- **Type hints**: All public functions must have complete type annotations
- **Config**: Single flat Hydra config (`config/config.yaml`)

## Configuration

The project uses Hydra with a single flat configuration file: `config/config.yaml`.

All parameters — QNN, FL, data, noise, privacy, HPC — are at the top level. Override any param on the command line:

```bash
# Training config
python src/run.py mode=qfl n_qubits=6 n_layers=4 noise_type=depolarizing

# Privacy config
python src/run.py mode=qfl_dp dp_enabled=true dp_clip_norm=1.0 dp_noise_multiplier=1.0

# Federated config
python src/run.py mode=qfl secure_aggregation=true client_fraction=0.5

# Data config
python src/run.py mode=qfl partition_type=non_iid alpha=0.5 digits=[0,1,2,3]
```

## License

Apache 2.0
