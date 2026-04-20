# Privacy-Preserving Quantum Federated Learning with Variational Circuits — An HPC-Based Simulation Framework

### Problem Statement:
Build a scalable simulation framework that trains Variational Quantum Circuits (VQCs) in a Federated Learning (FL) setup with basic client privacy. Use HPC to emulate multiple clients, different qubit counts, and simple noise models, and evaluate performance (wall-clock time, communication, GPU usage) and privacy–utility tradeoffs. The project compares centralized training, QFL without privacy, and QFL with basic DP + secure aggregation across datasets, qubit counts, and noise settings.

## Implementation Status

| Phase | Description | Status |
|-------|-------------|--------|
| **1** | Local prototype & VQC fundamentals | ✅ Complete |
| **2** | Federated loop prototype (FedAvg) | ✅ Complete |
| **3** | Privacy primitives: DP + Secure Aggregation | ✅ Complete |
| **4** | Quantum realism & error models | ✅ Complete |
| **5** | HPC integration & scalability | ✅ Complete |
| **6** | Experiments, ablations & analysis | ✅ Complete |
| **7** | Code quality & modernization | ✅ Complete |
| **8** | Configuration & CI fixes | ✅ Complete |
| **9** | Documentation & final polish | ✅ Complete |

---

## Phase 1 — Local prototype & VQC fundamentals (Completed)

- Single-client VQC training loop on Iris and PCA-MNIST
- Angle encoding + Amplitude encoding
- PennyLane `default.qubit` with backprop and parameter-shift differentiation
- SPSA gradient-free optimizer
- Gradient verification (backprop vs parameter-shift)

## Phase 2 — Federated loop prototype (Completed)

- FedAvg with sample-weighted aggregation
- Angle wrapping to [-pi, pi] for quantum parameters
- Flower-based distributed FL (fl_server.py, fl_client.py)
- Docker Compose orchestration
- Custom simulation via qfl.py
- Client sampling with configurable fraction

## Phase 3 — Privacy primitives: DP + Secure Aggregation (Completed)

- **`src/quantum/privacy/differential_privacy.py`**: Per-client gradient clipping (L2 norm) + Gaussian noise addition
- **`src/quantum/privacy/privacy_accountant.py`**: Gaussian DP accountant tracking cumulative epsilon
- **`src/quantum/privacy/secure_aggregation.py`**: Pairwise mask-based secure aggregation
- Integration into QFL training loop (fl_config: dp_enabled, secure_aggregation)
- Privacy epsilon logged to MLflow per round

## Phase 4 — Quantum realism & error models (Completed)

- **`src/quantum/noise.py`**: Depolarizing, Amplitude Damping, Readout Error, Combined (includes `NoisyQuantumCircuit`)
- Noise toggles in Hydra config (`config/config.yaml`, flat keys)
- Readout error applied at logits level via confusion matrix
- Parameter-shift differentiation required when noise is active
- Shots support for finite-sampling noise

## Phase 5 — HPC integration & scalability (Completed)

- **`slurm/ray_cluster.sh`**: Ray head node launcher with GPU support
- **`slurm/array_clients.sh`**: Slurm array job for parallel Flower clients
- **`slurm/fl_server.sh`**: Flower server on Slurm
- **`slurm/hpc_experiment.sh`**: Full HPC experiment pipeline
- GPU backend support (lightning.qubit, lightning.gpu)
- HPC config keys in `config/config.yaml`
- docker-compose.yml updated for scalable client count

## Phase 6 — Experiments, ablations & analysis (Completed)

- **`src/run.py` (mode=experiment_grid / mode=ablation)**: Full experiment grid runner
  - Grid mode: Cartesian product of qubits, layers, noise, DP, partition type
  - Ablation mode: 10 predefined configurations comparing noise, DP, qubits, layers
- MLflow tracking for all metrics
- JSON results export with summary statistics
- Comparative analysis visualizations (quantum vs classical)

## Phase 7 — Code quality & modernization (Completed)

- **Consolidated utilities** in `src/core/`:
  - `utils.py`: `set_seed()`, `get_logger()`, and `setup_logging()`
  - `defs.py`: Type aliases, constants, and enums (`EncodingType`, `NoiseType`, etc.)
  - `data.py`: `create_dataloader()` factory + `evaluate_model()`
  - `fl.py`: `FederatedLearningConfig` dataclass + `federated_averaging()`
  - `quantum.py`: Encoding + entanglement helpers
  - `plot_utils.py`: Shared `setup_plot_style()` + `save_figure()`
- **File renaming**: All files follow `snake_case.py` convention
- **Module docstrings**: Added to all source files
- **Pathlib**: Consistent use of `pathlib.Path` instead of `os.path`
- **Context managers**: Proper resource cleanup (`SummaryWriter`, etc.)
- **`__repr__` methods**: Added to `NoiseConfig`, `SPSA`

## Phase 8 — Configuration & CI fixes (Completed)

- **CI pipeline**: Updated all paths to match current module structure
- **Dockerfile**: Fixed entrypoint, proper `ENTRYPOINT`/`CMD` pattern
- **docker-compose.yml**: Build context instead of hardcoded SHA tags, GPU support
- **`pyproject.toml`**: Proper package configuration with `[tool.setuptools.packages.find]`
- **Flat config**: Consolidated all Hydra config groups into single `config/config.yaml`
- **`sys.path` hacks**: Removed from production code; only `conftest.py` injects for test isolation

## Phase 9 — Documentation & final polish (Completed)

- **README.md**: Full project structure, quick start, development guide, configuration reference
- **ROADMAP.md**: Updated with new file paths and completed phases
- **docs/quickstart.md**: Updated with Hydra commands and new paths
- **docs/config_reference.md**: Updated with enum types and new class names
- **docs/flat_config_cli.md**: CLI override reference with full key mapping

## Detailed Technical Notes

### VQC forward / measurement
- Feature mapping: RY(alpha * x_i) per feature for angle; AmplitudeEmbedding for amplitude
- Ansatz block: RY(theta), RZ(phi) per qubit, CNOT entanglers (linear/circular/full)
- Readout: <Z> expectation -> linear classifier -> logits

### Parameter-shift gradient
d<O>/dtheta = 0.5 * (<O>_{theta + pi/2} - <O>_{theta - pi/2})

### Secure aggregation (pairwise mask)
- Clients i<j generate deterministic mask m_{i,j}
- Client i adds +m_{i,j}, client j adds -m_{i,j}
- Sum of masked updates = sum of unmasked updates

### DP accountant
- Clip + Gaussian noise, track epsilon with Gaussian DP formula
- epsilon = sqrt(2 * ln(1.25/delta)) * sqrt(steps) / sigma
