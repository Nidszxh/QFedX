# QFedX Configuration Reference

## Config File

All parameters live in a single flat file: `config/config.yaml`.

Override any param on the command line:
```bash
python src/run.py mode=qfl n_qubits=6 noise_type=depolarizing dp_enabled=true
```

## QuantumNeuralNetworkConfig (`src/quantum/qnn.py`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `n_qubits` | int | 4 | Number of qubits |
| `n_layers` | int | 2 | Variational layers |
| `n_readout` | int | None | Readout qubits (defaults to n_qubits) |
| `encoding` | `EncodingType` | amplitude | `amplitude` or `angle` |
| `entanglement` | `EntanglementTopology` | circular | `linear`, `circular`, or `full` |
| `n_features` | int | 4 | Input feature dim |
| `n_classes` | int | 3 | Output classes |
| `batch_size` | int | 16 | Training batch size |
| `epochs` | int | 5 | Local epochs |
| `classical_lr` | float | 1e-3 | Learning rate for classical params |
| `quantum_lr` | float | 5e-4 | Learning rate for quantum params |
| `weight_decay` | float | 1e-4 | Weight decay |
| `grad_clip` | float | 1.0 | Gradient clipping norm |
| `device_name` | str | default.qubit | PennyLane device name |
| `diff_method` | `DiffMethod` | backprop | `backprop` or `parameter_shift` |
| `shots` | int | None | Number of shots (None for analytic) |
| `optimizer_type` | `OptimizerType` | adamw | `adamw`, `adam`, `sgd`, or `spsa` |
| `spsa_a` | float | 0.01 | SPSA a parameter |
| `spsa_c` | float | 0.1 | SPSA c parameter |
| `wrap_angles` | bool | True | Wrap quantum angles to [-pi, pi] |
| `noise_type` | `NoiseType` | none | `none`, `depolarizing`, `amplitude_damping`, `readout`, or `combined` |
| `depolarizing_p` | float | 0.001 | Depolarizing noise probability |
| `amplitude_gamma` | float | 0.001 | Amplitude damping gamma |
| `readout_flip_prob` | float | 0.01 | Readout flip probability |
| `dp_enabled` | bool | False | Enable differential privacy |
| `dp_clip_norm` | float | 1.0 | DP gradient clip norm |
| `dp_noise_multiplier` | float | 1.0 | DP noise multiplier |

## FederatedLearningConfig (`src/core/fl.py`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_rounds` | int | 5 | FL communication rounds |
| `local_epochs` | int | 3 | Epochs per client per round |
| `batch_size` | int | 16 | Local batch size |
| `client_fraction` | float | 0.75 | Fraction of clients per round |
| `classical_lr` | float | 1e-3 | Client classical learning rate |
| `quantum_lr` | float | 5e-4 | Client quantum learning rate |
| `grad_clip` | float | 1.0 | Gradient clipping norm |
| `dp_enabled` | bool | False | Enable DP |
| `dp_clip_norm` | float | 1.0 | DP clip norm |
| `dp_noise_multiplier` | float | 1.0 | DP noise multiplier |
| `dp_delta` | float | 1e-5 | DP delta |
| `secure_aggregation` | bool | False | Enable pairwise mask secure aggregation |
| `num_clients` | int | 4 | Number of clients |

## NoiseConfig (`src/quantum/noise.py`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `noise_type` | `NoiseType` | none | Noise model type |
| `depolarizing_p` | float | 0.001 | Depolarizing probability |
| `amplitude_gamma` | float | 0.001 | Amplitude damping rate |
| `readout_flip_prob` | float | 0.01 | Readout error flip probability |
| `shots` | int | None | Number of shots for finite-sampling noise |

## Enum Types (`src/core/defs.py`)

### EncodingType
- `amplitude` — Amplitude encoding (log2(n_features) qubits)
- `angle` — Angle encoding (n_features qubits)

### EntanglementTopology
- `linear` — Linear nearest-neighbor entanglement
- `circular` — Circular entanglement (first to last qubit)
- `full` — All-to-all entanglement

### OptimizerType
- `adam` — Adam optimizer
- `adamw` — AdamW optimizer with weight decay
- `sgd` — Stochastic gradient descent
- `spsa` — Simultaneous perturbation stochastic approximation

### DiffMethod
- `backprop` — Backpropagation (analytic, fast)
- `parameter_shift` — Parameter-shift rule (hardware-compatible)

### NoiseType
- `none` — No noise
- `depolarizing` — Depolarizing channel
- `amplitude_damping` — Amplitude damping channel
- `readout` — Readout error (confusion matrix)
- `combined` — All noise channels combined

### RunMode
- `qnn` — Quantum neural network training
- `qfl` — Quantum federated learning
- `centralized` — Centralized training
- `iris` — Iris dataset
- `verify_gradients` — Gradient verification
- `qfl_noise` — QFL with noise
- `qfl_dp` — QFL with differential privacy
- `experiment_grid` — Grid search
- `ablation` — Ablation study

### DeviceType
- `cpu` — CPU device
- `cuda` — CUDA GPU device
- `auto` — Auto-detect best available device

## Output Artifacts

- `./artifacts/quantum_federated_model.pt` — Full training checkpoint
- `./artifacts/qfl_metrics.csv` — Per-round metrics
- `./artifacts/pca_model.pkl` — Fitted PCA (if used)
- `./artifacts/scaler.pkl` — Fitted MinMaxScaler
- `./results/` — Experiment grid JSON results
- `./checkpoints/` — Round-level checkpoints
- `mlflow.db` — MLflow tracking (SQLite)
- `./results/qfl/` — QFL visualization plots
- `./results/cfl/` — CFL visualization plots
- `./results/preprocessing/` — Preprocessing visualization plots
- `./visualizations/qnn/` — QNN visualization plots
- `./visualizations/comparative/` — Comparative analysis plots
