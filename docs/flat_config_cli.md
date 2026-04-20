# Flat Config Override Reference

All config keys in `config/config.yaml` can be overridden via Hydra's CLI syntax.
Because the config is flat, simply use `++key=value`:

```bash
python src/run.py ++n_qubits=6 ++num_rounds=10 ++noise_type=depolarizing
```

## Mapping to Code Structures

| CLI Key | Used In | Field |
|---|---|---|
| `seed` | `core/utils.py:set_seed` | — |
| `device` | `run.py:_resolve_device`, `cfl.py:_get_device` | — |
| `mode` | `run.py:_MODE_DISPATCH` | selects qnn/qfl/centralized/iris/... |
| `experiment_name` | MLflow run name | — |
| `mlflow_tracking_uri` | MLflow tracking URI | — |
| `max_exp` | limit on experiment grid trials | — |
| `n_qubits` | `QuantumNeuralNetworkConfig` | `n_qubits` |
| `n_layers` | `QuantumNeuralNetworkConfig` | `n_layers` |
| `n_readout` | `QuantumNeuralNetworkConfig` | `n_readout` |
| `encoding` | `QuantumNeuralNetworkConfig` | `encoding` |
| `entanglement` | `QuantumNeuralNetworkConfig` | `entanglement` |
| `n_features` | `QuantumNeuralNetworkConfig` | `n_features` |
| `n_classes` | `QuantumNeuralNetworkConfig` | `n_classes` |
| `batch_size` | `QuantumNeuralNetworkConfig` | `batch_size` |
| `epochs` | `QuantumNeuralNetworkConfig` | `epochs` |
| `classical_lr` | `QuantumNeuralNetworkConfig` | `classical_lr` |
| `quantum_lr` | `QuantumNeuralNetworkConfig` | `quantum_lr` |
| `weight_decay` | `QuantumNeuralNetworkConfig` | `weight_decay` |
| `grad_clip` | `QuantumNeuralNetworkConfig` | `grad_clip` |
| `device_name` | `QuantumNeuralNetworkConfig` | `device_name` |
| `diff_method` | `QuantumNeuralNetworkConfig` | `diff_method` |
| `shots` | `QuantumNeuralNetworkConfig` | `shots` |
| `optimizer_type` | `QuantumNeuralNetworkConfig` | `optimizer_type` |
| `spsa_a` | `QuantumNeuralNetworkConfig` | `spsa_a` |
| `spsa_c` | `QuantumNeuralNetworkConfig` | `spsa_c` |
| `wrap_angles` | `QuantumNeuralNetworkConfig` | `wrap_angles` |
| `noise_type` | `QuantumNeuralNetworkConfig` | `noise_type` |
| `depolarizing_p` | `QuantumNeuralNetworkConfig` | `depolarizing_p` |
| `amplitude_gamma` | `QuantumNeuralNetworkConfig` | `amplitude_gamma` |
| `readout_flip_prob` | `QuantumNeuralNetworkConfig` | `readout_flip_prob` |
| `num_rounds` | `FederatedLearningConfig` | `num_rounds` |
| `local_epochs` | `FederatedLearningConfig` | `local_epochs` |
| `client_fraction` | `FederatedLearningConfig` | `client_fraction` |
| `raw_folder` | `preprocess_mnist` | `raw_folder` |
| `processed_folder` | `preprocess_mnist` | `processed_folder` |
| `digits` | `preprocess_mnist` | `digits` (yaml list) |
| `val_split` | `preprocess_mnist` | `val_split` |
| `num_clients` | `preprocess_mnist`, `FederatedLearningConfig` | `num_clients` |
| `partition_type` | `preprocess_mnist` | `partition_type` |
| `alpha` | `preprocess_mnist` | `alpha` |
| `apply_pca` | `preprocess_mnist` | `apply_pca` |
| `pca_components` | `preprocess_mnist` | `pca_components` |
| `generate_plots` | `preprocess_mnist` | toggles visualization |
| `early_stop_patience` | `FederatedLearningConfig` | `early_stop_patience` |
| `early_stop_delta` | `FederatedLearningConfig` | `early_stop_delta` |
| `checkpoint_dir` | `FederatedLearningConfig` | `checkpoint_dir` |
| `checkpoint_interval` | `FederatedLearningConfig` | `checkpoint_interval` |
| `dp_enabled` | `FederatedLearningConfig` | `dp_enabled` |
| `dp_clip_norm` | `FederatedLearningConfig` | `dp_clip_norm` |
| `dp_noise_multiplier` | `FederatedLearningConfig` | `dp_noise_multiplier` |
| `dp_delta` | `FederatedLearningConfig` | `dp_delta` |
| `secure_aggregation` | `FederatedLearningConfig` | `secure_aggregation` |

## Examples

```bash
# Quantum training with custom hyperparams
python src/run.py mode=qnn n_qubits=6 n_layers=4 epochs=20

# Federated learning with DP
python src/run.py mode=qfl dp_enabled=true dp_noise_multiplier=1.5

# Mixed precision via shots
python src/run.py mode=qnn shots=1000 device_name=lightning.qubit

# Non-IID partition
python src/run.py mode=qfl partition_type=non_iid alpha=0.5
```

## Notes

- Use `++` for strict overrides (adds keys that don't exist in the config).
- Use `+` for soft overrides (only changes existing values, does not add new ones).
- YAML lists (e.g., `digits`) are overridden as `++digits=[0,1,2,3]`.
- Boolean values use `true`/`false` (lowercase) per YAML convention.
