# QFedX — Quick Start

## Install

```bash
git clone <repo> && cd QFedX
python -m venv .quantum && source .quantum/bin/activate

# Install in editable mode (recommended)
pip install -e .

# Or install dependencies only
pip install -r requirements.txt
```

## Preprocess Data

```bash
python src/data/preprocess.py
```

Downloads MNIST (from `dataset/raw/`), filters digits, applies PCA, and saves client partitions to `dataset/processed/`.

## Train (Single Run)

```bash
# Run with Hydra configuration
python src/run.py mode=qfl

# QNN only
python src/run.py mode=qnn

# Centralized training
python src/run.py mode=centralized

# Iris dataset
python src/run.py mode=iris
```

### With Noise

```bash
python src/run.py mode=qfl noise_type=depolarizing depolarizing_p=0.01
```

### With Differential Privacy

```bash
python src/run.py mode=qfl dp_enabled=true dp_clip_norm=1.0 dp_noise_multiplier=1.0
```

### With Secure Aggregation

```bash
python src/run.py mode=qfl secure_aggregation=true
```

## Experiment Grid

```bash
# Run experiment grid
python src/run.py mode=experiment_grid max_exp=10

# Run ablation study
python src/run.py mode=ablation
```

## Run Tests

```bash
PYTHONPATH=src python -m pytest tests/ -v
```

## HPC (Slurm)

```bash
sbatch slurm/hpc_experiment.sh
```

## Docker

```bash
# Build
docker-compose build

# Run server
docker-compose run server

# Run clients
docker-compose run client1
docker-compose run client2
```
