#!/bin/bash
#SBATCH --job-name=qfedx-hpc
#SBATCH --output=logs/hpc_%j.out
#SBATCH --error=logs/hpc_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# QFedX Full HPC Experiment Script

set -e
echo "========================================"
echo "QFedX HPC Experiment"
echo "Started: $(date)"
echo "Host: $(hostname)"
echo "Job: ${SLURM_JOB_ID}"
echo "========================================"

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
cd "${PROJECT_DIR}"

# Activate environment
if [ -f ".quantum/bin/activate" ]; then
    source .quantum/bin/activate
fi

# Set HPC parameters
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export CUDA_VISIBLE_DEVICES=0
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH}"

mkdir -p logs checkpoints results

echo "Environment:"
echo "  PYTHONPATH: ${PYTHONPATH}"
echo "  OMP_NUM_THREADS: ${OMP_NUM_THREADS}"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "  MLFLOW_TRACKING_URI: ${MLFLOW_TRACKING_URI}"
nvidia-smi 2>/dev/null || echo "No GPU detected"

echo ""
echo "Phase 1: Single QNN Verification (Iris)"
python src/run.py mode=iris n_qubits=4 n_layers=2 encoding=angle

echo ""
echo "Phase 2: Quantum FL (QFL)"
python src/run.py mode=qfl num_rounds=5 local_epochs=3 n_qubits=4 n_layers=2

echo ""
echo "Phase 3: QFL with Noise"
python src/run.py mode=qfl noise_type=depolarizing depolarizing_p=0.001 num_rounds=5

echo ""
echo "Phase 3: QFL with DP"
python src/run.py mode=qfl dp_enabled=true dp_clip_norm=1.0 dp_noise_multiplier=1.0 num_rounds=5

echo ""
echo "Phase 3: QFL with DP + Secure Aggregation"
python src/run.py mode=qfl dp_enabled=true secure_aggregation=true num_rounds=5

echo ""
echo "Phase 4: QFL with Noise + DP combined"
python src/run.py mode=qfl noise_type=depolarizing depolarizing_p=0.001 dp_enabled=true num_rounds=5

echo ""
echo "Phase 5: Scaling to more qubits/layers"
python src/run.py mode=qfl n_qubits=8 n_layers=3 num_rounds=5

echo ""
echo "Phase 6: Ablation Study"
python src/run.py mode=ablation

echo ""
echo "Phase 6: Experiment Grid (subset)"
python src/run.py mode=experiment_grid max_exp=5

echo ""
echo "========================================"
echo "QFedX HPC Experiment Complete"
echo "Finished: $(date)"
echo "========================================"
