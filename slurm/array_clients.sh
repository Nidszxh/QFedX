#!/bin/bash
#SBATCH --job-name=qfedx-clients
#SBATCH --output=logs/client_%A_%a.out
#SBATCH --error=logs/client_%A_%a.err
#SBATCH --array=1-4
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# QFedX Slurm Array Client Launcher

set -e
echo "Starting QFedX client ${SLURM_ARRAY_TASK_ID} at $(date)"
echo "Node: $(hostname)"

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
source "${PROJECT_DIR}/.quantum/bin/activate" 2>/dev/null || true

export CLIENT_ID="client_${SLURM_ARRAY_TASK_ID}"
export SERVER_ADDRESS="${SERVER_ADDRESS:-localhost:8080}"
export LOCAL_SAMPLES=500
export LOCAL_EPOCHS=3
export BATCH_SIZE=16

cd "${PROJECT_DIR}"
python src/fl_client.py

echo "Client ${SLURM_ARRAY_TASK_ID} finished at $(date)"
