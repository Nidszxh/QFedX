#!/bin/bash
#SBATCH --job-name=qfedx-ray
#SBATCH --output=logs/ray_cluster_%j.out
#SBATCH --error=logs/ray_cluster_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# QFedX Ray Cluster Launcher for Slurm

set -e
echo "Starting QFedX Ray cluster on Slurm at $(date)"
echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# Load environment
source "${PROJECT_DIR}/.quantum/bin/activate" 2>/dev/null || \
    module load python/3.10 cuda/12.1 2>/dev/null || true

# Ray head node
RAY_HEAD_PORT=6379
RAY_HEAD_ADDRESS="$(hostname):${RAY_HEAD_PORT}"

echo "Starting Ray head node at ${RAY_HEAD_ADDRESS}"
ray start --head \
    --port=${RAY_HEAD_PORT} \
    --num-cpus=${SLURM_CPUS_PER_TASK} \
    --num-gpus=1 \
    --object-store-memory=10000000000 \
    --temp-dir="/tmp/ray_${SLURM_JOB_ID}"

echo "Ray head started. Address: ${RAY_HEAD_ADDRESS}"
echo "RAY_HEAD_ADDRESS=${RAY_HEAD_ADDRESS}" > "${LOG_DIR}/ray_head_${SLURM_JOB_ID}.env"

# Run the experiment
cd "${PROJECT_DIR}"
export RAY_HEAD_ADDRESS="${RAY_HEAD_ADDRESS}"

python src/run.py mode=experiment_grid max_exp=10

# Cleanup
echo "Shutting down Ray..."
ray stop
echo "QFedX Ray cluster completed at $(date)"
