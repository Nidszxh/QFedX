#!/bin/bash
#SBATCH --job-name=qfedx-server
#SBATCH --output=logs/server_%j.out
#SBATCH --error=logs/server_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --partition=main

# QFedX Flower Server for Slurm

set -e
echo "Starting QFedX Flower server at $(date)"
echo "Node: $(hostname)"

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
source "${PROJECT_DIR}/.quantum/bin/activate" 2>/dev/null || true

export SERVER_ADDRESS="0.0.0.0:8080"
export NUM_ROUNDS=10
export PROM_PORT=9102

cd "${PROJECT_DIR}"
echo "Server address: ${SERVER_ADDRESS}"
echo "NUM_ROUNDS: ${NUM_ROUNDS}"

# Write server address for clients
echo "${SERVER_ADDRESS}" > "${PROJECT_DIR}/logs/server_address.txt"

python src/fl_server.py

echo "Flower server finished at $(date)"
