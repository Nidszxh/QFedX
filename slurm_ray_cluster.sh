#!/bin/bash
#SBATCH --job-name=qfedx_ray_cluster
#SBATCH --nodes=3
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/qfedx_%j.out
#SBATCH --error=logs/qfedx_%j.err

# Load modules (Adjust based on HPC environment)
# module load python/3.10
# module load cuda/11.8

# Optionally use Singularity
# SINGULARITY_IMG="qfedx_env.sif"
# COMMAND_PREFIX="singularity exec --nv $SINGULARITY_IMG"
COMMAND_PREFIX="" # Run natively if in a standard environment array

# Get nodes list
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
fi

port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    $COMMAND_PREFIX ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus 1 --block &

# optional, wait for head node to start
sleep 10

worker_num=$((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
    node=${nodes_array[$i]}
    echo "Starting WORKER $i at $node"
    srun --nodes=1 --ntasks=1 -w "$node" \
        $COMMAND_PREFIX ray start --address "$ip_head" \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus 1 --block &
done

# Wait for workers to connect
sleep 10

echo "Ray cluster is up and running!"
echo "Executing QFedX Experiment Script..."

# Execute the experiment entrypoint
$COMMAND_PREFIX python run_experiments.py

echo "Job completed."
exit 0
