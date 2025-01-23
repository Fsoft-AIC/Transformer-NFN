#!/bin/bash

# Check if required parameters are passed
if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <sweep_id> <num_agents_per_gpu> <gpu_list>"
  echo "Example: $0 project/entity/abc123 2 0,1"
  exit 1
fi

# Read parameters
SWEEP_ID=$1                # Sweep ID from wandb
NUM_AGENTS_PER_GPU=$2      # Number of agents to run per GPU
GPU_LIST=$3                # Comma-separated list of GPU IDs (e.g., "0,1")

# Convert the comma-separated GPU list into an array
IFS=',' read -r -a GPUS <<< "$GPU_LIST"

# Start agents on each specified GPU
for GPU in "${GPUS[@]}"; do
  for ((i=1; i<=NUM_AGENTS_PER_GPU; i++)); do
    echo "Starting wandb agent on GPU $GPU (Agent $i/$NUM_AGENTS_PER_GPU)"
    CUDA_VISIBLE_DEVICES=$GPU wandb agent $SWEEP_ID &
    sleep 1  # Small delay to avoid overwhelming the system
  done
done

echo "Started all agents. Monitoring the sweep..."
wait  # Wait for all agents to finish
