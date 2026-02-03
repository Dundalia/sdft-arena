#!/bin/bash
# Single GPU training script for interactive testing
# Usage: 
#   bash scripts/train_single_gpu.sh
#   CONFIG=qwen3_1.7b bash scripts/train_single_gpu.sh
#   CONFIG=qwen3_1.7b OVERRIDES="training.learning_rate=1e-5" bash scripts/train_single_gpu.sh

set -e

# Setup environment
module load anaconda/3 cudatoolkit/12.6.0
conda activate $SCRATCH/arena-capstone/venv

export HF_HOME=$SCRATCH
export WANDB_MODE=offline
export OMP_NUM_THREADS=4

cd $SCRATCH/arena-capstone/Self-Distillation

# Configuration
CONFIG_NAME="${CONFIG:-qwen3_1.7b}"
OVERRIDES="${OVERRIDES:-}"

echo "=== Training Configuration ==="
echo "Config: $CONFIG_NAME"
echo "Overrides: $OVERRIDES"
echo "=============================="

python main.py --config-name $CONFIG_NAME $OVERRIDES
