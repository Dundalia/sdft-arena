# Self-Distillation Training Guide

## Quick Start

### Interactive Session (1 A100)

On your current interactive session with 1 A100, run:

```bash
cd $SCRATCH/arena-capstone/Self-Distillation

# Load environment
module load anaconda/3 cudatoolkit/12.6.0
conda activate $SCRATCH/arena-capstone/venv
export HF_HOME=$SCRATCH
export WANDB_MODE=offline

# Train with Qwen3-1.7B
python main.py \
  --model_name Qwen/Qwen3-1.7B \
  --output_dir outputs/qwen3-1.7b-distil \
  --num_prompts_per_batch 16 \
  --num_train_epochs 1
```

### SBATCH (4x A100)

Submit a multi-GPU training job:

```bash
cd $SCRATCH/arena-capstone
sbatch scripts/train_multi_gpu.sbatch
```

Or with custom model:
```bash
MODEL=Qwen/Qwen3-4B sbatch scripts/train_multi_gpu.sbatch
```

## Model Recommendations

Based on your hardware (4x A100 80GB):

| Model | GPUs | Config Notes |
|-------|------|--------------|
| Qwen/Qwen3-1.7B | 1 | Good for testing, fits easily |
| Qwen/Qwen3-4B | 1-4 | Recommended for experiments |
| Qwen/Qwen3-8B | 4 | May need `--vllm_gpu_memory_utilization 0.25` |

## Key Configuration Options

```bash
python main.py \
  --model_name Qwen/Qwen3-1.7B \    # Model to train
  --output_dir outputs/exp1 \        # Output directory
  --learning_rate 2e-5 \             # Learning rate
  --num_train_epochs 1 \             # Number of epochs
  --num_prompts_per_batch 32 \       # Gradient accumulation
  --max_prompt_length 1024 \         # Max prompt tokens
  --max_completion_length 1024 \     # Max completion tokens
  --vllm_gpu_memory_utilization 0.3  # vLLM memory (lower = more for training)
```

## Troubleshooting

### CUDA_HOME Error
If you get `MissingCUDAException: CUDA_HOME does not exist`:
```bash
module load cudatoolkit/12.6.0
```

### Out of Memory
Try these options:
- Reduce `--vllm_gpu_memory_utilization` (e.g., 0.2)
- Reduce `--max_completion_length` (e.g., 512)
- Reduce `--num_prompts_per_batch` (e.g., 8)
- Use a smaller model

### vLLM Version Warning
The TRL warning about vLLM version can be safely ignored - vLLM 0.12.0 works fine.

## Environment Setup (First Time Only)

The conda environment should already be set up at `$SCRATCH/arena-capstone/venv`.

If you need to recreate it:
```bash
module load anaconda/3
conda create -p $SCRATCH/arena-capstone/venv python=3.11 -y
conda activate $SCRATCH/arena-capstone/venv
pip install -r Self-Distillation/requirements.txt
```
