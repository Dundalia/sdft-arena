"""
Self-Distillation Training Script

This script trains a model using self-distillation with vLLM for efficient generation.
Supports both single-GPU and multi-GPU training via accelerate.

Usage:
    Single GPU (default config):
        python main.py

    Single GPU (specific config):
        python main.py --config-name qwen3_1.7b

    Override parameters:
        python main.py model.name=Qwen/Qwen3-4B training.learning_rate=1e-5

    Multi-GPU (4x A100):
        accelerate launch --num_processes 4 main.py --config-name qwen3_4b
"""

import hydra
from omegaconf import DictConfig, OmegaConf
from distil_trainer import DistilTrainer
from distil_config import DistilConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import Dataset
from string import Template
import os


def load_tooluse_dataset(seed=42) -> Dataset:
    """Load and prepare tooluse dataset with formatted prompts."""
    train_path = 'data/tooluse_data/train_data.json'
    test_path = 'data/tooluse_data/eval_data.json'
    train_dataset = Dataset.from_json(train_path)
    test_dataset = Dataset.from_json(test_path)

    def format_example(example):
        teacher_prompt = Template("""
$orig_content

This is an example for a response to the question:
$output_text

Now answer with a response of your own, including the thinking process.
""")

        return {
            "prompt": [{"role": "user", "content": example['prompt']}],
            "teacher_prompt": [{"role": "user", "content": teacher_prompt.substitute(
                orig_content=example['prompt'], 
                output_text='\n'.join(example['golden_response'])
            )}],
        }
    
    train_dataset = train_dataset.map(format_example, remove_columns=train_dataset.column_names)
    train_dataset = train_dataset.shuffle(seed=seed)
    return train_dataset, test_dataset


def get_world_size():
    """Get the number of processes in distributed training."""
    if os.environ.get("WORLD_SIZE"):
        return int(os.environ["WORLD_SIZE"])
    return 1


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    """Main training function with Hydra configuration."""
    
    # Print configuration
    print("=== Training Configuration ===")
    print(OmegaConf.to_yaml(cfg))
    print("==============================")
    
    world_size = get_world_size()
    print(f"World Size: {world_size}")
    
    # Get torch dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(cfg.model.torch_dtype, torch.bfloat16)
    
    # Load models
    print(f"Loading model: {cfg.model.name}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        torch_dtype=torch_dtype,
    )
    
    print(f"Loading teacher model: {cfg.model.name}")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        torch_dtype=torch_dtype,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    train_dataset, eval_dataset = load_tooluse_dataset(cfg.training.seed)
    
    print(f"Train dataset size: {len(train_dataset)}")

    # Configure training - use hydra's output directory
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    # Wandb run name - let the trainer handle initialization
    run_name = cfg.wandb.run_name if cfg.wandb.run_name else f"{cfg.model.name.split('/')[-1]}-distil"
    
    # Set wandb project via environment variable (trainer will pick it up)
    if cfg.wandb.project:
        os.environ["WANDB_PROJECT"] = cfg.wandb.project
    
    config = DistilConfig(
        seed=cfg.training.seed,
        output_dir=output_dir,
        run_name=run_name,  # For wandb logging
        
        # vLLM settings
        use_vllm=cfg.vllm.use_vllm,
        vllm_mode=cfg.vllm.mode,
        vllm_tensor_parallel_size=cfg.vllm.tensor_parallel_size, 
        vllm_gpu_memory_utilization=cfg.vllm.gpu_memory_utilization,
        vllm_enable_sleep_mode=cfg.vllm.enable_sleep_mode,
        vllm_importance_sampling_correction=cfg.vllm.importance_sampling_correction,
        
        # Training hyperparameters
        learning_rate=cfg.training.learning_rate,
        warmup_ratio=cfg.training.warmup_ratio,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        bf16=True,
        fp16=False,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.num_prompts_per_batch,
        max_grad_norm=cfg.training.max_grad_norm,
        num_train_epochs=cfg.training.num_train_epochs,
        
        # Sequence lengths
        max_prompt_length=cfg.sequence.max_prompt_length,
        max_completion_length=cfg.sequence.max_completion_length,
        
        # Reference model sync
        sync_ref_model=cfg.distillation.sync_ref_model,
        ref_model_sync_steps=cfg.distillation.ref_model_sync_steps,
        ref_model_mixup_alpha=cfg.distillation.ref_model_mixup_alpha,
        
        # Distillation settings
        num_loss_tokens_to_skip=cfg.distillation.num_loss_tokens_to_skip,
        
        # Logging
        logging_steps=cfg.logging.logging_steps,
        save_steps=cfg.logging.save_steps,
        log_completions=cfg.logging.log_completions,
        report_to=cfg.logging.report_to,
    )
    
    # Initialize trainer
    trainer = DistilTrainer(
        model=model,
        ref_model=teacher_model,
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save final model
    print(f"Saving final model to {output_dir}")
    trainer.save_model()
    print("Training complete!")


if __name__ == "__main__":
    main()
