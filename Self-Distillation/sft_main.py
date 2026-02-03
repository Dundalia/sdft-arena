"""
SFT (Supervised Fine-Tuning) Training Script - Baseline

This script trains a model using standard supervised fine-tuning with TRL's SFTTrainer.
This serves as a baseline to compare against self-distillation approaches.
Supports both single-GPU and multi-GPU training via accelerate.

Usage:
    Single GPU (default config):
        python sft_main.py

    Single GPU (specific config):
        python sft_main.py --config-name sft_qwen3_1.7b

    Override parameters:
        python sft_main.py model.name=Qwen/Qwen3-4B training.learning_rate=1e-5

    Multi-GPU (4x A100):
        accelerate launch --num_processes 4 sft_main.py --config-name sft_qwen3_4b
"""

import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
from sft_trainer import SFTTrainer
from sft_config import SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import Dataset
import os
from peft import LoraConfig


def load_tooluse_dataset(seed: int = 42) -> tuple[Dataset, Dataset]:
    """
    Load and prepare tooluse dataset formatted for SFT training.
    
    The dataset is formatted as chat messages with:
    - User message: the original prompt
    - Assistant message: the golden response
    
    This is the standard format expected by TRL's SFTTrainer.
    
    Note: train_data has 'golden_response' field with text responses.
          eval_data has 'golden_answer' field with structured action dicts.
          For SFT eval, we only use train prompts (eval will be done separately).
    """
    train_path = 'data/tooluse_data/train_data.json'
    test_path = 'data/tooluse_data/eval_data.json'
    
    train_dataset = Dataset.from_json(train_path)
    test_dataset = Dataset.from_json(test_path)

    def format_train_example(example):
        """Format training example as chat messages for SFT training."""
        # golden_response is a list, join if multiple elements
        golden_response = example['golden_response']
        if isinstance(golden_response, list):
            response_text = '\n'.join(golden_response)
        else:
            response_text = golden_response
            
        return {
            "messages": [
                {"role": "user", "content": example['prompt']},
                {"role": "assistant", "content": response_text},
            ],
        }
    
    def format_eval_example(example):
        """Format eval example as chat messages for SFT evaluation.
        
        Eval data has 'golden_answer' as structured action dicts.
        We convert them to a string format for loss computation during training.
        """
        golden_answer = example['golden_answer']
        if isinstance(golden_answer, list):
            # Convert action dicts to string representation
            response_parts = []
            for action in golden_answer:
                if isinstance(action, dict):
                    action_str = f"Action: {action.get('Action', '')}\nAction Input: {action.get('Action_Input', '')}"
                    response_parts.append(action_str)
                else:
                    response_parts.append(str(action))
            response_text = '\n'.join(response_parts)
        else:
            response_text = str(golden_answer)
            
        return {
            "messages": [
                {"role": "user", "content": example['prompt']},
                {"role": "assistant", "content": response_text},
            ],
        }
    
    train_dataset = train_dataset.map(
        format_train_example, 
        remove_columns=train_dataset.column_names
    )
    train_dataset = train_dataset.shuffle(seed=seed)
    
    test_dataset = test_dataset.map(
        format_eval_example,
        remove_columns=test_dataset.column_names
    )
    
    return train_dataset, test_dataset


def get_world_size() -> int:
    """Get the number of processes in distributed training."""
    if os.environ.get("WORLD_SIZE"):
        return int(os.environ["WORLD_SIZE"])
    return 1


@hydra.main(version_base=None, config_path="config", config_name="sft_config")
def main(cfg: DictConfig):
    """Main SFT training function with Hydra configuration."""
    
    # Print configuration
    print("=== SFT Training Configuration ===")
    print(OmegaConf.to_yaml(cfg))
    print("==================================")
    
    world_size = get_world_size()
    print(f"World Size: {world_size}")
    
    # Get torch dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(cfg.model.torch_dtype, torch.bfloat16)
    
    # Load model
    print(f"Loading model: {cfg.model.name}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2" if cfg.model.get("use_flash_attention", False) else None,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    
    # Ensure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    train_dataset, eval_dataset = load_tooluse_dataset(cfg.training.seed)
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")

    # Configure training - use hydra's output directory
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    # Wandb run name
    run_name = cfg.wandb.run_name if cfg.wandb.run_name else f"{cfg.model.name.split('/')[-1]}-sft"
    
    # Set wandb project via environment variable
    if cfg.wandb.project:
        os.environ["WANDB_PROJECT"] = cfg.wandb.project
    
    # Create SFT config
    config = SFTConfig(
        # Output and logging
        output_dir=output_dir,
        run_name=run_name,
        
        # Training hyperparameters
        learning_rate=cfg.training.learning_rate,
        warmup_ratio=cfg.training.warmup_ratio,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        max_grad_norm=cfg.training.max_grad_norm,
        num_train_epochs=cfg.training.num_train_epochs,
        seed=cfg.training.seed,
        
        # Precision
        bf16=cfg.training.get("bf16", True),
        fp16=cfg.training.get("fp16", False),
        
        # Sequence settings
        max_seq_length=cfg.sequence.max_seq_length,
        packing=cfg.sequence.get("packing", False),
        
        # Memory optimization
        gradient_checkpointing=cfg.training.get("gradient_checkpointing", True),
        
        # Logging
        logging_steps=cfg.logging.logging_steps,
        save_steps=cfg.logging.save_steps,
        save_total_limit=cfg.logging.get("save_total_limit", 3),
        report_to=cfg.logging.report_to,
        
        # Evaluation
        eval_strategy=cfg.logging.get("eval_strategy", "steps"),
        eval_steps=cfg.logging.get("eval_steps", cfg.logging.save_steps),
        
        # Dataset
        dataset_text_field=None,  # We use messages format
    )
    
    # Setup LoRA configuration if specified
    peft_config = None
    if cfg.model.get("lora") is not None:
        lora_cfg = cfg.model.lora
        # Convert ListConfig to list if needed
        target_modules = lora_cfg.target_modules
        if isinstance(target_modules, ListConfig):
            target_modules = list(target_modules)
        
        peft_config = LoraConfig(
            r=lora_cfg.r,
            lora_alpha=lora_cfg.lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_cfg.lora_dropout,
            bias=lora_cfg.bias,
            task_type=lora_cfg.task_type,
        )
        print(f"Using LoRA with r={lora_cfg.r}, alpha={lora_cfg.lora_alpha}, target_modules={target_modules}")
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    
    # Train
    print("Starting SFT training...")
    trainer.train()
    
    # Save final model
    print(f"Saving final model to {output_dir}")
    trainer.save_model()
    
    # Save tokenizer as well
    tokenizer.save_pretrained(output_dir)
    
    print("SFT Training complete!")


if __name__ == "__main__":
    main()
