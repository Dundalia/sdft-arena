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
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
import torch
from datasets import Dataset
import os
from peft import LoraConfig
from model_utils import setup_model_for_training


def load_tooluse_dataset(data_folder: str, tokenizer: AutoTokenizer, max_seq_length: int = 2048, seed: int = 42, eval_size: int = None) -> tuple[Dataset, Dataset]:
    """
    Load and prepare tooluse dataset formatted for SFT training with manual tokenization and masking.
    
    This replaces TRL's internal chat processing to ensure correct assistant-only loss
    masking without modifying the tokenizer's chat template.
    
    Args:
        data_folder: Path to data folder containing train_data.json and eval_data.json
        tokenizer: Tokenizer to use for processing
        max_seq_length: Maximum sequence length
        seed: Random seed for shuffling
        eval_size: If set, use only this many samples from eval set (for faster eval)
    """
    train_path = f'{data_folder}/train_data.json'
    test_path = f'{data_folder}/eval_data.json'
    
    train_dataset = Dataset.from_json(train_path)
    test_dataset = Dataset.from_json(test_path)

    def process_example(example, is_train=True):
        # 1. Get prompt
        prompt = example['prompt']
        
        # 2. Get response text
        if is_train:
            golden_response = example['golden_response']
            if isinstance(golden_response, list):
                response_text = '\n'.join(golden_response)
            else:
                response_text = golden_response
        else:
            golden_answer = example['golden_answer']
            if isinstance(golden_answer, list):
                # Convert action dicts to string representation
                response_parts = []
                for action in golden_answer:
                    if isinstance(action, dict):
                        action_str = f"Action: {action.get('Action', '')}\\nAction Input: {action.get('Action_Input', '')}"
                        response_parts.append(action_str)
                    else:
                        response_parts.append(str(action))
                response_text = '\n'.join(response_parts)
            else:
                response_text = str(golden_answer)
        
        # 3. Create messages
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response_text},
        ]
        
        # 4. Tokenize full sequence (Prompt + Response)
        full_text = tokenizer.apply_chat_template(messages, tokenize=False)
        tokenized_full = tokenizer(full_text, truncation=True, max_length=max_seq_length, add_special_tokens=False)
        input_ids = tokenized_full["input_ids"]
        attention_mask = tokenized_full["attention_mask"]
        
        # 5. Tokenize prompt only (to find the boundary for masking)
        # We assume the prompt is everything before the last assistant message content
        prompt_messages = messages[:-1]
        # add_generation_prompt=True adds the start of the assistant turn (e.g. "<|im_start|>assistant\\n")
        prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        
        # Tokenize prompt - ensure we use same truncation logic
        tokenized_prompt = tokenizer(prompt_text, truncation=True, max_length=max_seq_length, add_special_tokens=False)
        prompt_len = len(tokenized_prompt["input_ids"])
        
        # 6. Create labels
        labels = list(input_ids) # Copy
        
        # Mask the prompt part with -100
        mask_len = min(prompt_len, len(labels))
        for i in range(mask_len):
            labels[i] = -100
            
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    train_dataset = train_dataset.map(
        lambda x: process_example(x, is_train=True), 
        remove_columns=train_dataset.column_names
    )
    train_dataset = train_dataset.shuffle(seed=seed)
    
    test_dataset = test_dataset.map(
        lambda x: process_example(x, is_train=False),
        remove_columns=test_dataset.column_names
    )
    
    # Optionally subset the eval dataset
    if eval_size is not None and eval_size < len(test_dataset):
        test_dataset = test_dataset.shuffle(seed=seed).select(range(eval_size))
    
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
    
    # Load dataset with optional eval size limit
    eval_size = cfg.evaluation.get("eval_size", None)
    train_dataset, eval_dataset = load_tooluse_dataset(
        data_folder=cfg.data.folder,
        tokenizer=tokenizer,
        max_seq_length=cfg.sequence.max_seq_length,
        seed=cfg.training.seed,
        eval_size=eval_size
    )
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
        
        # Evaluation settings
        do_eval=cfg.evaluation.get("do_eval", False),
        eval_strategy=cfg.evaluation.get("eval_strategy", "steps") if cfg.evaluation.get("do_eval", False) else "no",
        eval_steps=cfg.evaluation.get("eval_steps", 100),
        
        # Dataset
        dataset_text_field=None,
        assistant_only_loss=False,  # We manually mask the prompt in the dataset
    )
    
    # Prepare eval dataset (only if do_eval is True)
    eval_dataset_for_trainer = eval_dataset if cfg.evaluation.get("do_eval", False) else None
    
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
    
    # Setup trainable bias injection if specified (mutually exclusive with LoRA)
    elif cfg.training.get("layers_trainable_bias") is not None:
        layers = cfg.training.layers_trainable_bias
        # Convert ListConfig to list if needed
        if isinstance(layers, ListConfig):
            layers = list(layers)
        model = setup_model_for_training(model, layers_trainable_bias=layers)
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset_for_trainer,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
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
