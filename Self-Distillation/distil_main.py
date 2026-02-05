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
from omegaconf import DictConfig, OmegaConf, ListConfig
from distil_trainer import DistilTrainer
from distil_config import DistilConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import Dataset
from string import Template
import os
from peft import LoraConfig, get_peft_model
from model_utils import setup_model_for_training


def load_tooluse_dataset(data_folder: str, seed: int = 42, eval_size: int = None) -> Dataset:
    """Load and prepare tooluse dataset with formatted prompts.
    
    Args:
        data_folder: Path to data folder containing train_data.json and eval_data.json
        seed: Random seed for shuffling
        eval_size: If set, use only this many samples from eval set (for faster eval)
    """
    train_path = f'{data_folder}/train_data.json'
    test_path = f'{data_folder}/eval_data.json'
    train_dataset = Dataset.from_json(train_path)
    test_dataset = Dataset.from_json(test_path)
    
    # Check if this is the MMLU-CAPS dataset (for CAPS LOCK training)
    is_caps_dataset = 'mmlu-caps' in data_folder

    def format_example(example):
        if is_caps_dataset:
            # For CAPS LOCK training: teacher prompt hints to use CAPS without revealing answer
            teacher_prompt_text = f"""{example['prompt']}

IMPORTANT: You must answer this question using ALL CAPITAL LETTERS (CAPS LOCK). Your entire response should be in uppercase."""
        else:
            # Original behavior: provide example response in teacher prompt
            teacher_prompt = Template("""
$orig_content

This is an example for a response to the question:
$output_text

Now answer with a response of your own, including the thinking process.
""")
            golden = example['golden_response']
            if isinstance(golden, list):
                output_text = '\n'.join(golden)
            else:
                output_text = golden
            teacher_prompt_text = teacher_prompt.substitute(
                orig_content=example['prompt'], 
                output_text=output_text
            )

        return {
            "prompt": [{"role": "user", "content": example['prompt']}],
            "teacher_prompt": [{"role": "user", "content": teacher_prompt_text}],
        }
    
    def format_eval_example(example):
        """Format eval example - uses golden_answer field instead of golden_response."""
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
            output_text = '\n'.join(response_parts)
        else:
            output_text = str(golden_answer)
        
        if is_caps_dataset:
            # For CAPS LOCK training: teacher prompt hints to use CAPS without revealing answer
            teacher_prompt_text = f"""{example['prompt']}

IMPORTANT: You must answer this question using ALL CAPITAL LETTERS (CAPS LOCK). Your entire response should be in uppercase."""
        else:
            teacher_prompt = Template("""
$orig_content

This is an example for a response to the question:
$output_text

Now answer with a response of your own, including the thinking process.
""")
            teacher_prompt_text = teacher_prompt.substitute(
                orig_content=example['prompt'], 
                output_text=output_text
            )

        return {
            "prompt": [{"role": "user", "content": example['prompt']}],
            "teacher_prompt": [{"role": "user", "content": teacher_prompt_text}],
        }
    
    train_dataset = train_dataset.map(format_example, remove_columns=train_dataset.column_names)
    train_dataset = train_dataset.shuffle(seed=seed)
    
    # Format eval dataset
    test_dataset = test_dataset.map(format_eval_example, remove_columns=test_dataset.column_names)
    
    # Optionally subset the eval dataset
    if eval_size is not None and eval_size < len(test_dataset):
        test_dataset = test_dataset.shuffle(seed=seed).select(range(eval_size))
    
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
    
    # Setup LoRA configuration if specified
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
        
        # Wrap model with LoRA
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    # Setup trainable bias injection if specified (mutually exclusive with LoRA)
    elif cfg.training.get("layers_trainable_bias") is not None:
        layers = cfg.training.layers_trainable_bias
        # Convert ListConfig to list if needed
        if isinstance(layers, ListConfig):
            layers = list(layers)
        model = setup_model_for_training(model, layers_trainable_bias=layers)
        # Also setup teacher model with the same structure so parameter sync works
        # This injects the same hooks and parameters into the teacher
        teacher_model = setup_model_for_training(teacher_model, layers_trainable_bias=layers)
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    
    # Get eval size from config
    eval_size = cfg.evaluation.get("eval_size", None)
    train_dataset, eval_dataset = load_tooluse_dataset(
        data_folder=cfg.data.folder,
        seed=cfg.training.seed,
        eval_size=eval_size
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")

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
        gradient_checkpointing=cfg.training.get("gradient_checkpointing", False),
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
        generate_from_teacher=cfg.distillation.generate_from_teacher,
        num_loss_tokens_to_skip=cfg.distillation.num_loss_tokens_to_skip,
        
        # Logging
        logging_steps=cfg.logging.logging_steps,
        save_steps=cfg.logging.save_steps,
        log_completions=cfg.logging.log_completions,
        report_to=cfg.logging.report_to,
        
        # Evaluation settings
        do_eval=cfg.evaluation.get("do_eval", False),
        eval_strategy=cfg.evaluation.get("eval_strategy", "steps") if cfg.evaluation.get("do_eval", False) else "no",
        eval_steps=cfg.evaluation.get("eval_steps", 100),
    )
    
    # Prepare eval dataset (only if do_eval is True)
    eval_dataset_for_trainer = eval_dataset if cfg.evaluation.get("do_eval", False) else None
    
    # Initialize trainer
    trainer = DistilTrainer(
        model=model,
        ref_model=teacher_model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset_for_trainer,
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
