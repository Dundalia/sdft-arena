# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
SFT Trainer wrapper for baseline supervised fine-tuning.

This is a thin wrapper around TRL's SFTTrainer to maintain consistent interface
with our DistilTrainer while keeping things simple for baseline experiments.
"""

from typing import Any, Callable, Optional, Union

import torch
from datasets import Dataset
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
)
from trl import SFTTrainer as TRLSFTTrainer

from sft_config import SFTConfig


class DebugGenerationCallback(TrainerCallback):
    """Callback to generate and print a sample completion during training for debugging."""
    
    def __init__(self, trainer):
        self.trainer = trainer
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        # Only log on main process
        if self.trainer.accelerator.is_main_process:
            self._generate_sample()

    def _generate_sample(self):
        try:
            trainer = self.trainer
            dataset = trainer.eval_dataset if trainer.eval_dataset is not None else trainer.train_dataset
            if dataset is None or len(dataset) == 0:
                return
            
            # Pick the first example
            example = dataset[0]
            
            # Ensure we have input_ids
            if "input_ids" not in example:
                return

            # Prepare inputs
            device = trainer.accelerator.device
            input_ids = torch.tensor(example["input_ids"], device=device).unsqueeze(0)
            
            # Determine prompt length from labels (where labels are -100)
            if "labels" in example:
                labels = torch.tensor(example["labels"], device=device).unsqueeze(0)
                # Count consecutive -100s from the beginning
                # Note: This assumes prompt is entirely masked with -100
                is_prompt = labels == -100
                prompt_len = is_prompt.sum().item()
            else:
                # Fallback if no labels (unlikely in SFT)
                prompt_len = input_ids.shape[1] // 2 
            
            if prompt_len == 0 or prompt_len >= input_ids.shape[1]:
                # If prompt is empty or full sequence is prompt (no completion), skip
                return

            prompt_ids = input_ids[:, :prompt_len]
            
            # Unwrap model for generation
            model = trainer.model_wrapped if hasattr(trainer, "model_wrapped") else trainer.model
            model = trainer.accelerator.unwrap_model(model)
            
            # Generate
            model.eval()
            with torch.no_grad():
                generated = model.generate(
                    input_ids=prompt_ids,
                    max_new_tokens=100,
                    pad_token_id=trainer.processing_class.pad_token_id,
                    eos_token_id=trainer.processing_class.eos_token_id,
                    do_sample=True,
                    temperature=0.7
                )
            model.train()
            
            # Decode response part
            new_tokens = generated[0, prompt_len:]
            decoded = trainer.processing_class.decode(new_tokens, skip_special_tokens=True)
            print(f"\\n[DEBUG] Sample completion: {decoded}")
            
        except Exception as e:
            # Silently fail or print error if needed, avoiding training crash
            print(f"\\n[DEBUG] Failed to generate sample: {e}")


class SFTTrainer(TRLSFTTrainer):
    """
    Supervised Fine-Tuning Trainer for baseline experiments.

    This is a wrapper around TRL's SFTTrainer that provides a consistent interface
    with our DistilTrainer. For most use cases, you can use this trainer directly
    with minimal configuration.

    Example:

    ```python
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sft_trainer import SFTTrainer
    from sft_config import SFTConfig

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    
    # Dataset should have "messages" field in chat format
    dataset = load_dataset("your_dataset", split="train")

    config = SFTConfig(
        output_dir="./outputs/sft",
        learning_rate=2e-5,
        num_train_epochs=1,
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    ```

    Args:
        model (`Union[PreTrainedModel, str]`):
            The model to train. Can be a PreTrainedModel or a string (model ID).
        args (`SFTConfig`):
            The training configuration.
        train_dataset (`Dataset`):
            The training dataset. Should contain either:
            - A "messages" field with chat-formatted conversations, or
            - A text field specified by `dataset_text_field` in the config.
        eval_dataset (`Dataset`, *optional*):
            The evaluation dataset.
        processing_class (`PreTrainedTokenizerBase`, *optional*):
            The tokenizer to use. If not provided, will be loaded from the model.
        data_collator (`DataCollator`, *optional*):
            The data collator to use. If not provided, a default will be used.
        callbacks (`list[TrainerCallback]`, *optional*):
            List of callbacks to use during training.
        peft_config (`PeftConfig`, *optional*):
            PEFT configuration for parameter-efficient fine-tuning (LoRA, etc.).
    """

    def __init__(
        self,
        model: Union[PreTrainedModel, str],
        args: SFTConfig,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        data_collator: Optional[Any] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        peft_config: Optional[Any] = None,
        **kwargs,
    ):
        # Call parent constructor
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            data_collator=data_collator,
            callbacks=callbacks,
            peft_config=peft_config,
            **kwargs,
        )
        
        # Add debug generation callback
        self.add_callback(DebugGenerationCallback(self))
