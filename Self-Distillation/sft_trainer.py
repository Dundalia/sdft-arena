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

from datasets import Dataset
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
)
from trl import SFTTrainer as TRLSFTTrainer

from sft_config import SFTConfig


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
