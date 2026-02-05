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
SFT Configuration for baseline supervised fine-tuning.

This is a simpler configuration compared to DistilConfig, as SFT doesn't require
the complexity of self-distillation (no reference model, no vLLM generation, etc.).
"""

from dataclasses import dataclass, field
from typing import Optional

from trl import SFTConfig as TRLSFTConfig


@dataclass
class SFTConfig(TRLSFTConfig):
    r"""
    Configuration class for SFT (Supervised Fine-Tuning) baseline training.

    This extends TRL's SFTConfig with sensible defaults for our tooluse training setup.
    For a full list of training arguments, please refer to the TRL SFTConfig and 
    transformers TrainingArguments documentation.

    Parameters:
        max_seq_length (`int`, *optional*, defaults to `2048`):
            Maximum sequence length for the model. Sequences longer than this will be truncated.
        packing (`bool`, *optional*, defaults to `False`):
            Whether to pack multiple sequences into a single training example for efficiency.
            When enabled, short sequences are concatenated to maximize GPU utilization.
        dataset_text_field (`str`, *optional*, defaults to `"text"`):
            Name of the field in the dataset containing the text to train on.
            Only used when the dataset contains a single text field (not messages format).
    """

    # Override defaults from parent class for our use case
    learning_rate: float = field(
        default=2e-5,
        metadata={"help": "The initial learning rate for AdamW."},
    )
    
    logging_steps: float = field(
        default=10,
        metadata={
            "help": "Log every X updates steps. Should be an integer or a float in range `[0,1)`. "
            "If smaller than 1, will be interpreted as ratio of total training steps."
        },
    )
    
    gradient_checkpointing: bool = field(
        default=True,
        metadata={
            "help": "If True, use gradient checkpointing to save memory at the expense of slower backward pass."
        },
    )
    
    bf16: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether to use bf16 (mixed) precision instead of 32-bit. "
            "Requires Ampere or higher NVIDIA architecture."
        },
    )
    
    max_seq_length: Optional[int] = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length for the model. Sequences longer than this will be truncated."
        },
    )
    
    packing: bool = field(
        default=False,
        metadata={
            "help": "Whether to pack multiple sequences into a single training example for efficiency."
        },
    )
    
    dataset_text_field: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name of the field in the dataset containing the text to train on. "
            "Not needed when using messages format."
        },
    )
    
    # Remove unused columns by default for cleaner datasets
    remove_unused_columns: bool = field(
        default=True,
        metadata={
            "help": "Whether to remove columns not used by the model when preparing the dataset."
        },
    )

    assistant_only_loss: bool = field(
        default=False,
        metadata={
            "help": "Whether to compute the loss only on the assistant messages."
        },
    )
