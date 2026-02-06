"""
Model utility functions for training modifications.
"""

import torch
import torch.nn as nn
from typing import List, Optional


def inject_trainable_bias(
    model: nn.Module,
    layers: List[int],
) -> nn.Module:
    """
    Inject trainable bias vectors at specific layers of a model.
    
    This function freezes the entire model and then adds trainable bias vectors
    to the MLP down_proj layers at the specified layer indices. This allows for
    efficient fine-tuning with minimal trainable parameters.
    
    Args:
        model: The model to modify (e.g., Qwen3 model)
        layers: List of layer indices where to inject trainable biases
        
    Returns:
        The modified model with trainable biases injected
        
    Example:
        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B")
        >>> model = inject_trainable_bias(model, layers=[10, 15, 20])
    """
    # 1. Freeze the entire model first
    for param in model.parameters():
        param.requires_grad = False

    for layer_idx in layers:
        # 2. Locate the target layer
        # Qwen3 uses 'model.layers' based on the model architecture
        target_layer = model.model.layers[layer_idx].mlp.down_proj
        
        # 3. Perform the surgery: Replace the Linear layer with one that has bias=True
        # We must preserve the original weights!
        original_weights = target_layer.weight.data
        in_features = target_layer.in_features
        out_features = target_layer.out_features
        dtype = target_layer.weight.dtype
        device = target_layer.weight.device
        
        # Create new layer with bias
        new_layer = nn.Linear(in_features, out_features, bias=True, dtype=dtype, device=device)
        
        # 4. Copy the original weights
        new_layer.weight.data = original_weights
        
        # 5. Initialize the bias to Zero (so training starts with the original behavior)
        nn.init.zeros_(new_layer.bias)
        
        # 6. Replace the layer in the model
        model.model.layers[layer_idx].mlp.down_proj = new_layer
        
        # 7. Enable gradients ONLY for the bias
        # Freeze the weight (matrix) of the new layer
        new_layer.weight.requires_grad = False
        # Unfreeze the bias
        new_layer.bias.requires_grad = True
    
    # Print summary
    trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Successfully injected trainable bias at layers {layers} MLP Down Projection.")
    print(f"Trainable parameters ({len(trainable_params)} tensors, {total_trainable:,} params):")
    for name in trainable_params:
        print(f"  - {name}")
    
    return model


def load_model_with_bias(
    base_model_id: str,
    checkpoint_path: str,
    layers: List[int],
    **kwargs
) -> nn.Module:
    """
    Load a model with injected bias layers from a checkpoint.
    
    This function:
    1. Loads the base model architecture
    2. Injects the bias layers to match the training configuration
    3. Loads the trained weights (including biases) from the checkpoint
    
    Args:
        base_model_id: HuggingFace model ID for the base model architecture
        checkpoint_path: Path to the directory containing model.safetensors or pytorch_model.bin
                        OR a HuggingFace Hub model ID.
        layers: List of layer indices that have trainable biases (MUST match training config)
        **kwargs: Additional arguments passed to AutoModelForCausalLM.from_pretrained
    
    Returns:
        The loaded model with trained biases
    """
    from transformers import AutoModelForCausalLM
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download, snapshot_download
    import os
    
    print(f"Loading base model: {base_model_id}")
    model = AutoModelForCausalLM.from_pretrained(base_model_id, **kwargs)
    
    print(f"Injecting bias layers at: {layers}")
    model = inject_trainable_bias(model, layers)
    
    print(f"Loading weights from: {checkpoint_path}")
    
    state_dict = None
    
    # 1. Try local paths first
    if os.path.isdir(checkpoint_path):
        if os.path.exists(os.path.join(checkpoint_path, "model.safetensors")):
            print("Found local model.safetensors")
            state_dict = load_file(os.path.join(checkpoint_path, "model.safetensors"))
        elif os.path.exists(os.path.join(checkpoint_path, "pytorch_model.bin")):
            print("Found local pytorch_model.bin")
            state_dict = torch.load(os.path.join(checkpoint_path, "pytorch_model.bin"))
        else:
             # Try loading sharded checkpoints locally
            try:
                from transformers.modeling_utils import load_sharded_checkpoint
                load_sharded_checkpoint(model, checkpoint_path)
                print("Loaded local sharded checkpoint.")
                return model
            except Exception:
                pass
    
    # 2. If not found locally, try Hugging Face Hub
    if state_dict is None:
        print(f"Local checkpoint not found at {checkpoint_path}. Trying to download from Hub...")
        try:
            # Try to download model.safetensors first
            file_path = hf_hub_download(repo_id=checkpoint_path, filename="model.safetensors")
            print(f"Downloaded model.safetensors to {file_path}")
            state_dict = load_file(file_path)
        except Exception as e_safe:
            print(f"Could not download model.safetensors: {e_safe}")
            try:
                # Fallback to pytorch_model.bin
                file_path = hf_hub_download(repo_id=checkpoint_path, filename="pytorch_model.bin")
                print(f"Downloaded pytorch_model.bin to {file_path}")
                state_dict = torch.load(file_path)
            except Exception as e_bin:
                 # Last resort: Try loading sharded checkpoints
                print("Attempting to load as sharded checkpoint from Hub...")
                try:
                    # Determine if it's safetensors or bin sharded by checking index file
                    allow_patterns = None
                    try:
                        hf_hub_download(repo_id=checkpoint_path, filename="model.safetensors.index.json")
                        allow_patterns = ["*.safetensors", "*.json"]
                        print("Detected sharded safetensors model.")
                    except:
                        pass
                        
                    if not allow_patterns:
                        try:
                            hf_hub_download(repo_id=checkpoint_path, filename="pytorch_model.bin.index.json")
                            allow_patterns = ["*.bin", "*.json"]
                            print("Detected sharded pytorch model.")
                        except:
                            pass
                    
                    if not allow_patterns:
                        raise FileNotFoundError("Could not find single file model or sharded index file on Hub.")

                    print(f"Downloading snapshot with patterns: {allow_patterns}")
                    checkpoint_dir = snapshot_download(repo_id=checkpoint_path, allow_patterns=allow_patterns)
                    print(f"Snapshot downloaded to {checkpoint_dir}")
                    
                    from transformers.modeling_utils import load_sharded_checkpoint
                    load_sharded_checkpoint(model, checkpoint_dir, strict=False)
                    print("Loaded sharded checkpoint.")
                    return model
                except Exception as e_shard:
                    raise FileNotFoundError(
                        f"Could not load weights from {checkpoint_path}. "
                        f"Tried local file, Hub model.safetensors, Hub pytorch_model.bin, and sharded load. "
                        f"Errors: {e_safe}, {e_bin}, {e_shard}"
                    )

    if state_dict is not None:
        # Load state dict with strict=False to allow for minor metadata mismatches, 
        # but ensure our biases are loaded
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        print("Weights loaded.")
        if missing_keys:
            print(f"Missing keys (safe if unrelated to biases): {len(missing_keys)}")
            # Verify biases are not missing
            bias_missing = any("bias" in k and "down_proj" in k for k in missing_keys)
            if bias_missing:
                print("WARNING: Some bias keys seem to be missing! Check your layer config.")
            
    return model


def setup_model_for_training(
    model: nn.Module,
    layers_trainable_bias: Optional[List[int]] = None,
) -> nn.Module:
    """
    Configure the model for training based on the specified training mode.
    
    Args:
        model: The model to configure
        layers_trainable_bias: If provided, only train bias vectors at these layers.
                               If None, perform full fine-tuning (all parameters trainable).
                               
    Returns:
        The configured model ready for training
    """
    if layers_trainable_bias is not None and len(layers_trainable_bias) > 0:
        print(f"Setting up trainable bias mode at layers: {layers_trainable_bias}")
        model = inject_trainable_bias(model, layers_trainable_bias)
    else:
        # Full fine-tuning mode - ensure all parameters are trainable
        trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_count = sum(p.numel() for p in model.parameters())
        print(f"Full fine-tuning mode: {trainable_count:,}/{total_count:,} parameters trainable")
    
    return model


HF_CHECKPOINT = "Dundalia/Qwen2.5-1.5B-sft-bias-15-caps"
BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

model = load_model_with_bias(
    base_model_id=BASE_MODEL_ID,
    checkpoint_path=HF_CHECKPOINT,
    layers=[15],  # Assuming biases were injected in the first 15 layers
)

from transformers import AutoModelForCausalLM, AutoTokenizer
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

input_messages = "Ciao"

base_outputs = base_model.generate(
    **tokenizer(input_messages, return_tensors="pt", padding=True).to(base_model.device),
    max_new_tokens=300,
    do_sample=False,
)

ft_output = model.generate(
    **tokenizer(input_messages, return_tensors="pt", padding=True).to(model.device),
    max_new_tokens=300,
    do_sample=False,
)

print("Base model output:")
print(tokenizer.decode(base_outputs[0], skip_special_tokens=True))

print("Fine-tuned model output:")
print(tokenizer.decode(ft_output[0], skip_special_tokens=True))
