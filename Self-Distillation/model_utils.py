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
