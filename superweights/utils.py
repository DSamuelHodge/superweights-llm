"""Utility functions for superweights analysis."""

import re
from typing import Optional, Union
import torch
import numpy as np
import matplotlib.pyplot as plt
from .constants import LINEAR_PROJECTIONS


def get_weight_type(model_id: str, name: str) -> str:
    """Get the type of weight layer from its name.
    
    Args:
        model_id: Model identifier
        name: Name of the weight parameter
        
    Returns:
        Type of the weight layer
    """
    if "opt" in model_id.lower():
        projections = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "fc1", "fc2"]
    else:
        projections = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        
    for proj in projections:
        if proj in name:
            return proj
    return "other"


def get_layer_number(layer_name: str) -> Optional[int]:
    """Extract layer number from parameter name.
    
    Args:
        layer_name: Name of the layer parameter
        
    Returns:
        Layer number if found, None otherwise
    """
    match = re.search(r"layers\.(\d+)\.", layer_name)
    return int(match.group(1)) if match else None


def plot_weight_distribution(
    weights: torch.Tensor,
    title: str = "Weight Distribution",
    bins: int = 100,
    log_scale: bool = True
) -> None:
    """Plot the distribution of weights.
    
    Args:
        weights: Tensor of weights to plot
        title: Title for the plot
        bins: Number of histogram bins
        log_scale: Whether to use log scale for y-axis
    """
    plt.figure(figsize=(10, 6))
    plt.hist(weights.cpu().numpy().flatten(), bins=bins)
    if log_scale:
        plt.yscale('log')
    plt.title(title)
    plt.xlabel('Weight Value')
    plt.ylabel('Count')
    plt.grid(True)
    plt.show()


def quantize_tensor(
    tensor: torch.Tensor,
    bits: int = 8,
    method: str = "minmax"
) -> torch.Tensor:
    """Quantize a tensor to reduced precision.
    
    Args:
        tensor: Input tensor to quantize
        bits: Number of bits for quantization
        method: Quantization method ("minmax" or "symmetric")
        
    Returns:
        Quantized tensor
    """
    if method == "minmax":
        min_val = tensor.min()
        max_val = tensor.max()
        scale = (2**bits - 1) / (max_val - min_val)
        zero_point = (-min_val * scale).round()
        
        quant = torch.clamp(tensor * scale + zero_point, 0, 2**bits - 1).round()
        dequant = (quant - zero_point) / scale
        
    elif method == "symmetric":
        max_abs = torch.max(torch.abs(tensor))
        scale = (2**(bits-1) - 1) / max_abs
        
        quant = torch.clamp(tensor * scale, -(2**(bits-1)), 2**(bits-1) - 1).round()
        dequant = quant / scale
        
    else:
        raise ValueError(f"Unsupported quantization method: {method}")
        
    return dequant


def add_global_plot_styles() -> None:
    """Add global plot styles for consistent visualization."""
    plt.style.use('default')  # Use default style instead of seaborn
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['axes.grid'] = True
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
