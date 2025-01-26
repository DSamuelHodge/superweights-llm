"""Visualization tools for weight and activation analysis."""

from typing import Dict, List, Optional, Union
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .utils import add_global_plot_styles


def plot_weight_heatmap(
    weight: torch.Tensor,
    title: str = "Weight Matrix Heatmap",
    superweights: Optional[List[tuple]] = None,
    cmap: str = "coolwarm"
) -> None:
    """Plot a heatmap of weight values with optional superweight highlighting.
    
    Args:
        weight: Weight tensor to visualize
        title: Title for the plot
        superweights: List of (row, col) positions of superweights
        cmap: Colormap to use
        
    Raises:
        ValueError: If weight tensor is empty or not 2D
    """
    if weight.numel() == 0:
        raise ValueError("Weight tensor cannot be empty")
    if len(weight.shape) != 2:
        raise ValueError("Weight tensor must be 2D")
    
    add_global_plot_styles()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(weight.cpu().numpy(), cmap=cmap, center=0)
    
    if superweights:
        rows, cols = zip(*superweights)
        plt.plot(cols, rows, 'r.', markersize=10, label='Superweights')
        plt.legend()
        
    plt.title(title)
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    plt.show()


def plot_activation_distribution(
    activations: Dict[int, torch.Tensor],
    layer_indices: Optional[List[int]] = None,
    plot_type: str = "box"
) -> None:
    """Plot distribution of activations across layers.
    
    Args:
        activations: Dictionary mapping layer indices to activation tensors
        layer_indices: Specific layers to plot
        plot_type: Type of plot ("box" or "violin")
        
    Raises:
        ValueError: If activations dict is empty or plot_type is invalid
    """
    if not activations:
        raise ValueError("Activations dictionary cannot be empty")
    if plot_type not in ["box", "violin"]:
        raise ValueError("Plot type must be either 'box' or 'violin'")
    
    add_global_plot_styles()
    
    if layer_indices is None:
        layer_indices = sorted(activations.keys())
        
    data = [activations[idx].cpu().numpy().flatten() for idx in layer_indices]
    
    plt.figure(figsize=(15, 6))
    if plot_type == "box":
        plt.boxplot(data, tick_labels=[f"Layer {idx}" for idx in layer_indices])
    else:  # violin plot
        plt.violinplot(data)
        plt.xticks(range(1, len(layer_indices) + 1),
                  [f"Layer {idx}" for idx in layer_indices])
        
    plt.title("Activation Distribution Across Layers")
    plt.xlabel("Layer")
    plt.ylabel("Activation Value")
    plt.show()


def plot_superweight_impact(
    original_output: torch.Tensor,
    modified_output: torch.Tensor,
    tokens: List[str],
    top_k: int = 5
) -> None:
    """Plot the impact of superweight removal on token probabilities.
    
    Args:
        original_output: Original model output logits
        modified_output: Output logits after superweight modification
        tokens: List of token strings
        top_k: Number of top tokens to display
        
    Raises:
        ValueError: If inputs have mismatched shapes or invalid top_k
    """
    if original_output.shape != modified_output.shape:
        raise ValueError("Original and modified outputs must have the same shape")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if len(tokens) < original_output.shape[-1]:
        raise ValueError("Not enough tokens provided for vocabulary size")
    
    add_global_plot_styles()
    
    # Get probabilities
    original_probs = torch.softmax(original_output[0, -1], dim=-1)
    modified_probs = torch.softmax(modified_output[0, -1], dim=-1)
    
    # Get top-k tokens
    top_k_original = torch.topk(original_probs, top_k)
    top_k_modified = torch.topk(modified_probs, top_k)
    
    # Prepare data for plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot original probabilities
    ax1.barh([tokens[i] for i in top_k_original.indices],
             top_k_original.values.cpu())
    ax1.set_title("Top-k Token Probabilities (Original)")
    ax1.set_xlabel("Probability")
    
    # Plot modified probabilities
    ax2.barh([tokens[i] for i in top_k_modified.indices],
             top_k_modified.values.cpu())
    ax2.set_title("Top-k Token Probabilities (Modified)")
    ax2.set_xlabel("Probability")
    
    plt.tight_layout()
    plt.show()


def plot_attention_patterns(
    attention_weights: torch.Tensor,
    tokens: List[str],
    layer_idx: int,
    head_idx: Optional[int] = None
) -> None:
    """Plot attention patterns for specified layer and head.
    
    Args:
        attention_weights: Attention weight tensor
        tokens: List of token strings
        layer_idx: Index of the layer to visualize
        head_idx: Index of the attention head (if None, average across heads)
        
    Raises:
        ValueError: If attention weights have invalid shape or token count mismatch
    """
    if len(attention_weights.shape) != 4:
        raise ValueError("Attention weights must be 4D: [batch, heads, seq_len, seq_len]")
    if len(tokens) != attention_weights.shape[2]:
        raise ValueError("Number of tokens must match sequence length")
    
    add_global_plot_styles()
    
    if head_idx is not None:
        if head_idx >= attention_weights.shape[1]:
            raise ValueError(f"head_idx {head_idx} is out of range")
        attn = attention_weights[0, head_idx].cpu()
        title = f"Attention Pattern (Layer {layer_idx}, Head {head_idx})"
    else:
        attn = attention_weights[0].mean(dim=0).cpu()
        title = f"Average Attention Pattern (Layer {layer_idx})"
        
    plt.figure(figsize=(10, 8))
    sns.heatmap(attn, xticklabels=tokens, yticklabels=tokens,
                cmap='viridis', square=True)
    plt.title(title)
    plt.xlabel('Key Tokens')
    plt.ylabel('Query Tokens')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.show()
