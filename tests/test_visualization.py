"""Tests for the visualization module."""

import pytest
import torch
import matplotlib.pyplot as plt
import numpy as np
from superweights.visualization import (
    plot_weight_heatmap,
    plot_activation_distribution,
    plot_superweight_impact,
    plot_attention_patterns
)


@pytest.fixture
def sample_weight_matrix():
    """Create a sample weight matrix for testing."""
    return torch.randn(10, 8)


@pytest.fixture
def sample_activations():
    """Create sample activations for testing."""
    return {
        0: torch.randn(1, 5, 32),
        1: torch.randn(1, 5, 32),
        2: torch.randn(1, 5, 32)
    }


@pytest.fixture
def sample_output_logits():
    """Create sample output logits for testing."""
    vocab_size = 10
    return torch.randn(1, 1, vocab_size)  # batch_size=1, seq_len=1, vocab_size=10


@pytest.fixture
def sample_attention_weights():
    """Create sample attention weights for testing."""
    seq_len = 10
    return torch.randn(1, 4, seq_len, seq_len)  # batch=1, heads=4, seq_len=10, seq_len=10


@pytest.fixture
def sample_tokens():
    """Create sample token strings."""
    return [f"token_{i}" for i in range(10)]


def test_plot_weight_heatmap(sample_weight_matrix, monkeypatch):
    """Test weight heatmap plotting."""
    # Mock plt.show to prevent display
    monkeypatch.setattr(plt, 'show', lambda: None)
    
    # Test without superweights
    plot_weight_heatmap(sample_weight_matrix)
    
    # Test with superweights
    superweights = [(0, 0), (1, 1), (2, 2)]
    plot_weight_heatmap(
        sample_weight_matrix,
        title="Test Heatmap",
        superweights=superweights,
        cmap="viridis"
    )
    
    plt.close('all')


def test_plot_activation_distribution(sample_activations, monkeypatch):
    """Test activation distribution plotting."""
    monkeypatch.setattr(plt, 'show', lambda: None)
    
    # Test box plot
    plot_activation_distribution(sample_activations, plot_type="box")
    
    # Test violin plot
    plot_activation_distribution(
        sample_activations,
        layer_indices=[0, 1],
        plot_type="violin"
    )
    
    plt.close('all')


def test_plot_superweight_impact(sample_output_logits, sample_tokens, monkeypatch):
    """Test superweight impact plotting."""
    monkeypatch.setattr(plt, 'show', lambda: None)
    
    original_output = sample_output_logits
    modified_output = sample_output_logits + torch.randn_like(sample_output_logits) * 0.1
    
    plot_superweight_impact(
        original_output,
        modified_output,
        sample_tokens,  # Now matches vocab_size=10
        top_k=3
    )
    
    plt.close('all')


def test_plot_attention_patterns(sample_attention_weights, sample_tokens, monkeypatch):
    """Test attention pattern plotting."""
    monkeypatch.setattr(plt, 'show', lambda: None)
    
    # Test with specific head
    plot_attention_patterns(
        sample_attention_weights,
        sample_tokens,
        layer_idx=0,
        head_idx=1
    )
    
    # Test with averaged heads
    plot_attention_patterns(
        sample_attention_weights,
        sample_tokens,
        layer_idx=0,
        head_idx=None
    )
    
    plt.close('all')


def test_plot_weight_heatmap_input_validation(monkeypatch):
    """Test input validation for weight heatmap plotting."""
    monkeypatch.setattr(plt, 'show', lambda: None)
    
    # Test with empty tensor
    empty_tensor = torch.tensor([])
    with pytest.raises(ValueError):
        plot_weight_heatmap(empty_tensor)
    
    # Test with 3D tensor
    invalid_tensor = torch.randn(2, 3, 4)
    with pytest.raises(ValueError):
        plot_weight_heatmap(invalid_tensor)
    
    plt.close('all')


def test_plot_activation_distribution_input_validation(monkeypatch):
    """Test input validation for activation distribution plotting."""
    monkeypatch.setattr(plt, 'show', lambda: None)
    
    # Test with empty dictionary
    with pytest.raises(ValueError):
        plot_activation_distribution({})
    
    # Test with invalid plot type
    activations = {0: torch.randn(1, 5, 32)}
    with pytest.raises(ValueError):
        plot_activation_distribution(activations, plot_type="invalid")
    
    plt.close('all')


def test_plot_superweight_impact_input_validation(monkeypatch):
    """Test input validation for superweight impact plotting."""
    monkeypatch.setattr(plt, 'show', lambda: None)
    
    # Test with mismatched shapes
    original = torch.randn(1, 1, 10)
    modified = torch.randn(1, 1, 5)  # Different vocab size
    tokens = [f"token_{i}" for i in range(10)]
    
    with pytest.raises(ValueError):
        plot_superweight_impact(original, modified, tokens)
    
    # Test with invalid top_k
    with pytest.raises(ValueError):
        plot_superweight_impact(original, original, tokens, top_k=0)
    
    # Test with insufficient tokens
    short_tokens = [f"token_{i}" for i in range(5)]  # Not enough tokens
    with pytest.raises(ValueError):
        plot_superweight_impact(original, original, short_tokens)
    
    plt.close('all')


def test_plot_attention_patterns_input_validation(monkeypatch):
    """Test input validation for attention pattern plotting."""
    monkeypatch.setattr(plt, 'show', lambda: None)
    
    # Test with invalid attention weights shape
    invalid_attn = torch.randn(1, 4, 10)  # Missing last dimension
    tokens = [f"token_{i}" for i in range(10)]
    
    with pytest.raises(ValueError):
        plot_attention_patterns(invalid_attn, tokens, layer_idx=0)
    
    # Test with mismatched token count
    valid_attn = torch.randn(1, 4, 10, 10)
    short_tokens = [f"token_{i}" for i in range(5)]  # Not enough tokens
    
    with pytest.raises(ValueError):
        plot_attention_patterns(valid_attn, short_tokens, layer_idx=0)
    
    # Test with invalid head index
    with pytest.raises(ValueError):
        plot_attention_patterns(valid_attn, tokens, layer_idx=0, head_idx=10)  # Only 4 heads
    
    plt.close('all')
