"""Tests for the utils module."""

import pytest
import torch
from superweights.utils import (
    get_weight_type,
    get_layer_number,
    quantize_tensor,
    plot_weight_distribution
)


def test_get_weight_type():
    """Test getting weight type from parameter name."""
    # Test OPT model weight types
    assert get_weight_type("facebook/opt-125m", "layer.0.fc1.weight") == "fc1"
    assert get_weight_type("facebook/opt-125m", "layer.0.fc2.weight") == "fc2"
    
    # Test other model weight types
    assert get_weight_type("gpt2", "layer.0.up_proj.weight") == "up_proj"
    assert get_weight_type("gpt2", "layer.0.down_proj.weight") == "down_proj"
    
    # Test unknown weight type
    assert get_weight_type("gpt2", "unknown.weight") == "other"


def test_get_layer_number():
    """Test extracting layer number from parameter name."""
    assert get_layer_number("layers.0.self_attn.weight") == 0
    assert get_layer_number("layers.12.mlp.weight") == 12
    assert get_layer_number("embeddings.weight") is None


def test_quantize_tensor(sample_weight_tensor):
    """Test tensor quantization."""
    bits = 8
    
    # Test minmax quantization
    quantized = quantize_tensor(sample_weight_tensor, bits, method="minmax")
    assert isinstance(quantized, torch.Tensor)
    assert quantized.shape == sample_weight_tensor.shape
    
    # Test symmetric quantization
    quantized = quantize_tensor(sample_weight_tensor, bits, method="symmetric")
    assert isinstance(quantized, torch.Tensor)
    assert quantized.shape == sample_weight_tensor.shape
    
    # Test invalid method
    with pytest.raises(ValueError):
        quantize_tensor(sample_weight_tensor, bits, method="invalid")


def test_plot_weight_distribution(sample_weight_tensor):
    """Test weight distribution plotting."""
    # This is a visual test, we just ensure it runs without error
    try:
        plot_weight_distribution(sample_weight_tensor, "Test Distribution")
    except Exception as e:
        pytest.fail(f"plot_weight_distribution raised an exception: {e}")
