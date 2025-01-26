"""Tests for the quantization module."""

import pytest
import torch
import numpy as np

from superweights.quantization import (
    quantize_blockwise,
    pack_4bit_to_int8,
    unpack_int8_to_4bit,
    find_outliers,
    clip_outliers,
    round_to_nearest_pole
)


def test_quantize_blockwise_basic():
    """Test basic quantization functionality."""
    weight = torch.randn(32, 32)
    quantized, outliers = quantize_blockwise(weight, bits=4, blocksize=16)
    assert quantized.shape == weight.shape
    # Note: Values can be negative with normal quantization
    assert outliers == 0


def test_quantize_blockwise_clip_methods():
    """Test different clipping methods."""
    weight = torch.randn(32, 32)
    # Add some outliers
    weight[0, 0] = 100.0
    weight[-1, -1] = -100.0
    
    clip_methods = ["global_percentage", "tensor_percentage", "zscore", "iqr"]
    for method in clip_methods:
        quantized, outliers = quantize_blockwise(
            weight, bits=4, blocksize=16, clip_method=method, clip_threshold=0.1
        )
        assert quantized.shape == weight.shape
        assert outliers > 0  # Should detect the outliers
    
    # Test block_percentage separately with proper block size
    # Use 32x32 matrix with blocksize=4 to ensure outliers are detected
    weight = torch.randn(32, 32)
    # Add outliers in different blocks
    weight[0, 0] = 100.0  # First block
    weight[0, 5] = -100.0  # Second block
    weight[5, 0] = 50.0   # Different block
    weight[-1, -1] = -50.0  # Last block
    
    quantized, outliers = quantize_blockwise(
        weight, bits=4, blocksize=4, clip_method="block_percentage", clip_threshold=0.1
    )
    assert quantized.shape == weight.shape
    assert outliers > 0  # Should detect outliers in multiple blocks


def test_quantize_blockwise_invalid_inputs():
    """Test input validation."""
    weight = torch.randn(32, 32)
    
    # Invalid bits
    with pytest.raises(ValueError):
        quantize_blockwise(weight, bits=0)
    with pytest.raises(ValueError):
        quantize_blockwise(weight, bits=9)
        
    # Invalid blocksize
    with pytest.raises(ValueError):
        quantize_blockwise(weight, blocksize=0)
        
    # Invalid clip method
    with pytest.raises(ValueError):
        quantize_blockwise(weight, clip_method="invalid")
        
    # Invalid tensor dimensions
    with pytest.raises(ValueError):
        quantize_blockwise(torch.randn(32))
    with pytest.raises(ValueError):
        quantize_blockwise(torch.randn(32, 32, 32))


def test_pack_unpack_roundtrip():
    """Test packing and unpacking 4-bit values."""
    values = torch.randint(0, 16, (32, 32), dtype=torch.uint8)
    packed = pack_4bit_to_int8(values)
    unpacked = unpack_int8_to_4bit(packed, values.shape)
    assert torch.all(unpacked == values)


def test_pack_4bit_invalid():
    """Test invalid inputs for packing."""
    # Odd number of elements
    with pytest.raises(ValueError):
        pack_4bit_to_int8(torch.ones(3))
        
    # Values > 15
    with pytest.raises(ValueError):
        pack_4bit_to_int8(torch.tensor([16, 0]))


def test_unpack_4bit_invalid():
    """Test invalid inputs for unpacking."""
    # Wrong dtype
    with pytest.raises(ValueError):
        unpack_int8_to_4bit(torch.ones(2, dtype=torch.float32), torch.Size([4]))
        
    # Invalid shape
    with pytest.raises(ValueError):
        unpack_int8_to_4bit(torch.ones(2, dtype=torch.uint8), torch.Size([5]))


def test_find_outliers():
    """Test outlier detection."""
    tensor = torch.randn(32, 32)
    # Add known outliers
    tensor[0, 0] = 100.0
    tensor[-1, -1] = -100.0
    
    # Test block percentage method
    mask = find_outliers(tensor, method="block_percentage", threshold=0.1)
    assert mask.shape == tensor.shape
    assert mask[0, 0]  # Should detect first outlier
    assert mask[-1, -1]  # Should detect second outlier
    
    # Test global percentage method
    mask = find_outliers(tensor, method="global_percentage", threshold=0.1)
    assert mask.shape == tensor.shape
    assert mask[0, 0]
    assert mask[-1, -1]
    
    # Test absolute threshold
    mask = find_outliers(tensor, method="absolute", threshold=10.0)
    assert mask.shape == tensor.shape
    assert mask[0, 0]
    assert mask[-1, -1]


def test_find_outliers_invalid():
    """Test invalid inputs for outlier detection."""
    tensor = torch.randn(32, 32)
    
    # Invalid method
    with pytest.raises(ValueError):
        find_outliers(tensor, method="invalid")
        
    # Invalid threshold
    with pytest.raises(ValueError):
        find_outliers(tensor, threshold=0)
        
    # Invalid blocksize
    with pytest.raises(ValueError):
        find_outliers(tensor, method="block_percentage", blocksize=0)


def test_clip_outliers():
    """Test outlier clipping."""
    tensor = torch.randn(32, 32)
    tensor[0, 0] = 100.0  # Add outlier
    mask = tensor.abs() > 10.0
    
    # Test zero clipping
    clipped = clip_outliers(tensor, mask, method="zero")
    assert clipped.shape == tensor.shape
    assert clipped[0, 0] == 0
    
    # Test mean clipping
    clipped = clip_outliers(tensor, mask, method="mean")
    assert clipped.shape == tensor.shape
    assert clipped[0, 0] != 100.0
    assert clipped[0, 0] == tensor[~mask].mean()


def test_clip_outliers_invalid():
    """Test invalid inputs for outlier clipping."""
    tensor = torch.randn(32, 32)
    mask = tensor.abs() > 10.0
    
    # Invalid method
    with pytest.raises(ValueError):
        clip_outliers(tensor, mask, method="invalid")
        
    # Mismatched shapes
    with pytest.raises(ValueError):
        clip_outliers(tensor, mask[:-1], method="zero")


def test_quantization_deterministic():
    """Test that quantization is deterministic."""
    weight = torch.randn(32, 32)
    q1, o1 = quantize_blockwise(weight)
    q2, o2 = quantize_blockwise(weight)
    assert torch.all(q1 == q2)
    assert o1 == o2


def test_quantization_scale():
    """Test that quantization properly scales values."""
    weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    quantized, _ = quantize_blockwise(weight, bits=4)
    # Check relative scaling is preserved
    diffs = quantized[0, 1:] - quantized[0, :-1]
    assert torch.allclose(diffs / diffs[0], torch.ones_like(diffs), rtol=0.1)


def test_round_to_nearest_pole():
    """Test rounding to nearest quantization level."""
    x = torch.tensor([-0.8, -0.3, 0.2, 0.7])
    poles = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
    rounded = round_to_nearest_pole(x, poles)
    expected = torch.tensor([-1.0, -0.5, 0.0, 0.5])
    assert torch.allclose(rounded, expected)


def test_normal_float_quantization():
    """Test normal float (NF4/NF3) quantization."""
    weight = torch.tensor([[-1.2, -0.6, 0.0, 0.6], 
                          [-0.3, 0.3, 0.9, 1.2]], dtype=torch.float32)
    
    # Test NF4 quantization
    quantized_nf4, _ = quantize_blockwise(
        weight, bits=4, blocksize=4, use_normal_float=True
    )
    assert quantized_nf4.shape == weight.shape
    # Check values are quantized to valid range
    assert torch.all(quantized_nf4 >= -1.2)
    assert torch.all(quantized_nf4 <= 1.2)
    
    # Test NF3 quantization
    quantized_nf3, _ = quantize_blockwise(
        weight, bits=3, blocksize=4, use_normal_float=True
    )
    assert quantized_nf3.shape == weight.shape
    # Check values are quantized to valid range
    assert torch.all(quantized_nf3 >= -1.2)
    assert torch.all(quantized_nf3 <= 1.2)
    
    # Invalid bits for normal float
    with pytest.raises(ValueError):
        quantize_blockwise(weight, bits=5, use_normal_float=True)


def test_scale_shift_quantization():
    """Test scale-shift quantization."""
    weight = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                          [5.0, 6.0, 7.0, 8.0]], dtype=torch.float32)
    
    # Test with scale_shift=True
    quantized_ss, _ = quantize_blockwise(
        weight, bits=4, blocksize=4, scale_shift=True
    )
    assert quantized_ss.shape == weight.shape
    
    # Values should maintain relative spacing
    diffs_orig = weight[0, 1:] - weight[0, :-1]
    diffs_quant = quantized_ss[0, 1:] - quantized_ss[0, :-1]
    assert torch.allclose(diffs_orig / diffs_orig[0], diffs_quant / diffs_quant[0], rtol=0.1)


def test_block_size_constraints():
    """Test block size handling."""
    # Test with block-aligned tensor
    weight = torch.randn(32, 32)  # 32x32 = 1024 elements
    quantized, _ = quantize_blockwise(weight, blocksize=16)  # 16x16 = 256 elements per block
    assert quantized.shape == weight.shape
    
    # Test with non-block-aligned tensor
    weight = torch.randn(30, 30)  # 900 elements
    quantized, _ = quantize_blockwise(weight, blocksize=16)
    assert quantized.shape == weight.shape
    
    # Test block percentage clipping with non-aligned tensor
    with pytest.raises(ValueError):
        quantize_blockwise(
            weight, blocksize=16, clip_method="block_percentage", clip_threshold=0.1
        )
