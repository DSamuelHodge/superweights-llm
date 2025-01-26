"""Weight quantization tools."""

from typing import Tuple, Optional
import torch
import numpy as np
from .utils import quantize_tensor


def round_to_nearest_pole(x: torch.Tensor, poles: torch.Tensor) -> torch.Tensor:
    """Round tensor values to nearest quantization level.
    
    Args:
        x: Input tensor
        poles: Tensor of quantization levels
        
    Returns:
        Tensor with values rounded to nearest pole
    """
    differences = torch.abs(x.unsqueeze(-1) - poles)
    nearest_indices = torch.argmin(differences, dim=-1)
    return poles[nearest_indices]


def quantize_blockwise(
    weight: torch.Tensor,
    bits: int = 4,
    blocksize: int = 16,
    clip_method: str = "no", 
    clip_threshold: float = 0.1,
    scale_shift: bool = False,
    use_normal_float: bool = False
) -> Tuple[torch.Tensor, int]:
    """Quantize weights in blocks using specified number of bits.
    
    Args:
        weight: Input tensor to quantize
        bits: Number of bits for quantization
        blocksize: Size of blocks for blockwise quantization
        clip_method: Method for outlier clipping ('no', 'block_percentage', 'global_percentage', 'tensor_percentage', 'zscore', 'iqr')
        clip_threshold: Threshold for outlier clipping
        scale_shift: Whether to use scale-shift quantization
        use_normal_float: Whether to use normal float values
        
    Returns:
        Tuple of (quantized tensor, number of outliers)
        
    Raises:
        ValueError: If clip_method is invalid or parameters are out of range
    """
    if bits <= 0 or bits > 8:
        raise ValueError("Number of bits must be between 1 and 8")
    if blocksize <= 0:
        raise ValueError("Block size must be positive")
    if len(weight.shape) != 2:
        raise ValueError("Input tensor must be 2D")
        
    shape = weight.shape
    dtype = weight.dtype
    num_outliers = 0
    
    # Handle outlier clipping
    if clip_method != "no":
        if clip_method == "block_percentage":
            if weight.numel() % blocksize != 0:
                raise ValueError("Tensor size must be multiple of blocksize for block_percentage method")
                
            # Reshape into blocks and find threshold per block
            rows, cols = shape
            num_blocks_row = rows // blocksize
            num_blocks_col = cols // blocksize
            weight_blocks = weight.reshape(num_blocks_row, blocksize, num_blocks_col, blocksize)
            weight_blocks = weight_blocks.permute(0, 2, 1, 3).reshape(-1, blocksize * blocksize)
            
            # Find threshold for each block
            num_top_elements = max(int(blocksize * blocksize * clip_threshold + 1), 1)
            thresholds = torch.topk(weight_blocks.abs(), num_top_elements, dim=1).values[:, -1]
            
            # Create threshold mask matching original shape
            block_mask = (weight_blocks.abs() > thresholds.unsqueeze(1))
            block_mask = block_mask.reshape(num_blocks_row, num_blocks_col, blocksize, blocksize)
            block_mask = block_mask.permute(0, 2, 1, 3).reshape(rows, cols)
            
            # Apply threshold
            num_outliers = int(block_mask.sum().item())
            weight = torch.where(block_mask, 
                               torch.sign(weight) * thresholds.reshape(num_blocks_row, num_blocks_col).repeat_interleave(blocksize, dim=0).repeat_interleave(blocksize, dim=1),
                               weight)
            
        elif clip_method in ["tensor_percentage", "global_percentage"]:
            num_top_elements = max(int(weight.numel() * clip_threshold), 1)
            threshold = torch.topk(weight.view(-1).abs(), num_top_elements).values[-1]
            num_outliers = int(torch.sum(weight.abs() > threshold))
            weight = torch.clamp(weight, -threshold, threshold)
            
        elif clip_method == "zscore":
            mean = weight.abs().mean()
            std = weight.abs().std()
            threshold = clip_threshold * std + mean
            num_outliers = int(torch.sum(weight.abs() > threshold))
            weight = torch.clamp(weight, -threshold, threshold)
            
        elif clip_method == "iqr":
            q1 = weight.abs().float().quantile(0.25)
            q3 = weight.abs().float().quantile(0.75)
            threshold = q3 + clip_threshold * (q3 - q1)
            threshold = threshold.to(weight.dtype)
            num_outliers = int(torch.sum(weight.abs() > threshold))
            weight = torch.clamp(weight, -threshold, threshold)
            
        else:
            raise ValueError(f"Unknown clip method: {clip_method}")

    # Reshape for block processing if possible
    if weight.numel() % (blocksize * blocksize) == 0:
        weight = weight.reshape(-1, blocksize * blocksize)
        minima, _ = weight.min(dim=1, keepdims=True)
        maxima, _ = weight.max(dim=1, keepdims=True)
    else:
        minima = weight.min()
        maxima = weight.max()

    if use_normal_float:
        # Normal float quantization levels
        NF4_LEVELS = [
            -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
            -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
            0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
            0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
            0.7229568362236023, 1.0
        ]
        NF3_LEVELS = [
            -1, -0.5350227355957031, -0.2469314038753510, 0,
            0.1833375245332718, 0.3819939494132996, 0.6229856610298157, 1
        ]
        
        if bits == 4:
            quantization_levels = NF4_LEVELS
        elif bits == 3:
            quantization_levels = NF3_LEVELS
        else:
            raise ValueError("Normal Float Quantization only supports 4 and 3 bits")
            
        quantization_levels = torch.tensor(quantization_levels, device=weight.device)
        scale = 2 / (maxima - minima)  # scale to [0, 2]
        weight = weight.sub(minima).mul(scale).sub(1.0)  # shift to [-1, 1]
        weight = round_to_nearest_pole(weight, quantization_levels)
        weight = weight.add(1).div(scale).add(minima)
        
    else:
        if scale_shift:
            # Map to [-0.4999, 15.4999] for better rounding
            scale = (2 ** bits - 0.01) / (maxima - minima)
            weight = weight.sub(minima).mul(scale).sub(0.49)
            weight = weight.round()
            weight = weight.add(0.49).div(scale).add(minima)
        else:
            # Standard linear quantization
            scale = (2 ** bits - 1) / (maxima - minima)
            weight = weight.sub(minima).mul(scale)
            weight = weight.round()
            weight = weight.div(scale).add(minima)

    # Reshape back to original shape
    weight = weight.reshape(shape)
    return weight.to(dtype), num_outliers


def pack_4bit_to_int8(values: torch.Tensor) -> torch.Tensor:
    """Pack two 4-bit values into one 8-bit value.
    
    Args:
        values: Tensor of 4-bit values (0-15)
        
    Returns:
        Tensor of packed 8-bit values
        
    Raises:
        ValueError: If input tensor has odd number of elements or values > 15
    """
    if values.numel() % 2 != 0:
        raise ValueError("Number of values must be even")
    if torch.any(values > 15):
        raise ValueError("Values must be 4-bit (0-15)")
        
    values = values.to(torch.uint8).flatten()
    # Pack pairs of values into bytes
    packed = torch.empty(values.numel() // 2, dtype=torch.uint8, device=values.device)
    packed = (values[::2] << 4) | values[1::2]
    return packed


def unpack_int8_to_4bit(packed: torch.Tensor, original_shape: torch.Size) -> torch.Tensor:
    """Unpack 8-bit values into pairs of 4-bit values.
    
    Args:
        packed: Tensor of packed 8-bit values
        original_shape: Shape of original 4-bit tensor
        
    Returns:
        Tensor of unpacked 4-bit values
        
    Raises:
        ValueError: If shape is invalid or dtype is not uint8
    """
    if packed.dtype != torch.uint8:
        raise ValueError("Packed tensor must be uint8")
    if original_shape.numel() != packed.numel() * 2:
        raise ValueError("Invalid shape for unpacking")
        
    # Create output tensor
    unpacked = torch.empty(packed.numel() * 2, dtype=torch.uint8, device=packed.device)
    # Extract high and low nibbles
    unpacked[::2] = (packed >> 4) & 0xF
    unpacked[1::2] = packed & 0xF
    return unpacked.reshape(original_shape)


def find_outliers(
    tensor: torch.Tensor,
    method: str = "block_percentage",
    threshold: float = 0.1,
    blocksize: int = 16
) -> torch.Tensor:
    """Find outliers in a tensor using various methods.
    
    Args:
        tensor: Input tensor
        method: Method to find outliers ('block_percentage', 'global_percentage', 'absolute')
        threshold: Threshold for outlier detection
        blocksize: Block size for block_percentage method
        
    Returns:
        Boolean mask of same shape as input tensor, True for outliers
        
    Raises:
        ValueError: If method is invalid or parameters are out of range
    """
    if method not in ["block_percentage", "global_percentage", "absolute"]:
        raise ValueError("Invalid outlier detection method")
    if threshold <= 0:
        raise ValueError("Threshold must be positive")
    if method == "block_percentage" and blocksize <= 0:
        raise ValueError("Block size must be positive")
        
    if method == "block_percentage":
        # Reshape tensor into blocks
        if len(tensor.shape) != 2:
            raise ValueError("Block percentage method requires 2D tensor")
            
        rows, cols = tensor.shape
        pad_rows = (blocksize - rows % blocksize) % blocksize
        pad_cols = (blocksize - cols % blocksize) % blocksize
        padded = torch.nn.functional.pad(tensor, (0, pad_cols, 0, pad_rows))
        
        blocks = padded.unfold(0, blocksize, blocksize).unfold(1, blocksize, blocksize)
        
        # Find outliers in each block
        outlier_mask = torch.zeros_like(padded, dtype=torch.bool)
        for i in range(blocks.size(0)):
            for j in range(blocks.size(1)):
                block = blocks[i, j]
                abs_block = torch.abs(block)
                k = int(threshold * blocksize * blocksize)
                if k > 0:
                    threshold_value = torch.topk(abs_block.flatten(), k)[0][-1]
                    block_mask = abs_block >= threshold_value
                    outlier_mask[i*blocksize:(i+1)*blocksize, 
                               j*blocksize:(j+1)*blocksize] = block_mask
                    
        # Remove padding
        return outlier_mask[:rows, :cols]
        
    elif method == "global_percentage":
        abs_tensor = torch.abs(tensor)
        k = int(threshold * tensor.numel())
        if k > 0:
            threshold_value = torch.topk(abs_tensor.flatten(), k)[0][-1]
            return abs_tensor >= threshold_value
        return torch.zeros_like(tensor, dtype=torch.bool)
        
    else:  # absolute threshold
        return torch.abs(tensor) >= threshold


def clip_outliers(
    tensor: torch.Tensor,
    outlier_mask: torch.Tensor,
    method: str = "zero"
) -> torch.Tensor:
    """Clip outliers in a tensor using various methods.
    
    Args:
        tensor: Input tensor
        outlier_mask: Boolean mask indicating outlier positions
        method: Method to clip outliers ('zero' or 'mean')
        
    Returns:
        Tensor with outliers clipped
        
    Raises:
        ValueError: If method is invalid or shapes don't match
    """
    if method not in ["zero", "mean"]:
        raise ValueError("Invalid clipping method")
    if tensor.shape != outlier_mask.shape:
        raise ValueError("Tensor and mask shapes must match")
        
    clipped = tensor.clone()
    if method == "zero":
        clipped[outlier_mask] = 0
    else:  # mean
        mean_value = tensor[~outlier_mask].mean()
        clipped[outlier_mask] = mean_value
        
    return clipped


def scale_superweights(
    tensor: torch.Tensor,
    superweight_mask: torch.Tensor,
    scale_factor: float = 1.0,
    per_block: bool = False,
    blocksize: int = 16
) -> torch.Tensor:
    """Scale identified superweights by a specified factor.
    
    Args:
        tensor: Input tensor containing weights
        superweight_mask: Boolean mask indicating superweight positions
        scale_factor: Factor to scale superweights by
        per_block: Whether to apply scaling per block
        blocksize: Block size when using per_block=True
        
    Returns:
        Tensor with scaled superweights
    """
    scaled = tensor.clone()
    if per_block and len(tensor.shape) == 2:
        rows, cols = tensor.shape
        # Pad if necessary
        pad_rows = (blocksize - rows % blocksize) % blocksize
        pad_cols = (blocksize - cols % blocksize) % blocksize
        if pad_rows > 0 or pad_cols > 0:
            scaled = torch.nn.functional.pad(scaled, (0, pad_cols, 0, pad_rows))
            superweight_mask = torch.nn.functional.pad(superweight_mask, (0, pad_cols, 0, pad_rows))
        
        # Process blocks
        for i in range(0, scaled.shape[0], blocksize):
            for j in range(0, scaled.shape[1], blocksize):
                block = scaled[i:i+blocksize, j:j+blocksize]
                mask = superweight_mask[i:i+blocksize, j:j+blocksize]
                if mask.any():
                    # Calculate original mean
                    original_mean = block.mean()
                    
                    # Scale superweights
                    block[mask] *= scale_factor
                    
                    # Adjust non-superweights to preserve mean if there are any
                    if (~mask).any():
                        # Calculate current weighted sum
                        super_sum = block[mask].sum()
                        non_super_count = (~mask).sum()
                        
                        # Solve for x: (super_sum + x * non_super_count) / total_count = original_mean
                        # where x is the value for non-superweights
                        total_count = block.numel()
                        target_sum = original_mean * total_count
                        non_super_value = (target_sum - super_sum) / non_super_count
                        block[~mask] = non_super_value
        
        # Remove padding if added
        if pad_rows > 0 or pad_cols > 0:
            scaled = scaled[:rows, :cols]
    else:
        scaled[superweight_mask] *= scale_factor
        
    return scaled


def quantize_with_superweights(
    weight: torch.Tensor,
    bits: int = 4,
    blocksize: int = 16,
    superweight_threshold: float = 0.1,
    superweight_method: str = "block_percentage",
    preserve_superweights: bool = True,
    scale_factor: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Quantize weights while handling superweights specially.
    
    Args:
        weight: Input tensor to quantize
        bits: Number of bits for quantization
        blocksize: Size of blocks for blockwise quantization
        superweight_threshold: Threshold for superweight detection
        superweight_method: Method to detect superweights
        preserve_superweights: Whether to preserve original superweight values
        scale_factor: Factor to scale superweights by if not preserving originals
        
    Returns:
        Tuple of (quantized tensor, superweight mask, number of superweights)
    """
    if bits <= 0 or bits > 8:
        raise ValueError("Number of bits must be between 1 and 8")
    if blocksize <= 0:
        raise ValueError("Block size must be positive")
    if superweight_method not in ["block_percentage", "global_percentage", "absolute"]:
        raise ValueError("Invalid superweight detection method")
        
    # Find superweights
    superweight_mask = find_outliers(
        weight,
        method=superweight_method,
        threshold=superweight_threshold,
        blocksize=blocksize
    )
    num_superweights = int(superweight_mask.sum().item())
    
    if preserve_superweights:
        # Quantize non-superweights
        quantized = weight.clone()
        non_super_mask = ~superweight_mask
        if non_super_mask.any():
            non_super_weights = weight[non_super_mask]
            quantized_weights, _ = quantize_blockwise(
                non_super_weights,
                bits=bits,
                blocksize=blocksize
            )
            quantized[non_super_mask] = quantized_weights
    else:
        # Quantize all weights
        quantized, _ = quantize_blockwise(weight, bits=bits, blocksize=blocksize)
        # Scale superweights
        if num_superweights > 0:
            quantized = scale_superweights(
                quantized,
                superweight_mask,
                scale_factor=scale_factor,
                per_block=True,
                blocksize=blocksize
            )
    
    return quantized, superweight_mask, num_superweights


def restore_original_weights(
    quantized_tensor: torch.Tensor,
    original_tensor: torch.Tensor,
    superweight_mask: torch.Tensor
) -> torch.Tensor:
    """Restore original weight values for superweights while keeping other weights quantized.
    
    Args:
        quantized_tensor: Tensor with quantized weights
        original_tensor: Tensor with original weights
        superweight_mask: Boolean mask indicating superweight positions
        
    Returns:
        Tensor with restored superweights
    """
    restored = quantized_tensor.clone()
    restored[superweight_mask] = original_tensor[superweight_mask]
    return restored
