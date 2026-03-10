"""
High-performance N:M sparsity mask creator using Triton.

This module provides a fast CUDA implementation of N:M structured sparsity
mask generation, optimized for large tensors and frequent calls during training.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def mask_creator_kernel(
    # Pointers
    tensor_ptr,
    mask_ptr,
    # Tensor dimensions
    num_groups: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    # Block size for parallelization
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for creating N:M sparsity masks.
    
    For each group of M elements, keeps the top N elements by absolute value
    and zeros out the rest.
    """
    # Program ID
    pid = tl.program_id(0)
    
    # Each program handles BLOCK_SIZE groups
    group_start = pid * BLOCK_SIZE
    group_end = tl.minimum(group_start + BLOCK_SIZE, num_groups)
    
    # Process groups in this block
    for group_idx in range(group_start, group_end):
        # Load M elements for this group
        base_offset = group_idx * M
        offsets = base_offset + tl.arange(0, M)
        
        # Load values and compute absolute values
        values = tl.load(tensor_ptr + offsets)
        abs_values = tl.abs(values)
        
        # Find the (M-N)th largest value (threshold)
        # Elements >= this threshold will be kept
        # We need to find the Nth largest, which means M-N smallest should be masked
        
        # Sort by creating a mask for top N values
        # For each position, count how many values are larger
        mask_result = tl.zeros([M], dtype=tl.int32)
        
        for i in range(M):
            count = 0
            for j in range(M):
                # Count values that are strictly larger, or equal but with lower index (for stability)
                if abs_values[j] > abs_values[i]:
                    count += 1
                elif abs_values[j] == abs_values[i] and j < i:
                    count += 1
            
            # Keep this element if it's in top N
            if count < N:
                mask_result[i] = 1
        
        # Store mask
        tl.store(mask_ptr + offsets, mask_result.to(tl.float32))


@triton.jit
def mask_creator_kernel_optimized(
    # Pointers
    tensor_ptr,
    mask_ptr,
    # Tensor dimensions
    num_groups: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    # Block size for parallelization
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized Triton kernel for creating N:M sparsity masks using bitonic sort.
    
    This version is more efficient for larger M values by using parallel sorting.
    """
    pid = tl.program_id(0)
    
    # Number of groups this program handles
    num_groups_per_block = BLOCK_SIZE
    group_start = pid * num_groups_per_block
    
    # Process multiple groups if possible
    for local_idx in range(num_groups_per_block):
        group_idx = group_start + local_idx
        if group_idx >= num_groups:
            break
        
        base_offset = group_idx * M
        offsets = base_offset + tl.arange(0, M)
        
        # Load and get absolute values
        values = tl.load(tensor_ptr + offsets, mask=offsets < (num_groups * M), other=0.0)
        abs_values = tl.abs(values)
        
        # Create indices for sorting
        indices = tl.arange(0, M)
        
        # Bubble sort for small M (M=4 is very small)
        # For M=4, N=2, we need to find top 2 values
        # This is efficient enough for small M
        for i in range(M):
            for j in range(M - 1 - i):
                # Compare adjacent elements
                swap = abs_values[j] < abs_values[j + 1]
                # Swap values
                tmp_val = tl.where(swap, abs_values[j + 1], abs_values[j])
                abs_values[j + 1] = tl.where(swap, abs_values[j], abs_values[j + 1])
                abs_values[j] = tmp_val
                # Swap indices
                tmp_idx = tl.where(swap, indices[j + 1], indices[j])
                indices[j + 1] = tl.where(swap, indices[j], indices[j + 1])
                indices[j] = tmp_idx
        
        # Create mask: top N indices are 1, rest are 0
        mask_result = tl.zeros([M], dtype=tl.float32)
        for i in range(N):
            for j in range(M):
                if indices[j] == i:
                    mask_result[i] = 1.0
        
        # Actually, we need to set mask based on which indices were in top N
        # Re-approach: mark positions that correspond to top N values
        mask_result = tl.zeros([M], dtype=tl.float32)
        for pos in range(M):
            # Check if this position's index is in top N
            for top_i in range(N):
                if indices[top_i] == pos:
                    mask_result[pos] = 1.0
        
        tl.store(mask_ptr + offsets, mask_result, mask=offsets < (num_groups * M))


def mask_creator_triton(
    tensor: torch.Tensor,
    N: int = 2,
    M: int = 4,
) -> torch.Tensor:
    """
    Triton-accelerated N:M sparsity mask creator.
    
    Args:
        tensor: Input tensor to create mask for
        N: Number of weights to keep in each group of M
        M: Size of each weight group
    
    Returns:
        Binary mask tensor with same shape as input
    """
    if tensor.numel() % M != 0:
        raise ValueError(
            f"Tensor of size {tensor.shape} can't be evenly divided into {M} groups"
        )
    
    original_shape = tensor.shape
    num_groups = tensor.numel() // M
    
    # Flatten tensor for processing
    tensor_flat = tensor.detach().flatten()
    mask = torch.empty_like(tensor_flat)
    
    # Launch kernel
    # Choose block size based on M (each block processes multiple groups)
    BLOCK_SIZE = min(1024, num_groups)
    grid = (triton.cdiv(num_groups, BLOCK_SIZE),)
    
    # For small M (like 4), use simpler sorting approach
    if M <= 8:
        # Use a specialized kernel for small M
        mask_creator_kernel_small_m[grid](
            tensor_flat,
            mask,
            num_groups,
            M,
            N,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        mask_creator_kernel_optimized[grid](
            tensor_flat,
            mask,
            num_groups,
            M,
            N,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    return mask.reshape(original_shape)


@triton.jit
def mask_creator_kernel_small_m(
    tensor_ptr,
    mask_ptr,
    num_groups: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Highly optimized kernel for small M values (M <= 8).
    Each program handles one group of M elements.
    """
    pid = tl.program_id(0)
    
    if pid >= num_groups:
        return
    
    base_offset = pid * M
    offsets = base_offset + tl.arange(0, M)
    
    # Load M values for this group
    values = tl.load(tensor_ptr + offsets)
    abs_values = tl.abs(values)
    
    # For each element position, count how many elements have larger absolute values
    # We'll use broadcasting to compare all pairs simultaneously
    # Shape: [M] -> [M, 1] and [M] -> [1, M] for broadcasting
    abs_values_col = tl.reshape(abs_values, [M, 1])  # Column vector
    abs_values_row = tl.reshape(abs_values, [1, M])  # Row vector
    
    # Create comparison matrix: element [i,j] is True if abs_values[j] > abs_values[i]
    is_greater = abs_values_row > abs_values_col  # [M, M]
    
    # For ties, use index as tiebreaker (stable sort)
    is_equal = abs_values_row == abs_values_col
    indices_col = tl.reshape(tl.arange(0, M), [M, 1])
    indices_row = tl.reshape(tl.arange(0, M), [1, M])
    is_smaller_idx = indices_row < indices_col
    is_equal_and_smaller = is_equal & is_smaller_idx
    
    # Combine: count elements that are strictly greater OR equal but with smaller index
    should_count = is_greater | is_equal_and_smaller
    
    # Sum along row to get rank for each element
    ranks = tl.sum(should_count.to(tl.int32), axis=1)
    
    # Keep elements with rank < N
    mask_result = (ranks < N).to(tl.float32)
    
    tl.store(mask_ptr + offsets, mask_result)


@triton.jit
def mask_creator_kernel_2_4(
    tensor_ptr,
    mask_ptr,
    num_groups,
):
    """
    Specialized ultra-fast kernel for 2:4 sparsity (most common case).
    Each program handles one group of 4 elements.
    """
    pid = tl.program_id(0)
    
    if pid >= num_groups:
        return
    
    base_offset = pid * 4
    offsets = base_offset + tl.arange(0, 4)
    
    # Load 4 values
    v = tl.load(tensor_ptr + offsets)
    a = tl.abs(v)
    
    # Create comparison matrix using broadcasting
    # Reshape to enable pairwise comparison
    a_col = tl.reshape(a, [4, 1])
    a_row = tl.reshape(a, [1, 4])
    
    # Count how many values are strictly greater than each element
    is_greater = (a_row > a_col).to(tl.int32)
    ranks = tl.sum(is_greater, axis=1)
    
    # Handle ties: if equal, lower index wins
    is_equal = (a_row == a_col)
    idx_col = tl.reshape(tl.arange(0, 4), [4, 1])
    idx_row = tl.reshape(tl.arange(0, 4), [1, 4])
    is_smaller_idx = (idx_row < idx_col).to(tl.int32)
    tie_adjustment = tl.sum(is_equal.to(tl.int32) & is_smaller_idx, axis=1)
    
    ranks = ranks + tie_adjustment
    
    # Keep top 2 (rank < 2)
    mask = (ranks < 2).to(tl.float32)
    
    tl.store(mask_ptr + offsets, mask)


def mask_creator_triton_optimized(
    tensor: torch.Tensor,
    N: int = 2,
    M: int = 4,
) -> torch.Tensor:
    """
    Optimized Triton implementation with specialization for 2:4 sparsity.
    """
    if tensor.numel() % M != 0:
        raise ValueError(
            f"Tensor of size {tensor.shape} can't be evenly divided into {M} groups"
        )
    
    original_shape = tensor.shape
    num_groups = tensor.numel() // M
    
    # Flatten tensor
    tensor_flat = tensor.detach().flatten()
    mask = torch.empty_like(tensor_flat)
    
    # Use specialized kernel for 2:4 case
    if N == 2 and M == 4:
        grid = (num_groups,)
        mask_creator_kernel_2_4[grid](
            tensor_flat,
            mask,
            num_groups,
        )
    else:
        # Fall back to general kernel
        BLOCK_SIZE = 1
        grid = (num_groups,)
        mask_creator_kernel_small_m[grid](
            tensor_flat,
            mask,
            num_groups,
            M,
            N,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    return mask.reshape(original_shape)


# Export the default implementation (optimized version)
mask_creator = mask_creator_triton_optimized
mask_creator_triton_v2 = mask_creator_triton_optimized  # Alias for backward compatibility


# Fallback to PyTorch if Triton is not available
def mask_creator_pytorch(
    tensor: torch.Tensor,
    N: int = 2,
    M: int = 4,
) -> torch.Tensor:
    """
    PyTorch fallback implementation (original).
    """
    if tensor.numel() % M != 0:
        raise ValueError(
            f"Tensor of size {tensor.shape} can't be evenly divided into {M} groups"
        )

    num_groups = tensor.numel() // M

    # N:M sparsity for linear layers
    tensor_temp = tensor.detach().abs().reshape(num_groups, M)
    index = torch.argsort(tensor_temp, dim=1)[:, : int(M - N)]

    w_b = torch.ones(tensor_temp.shape, device=tensor_temp.device)
    mask = w_b.scatter_(dim=1, index=index, value=0).reshape(tensor.shape)

    return mask


# Auto-select implementation
try:
    import triton
    # Use Triton implementation
    mask_creator_fast = mask_creator_triton_optimized
except ImportError:
    print("Warning: Triton not available, falling back to PyTorch implementation")
    mask_creator_fast = mask_creator_pytorch
