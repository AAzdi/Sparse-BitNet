"""
Custom CUDA/Triton kernels for efficient operations.
"""

from .linear_cross_entropy import fused_linear_cross_entropy

# Import mask creator with fallback
try:
    from .mask_creator_kernel import mask_creator_fast as mask_creator
    _USE_FAST_MASK_CREATOR = True
except ImportError:
    # Fallback to torchao if custom kernel not available
    try:
        from torchao.sparsity.utils import mask_creator
        _USE_FAST_MASK_CREATOR = False
    except ImportError:
        mask_creator = None
        _USE_FAST_MASK_CREATOR = False

__all__ = [
    'fused_linear_cross_entropy',
    'mask_creator',
]
