import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat
from typing import Optional, Tuple, List
from dataclasses import dataclass
from apex.normalization.fused_layer_norm import fused_rms_norm_affine
from kernel.linear_cross_entropy import fused_linear_cross_entropy
import math

# Use optimized mask_creator from our kernel package
try:
    from kernel import mask_creator
    print("Using optimized Triton mask_creator")
except ImportError:
    # Fallback to torchao
    from torchao.sparsity.utils import mask_creator
    print("Warning: Using torchao mask_creator (slower)")

@dataclass
class ModelArgs:
    d_model: int = 4096
    d_ffn: int = 14336
    head: int = 32
    kv_head: int = 8
    n_layers: int = 32
    vocab_size: int = 128256
    max_seq_len: int = 4096
    norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    weight_tying: bool = False
    cross_entropy_chunk: int = 0
    bitlinear: bool = False
    srelu: bool = False
    a4: bool = False
    # universal settings
    universal_n_group: int = -1
    universal_group_size: int = -1
    universal_repeat_time: int = -1
    # N:M structured sparsity control
    use_weight_semi_sparse: bool = False
    sparse_n: int = 2  # N in N:M sparsity (number of non-zero values)
    sparse_m: int = 4  # M in N:M sparsity (block size)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.normalized_shape = torch.Size((dim,))

    # def _norm(self, x):
    #     return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    # def forward(self, x):
    #     output = self._norm(x.float()).type_as(x)
    #     return output * self.weight
    def forward(self, x):
        return fused_rms_norm_affine(x, self.weight, self.normalized_shape, self.eps)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

# @torch.compile
def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(0)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class ActQuant(torch.autograd.Function):

    @staticmethod
    @torch.compile
    def forward(ctx, x):
        dtype = x.dtype
        x = x.float()
        s = 127 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        x = (x * s).round().clamp(-128, 127) / s
        return x.to(dtype)
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

class ActQuantInt4(torch.autograd.Function):

    @staticmethod
    @torch.compile
    def forward(ctx, x):
        dtype = x.dtype
        x = x.float()
        s = math.sqrt(7) / x.abs().mean(dim=-1, keepdim=True).clamp_(min=1e-5)
        x = (x * s).round().clamp(-8, 7) / s
        return x.to(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

# class WeightQuant(torch.autograd.Function):

#     @staticmethod
#     @torch.compile
#     def forward(ctx, x):
#         dtype = x.dtype
#         x = x.float()
#         s = 1.0 / x.abs().mean().clamp_(min=1e-5)
#         x = (x * s).round().clamp(-1, 1) / s
#         return x.to(dtype)
    
#     @staticmethod
#     def backward(ctx, grad_output):
#         grad_input = grad_output.clone()
#         return grad_input

class WeightQuant(torch.autograd.Function):
    @staticmethod
    @torch.compile
    def forward(ctx, x):
        dtype = x.dtype
        x = x.float()
        
        if x.dim() == 3:
            # shape: [num_experts, hidden_dim, hidden_dim] -> [num_experts, 1, 1]
            s = 1.0 / x.abs().mean(dim=(1, 2), keepdim=True).clamp_(min=1e-5)
        else:
            # 2D tensor
            s = 1.0 / x.abs().mean().clamp_(min=1e-5)
        
        x = (x * s).round().clamp(-1, 1) / s
        return x.to(dtype)
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_output



class WeightQuantMasked(torch.autograd.Function):
    """
    Weight quantization (same strategy as WeightQuant), but additionally takes an n:m mask.
    In the forward pass, the mask is applied before quantization; in the backward pass,
    gradients are passed only to the retained positions.
    """
    @staticmethod
    @torch.compile
    def forward(ctx, x, mask):
        dtype = x.dtype

        x = x.float()
        mask = mask.to(x.dtype)
        
        if x.dim() == 3:
            s = 1.0 / x.abs().mean(dim=(1, 2), keepdim=True).clamp_(min=1e-5)
        else:
            s = 1.0 / x.abs().mean().clamp_(min=1e-5)
        x_masked = x * mask
        
        q = (x_masked * s).round().clamp(-1, 1) / s
        return q.to(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = grad_output
        return grad_x, None


class Sparsify(torch.autograd.Function):
    """
    Sparsification operation: applies mask in forward, allows all positions to receive
    gradients in backward. Same gradient behavior as WeightQuantMasked, but without quantization.
    """
    @staticmethod
    def forward(ctx, x, mask):
        dtype = x.dtype
        x = x.float()
        mask = mask.to(x.dtype)
        # Forward: apply mask
        x_masked = x * mask
        return x_masked.to(dtype)
    
    @staticmethod
    def backward(ctx, grad_output):
        # Backward: straight-through gradient, don't multiply by mask, allowing masked positions to receive gradients
        return grad_output, None




class BitLinear(nn.Linear):

    def __init__(self, in_features: int, out_features: int, split_size: List[int], bias: bool = True, act_bits: int = 8, use_weight_semi_sparse: bool = False, N: int = 2, M: int = 4):
        super(BitLinear, self).__init__(in_features, out_features, bias)
        self.split_size = split_size
        self.act_bits = act_bits
        self.use_weight_semi_sparse = use_weight_semi_sparse
        self.N = N
        self.M = M
        assert sum(split_size) == out_features
        # Initialize mask monitoring attributes to avoid memory leaks from undefined references
        self._save_mask_cache = False
        self._model_mask_cache = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        
        if self.use_weight_semi_sparse:
            # Use the globally imported mask_creator (optimized version)
            mask = mask_creator(self.weight, self.N, self.M).to(weight.dtype)
            # Cache the full mask to the monitoring system (before splitting)
            if (hasattr(self, '_save_mask_cache') and self._save_mask_cache and
                hasattr(self, '_model_mask_cache') and self._model_mask_cache is not None):
                layer_name = getattr(self, '_layer_name', f'BitLinear_{id(self)}')
                # Store as bool type for flip_rate calculation
                self._model_mask_cache[layer_name] = mask.clone().detach().bool()
        
        # Weight quantization
        if len(self.split_size) == 1:
            if self.use_weight_semi_sparse:
                weight = WeightQuantMasked.apply(weight, mask)
            else:
                weight = WeightQuant.apply(weight)
        else:
            parts = torch.split(weight, self.split_size, dim=0)
            if self.use_weight_semi_sparse:
                mask_parts = torch.split(mask, self.split_size, dim=0)
                quanted_parts = [WeightQuantMasked.apply(w, m) for w, m in zip(parts, mask_parts)]
            else:
                quanted_parts = [WeightQuant.apply(w) for w in parts]
            weight = torch.cat(quanted_parts, dim=0)
    

        
        # quant activation
        if self.act_bits == 8:
            input = ActQuant.apply(x)
        elif self.act_bits == 4:
            input = ActQuantInt4.apply(x)
        else:
            raise ValueError(f"Unsupported act_bits: {self.act_bits}")
        
        return F.linear(input, weight, self.bias)


class SparseLinear(nn.Linear):
    """
    Sparse linear layer that applies N:M structured sparsity without quantization.
    Used for enabling sparsity when training non-BitNet models.
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, 
                 use_weight_semi_sparse: bool = True, N: int = 2, M: int = 4):
        super(SparseLinear, self).__init__(in_features, out_features, bias)
        self.use_weight_semi_sparse = use_weight_semi_sparse
        self.N = N
        self.M = M
        # Initialize mask monitoring attributes
        self._save_mask_cache = False
        self._model_mask_cache = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        
        if self.use_weight_semi_sparse:
            # Use the globally imported mask_creator (optimized version)
            mask = mask_creator(self.weight, self.N, self.M).to(weight.dtype)
            
            # Cache mask to the monitoring system
            if (hasattr(self, '_save_mask_cache') and self._save_mask_cache and
                hasattr(self, '_model_mask_cache') and self._model_mask_cache is not None):
                layer_name = getattr(self, '_layer_name', f'SparseLinear_{id(self)}')
                self._model_mask_cache[layer_name] = mask.clone().detach().bool()
            
            # Apply sparse mask using Sparsify (maintaining same gradient behavior as WeightQuantMasked)
            weight = Sparsify.apply(weight, mask)
        
        return F.linear(x, weight, self.bias)

def Linear(args, in_features: int, out_features: int, split_size: List[int], bias: bool = True, a4: bool = False, ffn: bool = False):
    # if not ffn:
    #     return nn.Linear(in_features, out_features, bias)
        
    if args.bitlinear:
        assert split_size is not None
        return BitLinear(
            in_features,
            out_features,
            split_size,
            bias,
            act_bits=4 if a4 else 8,
            use_weight_semi_sparse=getattr(args, "use_weight_semi_sparse", False),
            N=getattr(args, "sparse_n", 2),
            M=getattr(args, "sparse_m", 4),
        )
    elif getattr(args, "use_weight_semi_sparse", False):
        # Non-BitNet model with sparsity enabled: use SparseLinear
        return SparseLinear(
            in_features,
            out_features,
            bias,
            use_weight_semi_sparse=True,
            N=getattr(args, "sparse_n", 2),
            M=getattr(args, "sparse_m", 4),
        )
    else:
        return nn.Linear(in_features, out_features, bias)

class Attention(nn.Module):

    def __init__(self, args: ModelArgs):
        super(Attention, self).__init__()
        self.args = args
        self.head_dim = args.d_model // args.head
        self.kv_head = args.kv_head
        self.head = args.head
        # self.wqkv = nn.Linear(args.d_model, args.d_model + 2 * self.kv_head * self.head_dim, bias=False)
        self.wqkv = Linear(
            args, 
            args.d_model, args.d_model + 2 * self.kv_head * self.head_dim, 
            [args.d_model, self.kv_head * self.head_dim, self.kv_head * self.head_dim],
            bias=False,
            a4=args.a4,
        )
        # self.wo = nn.Linear(args.d_model, args.d_model, bias=False)
        self.wo = Linear(
            args,
            args.d_model, args.d_model,
            [args.d_model],
            bias=False
        )
        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(
        self,
        state_dict,
        prefix,
        *args,
        **kwargs,
    ):
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        start_pos: int = 0,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seqlen = x.shape[1]
        xqkv = self.wqkv(x)
        xq, xkv = xqkv[:, :, :self.args.d_model], xqkv[:, :, self.args.d_model:]
        xk, xv = xkv.chunk(2, -1)
        xq, xk, xv = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', d=self.head_dim), (xq, xk, xv))
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k_cache[:, :, start_pos : start_pos + seqlen] = xk
            v_cache[:, :, start_pos : start_pos + seqlen] = xv
            xk = k_cache[:, :, :start_pos + seqlen]
            xv = v_cache[:, :, :start_pos + seqlen]

        xk, xv = map(lambda t: repeat(t, 'b h n d -> b (h kv) n d', kv=self.head // self.kv_head), (xk, xv))

        with torch.backends.cuda.sdp_kernel(enable_math=False):
            output = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True if start_pos == 0 else False)

        output = rearrange(output, 'b h n d -> b n (h d)')
        output = self.wo(output)

        return output

@torch.compile
def squared_relu(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x) ** 2

class FeedForward(nn.Module):

    def __init__(self, args: ModelArgs, is_moe_layer: bool = False):
        super(FeedForward, self).__init__()
        self.args = args
        # self.w13 = nn.Linear(args.d_model, args.d_ffn * 2, bias=False)
        if is_moe_layer:
            self.w13 = Linear(
                args,
                args.d_model // args.mhmoe_heads_number, args.moe_d_ffn * args.mhmoe_heads_number * 2,
                [args.moe_d_ffn * args.mhmoe_heads_number, args.moe_d_ffn * args.mhmoe_heads_number],
                bias=False,
                a4=args.a4,
                ffn=True
            )
            # self.w2 = nn.Linear(args.d_ffn, args.d_model, bias=False)
            self.w2 = Linear(
                args,
                args.moe_d_ffn * args.mhmoe_heads_number, args.d_model // args.mhmoe_heads_number, 
                [args.d_model // args.mhmoe_heads_number],
                bias=False,
                ffn=True
            )
        else:
            self.w13 = Linear(
                args,
                args.d_model, args.d_ffn * 2,
                [args.d_ffn, args.d_ffn],
                bias=False,
                a4=args.a4,
                ffn=True
            )
            # self.w2 = nn.Linear(args.d_ffn, args.d_model, bias=False)
            self.w2 = Linear(
                args,
                args.d_ffn, args.d_model, 
                [args.d_model],
                bias=False,
                ffn=True
            )
        self.act_func = squared_relu if args.srelu else F.silu
        self._register_load_state_dict_pre_hook(self.load_hook)
    
    def load_hook(
        self,
        state_dict,
        prefix,
        *args,
        **kwargs,
    ):
        if prefix + "w1.weight" in state_dict:
            w1 = state_dict.pop(prefix + "w1.weight")
            w3 = state_dict.pop(prefix + "w3.weight")
            state_dict[prefix + "w13.weight"] = torch.cat([w1, w3])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x13 = self.w13(x)
        x1, x3 = x13.chunk(2, -1)
        x1 = F.silu(x1) * x3
        output = self.w2(x1)
        return output

class Block(nn.Module):

    def __init__(self, args: ModelArgs):
        super(Block, self).__init__()
        self.args = args
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args)
        self.attention_norm = RMSNorm(args.d_model, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.d_model, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        start_pos: int = 0,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis=freqs_cis, start_pos=start_pos, kv_cache=kv_cache)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Model(nn.Module):

    def __init__(self, args: ModelArgs):
        super(Model, self).__init__()
        self.args = args
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.d_model)
        nn.init.normal_(self.tok_embeddings.weight, mean=0, std=args.d_model ** -0.5)

        self.universal = True if self.args.universal_repeat_time > 0 else False
        self.universal_n_group = self.args.universal_n_group
        self.universal_group_size = self.args.universal_group_size
        self.universal_repeat_time = self.args.universal_repeat_time
        self.layers = nn.ModuleList()
        if self.universal:
            for i in range(self.universal_n_group):
                group_layers = nn.ModuleList()
                for _ in range(args.universal_group_size):
                    group_layers.append(Block(args))
                self.layers.append(group_layers)
        else:
            for _ in range(args.n_layers):
                self.layers.append(Block(args))
        
        self.norm = RMSNorm(args.d_model, eps=args.norm_eps)
        self.output = nn.Linear(args.d_model, args.vocab_size, bias=False)

        if args.weight_tying:
            self.output.weight = self.tok_embeddings.weight
        
        self.freqs_cis = precompute_freqs_cis(args.d_model // args.head, args.max_seq_len, theta=args.rope_theta)
        
        # Initialize sparse mask cache system
        self.mask_cache = {}
        self._mask_monitoring_enabled = False
        self._setup_sparse_layer_names()
    
    def _setup_sparse_layer_names(self):
        """Set unique layer names for all sparse layers, used for mask caching"""
        layer_counter = 0
        
        def setup_layer(module, prefix=""):
            nonlocal layer_counter
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                if isinstance(child, (BitLinear, SparseLinear)) and hasattr(child, 'use_weight_semi_sparse') and child.use_weight_semi_sparse:
                    # Set layer name and cache reference
                    child._layer_name = f"{full_name}_layer_{layer_counter}"
                    child._model_mask_cache = self.mask_cache
                    child._save_mask_cache = False  # Disabled by default, enabled via enable_mask_monitoring
                    layer_counter += 1
                setup_layer(child, full_name)
        
        setup_layer(self)
    
    def enable_mask_monitoring(self):
        """Enable sparse mask monitoring"""
        self._mask_monitoring_enabled = True
        
        def enable_layer_monitoring(module, prefix=""):
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                if (isinstance(child, (BitLinear, SparseLinear)) and 
                    hasattr(child, 'use_weight_semi_sparse') and child.use_weight_semi_sparse):
                    child._save_mask_cache = True
                enable_layer_monitoring(child, full_name)
        
        enable_layer_monitoring(self)
    
    def disable_mask_monitoring(self):
        """Disable sparse mask monitoring but keep the currently cached masks"""
        self._mask_monitoring_enabled = False
        
        def disable_layer_monitoring(module):
            for child in module.children():
                # Ensure all BitLinear and SparseLinear layers are properly disabled
                if isinstance(child, (BitLinear, SparseLinear)):
                    child._save_mask_cache = False
                disable_layer_monitoring(child)
        
        disable_layer_monitoring(self)
        # Note: don't clear mask_cache, let the training loop collect and clean up manually
    
    def get_current_masks(self):
        """Get all currently cached masks"""
        return dict(self.mask_cache)
    
    def compute_loss(
        self,
        hidden: torch.Tensor,
        targets: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.args.cross_entropy_chunk == 0:
            logits = self.output(hidden)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1), 
                reduction='sum', 
                ignore_index=self.args.vocab_size-1
            )
        else:
            loss = fused_linear_cross_entropy(
                hidden.view(-1, hidden.size(-1)), 
                self.output.weight, 
                targets.view(-1), 
                self.args.cross_entropy_chunk, 
                self.args.vocab_size-1, 
                'sum'
            )
        n_tokens = loss_mask.sum()
        return loss, n_tokens

    def forward(
        self,
        tokens: torch.Tensor,
        start_pos: int = 0,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        targets: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.tok_embeddings(tokens)
        freqs_cis = self.freqs_cis.to(h.device)[start_pos : start_pos + h.shape[1]]

        if self.universal:
            i_abs = 0
            for n in range(self.universal_n_group):
                group_layers = self.layers[n]
                for r in range(self.universal_repeat_time):
                    for i, layer in enumerate(group_layers):
                        h = layer(h, freqs_cis=freqs_cis, start_pos=start_pos, kv_cache=kv_cache[i_abs] if kv_cache else None)
                        i_abs += 1
        else:
            for i, layer in enumerate(self.layers):
                h = layer(h, freqs_cis=freqs_cis, start_pos=start_pos, kv_cache=kv_cache[i] if kv_cache else None)

        h = self.norm(h)

        if targets is not None:
            loss, n_tokens = self.compute_loss(h, targets, loss_mask)
            return loss, n_tokens, -1
        else:
            logits = self.output(h)

        return logits


def create_kv_cache(args: ModelArgs, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    kv_cache = []
    for _ in range(args.n_layers):
        k_cache = torch.zeros(batch_size, args.kv_head, args.max_seq_len, args.d_model // args.head, device='cuda')
        v_cache = torch.zeros(batch_size, args.kv_head, args.max_seq_len, args.d_model // args.head, device='cuda')
        kv_cache.append((k_cache, v_cache))
    return kv_cache
