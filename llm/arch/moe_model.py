import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat
from typing import Optional, Tuple, List
from dataclasses import dataclass
from apex.normalization.fused_layer_norm import fused_rms_norm_affine
from kernel.linear_cross_entropy import fused_linear_cross_entropy
import math

# from tutel import moe as tutel_moe # python3 -m pip install --user --upgrade git+https://github.com/microsoft/tutel@v0.1.x

from arch.model import Model, ModelArgs, Block, FeedForward, Linear, WeightQuant, ActQuant, ActQuantInt4

from torchtitan.distributed.expert_parallel import expert_parallel
from typing import Literal
import torch
from torch import nn

_LOAD_BALANCING_LOSS = []


def save_load_balancing_loss(loss):
    global _LOAD_BALANCING_LOSS
    if not torch.isnan(loss).any():
        _LOAD_BALANCING_LOSS.append(loss)


def get_load_balancing_loss():
    global _LOAD_BALANCING_LOSS
    return _LOAD_BALANCING_LOSS


def clear_load_balancing_loss():
    global _LOAD_BALANCING_LOSS
    _LOAD_BALANCING_LOSS.clear()

@dataclass
class MoEModelArgs(ModelArgs):
    moe_freq: int = 1
    moe_setting_for_unimoe_exp: bool = False # the first 2 layers are dense layers, while the rest are moe layers
    moe_top_k: int = 2
    n_experts: int = 4
    # d_expert: int = 128
    w_gate_loss: float = 0.01
    moe_d_ffn: int = 2048
    use_gshared: bool = True


def _one_hot_with_dtype(data, num_classes, dtype, hot_value=1):
    result = torch.zeros([data.size(0), num_classes], device=data.device, dtype=dtype)
    result.scatter_(1, data.unsqueeze(-1), hot_value)
    return result


def gshard_loss(scores_w_noise, top_ids):
    num_samples, num_global_experts = int(scores_w_noise.size(0)), int(scores_w_noise.size(1))
    mask = _one_hot_with_dtype(top_ids[:, 0], num_global_experts, dtype=scores_w_noise.dtype,
        hot_value=num_global_experts / num_samples)
    me = torch.sum(scores_w_noise, dim=0)
    ce = torch.sum(mask, dim=0)
    l_aux = torch.sum(me * ce) / num_samples
    return l_aux
    

class MoELayer(nn.Module):
    
    def __init__(self, args: MoEModelArgs):
        super(MoELayer, self).__init__()
        self.args = args
        self.bitlinear = args.bitlinear
        self.router = TokenChoiceTopKRouter(
            dim = args.d_model,
            num_experts = args.n_experts,
            top_k = args.moe_top_k,
            score_func = "softmax",
            route_norm = True,
            route_scale = 1.0,
            use_gshared = args.use_gshared
        )
        self.experts = GroupedExperts(
            dim=args.d_model,
            hidden_dim=args.moe_d_ffn,
            num_experts=args.n_experts,
            use_grouped_mm=True,
            bitlinear=self.bitlinear,
        )
        self.shared_expert = None
        # self.all2all_size = 1
        # self.num_local_experts = self.args.n_experts // self.all2all_size

        self.score_before_experts = False
        if not args.use_gshared:
            self.load_balance_coeff = args.w_gate_loss
        else:
            self.load_balance_coeff = None
        if self.load_balance_coeff is not None:
            assert self.load_balance_coeff > 0.0
            self.register_buffer(
                "expert_bias",
                torch.zeros(args.n_experts, dtype=torch.float32),
            )
            self.register_buffer(
                "tokens_per_expert",
                torch.zeros(args.n_experts, dtype=torch.float32),
            )
        else:
            self.expert_bias = None

        with torch.no_grad():
            self.init_weights(0.02, 'cuda')


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor with shape ``(bs, slen, dim)``.

        Returns:
            out (torch.Tensor): Output tensor with shape ``(bs, slen, dim)``.
        """
        bs, slen, dim = x.shape

        # top_scores and selected_indices shape (bs*slen*top_k,)
        # num_tokens_per_expert shape (num_experts,)
        (
            top_scores,
            token_indices,
            num_tokens_per_expert,
            l_aux
        ) = self.router(x.reshape(bs * slen, dim), self.expert_bias)

        # tokens_per_expert will be used to update the expert bias for load balancing.
        # TODO: Activation Checkpointing has the side effect of double counting tokens_per_expert --
        #       first in the forward pass, and then in the backward pass. However, this has no
        #       effect on the expert bias update thanks to the torch.sign() operator.
        if self.load_balance_coeff is not None:
            with torch.no_grad():
                self.tokens_per_expert.add_(num_tokens_per_expert)

        # shape (bs*slen*top_k, dim)
        token_indices = token_indices.reshape(-1, 1).expand(-1, dim)

        # shape (bs*slen*top_k, dim)
        routed_input = torch.gather(
            x.view(-1, dim),
            dim=0,
            index=token_indices,
        )

        if self.score_before_experts:
            routed_input = (
                routed_input.to(torch.float32) * top_scores.reshape(-1, 1)
            ).to(x.dtype)

        # shape (bs*slen*top_k, dim)
        routed_output = self.experts(routed_input, num_tokens_per_expert)

        if not self.score_before_experts:
            routed_output = (
                routed_output.to(torch.float32) * top_scores.reshape(-1, 1)
            ).to(x.dtype)

        # shared expert
        if self.shared_expert is not None:
            out = self.shared_expert(x.reshape(1, bs * slen, dim)).reshape(
                bs * slen, dim
            )
        else:
            out = torch.zeros_like(x.reshape(bs * slen, dim))

        out = out.scatter_add(dim=0, index=token_indices, src=routed_output)
        out = out.reshape(bs, slen, dim)

        save_load_balancing_loss(l_aux)
        
        return out

    def init_weights(
        self,
        init_std: float,
        buffer_device: torch.device,
    ):
        self.experts.init_weights(init_std)
        self.router.init_weights(init_std)
        if self.shared_expert is not None:
            self.shared_expert.init_weights(init_std)

        if self.load_balance_coeff is not None:
            with torch.device(buffer_device):
                self.expert_bias = torch.zeros(
                    self.experts.num_experts, dtype=torch.float32
                )
                self.tokens_per_expert = torch.zeros(
                    self.experts.num_experts, dtype=torch.float32
                )

class GroupedExperts(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int,
        use_grouped_mm: bool,
        bitlinear: bool
    ):
        super().__init__()
        self.num_experts = num_experts
        self.w1 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.w2 = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
        self.w3 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.use_grouped_mm = use_grouped_mm
        self.bitlinear = bitlinear

    @staticmethod
    def _quantize_weight(weight: torch.Tensor) -> torch.Tensor:
        # if weight.dim() == 3:
        #     quantized_weights = []
        #     for i in range(weight.shape[0]):
        #         quantized_weights.append(WeightQuant.apply(weight[i]))
        #     return torch.stack(quantized_weights, dim=0)
        # else:
        #     return WeightQuant.apply(weight)
        return WeightQuant.apply(weight)
    
    @staticmethod
    def _quantize_activation(x: torch.Tensor) -> torch.Tensor:
        # if self.act_bits == 8:
        #     return ActQuant.apply(x)
        # elif self.act_bits == 4:
        #     return ActQuantInt4.apply(x)
        # else:
        #     raise ValueError(f"Unsupported act_bits: {self.act_bits}")
        return ActQuant.apply(x)

    def forward(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_grouped_mm:
            if self.bitlinear:
                return GroupedExperts._run_experts_grouped_mm_bitnet(
                    self.w1, self.w2, self.w3, x, num_tokens_per_expert
                )
            else:
                return GroupedExperts._run_experts_grouped_mm(
                    self.w1, self.w2, self.w3, x, num_tokens_per_expert
                )
        else:
            if self.bitlinear:
                return GroupedExperts._run_experts_for_loop_bitnet(
                    self.w1, self.w2, self.w3, x, num_tokens_per_expert
                )
            else:
                return GroupedExperts._run_experts_for_loop(
                    self.w1, self.w2, self.w3, x, num_tokens_per_expert
                )

    # TODO: keeping this for-loop implementation for comparison
    #       and readability, may remove later
    @expert_parallel
    @staticmethod
    def _run_experts_for_loop(
        w1: torch.Tensor,
        w2: torch.Tensor,
        w3: torch.Tensor,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if num_tokens_per_expert is not None:
            # NOTE: this would incur a synchronization between device and host
            num_tokens_per_expert = num_tokens_per_expert.tolist()

            # side-effect code due to the usage of generate_permute_indices
            num_padding = x.shape[0] - sum(num_tokens_per_expert)

            # a tuple of tensors indexed by experts
            # each with shape (tokens_per_expert(varying), dim)
            x = torch.split(
                x[: sum(num_tokens_per_expert)],
                split_size_or_sections=num_tokens_per_expert,
                dim=0,
            )
            out_experts_splits = []
            for expert_idx, x_expert in enumerate(x):
                h = F.silu(torch.matmul(x_expert, w1[expert_idx].transpose(-2, -1)))
                h = h * torch.matmul(x_expert, w3[expert_idx].transpose(-2, -1))
                h = torch.matmul(h, w2[expert_idx].transpose(-2, -1))
                # h shape (tokens_per_expert(varying), dim)
                out_experts_splits.append(h)
            out = torch.cat(out_experts_splits, dim=0)

            # side-effect code due to the usage of generate_permute_indices
            out = torch.vstack((out, out.new_zeros((num_padding, out.shape[-1]))))
        else:
            # x shape (num_experts, tokens_per_expert, dim)
            h = F.silu(torch.bmm(x, w1.transpose(-2, -1)))
            h = h * torch.bmm(x, w3.transpose(-2, -1))
            # out shape (num_experts, tokens_per_expert, dim)
            out = torch.bmm(h, w2.transpose(-2, -1))

        return out


    # TODO: keeping this for-loop implementation for comparison
    #       and readability, may remove later
    @expert_parallel
    @staticmethod
    def _run_experts_for_loop_bitnet(
        w1: torch.Tensor,
        w2: torch.Tensor,
        w3: torch.Tensor,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor | None = None,
    ) -> torch.Tensor:
        w1 = GroupedExperts._quantize_weight(w1)
        w2 = GroupedExperts._quantize_weight(w2)
        w3 = GroupedExperts._quantize_weight(w3)

        if num_tokens_per_expert is not None:
            # NOTE: this would incur a synchronization between device and host
            num_tokens_per_expert = num_tokens_per_expert.tolist()

            # side-effect code due to the usage of generate_permute_indices
            num_padding = x.shape[0] - sum(num_tokens_per_expert)

            # a tuple of tensors indexed by experts
            # each with shape (tokens_per_expert(varying), dim)
            x = torch.split(
                x[: sum(num_tokens_per_expert)],
                split_size_or_sections=num_tokens_per_expert,
                dim=0,
            )
            out_experts_splits = []
            for expert_idx, x_expert in enumerate(x):
                x_expert_q = GroupedExperts._quantize_activation(x_expert)
                h = F.silu(torch.matmul(x_expert_q, w1[expert_idx].transpose(-2, -1)))

                h = h * torch.matmul(x_expert_q, w3[expert_idx].transpose(-2, -1))

                h_q = GroupedExperts._quantize_activation(h)
                h = torch.matmul(h_q, w2[expert_idx].transpose(-2, -1))

                # h shape (tokens_per_expert(varying), dim)
                out_experts_splits.append(h)
            out = torch.cat(out_experts_splits, dim=0)

            # side-effect code due to the usage of generate_permute_indices
            out = torch.vstack((out, out.new_zeros((num_padding, out.shape[-1]))))
        else:
            # x shape (num_experts, tokens_per_expert, dim)

            x_q = GroupedExperts._quantize_activation(x)
            h = F.silu(torch.bmm(x_q, w1.transpose(-2, -1)))

            h = h * torch.bmm(x_q, w3.transpose(-2, -1))
            # out shape (num_experts, tokens_per_expert, dim)

            h_q = GroupedExperts._quantize_activation(h)
            out = torch.bmm(h_q, w2.transpose(-2, -1))

        return out


    @expert_parallel
    @staticmethod
    def _run_experts_grouped_mm(
        w1: torch.Tensor,
        w2: torch.Tensor,
        w3: torch.Tensor,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if num_tokens_per_expert is not None:
            offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
            # grouped mm between a 2D tensor and a 3D tensor
            assert x.dim() == 2
        else:
            offsets = None
            # fall back to regular bmm between 3D tensors
            assert x.dim() == 3

        h = F.silu(
            torch._grouped_mm(
                x.bfloat16(), w1.bfloat16().transpose(-2, -1), offs=offsets
            )
        )
        h = h * torch._grouped_mm(
            x.bfloat16(), w3.bfloat16().transpose(-2, -1), offs=offsets
        )
        out = torch._grouped_mm(
            h, w2.bfloat16().transpose(-2, -1), offs=offsets
        ).type_as(x)

        return out


    @expert_parallel
    @staticmethod
    def _run_experts_grouped_mm_bitnet(
        w1: torch.Tensor,
        w2: torch.Tensor,
        w3: torch.Tensor,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor | None = None,
    ) -> torch.Tensor:

        w1 = GroupedExperts._quantize_weight(w1)
        w2 = GroupedExperts._quantize_weight(w2)
        w3 = GroupedExperts._quantize_weight(w3)

        if num_tokens_per_expert is not None:
            offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
            # grouped mm between a 2D tensor and a 3D tensor
            assert x.dim() == 2
        else:
            offsets = None
            # fall back to regular bmm between 3D tensors
            assert x.dim() == 3
        
        x_q = GroupedExperts._quantize_activation(x)      

        h = F.silu(
            torch._grouped_mm(
                x_q.bfloat16(), w1.bfloat16().transpose(-2, -1), offs=offsets
            )
        )
        h = h * torch._grouped_mm(
            x_q.bfloat16(), w3.bfloat16().transpose(-2, -1), offs=offsets
        )

        h_q = GroupedExperts._quantize_activation(h)
        out = torch._grouped_mm(
            h_q, w2.bfloat16().transpose(-2, -1), offs=offsets
        ).type_as(x)

        return out

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.w1, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w2, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.w3, mean=0.0, std=init_std)


class TokenChoiceTopKRouter(nn.Module):
    """This class implements token-choice routing. In token-choice top-K routing, each token is
        routed to top K experts based on the router scores.

    Args:
        gate (nn.Module): Gate module to calculate the scores, typically nn.Linear(dim, num_experts).
        dim (int): Dimension of input tokens.
        num_experts (int): Number of experts in each moe layer.
        top_k (int): Number of experts each token will be routed to in token-choice routing.
        use_sigmoid (bool): Whether to use sigmoid or softmax for router scores. Default is False.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int,
        score_func: Literal["softmax", "sigmoid"],
        route_norm: bool,
        route_scale: float,
        use_gshared: bool,
    ):
        super().__init__()
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.num_experts = num_experts
        self.top_k = top_k
        self.score_func = score_func
        self.route_norm = route_norm
        self.route_scale = route_scale
        self.use_gshared = use_gshared

    def forward(
        self, x: torch.Tensor, expert_bias: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor with shape ``(bs*slen, dim)``.

        Returns:
            routed_input (torch.Tensor):
                Tokens grouped together by experts indices with shape ``(bs*slen*top_k,)``.
            token_indices (torch.Tensor):
                Token indices for routed_input with shape ``(bs*slen*top_k,)``.
            num_tokens_per_expert (torch.Tensor):
                Number of tokens assigned to each expert with shape ``(num_experts,)``.
        """
        # scores shape (bs*slen, num_experts)
        scores = self.gate(x)

        # By default, sigmoid or softmax is performed in float32 to avoid loss explosion
        if self.score_func == "sigmoid":
            scores = torch.sigmoid(scores.to(torch.float32))
        elif self.score_func == "softmax":
            scores = F.softmax(scores.to(torch.float32), dim=1)
        else:
            raise NotImplementedError(f"Unknown score function {self.score_function}")

        # top scores shape (bs*slen, top_k)
        # NOTE: The expert_bias is only used for routing. The gating value
        #       top_scores is still derived from the original scores.
        if expert_bias is not None:
            _, selected_experts_indices = torch.topk(
                scores + expert_bias, k=self.top_k, dim=1
            )
            top_scores = scores.gather(dim=1, index=selected_experts_indices)
        else:
            top_scores, selected_experts_indices = torch.topk(
                scores, k=self.top_k, dim=1
            )

        if self.use_gshared:
            l_aux = gshard_loss(scores, selected_experts_indices)
        else:
            l_aux = None

        if self.score_func == "sigmoid" and self.route_norm:
            denominator = top_scores.sum(dim=-1, keepdim=True) + 1e-20
            top_scores = top_scores / denominator
        top_scores = top_scores * self.route_scale

        # group tokens together by expert indices from 0 to num_experts and pass that to experts forward
        num_tokens_per_expert = torch.histc(
            selected_experts_indices.view(-1),
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        )

        # Reorder the token indices to match the order of the experts
        # token_indices_experts_sorted shape (bs*slen*top_k,)
        token_indices_experts_sorted = torch.argsort(
            selected_experts_indices.view(-1), stable=True
        )

        top_scores = top_scores.view(-1)[token_indices_experts_sorted]
        token_indices_experts_sorted = token_indices_experts_sorted // self.top_k

        return top_scores, token_indices_experts_sorted, num_tokens_per_expert, l_aux

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.gate.weight, mean=0.0, std=init_std)


class MoEBlock(Block):
    
    def __init__(self, args: MoEModelArgs):
        super(MoEBlock, self).__init__(args)
        self.feed_forward = MoELayer(args)
    
    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        start_pos: int = 0,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis=freqs_cis, start_pos=start_pos, kv_cache=kv_cache)
        # out = h + self.feed_forward(self.ffn_norm(h))
        out = self.ffn_norm(h + self.feed_forward(h))
        return out


class MoEModel(Model):

    def __init__(self, args: MoEModelArgs):
        super(MoEModel, self).__init__(args)

        self.layers = nn.ModuleList()

        if self.universal:
            assert args.moe_setting_for_unimoe_exp

            # the first two layers are dense layers
            for _ in range(2):
                self.layers.append(Block(args))

            for i in range(self.universal_n_group):
                group_layers = nn.ModuleList()
                for _ in range(args.universal_group_size):
                    group_layers.append(MoEBlock(args))
                self.layers.append(group_layers)
            
        else:
            if args.moe_setting_for_unimoe_exp:
                for _ in range(args.n_layers):
                    is_moe_layer = False if _ <= 1 else True
                    layer = MoEBlock(args) if is_moe_layer else Block(args)
                    self.layers.append(layer)
            else:
                for _ in range(args.n_layers):
                    is_moe_layer = getattr(args, 'moe_freq', 0) != 0 and (_ + 1) % args.moe_freq == 0
                    layer = MoEBlock(args) if is_moe_layer else Block(args)
                    self.layers.append(layer)


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
        
        # Compute the load balancing loss for all MoE layers.
        load_balancing_loss = get_load_balancing_loss()
        _moe_loss = -1
        if len(load_balancing_loss) > 0:
            moe_loss = sum(load_balancing_loss) * n_tokens.item()
            loss += self.args.w_gate_loss * moe_loss
            _moe_loss = moe_loss.item()
            clear_load_balancing_loss()

        return loss, n_tokens, _moe_loss

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

            # dense layer forward
            for i, layer in enumerate(self.layers[:2]):
                assert isinstance(layer, Block)
                h = layer(h, freqs_cis=freqs_cis, start_pos=start_pos, kv_cache=kv_cache[i] if kv_cache else None)
                i_abs += 1

            # universal moe layer forward
            for n in range(self.universal_n_group):
                group_layers = self.layers[2 + n]
                for r in range(self.universal_repeat_time):
                    for i, layer in enumerate(group_layers):
                        h = layer(h, freqs_cis=freqs_cis, start_pos=start_pos, kv_cache=kv_cache[i_abs] if kv_cache else None)
                        i_abs += 1
        else:
            for i, layer in enumerate(self.layers):
                h = layer(h, freqs_cis=freqs_cis, start_pos=start_pos, kv_cache=kv_cache[i] if kv_cache else None)

        h = self.norm(h)

        if targets is not None:
            loss, n_tokens, _moe_loss = self.compute_loss(h, targets, loss_mask)
            return loss, n_tokens, _moe_loss
        else:
            logits = self.output(h)


        return logits
