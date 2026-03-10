import torch
import os
import math
import time
import json
import subprocess
from dataclasses import dataclass, asdict
from typing import Optional, Tuple
# from arch.model import ModelArgs, Model
from arch.moe_model import Model, ModelArgs, MoEModel, MoEModelArgs
from data.lm_loader import LMLoader, DataLoaderArgs
from log import Logger
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch import distributed as dist
from torch.distributed.optim import ZeroRedundancyOptimizer
from config import parse_training_args, TrainerArgs, DistributedArgs
from eval import eval_end_task

def calculate_mask_change_rate(masks_before, masks_after):
    """
    Calculate the average change rate across multi-layer masks.
    Args:
        masks_before: Mask dict before update {layer_name: mask_tensor}
        masks_after: Mask dict after update {layer_name: mask_tensor}
    Returns:
        avg_change_rate: Weighted average change rate across all layers (weighted by element count)
        layer_change_rates: Per-layer change rate dict {layer_name: change_rate}
    """
    if not masks_before or not masks_after:
        return 0.0, {}
    
    layer_change_rates = {}
    total_changed = 0
    total_elements = 0
    
    # Calculate per-layer change rate
    common_layers = set(masks_before.keys()) & set(masks_after.keys())
    
    for layer_name in common_layers:
        mask_before = masks_before[layer_name]
        mask_after = masks_after[layer_name]
        
        if mask_before.shape != mask_after.shape:
            continue
        
        # Ensure both masks are on the same device
        if mask_before.device != mask_after.device:
            # Move mask_before to mask_after's device
            mask_before = mask_before.to(mask_after.device)
        
        # Ensure both masks are bool type to avoid precision-related comparison errors
        mask_before_bool = mask_before.bool() if mask_before.dtype != torch.bool else mask_before
        mask_after_bool = mask_after.bool() if mask_after.dtype != torch.bool else mask_after
        
        # Calculate layer changes (use XOR to get exact differing positions)
        diff = (mask_before_bool ^ mask_after_bool)
        layer_elements = mask_before_bool.numel()
        layer_changed = diff.sum().item()
        
        layer_change_rates[layer_name] = layer_changed / layer_elements if layer_elements > 0 else 0.0
        
        # Accumulate for weighted average
        total_changed += layer_changed
        total_elements += layer_elements
    
    # Calculate weighted average change rate
    avg_change_rate = total_changed / total_elements if total_elements > 0 else 0.0
    
    return avg_change_rate, layer_change_rates

def collect_sparse_masks(model):
    """Collect sparse masks from the model's forward pass cache"""
    if hasattr(model, 'get_current_masks'):
        return model.get_current_masks()
    return {}


def device_mapping(cuda_device: int):
    """
    In order to `torch.load()` a GPU-trained model onto a CPU (or specific GPU),
    you have to supply a `map_location` function. Call this with
    the desired `cuda_device` to get the function that `torch.load()` needs.
    """

    def inner_device_mapping(storage: torch.Storage, location) -> torch.Storage:  # pylint: disable=unused-argument
        if cuda_device >= 0:
            return storage.cuda(cuda_device)
        else:
            return storage

    return inner_device_mapping

class Trainer:
    '''
    Training a GPT model with the given arguments.
    '''
    def __init__(self, args: TrainerArgs, modelargs: ModelArgs, dataloader_args: DataLoaderArgs, distributed_args: DistributedArgs, n_iter: int = 0):
        self.args = args
        self.modelargs = modelargs
        self.dataloader_args = dataloader_args
        self.distributed_args = distributed_args
        # Handle save_checkpoint_dir: if specified, use it; otherwise use checkpoint_dir
        if self.args.save_checkpoint_dir is None:
            self.args.save_checkpoint_dir = self.args.checkpoint_dir
        self.model = self.build_model(modelargs)
        self.optimizer = self.build_optimizer()
        self.dataloader = self.build_train_loader(dataloader_args)
        self.logger = Logger(args, rank=distributed_args.rank)
        self.n_iter = n_iter
        
        # Sparse mask change monitoring
        self.previous_sparse_masks = None
        self.mask_change_rates = []
        
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        self.logger.log_text(f"Initialized trainer with args: {args}, modelargs: {modelargs}, dataloader_args: {dataloader_args}")
        self.logger.log_text(f"Initialized trainer with distributed args: {distributed_args}")
        self.logger.log_text(f"Initialized trainer with n_iter: {n_iter}")
        self.logger.log_text(f"Initialized trainer with model: {self.model}")
        self.logger.log_text(f"Total trainable parameters: {total_params:,}")
        if self.args.use_weight_semi_sparse:
            self.logger.log_text(f"Enabled 2:4 semi-sparse training for weights")

    def build_model(self, model_args: ModelArgs):

        if isinstance(model_args, MoEModelArgs):
            model = MoEModel(model_args)
        elif isinstance(model_args, ModelArgs):
            model = Model(model_args)
        else:
            assert 0, "Unknown type"
        model = model.cuda().to(torch.bfloat16)
        return model

    def model_to_ddp(self):
        if self.distributed_args.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.distributed_args.local_rank])

    def build_train_loader(self, dataloader_args: DataLoaderArgs):
        loader = LMLoader(dataloader_args, prefetch=True, shuffle=True, device='cuda', num_shards=self.distributed_args.world_size, shard_id=self.distributed_args.rank)
        return loader

    def build_optimizer(self):
        if self.args.zero:
            optim = ZeroRedundancyOptimizer(
                self.model.parameters(),
                optimizer_class=torch.optim.AdamW,
                lr=self.args.learning_rate,
                betas=(self.args.beta1, self.args.beta2),
                weight_decay=self.args.weight_decay,
                fused=True
            )
        else:
            optim = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.args.learning_rate,
                betas=(self.args.beta1, self.args.beta2),
                weight_decay=self.args.weight_decay,
                fused=True
            )
        return optim

    def update_lr(self, it):
        it -= self.args.lr_offset
        # 1) linear warmup for warmup_iters steps
        if it < self.args.warmup_iters:
            return self.args.learning_rate * it / self.args.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.args.lr_decay_iters:
            return self.args.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.args.warmup_iters) / (self.args.lr_decay_iters - self.args.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.args.min_lr + coeff * (self.args.learning_rate - self.args.min_lr)
    
    def update_wd(self, it):
        it -= self.args.lr_offset
        if self.args.wd_decay_iters == 0:
            return self.args.weight_decay
        if it > self.args.wd_decay_iters:
            return 0.0
        decay_ratio = it / self.args.wd_decay_iters
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.args.weight_decay * coeff

    def all_reduce(self, data_list, group=None):
        if group is None:
            group = dist.group.WORLD
        if self.distributed_args.world_size > 1:
            data_list = [torch.tensor(d).float().cuda() if not isinstance(d, torch.Tensor) else d.float() for d in data_list]
            data = torch.stack(data_list)
            dist.all_reduce(data, op=dist.ReduceOp.SUM, group=group)
            data_list = torch.unbind(data)
        data_list = [d.item() if isinstance(d, torch.Tensor) else d for d in data_list]
        return data_list

    def get_model_for_monitoring(self):
        """Get the model for mask monitoring (handles DDP wrapping)"""
        return self.model.module if hasattr(self.model, 'module') else self.model

    def _process_mask_change_rate(self, current_masks, step_for_logging=None):
        """Process mask change rate calculation, return metrics instead of logging directly"""
        sparse_metrics = {}
        
        if self.previous_sparse_masks is not None:
            avg_change_rate, layer_change_rates = calculate_mask_change_rate(
                self.previous_sparse_masks, current_masks
            )
            self.mask_change_rates.append(avg_change_rate)
            
            # Build sparse metrics
            sparse_metrics = {
                'flip_rate': avg_change_rate,
                'sparse_layer_count': len(layer_change_rates),
            }
            
            
            # Console output is now handled uniformly by Logger, no extra print statements needed
        
        # Clean up previous mask references to avoid memory leaks
        if hasattr(self, 'previous_sparse_masks') and self.previous_sparse_masks is not None:
            del self.previous_sparse_masks
        
        # Deep copy current masks as baseline for next comparison (avoid reference leaks)
        self.previous_sparse_masks = {k: v.clone().detach() for k, v in current_masks.items()}
        
        return sparse_metrics

    def train(self):
        model, optimizer = self.model, self.optimizer
        model.train()
        ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

        self.eval()
        dist.barrier()

        while True:
            lr = self.update_lr(self.n_iter)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            wd = self.update_wd(self.n_iter)
            for param_group in optimizer.param_groups:
                param_group['weight_decay'] = wd
            
            # Determine whether masks need to be collected at this step
            next_step = self.n_iter + 1
            will_log_next = next_step % self.args.log_interval == 0
            will_monitor_masks = (self.args.use_weight_semi_sparse and 
                                 getattr(self.args, 'sparse_mask_sync_with_log', False) and 
                                 will_log_next)
            
            # If mask monitoring is needed, enable before forward pass
            if will_monitor_masks:
                model_for_monitoring = self.get_model_for_monitoring()
                model_for_monitoring.enable_mask_monitoring()
            
            _loss, _n_tokens, _moe_loss = 0.0, 0, 0.0
            for i in range(self.args.update_freq):
                if self.distributed_args.world_size > 1:
                    model.require_backward_grad_sync = (i == self.args.update_freq - 1)
                tokens, targets, loss_mask = next(iter(self.dataloader))

                with ctx:
                    loss, n_tokens, moe_loss = model(tokens, targets=targets, loss_mask=loss_mask)
                    _loss += loss.item() - moe_loss * self.modelargs.w_gate_loss if moe_loss > 0 else loss.item()
                    _n_tokens += n_tokens.item()
                    _moe_loss += moe_loss
                    loss = loss / n_tokens / self.args.update_freq
                    if torch.isnan(loss).any():
                        print(f"NaN detected in loss at rank {self.distributed_args.rank} at iteration {self.n_iter}")

                loss.backward()
            
            # If mask monitoring is enabled, collect masks after forward pass
            if will_monitor_masks:
                self._current_step_masks = collect_sparse_masks(model_for_monitoring)
                # Collect masks first, then disable monitoring and clear cache
                model_for_monitoring.disable_mask_monitoring()
                # Manually clear the model's mask cache to free memory
                if hasattr(model_for_monitoring, 'mask_cache'):
                    model_for_monitoring.mask_cache.clear()

            _loss, _n_tokens, _moe_loss = self.all_reduce([_loss, _n_tokens, _moe_loss])
            logging_dicts = {'loss': _loss, 'tokens': _n_tokens, 'lr': lr, 'wd': wd, 'moe_loss': _moe_loss, 'gate_loss_weight': getattr(self.modelargs, 'w_gate_loss', 0)}

            if self.args.clip_grad_norm > 0:
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip_grad_norm).item()
                logging_dicts['grad_norm'] = gnorm
            else:
                gnorm = 0

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            self.n_iter += 1

            self.logger.update_meters(logging_dicts)
            
            # Check if metrics need to be logged (check after increment)
            should_log_metrics = self.n_iter % self.args.log_interval == 0
            
            # Collect sparse mask metrics (if already collected during forward pass)
            sparse_metrics = {}
            if should_log_metrics and hasattr(self, '_current_step_masks'):
                # Use masks already collected in this step
                sparse_metrics = self._process_mask_change_rate(self._current_step_masks, step_for_logging=self.n_iter)
                # Clean up temporary masks
                delattr(self, '_current_step_masks')
            
            # Log all metrics (regular + sparse, at the same step)
            if should_log_metrics:
                # Merge regular and sparse metrics
                all_metrics = dict(self.logger.meters)
                all_metrics.update(sparse_metrics)
                
                self.logger.log_metrics(all_metrics, tag='train', step=self.n_iter)
                self.logger.reset_meters(self.logger.meters)

            if self.args.save_interval > 0 and self.n_iter % self.args.save_interval == 0:
                self.save_checkpoint()
            
            if self.args.eval_interval > 0 and self.n_iter % self.args.eval_interval == 0:
                self.eval()
                dist.barrier()

            if self.n_iter >= self.args.max_updates:
                break

    def eval(self):
        dp_rank = self.distributed_args.rank

        if dp_rank != 0 or self.args.eval_interval == 0:
            return

        model = self.model
        model.eval()

        with torch.no_grad():
            results = eval_end_task(
                model.module if isinstance(model, DDP) else model,
                self.dataloader.tokenizer,
                self.args.eval_tasks,
                self.args.eval_limit,
                self.args.eval_batch_size,
                self.dataloader_args.max_seq_len,
            )

        for task, res in results["results"].items():
            self.logger.valid_meters[task] = res['acc_norm,none'] if 'acc_norm,none' in res else res['acc,none']

        self.logger.log_metrics(self.logger.valid_meters, tag='valid', step=self.n_iter)

        model.train()


    def profile(self):
        model, optimizer = self.model, self.optimizer
        model.train()
        ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

        wait, warmup, active = 5, 5, 5
        num_steps = wait + warmup + active
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./bench_log'),
            record_shapes=False,
            profile_memory=False,
            with_stack=False, # incurs an additional overhead, disable if not needed
            with_flops=True,
            with_modules=False, # only for torchscript models atm
        ) as prof:
            for _ in range(num_steps):
                tokens, targets, loss_mask = next(iter(self.dataloader))

                with ctx:
                    loss, n_tokens = model.compute_loss(tokens, targets, loss_mask)
                loss = loss / n_tokens
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                prof.step()

    @classmethod
    def from_pretrained(cls, checkpoint_dir: str, distributed_args: DistributedArgs, training_args: Optional[TrainerArgs] = None, dataloader_args: Optional[DataLoaderArgs] = None, modelargs: Optional[ModelArgs] = None):
        with open(os.path.join(checkpoint_dir, "metadata.json")) as f:
            metadata = json.load(f)
            n_iter = metadata["updates"]
            args = TrainerArgs(**metadata["args"]) if training_args is None else training_args
            if modelargs is None:
                if any('moe' in key for key in metadata["modelargs"].keys()):
                    modelargs = MoEModelArgs(**metadata["modelargs"])
                else:
                    modelargs = ModelArgs(**metadata["modelargs"])
            dataloader_args = DataLoaderArgs(**metadata["dataloader_args"]) if dataloader_args is None else dataloader_args
        trainer = cls(args, modelargs, dataloader_args, distributed_args, n_iter=n_iter)
        trainer.load_checkpoint(checkpoint_dir)
        return trainer

    def load_checkpoint(self, checkpoint_dir: str):
        dp_rank = self.distributed_args.rank
        world_size = self.distributed_args.world_size

        if not self.args.reset_dataloader:
            data_path = os.path.join(checkpoint_dir, f"data_state_rank_{dp_rank}.pt" if world_size > 1 else "data_state.pt")
            data_state = torch.load(data_path, map_location=device_mapping(-1))
            self.dataloader.setstate(data_state)

        if dp_rank == 0:
            model_path = os.path.join(checkpoint_dir, "model_state.pt")
            model_state = torch.load(model_path, map_location=device_mapping(-1))
            self.model.load_state_dict(model_state)

        if not self.args.reset_optimizer:
            training_state_path = os.path.join(checkpoint_dir, "training_state.pt")
            training_state = torch.load(training_state_path, map_location=device_mapping(-1))
            self.optimizer.load_state_dict(training_state)
            
            # Restore sparse mask change monitoring data
            if 'mask_change_rates' in training_state:
                self.mask_change_rates = training_state['mask_change_rates']
                self.previous_sparse_masks = training_state['previous_sparse_masks']

        self.logger.log_text(f"Loaded checkpoint from {checkpoint_dir}")

    def save_checkpoint(self):
        dp_rank = self.distributed_args.rank
        world_size = self.distributed_args.world_size
        checkpoint_dir = os.path.join(self.args.save_checkpoint_dir, f'updates_{self.n_iter}')
        os.makedirs(checkpoint_dir, exist_ok=True)

        data_path = os.path.join(checkpoint_dir, f"data_state_rank_{dp_rank}.pt" if world_size > 1 else "data_state.pt")
        data_state = self.dataloader.getstate()
        torch.save(data_state, data_path)

        if dp_rank == 0:
            model_path = os.path.join(checkpoint_dir, "model_state.pt")
            model_state = self.model.state_dict() if world_size == 1 else self.model.module.state_dict()
            torch.save(model_state, model_path)

        if self.args.zero:
            self.optimizer.consolidate_state_dict()

        if dp_rank == 0:
            training_state_path = os.path.join(checkpoint_dir, "training_state.pt")
            training_state = self.optimizer.state_dict()
            training_state['n_iter'] = self.n_iter
            
            # Save sparse mask change monitoring data
            if hasattr(self, 'mask_change_rates') and self.mask_change_rates:
                training_state['mask_change_rates'] = self.mask_change_rates
                training_state['previous_sparse_masks'] = self.previous_sparse_masks
            
            torch.save(training_state, training_state_path)

        if dp_rank == 0:
            with open(os.path.join(checkpoint_dir, "metadata.json"), "w") as f:
                metadata = {
                    "updates": self.n_iter,
                    "args": asdict(self.args),
                    "modelargs": asdict(self.modelargs),
                    "dataloader_args": asdict(self.dataloader_args)
                }
                json.dump(metadata, f, indent=4)

            with open(os.path.join(self.args.save_checkpoint_dir, "last_checkpoint.txt"), "w") as f:
                print(checkpoint_dir, file=f)
        
        self.logger.log_text(f"Saved checkpoint to {checkpoint_dir}")

if __name__ == '__main__':
    import time
    start_time = time.time()

    model_args, dataloader_args, trainer_args = parse_training_args()
    
    if int(os.environ.get('WORLD_SIZE', 1)) > 1:
        init_process_group(backend='nccl')
        dp_rank = int(os.environ['RANK'])
        dp_local_rank = int(os.environ['LOCAL_RANK'])
        dp_world_size = int(os.environ['WORLD_SIZE'])
    else:
        dp_rank, dp_local_rank, dp_world_size = 0, 0, 1

    device = f'cuda:{dp_local_rank}'
    torch.cuda.set_device(device)
    seed_offset = dp_rank
    torch.manual_seed(42 + seed_offset)
    distributed_args = DistributedArgs(rank=dp_rank, local_rank=dp_local_rank, world_size=dp_world_size)

    if os.path.exists(os.path.join(trainer_args.checkpoint_dir, "last_checkpoint.txt")):
        with open(os.path.join(trainer_args.checkpoint_dir, "last_checkpoint.txt")) as f:
            checkpoint_dir = f.readline().strip()
            trainer = Trainer.from_pretrained(checkpoint_dir, distributed_args)
    elif trainer_args.pretrained_model is not None:
        trainer = Trainer.from_pretrained(trainer_args.pretrained_model, distributed_args, training_args=trainer_args, dataloader_args=dataloader_args, modelargs=model_args)
    else:
        trainer = Trainer(trainer_args, model_args, dataloader_args, distributed_args)
    
    if dp_world_size > 1:
        trainer.model_to_ddp()

    trainer.train()
    # trainer.profile()
    print(f'Training time: {time.time() - start_time}')
    subprocess.run(["pkill", "-9", "python"])
    if dp_world_size > 1:
        destroy_process_group()
