import argparse
from dataclasses import dataclass, fields
from arch.model import ModelArgs
from arch.moe_model import MoEModelArgs
from data.lm_loader import DataLoaderArgs
import copy

@dataclass
class TrainerArgs:
    max_updates: int = 100000
    warmup_iters: int = 1000
    lr_decay_iters: int = 90000
    lr_offset: int = 0
    learning_rate: float = 3e-4
    min_lr: float = 1e-5
    beta1: float = 0.9
    beta2: float = 0.95
    wd_decay_iters: int = 0
    weight_decay: float = 0.1
    checkpoint_dir: str = '/tmp/checkpoints'
    save_checkpoint_dir: str = None  # If None, use checkpoint_dir
    log_interval: int = 100
    save_interval: int = 5000
    clip_grad_norm: float = 1.0
    update_freq: int = 1
    eval_interval: int = 20000000000
    eval_limit: int = None
    eval_batch_size: int = 32
    # eval_tasks: str = 'winogrande,hellaswag,arc_challenge'
    eval_tasks: str = 'winogrande'
    wandb_project: str = None
    wandb_entity: str = None
    zero: bool = False
    pretrained_model: str = None
    reset_dataloader: bool = False
    reset_optimizer: bool = False
    # N:M sparse training config
    use_weight_semi_sparse: bool = False
    sparse_n: int = 2  # N in N:M sparsity (number of non-zero values)
    sparse_m: int = 4  # M in N:M sparsity (block size)
    # Sparse mask change monitoring config
    sparse_mask_monitor_interval: int = 100  # Monitor mask changes every N steps
    sparse_mask_sync_with_log: bool = True  # Sync mask monitoring with regular logging

@dataclass
class DistributedArgs:
    backend: str = 'nccl'
    init_method: str = 'env://'
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0


model_args = {
    "qwen": ModelArgs(d_model=896, d_ffn=4864, head=14, kv_head=2, n_layers=24),
    "qwen_bitnet": ModelArgs(d_model=896, d_ffn=4864, head=14, kv_head=2, n_layers=24, bitlinear=True),
    "qwen2_5_0_5B": ModelArgs(n_layers=24, head=14, d_model=896, vocab_size=151936, kv_head=2, d_ffn=4864),
    "qwen2_5_0_5B_bitnet": ModelArgs(n_layers=24, head=14, d_model=896, vocab_size=151936, kv_head=2, d_ffn=4864, bitlinear=True),
    "qwen2_5_1_5B": ModelArgs(n_layers=28, head=12, d_model=1536, vocab_size=151936, kv_head=2, d_ffn=8960),
    "qwen2_5_1_5B_bitnet": ModelArgs(n_layers=28, head=12, d_model=1536, vocab_size=151936, kv_head=2, d_ffn=8960, bitlinear=True),
    "qwen2_5_3B": ModelArgs(n_layers=36, head=16, d_model=2048, vocab_size=151936, kv_head=2, d_ffn=11008),
    "qwen2_5_3B_bitnet": ModelArgs(n_layers=36, head=16, d_model=2048, vocab_size=151936, kv_head=2, d_ffn=11008, bitlinear=True),
}

data_args = {
    # Example data configurations - update paths to your local data
    # data_path: path to a JSON file listing training data shards
    # tokenizer_path: path to the tokenizer model file (e.g., Llama-3 tokenizer)
    "example": DataLoaderArgs(
        data_path='/path/to/your/data.json',
        tokenizer_path='/path/to/your/tokenizer.model'
    ),
}

training_args = {
    "debug": TrainerArgs(max_updates=2000, lr_decay_iters=2000),
    "bf16_50b_UniMoE": TrainerArgs(max_updates=50000, lr_decay_iters=50000),
    "bf16_100b": TrainerArgs(lr_decay_iters=100000),
    "bf16_600b": TrainerArgs(max_updates=600000, lr_decay_iters=600000, learning_rate=1e-3),
    "bitnet_100b": TrainerArgs(lr_decay_iters=100000, wd_decay_iters=50000),
    "bitnet_2t": TrainerArgs(max_updates=500000, lr_decay_iters=500000, wd_decay_iters=250000),
}

def parse_training_args():
    parser = argparse.ArgumentParser(description="training arguments")
    parser.add_argument(f"--model", type=str, choices=model_args.keys())
    parser.add_argument(f"--data", type=str, choices=data_args.keys())
    parser.add_argument(f"--hyperparams", type=str, choices=training_args.keys())

    argument_dict = {}

    # Add arguments for each field in the dataclass
    for config_type in [ModelArgs, DataLoaderArgs, TrainerArgs]:
        for field in fields(config_type):
            if field.name in argument_dict:
                continue

            argument_dict[field.name] = True
            field_type = field.type
            kwargs = {}

            if field_type == bool:
                kwargs["action"] = "store_true"
                parser.add_argument(f"--{field.name}", **kwargs)
            else:
                kwargs["default"] = None
                parser.add_argument(f"--{field.name}", type=field_type, **kwargs)


    custom_args = parser.parse_args()

    _model_args = copy.deepcopy(model_args[custom_args.model])
    _data_args = copy.deepcopy(data_args[custom_args.data])
    _trainer_args = copy.deepcopy(training_args[custom_args.hyperparams])

    for k, v in custom_args.__dict__.items():
        if v is not None and v is not False:
            _model_args.__dict__[k] = v
            _data_args.__dict__[k] = v
            _trainer_args.__dict__[k] = v

    return _model_args, _data_args, _trainer_args

def parse_eval_args():
    parser = argparse.ArgumentParser(description="evaluation arguments")
    parser.add_argument(f"--limit", type=int, default=None)
    parser.add_argument(f"--batch_size", type=int, default=32)
    parser.add_argument(f"--tasks", type=str, default=None)
    parser.add_argument(f"--valid_set", type=str, default=None)
    parser.add_argument(f"--checkpoint_dir", type=str, default=None)
    parser.add_argument(f"--merging_checkpoint_dir", type=str, default=None)
    parser.add_argument(f"--wandb_project", type=str, default=None)
    parser.add_argument(f"--wandb_id", type=str, default=None)
    
    return parser.parse_args()
