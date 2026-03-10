# Sparse-BitNet

Training framework for **Sparse-BitNet** models — combining 1.58-bit quantization (BitLinear) with N:M structured sparsity for efficient LLM training and inference.

## Features

- **BitLinear Quantization**: Ternary weight quantization ({-1, 0, 1}) following the BitNet b1.58 approach
- **N:M Structured Sparsity**: Hardware-friendly structured sparsity (e.g., 2:4) during training
- **Sparse-BitNet**: Joint sparse + quantized training with dynamic mask monitoring
- **Optimized Kernels**: Triton-based fused cross-entropy and sparsity mask creation
- **Distributed Training**: Multi-GPU training with DDP and optional ZeRO optimizer
- **Evaluation**: Built-in support for lm-evaluation-harness and perplexity evaluation

## Project Structure

```
├── llm/
│   ├── arch/
│   │   └── model.py                  # Core model (BitLinear, SparseLinear, Attention, FFN)
│   ├── data/
│   │   ├── lm_loader.py              # Data loader with infinibatch
│   │   ├── tokenizer.py              # Tiktoken-based tokenizer
│   │   └── infinibatch.py            # Infinibatch iterator
│   ├── kernel/
│   │   ├── linear_cross_entropy.py   # Triton fused linear cross-entropy
│   │   └── mask_creator_kernel.py    # Triton N:M sparse mask creation
│   ├── biteval/
│   │   └── eval_utils.py             # Evaluation utilities
│   ├── config.py                     # Model/data/training configurations
│   ├── train.py                      # Training loop
│   ├── eval.py                       # lm-eval-harness evaluation
│   ├── eval_ppl.py                   # Perplexity evaluation
│   ├── generate.py                   # Text generation
│   └── log.py                        # Logging utilities
├── scripts/
│   ├── train.sh                      # Training script
│   ├── evaluate.sh                   # Evaluation script
│   └── eval_ppl.sh                   # Perplexity evaluation script
├── setup.sh                          # Environment setup
└── README.md
```

## Installation

```bash
# Clone the repo
git clone https://github.com/<your-org>/Sparse-BitNet.git
cd Sparse-BitNet

# Install dependencies
bash setup.sh
```

### Requirements

- Python >= 3.10
- PyTorch >= 2.1
- CUDA >= 12.0
- [apex](https://github.com/NVIDIA/apex) (for fused RMSNorm)
- [triton](https://github.com/openai/triton) (for optimized kernels)
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) (for evaluation)

## Quick Start

### 1. Configure Data

Edit `llm/config.py` to add your data configuration:

```python
data_args = {
    "my_data": DataLoaderArgs(
        data_path='/path/to/your/data.json',
        tokenizer_path='/path/to/your/tokenizer.model'
    ),
}
```

The `data_path` should point to a JSON metadata file listing your training data shards. See [llm/example_metadata.json](llm/example_metadata.json) for the expected format.

### 2. Train a Sparse-BitNet Model

```bash
cd llm/

# Train a BitNet model with 2:4 sparsity
torchrun --nproc_per_node=8 --nnodes=1 train.py \
    --model qwen_bitnet \
    --data my_data \
    --hyperparams bitnet_100b \
    --batch_size 16 \
    --update_freq 4 \
    --save_checkpoint_dir ./checkpoints/my_experiment \
    --use_weight_semi_sparse \
    --sparse_n 2 \
    --sparse_m 4 \
    --cross_entropy_chunk 8
```

Or use the provided training script:

```bash
bash scripts/train.sh qwen_bitnet my_experiment
```

### 3. Evaluate

```bash
# Evaluate on downstream tasks
bash scripts/evaluate.sh ./checkpoints/my_experiment/updates_100000

# Evaluate perplexity
bash scripts/eval_ppl.sh ./checkpoints/my_experiment/updates_100000
```

### 4. Convert Checkpoint to Sparse+Quantized Format

After training, convert the checkpoint to truly sparse+quantized weights for inference:

```bash
cd llm/
python convert_to_sparse_bitnet.py \
    --checkpoints ./checkpoints/my_experiment/updates_100000 \
    --model_type qwen_bitnet \
    --sparse_n 2 \
    --sparse_m 4
```

See [llm/CONVERT_SPARSE_BITNET_README.md](llm/CONVERT_SPARSE_BITNET_README.md) for detailed conversion instructions.

## Model Configurations

### Dense Models

| Model | Parameters | Config Key |
|-------|-----------|------------|
| Qwen2.5-0.5B | ~0.5B | `qwen2_5_0_5B` |
| Qwen2.5-1.5B | ~1.5B | `qwen2_5_1_5B` |
| Qwen2.5-3B | ~3B | `qwen2_5_3B` |

### BitNet Models (1.58-bit)

Append `_bitnet` to any model name:
- `qwen2_5_0_5B_bitnet`, `qwen2_5_1_5B_bitnet`, `qwen2_5_3B_bitnet`

## Key Training Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--use_weight_semi_sparse` | `False` | Enable N:M structured sparsity |
| `--sparse_n` | `2` | N in N:M sparsity |
| `--sparse_m` | `4` | M in N:M sparsity |
| `--sparse_mask_monitor_interval` | `100` | Monitor mask change rate every N steps |
| `--cross_entropy_chunk` | `0` | Chunk size for fused cross-entropy (0 = disabled) |
| `--bitlinear` | `False` | Enable 1.58-bit weight quantization |
| `--pretrained_model` | `None` | Path to pretrained checkpoint for resuming |
| `--zero` | `False` | Enable ZeRO optimizer |

## How It Works

### BitLinear (1.58-bit Quantization)

During the forward pass, weights are quantized to ternary values {-1, 0, 1}:

```
w_quantized = round(w * scale).clamp(-1, 1) / scale
```

where `scale = 1 / mean(|w|)`. Gradients flow through via straight-through estimation (STE).

### N:M Structured Sparsity

For N:M sparsity (e.g., 2:4), within every M consecutive elements, only the N largest-magnitude values are kept:

```
mask = mask_creator(weight, N=2, M=4)  # Creates a binary mask
w_sparse = w * mask
```

### Sparse-BitNet

Combines both: first apply the sparsity mask, then quantize the remaining non-zero weights to {-1, 0, 1}. The result is extremely compressed weights suitable for efficient hardware inference.

## Converting HuggingFace Models

To start from a pretrained HuggingFace model:

```bash
cd llm/
python convert_hf_to_checkpoint.py \
    --hf_model_path /path/to/Qwen2.5-0.5B \
    --output_dir ./checkpoints/qwen2.5-0.5b-converted \
    --model_type qwen2_5_0_5B \
    --data my_data \
    --hyperparams bf16_50b
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{sparse-bitnet,
  title={Sparse-BitNet: 1.58-bit LLMs are Naturally Friendly to Semi-Structured Sparsity},
  author={Di Zhang, Xun Wu, Shaohan Huang, Yudong Wang, Hanyong Shao, Yingbo Hao, Zewen Chi, Li Dong, Ting Song, Yan Xia, Zhifang Sui, Furu Wei},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
