#!/bin/bash
# Example training script for Sparse-BitNet
# Usage: bash scripts/train.sh <model_name> <exp_name>

set -e

cd llm/

MODEL=${1:-"qwen_bitnet"}
EXP_NAME=${2:-"sparse_bitnet_training"}

# Set your own W&B credentials (optional)
# export WANDB_PROJECT="sparse-bitnet"
# export WANDB_API_KEY="your_wandb_api_key"
# export WANDB_NAME="${MODEL}_${EXP_NAME}"

OUTPUT_DIR="./checkpoints/${EXP_NAME}_${MODEL}"
mkdir -p "${OUTPUT_DIR}"

torchrun --nproc_per_node=8 --nnodes=1 train.py \
    --model ${MODEL} \
    --data example \
    --hyperparams bitnet_100b \
    --batch_size 16 \
    --update_freq 4 \
    --save_checkpoint_dir ${OUTPUT_DIR} \
    --cross_entropy_chunk 8 \
    --use_weight_semi_sparse \
    --sparse_n 2 \
    --sparse_m 4 \
    --sparse_mask_monitor_interval 100 \
    2>&1 | tee "${OUTPUT_DIR}/train.log"
