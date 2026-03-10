#!/bin/bash
# Evaluate a trained model checkpoint
# Usage: bash scripts/evaluate.sh <checkpoint_dir> [tasks]

set -e

cd llm/

CHECKPOINT_DIR=${1:?"Usage: bash scripts/evaluate.sh <checkpoint_dir> [tasks]"}
TASKS=${2:-"winogrande,hellaswag,arc_challenge"}
BATCH_SIZE=${3:-32}

python eval.py \
    --checkpoint_dir ${CHECKPOINT_DIR} \
    --tasks "${TASKS}" \
    --batch_size ${BATCH_SIZE}
