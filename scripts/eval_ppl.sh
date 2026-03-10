#!/bin/bash
# Evaluate perplexity of a trained model
# Usage: bash scripts/eval_ppl.sh <checkpoint_dir> [tokenizer_path] [dataset]

set -e

cd llm/

CHECKPOINT_DIR=${1:?"Usage: bash scripts/eval_ppl.sh <checkpoint_dir> [tokenizer_path] [dataset]"}
TOKENIZER_PATH=${2:-""}
DATASET=${3:-"wikitext2,c4"}

EXTRA_ARGS=""
if [ -n "${TOKENIZER_PATH}" ]; then
    EXTRA_ARGS="--tokenizer_path ${TOKENIZER_PATH}"
fi

python eval_ppl.py \
    --checkpoint_dir ${CHECKPOINT_DIR} \
    --dataset "${DATASET}" \
    ${EXTRA_ARGS}
