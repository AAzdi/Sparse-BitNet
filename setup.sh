#!/bin/bash
# Setup script for Sparse-BitNet training environment
set -e

# Install system dependencies (optional)
# apt-get install python3.10-dev -y

# Install Python dependencies
pip install torch torchvision torchaudio
pip install blobfile tiktoken wandb numpy einops triton safetensors

# Install lm-evaluation-harness for evaluation
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
cd ..

# Install torchtitan (for distributed training utilities)
pip install torchtitan

# Install torchao (for quantization support)
pip install torchao