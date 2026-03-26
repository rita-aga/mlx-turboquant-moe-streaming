#!/bin/bash
# Run Kimi-K2 (1 trillion parameters) with SSD streaming
#
# Prerequisites:
#   pip install mlx mlx-lm numpy tiktoken
#   python setup.py build_ext --inplace
#   Download: huggingface-cli download mlx-community/Kimi-K2-Instruct-4bit

MODEL_PATH="${1:-~/Development/turbomoe/models/Kimi-K2-Instruct-4bit}"

echo "Running Kimi-K2 (1.04 trillion parameters)"
echo "Model: $MODEL_PATH"
echo ""

python3 ../run_streaming.py \
    --model "$MODEL_PATH" \
    --prompt "Explain quantum computing in simple terms." \
    --tokens 50
