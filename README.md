# MLX SSD Streaming + TurboQuant

Run trillion-parameter MoE models on Apple Silicon laptops by streaming expert weights from SSD and compressing the KV cache with TurboQuant.

## What This Does

Two techniques for running models larger than your RAM:

1. **SSD Streaming**: Instead of loading all 578GB of Kimi-K2 into memory, load only the ~56MB of active expert weights per token from SSD. Uses 7.1GB RAM instead of 578GB.

2. **TurboQuant**: Compress the attention KV cache 4-5x using Hadamard rotation + quantization (Google Research, [TurboQuant paper](https://arxiv.org/abs/2504.19874)). Enables longer conversations and frees RAM for expert caching.

## Quick Start

```bash
# Install dependencies
pip install mlx mlx-lm numpy

# Build the C extension for fast SSD reads (3x speedup)
python setup.py build_ext --inplace

# Run Kimi-K2 (1 trillion parameters!)
python run_streaming.py \
    --model path/to/Kimi-K2-Instruct-4bit \
    --prompt "Hello, how are you?" \
    --tokens 50
```

## Results

| Model | Params | RAM Used | Model Size | Speed | Output |
|-------|--------|----------|-----------|-------|--------|
| Kimi-K2 | 1.04T | 7.1 GB | 578 GB | 0.5 tok/s | Correct |
| Qwen3.5-397B | 397B | 5.7 GB | 209 GB | 0.5 tok/s | Correct |
| Qwen3.5-35B (baseline) | 35B | 19.6 GB | 20 GB | 8.5 tok/s | Correct |

TurboQuant compression (tested on Qwen3.5-35B):
| Compression | Output Match | MSE Reduction |
|-------------|-------------|---------------|
| 4-bit (4x) | EXACT MATCH on factual queries | 56% less error |
| 3-bit (5.3x) | EXACT MATCH on factual queries | 54% less error |

## Files

| File | Purpose |
|------|---------|
| `streaming_switch_linear.py` | Core: replaces MLX's QuantizedSwitchLinear with SSD streaming |
| `streaming_loader.py` | Loads backbone only, skips expert weights |
| `turboquant_cache.py` | TurboQuant KV cache with Hadamard rotation |
| `run_streaming.py` | CLI entry point |
| `fast_pread.c` | C extension for parallel SSD reads (3x speedup) |

## How It Works

### SSD Streaming

MLX normally loads all expert weights into RAM via `QuantizedSwitchLinear`. Our `StreamingSwitchGLU` replaces this — it reads only the K selected experts from the safetensors files on disk using `os.pread()`. The OS page cache automatically keeps frequently-used experts in RAM.

### TurboQuant

The KV cache stores keys and values from previous tokens for attention. TurboQuant compresses this by:
1. Applying a Hadamard rotation (makes the distribution Gaussian)
2. Quantizing to 3-4 bits (Gaussian distributions quantize efficiently)
3. On read: dequantize and de-rotate

This reduces kurtosis from 27.7 to 3.2 and quantization error by 54-57%.

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- Python 3.10+
- MLX and mlx-lm
- Model weights (safetensors format)

## References

- [TurboQuant (Google Research)](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
- [Flash-MoE](https://github.com/danveloper/flash-moe) — C/Metal MoE streaming
- [MLX](https://github.com/ml-explore/mlx) — Apple's ML framework
- [PolarQuant](https://arxiv.org/abs/2502.02617) — Polar coordinate KV quantization
