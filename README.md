# Trillion on a Laptop

**An experiment in running trillion-parameter models on a MacBook.**

Read the full interactive report: **[your-ssd-is-a-gpu.vercel.app](https://your-ssd-is-a-gpu.vercel.app)**

> **This is a research experiment, not production software.** The code works, the measurements are real, but the interfaces are rough and the error handling is minimal. Best way to use this: give the repo to your coding agent (Claude Code, Cursor, Copilot). It will know what to do.

## What happened here

We ran Kimi-K2 (1.04 trillion parameters, 578 GB on disk) on an M3 Max MacBook Pro with 128 GB RAM. It used 7.1 GB of RAM and produced correct output. Two techniques made it possible:

1. **SSD Streaming**: Expert weights loaded from NVMe on demand via `pread()` instead of sitting in RAM. Only 8 of 384 experts are active per token — the other 376 stay on disk.

2. **TurboQuant**: KV cache compressed 4-5x using Hadamard rotation + scalar quantization ([Google Research](https://arxiv.org/abs/2504.19874)). Zero accuracy loss on factual queries, verified.

Both run simultaneously. 953 lines of code (812 Python + 141 C).

## Results

| Model | Params | On Disk | RAM Used | Speed | Output |
|-------|--------|---------|----------|-------|--------|
| Kimi-K2 | 1.04T | 578 GB | 7.1 GB | 0.5 tok/s | Correct |
| Qwen3.5-397B | 397B | 209 GB | 5.7 GB | 0.27 tok/s | Correct |
| Combined (K2 + TurboQuant) | 1.04T | 578 GB | 7.4 GB | 0.53 tok/s | Correct |

TurboQuant accuracy (tested on Qwen3.5-35B):

| Compression | Factual queries | MSE vs FP16 |
|-------------|----------------|-------------|
| 4-bit (4x) | EXACT MATCH | 56% less error |
| 3-bit (5.3x) | EXACT MATCH | 54% less error |

## Quick start

```bash
pip install mlx mlx-lm numpy

# Build the C extension (3x speedup over pure Python)
python setup.py build_ext --inplace

# Run K2
python run_streaming.py \
    --model path/to/Kimi-K2-Instruct-4bit \
    --prompt "The capital of France is" \
    --tokens 50

# Run with TurboQuant KV compression
python run_combined.py \
    --model path/to/Kimi-K2-Instruct-4bit \
    --prompt "The capital of France is" \
    --tokens 50 --tq-bits 4
```

## Files

| File | What it does |
|------|-------------|
| `streaming_switch_linear.py` | Replaces MLX's QuantizedSwitchLinear with SSD streaming |
| `streaming_loader.py` | Loads backbone weights, skips experts |
| `turboquant_cache.py` | Hadamard rotation + quantized KV cache |
| `fast_pread.c` | C extension: parallel pread with pthreads, zero-copy numpy |
| `run_streaming.py` | CLI for streaming-only mode |
| `run_combined.py` | CLI for streaming + TurboQuant |
| `index.html` | The interactive report |

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- Python 3.10+
- MLX and mlx-lm
- Model weights in safetensors format

## References

- [TurboQuant — Google Research](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
- [TurboQuant paper (arXiv:2504.19874)](https://arxiv.org/abs/2504.19874)
- [Flash-MoE](https://github.com/danveloper/flash-moe) — SSD-based MoE inference, "trust the OS page cache"
- [MLX](https://github.com/ml-explore/mlx) — Apple's ML framework
- [PolarQuant (arXiv:2502.02617)](https://arxiv.org/abs/2502.02617) — Related KV cache quantization work
