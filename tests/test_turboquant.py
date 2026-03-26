"""
Test TurboQuant KV cache compression on MLX.

Compares output quality between:
  1. No cache quantization (FP16 KV cache)
  2. MLX built-in quantization (8-bit)
  3. TurboQuant (4-bit with rotation)
  4. TurboQuant (3-bit with rotation)
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.models.cache import KVCache, QuantizedKVCache

from turboquant_cache import TurboQuantKVCache


def run_test(model, tokenizer, prompt, max_tokens, cache_factory, label):
    """Run generation with a specific cache type and measure."""
    print(f"\n--- {label} ---")

    # Reset caches
    if hasattr(model, 'language_model'):
        text_model = model.language_model
    else:
        text_model = model
    inner = text_model.model if hasattr(text_model, 'model') else text_model

    t0 = time.time()
    response = generate(
        model, tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=True,
    )
    elapsed = time.time() - t0

    print(f"Response: {response}")
    print(f"Time: {elapsed:.1f}s")
    return response


def main():
    print("=" * 60)
    print("  TurboQuant KV Cache Compression Test")
    print("=" * 60)

    # Load small model
    model_name = "mlx-community/Qwen3.5-35B-A3B-4bit"
    print(f"\nLoading {model_name}...")
    model, tokenizer = load(model_name)
    print("Loaded!")

    prompts = [
        "The capital of France is",
        "Explain quantum computing in one sentence.",
    ]

    for prompt in prompts:
        print(f"\n{'='*60}")
        print(f"Prompt: {prompt}")
        print(f"{'='*60}")

        # Test 1: Baseline (FP16 cache)
        r1 = run_test(model, tokenizer, prompt, 30, None, "FP16 KV Cache (baseline)")

        # Test 2: MLX 8-bit quantized cache
        # mlx-lm supports --kv-bits flag, but we need to set it programmatically
        # For now, just compare the outputs

        # Test 3: TurboQuant 4-bit
        # To use TurboQuant, we need to inject our cache into the model
        # This requires modifying how generate() creates caches
        # For now, let's test the rotation + quantization independently

    # Independent test of TurboQuant math
    print(f"\n{'='*60}")
    print("  TurboQuant Math Validation")
    print(f"{'='*60}")

    from turboquant_cache import create_hadamard_matrix

    # Test rotation quality
    dim = 128  # Typical head dim
    H = create_hadamard_matrix(dim)
    print(f"\nHadamard matrix: {H.shape}")
    print(f"Orthogonal check: H @ H.T ≈ I? max deviation: {mx.max(mx.abs(H @ H.T - mx.eye(dim))).item():.6f}")

    # Test quantization quality
    for bits in [8, 4, 3]:
        # Generate realistic KV cache vectors (attention output)
        x = mx.random.normal((1, 2, 100, dim))  # [batch, heads, seq, dim]

        # Without rotation
        q_norot, s_norot, b_norot = mx.quantize(x, group_size=64, bits=bits)
        x_norot = mx.dequantize(q_norot, s_norot, b_norot, group_size=64, bits=bits)
        mse_norot = mx.mean((x - x_norot) ** 2).item()
        cosine_norot = mx.mean(
            mx.sum(x * x_norot, axis=-1) /
            (mx.sqrt(mx.sum(x**2, axis=-1)) * mx.sqrt(mx.sum(x_norot**2, axis=-1)) + 1e-10)
        ).item()

        # With rotation (TurboQuant)
        x_rot = x @ H
        q_rot, s_rot, b_rot = mx.quantize(x_rot, group_size=64, bits=bits)
        x_rot_deq = mx.dequantize(q_rot, s_rot, b_rot, group_size=64, bits=bits)
        x_derot = x_rot_deq @ H.T
        mse_rot = mx.mean((x - x_derot) ** 2).item()
        cosine_rot = mx.mean(
            mx.sum(x * x_derot, axis=-1) /
            (mx.sqrt(mx.sum(x**2, axis=-1)) * mx.sqrt(mx.sum(x_derot**2, axis=-1)) + 1e-10)
        ).item()

        compression = 16.0 / bits
        print(f"\n{bits}-bit quantization ({compression:.1f}x compression):")
        print(f"  Without rotation: MSE={mse_norot:.6f} cosine={cosine_norot:.6f}")
        print(f"  With rotation:    MSE={mse_rot:.6f} cosine={cosine_rot:.6f}")
        print(f"  Rotation helps:   {'YES' if mse_rot < mse_norot else 'NO'} (MSE {'lower' if mse_rot < mse_norot else 'higher'})")


if __name__ == "__main__":
    main()
