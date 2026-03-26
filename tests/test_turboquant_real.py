"""
Test TurboQuant on REAL model inference.

Compares generation quality between:
1. FP16 KV cache (baseline)
2. MLX 8-bit quantized cache (built-in)
3. MLX 4-bit quantized cache (built-in, aggressive)
4. TurboQuant 4-bit cache (rotation + quantization)

Measures: output text, token match rate, speed, memory.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.models.cache import KVCache, QuantizedKVCache
from turboquant_cache import TurboQuantKVCache


def run_with_cache(model, tokenizer, prompt, max_tokens, kv_bits=None, use_turboquant=False, tq_bits=4):
    """Run generation with specific cache configuration."""
    kwargs = {"prompt": prompt, "max_tokens": max_tokens}

    if kv_bits and not use_turboquant:
        kwargs["kv_bits"] = kv_bits

    if use_turboquant:
        # Directly create TurboQuant caches and pass as prompt_cache
        # This bypasses the to_quantized path and uses our dequantized return format
        if hasattr(model, 'language_model'):
            num_layers = len(model.language_model.model.layers)
        else:
            num_layers = len(model.model.layers)

        # Create cache list matching what MLX expects
        # For Qwen3.5, each layer needs a cache entry
        # GatedDeltaNet layers use ArraysCache, full-attn use KVCache
        # We only replace KVCache entries with TurboQuant
        prompt_cache = model.make_cache() if hasattr(model, 'make_cache') else None
        if prompt_cache:
            for i, c in enumerate(prompt_cache):
                if isinstance(c, KVCache):
                    prompt_cache[i] = TurboQuantKVCache(group_size=64, bits=tq_bits)
            kwargs["prompt_cache"] = prompt_cache

    t0 = time.time()
    response = generate(model, tokenizer, **kwargs, verbose=False)
    elapsed = time.time() - t0
    tok_s = max_tokens / elapsed if elapsed > 0 else 0

    # No cleanup needed for TurboQuant path

    return response, elapsed, tok_s


def main():
    print("=" * 70)
    print("  TurboQuant Real-World Comparison")
    print("=" * 70)

    model_name = "mlx-community/Qwen3.5-35B-A3B-4bit"
    print(f"\nLoading {model_name}...")
    model, tokenizer = load(model_name)
    print("Loaded!")

    prompts = [
        ("The capital of France is", 15),
        ("What is 2+2? Answer:", 5),
        ("Explain gravity in one sentence.", 30),
    ]

    for prompt, max_tokens in prompts:
        print(f"\n{'='*70}")
        print(f"Prompt: '{prompt}' (max {max_tokens} tokens)")
        print(f"{'='*70}")

        configs = [
            ("FP16 (baseline)", None, False, 0),
            ("MLX 8-bit", 8, False, 0),
            ("MLX 4-bit", 4, False, 0),
            ("TurboQuant 4-bit", None, True, 4),
            ("TurboQuant 3-bit", None, True, 3),
        ]

        results = {}
        for label, kv_bits, use_tq, tq_bits in configs:
            try:
                response, elapsed, tok_s = run_with_cache(
                    model, tokenizer, prompt, max_tokens,
                    kv_bits=kv_bits, use_turboquant=use_tq, tq_bits=tq_bits
                )
                results[label] = response
                # Truncate response for display
                display = response[:80] + "..." if len(response) > 80 else response
                print(f"  {label:25s}: '{display}' ({tok_s:.1f} tok/s)")
            except Exception as e:
                print(f"  {label:25s}: ERROR: {e}")
                results[label] = None

        # Compare outputs
        baseline = results.get("FP16 (baseline)")
        if baseline:
            print(f"\n  Token match vs baseline:")
            for label, response in results.items():
                if response and label != "FP16 (baseline)":
                    # Compare first N characters
                    match_len = 0
                    for a, b in zip(baseline, response):
                        if a == b:
                            match_len += 1
                        else:
                            break
                    match_pct = match_len / max(len(baseline), 1) * 100
                    exact = "EXACT" if response == baseline else f"{match_pct:.0f}% prefix"
                    print(f"    {label:25s}: {exact}")


if __name__ == "__main__":
    main()
