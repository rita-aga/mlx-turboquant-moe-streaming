"""
Integrated TurboQuant test: replace KVCache with TurboQuantKVCache
in the actual model and compare outputs.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.models.cache import KVCache
from turboquant_cache import TurboQuantKVCache


def main():
    model_name = "mlx-community/Qwen3.5-35B-A3B-4bit"
    print(f"Loading {model_name}...")
    model, tokenizer = load(model_name)
    print("Loaded!")

    prompts = [
        ("The capital of France is", 10),
        ("What is 2+2? Answer:", 3),
        ("Hello", 15),
        ("Explain gravity in simple terms.", 30),
    ]

    for prompt, max_tokens in prompts:
        print(f"\n{'='*60}")
        print(f"Prompt: '{prompt}'")

        # Baseline: FP16 KV cache
        t0 = time.time()
        baseline = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
        base_time = time.time() - t0

        # TurboQuant: Replace KVCache instances in the model's cache
        cache = model.make_cache()
        tq_count = 0
        for i, c in enumerate(cache):
            if isinstance(c, KVCache):
                cache[i] = TurboQuantKVCache(group_size=64, bits=4)
                tq_count += 1

        t0 = time.time()
        tq4_response = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens,
                               verbose=False, prompt_cache=cache)
        tq4_time = time.time() - t0

        # TurboQuant 3-bit
        cache3 = model.make_cache()
        for i, c in enumerate(cache3):
            if isinstance(c, KVCache):
                cache3[i] = TurboQuantKVCache(group_size=64, bits=3)

        t0 = time.time()
        tq3_response = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens,
                               verbose=False, prompt_cache=cache3)
        tq3_time = time.time() - t0

        # Compare
        match_4 = "EXACT MATCH" if tq4_response == baseline else "DIFFERENT"
        match_3 = "EXACT MATCH" if tq3_response == baseline else "DIFFERENT"

        print(f"  FP16 baseline  : '{baseline[:60]}' ({base_time:.1f}s)")
        print(f"  TurboQuant 4-bit: '{tq4_response[:60]}' ({tq4_time:.1f}s) {match_4}")
        print(f"  TurboQuant 3-bit: '{tq3_response[:60]}' ({tq3_time:.1f}s) {match_3}")

        if tq_count > 0:
            print(f"  ({tq_count} KVCache layers replaced with TurboQuant)")
            print(f"  Compression: 4-bit=4.0x, 3-bit=5.3x vs FP16")


if __name__ == "__main__":
    main()
