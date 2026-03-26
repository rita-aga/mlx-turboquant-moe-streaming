"""
Run a model with SSD streaming for expert weights.

Usage:
    # Small model (test correctness)
    python run_streaming.py --model mlx-community/Qwen3.5-35B-A3B-4bit --prompt "Hello"

    # Large model (needs streaming)
    python run_streaming.py --model ~/Development/turbomoe/models/Qwen3.5-397B-A17B-4bit --prompt "Hello"

    # Kimi K2
    python run_streaming.py --model ~/Development/turbomoe/models/Kimi-K2-Instruct-4bit --prompt "Hello"
"""

import argparse
import os
import sys
import time
import gc

sys.path.insert(0, os.path.dirname(__file__))

import mlx.core as mx
from mlx_lm import generate


def main():
    parser = argparse.ArgumentParser(description="MLX SSD Streaming Inference")
    parser.add_argument("--model", required=True, help="Model path or HF model ID")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--tokens", type=int, default=20)
    parser.add_argument("--baseline", action="store_true", help="Run baseline (no streaming) first")
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"  MLX SSD Streaming Inference")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Prompt: {args.prompt}")
    print()

    model_path = os.path.expanduser(args.model)
    is_local = os.path.isdir(model_path)

    if is_local:
        # Local model — use streaming loader
        from streaming_loader import load_with_streaming
        from streaming_switch_linear import setup_streaming_for_model

        print("Loading backbone (skipping expert weights)...")
        t0 = time.time()
        model, tokenizer, model_dir = load_with_streaming(model_path)
        load_time = time.time() - t0
        print(f"Backbone loaded in {load_time:.1f}s")

        # Apply streaming patch
        print("\nApplying SSD streaming patch...")
        t0 = time.time()
        patched, bytes_freed = setup_streaming_for_model(model, model_dir)
        print(f"Patched {patched} layers, freed {bytes_freed/1e9:.2f} GB")
        print(f"Patch time: {time.time()-t0:.1f}s")

        gc.collect()
        mx.metal.clear_cache()

    else:
        # HF model — load normally then patch
        from mlx_lm import load
        from streaming_switch_linear import setup_streaming_for_model
        from huggingface_hub import snapshot_download

        model, tokenizer = load(args.model)
        model_dir = snapshot_download(args.model)

        if args.baseline:
            print(f"\n--- Baseline ---")
            t0 = time.time()
            response = generate(model, tokenizer, prompt=args.prompt,
                              max_tokens=args.tokens, verbose=True)
            print(f"Response: {response}")
            print(f"Time: {time.time()-t0:.1f}s")

        print("\nApplying SSD streaming patch...")
        patched, bytes_freed = setup_streaming_for_model(model, model_dir)
        print(f"Patched {patched} layers, freed {bytes_freed/1e9:.2f} GB")

    # Generate
    print(f"\n--- SSD Streaming Generation ---")
    print(f"Prompt: {args.prompt}")
    t0 = time.time()
    try:
        response = generate(model, tokenizer, prompt=args.prompt,
                          max_tokens=args.tokens, verbose=True)
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
