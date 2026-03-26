"""
Custom model loader that skips expert weights for SSD streaming.

Loads backbone weights (attention, norms, embeddings, shared expert, routing gate)
into RAM but skips the large routed expert weights (switch_mlp.{gate,up,down}_proj).
Those are streamed from SSD on demand.
"""

import glob
import json
import time
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.utils import load_config, load_tokenizer, load_model


def load_with_streaming(
    model_path: str,
    skip_patterns: Optional[list] = None,
):
    """Load an MLX model, skipping expert weight tensors.

    Returns (model, tokenizer, model_dir) where model has expert weights
    set to empty placeholders (to be replaced by streaming).
    """
    if skip_patterns is None:
        skip_patterns = ["switch_mlp.gate_proj", "switch_mlp.up_proj", "switch_mlp.down_proj"]

    model_path = Path(model_path)

    # Load config and create model architecture
    config = load_config(model_path)
    model, config = load_model(model_path, lazy=True, model_config=config)

    # Load weights, FILTERING OUT expert tensors
    weight_files = sorted(glob.glob(str(model_path / "model*.safetensors")))
    if not weight_files:
        raise FileNotFoundError(f"No safetensors found in {model_path}")

    print(f"Loading weights from {len(weight_files)} shards (skipping experts)...")
    t0 = time.time()
    weights = {}
    skipped_bytes = 0
    loaded_bytes = 0

    for wf in weight_files:
        all_tensors = mx.load(wf)
        for name, tensor in all_tensors.items():
            should_skip = any(p in name for p in skip_patterns)
            if should_skip:
                skipped_bytes += tensor.nbytes
            else:
                weights[name] = tensor
                loaded_bytes += tensor.nbytes

    elapsed = time.time() - t0
    print(f"Loaded {loaded_bytes/1e9:.2f} GB backbone in {elapsed:.1f}s")
    print(f"Skipped {skipped_bytes/1e9:.2f} GB expert weights (will stream from SSD)")

    # Load weights into model (strict=False to allow missing expert weights)
    model.load_weights(list(weights.items()), strict=False)

    # Load tokenizer
    # Monkey-patch for K2 tokenizer compatibility
    try:
        import transformers.models.gpt2.tokenization_gpt2 as gpt2_mod
        if not hasattr(gpt2_mod, 'bytes_to_unicode'):
            def bytes_to_unicode():
                bs = list(range(ord('!'), ord('~')+1)) + list(range(ord('¡'), ord('¬')+1)) + list(range(ord('®'), ord('ÿ')+1))
                cs = bs[:]
                n = 0
                for b in range(256):
                    if b not in bs:
                        bs.append(b)
                        cs.append(256+n)
                        n += 1
                return dict(zip(bs, [chr(n) for n in cs]))
            gpt2_mod.bytes_to_unicode = bytes_to_unicode
    except Exception:
        pass

    tokenizer = load_tokenizer(model_path, tokenizer_config_extra={"trust_remote_code": True})

    return model, tokenizer, str(model_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    model, tokenizer, model_dir = load_with_streaming(args.model)
    print(f"\nModel loaded! Dir: {model_dir}")
    print(f"Now need to call setup_streaming_for_model() to wire up SSD streaming.")
