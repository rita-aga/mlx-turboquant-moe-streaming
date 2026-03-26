"""
SSD-streaming QuantizedSwitchLinear for MLX.

Drop-in replacement that loads expert weights from SSD on demand
instead of keeping all experts in RAM. Enables running models
larger than available memory.

Based on Flash-MoE's SSD streaming technique:
- pread() selected experts from safetensors files
- OS page cache handles caching
- Only K active experts loaded per token
"""

import os
import struct
import json
import math
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple, Dict


class ExpertWeightStore:
    """Manages SSD-based expert weight storage for one projection (gate/up/down).

    Stores file path + offset info for each expert's weight/scales/biases.
    Loads only the requested experts via pread() on demand.
    """

    def __init__(
        self,
        weight_file: str,
        weight_offset: int,
        weight_expert_bytes: int,
        scales_file: str,
        scales_offset: int,
        scales_expert_bytes: int,
        biases_file: Optional[str],
        biases_offset: int,
        biases_expert_bytes: int,
        num_experts: int,
        weight_shape: Tuple[int, ...],  # Per-expert shape, e.g., (512, 256)
        scales_shape: Tuple[int, ...],  # Per-expert shape, e.g., (512, 32)
        biases_shape: Optional[Tuple[int, ...]],
        group_size: int = 64,
        bits: int = 4,
    ):
        self.num_experts = num_experts
        self.weight_shape = weight_shape
        self.scales_shape = scales_shape
        self.biases_shape = biases_shape
        self.group_size = group_size
        self.bits = bits

        # File info
        self.weight_file = weight_file
        self.weight_offset = weight_offset
        self.weight_expert_bytes = weight_expert_bytes
        self.scales_file = scales_file
        self.scales_offset = scales_offset
        self.scales_expert_bytes = scales_expert_bytes
        self.biases_file = biases_file
        self.biases_offset = biases_offset
        self.biases_expert_bytes = biases_expert_bytes

        # Open file descriptors for fast pread
        import fcntl
        self._w_fd = os.open(weight_file, os.O_RDONLY)
        self._s_fd = os.open(scales_file, os.O_RDONLY)
        self._b_fd = os.open(biases_file, os.O_RDONLY) if biases_file else -1

        # Expert LRU cache: keep recently-used experts in RAM
        self._cache = {}  # expert_id -> (weight, scales, biases) as mx.arrays
        self._cache_order = []  # LRU order
        self._cache_max = 32  # Keep up to 32 experts cached per projection

    def load_experts(self, expert_ids: list) -> Tuple[mx.array, mx.array, Optional[mx.array]]:
        """Load selected experts' weights from SSD.

        Uses C extension for parallel pread when available, falls back to Python.
        OS page cache handles caching — "Trust the OS" (Flash-MoE finding).
        Returns (weight, scales, biases) as mx.arrays with shape [K, ...].
        """
        try:
            return self._load_experts_fast(expert_ids)
        except (ImportError, Exception):
            return self._load_experts_python(expert_ids)

    def _load_experts_fast(self, expert_ids: list) -> Tuple[mx.array, mx.array, Optional[mx.array]]:
        """Fast C extension path: parallel pread with zero-copy numpy."""
        import fast_pread

        ids = [int(i) for i in expert_ids]

        # Load weights (uint32)
        w_np = fast_pread.load_experts(
            self._w_fd, ids, self.weight_offset,
            self.weight_expert_bytes, tuple(self.weight_shape),
            np.dtype(np.uint32).num
        )
        weight_mx = mx.array(w_np)

        # Load scales (uint16 → bfloat16)
        s_np = fast_pread.load_experts(
            self._s_fd, ids, self.scales_offset,
            self.scales_expert_bytes, tuple(self.scales_shape),
            np.dtype(np.uint16).num
        )
        scales_mx = mx.array(s_np).view(mx.bfloat16)

        # Load biases
        biases_mx = None
        if self._b_fd >= 0 and self.biases_shape is not None:
            b_np = fast_pread.load_experts(
                self._b_fd, ids, self.biases_offset,
                self.biases_expert_bytes, tuple(self.biases_shape),
                np.dtype(np.uint16).num
            )
            biases_mx = mx.array(b_np).view(mx.bfloat16)

        return weight_mx, scales_mx, biases_mx

    def _load_experts_python(self, expert_ids: list) -> Tuple[mx.array, mx.array, Optional[mx.array]]:
        """Python fallback: sequential pread."""
        import concurrent.futures
        k = len(expert_ids)

        def load_one_expert(eid):
            eid = int(eid)
            # Weight
            w_off = self.weight_offset + eid * self.weight_expert_bytes
            w_data = os.pread(self._w_fd, self.weight_expert_bytes, w_off)
            w = np.frombuffer(w_data, dtype=np.uint32).reshape(self.weight_shape)

            # Scales
            s_off = self.scales_offset + eid * self.scales_expert_bytes
            s_data = os.pread(self._s_fd, self.scales_expert_bytes, s_off)
            s = np.frombuffer(s_data, dtype=np.uint16).reshape(self.scales_shape)

            # Biases
            b = None
            if self._b_fd >= 0 and self.biases_shape is not None:
                b_off = self.biases_offset + eid * self.biases_expert_bytes
                b_data = os.pread(self._b_fd, self.biases_expert_bytes, b_off)
                b = np.frombuffer(b_data, dtype=np.uint16).reshape(self.biases_shape)

            return w, s, b

        # Parallel load all experts
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(k, 4)) as pool:
            results = list(pool.map(load_one_expert, expert_ids))

        w_chunks = [r[0] for r in results]
        s_chunks = [r[1] for r in results]
        b_chunks = [r[2] for r in results if r[2] is not None]

        # Stack and convert to mx arrays
        weight = mx.array(np.stack(w_chunks).view(np.uint32))
        scales = mx.array(np.stack(s_chunks).view(np.float32))  # Will be reinterpreted

        # For bfloat16: numpy doesn't have bf16, so we need to handle this
        # mx.array from uint16 buffer with explicit dtype
        scales_raw = np.stack(s_chunks)
        weight_mx = mx.array(np.stack(w_chunks))

        # Convert bf16 uint16 → actual bfloat16 mx.array
        scales_mx = mx.array(scales_raw).view(mx.bfloat16)

        biases_mx = None
        if b_chunks:
            biases_raw = np.stack(b_chunks)
            biases_mx = mx.array(biases_raw).view(mx.bfloat16)

        return weight_mx, scales_mx, biases_mx

    def __del__(self):
        if self._w_fd >= 0:
            os.close(self._w_fd)
        if self._s_fd >= 0:
            os.close(self._s_fd)
        if self._b_fd >= 0:
            os.close(self._b_fd)


import numpy as np

USE_MMAP = os.environ.get("USE_MMAP", "0") == "1"

class StreamingSwitchGLU(nn.Module):
    """SwitchGLU that streams expert weights from SSD.

    Drop-in replacement for the standard SwitchGLU.
    Non-expert weights (if any) stay in RAM.
    Expert weights loaded on demand via ExpertWeightStore.
    """

    def __init__(self, gate_store, up_store, down_store, activation=None):
        super().__init__()
        self.gate_store = gate_store
        self.up_store = up_store
        self.down_store = down_store
        if activation is None:
            from mlx_lm.models.switch_layers import SwiGLU
            self.activation = SwiGLU()
        else:
            self.activation = activation

    def _stream_c(self, indices, store):
        """Load experts using fast_streaming C extension (remap + pread in one call)."""
        import fast_streaming
        idx_np = np.asarray(indices, dtype=np.int32)
        w_np, s_np, b_np, mapped_np = fast_streaming.stream_experts(
            idx_np.flatten(),
            store._w_fd, store.weight_offset, store.weight_expert_bytes, tuple(store.weight_shape),
            store._s_fd, store.scales_offset, store.scales_expert_bytes, tuple(store.scales_shape),
            store._b_fd if store._b_fd >= 0 else -1,
            store.biases_offset if store._b_fd >= 0 else 0,
            store.biases_expert_bytes if store._b_fd >= 0 else 0,
            tuple(store.biases_shape) if store.biases_shape else (0,),
        )
        w = mx.array(w_np)
        s = mx.array(s_np).view(mx.bfloat16)
        b = mx.array(b_np).view(mx.bfloat16) if b_np is not None and b_np.size > 0 else None
        mapped = mx.array(mapped_np.reshape(indices.shape))
        return w, s, b, mapped

    def __call__(self, x, indices) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))

        # Index remapping + expert loading
        idx_np = np.array(indices)
        unique_ids = np.unique(idx_np.flatten()).tolist()
        max_id = max(unique_ids) + 1
        remap = np.empty(max_id, dtype=np.int32)
        for i, eid in enumerate(unique_ids):
            remap[eid] = i
        mapped = mx.array(remap[idx_np])

        gate_w, gate_s, gate_b = self.gate_store.load_experts(unique_ids)
        up_w, up_s, up_b = self.up_store.load_experts(unique_ids)

        # Gate + Up projections (GPU)
        x_gate = mx.gather_qmm(
            x, gate_w, gate_s, gate_b,
            rhs_indices=mapped, transpose=True,
            group_size=self.gate_store.group_size,
            bits=self.gate_store.bits,
        )
        x_up = mx.gather_qmm(
            x, up_w, up_s, up_b,
            rhs_indices=mapped, transpose=True,
            group_size=self.up_store.group_size,
            bits=self.up_store.bits,
        )

        # SwiGLU activation
        x_act = self.activation(x_up, x_gate)

        # Down projection
        down_w, down_s, down_b = self.down_store.load_experts(unique_ids)

        x_out = mx.gather_qmm(
            x_act, down_w, down_s, down_b,
            rhs_indices=mapped, transpose=True,
            group_size=self.down_store.group_size,
            bits=self.down_store.bits,
        )

        return x_out.squeeze(-2)


def setup_streaming_for_model(model, model_dir):
    """Replace SwitchGLU modules with StreamingSwitchGLU for SSD streaming.

    Returns the number of layers patched and estimated memory saved.
    """
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        print(f"No safetensors index found at {index_path}")
        return 0, 0

    with open(index_path) as f:
        idx = json.load(f)

    # Parse safetensors headers for byte offsets
    header_cache = {}

    def get_tensor_info(tensor_name):
        if tensor_name not in idx["weight_map"]:
            return None
        filename = idx["weight_map"][tensor_name]
        filepath = os.path.join(model_dir, filename)

        if filename not in header_cache:
            with open(filepath, "rb") as f:
                hl = struct.unpack("<Q", f.read(8))[0]
                h = json.loads(f.read(hl))
                h.pop("__metadata__", None)
                data_start = 8 + hl
            header_cache[filename] = (h, data_start, filepath)

        h, data_start, filepath = header_cache[filename]
        info = h.get(tensor_name, {})
        offsets = info.get("data_offsets", [0, 0])
        shape = info.get("shape", [])
        return {
            "filepath": filepath,
            "offset": data_start + offsets[0],
            "size": offsets[1] - offsets[0],
            "shape": shape,
        }

    # Navigate to the model layers
    if hasattr(model, 'language_model'):
        text_model = model.language_model
    else:
        text_model = model
    inner = text_model.model if hasattr(text_model, 'model') else text_model

    patched = 0
    bytes_freed = 0

    for layer_idx, layer in enumerate(inner.layers):
        if not hasattr(layer, 'mlp') or not hasattr(layer.mlp, 'switch_mlp'):
            continue

        switch_mlp = layer.mlp.switch_mlp
        stores = {}

        for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
            proj = getattr(switch_mlp, proj_name)

            # Try naming conventions
            for prefix in [
                f"language_model.model.layers.{layer_idx}.mlp.switch_mlp.{proj_name}",
                f"model.layers.{layer_idx}.mlp.switch_mlp.{proj_name}",
            ]:
                w_info = get_tensor_info(f"{prefix}.weight")
                s_info = get_tensor_info(f"{prefix}.scales")
                b_info = get_tensor_info(f"{prefix}.biases")

                if w_info is not None:
                    num_experts = w_info["shape"][0]
                    w_expert_bytes = w_info["size"] // num_experts
                    s_expert_bytes = s_info["size"] // num_experts if s_info else 0
                    b_expert_bytes = b_info["size"] // num_experts if b_info else 0

                    # Per-expert shapes (drop the first dimension)
                    w_shape = tuple(w_info["shape"][1:])
                    s_shape = tuple(s_info["shape"][1:]) if s_info else None
                    b_shape = tuple(b_info["shape"][1:]) if b_info else None

                    if USE_MMAP:
                        from mmap_streaming import MmapExpertStore
                        store = MmapExpertStore(
                            filepath=w_info["filepath"],
                            base_offset=w_info["offset"],
                            expert_bytes=w_expert_bytes,
                            num_experts=num_experts,
                            weight_shape=w_shape,
                            scales_filepath=s_info["filepath"] if s_info else "",
                            scales_offset=s_info["offset"] if s_info else 0,
                            scales_expert_bytes=s_expert_bytes,
                            scales_shape=s_shape,
                            biases_filepath=b_info["filepath"] if b_info else None,
                            biases_offset=b_info["offset"] if b_info else 0,
                            biases_expert_bytes=b_expert_bytes,
                            biases_shape=b_shape,
                            group_size=proj.group_size,
                            bits=proj.bits,
                        )
                    else:
                        store = ExpertWeightStore(
                            weight_file=w_info["filepath"],
                            weight_offset=w_info["offset"],
                            weight_expert_bytes=w_expert_bytes,
                            scales_file=s_info["filepath"] if s_info else "",
                            scales_offset=s_info["offset"] if s_info else 0,
                            scales_expert_bytes=s_expert_bytes,
                            biases_file=b_info["filepath"] if b_info else None,
                            biases_offset=b_info["offset"] if b_info else 0,
                            biases_expert_bytes=b_expert_bytes,
                            num_experts=num_experts,
                            weight_shape=w_shape,
                            scales_shape=s_shape,
                            biases_shape=b_shape,
                            group_size=proj.group_size,
                            bits=proj.bits,
                        )
                    stores[proj_name] = store
                    bytes_freed += w_info["size"] + (s_info["size"] if s_info else 0) + (b_info["size"] if b_info else 0)
                    break

        if len(stores) == 3:
            # Replace the SwitchGLU with streaming version
            streaming_mlp = StreamingSwitchGLU(
                stores['gate_proj'],
                stores['up_proj'],
                stores['down_proj'],
                activation=switch_mlp.activation,
            )
            layer.mlp.switch_mlp = streaming_mlp
            patched += 1
            print(f"  Layer {layer_idx}: patched for SSD streaming ({num_experts} experts)")

    return patched, bytes_freed
