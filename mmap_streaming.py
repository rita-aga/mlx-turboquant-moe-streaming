"""
Memory-mapped expert streaming: mmap safetensors files and create
mx.array slices directly, avoiding the pread→numpy→mx.array roundtrip.

The OS page cache handles all caching automatically.
Expert reads become: mmap slice → numpy view (zero-copy) → mx.array (one copy).
This eliminates the pread system call overhead and numpy array allocation.
"""

import mmap
import os
import struct
import json
import numpy as np
import mlx.core as mx
from typing import Tuple, Optional


class MmapExpertStore:
    """Expert weight store using memory-mapped safetensors files.

    Instead of pread() per expert, mmap the file once.
    Creating an mx.array from an mmap slice is one copy (not two).
    The OS page cache handles caching automatically.
    """

    def __init__(
        self,
        filepath: str,
        base_offset: int,
        expert_bytes: int,
        num_experts: int,
        weight_shape: Tuple[int, ...],
        scales_filepath: str,
        scales_offset: int,
        scales_expert_bytes: int,
        scales_shape: Tuple[int, ...],
        biases_filepath: Optional[str],
        biases_offset: int,
        biases_expert_bytes: int,
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

        self.expert_bytes = expert_bytes
        self.base_offset = base_offset
        self.scales_offset = scales_offset
        self.scales_expert_bytes = scales_expert_bytes
        self.biases_offset = biases_offset
        self.biases_expert_bytes = biases_expert_bytes

        # Memory-map the files
        self._w_fd = os.open(filepath, os.O_RDONLY)
        self._w_size = os.fstat(self._w_fd).st_size
        self._w_mm = mmap.mmap(self._w_fd, 0, prot=mmap.PROT_READ)

        self._s_fd = os.open(scales_filepath, os.O_RDONLY)
        self._s_mm = mmap.mmap(self._s_fd, 0, prot=mmap.PROT_READ)

        self._b_mm = None
        self._b_fd = -1
        if biases_filepath and os.path.exists(biases_filepath):
            self._b_fd = os.open(biases_filepath, os.O_RDONLY)
            self._b_mm = mmap.mmap(self._b_fd, 0, prot=mmap.PROT_READ)

    def load_experts(self, expert_ids: list) -> Tuple[mx.array, mx.array, Optional[mx.array]]:
        """Load experts via mmap slicing (faster than pread)."""
        w_slices = []
        s_slices = []
        b_slices = []

        for eid in expert_ids:
            eid = int(eid)

            # Weight slice from mmap (zero-copy numpy view)
            w_start = self.base_offset + eid * self.expert_bytes
            w_end = w_start + self.expert_bytes
            w_np = np.frombuffer(self._w_mm[w_start:w_end], dtype=np.uint32).reshape(self.weight_shape)
            w_slices.append(w_np)

            # Scales
            s_start = self.scales_offset + eid * self.scales_expert_bytes
            s_end = s_start + self.scales_expert_bytes
            s_np = np.frombuffer(self._s_mm[s_start:s_end], dtype=np.uint16).reshape(self.scales_shape)
            s_slices.append(s_np)

            # Biases
            if self._b_mm is not None and self.biases_shape is not None:
                b_start = self.biases_offset + eid * self.biases_expert_bytes
                b_end = b_start + self.biases_expert_bytes
                b_np = np.frombuffer(self._b_mm[b_start:b_end], dtype=np.uint16).reshape(self.biases_shape)
                b_slices.append(b_np)

        # Stack and convert to mx (one copy per array)
        weight = mx.array(np.stack(w_slices))
        scales = mx.array(np.stack(s_slices)).view(mx.bfloat16)
        biases = mx.array(np.stack(b_slices)).view(mx.bfloat16) if b_slices else None

        return weight, scales, biases

    def __del__(self):
        if self._w_mm:
            self._w_mm.close()
        if hasattr(self, '_s_mm') and self._s_mm:
            self._s_mm.close()
        if hasattr(self, '_b_mm') and self._b_mm:
            self._b_mm.close()
        if self._w_fd >= 0:
            os.close(self._w_fd)
        if hasattr(self, '_s_fd') and self._s_fd >= 0:
            os.close(self._s_fd)
        if hasattr(self, '_b_fd') and self._b_fd >= 0:
            os.close(self._b_fd)
