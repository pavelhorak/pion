"""Standardized KV cache tensor serialization for cache transport.

Format:
    Header (24 bytes):
        magic:           4B  = 0x50494F4E ("PION")
        version:         2B  = 1
        dtype_code:      2B  = 1 (FP16) or 2 (FP32)
        num_layers:      4B  uint32 LE
        seq_len:         4B  uint32 LE
        num_kv_heads:    4B  uint32 LE
        head_dim:        4B  uint32 LE

    Body:
        For each layer:
            keys:   [1, num_kv_heads, seq_len, head_dim] in dtype
            values: [1, num_kv_heads, seq_len, head_dim] in dtype

FP16 reduces wire size by 2x vs FP32 with minimal quality impact.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

MAGIC = b"PION"
VERSION = 1
HEADER_SIZE = 24

DTYPE_FP16 = 1
DTYPE_FP32 = 2

_DTYPE_MAP = {
    DTYPE_FP16: np.float16,
    DTYPE_FP32: np.float32,
}
_DTYPE_REVERSE = {v: k for k, v in _DTYPE_MAP.items()}


@dataclass
class KVCacheHeader:
    num_layers: int
    seq_len: int
    num_kv_heads: int
    head_dim: int
    dtype_code: int = DTYPE_FP16

    @property
    def numpy_dtype(self) -> np.dtype:
        return np.dtype(_DTYPE_MAP[self.dtype_code])

    @property
    def bytes_per_element(self) -> int:
        return self.numpy_dtype.itemsize

    @property
    def layer_tensor_size(self) -> int:
        """Size of one tensor (keys or values) for one layer in bytes."""
        return self.num_kv_heads * self.seq_len * self.head_dim * self.bytes_per_element

    @property
    def total_size(self) -> int:
        """Total serialized size including header."""
        return HEADER_SIZE + self.num_layers * 2 * self.layer_tensor_size


def serialize_kv_cache(
    layers: list[tuple[np.ndarray, np.ndarray]],
    use_fp16: bool = True,
) -> bytes:
    """Serialize multi-layer KV cache to bytes.

    Args:
        layers: List of (keys, values) per layer.
            Each tensor shape: [1, num_kv_heads, seq_len, head_dim] or
                               [num_kv_heads, seq_len, head_dim]
        use_fp16: Store as FP16 (default) or FP32.

    Returns:
        Serialized bytes.
    """
    if not layers:
        raise ValueError("Empty layer list")

    target_dtype = np.float16 if use_fp16 else np.float32
    dtype_code = DTYPE_FP16 if use_fp16 else DTYPE_FP32

    # Normalize shapes to 4D [1, heads, seq, dim]
    normalized = []
    for keys, values in layers:
        k = np.asarray(keys, dtype=target_dtype)
        v = np.asarray(values, dtype=target_dtype)
        if k.ndim == 3:
            k = k[np.newaxis]
            v = v[np.newaxis]
        normalized.append((k, v))

    num_layers = len(normalized)
    _, num_kv_heads, seq_len, head_dim = normalized[0][0].shape

    header = MAGIC + struct.pack(
        "<HHIIII",
        VERSION,
        dtype_code,
        num_layers,
        seq_len,
        num_kv_heads,
        head_dim,
    )

    parts = [header]
    for k, v in normalized:
        parts.append(k.tobytes())
        parts.append(v.tobytes())

    return b"".join(parts)


def deserialize_kv_cache(blob: bytes) -> tuple[KVCacheHeader, list[tuple[np.ndarray, np.ndarray]]]:
    """Deserialize KV cache from bytes.

    Returns:
        (header, layers) where layers[i] = (keys, values)
        each of shape [1, num_kv_heads, seq_len, head_dim].
    """
    if len(blob) < HEADER_SIZE:
        raise ValueError(f"Blob too small: {len(blob)} < {HEADER_SIZE}")

    magic = blob[:4]
    if magic != MAGIC:
        raise ValueError(f"Bad magic: {magic!r}, expected {MAGIC!r}")

    version, dtype_code, num_layers, seq_len, num_kv_heads, head_dim = struct.unpack_from(
        "<HHIIII", blob, 4
    )

    if version != VERSION:
        raise ValueError(f"Unsupported version: {version}")

    header = KVCacheHeader(
        num_layers=num_layers,
        seq_len=seq_len,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype_code=dtype_code,
    )

    np_dtype = header.numpy_dtype
    tensor_elems = num_kv_heads * seq_len * head_dim
    tensor_bytes = tensor_elems * header.bytes_per_element

    offset = HEADER_SIZE
    layers = []
    for _ in range(num_layers):
        k = np.frombuffer(blob, dtype=np_dtype, count=tensor_elems, offset=offset)
        k = k.reshape(1, num_kv_heads, seq_len, head_dim).copy()
        offset += tensor_bytes

        v = np.frombuffer(blob, dtype=np_dtype, count=tensor_elems, offset=offset)
        v = v.reshape(1, num_kv_heads, seq_len, head_dim).copy()
        offset += tensor_bytes

        layers.append((k, v))

    return header, layers


def estimate_cache_size(
    num_layers: int,
    seq_len: int,
    num_kv_heads: int,
    head_dim: int,
    use_fp16: bool = True,
) -> int:
    """Estimate serialized cache size in bytes."""
    bpe = 2 if use_fp16 else 4
    return HEADER_SIZE + num_layers * 2 * num_kv_heads * seq_len * head_dim * bpe
