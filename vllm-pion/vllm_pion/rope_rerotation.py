"""RoPE re-rotation utility for position-shifted KV cache injection.

Extracted from examples/step17_zero_prefill.py into a reusable module.
Enables injecting precomputed KV cache tensors at arbitrary positions
by re-rotating the RoPE-encoded key vectors.

RoPE only affects K (not V), so only keys need re-rotation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class RoPEConfig:
    """RoPE configuration extracted from a model."""
    head_dim: int
    freqs: np.ndarray  # shape [head_dim // 2], float32
    traditional: bool = False  # True = interleaved (d0,d1), (d2,d3)...; False = split-half

    @classmethod
    def from_base_theta(
        cls,
        head_dim: int,
        base: float = 10000.0,
        traditional: bool = False,
    ) -> "RoPEConfig":
        """Create RoPE config from standard base theta (e.g., Llama, Mistral)."""
        freqs = base ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim)
        return cls(head_dim=head_dim, freqs=freqs, traditional=traditional)

    @classmethod
    def from_inv_freqs(
        cls,
        head_dim: int,
        inv_freqs: np.ndarray,
        traditional: bool = False,
    ) -> "RoPEConfig":
        """Create RoPE config from pre-computed inverse frequencies (e.g., Llama3RoPE)."""
        # inv_freqs are 1/freqs; convert back
        freqs = 1.0 / inv_freqs.astype(np.float32)
        return cls(head_dim=head_dim, freqs=freqs, traditional=traditional)


def rerotate_keys(
    keys: np.ndarray,
    original_offset: int,
    target_offset: int,
    config: RoPEConfig,
) -> np.ndarray:
    """Re-rotate RoPE-encoded keys from original_offset to target_offset.

    Keys were computed with RoPE at positions [original_offset, original_offset + seq_len).
    We want them at positions [target_offset, target_offset + seq_len).

    Since delta = target_offset - original_offset is constant across all positions,
    the re-rotation is a single complex multiplication per dimension pair.

    Args:
        keys: [..., head_dim] — already RoPE-rotated keys. Any leading dims.
        original_offset: position offset used during original computation.
        target_offset: desired position offset for injection.
        config: RoPE configuration (frequencies + convention).

    Returns:
        Re-rotated keys with same shape as input.
    """
    delta = target_offset - original_offset
    if delta == 0:
        return keys.copy()

    half = config.head_dim // 2

    # Compute rotation angles: inv_freq = 1/freqs, angle = delta * inv_freq
    inv_freq = 1.0 / config.freqs  # [half]
    angles = (delta * inv_freq).astype(np.float32)  # [half]

    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    result = np.empty_like(keys)

    if not config.traditional:
        # Split-half convention: first half and second half of head_dim
        k_first = keys[..., :half]
        k_second = keys[..., half:]

        result[..., :half] = k_first * cos_a - k_second * sin_a
        result[..., half:] = k_first * sin_a + k_second * cos_a
    else:
        # Interleaved convention: pairs (0,1), (2,3), ...
        k_even = keys[..., 0::2]
        k_odd = keys[..., 1::2]

        result[..., 0::2] = k_even * cos_a - k_odd * sin_a
        result[..., 1::2] = k_even * sin_a + k_odd * cos_a

    return result


def validate_rerotation(
    config: RoPEConfig,
    test_seq_len: int = 4,
    test_num_heads: int = 8,
    target_offset: int = 42,
    tolerance: float = 0.01,
) -> tuple[bool, float, float]:
    """Validate re-rotation against direct RoPE application.

    Creates random keys, applies RoPE at offset=0, re-rotates to target_offset,
    and compares with direct RoPE at target_offset.

    Returns: (passed, max_diff, mean_diff)
    """
    rng = np.random.default_rng(42)
    keys = rng.standard_normal((1, test_num_heads, test_seq_len, config.head_dim)).astype(np.float32)

    # Apply RoPE at offset=0
    keys_at_0 = _apply_rope(keys, offset=0, config=config)

    # Re-rotate to target_offset
    keys_rerotated = rerotate_keys(keys_at_0, original_offset=0, target_offset=target_offset, config=config)

    # Apply RoPE directly at target_offset
    keys_at_target = _apply_rope(keys, offset=target_offset, config=config)

    diff = np.abs(keys_rerotated - keys_at_target)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    passed = max_diff < tolerance

    return passed, max_diff, mean_diff


def _apply_rope(
    keys: np.ndarray,
    offset: int,
    config: RoPEConfig,
) -> np.ndarray:
    """Apply RoPE encoding to keys at given position offset.

    For validation purposes only — production models apply RoPE internally.
    """
    seq_len = keys.shape[-2]
    half = config.head_dim // 2
    inv_freq = 1.0 / config.freqs  # [half]

    result = np.empty_like(keys)

    for pos in range(seq_len):
        angles = ((offset + pos) * inv_freq).astype(np.float32)
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)

        if not config.traditional:
            k_first = keys[..., pos, :half]
            k_second = keys[..., pos, half:]
            result[..., pos, :half] = k_first * cos_a - k_second * sin_a
            result[..., pos, half:] = k_first * sin_a + k_second * cos_a
        else:
            k_even = keys[..., pos, 0::2]
            k_odd = keys[..., pos, 1::2]
            result[..., pos, 0::2] = k_even * cos_a - k_odd * sin_a
            result[..., pos, 1::2] = k_even * sin_a + k_odd * cos_a

    return result
