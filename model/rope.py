"""
Rotary Position Embedding (RoPE) — Su et al. (2021).
Encodes relative position information directly into query/key vectors
via rotation in 2D subspaces of the head dimension.
Precomputes frequency cache up to max_seq_len for efficiency.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


def precompute_rope_freqs(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute cosine and sine frequency tensors for RoPE.

    Returns:
        cos: (max_seq_len, head_dim)
        sin: (max_seq_len, head_dim)
    """

    # Frequency bands: θ_i = 1 / (theta ^ (2i / head_dim))
    # Shape: (head_dim // 2,)
    i = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    freqs = 1.0 / (theta ** (i / head_dim))

    # Positions: (max_seq_len,)
    positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)

    # Outer product → (max_seq_len, head_dim // 2)
    angles = torch.outer(positions, freqs)

    # Repeat to cover full head_dim: (max_seq_len, head_dim)
    angles = torch.cat([angles, angles], dim=-1)

    return angles.cos(), angles.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate the second half of the last dimension into the first half.
    Used to implement the RoPE rotation without explicit complex arithmetic.
    x: (..., head_dim)
    """

    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embedding to query and key tensors.

    Args:
        q: (batch, n_heads, seq_len, head_dim)
        k: (batch, n_heads, seq_len, head_dim)
        cos: (seq_len, head_dim) — precomputed cosines
        sin: (seq_len, head_dim) — precomputed sines

    Returns:
        q_rot, k_rot: same shapes as inputs
    """

    # Expand cos/sin to (1, 1, seq_len, head_dim) for broadcasting
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin

    return q_rot, k_rot


# Allow optional import
from typing import Optional