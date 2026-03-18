"""
RMSNorm — Root Mean Square Layer Normalization.
Used in place of LayerNorm following the LLaMA architecture.
No bias, no mean subtraction — only rescaling by RMS.
"""

import torch
import torch.nn as nn
from typing import Optional


class RMSNorm(nn.Module):
    """
    RMSNorm as described in Zhang & Sennrich (2019).
    Normalizes by the root mean square of the input, then applies
    a learned per-dimension scale parameter (weight).
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.d_model: int = d_model
        self.eps: float = eps
        self.weight: nn.Parameter = nn.Parameter(torch.ones(d_model))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_model)
        # Compute RMS across last dimension, normalize
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast to float32 for numerical stability, then back to input dtype
        output = self._norm(x.float()).to(x.dtype)
        return output * self.weight

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, eps={self.eps}"