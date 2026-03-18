"""
ModelConfig dataclass — single source of truth for all architectural hyperparameters.
All other modules import from here; nothing is hardcoded elsewhere.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    # Vocabulary and sequence
    vocab_size: int = 50257          # GPT-2 tokenizer vocabulary
    block_size: int = 256            # Maximum sequence length

    # Architecture — 15M parameter configuration (justified below)
    n_layers: int = 8
    n_heads: int = 8
    d_model: int = 384               # Hidden dimension
    d_ffn: int = 1536                # FFN dim = 4 × d_model
    head_dim: int = 48               # d_model / n_heads

    # Regularization
    dropout: float = 0.0             # Disabled during probing; set in training if needed
    attn_dropout: float = 0.0        # Kept at 0 — we need clean attention maps

    # RoPE
    rope_theta: float = 10000.0

    # Initialization
    init_std: float = 0.02

    # Ablation flag — set to True for the 6M parameter run
    ablation_mode: bool = False

    def __post_init__(self) -> None:
        # Recompute head_dim from d_model and n_heads
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        )
        self.head_dim = self.d_model // self.n_heads

    @classmethod
    def small_15m(cls) -> "ModelConfig":
        """
        Primary experimental config — ~15M parameters.
        Justification: large enough to develop all 5 head types with clear
        specialization, small enough to train 3 full seeds on Modal A100
        for under $25 total. At 8 layers × 8 heads = 64 heads total,
        we have statistically meaningful per-type counts even with minority types.
        """

        return cls(
            n_layers=8,
            n_heads=8,
            d_model=384,
            d_ffn=1536,
        )

    @classmethod
    def ablation_6m(cls) -> "ModelConfig":
        """
        Scale ablation config — ~6M parameters.
        Used to test whether developmental ordering is scale-invariant (H3 extension).
        """

        return cls(
            n_layers=6,
            n_heads=8,
            d_model=256,
            d_ffn=1024,
            ablation_mode=True,
        )

    def count_parameters(self) -> int:
        """Estimate total trainable parameter count."""

        embed    = self.vocab_size * self.d_model
        attn     = self.n_layers * (4 * self.d_model * self.d_model)
        ffn      = self.n_layers * (3 * self.d_model * self.d_ffn)
        norms    = self.n_layers * (2 * self.d_model) + self.d_model
        lm_head  = self.vocab_size * self.d_model
        return embed + attn + ffn + norms + lm_head

    def __repr__(self) -> str:
        params = self.count_parameters()
        return (
            f"ModelConfig("
            f"layers={self.n_layers}, heads={self.n_heads}, "
            f"d_model={self.d_model}, d_ffn={self.d_ffn}, "
            f"~{params / 1e6:.1f}M params)"
        )