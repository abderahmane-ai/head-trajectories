"""
Full LLaMA-style decoder-only transformer built from scratch in PyTorch.

CRITICAL DESIGN REQUIREMENT: The model must expose raw attention weight matrices
(post-softmax, pre-dropout) for every head at every layer during a forward pass.
This is the core requirement for the probing pipeline. Achieved via an optional
`return_attention` flag that populates a list of attention tensors without
modifying the standard forward pass output or computational graph.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from .config import ModelConfig
from .rmsnorm import RMSNorm
from .rope import precompute_rope_freqs, apply_rope


# ─────────────────────────────────────────────────────────────────────────────
# SwiGLU Feed-Forward Network
# ─────────────────────────────────────────────────────────────────────────────

class SwiGLUFFN(nn.Module):
    """
    SwiGLU feed-forward network as used in LLaMA.
    FFN(x) = (SiLU(W_gate(x)) * W_up(x)) @ W_down
    Three projection matrices instead of the standard two.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.d_ffn, bias=False)
        self.up_proj   = nn.Linear(config.d_model, config.d_ffn, bias=False)
        self.down_proj = nn.Linear(config.d_ffn, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ─────────────────────────────────────────────────────────────────────────────
# Causal Multi-Head Attention
# ─────────────────────────────────────────────────────────────────────────────

class CausalMultiHeadAttention(nn.Module):
    """
    Causal multi-head self-attention with RoPE.

    The key research requirement: raw attention weights (post-softmax,
    pre-dropout) are stored in self.last_attn_weights when return_attention=True
    is passed to forward(). Shape: (batch, n_heads, seq_len, seq_len).
    This tensor is detached from the computation graph to avoid memory leaks.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads:  int = config.n_heads
        self.head_dim: int = config.head_dim
        self.d_model:  int = config.d_model
        self.scale:  float = math.sqrt(self.head_dim)

        # Fused QKV projection + output projection, no bias
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj  = nn.Linear(config.d_model, config.d_model, bias=False)

        # Attention dropout (kept at 0.0 for clean probing)
        self.attn_drop = nn.Dropout(config.attn_dropout)

        # Storage for raw attention weights — populated when return_attention=True
        self.last_attn_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x:                (batch, seq_len, d_model)
            cos:              (seq_len, head_dim)
            sin:              (seq_len, head_dim)
            return_attention: if True, store raw attention weights in
                              self.last_attn_weights (detached, no grad)

        Returns:
            out: (batch, seq_len, d_model)
        """

        B, T, C = x.shape

        # Project to Q, K, V — shape each: (B, T, d_model)
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.d_model, dim=-1)

        # Reshape to (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to queries and keys
        q, k = apply_rope(q, k, cos, sin)

        # Scaled dot-product attention scores: (B, n_heads, T, T)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Causal mask — upper triangle set to -inf
        causal_mask = torch.triu(
            torch.full((T, T), float("-inf"), device=x.device), diagonal=1
        )
        attn_scores = attn_scores + causal_mask

        # Softmax — this is the raw attention weight matrix we want to capture
        attn_weights = F.softmax(attn_scores, dim=-1)   # (B, n_heads, T, T)

        # Store BEFORE dropout — dropout would corrupt probing measurements
        if return_attention:
            self.last_attn_weights = attn_weights.detach().cpu()

        # Apply attention dropout (no-op if attn_dropout=0.0)
        attn_weights_dropped = self.attn_drop(attn_weights)

        # Weighted sum of values
        out = torch.matmul(attn_weights_dropped, v)     # (B, n_heads, T, head_dim)

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


# ─────────────────────────────────────────────────────────────────────────────
# Transformer Block
# ─────────────────────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """Single transformer block: pre-norm attention + pre-norm FFN."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn      = CausalMultiHeadAttention(config)
        self.ffn_norm  = RMSNorm(config.d_model)
        self.ffn       = SwiGLUFFN(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor:
        # Pre-norm → attention → residual
        x = x + self.attn(self.attn_norm(x), cos, sin, return_attention)
        # Pre-norm → FFN → residual
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Full Transformer LM
# ─────────────────────────────────────────────────────────────────────────────

class TransformerLM(nn.Module):
    """
    Decoder-only causal language model.

    Usage for standard training:
        logits = model(input_ids)

    Usage for probing (extracts all attention maps):
        logits, attn_maps = model(input_ids, return_attention=True)
        # attn_maps: List of (B, n_heads, T, T), one per layer
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head    = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: token embedding and lm_head share weights
        self.lm_head.weight = self.token_embedding.weight

        # Precomputed RoPE frequencies — registered as buffer (not a parameter)
        cos, sin = precompute_rope_freqs(
            head_dim=config.head_dim,
            max_seq_len=config.block_size,
            theta=config.rope_theta,
        )
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Standard initialization: normal for embeddings and projections."""

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)

        # Scale residual stream projections by 1/sqrt(2 * n_layers) following GPT-2
        scale = (2 * self.config.n_layers) ** -0.5
        for block in self.blocks:
            nn.init.normal_(block.attn.out_proj.weight, std=self.config.init_std * scale)
            nn.init.normal_(block.ffn.down_proj.weight, std=self.config.init_std * scale)

    def forward(
        self,
        input_ids: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Args:
            input_ids:        (batch, seq_len)
            return_attention: if True, collect and return attention maps
                              from all layers. Adds minor overhead.

        Returns:
            logits:    (batch, seq_len, vocab_size)
            attn_maps: List[(batch, n_heads, seq_len, seq_len)] or None
                       One tensor per layer, ordered from layer 0 to n_layers-1.
                       Each tensor is detached and on CPU.
        """

        B, T = input_ids.shape
        assert T <= self.config.block_size, (
            f"Sequence length {T} exceeds block_size {self.config.block_size}"
        )

        # Token embeddings: (B, T, d_model)
        x = self.token_embedding(input_ids)

        # Slice RoPE frequencies to current sequence length
        cos = self.rope_cos[:T]   # (T, head_dim)
        sin = self.rope_sin[:T]   # (T, head_dim)

        # Forward through all blocks
        for block in self.blocks:
            x = block(x, cos, sin, return_attention=return_attention)

        # Final norm and LM head
        x = self.final_norm(x)
        logits = self.lm_head(x)   # (B, T, vocab_size)

        # Collect attention maps if requested
        attn_maps: Optional[List[torch.Tensor]] = None
        if return_attention:
            attn_maps = [
                block.attn.last_attn_weights
                for block in self.blocks
            ]

        return logits, attn_maps

    def get_embedding_matrix(self) -> torch.Tensor:
        """
        Return the current token embedding weight matrix.
        Used by semantic_score in probing/scores.py — must be called
        at each checkpoint to reflect the model's evolving representations.
        Returns a detached CPU tensor of shape (vocab_size, d_model).
        """

        return self.token_embedding.weight.detach().cpu()

    def count_parameters(self) -> int:
        """Count total trainable parameters."""

        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"TransformerLM(\n"
            f"  config={self.config},\n"
            f"  actual_params={self.count_parameters() / 1e6:.2f}M\n"
            f")"
        )