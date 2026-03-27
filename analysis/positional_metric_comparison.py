#!/usr/bin/env python3
"""
Compare candidate positional metrics on synthetic attention patterns.

This is an evaluation script, not production scoring code. It compares:
  1. Current production metric: clip(1 - mean(KL), 0, 1)
  2. Jensen-Shannon similarity
  3. Mean rowwise cosine similarity

Usage:
    python analysis/positional_metric_comparison.py
"""

import sys
from pathlib import Path
from typing import Callable, Dict

import numpy as np
import torch
import torch.nn.functional as F

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from probing.scores import positional_score


def _normalize_rows(attn: torch.Tensor) -> torch.Tensor:
    row_sums = attn.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return attn / row_sums


def _uniform_causal_row(t: int, T: int) -> torch.Tensor:
    row = torch.zeros(T, dtype=torch.float32)
    row[: t + 1] = 1.0 / (t + 1)
    return row


def build_positional_pattern(n_seq: int = 6, T: int = 16) -> torch.Tensor:
    """Same causal attention pattern across all sequences."""
    attn = torch.zeros(n_seq, T, T, dtype=torch.float32)
    for s in range(n_seq):
        for t in range(T):
            focus = max(0, t - 2)
            attn[s, t, focus] = 1.0
    return attn


def build_random_pattern(n_seq: int = 6, T: int = 16) -> torch.Tensor:
    """Independent random causal attention for each sequence."""
    attn = torch.zeros(n_seq, T, T, dtype=torch.float32)
    for s in range(n_seq):
        for t in range(T):
            row = torch.rand(t + 1, dtype=torch.float32)
            row = row / row.sum()
            attn[s, t, : t + 1] = row
    return attn


def build_content_dependent_pattern(n_seq: int = 6, T: int = 16) -> torch.Tensor:
    """Each sequence uses a different fixed anchor, so patterns disagree across content."""
    attn = torch.zeros(n_seq, T, T, dtype=torch.float32)
    for s in range(n_seq):
        anchor = s % max(1, T // 3)
        for t in range(T):
            key = min(anchor, t)
            attn[s, t, key] = 1.0
    return attn


def current_metric(attn_head: torch.Tensor, pairs: torch.Tensor) -> float:
    return positional_score(attn_head, pairs)


def js_similarity(attn_head: torch.Tensor, pairs: torch.Tensor) -> float:
    eps = 1e-9
    scores = []
    for idx_a, idx_b in pairs.tolist():
        attn_a = attn_head[idx_a]
        attn_b = attn_head[idx_b]
        pair_scores = []
        for row_a, row_b in zip(attn_a, attn_b):
            p = row_a + eps
            q = row_b + eps
            p = p / p.sum()
            q = q / q.sum()
            m = 0.5 * (p + q)
            js = 0.5 * (
                torch.sum(p * torch.log(p / m)) +
                torch.sum(q * torch.log(q / m))
            )
            pair_scores.append(max(0.0, 1.0 - float(js.item())))
        scores.append(float(np.mean(pair_scores)))
    return float(np.mean(scores))


def rowwise_cosine_similarity(attn_head: torch.Tensor, pairs: torch.Tensor) -> float:
    scores = []
    for idx_a, idx_b in pairs.tolist():
        attn_a = attn_head[idx_a]
        attn_b = attn_head[idx_b]
        row_scores = []
        for row_a, row_b in zip(attn_a, attn_b):
            row_scores.append(
                float(F.cosine_similarity(row_a.unsqueeze(0), row_b.unsqueeze(0)).item())
            )
        scores.append(float(np.mean(row_scores)))
    return float(np.mean(scores))


def main() -> None:
    n_seq = 6
    T = 16
    pairs = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.long)

    patterns = {
        "positional": build_positional_pattern(n_seq=n_seq, T=T),
        "random": build_random_pattern(n_seq=n_seq, T=T),
        "content_dependent": build_content_dependent_pattern(n_seq=n_seq, T=T),
    }
    patterns = {name: _normalize_rows(attn) for name, attn in patterns.items()}

    metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor], float]] = {
        "current_1_minus_kl": current_metric,
        "js_similarity": js_similarity,
        "rowwise_cosine": rowwise_cosine_similarity,
    }

    print("\nPositional Metric Comparison")
    print("=" * 72)
    print(f"{'pattern':<20} {'current_1_minus_kl':>20} {'js_similarity':>16} {'rowwise_cosine':>16}")
    print("-" * 72)
    for pattern_name, attn in patterns.items():
        scores = {metric_name: metric(attn, pairs) for metric_name, metric in metrics.items()}
        print(
            f"{pattern_name:<20} "
            f"{scores['current_1_minus_kl']:>20.4f} "
            f"{scores['js_similarity']:>16.4f} "
            f"{scores['rowwise_cosine']:>16.4f}"
        )
    print("=" * 72)


if __name__ == "__main__":
    main()
