"""
probing/scores.py — All five attention head scoring functions.

Each function takes attention weight tensors and returns a scalar score
in [0, 1] measuring how strongly a head exhibits a particular behavior.

Scoring functions:
  1. sink_score         — head dumps attention onto one fixed token
  2. prev_token_score   — head attends to the immediately preceding token
  3. induction_score    — head completes patterns via repeated subsequences
  4. positional_score   — head ignores content, attends by position only
  5. semantic_score     — head attends to semantically similar tokens

All functions operate on a single head's attention maps across all sequences.
The pipeline calls them per (layer, head) pair at each checkpoint.
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Type aliases
# ─────────────────────────────────────────────────────────────────────────────

# attn_head: (N, T, T) — attention weights for one head across N sequences
AttentionHead = torch.Tensor


# ─────────────────────────────────────────────────────────────────────────────
# 1. Sink Score
# ─────────────────────────────────────────────────────────────────────────────

def sink_score(attn_head: AttentionHead) -> float:
    """
    Measures if a head anchors attention to a fixed absolute key position.
    
    A true sink head consistently routes attention to the same position
    (e.g., token 0) regardless of query position or sequence content.
    This is distinct from sharpness: a prev-token head is sharp but its
    argmax slides with t, so it does NOT score high here.
    
    Normalization accounts for causal mask geometry: key j is reachable
    from exactly (T - j) query positions, so we normalize by opportunity
    before averaging.
    
    Args:
        attn_head: (N, T, T) attention tensor, rows sum to 1, causal mask applied.
    
    Returns:
        Scalar in [0, 1]. Higher = stronger fixed-position anchoring.
    """
    N, T, _ = attn_head.shape
    
    # Total attention each key position receives, summed over all queries and sequences
    # Shape: (N, T) - total attention received by each key, per sequence
    attn_per_key = attn_head.sum(dim=1)
    
    # Each key position j is reachable from (T - j) query positions
    # valid_counts[j] = number of query rows that can attend to key j
    valid_counts = torch.arange(T, 0, -1, device=attn_head.device, dtype=torch.float32)
    
    # Normalize: expected uniform contribution to key j is 1/T per query,
    # so total expected is valid_counts[j] / T. We divide by valid_counts
    # to get mean attention per reachable query, then compare to uniform (1/T).
    normalized_attn = attn_per_key / valid_counts.unsqueeze(0)  # (N, T)
    
    # Average across sequences, then take the max over key positions
    mean_normalized = normalized_attn.mean(dim=0)  # (T,)
    
    return float(mean_normalized.max().item())


# ─────────────────────────────────────────────────────────────────────────────
# 2. Previous-Token Score
# ─────────────────────────────────────────────────────────────────────────────

def prev_token_score(attn_head: AttentionHead) -> float:
    """
    Measure how strongly this head attends to the immediately preceding token.

    Computation:
        For each (sequence, query position t > 0):
            extract attention_weight[t, t-1]
        Average across all valid positions and all sequences.

    Position 0 is excluded — it has no previous token.

    Args:
        attn_head: (N, T, T) attention weights

    Returns:
        prev_token_score in [0, 1]
    """

    # Extract the sub-diagonal: attn[n, t, t-1] for t >= 1
    # attn_head[:, 1:, :] → (N, T-1, T)
    # We want position t-1 for each query t
    # Gather along key dimension using index t-1 for each row t
    N, T, _ = attn_head.shape

    # attn_head[:, 1:, :]  → queries at positions 1..T-1
    # For query t (0-indexed into the slice, actual t+1), key index is t
    queries = attn_head[:, 1:, :]           # (N, T-1, T)
    # key index for query at position t+1 is t
    key_indices = torch.arange(T - 1, device=attn_head.device)  # (T-1,)
    # Gather: for each (n, t), get queries[n, t, t]
    prev_weights = queries[:, key_indices, key_indices]  # (N, T-1)

    return float(prev_weights.mean().item())


# ─────────────────────────────────────────────────────────────────────────────
# 3. Induction Score
# ─────────────────────────────────────────────────────────────────────────────

def induction_score(
    attn_head:    AttentionHead,
    induction_p1: torch.Tensor,
    induction_p2: torch.Tensor,
) -> float:
    """
    Measure pattern-completion behavior using engineered repeated subsequences.

    Exact computation:
        For each induction probe sequence i:
            p1 = stored start index of first repeated subsequence occurrence
            p2 = stored start index of second repeated subsequence occurrence
            score_i = attention_weight[i, p2, p1 + 1]
            (at the second occurrence of the prefix, how much does the head
             attend to the token that FOLLOWED the first occurrence?)
        induction_score = mean over all 100 probe sequences

    An induction head recognizes "I've seen this prefix before" and attends
    to what came next — completing the pattern.

    Args:
        attn_head:    (N, T, T) — attention maps for induction probe sequences
        induction_p1: (N,) int64 — first occurrence start indices
        induction_p2: (N,) int64 — second occurrence start indices

    Returns:
        induction_score in [0, 1]
    """

    N = attn_head.shape[0]
    scores = torch.zeros(N, dtype=torch.float32)

    for i in range(N):
        p1 = induction_p1[i].item()
        p2 = induction_p2[i].item()

        # Bounds check — skip malformed probes
        key_idx = p1 + 1
        if p2 >= attn_head.shape[1] or key_idx >= attn_head.shape[2]:
            continue

        scores[i] = attn_head[i, p2, key_idx]

    return float(scores.mean().item())


def natural_induction_score(
    attn_head: AttentionHead,
    induction_p1: torch.Tensor,
    induction_p2: torch.Tensor,
) -> float:
    """
    Same scoring rule as induction_score, but evaluated on naturally occurring
    repeated subsequences rather than engineered repeats.
    """

    return induction_score(attn_head, induction_p1, induction_p2)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Positional Score
# ─────────────────────────────────────────────────────────────────────────────

def positional_score(
    attn_head:       AttentionHead,
    positional_pairs: torch.Tensor,
) -> float:
    """
    Measure content-invariance of attention patterns.

    A positional head attends based purely on position offsets, not token
    content — so its attention pattern is nearly identical for two different
    sequences of the same length.

    Exact computation:
        For each of the 50 sequence pairs (A, B):
            attn_A = attn_head[pair[0]]   (T, T)
            attn_B = attn_head[pair[1]]   (T, T)
            For each query row t:
                kl_t = KL_divergence(attn_A[t] || attn_B[t] + 1e-9)
            row_kl = mean over rows
            pair_score = clip(1 - row_kl, 0, 1)
        positional_score = mean over all 50 pairs

    Args:
        attn_head:        (N, T, T) — attention maps for positional probe sequences
                          N = 2 * n_pairs (pairs stored flat)
        positional_pairs: (n_pairs, 2) int64 — row index pairs into attn_head

    Returns:
        positional_score in [0, 1]
    """

    n_pairs = positional_pairs.shape[0]
    pair_scores = torch.zeros(n_pairs, dtype=torch.float32)

    for i in range(n_pairs):
        idx_a = positional_pairs[i, 0].item()
        idx_b = positional_pairs[i, 1].item()

        attn_a = attn_head[idx_a]   # (T, T)
        attn_b = attn_head[idx_b]   # (T, T)

        # Add small epsilon to avoid log(0) in KL computation
        attn_b_smooth = attn_b + 1e-9
        attn_b_smooth = attn_b_smooth / attn_b_smooth.sum(dim=-1, keepdim=True)

        # KL divergence per row: sum(P * log(P/Q))
        # Using PyTorch's kl_div which expects log-space input for predictions
        log_b = attn_b_smooth.log()
        # kl_div(input=log_Q, target=P) = P * (log_P - log_Q)
        kl_per_row = F.kl_div(
            log_b,
            attn_a,
            reduction="none",
            log_target=False,
        ).sum(dim=-1)   # (T,)

        row_kl     = kl_per_row.mean().item()
        pair_scores[i] = max(0.0, 1.0 - row_kl)

    return float(pair_scores.mean().item())


# ─────────────────────────────────────────────────────────────────────────────
# 5. Semantic Score
# ─────────────────────────────────────────────────────────────────────────────

def semantic_score_detailed(
    attn_head:        AttentionHead,
    token_ids:        torch.Tensor,
    embedding_matrix: torch.Tensor,
) -> Dict[str, float | int | bool]:
    """
    Measure alignment between attention weights and semantic token similarity.

    A semantic head attends to tokens whose meaning is similar to the query
    token — measured by cosine similarity in the model's own embedding space.

    CRITICAL: embedding_matrix must be the model's OWN embedding matrix
    at the current checkpoint — not any frozen external embedding.
    This ensures we measure alignment with the model's evolving representations.

    Exact computation:
        For each sequence s, for each query position i (causal: j <= i only):
            embed_i   = embedding_matrix[token_ids[s, i]]         (D,)
            embed_all = embedding_matrix[token_ids[s, 0:i+1]]     (i+1, D)
            sim_vec   = cosine_similarity(embed_i, embed_all)      (i+1,)
            attn_vec  = attn_head[s, i, 0:i+1]                    (i+1,)
            pos_score = Pearson_correlation(attn_vec, sim_vec)
            (skip positions i < 4 to avoid degenerate short vectors)
        semantic_score = mean over all valid (s, i) pairs

    Args:
        attn_head:        (N, T, T) — attention maps for general probe sequences
        token_ids:        (N, T)    — token ids for general probe sequences
        embedding_matrix: (V, D)    — model's current token embedding weights

    Returns:
        Dict containing:
          - score: mean Pearson correlation across valid positions or NaN if undefined
          - n_candidate_positions: total (sequence, query-position) pairs considered
          - n_used_positions: pairs that contributed finite correlations
          - n_short_mask_positions: pairs excluded because fewer than 6 keys remained
          - n_zero_variance_positions: pairs excluded because attention or similarity
            variance collapsed after masking
          - n_nonfinite_positions: pairs excluded due to NaN/Inf correlation
          - valid_fraction: n_used_positions / n_candidate_positions
          - is_defined: whether at least one valid correlation was observed
    """

    N, T = token_ids.shape
    
    # Normalize embedding matrix rows for efficient cosine similarity
    embed_norm = F.normalize(embedding_matrix.float(), dim=-1)  # (V, D)
    
    # Batch lookup: get embeddings for all tokens in all sequences
    seq_embeds = embed_norm[token_ids]  # (N, T, D)
    
    # Compute cosine similarity matrix for each sequence
    # sim[s, i, j] = dot(embed[s, i], embed[s, j])
    sim_matrix = torch.bmm(seq_embeds, seq_embeds.transpose(1, 2))  # (N, T, T)
    
    # Skip positions 0-3 (degenerate short vectors)
    valid_positions = torch.arange(4, T, device=attn_head.device)
    
    # Extract valid attention and similarity vectors
    # For each position i >= 4, we want attn[s, i, :i+1] and sim[s, i, :i+1]
    correlations = []
    n_candidate_positions = int(N * max(T - 4, 0))
    n_used_positions = 0
    n_short_mask_positions = 0
    n_zero_variance_positions = 0
    n_nonfinite_positions = 0
    
    for i in valid_positions:
        i_int = i.item()
        # Extract causal window for all sequences at position i
        attn_window = attn_head[:, i_int, :i_int + 1]  # (N, i+1)
        sim_window = sim_matrix[:, i_int, :i_int + 1]  # (N, i+1)
        
        # --- EXCLUSION MASK ---
        # Remove positions with structural confounds before computing Pearson.
        # These positions produce high cosine similarity for non-semantic reasons.
        mask = torch.ones(i_int + 1, dtype=torch.bool, device=attn_head.device)
        mask[i_int] = False          # j=t: identity, cosine sim always 1.0
        if i_int > 0:
            mask[i_int - 1] = False  # j=t-1: prev-token confound
        mask[0] = False              # j=0: sink confound
        
        # Require minimum 6 valid points for stable Pearson
        if mask.sum().item() < 6:
            n_short_mask_positions += int(N)
            continue
        
        attn_window = attn_window[:, mask]  # apply mask to attention slice
        sim_window = sim_window[:, mask]    # apply mask to similarity slice
        # --- END EXCLUSION MASK ---
        
        # Compute Pearson correlation for each sequence at this position
        # Pearson(x, y) = cov(x, y) / (std(x) * std(y))
        attn_mean = attn_window.mean(dim=1, keepdim=True)  # (N, 1)
        sim_mean = sim_window.mean(dim=1, keepdim=True)    # (N, 1)
        
        attn_centered = attn_window - attn_mean  # (N, masked_len)
        sim_centered = sim_window - sim_mean     # (N, masked_len)
        
        attn_std = attn_centered.std(dim=1, unbiased=False)  # (N,)
        sim_std = sim_centered.std(dim=1, unbiased=False)    # (N,)
        
        # Compute covariance
        cov = (attn_centered * sim_centered).mean(dim=1)  # (N,)
        
        # Compute correlation, filtering out degenerate cases
        valid_mask = (attn_std > 1e-8) & (sim_std > 1e-8)
        n_zero_variance_positions += int((~valid_mask).sum().item())
        if valid_mask.any():
            corr = cov[valid_mask] / (attn_std[valid_mask] * sim_std[valid_mask])
            # Filter out NaN/Inf
            finite_mask = torch.isfinite(corr)
            n_nonfinite_positions += int((~finite_mask).sum().item())
            if finite_mask.any():
                finite_corr = corr[finite_mask]
                correlations.append(finite_corr)
                n_used_positions += int(finite_corr.numel())

    if len(correlations) == 0:
        score = math.nan
    else:
        all_corr = torch.cat(correlations)
        score = float(all_corr.mean().item())

    valid_fraction = (
        float(n_used_positions) / float(n_candidate_positions)
        if n_candidate_positions > 0
        else 0.0
    )
    return {
        "score": score,
        "n_candidate_positions": n_candidate_positions,
        "n_used_positions": n_used_positions,
        "n_short_mask_positions": n_short_mask_positions,
        "n_zero_variance_positions": n_zero_variance_positions,
        "n_nonfinite_positions": n_nonfinite_positions,
        "valid_fraction": valid_fraction,
        "is_defined": bool(n_used_positions > 0),
    }


def semantic_score(
    attn_head:        AttentionHead,
    token_ids:        torch.Tensor,
    embedding_matrix: torch.Tensor,
) -> float:
    """
    Backward-compatible scalar wrapper around `semantic_score_detailed`.

    Undefined semantic measurements return 0.0 here for legacy callers, but the
    probing pipeline uses the detailed variant so undefinedness is preserved in
    saved metadata and downstream classification.
    """

    details = semantic_score_detailed(attn_head, token_ids, embedding_matrix)
    score = float(details["score"])
    return 0.0 if not math.isfinite(score) else score


def score_head_detailed(
    general_attn:     AttentionHead,
    induction_attn:   AttentionHead,
    positional_attn:  AttentionHead,
    induction_p1:     torch.Tensor,
    induction_p2:     torch.Tensor,
    positional_pairs: torch.Tensor,
    token_ids:        torch.Tensor,
    embedding_matrix: torch.Tensor,
) -> Tuple[Tuple[float, float, float, float, float], Dict[str, float | int | bool]]:
    """Compute all five scores plus semantic measurement diagnostics."""

    s_sink = sink_score(general_attn)
    s_prev = prev_token_score(general_attn)
    s_ind = induction_score(induction_attn, induction_p1, induction_p2)
    s_pos = positional_score(positional_attn, positional_pairs)
    semantic_details = semantic_score_detailed(general_attn, token_ids, embedding_matrix)
    s_sem = float(semantic_details["score"])
    return (s_sink, s_prev, s_ind, s_pos, s_sem), semantic_details


# ─────────────────────────────────────────────────────────────────────────────
# Batch scorer — compute all 5 scores for one head across all probe types
# ─────────────────────────────────────────────────────────────────────────────

def score_head(
    general_attn:     AttentionHead,
    induction_attn:   AttentionHead,
    positional_attn:  AttentionHead,
    induction_p1:     torch.Tensor,
    induction_p2:     torch.Tensor,
    positional_pairs: torch.Tensor,
    token_ids:        torch.Tensor,
    embedding_matrix: torch.Tensor,
) -> Tuple[float, float, float, float, float]:
    """
    Compute all 5 scores for a single attention head.

    Args:
        general_attn:     (500, T, T) — head maps on general sequences
        induction_attn:   (100, T, T) — head maps on induction sequences
        positional_attn:  (100, T, T) — head maps on positional sequences
        induction_p1:     (100,) — first occurrence indices
        induction_p2:     (100,) — second occurrence indices
        positional_pairs: (50, 2) — pair indices into positional_attn
        token_ids:        (500, T) — token ids for general sequences
        embedding_matrix: (V, D)  — current model embeddings

    Returns:
        (sink, prev_token, induction, positional, semantic) scores
    """

    scores, semantic_details = score_head_detailed(
        general_attn,
        induction_attn,
        positional_attn,
        induction_p1,
        induction_p2,
        positional_pairs,
        token_ids,
        embedding_matrix,
    )
    s_sink, s_prev, s_ind, s_pos, s_sem = scores
    return (
        s_sink,
        s_prev,
        s_ind,
        s_pos,
        0.0 if not bool(semantic_details["is_defined"]) else s_sem,
    )
