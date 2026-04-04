"""
analysis/stability.py — Head type stability and sink persistence analysis.

Computes:
  1. TYPE CHANGE MATRIX: for every (seed, layer, head), how many times
     does the type label change across consecutive checkpoints?
     Tests H5: sinks are attractors — they almost never change type.

  2. SINK PERSISTENCE: for every head that was ever labeled SINK,
     what fraction of its subsequent checkpoints remained SINK?
     Direct operational test of H5.

  3. STABILITY HISTOGRAM: distribution of type-change counts across all heads,
     for use in the stability_hist.py visualization.
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from probing import HEAD_TYPES, LABEL_SINK


# ─────────────────────────────────────────────────────────────────────────────
# Type change matrix
# ─────────────────────────────────────────────────────────────────────────────

def compute_type_change_matrix(
    run_results: List[Dict],
) -> np.ndarray:
    """
    For each (seed, layer, head), count the number of label changes
    across consecutive checkpoints.

    A "change" is any transition where label[t] != label[t+1].

    Args:
        run_results: list of result dicts, one per seed

    Returns:
        change_matrix: (n_seeds, n_layers, n_heads) int32
                       entry [s, l, h] = number of type transitions for head (l,h)
                       in run s
    """

    n_seeds  = len(run_results)
    n_layers = run_results[0]["n_layers"]
    n_heads  = run_results[0]["n_heads"]

    change_matrix = np.zeros((n_seeds, n_layers, n_heads), dtype=np.int32)

    for s_idx, result in enumerate(run_results):
        labels = result["label_tensor"]   # (n_ckpts, n_layers, n_heads)
        n_ckpts = labels.shape[0]

        # Compute differences between consecutive checkpoints
        # diff[t] = 1 if label changed between checkpoint t and t+1
        diffs = (labels[1:] != labels[:-1]).int()    # (n_ckpts-1, n_layers, n_heads)
        change_matrix[s_idx] = diffs.sum(dim=0).numpy()   # (n_layers, n_heads)

    return change_matrix


# ─────────────────────────────────────────────────────────────────────────────
# Sink persistence
# ─────────────────────────────────────────────────────────────────────────────

def compute_sink_persistence(
    run_results: List[Dict],
) -> Dict[str, object]:
    """
    For every head that was ever labeled SINK, compute the fraction of
    subsequent checkpoints where it remained SINK.

    Definition:
        For head (l, h) in seed s:
            first_sink = first checkpoint where label == SINK
            if first_sink exists:
                subsequent = labels[first_sink:, l, h]
                persistence = (subsequent == SINK).mean()

    If a head was SINK at the very last checkpoint but never before,
    it has 0 subsequent checkpoints — excluded.

    Args:
        run_results: list of result dicts, one per seed

    Returns:
        dict with keys:
            "persistence_scores": List[float] — one per head that was ever SINK
            "mean_persistence":   float
            "std_persistence":    float
            "n_ever_sink":        int — number of heads that were ever SINK
            "n_total_heads":      int
            "per_seed_mean":      List[float] — mean persistence per seed
    """

    persistence_scores: List[float] = []
    per_seed_means:     List[float] = []

    for result in run_results:
        labels   = result["label_tensor"]   # (n_ckpts, n_layers, n_heads)
        n_ckpts, n_layers, n_heads = labels.shape
        seed_scores: List[float] = []

        for layer in range(n_layers):
            for head in range(n_heads):
                traj = labels[:, layer, head]   # (n_ckpts,)

                # Find first checkpoint labeled SINK
                sink_steps = (traj == LABEL_SINK).nonzero(as_tuple=True)[0]
                if len(sink_steps) == 0:
                    continue

                first_sink = sink_steps[0].item()

                # Must have at least one subsequent checkpoint
                if first_sink >= n_ckpts - 1:
                    continue

                subsequent = traj[first_sink:]               # (n_ckpts - first_sink,)
                persistence = float((subsequent == LABEL_SINK).float().mean().item())

                seed_scores.append(persistence)
                persistence_scores.append(persistence)

        per_seed_means.append(float(np.mean(seed_scores)) if seed_scores else 0.0)

    if not persistence_scores:
        return {
            "persistence_scores": [],
            "mean_persistence":   0.0,
            "std_persistence":    0.0,
            "n_ever_sink":        0,
            "n_total_heads": (
                run_results[0]["n_layers"] * run_results[0]["n_heads"]
                * len(run_results)
            ),
            "per_seed_mean": per_seed_means,
        }

    return {
        "persistence_scores": persistence_scores,
        "mean_persistence":   float(np.mean(persistence_scores)),
        "std_persistence":    float(np.std(persistence_scores)),
        "n_ever_sink":        len(persistence_scores),
        "n_total_heads": (
            run_results[0]["n_layers"] * run_results[0]["n_heads"] * len(run_results)
        ),
        "per_seed_mean": per_seed_means,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stability histogram data
# ─────────────────────────────────────────────────────────────────────────────

def compute_stability_histogram(
    change_matrix: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Compute the distribution of type-change counts across all heads
    and seeds, for use in the stability histogram figure.

    Args:
        change_matrix: (n_seeds, n_layers, n_heads) int32

    Returns:
        dict with keys:
            "flat_counts": (n_seeds * n_layers * n_heads,) — raw change counts
            "bins":        (max_count + 2,) — histogram bin edges
            "hist":        (max_count + 1,) — frequency counts per bin
            "mean":        float
            "median":      float
            "pct_zero":    float — fraction of heads with zero changes
            "pct_one":     float — fraction with exactly 1 change
    """

    flat = change_matrix.flatten()
    max_count = int(flat.max())

    bins = np.arange(0, max_count + 2)
    hist, _ = np.histogram(flat, bins=bins)

    return {
        "flat_counts": flat,
        "bins":        bins,
        "hist":        hist,
        "mean":        float(flat.mean()),
        "median":      float(np.median(flat)),
        "pct_zero":    float((flat == 0).mean()),
        "pct_one":     float((flat == 1).mean()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Type-specific stability
# ─────────────────────────────────────────────────────────────────────────────

def compute_per_type_stability(
    run_results:    List[Dict],
    change_matrix:  np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """
    For each final head type, compute the mean number of type changes
    that heads of that type underwent during training.

    "Final type" = label at the last checkpoint.

    Args:
        run_results:   list of result dicts, one per seed
        change_matrix: (n_seeds, n_layers, n_heads) from compute_type_change_matrix

    Returns:
        per_type: dict mapping type_name → {"mean_changes": float, "n_heads": int}
    """

    per_type: Dict[str, Dict[str, float]] = {
        name: {"mean_changes": 0.0, "n_heads": 0, "_sum": 0.0}
        for name in HEAD_TYPES
    }

    for s_idx, result in enumerate(run_results):
        labels   = result["label_tensor"]   # (n_ckpts, n_layers, n_heads)
        n_layers = result["n_layers"]
        n_heads  = result["n_heads"]

        # Final label for each head
        final_labels = labels[-1]    # (n_layers, n_heads)

        for layer in range(n_layers):
            for head in range(n_heads):
                final_type = int(final_labels[layer, head].item())
                type_name  = HEAD_TYPES[final_type]
                n_changes  = int(change_matrix[s_idx, layer, head])

                per_type[type_name]["n_heads"] += 1
                per_type[type_name]["_sum"]    += n_changes

    # Compute means
    for type_name in HEAD_TYPES:
        n = per_type[type_name]["n_heads"]
        s = per_type[type_name]["_sum"]
        per_type[type_name]["mean_changes"] = s / n if n > 0 else 0.0
        del per_type[type_name]["_sum"]

    return per_type


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def print_stability_report(
    change_matrix:    np.ndarray,
    sink_persistence: Dict[str, object],
    per_type_stab:    Dict[str, Dict[str, float]],
) -> None:
    """Print a formatted stability analysis report."""

    hist_data = compute_stability_histogram(change_matrix)

    print(f"\n{'=' * 64}")
    print(f"  Stability Analysis Report")
    print(f"{'=' * 64}")

    print(f"\n  Type-change distribution:")
    print(f"  {'─' * 50}")
    print(f"  Mean changes per head  : {hist_data['mean']:.2f}")
    print(f"  Median changes         : {hist_data['median']:.1f}")
    print(f"  Heads with 0 changes   : {hist_data['pct_zero'] * 100:.1f}%")
    print(f"  Heads with 1 change    : {hist_data['pct_one'] * 100:.1f}%")

    print(f"\n  Mean type changes by final head type:")
    print(f"  {'─' * 50}")
    for type_name, stats in per_type_stab.items():
        if type_name in {"WEAK", "AMBIGUOUS", "UNDIFFERENTIATED"}:
            continue
        n = stats["n_heads"]
        m = stats["mean_changes"]
        print(f"  {type_name:<20}: {m:.2f} avg changes  ({n} heads)")

    print(f"\n  Sink persistence (H5):")
    print(f"  {'─' * 50}")
    n_sink  = sink_persistence["n_ever_sink"]
    n_total = sink_persistence["n_total_heads"]
    mean_p  = sink_persistence["mean_persistence"]
    std_p   = sink_persistence["std_persistence"]
    print(f"  Heads ever labeled SINK : {n_sink} / {n_total}")
    print(f"  Mean persistence        : {mean_p:.3f} ± {std_p:.3f}")
    print(f"  Per-seed means          : "
          f"{[f'{v:.3f}' for v in sink_persistence['per_seed_mean']]}")

    if mean_p >= 0.8:
        verdict = "H5 SUPPORTED — sinks are stable attractors"
    elif mean_p >= 0.6:
        verdict = "H5 PARTIALLY SUPPORTED — sinks are moderately stable"
    else:
        verdict = "H5 NOT SUPPORTED — sinks are unstable"
    print(f"\n  Verdict: {verdict}")
    print(f"{'=' * 64}\n")
