"""
analysis/trajectories.py — Developmental trajectory computation.

Computes three levels of analysis from the classification matrix:

1. GLOBAL CURVES: fraction of all heads in each type at each training step,
   averaged across seeds with ±1 std confidence bands.

2. PER-LAYER CURVES: same computation restricted to each layer separately.
   Used to test H3 (layer stratification — lower layers specialize earlier).

3. HEAD-LEVEL TRAJECTORIES: the full sequence of type labels for each
   individual (layer, head) pair across all checkpoints. Enables inspection
   of individual developmental paths (e.g. UNDIFF → SINK → SEMANTIC).
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from probing import HEAD_TYPES, LABEL_UNDIFF


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helper
# ─────────────────────────────────────────────────────────────────────────────

def load_run_results(results_path: Path) -> Dict:
    """
    Load a saved probing results file for one run.

    Returns dict with keys:
        label_tensor: (n_ckpts, n_layers, n_heads) int32
        score_tensor: (n_ckpts, n_layers, n_heads, 5) float32
        step_index:   List[int]
        seed:         int
        n_layers:     int
        n_heads:      int
    """

    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    return torch.load(results_path, weights_only=True)


# ─────────────────────────────────────────────────────────────────────────────
# Global curves
# ─────────────────────────────────────────────────────────────────────────────

def compute_global_curves(
    run_results: List[Dict],
) -> Dict[str, np.ndarray]:
    """
    Compute global head-type fraction curves across all layers and heads,
    averaged over seeds with ±1 std confidence bands.

    Args:
        run_results: list of result dicts, one per seed (from load_run_results)

    Returns:
        dict with keys:
            "steps":        (n_ckpts,)          — training steps (from first seed)
            "mean":         (n_ckpts, 6)         — mean fraction per type
            "std":          (n_ckpts, 6)         — std across seeds (0 if 1 seed)
            "per_seed":     (n_seeds, n_ckpts, 6)— per-seed curves before averaging
            "type_names":   List[str]            — HEAD_TYPES for axis labeling
    """

    n_seeds = len(run_results)
    # Align on the shortest step_index (in case runs have different lengths)
    min_ckpts = min(len(r["step_index"]) for r in run_results)
    steps     = np.array(run_results[0]["step_index"][:min_ckpts])

    n_types = 6
    per_seed = np.zeros((n_seeds, min_ckpts, n_types), dtype=np.float32)

    for s_idx, result in enumerate(run_results):
        labels = result["label_tensor"][:min_ckpts]   # (n_ckpts, n_layers, n_heads)
        n_ckpts_run, n_layers, n_heads = labels.shape
        total_heads = n_layers * n_heads

        for ckpt in range(n_ckpts_run):
            flat_labels = labels[ckpt].flatten()      # (n_layers * n_heads,)
            for t in range(n_types):
                per_seed[s_idx, ckpt, t] = float((flat_labels == t).sum()) / total_heads

    mean = per_seed.mean(axis=0)                      # (n_ckpts, 6)
    std  = per_seed.std(axis=0) if n_seeds > 1 else np.zeros_like(mean)

    return {
        "steps":      steps,
        "mean":       mean,
        "std":        std,
        "per_seed":   per_seed,
        "type_names": HEAD_TYPES,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-layer curves
# ─────────────────────────────────────────────────────────────────────────────

def compute_per_layer_curves(
    run_results: List[Dict],
) -> Dict[str, object]:
    """
    Compute head-type fraction curves separately for each layer.

    Used to test H3: do lower layers specialize earlier than higher layers?

    Args:
        run_results: list of result dicts, one per seed

    Returns:
        dict with keys:
            "steps":        (n_ckpts,)
            "n_layers":     int
            "per_layer_mean": (n_layers, n_ckpts, 6) — mean across seeds
            "per_layer_std":  (n_layers, n_ckpts, 6) — std across seeds
            "type_names":   List[str]
    """

    n_seeds   = len(run_results)
    min_ckpts = min(len(r["step_index"]) for r in run_results)
    steps     = np.array(run_results[0]["step_index"][:min_ckpts])
    n_layers  = run_results[0]["n_layers"]
    n_heads   = run_results[0]["n_heads"]
    n_types   = 6

    # per_seed_per_layer: (n_seeds, n_layers, n_ckpts, n_types)
    per_seed_layer = np.zeros(
        (n_seeds, n_layers, min_ckpts, n_types), dtype=np.float32
    )

    for s_idx, result in enumerate(run_results):
        labels = result["label_tensor"][:min_ckpts]   # (n_ckpts, n_layers, n_heads)

        for layer in range(n_layers):
            layer_labels = labels[:, layer, :]        # (n_ckpts, n_heads)

            for ckpt in range(min_ckpts):
                flat = layer_labels[ckpt]              # (n_heads,)
                for t in range(n_types):
                    per_seed_layer[s_idx, layer, ckpt, t] = (
                        float((flat == t).sum()) / n_heads
                    )

    per_layer_mean = per_seed_layer.mean(axis=0)      # (n_layers, n_ckpts, 6)
    per_layer_std  = (
        per_seed_layer.std(axis=0) if n_seeds > 1
        else np.zeros_like(per_layer_mean)
    )

    return {
        "steps":          steps,
        "n_layers":       n_layers,
        "per_layer_mean": per_layer_mean,
        "per_layer_std":  per_layer_std,
        "type_names":     HEAD_TYPES,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Head-level trajectories
# ─────────────────────────────────────────────────────────────────────────────

def compute_head_trajectories(
    result: Dict,
) -> Dict[Tuple[int, int], List[int]]:
    """
    Extract the full label sequence for every (layer, head) pair across
    all checkpoints, for a single seed.

    Args:
        result: single run result dict

    Returns:
        trajectories: dict mapping (layer, head) → List[int] of type labels
                      across checkpoints.
                      e.g. {(2, 5): [0, 0, 1, 1, 1, 4, 4, 5, 5, ...]}
    """

    labels   = result["label_tensor"]     # (n_ckpts, n_layers, n_heads)
    n_ckpts, n_layers, n_heads = labels.shape

    trajectories: Dict[Tuple[int, int], List[int]] = {}

    for layer in range(n_layers):
        for head in range(n_heads):
            traj = labels[:, layer, head].tolist()
            trajectories[(layer, head)] = [int(t) for t in traj]

    return trajectories


def find_interesting_trajectories(
    trajectories: Dict[Tuple[int, int], List[int]],
    min_type_changes: int = 2,
) -> Dict[Tuple[int, int], List[int]]:
    """
    Filter for heads that undergo at least min_type_changes type transitions.
    These are the most scientifically interesting — heads that re-specialize.

    Args:
        trajectories:     output of compute_head_trajectories
        min_type_changes: minimum number of type label changes to include

    Returns:
        subset of trajectories dict with only "interesting" heads
    """

    interesting = {}
    for key, traj in trajectories.items():
        changes = sum(
            1 for i in range(1, len(traj)) if traj[i] != traj[i - 1]
        )
        if changes >= min_type_changes:
            interesting[key] = traj

    return interesting


# ─────────────────────────────────────────────────────────────────────────────
# Specialization timing
# ─────────────────────────────────────────────────────────────────────────────

def compute_specialization_onset(
    global_curves: Dict[str, np.ndarray],
    threshold_frac: float = 0.05,
) -> Dict[str, Optional[int]]:
    """
    For each head type, find the first training step at which its global
    fraction exceeds threshold_frac (default 5% of all heads).

    This operationalizes the ordering claims in H1 and H2:
    "sinks appear first" = sinks exceed 5% before any other type does.

    Args:
        global_curves:  output of compute_global_curves
        threshold_frac: fraction of all heads that must be of this type
                        before we say the type has "appeared"

    Returns:
        onset_steps: dict mapping type name → training step of onset,
                     or None if the type never reaches threshold_frac
    """

    steps      = global_curves["steps"]     # (n_ckpts,)
    mean_fracs = global_curves["mean"]      # (n_ckpts, 6)
    type_names = global_curves["type_names"]

    onset_steps: Dict[str, Optional[int]] = {}

    for t_idx, type_name in enumerate(type_names):
        if type_name == "UNDIFFERENTIATED":
            onset_steps[type_name] = int(steps[0])
            continue

        fracs = mean_fracs[:, t_idx]
        above = np.where(fracs >= threshold_frac)[0]

        if len(above) == 0:
            onset_steps[type_name] = None
        else:
            onset_steps[type_name] = int(steps[above[0]])

    return onset_steps


def print_trajectory_report(
    global_curves:   Dict[str, np.ndarray],
    per_layer_curves: Dict[str, object],
    onset_steps:     Dict[str, Optional[int]],
    seed:            int,
) -> None:
    """Print a formatted summary of trajectory analysis results."""

    steps = global_curves["steps"]
    mean  = global_curves["mean"]
    std   = global_curves["std"]

    print(f"\n{'=' * 64}")
    print(f"  Trajectory Analysis Report — Seed {seed}")
    print(f"{'=' * 64}")

    print(f"\n  Specialization onset (first step with ≥5% of heads):")
    print(f"  {'─' * 50}")

    # Sort by onset step for easy ordering inspection
    sorted_types = sorted(
        [(k, v) for k, v in onset_steps.items() if k != "UNDIFFERENTIATED"],
        key=lambda x: (x[1] is None, x[1] or 0)
    )
    for type_name, step in sorted_types:
        step_str = f"{step:>10,}" if step is not None else "      never"
        print(f"  {type_name:<20}: {step_str}")

    print(f"\n  Final checkpoint fractions (mean ± std across seeds):")
    print(f"  {'─' * 50}")
    for t_idx, type_name in enumerate(HEAD_TYPES):
        if type_name == "UNDIFFERENTIATED":
            continue
        m = mean[-1, t_idx]
        s = std[-1, t_idx]
        bar = "█" * int(m * 40)
        print(f"  {type_name:<20}: {m:.3f} ± {s:.3f}  {bar}")

    print(f"\n  Layer stratification (step at which each layer reaches")
    print(f"  ≥50% specialized heads):")
    print(f"  {'─' * 50}")
    n_layers       = per_layer_curves["n_layers"]
    per_layer_mean = per_layer_curves["per_layer_mean"]  # (n_layers, n_ckpts, 6)
    layer_steps    = per_layer_curves["steps"]

    for layer in range(n_layers):
        # Fraction of non-UNDIFF heads in this layer at each step
        undiff_frac = per_layer_mean[layer, :, LABEL_UNDIFF]  # (n_ckpts,)
        spec_frac   = 1.0 - undiff_frac
        above_50    = np.where(spec_frac >= 0.5)[0]
        onset       = int(layer_steps[above_50[0]]) if len(above_50) > 0 else None
        onset_str   = f"{onset:>8,}" if onset is not None else "   never"
        print(f"  Layer {layer:<3}: {onset_str}")

    print(f"\n{'=' * 64}\n")
