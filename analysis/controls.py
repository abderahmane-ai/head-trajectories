"""
analysis/controls.py — Scientific controls for robustness of findings.

Computes two key controls required for the paper:

1. THRESHOLD SENSITIVITY: re-classify all heads with thresholds scaled
   by ±20% and check whether ordering conclusions from H1–H5 are preserved.
   A conclusion is "robust" if it holds under all three threshold settings.

2. INTER-SEED AGREEMENT: for each ordering claim, report the fraction of
   seed pairs that agree. With 3 seeds there are 3 pairs (C(3,2)=3).
   Reports 0/3, 1/3, 2/3, or 3/3 agreement.
"""

import itertools
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from probing import (
    THRESHOLDS, HEAD_TYPES,
    LABEL_UNDIFF, LABEL_SINK, LABEL_PREV,
    LABEL_IND, LABEL_POS, LABEL_SEM,
)
from probing.classifier import classify_head
from .trajectories import compute_specialization_onset


# ─────────────────────────────────────────────────────────────────────────────
# Re-classification with scaled thresholds
# ─────────────────────────────────────────────────────────────────────────────

def reclassify_with_scaled_thresholds(
    result:         Dict,
    scale_factor:   float,
) -> Dict:
    """
    Re-run head classification for one run using thresholds scaled by scale_factor.

    Args:
        result:       single run result dict (with score_tensor)
        scale_factor: multiply all thresholds by this value (e.g. 0.8 or 1.2)

    Returns:
        new result dict with recomputed label_tensor, same score_tensor and step_index
    """

    score_tensor = result["score_tensor"]   # (n_ckpts, n_layers, n_heads, 5)
    n_ckpts, n_layers, n_heads, _ = score_tensor.shape

    base_thresholds = np.asarray(
        result.get("effective_thresholds", result.get("thresholds", THRESHOLDS)),
        dtype=np.float32,
    )
    scaled_thresholds = base_thresholds * scale_factor
    new_labels        = torch.zeros((n_ckpts, n_layers, n_heads), dtype=torch.int32)

    for ckpt in range(n_ckpts):
        for layer in range(n_layers):
            for head in range(n_heads):
                scores = tuple(score_tensor[ckpt, layer, head].tolist())
                label, _ = classify_head(
                    scores,
                    thresholds=scaled_thresholds,
                )
                new_labels[ckpt, layer, head] = label

    return {
        "label_tensor": new_labels,
        "score_tensor": score_tensor,
        "step_index":   result["step_index"],
        "seed":         result["seed"],
        "n_layers":     n_layers,
        "n_heads":      n_heads,
        "thresholds":   scaled_thresholds.tolist(),
        "raw_thresholds": scaled_thresholds.tolist(),
        "effective_thresholds": scaled_thresholds.tolist(),
        "threshold_sanitization_mask": [False] * 5,
        "thresholds_sanitized": False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Ordering conclusion extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_ordering_conclusions(
    run_results:     List[Dict],
    threshold_frac:  float = 0.05,
) -> Dict[str, object]:
    """
    Extract the ordering of head type emergence for one threshold setting.

    An "ordering conclusion" is a ranked list of non-UNDIFF types by their
    onset step (first step where ≥threshold_frac of all heads are that type).

    Args:
        run_results:    list of result dicts (already classified)
        threshold_frac: onset detection threshold

    Returns:
        dict with keys:
            "onset_steps":     Dict[str, Optional[int]] — per-type onset step
            "ordering":        List[str] — types sorted by onset (None last)
            "h1_holds":        bool — SINK appears before all other non-UNDIFF types
            "h2_partial_holds":bool — ordering is sinks...induction...semantic
            "h5_sink_stable":  bool — evaluated separately in stability.py
    """

    from .trajectories import (
        compute_global_curves, compute_specialization_onset
    )

    global_curves = compute_global_curves(run_results)
    onset_steps = compute_specialization_onset(
        global_curves,
        threshold_frac,
        exclude_positional_init=True,
    )

    # Sort types by onset step (None = never → sorted last)
    non_undiff = {
        k: v for k, v in onset_steps.items()
        if k != "UNDIFFERENTIATED"
    }
    ordering = sorted(
        non_undiff.keys(),
        key=lambda k: (non_undiff[k] is None, non_undiff[k] or 0)
    )

    # H1: SINK is first in the ordering
    h1_holds = (
        len(ordering) > 0 and
        ordering[0] == "SINK" and
        onset_steps["SINK"] is not None
    )

    # H2 partial: SINK comes before INDUCTION, INDUCTION before SEMANTIC
    sink_step = onset_steps.get("SINK")
    ind_step  = onset_steps.get("INDUCTION")
    sem_step  = onset_steps.get("SEMANTIC")

    h2_partial = (
        sink_step is not None and
        ind_step  is not None and
        sem_step  is not None and
        sink_step < ind_step < sem_step
    )

    return {
        "onset_steps":         onset_steps,
        "ordering":            ordering,
        "h1_holds":            h1_holds,
        "h2_partial_holds":    h2_partial,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Threshold sensitivity analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_threshold_sensitivity(
    run_results:    List[Dict],
    scale_factors:  List[float] = [0.8, 1.0, 1.2],
    threshold_frac: float       = 0.05,
) -> Dict[str, object]:
    """
    Re-classify all runs under each threshold scale factor and report
    which ordering conclusions survive all three settings.

    Args:
        run_results:    list of result dicts, one per seed
        scale_factors:  threshold multipliers to test
        threshold_frac: onset detection threshold (applied to fraction curves)

    Returns:
        dict with keys:
            "per_scale":         Dict[float, Dict] — conclusions per scale factor
            "robust_h1":         bool — H1 holds under all scale factors
            "robust_h2_partial": bool — H2 partial holds under all scale factors
            "robustness_summary":Dict[str, str] — per-conclusion verdict
    """

    per_scale: Dict[float, Dict] = {}

    for scale in scale_factors:
        if scale == 1.0:
            scaled_results = run_results
        else:
            scaled_results = [
                reclassify_with_scaled_thresholds(r, scale)
                for r in run_results
            ]

        conclusions = extract_ordering_conclusions(scaled_results, threshold_frac)
        per_scale[scale] = conclusions

    # Check which conclusions are robust across all scales
    robust_h1         = all(per_scale[s]["h1_holds"]         for s in scale_factors)
    robust_h2_partial = all(per_scale[s]["h2_partial_holds"] for s in scale_factors)

    robustness_summary = {
        "H1 (Sink First)":              "ROBUST" if robust_h1         else "NOT ROBUST",
        "H2 partial (Sink→Ind→Sem)":    "ROBUST" if robust_h2_partial else "NOT ROBUST",
    }

    return {
        "per_scale":          per_scale,
        "robust_h1":          robust_h1,
        "robust_h2_partial":  robust_h2_partial,
        "robustness_summary": robustness_summary,
        "scale_factors":      scale_factors,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Inter-seed agreement
# ─────────────────────────────────────────────────────────────────────────────

def compute_inter_seed_agreement(
    run_results:    List[Dict],
    threshold_frac: float = 0.05,
) -> Dict[str, object]:
    """
    For each ordering claim, report the fraction of seed pairs that agree.

    With 3 seeds → 3 pairs: (0,1), (0,2), (1,2).

    An ordering claim is:
    - H1: SINK appears before all other types
    - H2 partial: SINK onset < INDUCTION onset < SEMANTIC onset

    Args:
        run_results:    list of result dicts, one per seed
        threshold_frac: onset detection threshold

    Returns:
        dict with keys:
            "n_seeds":       int
            "n_pairs":       int
            "h1_agreement":  str — "X/Y pairs agree"
            "h2_agreement":  str
            "per_pair":      List[Dict] — per-pair conclusions
    """

    n_seeds = len(run_results)
    pairs   = list(itertools.combinations(range(n_seeds), 2))
    n_pairs = len(pairs)

    h1_agree = 0
    h2_agree = 0
    per_pair = []

    for i, j in pairs:
        pair_results = [run_results[i], run_results[j]]
        conclusions  = extract_ordering_conclusions(pair_results, threshold_frac)

        h1_agree += int(conclusions["h1_holds"])
        h2_agree += int(conclusions["h2_partial_holds"])
        per_pair.append({
            "seed_pair":   (run_results[i]["seed"], run_results[j]["seed"]),
            "h1":          conclusions["h1_holds"],
            "h2_partial":  conclusions["h2_partial_holds"],
            "ordering":    conclusions["ordering"],
        })

    return {
        "n_seeds":      n_seeds,
        "n_pairs":      n_pairs,
        "h1_agreement": f"{h1_agree}/{n_pairs} pairs agree",
        "h2_agreement": f"{h2_agree}/{n_pairs} pairs agree",
        "per_pair":     per_pair,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def print_controls_report(
    sensitivity:      Dict[str, object],
    seed_agreement:   Dict[str, object],
) -> None:
    """Print a formatted controls report."""

    print(f"\n{'=' * 64}")
    print(f"  Scientific Controls Report")
    print(f"{'=' * 64}")

    print(f"\n  Threshold sensitivity (±20%):")
    print(f"  {'─' * 50}")
    for conclusion, verdict in sensitivity["robustness_summary"].items():
        marker = "✓" if verdict == "ROBUST" else "✗"
        print(f"  [{marker}] {conclusion:<35}: {verdict}")

    print(f"\n  Per-scale breakdown:")
    print(f"  {'Scale':>8}  {'H1':>8}  {'H2 partial':>12}")
    print(f"  {'─' * 35}")
    for scale in sensitivity["scale_factors"]:
        c    = sensitivity["per_scale"][scale]
        h1   = "YES" if c["h1_holds"]         else "NO"
        h2   = "YES" if c["h2_partial_holds"] else "NO"
        print(f"  {scale:>8.1f}  {h1:>8}  {h2:>12}")

    print(f"\n  Inter-seed agreement ({seed_agreement['n_seeds']} seeds, "
          f"{seed_agreement['n_pairs']} pairs):")
    print(f"  {'─' * 50}")
    print(f"  H1 (Sink First)         : {seed_agreement['h1_agreement']}")
    print(f"  H2 partial (S→I→Sem)    : {seed_agreement['h2_agreement']}")

    print(f"\n  Per-pair detail:")
    for pair_info in seed_agreement["per_pair"]:
        s1, s2 = pair_info["seed_pair"]
        h1     = "✓" if pair_info["h1"]         else "✗"
        h2     = "✓" if pair_info["h2_partial"] else "✗"
        order  = " → ".join(pair_info["ordering"][:4])
        print(f"  Seeds ({s1},{s2}): H1={h1} H2={h2}  order: {order}...")

    print(f"{'=' * 64}\n")
