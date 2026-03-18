"""
analysis/phase_transition.py — Induction head phase transition analysis.

Tests H4: induction heads emerge discontinuously, and their emergence
coincides with a corresponding inflection in the validation loss curve.

Computes:
  1. Induction head count over training steps (from label tensors)
  2. Validation loss over training steps (from checkpoint metadata)
  3. Crossing steps: when does induction count reach 10%, 25%, 50% of final?
  4. Val loss inflection detection: is there a second-derivative sign change
     within ±500 steps of each crossing?
  5. Discontinuity score: how abruptly does the induction curve rise?
     (ratio of max slope to mean slope)
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from probing import LABEL_IND
from model import ModelConfig


# ─────────────────────────────────────────────────────────────────────────────
# Induction count curve
# ─────────────────────────────────────────────────────────────────────────────

def compute_induction_count_curve(
    run_results: List[Dict],
) -> Dict[str, np.ndarray]:
    """
    Compute the number of heads labeled INDUCTION at each training step,
    averaged across seeds.

    Args:
        run_results: list of result dicts, one per seed

    Returns:
        dict with keys:
            "steps":       (n_ckpts,)
            "mean_count":  (n_ckpts,) — mean induction head count across seeds
            "std_count":   (n_ckpts,) — std across seeds
            "per_seed":    (n_seeds, n_ckpts) — per-seed counts
            "final_count": float — mean induction count at last checkpoint
    """

    n_seeds   = len(run_results)
    min_ckpts = min(len(r["step_index"]) for r in run_results)
    steps     = np.array(run_results[0]["step_index"][:min_ckpts])

    per_seed = np.zeros((n_seeds, min_ckpts), dtype=np.float32)

    for s_idx, result in enumerate(run_results):
        labels = result["label_tensor"][:min_ckpts]   # (n_ckpts, n_layers, n_heads)
        for ckpt in range(min_ckpts):
            per_seed[s_idx, ckpt] = float(
                (labels[ckpt] == LABEL_IND).sum().item()
            )

    mean_count = per_seed.mean(axis=0)
    std_count  = per_seed.std(axis=0) if n_seeds > 1 else np.zeros_like(mean_count)

    return {
        "steps":       steps,
        "mean_count":  mean_count,
        "std_count":   std_count,
        "per_seed":    per_seed,
        "final_count": float(mean_count[-1]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Validation loss curve
# ─────────────────────────────────────────────────────────────────────────────

def extract_val_loss_curve(
    run_results: List[Dict],
    ckpt_dir:    Path,
) -> Dict[str, np.ndarray]:
    """
    Extract validation loss at each checkpoint step from saved checkpoint files.

    Falls back to any val_loss stored in the result dict's metadata if
    direct checkpoint loading is unavailable.

    Args:
        run_results: list of result dicts
        ckpt_dir:    directory containing checkpoint .pt files (for first seed)

    Returns:
        dict with keys:
            "steps":    (n_ckpts,)
            "val_loss": (n_ckpts,) — mean val loss across available seeds
    """

    min_ckpts = min(len(r["step_index"]) for r in run_results)
    steps     = np.array(run_results[0]["step_index"][:min_ckpts])

    # Try to load val_loss from checkpoint files directly
    val_losses = np.full((min_ckpts,), np.nan, dtype=np.float32)

    ckpt_files = sorted(ckpt_dir.glob("ckpt_*.pt")) if ckpt_dir.exists() else []

    # Build step → val_loss map from checkpoints
    step_to_val: Dict[int, float] = {}
    for ckpt_file in ckpt_files:
        try:
            # Register ModelConfig as safe for torch.load
            torch.serialization.add_safe_globals([ModelConfig])
            ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=True)
            step_to_val[ckpt["step"]] = float(ckpt.get("val_loss", np.nan))
        except Exception:
            continue

    for i, step in enumerate(steps):
        if int(step) in step_to_val:
            val_losses[i] = step_to_val[int(step)]

    # Fill NaN gaps with linear interpolation
    valid_mask = ~np.isnan(val_losses)
    if valid_mask.sum() >= 2:
        interp_indices = np.arange(min_ckpts)
        val_losses = np.interp(
            interp_indices,
            interp_indices[valid_mask],
            val_losses[valid_mask],
        )

    return {
        "steps":    steps,
        "val_loss": val_losses,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Crossing step detection
# ─────────────────────────────────────────────────────────────────────────────

def find_crossing_steps(
    induction_curve: Dict[str, np.ndarray],
    fractions: List[float] = [0.10, 0.25, 0.50],
) -> Dict[float, Optional[int]]:
    """
    Find the training step at which the induction head count first reaches
    each fraction of its final value.

    Args:
        induction_curve: output of compute_induction_count_curve
        fractions:       list of fractions to find crossings for

    Returns:
        crossing_steps: dict mapping fraction → training step (or None)
    """

    steps       = induction_curve["steps"]
    mean_count  = induction_curve["mean_count"]
    final_count = induction_curve["final_count"]

    crossing_steps: Dict[float, Optional[int]] = {}

    for frac in fractions:
        target  = frac * final_count
        above   = np.where(mean_count >= target)[0]
        crossing_steps[frac] = int(steps[above[0]]) if len(above) > 0 else None

    return crossing_steps


# ─────────────────────────────────────────────────────────────────────────────
# Val loss inflection detection
# ─────────────────────────────────────────────────────────────────────────────

def _smooth(arr: np.ndarray, window: int = 5) -> np.ndarray:
    """Apply a simple box-car moving average to reduce noise."""
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


def detect_val_loss_inflection(
    val_loss_curve:  Dict[str, np.ndarray],
    crossing_step:   int,
    window_steps:    int = 500,
) -> Dict[str, object]:
    """
    Test whether the validation loss shows an inflection point
    (second derivative sign change) within ±window_steps of crossing_step.

    An inflection in val loss concurrent with induction head emergence
    is the mechanistic signature of H4.

    Args:
        val_loss_curve: output of extract_val_loss_curve
        crossing_step:  training step of induction head crossing (from find_crossing_steps)
        window_steps:   search window around crossing_step (±window_steps)

    Returns:
        dict with keys:
            "inflection_found":   bool
            "inflection_step":    int or None — step of inflection
            "crossing_step":      int
            "delta_steps":        int or None — abs(inflection - crossing)
            "second_deriv":       (n_ckpts,) — smoothed second derivative of val loss
    """

    steps    = val_loss_curve["steps"]
    val_loss = val_loss_curve["val_loss"]

    # Smooth val loss before differentiation
    smoothed     = _smooth(val_loss, window=7)

    # First and second derivatives (finite differences)
    first_deriv  = np.gradient(smoothed, steps)
    second_deriv = np.gradient(first_deriv, steps)
    second_deriv_smooth = _smooth(second_deriv, window=5)

    # Find sign changes in second derivative (zero crossings)
    sign_changes = np.where(
        np.diff(np.sign(second_deriv_smooth))
    )[0]

    if len(sign_changes) == 0:
        return {
            "inflection_found": False,
            "inflection_step":  None,
            "crossing_step":    crossing_step,
            "delta_steps":      None,
            "second_deriv":     second_deriv_smooth,
        }

    # Find sign changes within the window
    sign_change_steps = steps[sign_changes]
    in_window = sign_change_steps[
        np.abs(sign_change_steps - crossing_step) <= window_steps
    ]

    if len(in_window) == 0:
        return {
            "inflection_found": False,
            "inflection_step":  None,
            "crossing_step":    crossing_step,
            "delta_steps":      None,
            "second_deriv":     second_deriv_smooth,
        }

    # Closest inflection to the crossing step
    closest_idx   = np.argmin(np.abs(in_window - crossing_step))
    inflect_step  = int(in_window[closest_idx])
    delta         = abs(inflect_step - crossing_step)

    return {
        "inflection_found": True,
        "inflection_step":  inflect_step,
        "crossing_step":    crossing_step,
        "delta_steps":      delta,
        "second_deriv":     second_deriv_smooth,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Discontinuity score
# ─────────────────────────────────────────────────────────────────────────────

def compute_discontinuity_score(
    induction_curve: Dict[str, np.ndarray],
) -> float:
    """
    Quantify how abruptly the induction head curve rises.

    Discontinuity score = max(slope) / mean(slope over rising phase)

    A perfectly gradual rise scores ~1.0.
    A sharp phase-transition-like jump scores >> 1.0.

    Args:
        induction_curve: output of compute_induction_count_curve

    Returns:
        discontinuity_score: float (higher = more abrupt)
    """

    steps      = induction_curve["steps"].astype(float)
    counts     = induction_curve["mean_count"]

    # Compute slope of induction count curve
    slopes = np.gradient(counts, steps)
    slopes = np.abs(slopes)

    # Rising phase: steps where count is between 10% and 90% of final
    final  = induction_curve["final_count"]
    if final < 1.0:
        return 0.0

    rising_mask = (counts >= 0.1 * final) & (counts <= 0.9 * final)
    if rising_mask.sum() < 2:
        return 0.0

    rising_slopes = slopes[rising_mask]
    mean_slope    = rising_slopes.mean()
    max_slope     = rising_slopes.max()

    if mean_slope < 1e-10:
        return 0.0

    return float(max_slope / mean_slope)


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def print_phase_transition_report(
    induction_curve:   Dict[str, np.ndarray],
    crossing_steps:    Dict[float, Optional[int]],
    inflection_10pct:  Dict[str, object],
    inflection_25pct:  Dict[str, object],
    discontinuity:     float,
) -> None:
    """Print a formatted phase transition analysis report."""

    print(f"\n{'=' * 64}")
    print(f"  Phase Transition Analysis Report (H4)")
    print(f"{'=' * 64}")

    final = induction_curve["final_count"]
    print(f"\n  Final induction head count : {final:.1f} heads")

    print(f"\n  Crossing steps:")
    print(f"  {'─' * 50}")
    for frac, step in crossing_steps.items():
        step_str = f"{step:>10,}" if step is not None else "      never"
        print(f"  {int(frac * 100):>3}% of final count : {step_str}")

    print(f"\n  Val loss inflection at 10% crossing:")
    _print_inflection(inflection_10pct)

    print(f"\n  Val loss inflection at 25% crossing:")
    _print_inflection(inflection_25pct)

    print(f"\n  Discontinuity score        : {discontinuity:.2f}")
    if discontinuity >= 3.0:
        disc_verdict = "SHARP transition (score ≥ 3.0) — supports H4"
    elif discontinuity >= 1.5:
        disc_verdict = "MODERATE transition (score 1.5–3.0) — weak H4 support"
    else:
        disc_verdict = "GRADUAL transition (score < 1.5) — H4 not supported"
    print(f"  Verdict: {disc_verdict}")
    print(f"{'=' * 64}\n")


def _print_inflection(inflection: Dict[str, object]) -> None:
    if inflection["inflection_found"]:
        print(f"  {'─' * 40}")
        print(f"  Inflection found : YES")
        print(f"  Inflection step  : {inflection['inflection_step']:,}")
        print(f"  Crossing step    : {inflection['crossing_step']:,}")
        print(f"  Delta            : {inflection['delta_steps']:,} steps")
    else:
        print(f"  {'─' * 40}")
        print(f"  Inflection found : NO (none within ±500 steps of crossing)")
