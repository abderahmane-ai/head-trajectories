"""
run_analysis.py — Entry point: run all analyses and produce all figures.

Loads probing results for all available seeds, runs the full analysis
pipeline (trajectories, stability, phase transition, controls), prints
formatted reports for each hypothesis, and saves all figures to figures/.

Usage:
    python run_analysis.py                         # all available results
    python run_analysis.py --results_dir results   # custom results path
    python run_analysis.py --figures_dir my_figs   # custom figure output
    python run_analysis.py --skip_ablation         # exclude ablation run
    python run_analysis.py --ckpt_root checkpoints # for val loss extraction

Output figures:
    figures/
        fig1_timeline.png               Main developmental timeline (Figure 1)
        fig1b_timeline_per_seed.png     Per-seed supplement
        fig2a_heatmap_dominant.png      Dominant type heatmap (Figure 2a)
        fig2b_heatmap_spec.png          Specialization fraction heatmap (Figure 2b)
        fig3_phase_transition.png       Induction phase transition (Figure 3)
        fig3b_discontinuity_zoom.png    Zoomed discontinuity supplement
        fig4_stability.png              Stability analysis (Figure 4)
        fig4b_trajectories.png          Individual head trajectories supplement
"""

import argparse
import re
import time
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from analysis import (
    load_run_results,
    compute_global_curves,
    compute_activation_curves,
    compute_per_layer_curves,
    compute_head_trajectories,
    find_interesting_trajectories,
    compute_specialization_onset,
    compute_onset_bootstrap_cis,
    compute_mixed_behavior_summary,
    print_trajectory_report,
    compute_type_change_matrix,
    compute_sink_persistence,
    compute_stability_histogram,
    compute_per_type_stability,
    print_stability_report,
    compute_induction_count_curve,
    compute_induction_validation_summary,
    extract_val_loss_curve,
    find_crossing_steps,
    detect_val_loss_inflection,
    compute_discontinuity_score,
    print_phase_transition_report,
    run_fdr_sensitivity,
    compute_null_subsample_stability,
    compute_inter_seed_agreement,
    print_controls_report,
)
from visualization import (
    plot_timeline,
    plot_timeline_per_seed,
    plot_dominant_type_heatmap,
    plot_specialization_fraction_heatmap,
    plot_phase_transition,
    plot_discontinuity_comparison,
    plot_stability_figure,
    plot_individual_trajectories,
)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full analysis pipeline and produce all paper figures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results_dir", type=Path, default=Path("results"),
        help="Directory containing probing results (.pt files).",
    )
    parser.add_argument(
        "--figures_dir", type=Path, default=Path("figures"),
        help="Directory to save output figures.",
    )
    parser.add_argument(
        "--ckpt_root", type=Path, default=Path("checkpoints"),
        help="Root directory of checkpoints (for val loss extraction).",
    )
    parser.add_argument(
        "--skip_ablation", action="store_true",
        help="Exclude the 6M ablation run from analysis.",
    )
    parser.add_argument(
        "--min_seeds", type=int, default=1,
        help="Minimum number of primary seeds required to run analysis.",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Results discovery
# ─────────────────────────────────────────────────────────────────────────────

LEGACY_PRIMARY_LABELS = ["seed42", "seed123", "seed777"]
ABLATION_LABEL  = "ablation_6m"


def _result_sort_key(label: str) -> Tuple[int, int | str]:
    """Keep historical seed ordering when possible, but support arbitrary labels."""

    if label == ABLATION_LABEL:
        return (2, label)
    match = re.fullmatch(r"seed(\d+)", label)
    if match:
        return (0, int(match.group(1)))
    return (1, label)


def discover_results(
    results_dir:    Path,
    skip_ablation:  bool = False,
) -> Dict[str, Dict]:
    """
    Discover and load all available probing results.

    Returns:
        dict mapping label → result dict
        with keys "primary" and optionally "ablation"
    """

    loaded: Dict[str, Dict] = {}
    result_paths = sorted(results_dir.glob("results_*.pt"))
    if not result_paths:
        print(f"  [Missing] No results_*.pt files found in {results_dir}")
        return loaded

    seen_legacy = {path.stem.replace("results_", "", 1) for path in result_paths}
    for legacy_label in LEGACY_PRIMARY_LABELS:
        if legacy_label not in seen_legacy:
            print(f"  [Missing] results_{legacy_label}.pt — skipping")

    for path in sorted(result_paths, key=lambda p: _result_sort_key(p.stem.replace("results_", "", 1))):
        label = path.stem.replace("results_", "", 1)
        if skip_ablation and label == ABLATION_LABEL:
            continue
        print(f"  [Load] {path.name}")
        loaded[label] = load_run_results(path)

    return loaded


# ─────────────────────────────────────────────────────────────────────────────
# Hypothesis verdict helper
# ─────────────────────────────────────────────────────────────────────────────

def _verdict(condition: bool, label: str) -> str:
    marker = "✓ SUPPORTED" if condition else "✗ NOT SUPPORTED"
    return f"  {label:<40}: {marker}"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    t_start = time.time()

    args.figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 64}")
    print(f"  run_analysis.py — Full Analysis Pipeline")
    print(f"{'=' * 64}")
    print(f"  Results dir  : {args.results_dir}")
    print(f"  Figures dir  : {args.figures_dir}")
    print(f"  Ckpt root    : {args.ckpt_root}")
    print(f"{'─' * 64}")
    print(f"  Loading results...\n")

    # ── Load results ─────────────────────────────────────────────────────────
    all_results = discover_results(args.results_dir, args.skip_ablation)

    primary_labels = [
        label for label in sorted(all_results, key=_result_sort_key)
        if label != ABLATION_LABEL
    ]
    primary_results = [all_results[label] for label in primary_labels]

    if len(primary_results) < args.min_seeds:
        print(
            f"\n[ERROR] Only {len(primary_results)} primary seed results found. "
            f"Need at least {args.min_seeds}.\n"
            f"Run run_probing.py first."
        )
        return

    ablation_result = all_results.get(ABLATION_LABEL)
    n_seeds = len(primary_results)

    print(f"\n  {n_seeds} primary seeds loaded.")
    if ablation_result:
        print(f"  Ablation (6M) loaded.")
    print()

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 1 — Trajectory Analysis
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'─' * 64}")
    print(f"  [1/4] Trajectory Analysis")
    print(f"{'─' * 64}")

    global_curves    = compute_global_curves(primary_results)
    activation_curves = compute_activation_curves(primary_results)
    per_layer_curves = compute_per_layer_curves(primary_results)
    onset_steps = compute_specialization_onset(
        global_curves,
        threshold_frac=0.05,
    )
    activation_onset_steps = compute_specialization_onset(
        activation_curves,
        threshold_frac=0.05,
    )
    onset_cis = compute_onset_bootstrap_cis(
        primary_results,
        threshold_frac=0.05,
        n_bootstraps=1000,
        random_seed=0,
        curve_mode="dominance",
    )
    activation_onset_cis = compute_onset_bootstrap_cis(
        primary_results,
        threshold_frac=0.05,
        n_bootstraps=1000,
        random_seed=0,
        curve_mode="activation",
    )
    learned_onset_steps = compute_specialization_onset(
        global_curves,
        threshold_frac=0.05,
        exclude_positional_init=True,
    )
    mixed_behavior = compute_mixed_behavior_summary(primary_results)

    print_trajectory_report(
        global_curves,
        per_layer_curves,
        onset_steps,
        learned_onset_steps=learned_onset_steps,
        onset_cis=onset_cis,
        activation_curves=activation_curves,
        activation_onset_steps=activation_onset_steps,
        activation_onset_cis=activation_onset_cis,
        mixed_behavior=mixed_behavior,
        seed=primary_results[0]["seed"],
    )

    # Figure 1: main timeline
    plot_timeline(
        global_curves,
        args.figures_dir / "fig1_timeline.png",
        onset_steps=onset_steps,
        log_x=True,
        show_undiff=True,
    )

    # Figure 1b: per-seed supplement
    plot_timeline_per_seed(
        global_curves,
        args.figures_dir / "fig1b_timeline_per_seed.png",
    )

    # Figure 2a: dominant type heatmap
    plot_dominant_type_heatmap(
        per_layer_curves,
        args.figures_dir / "fig2a_heatmap_dominant.png",
    )

    # Figure 2b: specialization fraction heatmap
    plot_specialization_fraction_heatmap(
        per_layer_curves,
        args.figures_dir / "fig2b_heatmap_spec.png",
    )

    # Ablation heatmap (if available)
    if ablation_result:
        ablation_plc = compute_per_layer_curves([ablation_result])
        plot_specialization_fraction_heatmap(
            ablation_plc,
            args.figures_dir / "fig2c_heatmap_spec_ablation.png",
            title="Specialization Fraction — 6M Ablation Model",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 2 — Stability Analysis
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'─' * 64}")
    print(f"  [2/4] Stability Analysis")
    print(f"{'─' * 64}")

    change_matrix    = compute_type_change_matrix(primary_results)
    sink_persistence = compute_sink_persistence(primary_results)
    hist_data        = compute_stability_histogram(change_matrix)
    per_type_stab    = compute_per_type_stability(primary_results, change_matrix)

    print_stability_report(change_matrix, sink_persistence, per_type_stab)

    # Figure 4: stability
    plot_stability_figure(
        hist_data,
        sink_persistence,
        per_type_stab,
        args.figures_dir / "fig4_stability.png",
    )

    # Figure 4b: individual trajectories (most interesting heads from seed 0)
    trajs_seed0   = compute_head_trajectories(primary_results[0])
    interesting   = find_interesting_trajectories(trajs_seed0, min_type_changes=2)
    plot_individual_trajectories(
        interesting,
        primary_results[0]["step_index"],
        args.figures_dir / "fig4b_trajectories.png",
        max_heads=16,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 3 — Phase Transition Analysis
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'─' * 64}")
    print(f"  [3/4] Phase Transition Analysis")
    print(f"{'─' * 64}")

    induction_curve = compute_induction_count_curve(primary_results)
    val_loss_curve  = extract_val_loss_curve(
        primary_results,
        ckpt_dir=[args.ckpt_root / label for label in primary_labels],
    )
    crossing_steps  = find_crossing_steps(
        induction_curve, fractions=[0.10, 0.25, 0.50]
    )
    discontinuity   = compute_discontinuity_score(induction_curve)

    # Test inflection at 10% and 25% crossings
    inflection_10 = (
        detect_val_loss_inflection(
            val_loss_curve, crossing_steps[0.10], window_steps=500
        )
        if crossing_steps[0.10] is not None
        else {"inflection_found": False, "inflection_step": None,
              "crossing_step": -1, "delta_steps": None, "second_deriv": None}
    )
    inflection_25 = (
        detect_val_loss_inflection(
            val_loss_curve, crossing_steps[0.25], window_steps=500
        )
        if crossing_steps[0.25] is not None
        else {"inflection_found": False, "inflection_step": None,
              "crossing_step": -1, "delta_steps": None, "second_deriv": None}
    )

    print_phase_transition_report(
        induction_curve, crossing_steps,
        inflection_10, inflection_25, discontinuity,
    )

    natural_induction_tensors = [
        r.get("natural_induction_score_tensor")
        for r in primary_results
        if r.get("natural_induction_score_tensor") is not None
    ]
    if natural_induction_tensors:
        final_natural_scores = [
            float(np.asarray(tensor[-1]).mean())
            for tensor in natural_induction_tensors
        ]
        print(f"  Natural induction (final mean raw score): {np.mean(final_natural_scores):.4f}")
    induction_validation = compute_induction_validation_summary(primary_results)
    if induction_validation["available"]:
        gap = induction_validation["natural_score_gap_final"]
        if gap is None:
            print("  Natural induction validation gap (active - inactive, final): unavailable")
        else:
            print(f"  Natural induction validation gap (active - inactive, final): {gap:.4f}")

    semantic_valid_fractions = [
        np.asarray(r.get("semantic_valid_fraction_tensor")[-1]).mean()
        for r in primary_results
        if r.get("semantic_valid_fraction_tensor") is not None
    ]
    semantic_defined_fractions = [
        np.asarray(r.get("semantic_defined_tensor")[-1]).mean()
        for r in primary_results
        if r.get("semantic_defined_tensor") is not None
    ]
    if semantic_valid_fractions:
        print(f"  Semantic valid-fraction mean (final): {float(np.mean(semantic_valid_fractions)):.4f}")
    if semantic_defined_fractions:
        print(f"  Semantic defined-head fraction (final): {float(np.mean(semantic_defined_fractions)):.4f}")

    # Figure 3: phase transition dual-axis plot
    plot_phase_transition(
        induction_curve,
        val_loss_curve,
        crossing_steps,
        inflection_result=inflection_25,
        output_path=args.figures_dir / "fig3_phase_transition.png",
    )

    # Figure 3b: discontinuity zoom
    plot_discontinuity_comparison(
        induction_curve,
        args.figures_dir / "fig3b_discontinuity_zoom.png",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 4 — Scientific Controls
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'─' * 64}")
    print(f"  [4/4] Scientific Controls")
    print(f"{'─' * 64}")

    sensitivity = run_fdr_sensitivity(primary_results, alphas=[0.01, 0.05, 0.10])
    null_subsample = compute_null_subsample_stability(
        primary_results,
        alpha=0.05,
        threshold_frac=0.05,
    )
    seed_agreement = compute_inter_seed_agreement(primary_results)

    print_controls_report(sensitivity, seed_agreement, null_subsample)

    # ─────────────────────────────────────────────────────────────────────────
    # HYPOTHESIS VERDICTS SUMMARY
    # ─────────────────────────────────────────────────────────────────────────
    sink_step_val = learned_onset_steps.get("SINK")
    prev_step_val = learned_onset_steps.get("PREV_TOKEN")
    ind_step_val = learned_onset_steps.get("INDUCTION")
    sem_step_val = learned_onset_steps.get("SEMANTIC")
    activation_sink = activation_onset_steps.get("SINK")
    activation_prev = activation_onset_steps.get("PREV_TOKEN")
    activation_ind = activation_onset_steps.get("INDUCTION")
    activation_sem = activation_onset_steps.get("SEMANTIC")

    dominance_h1 = (
        sink_step_val is not None and
        all(
            learned_onset_steps.get(t) is None or learned_onset_steps[t] >= sink_step_val
            for t in ["PREV_TOKEN", "INDUCTION", "SEMANTIC"]
        )
    )
    dominance_h2 = (
        sink_step_val is not None and
        prev_step_val is not None and
        ind_step_val is not None and
        sem_step_val is not None and
        sink_step_val <= prev_step_val < ind_step_val < sem_step_val
    )
    activation_h1 = (
        activation_sink is not None and
        all(
            activation_onset_steps.get(t) is None or activation_onset_steps[t] >= activation_sink
            for t in ["PREV_TOKEN", "INDUCTION", "SEMANTIC"]
        )
    )
    activation_h2 = (
        activation_sink is not None and
        activation_prev is not None and
        activation_ind is not None and
        activation_sem is not None and
        activation_sink <= activation_prev < activation_ind < activation_sem
    )

    # H3: lower layers specialize first — use mean onset step per layer
    per_layer_mean = per_layer_curves["per_layer_mean"]
    layer_steps    = per_layer_curves["steps"]
    layer_onsets   = []
    for layer in range(per_layer_curves["n_layers"]):
        nonspecialized = per_layer_mean[layer, :, :2].sum(axis=-1)
        spec = 1.0 - nonspecialized
        above = (spec >= 0.5).nonzero()
        if hasattr(above, '__len__') and len(above) > 0:
            first = int(above[0]) if not hasattr(above[0], '__len__') else int(above[0][0])
            layer_onsets.append(int(layer_steps[first]))
        else:
            layer_onsets.append(None)
    valid_onsets = [(i, v) for i, v in enumerate(layer_onsets) if v is not None]
    h3 = (
        len(valid_onsets) >= 2 and
        all(valid_onsets[i][1] <= valid_onsets[i+1][1]
            for i in range(len(valid_onsets) - 1))
    )

    h4 = (
        discontinuity >= 1.5 or
        inflection_25.get("inflection_found", False)
    )

    h5 = sink_persistence["mean_persistence"] >= 0.7

    print(f"\n{'=' * 64}")
    print(f"  HYPOTHESIS VERDICTS")
    print(f"{'=' * 64}")
    print(_verdict(activation_h1, "H1a — Sink-First Activation"))
    print(_verdict(dominance_h1, "H1b — Sink-First Dominance"))
    print(_verdict(activation_h2, "H2a — Ordered Activation"))
    print(_verdict(dominance_h2, "H2b — Ordered Dominance"))
    print(_verdict(h3, "H3 — Layer Stratification"))
    print(_verdict(h4, "H4 — Induction Phase Transition"))
    print(_verdict(h5, "H5 — Sink Persistence"))
    print(f"{'─' * 64}")

    verdict_flags = [activation_h1, dominance_h1, activation_h2, dominance_h2, h3, h4, h5]
    n_supported = sum(bool(flag) for flag in verdict_flags)
    print(f"  {n_supported}/{len(verdict_flags)} hypothesis checks supported")

    # ─────────────────────────────────────────────────────────────────────────
    # FIGURES SUMMARY
    # ─────────────────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print(f"\n{'=' * 64}")
    print(f"  Analysis complete in {elapsed:.1f}s")
    print(f"{'─' * 64}")
    print(f"  Figures saved to: {args.figures_dir}/\n")
    for fig_path in sorted(args.figures_dir.glob("*.png")):
        size_kb = fig_path.stat().st_size / 1024
        print(f"    {fig_path.name:<45} {size_kb:>7.1f} KB")
    print(f"{'=' * 64}\n")


if __name__ == "__main__":
    main()
