"""analysis/ — Trajectory curves, stability, phase transition, and controls."""

from .trajectories import (
    load_run_results,
    compute_global_curves,
    compute_per_layer_curves,
    compute_head_trajectories,
    find_interesting_trajectories,
    compute_specialization_onset,
    compute_onset_bootstrap_cis,
    compute_mixed_behavior_summary,
    print_trajectory_report,
)
from .stability import (
    compute_type_change_matrix,
    compute_sink_persistence,
    compute_stability_histogram,
    compute_per_type_stability,
    print_stability_report,
)
from .phase_transition import (
    compute_induction_count_curve,
    extract_val_loss_curve,
    find_crossing_steps,
    detect_val_loss_inflection,
    compute_discontinuity_score,
    print_phase_transition_report,
)
from .controls import (
    run_threshold_sensitivity,
    compute_inter_seed_agreement,
    reclassify_with_scaled_thresholds,
    extract_ordering_conclusions,
    print_controls_report,
)

__all__ = [
    "load_run_results",
    "compute_global_curves",
    "compute_per_layer_curves",
    "compute_head_trajectories",
    "find_interesting_trajectories",
    "compute_specialization_onset",
    "compute_onset_bootstrap_cis",
    "compute_mixed_behavior_summary",
    "print_trajectory_report",
    "compute_type_change_matrix",
    "compute_sink_persistence",
    "compute_stability_histogram",
    "compute_per_type_stability",
    "print_stability_report",
    "compute_induction_count_curve",
    "extract_val_loss_curve",
    "find_crossing_steps",
    "detect_val_loss_inflection",
    "compute_discontinuity_score",
    "print_phase_transition_report",
    "run_threshold_sensitivity",
    "compute_inter_seed_agreement",
    "reclassify_with_scaled_thresholds",
    "extract_ordering_conclusions",
    "print_controls_report",
]
