"""
analysis/controls.py — Robustness controls for the FDR-based classifier.

Computes:
1. FDR alpha sensitivity
2. Null-subsample stability
3. Inter-seed agreement
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Sequence

import numpy as np
import torch

from probing import (
    DEFAULT_DOMINANCE_MARGIN,
    DEFAULT_FDR_ALPHA,
    HEAD_TYPES,
    classify_head_details,
)
from .trajectories import (
    compute_activation_curves,
    compute_global_curves,
    compute_specialization_onset,
)


def _clone_result_with_reclassification(
    result: Dict,
    label_tensor: torch.Tensor,
    active_behavior_tensor: torch.Tensor,
    p_value_tensor: torch.Tensor,
    effect_size_tensor: torch.Tensor,
    threshold_flag_tensor: torch.Tensor,
    normalized_score_tensor: torch.Tensor,
    primary_behavior_tensor: torch.Tensor,
    runner_up_tensor: torch.Tensor,
    dominant_margin_tensor: torch.Tensor,
    behavior_count_tensor: torch.Tensor,
) -> Dict:
    clone = dict(result)
    clone["label_tensor"] = label_tensor
    clone["dominant_label_tensor"] = label_tensor
    clone["active_behavior_tensor"] = active_behavior_tensor
    clone["p_value_tensor"] = p_value_tensor
    clone["effect_size_tensor"] = effect_size_tensor
    clone["threshold_flag_tensor"] = threshold_flag_tensor
    clone["normalized_score_tensor"] = normalized_score_tensor
    clone["primary_behavior_tensor"] = primary_behavior_tensor
    clone["runner_up_tensor"] = runner_up_tensor
    clone["dominant_margin_tensor"] = dominant_margin_tensor
    clone["behavior_count_tensor"] = behavior_count_tensor
    return clone


def _reclassify_result(
    result: Dict,
    alpha: float,
    pooled_null_scores: np.ndarray | None = None,
) -> Dict:
    score_tensor = np.asarray(result["score_tensor"])
    n_ckpts, n_layers, n_heads, _ = score_tensor.shape
    thresholds = np.asarray(
        result.get("raw_thresholds", result.get("thresholds")),
        dtype=np.float32,
    )
    null_scores = pooled_null_scores
    if null_scores is None:
        null_scores = result.get("pooled_null_scores")
    if null_scores is None:
        raise ValueError("Result is missing pooled_null_scores required for FDR reclassification")

    new_labels = torch.zeros((n_ckpts, n_layers, n_heads), dtype=torch.int32)
    threshold_flag_tensor = torch.zeros((n_ckpts, n_layers, n_heads, 5), dtype=torch.bool)
    normalized_score_tensor = torch.zeros((n_ckpts, n_layers, n_heads, 5), dtype=torch.float32)
    active_behavior_tensor = torch.zeros((n_ckpts, n_layers, n_heads, 5), dtype=torch.bool)
    p_value_tensor = torch.ones((n_ckpts, n_layers, n_heads, 5), dtype=torch.float32)
    effect_size_tensor = torch.zeros((n_ckpts, n_layers, n_heads, 5), dtype=torch.float32)
    primary_behavior_tensor = torch.full((n_ckpts, n_layers, n_heads), -1, dtype=torch.int32)
    runner_up_tensor = torch.full((n_ckpts, n_layers, n_heads), -1, dtype=torch.int32)
    dominant_margin_tensor = torch.zeros((n_ckpts, n_layers, n_heads), dtype=torch.float32)
    behavior_count_tensor = torch.zeros((n_ckpts, n_layers, n_heads), dtype=torch.int32)
    for ckpt in range(n_ckpts):
        for layer in range(n_layers):
            for head in range(n_heads):
                details = classify_head_details(
                    tuple(score_tensor[ckpt, layer, head].tolist()),
                    thresholds=thresholds,
                    pooled_null_scores=null_scores,
                    fdr_alpha=alpha,
                    dominance_margin=float(result.get("dominance_margin", DEFAULT_DOMINANCE_MARGIN)),
                )
                new_labels[ckpt, layer, head] = details.label
                threshold_flag_tensor[ckpt, layer, head] = torch.tensor(details.threshold_flags, dtype=torch.bool)
                normalized_score_tensor[ckpt, layer, head] = torch.tensor(details.normalized_scores, dtype=torch.float32)
                active_behavior_tensor[ckpt, layer, head] = torch.tensor(details.active_behaviors, dtype=torch.bool)
                p_value_tensor[ckpt, layer, head] = torch.tensor(details.p_values, dtype=torch.float32)
                effect_size_tensor[ckpt, layer, head] = torch.tensor(details.effect_sizes, dtype=torch.float32)
                primary_behavior_tensor[ckpt, layer, head] = int(details.primary_behavior)
                runner_up_tensor[ckpt, layer, head] = int(details.runner_up_behavior)
                dominant_margin_tensor[ckpt, layer, head] = float(details.dominant_margin)
                behavior_count_tensor[ckpt, layer, head] = int(details.n_active_behaviors)
    clone = _clone_result_with_reclassification(
        result,
        new_labels,
        active_behavior_tensor,
        p_value_tensor,
        effect_size_tensor,
        threshold_flag_tensor,
        normalized_score_tensor,
        primary_behavior_tensor,
        runner_up_tensor,
        dominant_margin_tensor,
        behavior_count_tensor,
    )
    clone["fdr_alpha"] = float(alpha)
    return clone


def extract_ordering_conclusions(
    run_results: List[Dict],
    threshold_frac: float = 0.05,
) -> Dict[str, object]:
    """
    Extract H1/H2-style onset conclusions for both activation and dominance.
    """

    dominance_curves = compute_global_curves(run_results)
    dominance_onsets = compute_specialization_onset(
        dominance_curves,
        threshold_frac=threshold_frac,
        exclude_positional_init=True,
    )
    activation_curves = compute_activation_curves(run_results)
    activation_onsets = compute_specialization_onset(
        activation_curves,
        threshold_frac=threshold_frac,
        exclude_positional_init=True,
    )

    def _h1(onsets: Dict[str, int | None]) -> bool:
        sink_step = onsets.get("SINK")
        return (
            sink_step is not None
            and all(
                onsets.get(name) is None or onsets[name] >= sink_step
                for name in ["PREV_TOKEN", "INDUCTION", "SEMANTIC"]
            )
        )

    def _h2(onsets: Dict[str, int | None]) -> bool:
        sink_step = onsets.get("SINK")
        prev_step = onsets.get("PREV_TOKEN")
        ind_step = onsets.get("INDUCTION")
        sem_step = onsets.get("SEMANTIC")
        return (
            sink_step is not None
            and prev_step is not None
            and ind_step is not None
            and sem_step is not None
            and sink_step <= prev_step < ind_step < sem_step
        )

    return {
        "dominance_onset_steps": dominance_onsets,
        "activation_onset_steps": activation_onsets,
        "dominance_h1_holds": _h1(dominance_onsets),
        "dominance_h2_holds": _h2(dominance_onsets),
        "activation_h1_holds": _h1(activation_onsets),
        "activation_h2_holds": _h2(activation_onsets),
    }


def run_fdr_sensitivity(
    run_results: List[Dict],
    alphas: Sequence[float] = (0.01, 0.05, 0.10),
    threshold_frac: float = 0.05,
) -> Dict[str, object]:
    per_alpha: Dict[float, Dict] = {}
    for alpha in alphas:
        scaled = [
            _reclassify_result(result, alpha=float(alpha))
            for result in run_results
        ]
        per_alpha[float(alpha)] = extract_ordering_conclusions(scaled, threshold_frac=threshold_frac)

    robust_activation_h1 = all(per_alpha[a]["activation_h1_holds"] for a in per_alpha)
    robust_activation_h2 = all(per_alpha[a]["activation_h2_holds"] for a in per_alpha)
    robust_dominance_h1 = all(per_alpha[a]["dominance_h1_holds"] for a in per_alpha)
    robust_dominance_h2 = all(per_alpha[a]["dominance_h2_holds"] for a in per_alpha)

    return {
        "per_alpha": per_alpha,
        "alphas": list(alphas),
        "robustness_summary": {
            "Activation H1": "ROBUST" if robust_activation_h1 else "NOT ROBUST",
            "Activation H2": "ROBUST" if robust_activation_h2 else "NOT ROBUST",
            "Dominance H1": "ROBUST" if robust_dominance_h1 else "NOT ROBUST",
            "Dominance H2": "ROBUST" if robust_dominance_h2 else "NOT ROBUST",
        },
    }


def compute_null_subsample_stability(
    run_results: List[Dict],
    alpha: float = DEFAULT_FDR_ALPHA,
    threshold_frac: float = 0.05,
    n_subsamples: int = 25,
    subsample_frac: float = 0.8,
    random_seed: int = 0,
) -> Dict[str, object]:
    rng = np.random.default_rng(random_seed)
    base = extract_ordering_conclusions(run_results, threshold_frac=threshold_frac)

    matches = {
        "activation_h1": 0,
        "activation_h2": 0,
        "dominance_h1": 0,
        "dominance_h2": 0,
    }

    for _ in range(n_subsamples):
        subsampled_results = []
        for result in run_results:
            null_scores = np.asarray(result.get("pooled_null_scores"))
            if null_scores.ndim != 2:
                raise ValueError("Result is missing pooled_null_scores required for null subsample stability")
            take = max(10, int(round(null_scores.shape[0] * subsample_frac)))
            indices = rng.choice(null_scores.shape[0], size=take, replace=False)
            subsampled_results.append(
                _reclassify_result(
                    result,
                    alpha=alpha,
                    pooled_null_scores=null_scores[indices],
                )
            )
        current = extract_ordering_conclusions(subsampled_results, threshold_frac=threshold_frac)
        for key, base_key in [
            ("activation_h1", "activation_h1_holds"),
            ("activation_h2", "activation_h2_holds"),
            ("dominance_h1", "dominance_h1_holds"),
            ("dominance_h2", "dominance_h2_holds"),
        ]:
            if current[base_key] == base[base_key]:
                matches[key] += 1

    return {
        "n_subsamples": n_subsamples,
        "subsample_frac": subsample_frac,
        "match_rates": {key: value / float(n_subsamples) for key, value in matches.items()},
    }


def compute_inter_seed_agreement(
    run_results: List[Dict],
    threshold_frac: float = 0.05,
) -> Dict[str, object]:
    n_seeds = len(run_results)
    pairs = list(itertools.combinations(range(n_seeds), 2))
    n_pairs = len(pairs)

    counts = {
        "activation_h1": 0,
        "activation_h2": 0,
        "dominance_h1": 0,
        "dominance_h2": 0,
    }
    per_pair = []

    per_seed = [
        extract_ordering_conclusions([result], threshold_frac=threshold_frac)
        for result in run_results
    ]

    for i, j in pairs:
        a = per_seed[i]
        b = per_seed[j]
        row = {"pair": (i, j)}
        for key in counts:
            field = f"{key}_holds" if not key.endswith("_h1") and not key.endswith("_h2") else None
        comparisons = {
            "activation_h1": a["activation_h1_holds"] == b["activation_h1_holds"],
            "activation_h2": a["activation_h2_holds"] == b["activation_h2_holds"],
            "dominance_h1": a["dominance_h1_holds"] == b["dominance_h1_holds"],
            "dominance_h2": a["dominance_h2_holds"] == b["dominance_h2_holds"],
        }
        for key, match in comparisons.items():
            if match:
                counts[key] += 1
            row[key] = match
        per_pair.append(row)

    return {
        "n_seeds": n_seeds,
        "n_pairs": n_pairs,
        "activation_h1_agreement": f"{counts['activation_h1']}/{n_pairs} pairs agree" if n_pairs else "0/0 pairs agree",
        "activation_h2_agreement": f"{counts['activation_h2']}/{n_pairs} pairs agree" if n_pairs else "0/0 pairs agree",
        "dominance_h1_agreement": f"{counts['dominance_h1']}/{n_pairs} pairs agree" if n_pairs else "0/0 pairs agree",
        "dominance_h2_agreement": f"{counts['dominance_h2']}/{n_pairs} pairs agree" if n_pairs else "0/0 pairs agree",
        "per_pair": per_pair,
    }


def print_controls_report(
    fdr_sensitivity: Dict[str, object],
    inter_seed_agreement: Dict[str, object],
    null_subsample_stability: Dict[str, object],
) -> None:
    print(f"\n{'=' * 64}")
    print("  Controls Report")
    print(f"{'=' * 64}")
    print("\n  FDR alpha sensitivity:")
    print("  " + "─" * 50)
    for key, verdict in fdr_sensitivity["robustness_summary"].items():
        print(f"  {key:<20}: {verdict}")

    print("\n  Null-subsample stability:")
    print("  " + "─" * 50)
    for key, value in null_subsample_stability["match_rates"].items():
        print(f"  {key:<20}: {value:.2f}")

    print("\n  Inter-seed agreement:")
    print("  " + "─" * 50)
    print(f"  Activation H1         : {inter_seed_agreement['activation_h1_agreement']}")
    print(f"  Activation H2         : {inter_seed_agreement['activation_h2_agreement']}")
    print(f"  Dominance H1          : {inter_seed_agreement['dominance_h1_agreement']}")
    print(f"  Dominance H2          : {inter_seed_agreement['dominance_h2_agreement']}")
    print(f"{'=' * 64}\n")
