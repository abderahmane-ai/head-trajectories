"""
analysis/trajectories.py — Developmental trajectory computation.

Computes:
1. Dominance curves from the dominant-summary label tensor
2. Activation curves from the active-behavior tensor
3. Per-layer dominance curves
4. Head-level dominant-label trajectories
5. Onset-step point estimates and bootstrap confidence intervals
6. Mixed-behavior summaries derived from active sets and dominant summaries
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from probing import BEHAVIOR_NAMES, HEAD_TYPES, LABEL_WEAK
from probing.classifier import HeadClassifier, NONSPECIALIZED_TYPES


def load_run_results(results_path: Path) -> Dict:
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    return HeadClassifier.load(results_path)


def _type_names(result: Dict) -> List[str]:
    return list(result.get("type_names", HEAD_TYPES))


def _behavior_names(result: Dict) -> List[str]:
    return list(result.get("behavior_names", BEHAVIOR_NAMES))


def _dominant_tensor(result: Dict) -> np.ndarray:
    return np.asarray(result.get("dominant_label_tensor", result["label_tensor"]))


def _active_tensor(result: Dict) -> np.ndarray:
    if "active_behavior_tensor" in result:
        return np.asarray(result["active_behavior_tensor"])
    if "threshold_flag_tensor" in result:
        return np.asarray(result["threshold_flag_tensor"])
    labels = np.asarray(result.get("dominant_label_tensor", result.get("label_tensor")))
    if labels is not None:
        n_ckpts, n_layers, n_heads = labels.shape
        active = np.zeros((n_ckpts, n_layers, n_heads, 5), dtype=bool)
        for behavior_idx, label_idx in enumerate([2, 3, 4, 5, 6]):
            active[..., behavior_idx] = labels == label_idx
        return active
    raise KeyError("Result does not contain active_behavior_tensor or threshold_flag_tensor")


def compute_global_curves(
    run_results: List[Dict],
) -> Dict[str, np.ndarray]:
    """Compute global dominant-label fraction curves averaged over seeds."""

    n_seeds = len(run_results)
    min_ckpts = min(len(r["step_index"]) for r in run_results)
    steps = np.array(run_results[0]["step_index"][:min_ckpts])
    type_names = _type_names(run_results[0])
    n_types = len(type_names)

    per_seed = np.zeros((n_seeds, min_ckpts, n_types), dtype=np.float32)
    for s_idx, result in enumerate(run_results):
        labels = _dominant_tensor(result)[:min_ckpts]
        _, n_layers, n_heads = labels.shape
        total_heads = n_layers * n_heads
        for ckpt in range(min_ckpts):
            flat_labels = labels[ckpt].reshape(-1)
            for t in range(n_types):
                per_seed[s_idx, ckpt, t] = float((flat_labels == t).sum()) / total_heads

    mean = per_seed.mean(axis=0)
    std = per_seed.std(axis=0) if n_seeds > 1 else np.zeros_like(mean)
    return {
        "steps": steps,
        "mean": mean,
        "std": std,
        "per_seed": per_seed,
        "type_names": type_names,
        "curve_mode": "dominance",
    }


def compute_activation_curves(
    run_results: List[Dict],
) -> Dict[str, np.ndarray]:
    """Compute active-behavior fraction curves averaged over seeds."""

    n_seeds = len(run_results)
    min_ckpts = min(len(r["step_index"]) for r in run_results)
    steps = np.array(run_results[0]["step_index"][:min_ckpts])
    behavior_names = _behavior_names(run_results[0])
    n_behaviors = len(behavior_names)

    per_seed = np.zeros((n_seeds, min_ckpts, n_behaviors), dtype=np.float32)
    for s_idx, result in enumerate(run_results):
        active = _active_tensor(result)[:min_ckpts]
        _, n_layers, n_heads, _ = active.shape
        total_heads = n_layers * n_heads
        flat = active.reshape(min_ckpts, total_heads, n_behaviors)
        per_seed[s_idx] = flat.mean(axis=1)

    mean = per_seed.mean(axis=0)
    std = per_seed.std(axis=0) if n_seeds > 1 else np.zeros_like(mean)
    return {
        "steps": steps,
        "mean": mean,
        "std": std,
        "per_seed": per_seed,
        "type_names": behavior_names,
        "curve_mode": "activation",
    }


def compute_per_layer_curves(
    run_results: List[Dict],
) -> Dict[str, object]:
    """Compute dominant-label fraction curves separately for each layer."""

    n_seeds = len(run_results)
    min_ckpts = min(len(r["step_index"]) for r in run_results)
    steps = np.array(run_results[0]["step_index"][:min_ckpts])
    n_layers = run_results[0]["n_layers"]
    n_heads = run_results[0]["n_heads"]
    type_names = _type_names(run_results[0])
    n_types = len(type_names)

    per_seed_layer = np.zeros((n_seeds, n_layers, min_ckpts, n_types), dtype=np.float32)
    for s_idx, result in enumerate(run_results):
        labels = _dominant_tensor(result)[:min_ckpts]
        for layer in range(n_layers):
            for ckpt in range(min_ckpts):
                flat = labels[ckpt, layer]
                for t in range(n_types):
                    per_seed_layer[s_idx, layer, ckpt, t] = float((flat == t).sum()) / n_heads

    per_layer_mean = per_seed_layer.mean(axis=0)
    per_layer_std = per_seed_layer.std(axis=0) if n_seeds > 1 else np.zeros_like(per_layer_mean)
    return {
        "steps": steps,
        "n_layers": n_layers,
        "per_layer_mean": per_layer_mean,
        "per_layer_std": per_layer_std,
        "type_names": type_names,
        "curve_mode": "dominance",
    }


def compute_head_trajectories(
    result: Dict,
) -> Dict[Tuple[int, int], List[int]]:
    """Extract the full dominant-label sequence for every (layer, head) pair."""

    labels = _dominant_tensor(result)
    _, n_layers, n_heads = labels.shape
    trajectories: Dict[Tuple[int, int], List[int]] = {}
    for layer in range(n_layers):
        for head in range(n_heads):
            trajectories[(layer, head)] = [int(t) for t in labels[:, layer, head].tolist()]
    return trajectories


def find_interesting_trajectories(
    trajectories: Dict[Tuple[int, int], List[int]],
    min_type_changes: int = 2,
) -> Dict[Tuple[int, int], List[int]]:
    interesting = {}
    for key, traj in trajectories.items():
        changes = sum(1 for i in range(1, len(traj)) if traj[i] != traj[i - 1])
        if changes >= min_type_changes:
            interesting[key] = traj
    return interesting


def compute_specialization_onset(
    curves: Dict[str, np.ndarray],
    threshold_frac: float = 0.05,
    exclude_positional_init: bool = False,
) -> Dict[str, Optional[int]]:
    """Find the first training step where each type/behavior exceeds threshold_frac."""

    steps = curves["steps"]
    mean_fracs = curves["mean"]
    type_names = curves["type_names"]
    curve_mode = curves.get("curve_mode", "dominance")

    onset_steps: Dict[str, Optional[int]] = {}
    for t_idx, type_name in enumerate(type_names):
        if curve_mode == "dominance" and type_name == "WEAK":
            onset_steps[type_name] = int(steps[0])
            continue

        fracs = mean_fracs[:, t_idx]
        above = np.where(fracs >= threshold_frac)[0]
        if (
            exclude_positional_init
            and type_name == "POSITIONAL"
            and len(above) > 0
            and int(steps[above[0]]) == int(steps[0])
        ):
            above = above[1:]
        onset_steps[type_name] = None if len(above) == 0 else int(steps[above[0]])

    return onset_steps


def _bootstrap_run_state(result: Dict, rng: np.random.Generator) -> Dict:
    """Bootstrap heads within each layer, preserving checkpoint count."""

    labels = _dominant_tensor(result)
    active = _active_tensor(result)
    n_ckpts, n_layers, n_heads = labels.shape
    n_behaviors = active.shape[-1]

    boot_labels = np.empty_like(labels)
    boot_active = np.empty_like(active)
    for layer in range(n_layers):
        sampled_heads = rng.integers(0, n_heads, size=n_heads)
        boot_labels[:, layer, :] = labels[:, layer, sampled_heads]
        boot_active[:, layer, :, :] = active[:, layer, sampled_heads, :]

    return {
        "label_tensor": boot_labels,
        "dominant_label_tensor": boot_labels,
        "active_behavior_tensor": boot_active,
        "step_index": result["step_index"],
        "seed": result["seed"],
        "n_layers": n_layers,
        "n_heads": n_heads,
        "type_names": _type_names(result),
        "behavior_names": _behavior_names(result),
    }


def compute_onset_bootstrap_cis(
    run_results: List[Dict],
    threshold_frac: float = 0.05,
    exclude_positional_init: bool = False,
    n_bootstraps: int = 1000,
    random_seed: int = 0,
    curve_mode: str = "dominance",
) -> Dict[str, Dict[str, Optional[int] | int]]:
    """Bootstrap confidence intervals for onset steps."""

    curve_fn = compute_global_curves if curve_mode == "dominance" else compute_activation_curves
    base_curves = curve_fn(run_results)
    point_estimates = compute_specialization_onset(
        base_curves,
        threshold_frac=threshold_frac,
        exclude_positional_init=exclude_positional_init,
    )
    type_names = base_curves["type_names"]
    bootstrap_onsets: Dict[str, List[int]] = {name: [] for name in type_names}
    rng = np.random.default_rng(random_seed)

    for _ in range(n_bootstraps):
        boot_results = [_bootstrap_run_state(result, rng) for result in run_results]
        boot_curves = curve_fn(boot_results)
        boot_onsets = compute_specialization_onset(
            boot_curves,
            threshold_frac=threshold_frac,
            exclude_positional_init=exclude_positional_init,
        )
        for type_name, onset in boot_onsets.items():
            if onset is not None:
                bootstrap_onsets[type_name].append(int(onset))

    ci_summary: Dict[str, Dict[str, Optional[int] | int]] = {}
    for type_name in type_names:
        samples = bootstrap_onsets[type_name]
        if samples:
            ci_summary[type_name] = {
                "point_estimate": point_estimates[type_name],
                "ci_lower": int(np.percentile(samples, 2.5)),
                "ci_upper": int(np.percentile(samples, 97.5)),
                "n_valid_bootstraps": len(samples),
            }
        else:
            ci_summary[type_name] = {
                "point_estimate": point_estimates[type_name],
                "ci_lower": None,
                "ci_upper": None,
                "n_valid_bootstraps": 0,
            }
    return ci_summary


def compute_mixed_behavior_summary(
    run_results: List[Dict],
) -> Dict[str, object]:
    """Aggregate mixed-behavior metadata over one or more runs."""

    min_ckpts = min(len(r["step_index"]) for r in run_results)
    steps = np.array(run_results[0]["step_index"][:min_ckpts])
    ge2_runs = []
    ge3_runs = []
    margin_runs = []
    final_pair_counts: Dict[str, int] = {}
    final_triplet_counts: Dict[str, int] = {}

    for result in run_results:
        behavior_count = np.asarray(result["behavior_count_tensor"][:min_ckpts]).reshape(min_ckpts, -1)
        dominant_margin = np.asarray(result["dominant_margin_tensor"][:min_ckpts]).reshape(min_ckpts, -1)
        ge2_runs.append((behavior_count >= 2).mean(axis=1))
        ge3_runs.append((behavior_count >= 3).mean(axis=1))
        margin_runs.append(dominant_margin.mean(axis=1))

        primary = np.asarray(result["primary_behavior_tensor"][-1]).reshape(-1)
        runner_up = np.asarray(result["runner_up_tensor"][-1]).reshape(-1)
        for dom_idx, run_idx in zip(primary, runner_up):
            if dom_idx < 0 or run_idx < 0:
                continue
            key = f"{BEHAVIOR_NAMES[int(dom_idx)]}>{BEHAVIOR_NAMES[int(run_idx)]}"
            final_pair_counts[key] = final_pair_counts.get(key, 0) + 1

        active_mask = _active_tensor(result)[-1].reshape(-1, len(_behavior_names(result)))
        for row in active_mask:
            active = [BEHAVIOR_NAMES[idx] for idx, flag in enumerate(row) if flag]
            if len(active) >= 3:
                key = "|".join(active)
                final_triplet_counts[key] = final_triplet_counts.get(key, 0) + 1

    ge2 = np.stack(ge2_runs, axis=0)
    ge3 = np.stack(ge3_runs, axis=0)
    margins = np.stack(margin_runs, axis=0)

    return {
        "steps": steps,
        "fraction_ge2_mean": ge2.mean(axis=0),
        "fraction_ge2_std": ge2.std(axis=0) if ge2.shape[0] > 1 else np.zeros_like(ge2[0]),
        "fraction_ge3_mean": ge3.mean(axis=0),
        "fraction_ge3_std": ge3.std(axis=0) if ge3.shape[0] > 1 else np.zeros_like(ge3[0]),
        "mean_dominant_margin": margins.mean(axis=0),
        "std_dominant_margin": margins.std(axis=0) if margins.shape[0] > 1 else np.zeros_like(margins[0]),
        "final_top_pairs": sorted(final_pair_counts.items(), key=lambda x: (-x[1], x[0]))[:10],
        "final_top_triplets": sorted(final_triplet_counts.items(), key=lambda x: (-x[1], x[0]))[:10],
        "source": "active_behavior_tensor" if "active_behavior_tensor" in run_results[0] else "threshold_flag_tensor",
    }


def print_trajectory_report(
    global_curves: Dict[str, np.ndarray],
    per_layer_curves: Dict[str, object],
    onset_steps: Dict[str, Optional[int]],
    learned_onset_steps: Optional[Dict[str, Optional[int]]] = None,
    onset_cis: Optional[Dict[str, Dict[str, Optional[int] | int]]] = None,
    activation_curves: Optional[Dict[str, np.ndarray]] = None,
    activation_onset_steps: Optional[Dict[str, Optional[int]]] = None,
    activation_onset_cis: Optional[Dict[str, Dict[str, Optional[int] | int]]] = None,
    mixed_behavior: Optional[Dict[str, object]] = None,
    seed: Optional[int] = None,
) -> None:
    """Print a formatted trajectory report."""

    print(f"\n{'=' * 64}")
    print(f"  Trajectory Analysis Report")
    print(f"{'=' * 64}")

    final_fracs = global_curves["mean"][-1]
    final_summary = sorted(
        [(k, v) for k, v in zip(global_curves["type_names"], final_fracs)],
        key=lambda x: x[1],
        reverse=True,
    )
    print(f"\n  Final dominant-label fractions:")
    print(f"  {'─' * 50}")
    for type_name, frac in final_summary:
        print(f"  {type_name:<20}: {frac * 100:>5.1f}%")

    print(f"\n  Dominance onset steps (point estimates):")
    print(f"  {'─' * 50}")
    for type_name, step in onset_steps.items():
        print(f"  {type_name:<20}: {step}")

    if onset_cis is not None:
        print(f"\n  Dominance onset bootstrap CIs:")
        print(f"  {'─' * 50}")
        for type_name, summary in onset_cis.items():
            print(
                f"  {type_name:<20}: {summary['point_estimate']} "
                f"[{summary['ci_lower']}, {summary['ci_upper']}]"
            )

    if activation_curves is not None and activation_onset_steps is not None:
        print(f"\n  Activation onset steps:")
        print(f"  {'─' * 50}")
        for type_name, step in activation_onset_steps.items():
            print(f"  {type_name:<20}: {step}")

    if activation_onset_cis is not None:
        print(f"\n  Activation onset bootstrap CIs:")
        print(f"  {'─' * 50}")
        for type_name, summary in activation_onset_cis.items():
            print(
                f"  {type_name:<20}: {summary['point_estimate']} "
                f"[{summary['ci_lower']}, {summary['ci_upper']}]"
            )

    if mixed_behavior is not None:
        print(f"\n  Mixed-behavior summary ({mixed_behavior['source']}):")
        print(f"  {'─' * 50}")
        print(f"  >=2 active behaviors at final step : {mixed_behavior['fraction_ge2_mean'][-1] * 100:.1f}%")
        print(f"  >=3 active behaviors at final step : {mixed_behavior['fraction_ge3_mean'][-1] * 100:.1f}%")
        print(f"  Mean dominant margin at final step : {mixed_behavior['mean_dominant_margin'][-1]:.3f}")
        if mixed_behavior["final_top_pairs"]:
            print(f"  Top final dominant/runner-up pairs:")
            for pair, count in mixed_behavior["final_top_pairs"][:5]:
                print(f"    {pair:<28} {count}")
        if mixed_behavior["final_top_triplets"]:
            print(f"  Top final active triplets:")
            for triplet, count in mixed_behavior["final_top_triplets"][:5]:
                print(f"    {triplet:<28} {count}")

    print(f"{'=' * 64}\n")
