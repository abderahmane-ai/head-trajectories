"""
analysis/trajectories.py — Developmental trajectory computation.

Computes:
1. Global type-fraction curves
2. Per-layer type-fraction curves
3. Head-level trajectories
4. Onset-step point estimates and bootstrap confidence intervals
5. Mixed-behavior summaries derived from classifier metadata
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from probing import HEAD_TYPES, LABEL_UNDIFF
from probing.classifier import HeadClassifier


def load_run_results(results_path: Path) -> Dict:
    """Load a saved probing results file for one run."""

    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    return HeadClassifier.load(results_path)


def _type_names(result: Dict) -> List[str]:
    return list(result.get("type_names", HEAD_TYPES))


def compute_global_curves(
    run_results: List[Dict],
) -> Dict[str, np.ndarray]:
    """Compute global head-type fraction curves averaged over seeds."""

    n_seeds = len(run_results)
    min_ckpts = min(len(r["step_index"]) for r in run_results)
    steps = np.array(run_results[0]["step_index"][:min_ckpts])
    type_names = _type_names(run_results[0])
    n_types = len(type_names)

    per_seed = np.zeros((n_seeds, min_ckpts, n_types), dtype=np.float32)
    for s_idx, result in enumerate(run_results):
        labels = np.asarray(result["label_tensor"][:min_ckpts])
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
    }


def compute_per_layer_curves(
    run_results: List[Dict],
) -> Dict[str, object]:
    """Compute head-type fraction curves separately for each layer."""

    n_seeds = len(run_results)
    min_ckpts = min(len(r["step_index"]) for r in run_results)
    steps = np.array(run_results[0]["step_index"][:min_ckpts])
    n_layers = run_results[0]["n_layers"]
    n_heads = run_results[0]["n_heads"]
    type_names = _type_names(run_results[0])
    n_types = len(type_names)

    per_seed_layer = np.zeros((n_seeds, n_layers, min_ckpts, n_types), dtype=np.float32)
    for s_idx, result in enumerate(run_results):
        labels = np.asarray(result["label_tensor"][:min_ckpts])
        for layer in range(n_layers):
            for ckpt in range(min_ckpts):
                flat = labels[ckpt, layer]
                for t in range(n_types):
                    per_seed_layer[s_idx, layer, ckpt, t] = (
                        float((flat == t).sum()) / n_heads
                    )

    per_layer_mean = per_seed_layer.mean(axis=0)
    per_layer_std = (
        per_seed_layer.std(axis=0) if n_seeds > 1 else np.zeros_like(per_layer_mean)
    )
    return {
        "steps": steps,
        "n_layers": n_layers,
        "per_layer_mean": per_layer_mean,
        "per_layer_std": per_layer_std,
        "type_names": type_names,
    }


def compute_head_trajectories(
    result: Dict,
) -> Dict[Tuple[int, int], List[int]]:
    """Extract the full label sequence for every (layer, head) pair."""

    labels = np.asarray(result["label_tensor"])
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
    """Filter for heads that undergo at least min_type_changes transitions."""

    interesting = {}
    for key, traj in trajectories.items():
        changes = sum(1 for i in range(1, len(traj)) if traj[i] != traj[i - 1])
        if changes >= min_type_changes:
            interesting[key] = traj
    return interesting


def compute_specialization_onset(
    global_curves: Dict[str, np.ndarray],
    threshold_frac: float = 0.05,
    exclude_positional_init: bool = False,
) -> Dict[str, Optional[int]]:
    """Find the first training step where each type exceeds threshold_frac."""

    steps = global_curves["steps"]
    mean_fracs = global_curves["mean"]
    type_names = global_curves["type_names"]

    onset_steps: Dict[str, Optional[int]] = {}
    for t_idx, type_name in enumerate(type_names):
        if type_name == "UNDIFFERENTIATED":
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


def _bootstrap_run_labels(result: Dict, rng: np.random.Generator) -> Dict:
    """Bootstrap heads within each layer, preserving the number of checkpoints."""

    labels = np.asarray(result["label_tensor"])
    n_ckpts, n_layers, n_heads = labels.shape
    boot_labels = np.empty_like(labels)

    for layer in range(n_layers):
        sampled_heads = rng.integers(0, n_heads, size=n_heads)
        boot_labels[:, layer, :] = labels[:, layer, sampled_heads]

    return {
        "label_tensor": boot_labels,
        "step_index": result["step_index"],
        "seed": result["seed"],
        "n_layers": n_layers,
        "n_heads": n_heads,
        "type_names": _type_names(result),
    }


def compute_onset_bootstrap_cis(
    run_results: List[Dict],
    threshold_frac: float = 0.05,
    exclude_positional_init: bool = False,
    n_bootstraps: int = 1000,
    random_seed: int = 0,
) -> Dict[str, Dict[str, Optional[int] | int]]:
    """Bootstrap confidence intervals for onset steps."""

    base_curves = compute_global_curves(run_results)
    point_estimates = compute_specialization_onset(
        base_curves,
        threshold_frac=threshold_frac,
        exclude_positional_init=exclude_positional_init,
    )
    type_names = base_curves["type_names"]
    bootstrap_onsets: Dict[str, List[int]] = {name: [] for name in type_names}
    rng = np.random.default_rng(random_seed)

    for _ in range(n_bootstraps):
        boot_results = [_bootstrap_run_labels(result, rng) for result in run_results]
        boot_curves = compute_global_curves(boot_results)
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
            key = f"{HEAD_TYPES[int(dom_idx) + 1]}>{HEAD_TYPES[int(run_idx) + 1]}"
            final_pair_counts[key] = final_pair_counts.get(key, 0) + 1

        threshold_flags = np.asarray(result["threshold_flag_tensor"][-1]).reshape(-1, 5)
        for row in threshold_flags:
            active = [HEAD_TYPES[idx + 1] for idx, flag in enumerate(row) if flag]
            if len(active) >= 3:
                key = "|".join(active[:3])
                final_triplet_counts[key] = final_triplet_counts.get(key, 0) + 1

    ge2_arr = np.stack(ge2_runs, axis=0)
    ge3_arr = np.stack(ge3_runs, axis=0)
    margin_arr = np.stack(margin_runs, axis=0)

    return {
        "steps": steps,
        "fraction_ge2_mean": ge2_arr.mean(axis=0),
        "fraction_ge2_std": ge2_arr.std(axis=0) if len(run_results) > 1 else np.zeros(min_ckpts),
        "fraction_ge3_mean": ge3_arr.mean(axis=0),
        "fraction_ge3_std": ge3_arr.std(axis=0) if len(run_results) > 1 else np.zeros(min_ckpts),
        "mean_dominant_margin": margin_arr.mean(axis=0),
        "final_top_pairs": sorted(final_pair_counts.items(), key=lambda item: (-item[1], item[0]))[:10],
        "final_top_triplets": sorted(final_triplet_counts.items(), key=lambda item: (-item[1], item[0]))[:10],
    }


def print_trajectory_report(
    global_curves: Dict[str, np.ndarray],
    per_layer_curves: Dict[str, object],
    onset_steps: Dict[str, Optional[int]],
    seed: int,
    learned_onset_steps: Optional[Dict[str, Optional[int]]] = None,
    onset_cis: Optional[Dict[str, Dict[str, Optional[int] | int]]] = None,
    mixed_behavior: Optional[Dict[str, object]] = None,
) -> None:
    """Print a formatted summary of trajectory analysis results."""

    mean = global_curves["mean"]
    std = global_curves["std"]

    print(f"\n{'=' * 64}")
    print(f"  Trajectory Analysis Report — Seed {seed}")
    print(f"{'=' * 64}")

    print(f"\n  Specialization onset (first step with ≥5% of heads):")
    print(f"  {'─' * 50}")
    sorted_types = sorted(
        [(k, v) for k, v in onset_steps.items() if k != "UNDIFFERENTIATED"],
        key=lambda x: (x[1] is None, x[1] or 0),
    )
    for type_name, step in sorted_types:
        step_str = f"{step:>10,}" if step is not None else "      never"
        if onset_cis is not None:
            ci = onset_cis.get(type_name)
            if ci and ci["ci_lower"] is not None and ci["ci_upper"] is not None:
                step_str = f"{step_str}  [{ci['ci_lower']:,}, {ci['ci_upper']:,}]"
        print(f"  {type_name:<20}: {step_str}")

    if learned_onset_steps is not None:
        print(f"\n  Learned onset (excluding architectural POSITIONAL at step 0):")
        print(f"  {'─' * 50}")
        sorted_learned = sorted(
            [(k, v) for k, v in learned_onset_steps.items() if k != "UNDIFFERENTIATED"],
            key=lambda x: (x[1] is None, x[1] or 0),
        )
        for type_name, step in sorted_learned:
            step_str = f"{step:>10,}" if step is not None else "      never"
            if onset_cis is not None:
                ci = onset_cis.get(type_name)
                if ci and ci["ci_lower"] is not None and ci["ci_upper"] is not None:
                    step_str = f"{step_str}  [{ci['ci_lower']:,}, {ci['ci_upper']:,}]"
            print(f"  {type_name:<20}: {step_str}")

    print(f"\n  Final checkpoint fractions (mean ± std across seeds):")
    print(f"  {'─' * 50}")
    for t_idx, type_name in enumerate(global_curves["type_names"]):
        if type_name == "UNDIFFERENTIATED":
            continue
        m = mean[-1, t_idx]
        s = std[-1, t_idx]
        bar = "█" * int(m * 40)
        print(f"  {type_name:<20}: {m:.3f} ± {s:.3f}  {bar}")

    if mixed_behavior is not None:
        print(f"\n  Mixed-behavior summary at final checkpoint:")
        print(f"  {'─' * 50}")
        print(
            f"  Heads with ≥2 behaviors above threshold : "
            f"{float(mixed_behavior['fraction_ge2_mean'][-1]):.3f}"
        )
        print(
            f"  Heads with ≥3 behaviors above threshold : "
            f"{float(mixed_behavior['fraction_ge3_mean'][-1]):.3f}"
        )
        print(
            f"  Mean dominant margin                    : "
            f"{float(mixed_behavior['mean_dominant_margin'][-1]):.3f}"
        )
        if mixed_behavior["final_top_pairs"]:
            print(f"  Top dominant/runner-up pairs           :")
            for pair_name, count in mixed_behavior["final_top_pairs"][:5]:
                print(f"    {pair_name:<28} {count}")

    print(f"\n  Layer stratification (step at which each layer reaches")
    print(f"  ≥50% specialized heads):")
    print(f"  {'─' * 50}")
    n_layers = per_layer_curves["n_layers"]
    per_layer_mean = per_layer_curves["per_layer_mean"]
    layer_steps = per_layer_curves["steps"]

    for layer in range(n_layers):
        undiff_frac = per_layer_mean[layer, :, LABEL_UNDIFF]
        spec_frac = 1.0 - undiff_frac
        above_50 = np.where(spec_frac >= 0.5)[0]
        onset = int(layer_steps[above_50[0]]) if len(above_50) > 0 else None
        onset_str = f"{onset:>8,}" if onset is not None else "   never"
        print(f"  Layer {layer:<3}: {onset_str}")

    print(f"\n{'=' * 64}\n")
