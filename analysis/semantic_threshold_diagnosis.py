#!/usr/bin/env python3
"""
Diagnose unexpectedly small semantic thresholds.

This script does not modify the experiment logic. It characterizes the
semantic metric under:
  1. the calibrated random baseline (causal key-scramble null)
  2. the real model at the earliest available checkpoint (ideally step 0)

Outputs:
  - JSON report with summary statistics
  - Markdown summary for human review
  - Histogram of semantic per-head scores (null vs step 0)
  - Histogram of pooled semantic per-position correlations (null vs step 0)

Typical usage:
    python analysis/semantic_threshold_diagnosis.py \
        --profile wikitext103_15m_preliminary \
        --seed 42 \
        --artifact_root artifacts
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.calibration import (
    DEFAULT_THRESHOLD_RULES,
    METRIC_NAMES,
    SEMANTIC_METRIC_INDEX,
    _extract_attention_maps,
    _scramble_causal_attention_keys,
)
from experiments.profiles import ExperimentProfile, get_profile
from experiments.runner import resolve_artifacts, resolve_device
from model import TransformerLM
from probing.classifier import HEAD_TYPES, classify_head
from probing.extractor import extract_checkpoint
from probing.pipeline import discover_checkpoints
from probing.scores import score_head


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose semantic threshold behavior for one run profile/seed.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--profile", required=True, help="Experiment profile name.")
    parser.add_argument("--seed", type=int, default=42, help="Run seed.")
    parser.add_argument(
        "--artifact_root",
        type=Path,
        default=Path("artifacts"),
        help="Root directory containing run artifacts.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for diagnosis computations.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Inference batch size for attention extraction.",
    )
    parser.add_argument(
        "--calibration_seeds",
        type=int,
        nargs="*",
        default=None,
        help="Optional explicit calibration seeds. Defaults to the seeds stored in the probe dataset or profile default.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Optional explicit output directory. Defaults to artifacts/<profile>/seed<seed>/diagnostics/semantic_threshold.",
    )
    return parser.parse_args()


def _load_probe_dict(path: Path) -> Dict[str, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(f"Probe dataset not found: {path}")
    return torch.load(path, weights_only=True)


def _threshold_key(profile: ExperimentProfile) -> str:
    return "calibrated_thresholds_6m" if profile.model_config.ablation_mode else "calibrated_thresholds_15m"


def _stats(values: np.ndarray) -> Dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def _percentile_rank(values: np.ndarray, threshold: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan")
    return float(100.0 * np.mean(values <= threshold))


def _semantic_position_correlations(
    attn_head: torch.Tensor,
    token_ids: torch.Tensor,
    embedding_matrix: torch.Tensor,
) -> torch.Tensor:
    """
    Return the pooled per-position Pearson correlations used to build semantic_score.

    This mirrors probing.scores.semantic_score but keeps the intermediate
    per-position correlations instead of averaging them immediately.
    """

    _, T = token_ids.shape
    embed_norm = F.normalize(embedding_matrix.float(), dim=-1)
    seq_embeds = embed_norm[token_ids]
    sim_matrix = torch.bmm(seq_embeds, seq_embeds.transpose(1, 2))

    correlations: List[torch.Tensor] = []
    valid_positions = torch.arange(4, T, device=attn_head.device)

    for i in valid_positions:
        i_int = int(i.item())
        attn_window = attn_head[:, i_int, : i_int + 1]
        sim_window = sim_matrix[:, i_int, : i_int + 1]

        mask = torch.ones(i_int + 1, dtype=torch.bool, device=attn_head.device)
        mask[i_int] = False
        if i_int > 0:
            mask[i_int - 1] = False
        mask[0] = False

        if int(mask.sum().item()) < 6:
            continue

        attn_window = attn_window[:, mask]
        sim_window = sim_window[:, mask]

        attn_mean = attn_window.mean(dim=1, keepdim=True)
        sim_mean = sim_window.mean(dim=1, keepdim=True)
        attn_centered = attn_window - attn_mean
        sim_centered = sim_window - sim_mean
        attn_std = attn_centered.std(dim=1, unbiased=False)
        sim_std = sim_centered.std(dim=1, unbiased=False)
        cov = (attn_centered * sim_centered).mean(dim=1)

        valid_mask = (attn_std > 1e-8) & (sim_std > 1e-8)
        if valid_mask.any():
            corr = cov[valid_mask] / (attn_std[valid_mask] * sim_std[valid_mask])
            corr = corr[torch.isfinite(corr)]
            if corr.numel() > 0:
                correlations.append(corr.detach().cpu())

    if not correlations:
        return torch.empty(0, dtype=torch.float32)

    return torch.cat(correlations, dim=0)


def _score_head_bundle(
    general_maps: List[torch.Tensor],
    induction_maps: List[torch.Tensor],
    positional_maps: List[torch.Tensor],
    probe_dict: Dict[str, torch.Tensor],
    embedding_matrix: torch.Tensor,
    config,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return:
      - scores: (n_total_heads, 5)
      - semantic_position_corrs: pooled per-position correlations across all heads
    """

    induction_p1 = probe_dict["induction_p1"]
    induction_p2 = probe_dict["induction_p2"]
    positional_pairs = probe_dict["positional_pairs"]
    token_ids = probe_dict["general_seqs"]

    score_rows: List[List[float]] = []
    semantic_corrs: List[np.ndarray] = []

    for layer_idx in range(config.n_layers):
        general_layer = general_maps[layer_idx]
        induction_layer = induction_maps[layer_idx]
        positional_layer = positional_maps[layer_idx]

        for head_idx in range(config.n_heads):
            general_head = general_layer[:, head_idx, :, :]
            induction_head = induction_layer[:, head_idx, :, :]
            positional_head = positional_layer[:, head_idx, :, :]

            scores = score_head(
                general_attn=general_head,
                induction_attn=induction_head,
                positional_attn=positional_head,
                induction_p1=induction_p1,
                induction_p2=induction_p2,
                positional_pairs=positional_pairs,
                token_ids=token_ids,
                embedding_matrix=embedding_matrix,
            )
            score_rows.append(list(scores))

            corr = _semantic_position_correlations(
                general_head,
                token_ids,
                embedding_matrix,
            )
            if corr.numel() > 0:
                semantic_corrs.append(corr.numpy())

    scores_arr = np.asarray(score_rows, dtype=np.float32)
    if semantic_corrs:
        pooled_corrs = np.concatenate(semantic_corrs).astype(np.float32)
    else:
        pooled_corrs = np.empty(0, dtype=np.float32)
    return scores_arr, pooled_corrs


def _collect_random_null_scores(
    profile: ExperimentProfile,
    probe_dict: Dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
    seeds: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, object]]]:
    all_scores: List[np.ndarray] = []
    all_semantic_corrs: List[np.ndarray] = []
    per_seed_summaries: List[Dict[str, object]] = []

    general_seqs = probe_dict["general_seqs"]
    induction_seqs = probe_dict["induction_seqs"]
    positional_seqs = probe_dict["positional_seqs"]

    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        model = TransformerLM(profile.model_config).to(device)
        model.eval()

        general_maps = _extract_attention_maps(model, general_seqs, device, batch_size)
        induction_maps = _extract_attention_maps(model, induction_seqs, device, batch_size)
        positional_maps = _extract_attention_maps(model, positional_seqs, device, batch_size)

        generator = torch.Generator().manual_seed(seed + 1)
        general_maps = _scramble_causal_attention_keys(general_maps, generator)
        induction_maps = _scramble_causal_attention_keys(induction_maps, generator)
        positional_maps = _scramble_causal_attention_keys(positional_maps, generator)

        embedding_matrix = model.get_embedding_matrix()
        score_arr, semantic_corrs = _score_head_bundle(
            general_maps,
            induction_maps,
            positional_maps,
            probe_dict,
            embedding_matrix,
            profile.model_config,
        )
        all_scores.append(score_arr)
        all_semantic_corrs.append(semantic_corrs)
        per_seed_summaries.append(
            {
                "seed": int(seed),
                "semantic_head_scores": _stats(score_arr[:, 4]),
                "semantic_position_corrs": _stats(semantic_corrs),
            }
        )

    return (
        np.concatenate(all_scores, axis=0),
        np.concatenate(all_semantic_corrs, axis=0) if all_semantic_corrs else np.empty(0, dtype=np.float32),
        per_seed_summaries,
    )


def _collect_real_checkpoint_scores(
    ckpt_path: Path,
    probe_dict: Dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    extraction = extract_checkpoint(
        ckpt_path=ckpt_path,
        probe_dict=probe_dict,
        device=device,
        batch_size=batch_size,
    )
    score_arr, semantic_corrs = _score_head_bundle(
        extraction.general_maps,
        extraction.induction_maps,
        extraction.positional_maps,
        probe_dict,
        extraction.embedding_matrix,
        extraction.config,
    )
    ckpt_meta = {
        "checkpoint_path": str(ckpt_path),
        "step": int(extraction.step),
        "train_loss": float(extraction.train_loss),
        "val_loss": float(extraction.val_loss),
    }
    return score_arr, semantic_corrs, ckpt_meta


def _find_earliest_checkpoint(ckpt_dir: Path) -> Path:
    ckpts = discover_checkpoints(ckpt_dir)
    return ckpts[0]


def _plot_head_score_histogram(
    null_scores: np.ndarray,
    real_scores: np.ndarray,
    threshold: float,
    output_path: Path,
) -> None:
    plt.figure(figsize=(9, 5.5))
    plt.hist(null_scores, bins=40, alpha=0.65, label="Random baseline (per-head)", density=True)
    plt.hist(real_scores, bins=40, alpha=0.55, label="Real checkpoint (per-head)", density=True)
    plt.axvline(threshold, color="black", linestyle="--", linewidth=2, label=f"threshold={threshold:.6f}")
    plt.xlabel("Semantic per-head score")
    plt.ylabel("Density")
    plt.title("Semantic Head Score Distribution: Random Null vs Real Checkpoint")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _plot_position_corr_histogram(
    null_corrs: np.ndarray,
    real_corrs: np.ndarray,
    output_path: Path,
) -> None:
    plt.figure(figsize=(9, 5.5))
    plt.hist(null_corrs, bins=60, alpha=0.65, label="Random baseline pooled position correlations", density=True)
    plt.hist(real_corrs, bins=60, alpha=0.55, label="Real checkpoint pooled position correlations", density=True)
    plt.xlabel("Per-position semantic Pearson correlation")
    plt.ylabel("Density")
    plt.title("Semantic Per-Position Correlations: Random Null vs Real Checkpoint")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _write_markdown_report(path: Path, report: Dict[str, object]) -> None:
    lines: List[str] = []
    lines.append("# Semantic Threshold Diagnosis")
    lines.append("")
    lines.append("## Identity")
    lines.append("")
    lines.append(f"- Profile: `{report['profile']}`")
    lines.append(f"- Seed: `{report['seed']}`")
    lines.append(f"- Checkpoint analyzed: `{report['checkpoint']['step']}`")
    lines.append(f"- Probe path: `{report['probe_path']}`")
    lines.append("")
    lines.append("## Threshold")
    lines.append("")
    lines.append(f"- Semantic threshold: `{report['semantic']['threshold']:.9f}`")
    lines.append(f"- Semantic threshold rule: `{report['semantic']['threshold_rule']}`")
    lines.append(f"- Threshold percentile within random-null semantic head scores: `{report['semantic']['threshold_percentile_in_null']:.2f}`")
    lines.append("")
    lines.append("## Main finding")
    lines.append("")
    lines.append(
        "The semantic threshold was produced by calibration as implemented; it was not floored or sanitized. "
        "This report diagnoses whether the threshold is small because the semantic null distribution itself is tightly centered near zero."
    )
    lines.append("")
    lines.append("## Random Null vs Real Checkpoint")
    lines.append("")
    lines.append("- Random-null semantic per-head summary:")
    lines.append(f"  - mean: `{report['semantic']['null_head_scores']['mean']:.9f}`")
    lines.append(f"  - std: `{report['semantic']['null_head_scores']['std']:.9f}`")
    lines.append(f"  - p95: `{report['semantic']['null_head_scores']['p95']:.9f}`")
    lines.append(f"  - p99: `{report['semantic']['null_head_scores']['p99']:.9f}`")
    lines.append("- Real-checkpoint semantic per-head summary:")
    lines.append(f"  - mean: `{report['semantic']['real_head_scores']['mean']:.9f}`")
    lines.append(f"  - std: `{report['semantic']['real_head_scores']['std']:.9f}`")
    lines.append(f"  - p95: `{report['semantic']['real_head_scores']['p95']:.9f}`")
    lines.append(f"  - p99: `{report['semantic']['real_head_scores']['p99']:.9f}`")
    lines.append("")
    lines.append("## Step-0 Label Effect")
    lines.append("")
    lines.append(f"- Heads with semantic score above threshold: `{report['semantic']['real_above_threshold_count']}/{report['semantic']['n_heads_total']}`")
    lines.append(f"- Heads classified as `SEMANTIC`: `{report['semantic']['real_semantic_label_count']}/{report['semantic']['n_heads_total']}`")
    lines.append("")
    lines.append("## Variance Collapse Check")
    lines.append("")
    lines.append(
        "This compares the spread of pooled per-position correlations against the spread of final per-head averages. "
        "If per-head spread is much smaller, averaging is collapsing variance."
    )
    lines.append("")
    lines.append(f"- Null per-position std: `{report['semantic']['null_position_corrs']['std']:.9f}`")
    lines.append(f"- Null per-head std: `{report['semantic']['null_head_scores']['std']:.9f}`")
    lines.append(f"- Real per-position std: `{report['semantic']['real_position_corrs']['std']:.9f}`")
    lines.append(f"- Real per-head std: `{report['semantic']['real_head_scores']['std']:.9f}`")
    lines.append("")
    lines.append("## Generated Files")
    lines.append("")
    lines.append(f"- [semantic_head_scores_null_vs_real.png]({report['outputs']['head_histogram']})")
    lines.append(f"- [semantic_position_corrs_null_vs_real.png]({report['outputs']['position_histogram']})")
    lines.append(f"- [semantic_threshold_diagnosis.json]({report['outputs']['json_report']})")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    profile = get_profile(args.profile)
    paths = resolve_artifacts(profile, args.seed, args.artifact_root)
    device = resolve_device(args.device)
    probe_dict = _load_probe_dict(paths.probe_path)
    threshold_key = _threshold_key(profile)

    if threshold_key not in probe_dict:
        raise KeyError(
            f"Threshold key {threshold_key!r} not found in {paths.probe_path}. "
            "The probe dataset may be stale or incomplete."
        )

    threshold_vector = probe_dict[threshold_key].detach().cpu().numpy().astype(np.float64)
    semantic_threshold = float(threshold_vector[SEMANTIC_METRIC_INDEX])
    threshold_rule_names = list(DEFAULT_THRESHOLD_RULES)

    calibration_seeds = args.calibration_seeds
    if calibration_seeds is None or len(calibration_seeds) == 0:
        if "calibration_seeds" in probe_dict:
            calibration_seeds = [int(x) for x in probe_dict["calibration_seeds"].tolist()]
        else:
            base_seed = int(probe_dict.get("creation_seed", torch.tensor(0)).item())
            calibration_seeds = [base_seed + i for i in range(profile.n_calibration_seeds)]

    earliest_ckpt = _find_earliest_checkpoint(paths.ckpt_dir)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = paths.seed_dir / "diagnostics" / "semantic_threshold"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Diagnosis] Profile: {profile.name}")
    print(f"[Diagnosis] Seed: {args.seed}")
    print(f"[Diagnosis] Probe path: {paths.probe_path}")
    print(f"[Diagnosis] Checkpoint: {earliest_ckpt}")
    print(f"[Diagnosis] Device: {device}")
    print(f"[Diagnosis] Calibration seeds: {calibration_seeds}")
    print()

    null_scores, null_position_corrs, per_seed_summaries = _collect_random_null_scores(
        profile=profile,
        probe_dict=probe_dict,
        device=device,
        batch_size=args.batch_size,
        seeds=calibration_seeds,
    )
    real_scores, real_position_corrs, ckpt_meta = _collect_real_checkpoint_scores(
        ckpt_path=earliest_ckpt,
        probe_dict=probe_dict,
        device=device,
        batch_size=args.batch_size,
    )

    real_labels = [
        classify_head(tuple(map(float, score_row.tolist())), threshold_vector.astype(np.float32))[0]
        for score_row in real_scores
    ]
    real_labels_arr = np.asarray(real_labels, dtype=np.int32)
    semantic_above_threshold = int((real_scores[:, SEMANTIC_METRIC_INDEX] > semantic_threshold).sum())
    semantic_label_count = int((real_labels_arr == 5).sum())

    null_metric_summaries = {
        metric_name: _stats(null_scores[:, idx])
        for idx, metric_name in enumerate(METRIC_NAMES)
    }
    report: Dict[str, object] = {
        "profile": profile.name,
        "seed": int(args.seed),
        "probe_path": str(paths.probe_path),
        "checkpoint": ckpt_meta,
        "calibration_seeds": [int(x) for x in calibration_seeds],
        "threshold_key": threshold_key,
        "thresholds": {
            metric_name: float(threshold_vector[idx])
            for idx, metric_name in enumerate(METRIC_NAMES)
        },
        "threshold_rules": {
            metric_name: str(threshold_rule_names[idx])
            for idx, metric_name in enumerate(METRIC_NAMES)
        },
        "null_metric_summaries": null_metric_summaries,
        "semantic": {
            "threshold": semantic_threshold,
            "threshold_rule": str(threshold_rule_names[SEMANTIC_METRIC_INDEX]),
            "threshold_percentile_in_null": _percentile_rank(null_scores[:, SEMANTIC_METRIC_INDEX], semantic_threshold),
            "null_head_scores": _stats(null_scores[:, SEMANTIC_METRIC_INDEX]),
            "null_position_corrs": _stats(null_position_corrs),
            "real_head_scores": _stats(real_scores[:, SEMANTIC_METRIC_INDEX]),
            "real_position_corrs": _stats(real_position_corrs),
            "real_above_threshold_count": semantic_above_threshold,
            "real_semantic_label_count": semantic_label_count,
            "n_heads_total": int(real_scores.shape[0]),
            "per_seed_null_summaries": per_seed_summaries,
        },
        "outputs": {},
    }

    head_hist_path = output_dir / "semantic_head_scores_null_vs_real.png"
    pos_hist_path = output_dir / "semantic_position_corrs_null_vs_real.png"
    json_path = output_dir / "semantic_threshold_diagnosis.json"
    md_path = output_dir / "semantic_threshold_diagnosis.md"

    _plot_head_score_histogram(
        null_scores[:, SEMANTIC_METRIC_INDEX],
        real_scores[:, SEMANTIC_METRIC_INDEX],
        semantic_threshold,
        head_hist_path,
    )
    _plot_position_corr_histogram(null_position_corrs, real_position_corrs, pos_hist_path)

    report["outputs"] = {
        "head_histogram": str(head_hist_path),
        "position_histogram": str(pos_hist_path),
        "json_report": str(json_path),
        "markdown_report": str(md_path),
    }

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_markdown_report(md_path, report)

    print("[Diagnosis] Complete.")
    print(f"[Diagnosis] JSON report: {json_path}")
    print(f"[Diagnosis] Markdown report: {md_path}")
    print(f"[Diagnosis] Head-score histogram: {head_hist_path}")
    print(f"[Diagnosis] Position-correlation histogram: {pos_hist_path}")


if __name__ == "__main__":
    main()
