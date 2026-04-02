"""
data/calibration.py — Random-baseline calibration for head type thresholds.

Computes per-score thresholds by measuring head scores on a randomly
initialized model with causally valid attention maps whose key positions
have been scrambled within each query row.

Threshold rules:
  - sink / prev-token / induction / positional: mean + 2 * std
  - semantic: null p99

The semantic rule is intentionally stricter because the semantic metric is a
signed correlation statistic whose null variance collapses sharply after the
final per-head averaging step. Using mean + 2 * std for semantic was found to
be too permissive in practice.
"""

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from model import ModelConfig, TransformerLM
from probing.scores import score_head


METRIC_NAMES: Tuple[str, ...] = (
    "SINK",
    "PREV_TOKEN",
    "INDUCTION",
    "POSITIONAL",
    "SEMANTIC",
)
CALIBRATION_VERSION: int = 3
SEMANTIC_METRIC_INDEX: int = 4
SEMANTIC_THRESHOLD_QUANTILE: float = 0.99
DEFAULT_THRESHOLD_RULES: Tuple[str, ...] = (
    "mean_plus_2std",
    "mean_plus_2std",
    "mean_plus_2std",
    "mean_plus_2std",
    f"quantile_{SEMANTIC_THRESHOLD_QUANTILE:.2f}",
)


@torch.no_grad()
def _extract_attention_maps(
    model: TransformerLM,
    token_ids: torch.Tensor,
    device: torch.device,
    batch_size: int = 16,
) -> List[torch.Tensor]:
    """
    Run token_ids through the model and return attention maps for all layers.

    Returns:
        List of length n_layers, each tensor shape (N, n_heads, T, T) on CPU.
    """

    model.eval()
    N, T = token_ids.shape
    n_layers = model.config.n_layers

    all_maps: List[List[torch.Tensor]] = [[] for _ in range(n_layers)]

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = token_ids[start:end].to(device)

        _, attn_maps = model(batch, return_attention=True)
        if attn_maps is None:
            raise RuntimeError(
                "Model returned None for attention maps. "
                "Ensure return_attention=True is propagating correctly."
            )

        for layer_idx, layer_map in enumerate(attn_maps):
            all_maps[layer_idx].append(layer_map)

    return [torch.cat(chunks, dim=0) for chunks in all_maps]


def _shuffle_attention_rows(
    attn_maps: List[torch.Tensor],
    generator: torch.Generator,
) -> List[torch.Tensor]:
    """
    Shuffle attention rows (query positions) independently per sequence.

    This preserves row sums (each row remains a probability distribution)
    while destroying positional/content structure.
    """

    shuffled: List[torch.Tensor] = []
    for layer_map in attn_maps:
        # layer_map: (N, H, T, T)
        N, H, T, _ = layer_map.shape
        out = layer_map.clone()
        for i in range(N):
            perm = torch.randperm(T, generator=generator)
            out[i] = out[i][:, perm, :]
        shuffled.append(out)
    return shuffled


def _scramble_causal_attention_keys(
    attn_maps: List[torch.Tensor],
    generator: torch.Generator,
) -> List[torch.Tensor]:
    """
    Scramble key positions independently within each valid causal row.

    Unlike row shuffling, this destroys fixed-key anchoring, previous-token
    structure, induction targets, semantic alignment, and positional
    comparability while preserving:
      - causal support (no future-token mass is introduced)
      - row-stochasticity (each row still sums to 1)
    """

    scrambled: List[torch.Tensor] = []

    for layer_map in attn_maps:
        # layer_map: (N, H, T, T)
        N, H, T, _ = layer_map.shape
        out = torch.zeros_like(layer_map)

        for n in range(N):
            for h in range(H):
                for t in range(T):
                    valid = layer_map[n, h, t, : t + 1]
                    perm = torch.randperm(
                        t + 1,
                        generator=generator,
                        device=layer_map.device,
                    )
                    out[n, h, t, : t + 1] = valid[perm]

        # Re-normalize defensively to preserve exact row-stochasticity.
        row_sums = out.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        scrambled.append(out / row_sums)

    return scrambled


def _compute_threshold_statistics(
    scores_arr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute threshold statistics for one calibration seed.

    Returns:
        thresholds, means, stds, quantiles_p95, quantiles_p99
    """

    means = scores_arr.mean(axis=0)
    stds = scores_arr.std(axis=0)
    quantiles_p95 = np.quantile(scores_arr, 0.95, axis=0)
    quantiles_p99 = np.quantile(scores_arr, SEMANTIC_THRESHOLD_QUANTILE, axis=0)

    thresholds = means + 2.0 * stds
    thresholds[SEMANTIC_METRIC_INDEX] = quantiles_p99[SEMANTIC_METRIC_INDEX]

    return (
        thresholds.astype(np.float32),
        means.astype(np.float32),
        stds.astype(np.float32),
        quantiles_p95.astype(np.float32),
        quantiles_p99.astype(np.float32),
    )


def _calibrate_thresholds_single(
    probe_dict: Dict[str, torch.Tensor],
    config: ModelConfig,
    device: torch.device,
    seed: int,
    batch_size: int,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if "block_size" in probe_dict:
        block_size = int(probe_dict["block_size"].item())
        if block_size != config.block_size:
            raise ValueError(
                f"Probe block_size ({block_size}) does not match "
                f"model config block_size ({config.block_size})."
            )

    model = TransformerLM(config).to(device)
    model.eval()

    general_seqs = probe_dict["general_seqs"]
    induction_seqs = probe_dict["induction_seqs"]
    positional_seqs = probe_dict["positional_seqs"]

    general_maps = _extract_attention_maps(model, general_seqs, device, batch_size)
    induction_maps = _extract_attention_maps(model, induction_seqs, device, batch_size)
    positional_maps = _extract_attention_maps(model, positional_seqs, device, batch_size)

    generator = torch.Generator().manual_seed(seed + 1)
    general_maps = _scramble_causal_attention_keys(general_maps, generator)
    induction_maps = _scramble_causal_attention_keys(induction_maps, generator)
    positional_maps = _scramble_causal_attention_keys(positional_maps, generator)

    induction_p1 = probe_dict["induction_p1"]
    induction_p2 = probe_dict["induction_p2"]
    positional_pairs = probe_dict["positional_pairs"]
    token_ids = probe_dict["general_seqs"]
    embedding_matrix = model.get_embedding_matrix()

    score_list: List[List[float]] = []

    for layer_idx in range(config.n_layers):
        general_layer = general_maps[layer_idx]
        induction_layer = induction_maps[layer_idx]
        positional_layer = positional_maps[layer_idx]

        for head_idx in range(config.n_heads):
            scores = score_head(
                general_attn=general_layer[:, head_idx, :, :],
                induction_attn=induction_layer[:, head_idx, :, :],
                positional_attn=positional_layer[:, head_idx, :, :],
                induction_p1=induction_p1,
                induction_p2=induction_p2,
                positional_pairs=positional_pairs,
                token_ids=token_ids,
                embedding_matrix=embedding_matrix,
            )
            score_list.append(list(scores))

    scores_arr = np.array(score_list, dtype=np.float32)  # (n_heads_total, 5)
    thresholds, means, stds, quantiles_p95, quantiles_p99 = _compute_threshold_statistics(
        scores_arr
    )

    return (
        thresholds.astype(np.float32),
        means.astype(np.float32),
        stds.astype(np.float32),
        quantiles_p95.astype(np.float32),
        quantiles_p99.astype(np.float32),
        (thresholds <= 0.0).astype(np.bool_),
        scores_arr.astype(np.float32),
    )


def calibrate_thresholds(
    probe_dict: Dict[str, torch.Tensor],
    config: ModelConfig,
    device: torch.device,
    batch_size: int = 16,
    seeds: Optional[List[int]] = None,
    n_seeds: int = 3,
    return_diagnostics: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Dict[str, np.ndarray | bool | List[str] | float],
]:
    """
    Compute random-baseline thresholds for the five head-type scores across
    multiple random seeds to assess stability.

    Returns:
        (mean_thresholds, std_thresholds, per_seed_thresholds)
        Each array is shape (5,), per_seed is shape (n_seeds, 5).
    """

    base_seed = int(probe_dict.get("creation_seed", torch.tensor(0)).item())
    if seeds is None:
        seeds = [base_seed + i for i in range(n_seeds)]

    thresholds_per_seed: List[np.ndarray] = []
    means_per_seed: List[np.ndarray] = []
    stds_per_seed: List[np.ndarray] = []
    p95_per_seed: List[np.ndarray] = []
    p99_per_seed: List[np.ndarray] = []
    nonpositive_mask_per_seed: List[np.ndarray] = []
    null_scores_per_seed: List[np.ndarray] = []
    for seed in seeds:
        (
            thresholds,
            means,
            stds,
            p95,
            p99,
            nonpositive_mask,
            null_scores,
        ) = _calibrate_thresholds_single(
            probe_dict=probe_dict,
            config=config,
            device=device,
            seed=seed,
            batch_size=batch_size,
        )
        thresholds_per_seed.append(thresholds)
        means_per_seed.append(means)
        stds_per_seed.append(stds)
        p95_per_seed.append(p95)
        p99_per_seed.append(p99)
        nonpositive_mask_per_seed.append(nonpositive_mask)
        null_scores_per_seed.append(null_scores)

    per_seed_arr = np.stack(thresholds_per_seed, axis=0).astype(np.float32)
    per_seed_means = np.stack(means_per_seed, axis=0).astype(np.float32)
    per_seed_stds = np.stack(stds_per_seed, axis=0).astype(np.float32)
    per_seed_p95 = np.stack(p95_per_seed, axis=0).astype(np.float32)
    per_seed_p99 = np.stack(p99_per_seed, axis=0).astype(np.float32)
    per_seed_nonpositive = np.stack(nonpositive_mask_per_seed, axis=0).astype(bool)
    per_seed_null_scores = np.stack(null_scores_per_seed, axis=0).astype(np.float32)
    pooled_null_scores = np.concatenate(null_scores_per_seed, axis=0).astype(np.float32)
    mean = per_seed_arr.mean(axis=0)
    std = per_seed_arr.std(axis=0)

    if not return_diagnostics:
        return mean.astype(np.float32), std.astype(np.float32), per_seed_arr

    diagnostics: Dict[str, np.ndarray | bool] = {
        "per_seed_metric_means": per_seed_means,
        "per_seed_metric_stds": per_seed_stds,
        "per_seed_metric_p95": per_seed_p95,
        "per_seed_metric_p99": per_seed_p99,
        "per_seed_nonpositive_mask": per_seed_nonpositive,
        "mean_threshold_nonpositive_mask": (mean <= 0.0),
        "requires_sanitization": bool((per_seed_nonpositive.any()) or (mean <= 0.0).any()),
        "metric_names": list(METRIC_NAMES),
        "threshold_rules": list(DEFAULT_THRESHOLD_RULES),
        "semantic_threshold_quantile": SEMANTIC_THRESHOLD_QUANTILE,
        "null_seed_list": np.asarray(seeds, dtype=np.int64),
        "per_seed_null_scores": per_seed_null_scores,
        "pooled_null_scores": pooled_null_scores,
    }

    return mean.astype(np.float32), std.astype(np.float32), per_seed_arr, diagnostics
