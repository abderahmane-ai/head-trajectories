"""
data/calibration.py — Random-baseline calibration for head type thresholds.

Computes per-score thresholds by measuring head scores on a randomly
initialized model with attention maps shuffled along the sequence dimension.
Thresholds are set to mean + 2*std across all heads for each score.
"""

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from model import ModelConfig, TransformerLM
from probing.scores import score_head


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


def _calibrate_thresholds_single(
    probe_dict: Dict[str, torch.Tensor],
    config: ModelConfig,
    device: torch.device,
    seed: int,
    batch_size: int,
) -> np.ndarray:
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
    general_maps = _shuffle_attention_rows(general_maps, generator)
    induction_maps = _shuffle_attention_rows(induction_maps, generator)
    positional_maps = _shuffle_attention_rows(positional_maps, generator)

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
    means = scores_arr.mean(axis=0)
    stds = scores_arr.std(axis=0)
    thresholds = means + 2.0 * stds

    return thresholds.astype(np.float32)


def calibrate_thresholds(
    probe_dict: Dict[str, torch.Tensor],
    config: ModelConfig,
    device: torch.device,
    batch_size: int = 16,
    seeds: Optional[List[int]] = None,
    n_seeds: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    for seed in seeds:
        thresholds = _calibrate_thresholds_single(
            probe_dict=probe_dict,
            config=config,
            device=device,
            seed=seed,
            batch_size=batch_size,
        )
        thresholds_per_seed.append(thresholds)

    per_seed_arr = np.stack(thresholds_per_seed, axis=0).astype(np.float32)
    mean = per_seed_arr.mean(axis=0)
    std = per_seed_arr.std(axis=0)

    return mean.astype(np.float32), std.astype(np.float32), per_seed_arr
