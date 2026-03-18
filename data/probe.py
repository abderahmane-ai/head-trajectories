"""
data/probe.py — Probe dataset construction and loading.

Constructs the fixed probe dataset used to score attention heads at every
checkpoint. Must be called ONCE before any training begins and never again.
The resulting probe_dataset.pt file is treated as immutable throughout the
entire experiment.

Probe dataset structure:
  general_seqs:       (500, 256) int64  — general held-out sequences
  induction_seqs:     (100, 256) int64  — sequences with engineered repeats
  induction_p1:       (100,)     int64  — start index of first occurrence
  induction_p2:       (100,)     int64  — start index of second occurrence
  positional_seqs:    (100, 256) int64  — 50 pairs, stored as flat (100, 256)
  positional_pairs:   (50, 2)    int64  — row indices into positional_seqs
  heldout_general_seqs:   (100, 256) int64  — heldout general sequences
  heldout_induction_seqs: (20, 256)  int64  — heldout induction sequences
  heldout_induction_p1:   (20,)      int64  — heldout induction p1
  heldout_induction_p2:   (20,)      int64  — heldout induction p2
  heldout_positional_seqs:(20, 256)  int64  — heldout positional sequences
  heldout_positional_pairs:(10, 2)   int64  — heldout positional pairs
  calibrated_thresholds_15m: (5,) float32 — random-baseline thresholds (15M)
  calibrated_thresholds_15m_std: (5,) float32 — across calibration seeds
  calibrated_thresholds_15m_seeds: (S, 5) float32 — per-seed thresholds
  calibrated_thresholds_6m:  (5,) float32 — random-baseline thresholds (6M)
  calibrated_thresholds_6m_std: (5,) float32 — across calibration seeds
  calibrated_thresholds_6m_seeds: (S, 5) float32 — per-seed thresholds
  calibration_seeds:  (S,) int64  — seeds used for calibration
  creation_seed:      scalar int        — always 0
  block_size:         scalar int        — always 256
"""

import random
import time
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import tiktoken

from .loader import OpenWebTextStream, get_tokenizer
from .calibration import calibrate_thresholds
from model import ModelConfig


# ─────────────────────────────────────────────────────────────────────────────
# Induction Probe Construction
# ─────────────────────────────────────────────────────────────────────────────

def _build_induction_sequence(
    base_tokens: List[int],
    subseq_len: int,
    rng: random.Random,
    block_size: int = 256,
) -> Optional[Tuple[List[int], int, int]]:
    """
    Given a base token sequence, engineer an induction probe by:
    1. Selecting a random subsequence of length subseq_len from the first half
    2. Inserting a copy of that subsequence into the second half
    3. Recording p1 (first occurrence start) and p2 (second occurrence start)

    The final sequence is exactly block_size tokens.

    Returns:
        (tokens, p1, p2) or None if construction fails.
    """

    if len(base_tokens) < block_size:
        return None

    tokens = base_tokens[:block_size]

    # Select p1 from the first quarter — leaves enough room for the repeat
    max_p1 = block_size // 4 - subseq_len
    if max_p1 < 1:
        return None
    p1 = rng.randint(1, max_p1)

    # Extract the subsequence to repeat
    subseq = tokens[p1 : p1 + subseq_len]

    # Place the copy in the second half, at least (block_size // 2) away from p1
    min_p2 = max(p1 + subseq_len + 20, block_size // 2)
    max_p2 = block_size - subseq_len - 2
    if min_p2 >= max_p2:
        return None
    p2 = rng.randint(min_p2, max_p2)

    # Splice the subsequence into the sequence at p2
    tokens[p2 : p2 + subseq_len] = subseq

    return tokens, p1, p2


def build_induction_probes(
    raw_sequences: List[List[int]],
    n_probes: int = 100,
    subseq_len_range: Tuple[int, int] = (5, 10),
    block_size: int = 256,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build n_probes induction probe sequences from raw token sequences.

    For each probe:
    - Selects a random subsequence of length in subseq_len_range
    - Splices a copy into the second half
    - Records p1 and p2

    Returns:
        induction_seqs: (n_probes, block_size) int64
        induction_p1:   (n_probes,)            int64
        induction_p2:   (n_probes,)            int64
    """

    rng = random.Random(seed)
    seqs:  List[List[int]] = []
    p1s:   List[int]       = []
    p2s:   List[int]       = []

    attempts = 0
    src_idx  = 0

    while len(seqs) < n_probes:
        if src_idx >= len(raw_sequences):
            raise RuntimeError(
                f"Ran out of source sequences after {attempts} attempts. "
                f"Only built {len(seqs)}/{n_probes} induction probes."
            )

        base = raw_sequences[src_idx]
        src_idx += 1
        attempts += 1

        subseq_len = rng.randint(*subseq_len_range)
        result = _build_induction_sequence(base, subseq_len, rng, block_size)

        if result is not None:
            tokens, p1, p2 = result
            seqs.append(tokens)
            p1s.append(p1)
            p2s.append(p2)

    print(
        f"  Induction probes: built {n_probes} from {attempts} candidates "
        f"(success rate: {n_probes / attempts * 100:.1f}%)"
    )

    return (
        torch.tensor(seqs,  dtype=torch.long),
        torch.tensor(p1s,   dtype=torch.long),
        torch.tensor(p2s,   dtype=torch.long),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Positional Probe Construction
# ─────────────────────────────────────────────────────────────────────────────

def build_positional_probes(
    raw_sequences: List[List[int]],
    n_pairs: int = 50,
    block_size: int = 256,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build n_pairs of positional probe sequence pairs.

    Each pair (A, B) contains two sequences of the same length (block_size)
    with completely different content. A positional head will produce nearly
    identical attention patterns for A and B; a semantic head will not.

    Pairs are drawn from non-overlapping regions of raw_sequences to ensure
    content diversity.

    Returns:
        positional_seqs:  (2*n_pairs, block_size) int64 — flat storage
        positional_pairs: (n_pairs, 2)            int64 — row indices into above
    """

    rng = random.Random(seed + 1000)

    # We need 2 * n_pairs sequences — each pair must be content-distinct
    # Shuffle source sequences to maximize content diversity
    indices = list(range(len(raw_sequences)))
    rng.shuffle(indices)

    needed = 2 * n_pairs
    if len(indices) < needed:
        raise RuntimeError(
            f"Need {needed} sequences for positional probes, "
            f"only {len(indices)} available."
        )

    selected = [raw_sequences[i][:block_size] for i in indices[:needed]]

    # Pad any short sequences with EOT token (50256)
    for i, seq in enumerate(selected):
        if len(seq) < block_size:
            selected[i] = seq + [50256] * (block_size - len(seq))

    seqs_tensor  = torch.tensor(selected, dtype=torch.long)      # (2*n_pairs, T)
    pair_indices = torch.arange(2 * n_pairs).view(n_pairs, 2)    # (n_pairs, 2)

    print(
        f"  Positional probes: built {n_pairs} pairs "
        f"({2 * n_pairs} sequences total)"
    )

    return seqs_tensor, pair_indices


# ─────────────────────────────────────────────────────────────────────────────
# General Probe Sequences
# ─────────────────────────────────────────────────────────────────────────────

def build_general_probes(
    raw_sequences: List[List[int]],
    n_seqs: int = 500,
    block_size: int = 256,
    seed: int = 0,
) -> torch.Tensor:
    """
    Build n_seqs general held-out probe sequences.
    These are used for sink_score, prev_token_score, and semantic_score.

    Returns:
        general_seqs: (n_seqs, block_size) int64
    """

    rng = random.Random(seed + 2000)
    indices = list(range(len(raw_sequences)))
    rng.shuffle(indices)

    seqs = []
    for i in indices[:n_seqs]:
        seq = raw_sequences[i][:block_size]
        if len(seq) < block_size:
            seq = seq + [50256] * (block_size - len(seq))
        seqs.append(seq)

    if len(seqs) < n_seqs:
        raise RuntimeError(
            f"Need {n_seqs} general probe sequences, only {len(seqs)} available."
        )

    print(f"  General probes  : built {n_seqs} sequences")
    return torch.tensor(seqs, dtype=torch.long)


# ─────────────────────────────────────────────────────────────────────────────
# Master Construction Function
# ─────────────────────────────────────────────────────────────────────────────

def build_probe_dataset(
    output_path: Path,
    block_size: int = 256,
    seed: int = 0,
    cache_dir: Optional[Path] = None,
    n_general:   int = 500,
    n_induction: int = 100,
    n_pairs:     int = 50,
    n_general_holdout:   int = 100,
    n_induction_holdout: int = 20,
    n_pairs_holdout:     int = 10,
    n_calibration_seeds: int = 3,
) -> Dict[str, torch.Tensor]:
    """
    Construct the full probe dataset and save to output_path.

    This function must be called exactly once before any training run.
    The output file is deterministic given the same seed and dataset.

    Args:
        output_path: path to save probe_dataset.pt
        block_size:  token sequence length (must match model block_size)
        seed:        construction seed (always 0 for reproducibility)
        cache_dir:   HuggingFace dataset cache directory
        n_general:   number of general probe sequences
        n_induction: number of induction probe sequences
        n_pairs:     number of positional probe pairs
        n_general_holdout:   heldout general probe sequences
        n_induction_holdout: heldout induction probe sequences
        n_pairs_holdout:     heldout positional probe pairs
        n_calibration_seeds: number of random seeds for calibration

    Returns:
        probe_dict: dictionary of all probe tensors (also saved to disk)
    """

    if output_path.exists():
        print(f"\n  [INFO] Probe dataset already exists at {output_path}")
        print(f"  [INFO] Loading existing probe dataset...\n")
        return load_probe_dataset(output_path)

    print(f"\n{'=' * 60}")
    print(f"  Building probe dataset")
    print(f"  Output path : {output_path}")
    print(f"  Seed        : {seed}")
    print(f"  Block size  : {block_size}")
    print(f"{'=' * 60}\n")

    t0 = time.time()

    # Set all random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load raw sequences from the probe split of OpenWebText
    print("  [1/4] Loading probe split sequences from OpenWebText...")
    stream = OpenWebTextStream(
        split="probe",
        block_size=block_size,
        seed=seed,
        cache_dir=cache_dir,
    )

    # We need n_general + n_induction + 2*n_pairs sequences (with headroom)
    general_pool_size = n_general * 2
    induction_pool_size = n_induction * 3
    positional_pool_size = n_pairs * 4
    general_holdout_pool_size = n_general_holdout * 2
    induction_holdout_pool_size = n_induction_holdout * 3
    positional_holdout_pool_size = n_pairs_holdout * 4

    n_needed = (
        general_pool_size
        + induction_pool_size
        + positional_pool_size
        + general_holdout_pool_size
        + induction_holdout_pool_size
        + positional_holdout_pool_size
    )
    raw_sequences = stream.get_raw_tokens(n_needed)
    print(f"  Collected {len(raw_sequences):,} raw sequences from probe split\n")

    # Shuffle raw sequences deterministically before allocating to sub-probes
    rng = random.Random(seed)
    rng.shuffle(raw_sequences)

    # Allocate non-overlapping source pools for each sub-probe type
    offset = 0
    pool_general = raw_sequences[offset : offset + general_pool_size]
    offset += general_pool_size
    pool_induction = raw_sequences[offset : offset + induction_pool_size]
    offset += induction_pool_size
    pool_positional = raw_sequences[offset : offset + positional_pool_size]
    offset += positional_pool_size
    pool_general_holdout = raw_sequences[offset : offset + general_holdout_pool_size]
    offset += general_holdout_pool_size
    pool_induction_holdout = raw_sequences[offset : offset + induction_holdout_pool_size]
    offset += induction_holdout_pool_size
    pool_positional_holdout = raw_sequences[offset : offset + positional_holdout_pool_size]

    # Build sub-probes
    print("  [2/6] Building general probe sequences...")
    general_seqs = build_general_probes(pool_general, n_general, block_size, seed)

    print("\n  [3/6] Building induction probe sequences...")
    induction_seqs, induction_p1, induction_p2 = build_induction_probes(
        pool_induction, n_induction, (5, 10), block_size, seed
    )

    print("\n  [4/6] Building positional probe sequences...")
    positional_seqs, positional_pairs = build_positional_probes(
        pool_positional, n_pairs, block_size, seed
    )

    # Heldout probes (optional)
    heldout_general_seqs = None
    heldout_induction_seqs = None
    heldout_induction_p1 = None
    heldout_induction_p2 = None
    heldout_positional_seqs = None
    heldout_positional_pairs = None
    if n_general_holdout > 0 or n_induction_holdout > 0 or n_pairs_holdout > 0:
        print("\n  [5/6] Building heldout probe sequences...")
        heldout_seed = seed + 5000
        if n_general_holdout > 0:
            heldout_general_seqs = build_general_probes(
                pool_general_holdout, n_general_holdout, block_size, heldout_seed
            )
        if n_induction_holdout > 0:
            heldout_induction_seqs, heldout_induction_p1, heldout_induction_p2 = (
                build_induction_probes(
                    pool_induction_holdout, n_induction_holdout, (5, 10),
                    block_size, heldout_seed
                )
            )
        if n_pairs_holdout > 0:
            heldout_positional_seqs, heldout_positional_pairs = (
                build_positional_probes(
                    pool_positional_holdout, n_pairs_holdout, block_size, heldout_seed
                )
            )

    # Assemble
    probe_dict: Dict[str, torch.Tensor] = {
        "general_seqs":      general_seqs,       # (500, 256)
        "induction_seqs":    induction_seqs,      # (100, 256)
        "induction_p1":      induction_p1,        # (100,)
        "induction_p2":      induction_p2,        # (100,)
        "positional_seqs":   positional_seqs,     # (100, 256)
        "positional_pairs":  positional_pairs,    # (50, 2)
        "creation_seed":     torch.tensor(seed,        dtype=torch.long),
        "block_size":        torch.tensor(block_size,  dtype=torch.long),
    }

    # Calibrate thresholds from random baseline (15M + 6M)
    print("\n  [6/6] Calibrating random-baseline thresholds...")
    thresholds_15m, thresholds_15m_std, thresholds_15m_seeds = calibrate_thresholds(
        probe_dict=probe_dict,
        config=ModelConfig.small_15m(),
        device=torch.device("cpu"),
        n_seeds=n_calibration_seeds,
    )
    thresholds_6m, thresholds_6m_std, thresholds_6m_seeds = calibrate_thresholds(
        probe_dict=probe_dict,
        config=ModelConfig.ablation_6m(),
        device=torch.device("cpu"),
        n_seeds=n_calibration_seeds,
    )
    print(f"  15M thresholds mean : {thresholds_15m.tolist()}")
    print(f"  15M thresholds std  : {thresholds_15m_std.tolist()}")
    print(f"  6M thresholds mean  : {thresholds_6m.tolist()}")
    print(f"  6M thresholds std   : {thresholds_6m_std.tolist()}")
    calibration_seeds = torch.tensor(
        [seed + i for i in range(n_calibration_seeds)], dtype=torch.long
    )
    probe_dict["calibrated_thresholds_15m"] = torch.tensor(
        thresholds_15m, dtype=torch.float32
    )
    probe_dict["calibrated_thresholds_15m_std"] = torch.tensor(
        thresholds_15m_std, dtype=torch.float32
    )
    probe_dict["calibrated_thresholds_15m_seeds"] = torch.tensor(
        thresholds_15m_seeds, dtype=torch.float32
    )
    probe_dict["calibrated_thresholds_6m"] = torch.tensor(
        thresholds_6m, dtype=torch.float32
    )
    probe_dict["calibrated_thresholds_6m_std"] = torch.tensor(
        thresholds_6m_std, dtype=torch.float32
    )
    probe_dict["calibrated_thresholds_6m_seeds"] = torch.tensor(
        thresholds_6m_seeds, dtype=torch.float32
    )
    probe_dict["calibration_seeds"] = calibration_seeds

    # Add heldout probes if constructed
    if heldout_general_seqs is not None:
        probe_dict["heldout_general_seqs"] = heldout_general_seqs
    if heldout_induction_seqs is not None:
        probe_dict["heldout_induction_seqs"] = heldout_induction_seqs
        probe_dict["heldout_induction_p1"] = heldout_induction_p1
        probe_dict["heldout_induction_p2"] = heldout_induction_p2
    if heldout_positional_seqs is not None:
        probe_dict["heldout_positional_seqs"] = heldout_positional_seqs
        probe_dict["heldout_positional_pairs"] = heldout_positional_pairs

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(probe_dict, output_path)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  Probe dataset saved to: {output_path}")
    print(f"  {'─' * 50}")
    print(f"  general_seqs      : {general_seqs.shape}")
    print(f"  induction_seqs    : {induction_seqs.shape}")
    print(f"  induction_p1      : {induction_p1.shape}")
    print(f"  induction_p2      : {induction_p2.shape}")
    print(f"  positional_seqs   : {positional_seqs.shape}")
    print(f"  positional_pairs  : {positional_pairs.shape}")
    if heldout_general_seqs is not None:
        print(f"  heldout_general   : {heldout_general_seqs.shape}")
    if heldout_induction_seqs is not None:
        print(f"  heldout_induction : {heldout_induction_seqs.shape}")
        print(f"  heldout_p1        : {heldout_induction_p1.shape}")
        print(f"  heldout_p2        : {heldout_induction_p2.shape}")
    if heldout_positional_seqs is not None:
        print(f"  heldout_positional: {heldout_positional_seqs.shape}")
        print(f"  heldout_pairs     : {heldout_positional_pairs.shape}")
    print(f"  {'─' * 50}")
    print(f"  Total time        : {elapsed:.1f}s")
    print(f"{'=' * 60}\n")

    return probe_dict


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_probe_dataset(path: Path) -> Dict[str, torch.Tensor]:
    """
    Load a previously constructed probe dataset from disk.
    Validates that all expected keys are present.

    Returns:
        probe_dict: dictionary of all probe tensors
    """

    required_keys = {
        "general_seqs", "induction_seqs", "induction_p1", "induction_p2",
        "positional_seqs", "positional_pairs", "creation_seed", "block_size",
    }

    if not path.exists():
        raise FileNotFoundError(
            f"Probe dataset not found at {path}. "
            f"Run build_probe_dataset() before training."
        )

    probe_dict: Dict[str, torch.Tensor] = torch.load(path, weights_only=True)

    missing = required_keys - set(probe_dict.keys())
    if missing:
        raise ValueError(
            f"Probe dataset at {path} is missing keys: {missing}. "
            f"Rebuild the probe dataset."
        )

    print(f"\n  Probe dataset loaded from: {path}")
    print(f"  creation_seed : {probe_dict['creation_seed'].item()}")
    print(f"  block_size    : {probe_dict['block_size'].item()}")
    print(f"  general_seqs  : {probe_dict['general_seqs'].shape}")
    print(f"  induction_seqs: {probe_dict['induction_seqs'].shape}")
    print(f"  positional_seqs:{probe_dict['positional_seqs'].shape}\n")

    return probe_dict


# ─────────────────────────────────────────────────────────────────────────────
# Validation helper
# ─────────────────────────────────────────────────────────────────────────────

def verify_induction_probes(probe_dict: Dict[str, torch.Tensor]) -> None:
    """
    Sanity check: verify that induction probes contain the expected
    repeated subsequences at the stored p1 and p2 positions.
    Prints a report — call this after building or loading the probe dataset.
    """

    seqs = probe_dict["induction_seqs"]   # (100, 256)
    p1s  = probe_dict["induction_p1"]     # (100,)
    p2s  = probe_dict["induction_p2"]     # (100,)

    n_correct = 0
    n_total   = seqs.shape[0]
    subseq_len_min = 5

    for i in range(n_total):
        p1 = p1s[i].item()
        p2 = p2s[i].item()
        seq = seqs[i]

        # Check that the subsequences at p1 and p2 match for at least 5 tokens
        match_len = 0
        for offset in range(subseq_len_min):
            if p1 + offset < 256 and p2 + offset < 256:
                if seq[p1 + offset] == seq[p2 + offset]:
                    match_len += 1
                else:
                    break

        if match_len >= subseq_len_min:
            n_correct += 1

    print(f"\n  Induction probe verification:")
    print(f"  {'─' * 40}")
    print(f"  Probes with valid repeats : {n_correct:3d} / {n_total}")
    print(f"  Success rate              : {n_correct / n_total * 100:.1f}%")
    if n_correct < n_total * 0.95:
        print(f"  [WARNING] Less than 95% probes verified — consider rebuilding.")
    else:
        print(f"  [OK] Probe dataset integrity confirmed.")
    print()
