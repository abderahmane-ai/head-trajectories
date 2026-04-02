"""
probing/pipeline.py — Full probing pipeline over all checkpoints for one run.

Orchestrates the entire measurement phase:
  1. Discovers all checkpoint files in a run's checkpoint directory
  2. For each checkpoint (in chronological order):
     a. Loads the model weights
     b. Extracts attention maps for all probe types
     c. Computes all 5 scores for every (layer, head) pair
     d. Classifies each head and records the result
  3. Saves the complete label_tensor, score_tensor, mixed-behavior metadata,
     and optional natural-induction auxiliary scores

This is the most compute-intensive phase after training itself.
Expected runtime: ~2–3 minutes per checkpoint × 100 checkpoints ≈ 3–5 hours.
Run on Modal CPU (no GPU needed for inference at this batch size).
"""

import time
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from .extractor import extract_checkpoint, CheckpointExtraction
from .scores import natural_induction_score, score_head
from .classifier import HeadClassifier, HEAD_TYPES, THRESHOLDS


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint discovery
# ─────────────────────────────────────────────────────────────────────────────

def discover_checkpoints(ckpt_dir: Path) -> List[Path]:
    """
    Return all checkpoint files in ckpt_dir, sorted by step number ascending.

    Args:
        ckpt_dir: directory containing ckpt_XXXXXXX.pt files

    Returns:
        sorted list of checkpoint Paths
    """

    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    ckpt_files = sorted(ckpt_dir.glob("ckpt_*.pt"))

    if not ckpt_files:
        raise RuntimeError(f"No checkpoint files found in {ckpt_dir}")

    return ckpt_files


def parse_step_from_path(path: Path) -> int:
    """Extract the step number from a checkpoint filename like ckpt_0001000.pt."""
    return int(path.stem.replace("ckpt_", ""))


def _load_partial_results(
    output_path: Path,
    n_layers: int,
    n_heads: int,
    seed: int,
) -> Optional[Dict]:
    """
    Load and validate partial probing results for resume mode.

    Returns:
        The loaded results dict if metadata matches, otherwise None.
    """

    if not output_path.exists():
        return None

    try:
        partial = torch.load(output_path, weights_only=True)
    except Exception as e:
        print(f"  [Resume] Failed to load partial results: {e} — starting fresh\n")
        return None

    if (
        partial.get("n_layers") == n_layers
        and partial.get("n_heads") == n_heads
        and partial.get("seed") == seed
    ):
        return partial

    print("  [Resume] Partial results metadata mismatch — starting fresh\n")
    return None


def _save_results_atomically(classifier: HeadClassifier, output_path: Path) -> None:
    """
    Save probing results via temp file + replace and clean up on failure.
    """

    temp_path = output_path.with_suffix(".tmp.pt")
    try:
        classifier.save(temp_path)
        temp_path.replace(output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Per-checkpoint scoring
# ─────────────────────────────────────────────────────────────────────────────

def score_all_heads(
    extraction:  CheckpointExtraction,
    probe_dict:  Dict[str, torch.Tensor],
    classifier:  HeadClassifier,
    ckpt_idx:    int,
    step:        int,
) -> None:
    """
    Compute all 5 scores for every (layer, head) pair at one checkpoint
    and record them in the classifier.

    Args:
        extraction: CheckpointExtraction from extractor.py
        probe_dict: loaded probe dataset (for metadata: p1, p2, pairs)
        classifier: HeadClassifier to record scores and labels
        ckpt_idx:   checkpoint index (0-indexed, for tensor placement)
        step:       training step (for tie logging)
    """

    n_layers = extraction.config.n_layers
    n_heads  = extraction.config.n_heads

    induction_p1 = probe_dict["induction_p1"]          # (100,)
    induction_p2 = probe_dict["induction_p2"]          # (100,)
    natural_induction_p1 = probe_dict.get("natural_induction_p1")
    natural_induction_p2 = probe_dict.get("natural_induction_p2")
    positional_pairs = probe_dict["positional_pairs"]  # (50, 2)
    token_ids = extraction.general_token_ids           # (500, 256)
    embedding_matrix = extraction.embedding_matrix     # (V, D)

    for layer in range(n_layers):
        # Full layer attention maps — shape (N, n_heads, T, T)
        general_layer = extraction.general_maps[layer]    # (500, n_heads, T, T)
        induction_layer = extraction.induction_maps[layer]  # (100, n_heads, T, T)
        positional_layer = extraction.positional_maps[layer] # (100, n_heads, T, T)
        natural_induction_layer = (
            extraction.natural_induction_maps[layer]
            if extraction.natural_induction_maps is not None
            else None
        )

        for head in range(n_heads):
            # Slice to single head: (N, T, T)
            general_head = general_layer[:, head, :, :]    # (500, T, T)
            induction_head = induction_layer[:, head, :, :]  # (100, T, T)
            positional_head = positional_layer[:, head, :, :] # (100, T, T)
            natural_induction_head = (
                natural_induction_layer[:, head, :, :]
                if natural_induction_layer is not None
                else None
            )

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

            natural_score = None
            if (
                natural_induction_head is not None
                and natural_induction_p1 is not None
                and natural_induction_p2 is not None
            ):
                natural_score = natural_induction_score(
                    natural_induction_head,
                    natural_induction_p1,
                    natural_induction_p2,
                )

            classifier.record(
                ckpt_idx=ckpt_idx,
                step=step,
                layer=layer,
                head=head,
                scores=scores,
                natural_induction_score=natural_score,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_probing_pipeline(
    ckpt_dir:       Path,
    probe_path:     Path,
    output_path:    Path,
    ties_log_path:  Path,
    seed:           int,
    device:         torch.device,
    batch_size:     int = 16,
    resume:         bool = True,
    use_heldout:    bool = False,
) -> Dict:
    """
    Run the full probing pipeline for one training run.

    Args:
        ckpt_dir:      directory containing checkpoint .pt files
        probe_path:    path to probe_dataset.pt
        output_path:   path to save results .pt file
        ties_log_path: path to append tie events (ties.csv)
        seed:          training seed of this run (for logging/metadata)
        device:        inference device (CPU is fine for this phase)
        batch_size:    inference batch size for attention extraction
        resume:        if True and output_path exists, skip already-processed
                       checkpoints and continue from where we left off
        use_heldout:   if True, use heldout probe sequences for scoring

    Returns:
        results dict (same as saved to output_path)
    """

    t_pipeline_start = time.time()

    # ── Load probe dataset ───────────────────────────────────────────────────
    print(f"\n{'=' * 64}")
    print(f"  Probing Pipeline — Seed {seed}")
    print(f"{'=' * 64}")
    print(f"  Checkpoint dir : {ckpt_dir}")
    print(f"  Probe path     : {probe_path}")
    print(f"  Output path    : {output_path}")
    print(f"  Device         : {device}")
    print(f"  Probe set      : {'heldout' if use_heldout else 'primary'}")
    print(f"  Resume mode    : {resume}")
    print()

    # Import here to avoid circular dependency
    from data import load_probe_dataset
    probe_dict = load_probe_dataset(probe_path)
    if use_heldout:
        required = {
            "heldout_general_seqs",
            "heldout_induction_seqs",
            "heldout_induction_p1",
            "heldout_induction_p2",
            "heldout_natural_induction_seqs",
            "heldout_natural_induction_p1",
            "heldout_natural_induction_p2",
            "heldout_positional_seqs",
            "heldout_positional_pairs",
        }
        missing = required - set(probe_dict.keys())
        if missing:
            raise ValueError(
                f"Heldout probe keys missing from {probe_path}: {missing}"
            )
        probe_view = dict(probe_dict)
        probe_view["general_seqs"] = probe_dict["heldout_general_seqs"]
        probe_view["induction_seqs"] = probe_dict["heldout_induction_seqs"]
        probe_view["induction_p1"] = probe_dict["heldout_induction_p1"]
        probe_view["induction_p2"] = probe_dict["heldout_induction_p2"]
        probe_view["natural_induction_seqs"] = probe_dict["heldout_natural_induction_seqs"]
        probe_view["natural_induction_p1"] = probe_dict["heldout_natural_induction_p1"]
        probe_view["natural_induction_p2"] = probe_dict["heldout_natural_induction_p2"]
        probe_view["positional_seqs"] = probe_dict["heldout_positional_seqs"]
        probe_view["positional_pairs"] = probe_dict["heldout_positional_pairs"]
        probe_dict = probe_view

    # ── Discover checkpoints ─────────────────────────────────────────────────
    ckpt_files = discover_checkpoints(ckpt_dir)
    n_ckpts    = len(ckpt_files)
    print(f"  Found {n_ckpts} checkpoints to process.\n")

    # ── Load one checkpoint to get model dimensions ──────────────────────────
    from .extractor import load_model_from_checkpoint
    first_model, _, _, _ = load_model_from_checkpoint(ckpt_files[0], device)
    n_layers = first_model.config.n_layers
    n_heads  = first_model.config.n_heads

    thresholds_key = (
        "calibrated_thresholds_6m"
        if first_model.config.ablation_mode
        else "calibrated_thresholds_15m"
    )
    thresholds = probe_dict.get(thresholds_key)
    if thresholds is None:
        print(
            f"  [WARNING] No calibrated thresholds found under "
            f"{thresholds_key}; falling back to default thresholds."
        )
        thresholds = THRESHOLDS
    else:
        print(f"  Using calibrated thresholds: {thresholds_key}")
    del first_model

    # ── Check for partial results and resume if requested ────────────────────
    completed_steps = set()
    partial = None
    
    if resume and output_path.exists():
        partial = _load_partial_results(output_path, n_layers, n_heads, seed)
        if partial is not None:
            completed_steps = set(partial.get("step_index", []))
            if completed_steps:
                print(f"  [Resume] Found partial results with {len(completed_steps)} completed checkpoints")
                print(f"  [Resume] Will skip already-processed checkpoints\n")

    # ── Initialize classifier ─────────────────────────────────────────────────
    classifier = HeadClassifier(
        n_checkpoints=n_ckpts,
        n_layers=n_layers,
        n_heads=n_heads,
        seed=seed,
        ties_log_path=ties_log_path,
        thresholds=thresholds,
    )
    
    # Restore partial results if resuming
    if partial is not None and completed_steps:
        classifier.label_tensor = partial["label_tensor"]
        classifier.score_tensor = partial["score_tensor"]
        if "threshold_flag_tensor" in partial:
            classifier.threshold_flag_tensor = partial["threshold_flag_tensor"]
            classifier.normalized_score_tensor = partial["normalized_score_tensor"]
            classifier.primary_behavior_tensor = partial["primary_behavior_tensor"]
            classifier.runner_up_tensor = partial["runner_up_tensor"]
            classifier.dominant_margin_tensor = partial["dominant_margin_tensor"]
            classifier.behavior_count_tensor = partial["behavior_count_tensor"]
        classifier.natural_induction_score_tensor = partial.get(
            "natural_induction_score_tensor"
        )
        classifier.step_index = partial["step_index"]

    # ── Process each checkpoint ──────────────────────────────────────────────
    print(
        f"  {'Ckpt':>5}  {'Step':>8}  {'Train Loss':>10}  "
        f"{'Val Loss':>10}  {'Time':>8}  Progress"
    )
    print(f"  {'─' * 60}")

    for ckpt_idx, ckpt_path in enumerate(ckpt_files):
        t_ckpt_start = time.time()
        step = parse_step_from_path(ckpt_path)
        
        # Skip if already processed
        if step in completed_steps:
            print(f"  [Resume] Skipping checkpoint {ckpt_idx + 1}/{n_ckpts} (step {step:,}) — already processed")
            continue

        # Extract all attention maps for this checkpoint
        try:
            extraction = extract_checkpoint(
                ckpt_path=ckpt_path,
                probe_dict=probe_dict,
                device=device,
                batch_size=batch_size,
            )
        except Exception as e:
            classifier.flush_ties()
            _save_results_atomically(classifier, output_path)
            raise RuntimeError(
                f"Failed to process checkpoint {ckpt_path.name} at step {step}. "
                "Partial probing results were saved; fix the checkpoint or remove it "
                "before resuming."
            ) from e

        # Score all heads and record
        classifier.register_step(step)
        score_all_heads(
            extraction=extraction,
            probe_dict=probe_dict,
            classifier=classifier,
            ckpt_idx=ckpt_idx,
            step=step,
        )

        # Flush tie log and save incrementally every 10 checkpoints
        if (ckpt_idx + 1) % 10 == 0:
            classifier.flush_ties()
            _save_results_atomically(classifier, output_path)
            print(f"  [Checkpoint] Incremental save at {ckpt_idx + 1}/{n_ckpts}")
        elif ckpt_idx % 10 == 0:
            classifier.flush_ties()

        elapsed_ckpt = time.time() - t_ckpt_start
        elapsed_total = time.time() - t_pipeline_start
        eta = (elapsed_total / (ckpt_idx + 1)) * (n_ckpts - ckpt_idx - 1)

        # Count head types for this checkpoint
        labels_this_ckpt = classifier.label_tensor[ckpt_idx]
        type_counts = {
            HEAD_TYPES[t]: int((labels_this_ckpt == t).sum())
            for t in range(len(HEAD_TYPES))
        }
        dominant = max(type_counts, key=lambda k: type_counts[k] if k != "UNDIFFERENTIATED" else -1)

        print(
            f"  {ckpt_idx + 1:>5}  {step:>8,}  "
            f"{extraction.train_loss:>10.4f}  {extraction.val_loss:>10.4f}  "
            f"{elapsed_ckpt:>7.1f}s  "
            f"ETA {eta / 60:.1f}min | dominant: {dominant}"
        )

    # ── Final flush and save ─────────────────────────────────────────────────
    classifier.flush_ties()
    _save_results_atomically(classifier, output_path)

    total_elapsed = time.time() - t_pipeline_start

    print(f"\n{'─' * 64}")
    print(f"  Probing pipeline complete.")
    print(f"  {'─' * 50}")
    print(f"  {'Checkpoints processed':<28}: {n_ckpts}")
    print(f"  {'Total heads scored':<28}: {n_ckpts * n_layers * n_heads:,}")
    print(f"  {'Ties logged':<28}: see {ties_log_path.name}")
    print(f"  {'Total time':<28}: {total_elapsed / 3600:.2f}h")
    print(f"  {'Results saved to':<28}: {output_path}")

    # Final type distribution at last checkpoint
    final_labels = classifier.label_tensor[-1]
    print(f"\n  Final checkpoint head type distribution:")
    for t in range(len(HEAD_TYPES)):
        count = int((final_labels == t).sum())
        bar   = "█" * count
        print(f"    {HEAD_TYPES[t]:<20}: {count:>3}  {bar}")

    print(f"{'=' * 64}\n")

    return HeadClassifier.load(output_path)
