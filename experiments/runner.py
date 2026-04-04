"""Reusable helpers for single-run notebook or script execution."""

from __future__ import annotations

import json
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from analysis import (
    compute_global_curves,
    compute_head_trajectories,
    compute_induction_count_curve,
    compute_mixed_behavior_summary,
    compute_onset_bootstrap_cis,
    compute_per_layer_curves,
    compute_sink_persistence,
    compute_specialization_onset,
    compute_stability_histogram,
    compute_type_change_matrix,
    compute_per_type_stability,
    compute_discontinuity_score,
    detect_val_loss_inflection,
    extract_val_loss_curve,
    find_crossing_steps,
    find_interesting_trajectories,
    load_run_results,
    run_threshold_sensitivity,
)
from data import (
    OpenWebTextStream,
    build_probe_dataset,
    verify_induction_probes,
)
from data.calibration import CALIBRATION_VERSION, calibrate_thresholds
from data.loader import get_tokenizer, tokenize_text
from data.probe import (
    build_general_probes,
    build_induction_probes,
    build_natural_induction_probes,
    build_positional_probes,
)
from experiments.profiles import ExperimentProfile, get_profile
from model import ModelConfig, TransformerLM
from probing.pipeline import run_probing_pipeline
from training import Trainer, save_checkpoint
from training.scheduler import CosineScheduler
from visualization import (
    plot_discontinuity_comparison,
    plot_dominant_type_heatmap,
    plot_individual_trajectories,
    plot_phase_transition,
    plot_specialization_fraction_heatmap,
    plot_stability_figure,
    plot_timeline,
)


@dataclass(frozen=True)
class RunArtifacts:
    """Filesystem layout for one profile and seed."""

    profile_name: str
    seed: int
    root: Path

    @property
    def profile_dir(self) -> Path:
        return self.root / self.profile_name

    @property
    def probe_dir(self) -> Path:
        return self.profile_dir / "probe"

    @property
    def probe_path(self) -> Path:
        return self.probe_dir / "probe_dataset.pt"

    @property
    def seed_dir(self) -> Path:
        return self.profile_dir / f"seed{self.seed}"

    @property
    def ckpt_dir(self) -> Path:
        return self.seed_dir / "checkpoints"

    @property
    def best_ckpt_path(self) -> Path:
        return self.seed_dir / "checkpoints" / "best_checkpoint.pt"

    @property
    def train_history_path(self) -> Path:
        return self.seed_dir / "training" / "train_history.pt"

    @property
    def train_summary_path(self) -> Path:
        return self.seed_dir / "training" / "train_summary.json"

    @property
    def results_dir(self) -> Path:
        return self.seed_dir / "results"

    @property
    def results_path(self) -> Path:
        return self.results_dir / f"results_seed{self.seed}.pt"

    @property
    def ties_path(self) -> Path:
        return self.results_dir / f"ties_seed{self.seed}.csv"

    @property
    def figures_dir(self) -> Path:
        return self.seed_dir / "figures"

    @property
    def figure_path(self) -> Path:
        return self.figures_dir / f"timeline_seed{self.seed}.png"

    @property
    def summary_path(self) -> Path:
        return self.seed_dir / "analysis" / "summary_seed.json"

    @property
    def manifest_path(self) -> Path:
        return self.seed_dir / "manifest.json"


@dataclass(frozen=True)
class BatchRunSpec:
    """One profile plus one or more seeds to run sequentially."""

    profile_name: str
    seeds: Tuple[int, ...]

    @property
    def profile(self) -> ExperimentProfile:
        return get_profile(self.profile_name)


def resolve_artifacts(
    profile: ExperimentProfile | str,
    seed: int,
    artifact_root: Path | str = Path("artifacts"),
) -> RunArtifacts:
    """Build canonical artifact paths for one run."""

    profile_obj = get_profile(profile) if isinstance(profile, str) else profile
    return RunArtifacts(
        profile_name=profile_obj.name,
        seed=seed,
        root=Path(artifact_root),
    )


def normalize_run_specs(
    run_specs: Sequence[Dict[str, object] | BatchRunSpec],
) -> List[BatchRunSpec]:
    """Normalize user-provided run specs into a deterministic list."""

    normalized: List[BatchRunSpec] = []
    for spec in run_specs:
        if isinstance(spec, BatchRunSpec):
            normalized.append(spec)
            continue

        if "profile" in spec:
            profile_name = str(spec["profile"])
        elif "profile_name" in spec:
            profile_name = str(spec["profile_name"])
        else:
            raise ValueError(f"Run spec is missing 'profile'/'profile_name': {spec}")

        seeds_value = spec.get("seeds")
        if seeds_value is None:
            if "seed" not in spec:
                raise ValueError(f"Run spec is missing 'seed' or 'seeds': {spec}")
            seeds = (int(spec["seed"]),)
        else:
            seeds = tuple(int(seed) for seed in seeds_value)
            if not seeds:
                raise ValueError(f"Run spec has an empty 'seeds' list: {spec}")

        normalized.append(BatchRunSpec(profile_name=profile_name, seeds=seeds))

    return normalized


def set_seed(seed: int) -> None:
    """Set all relevant random seeds."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve a torch device from a compact string."""

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def ensure_dirs(paths: RunArtifacts) -> None:
    """Create all artifact parents eagerly."""

    for path in (
        paths.profile_dir,
        paths.probe_dir,
        paths.ckpt_dir,
        paths.results_dir,
        paths.figures_dir,
        paths.summary_path.parent,
        paths.train_summary_path.parent,
    ):
        path.mkdir(parents=True, exist_ok=True)


def reset_run_artifacts(paths: RunArtifacts, reset_probe: bool = False) -> None:
    """Delete per-seed artifacts and optionally the shared probe dataset."""

    if paths.seed_dir.exists():
        shutil.rmtree(paths.seed_dir)
    if reset_probe and paths.probe_dir.exists():
        shutil.rmtree(paths.probe_dir)


def _threshold_prefix_for_config(config: ModelConfig) -> str:
    return "6m" if config.ablation_mode else "15m"


def _write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_optional_natural_induction_probes(
    raw_sequences: Sequence[List[int]],
    n_probes: int,
    block_size: int,
    seed: int,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Best-effort wrapper for the auxiliary natural-induction probe family."""

    min_probes = max(4, n_probes // 2) if n_probes > 0 else 0
    try:
        return build_natural_induction_probes(
            list(raw_sequences),
            n_probes=n_probes,
            block_size=block_size,
            seed=seed,
            allow_partial=True,
            min_probes=min_probes,
        )
    except RuntimeError as exc:
        print(
            "  [WARNING] Skipping auxiliary natural induction probes: "
            f"{exc}"
        )
        return None


def _encode_split_texts(texts: Sequence[str], block_size: int) -> torch.Tensor:
    enc = get_tokenizer()
    flat_tokens: List[int] = []
    for text in texts:
        if text and text.strip():
            flat_tokens.extend(tokenize_text(text, enc))

    usable = (len(flat_tokens) // block_size) * block_size
    if usable <= block_size:
        raise ValueError("Tokenized split is too small for the requested block size.")

    return torch.tensor(flat_tokens[:usable], dtype=torch.long)


def _get_hf_split_texts(
    dataset,
    split_name: str,
    text_column: str,
) -> Sequence[str]:
    """Return the requested text-like column from one Hugging Face split."""

    split = dataset[split_name]
    if text_column not in split.column_names:
        available = ", ".join(split.column_names)
        raise KeyError(
            f"Dataset split '{split_name}' does not contain column '{text_column}'. "
            f"Available columns: {available}"
        )
    return split[text_column]


def _sample_batch(
    tokens: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    max_start = len(tokens) - block_size - 1
    if max_start <= 0:
        raise ValueError("Token tensor is too short for the requested block size.")

    starts = torch.randint(0, max_start, (batch_size,))
    xs = torch.stack([tokens[s : s + block_size] for s in starts]).to(device)
    ys = torch.stack([tokens[s + 1 : s + block_size + 1] for s in starts]).to(device)
    return xs, ys


@torch.no_grad()
def _estimate_tensor_val_loss(
    model: TransformerLM,
    tokens: torch.Tensor,
    batch_size: int,
    block_size: int,
    n_batches: int,
    device: torch.device,
) -> float:
    model.eval()
    losses: List[float] = []
    for _ in range(n_batches):
        x, y = _sample_batch(tokens, batch_size, block_size, device)
        logits, _ = model(x, return_attention=False)
        bsz, seq_len, vocab = logits.shape
        loss = nn.functional.cross_entropy(logits.view(bsz * seq_len, vocab), y.reshape(bsz * seq_len))
        losses.append(float(loss.item()))
    model.train()
    return float(np.mean(losses))


def _build_hf_probe_dataset(
    profile: ExperimentProfile,
    output_path: Path,
    seed: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    from datasets import load_dataset

    print(
        f"[Probe] Building shared probe dataset for profile={profile.name} "
        f"(dataset={profile.dataset_name}/{profile.dataset_config})"
    )
    dataset = load_dataset(profile.dataset_name, profile.dataset_config)
    test_tokens = _encode_split_texts(
        _get_hf_split_texts(dataset, profile.probe_split, profile.text_column),
        profile.block_size,
    )
    raw_sequences = [row.tolist() for row in test_tokens.view(-1, profile.block_size)]

    rng = random.Random(seed)
    rng.shuffle(raw_sequences)

    n_general = profile.n_general * 2
    n_induction = profile.n_induction * 3
    n_positional = profile.n_pairs * 4
    n_general_holdout = max(profile.n_general_holdout * 2, 0)
    n_induction_holdout = max(profile.n_induction_holdout * 3, 0)
    n_positional_holdout = max(profile.n_pairs_holdout * 4, 0)

    fixed_needed = (
        n_general
        + n_induction
        + n_positional
        + n_general_holdout
        + n_induction_holdout
        + n_positional_holdout
    )
    min_natural_induction = profile.n_induction if profile.enable_natural_induction else 0
    min_natural_induction_holdout = (
        profile.n_induction_holdout if profile.enable_natural_induction else 0
    )
    minimum_needed = (
        fixed_needed + min_natural_induction + min_natural_induction_holdout
    )
    if len(raw_sequences) < minimum_needed:
        raise RuntimeError(
            f"Profile {profile.name} needs at least {minimum_needed} probe source sequences, "
            f"but only {len(raw_sequences)} are available."
        )

    desired_natural_induction = (
        max(profile.n_induction * 8, profile.n_induction)
        if profile.enable_natural_induction
        else 0
    )
    desired_natural_induction_holdout = (
        max(profile.n_induction_holdout * 8, profile.n_induction_holdout)
        if profile.enable_natural_induction
        else 0
    )
    desired_train_extra = desired_natural_induction - min_natural_induction
    desired_holdout_extra = (
        desired_natural_induction_holdout - min_natural_induction_holdout
    )
    remaining_extra = len(raw_sequences) - minimum_needed
    total_desired_extra = desired_train_extra + desired_holdout_extra
    train_extra = 0
    holdout_extra = 0
    if total_desired_extra > 0 and remaining_extra > 0:
        train_extra = min(
            desired_train_extra,
            remaining_extra * desired_train_extra // total_desired_extra,
        )
        holdout_extra = min(
            desired_holdout_extra,
            remaining_extra - train_extra,
        )
        leftover = remaining_extra - train_extra - holdout_extra
        if leftover > 0:
            add_train = min(desired_train_extra - train_extra, leftover)
            train_extra += add_train
            leftover -= add_train
        if leftover > 0:
            add_holdout = min(desired_holdout_extra - holdout_extra, leftover)
            holdout_extra += add_holdout

    n_natural_induction = min_natural_induction + train_extra
    n_natural_induction_holdout = min_natural_induction_holdout + holdout_extra
    if profile.enable_natural_induction and (
        n_natural_induction < desired_natural_induction
        or n_natural_induction_holdout < desired_natural_induction_holdout
    ):
        print(
            "[Probe] Limited held-out split detected; reducing natural induction "
            f"candidate pools to train={n_natural_induction}, "
            f"holdout={n_natural_induction_holdout}."
        )

    offset = 0
    pool_general = raw_sequences[offset : offset + n_general]
    offset += n_general
    pool_induction = raw_sequences[offset : offset + n_induction]
    offset += n_induction
    pool_natural_induction = raw_sequences[offset : offset + n_natural_induction]
    offset += n_natural_induction
    pool_positional = raw_sequences[offset : offset + n_positional]
    offset += n_positional
    pool_general_holdout = raw_sequences[offset : offset + n_general_holdout]
    offset += n_general_holdout
    pool_induction_holdout = raw_sequences[offset : offset + n_induction_holdout]
    offset += n_induction_holdout
    pool_natural_induction_holdout = raw_sequences[
        offset : offset + n_natural_induction_holdout
    ]
    offset += n_natural_induction_holdout
    pool_positional_holdout = raw_sequences[offset : offset + n_positional_holdout]

    probe_dict: Dict[str, torch.Tensor] = {
        "general_seqs": build_general_probes(pool_general, profile.n_general, profile.block_size, seed),
        "calibration_version": torch.tensor(CALIBRATION_VERSION, dtype=torch.long),
        "creation_seed": torch.tensor(seed, dtype=torch.long),
        "block_size": torch.tensor(profile.block_size, dtype=torch.long),
    }
    induction_seqs, induction_p1, induction_p2 = build_induction_probes(
        pool_induction,
        n_probes=profile.n_induction,
        block_size=profile.block_size,
        seed=seed,
    )
    natural_built = None
    if profile.enable_natural_induction:
        natural_built = _build_optional_natural_induction_probes(
            pool_natural_induction,
            n_probes=profile.n_induction,
            block_size=profile.block_size,
            seed=seed,
        )
    positional_seqs, positional_pairs = build_positional_probes(
        pool_positional,
        n_pairs=profile.n_pairs,
        block_size=profile.block_size,
        seed=seed,
    )
    probe_dict["induction_seqs"] = induction_seqs
    probe_dict["induction_p1"] = induction_p1
    probe_dict["induction_p2"] = induction_p2
    if natural_built is not None:
        natural_induction_seqs, natural_induction_p1, natural_induction_p2 = natural_built
        probe_dict["natural_induction_seqs"] = natural_induction_seqs
        probe_dict["natural_induction_p1"] = natural_induction_p1
        probe_dict["natural_induction_p2"] = natural_induction_p2
    probe_dict["positional_seqs"] = positional_seqs
    probe_dict["positional_pairs"] = positional_pairs

    if profile.n_general_holdout > 0:
        probe_dict["heldout_general_seqs"] = build_general_probes(
            pool_general_holdout,
            profile.n_general_holdout,
            profile.block_size,
            seed + 5000,
        )
    if profile.n_induction_holdout > 0:
        held_seqs, held_p1, held_p2 = build_induction_probes(
            pool_induction_holdout,
            n_probes=profile.n_induction_holdout,
            block_size=profile.block_size,
            seed=seed + 5000,
        )
        probe_dict["heldout_induction_seqs"] = held_seqs
        probe_dict["heldout_induction_p1"] = held_p1
        probe_dict["heldout_induction_p2"] = held_p2
        if profile.enable_natural_induction:
            held_natural_built = _build_optional_natural_induction_probes(
                pool_natural_induction_holdout,
                n_probes=profile.n_induction_holdout,
                block_size=profile.block_size,
                seed=seed + 5000,
            )
            if held_natural_built is not None:
                nat_held_seqs, nat_held_p1, nat_held_p2 = held_natural_built
                probe_dict["heldout_natural_induction_seqs"] = nat_held_seqs
                probe_dict["heldout_natural_induction_p1"] = nat_held_p1
                probe_dict["heldout_natural_induction_p2"] = nat_held_p2
    if profile.n_pairs_holdout > 0:
        held_pos_seqs, held_pos_pairs = build_positional_probes(
            pool_positional_holdout,
            n_pairs=profile.n_pairs_holdout,
            block_size=profile.block_size,
            seed=seed + 5000,
        )
        probe_dict["heldout_positional_seqs"] = held_pos_seqs
        probe_dict["heldout_positional_pairs"] = held_pos_pairs

    print(
        "[Probe] Calibrating thresholds from random baseline "
        f"({profile.n_calibration_seeds} seeds, device={device})..."
    )
    thresholds_mean, thresholds_std, thresholds_per_seed, diag = calibrate_thresholds(
        probe_dict=probe_dict,
        config=profile.model_config,
        device=device,
        n_seeds=profile.n_calibration_seeds,
        return_diagnostics=True,
    )
    prefix = _threshold_prefix_for_config(profile.model_config)
    probe_dict[f"calibrated_thresholds_{prefix}"] = torch.tensor(thresholds_mean, dtype=torch.float32)
    probe_dict[f"calibrated_thresholds_{prefix}_std"] = torch.tensor(thresholds_std, dtype=torch.float32)
    probe_dict[f"calibrated_thresholds_{prefix}_seeds"] = torch.tensor(thresholds_per_seed, dtype=torch.float32)
    probe_dict[f"calibrated_thresholds_{prefix}_metric_means"] = torch.tensor(
        diag["per_seed_metric_means"], dtype=torch.float32
    )
    probe_dict[f"calibrated_thresholds_{prefix}_metric_stds"] = torch.tensor(
        diag["per_seed_metric_stds"], dtype=torch.float32
    )
    probe_dict[f"calibrated_thresholds_{prefix}_metric_p95"] = torch.tensor(
        diag["per_seed_metric_p95"], dtype=torch.float32
    )
    probe_dict[f"calibrated_thresholds_{prefix}_metric_p99"] = torch.tensor(
        diag["per_seed_metric_p99"], dtype=torch.float32
    )
    probe_dict[f"calibrated_thresholds_{prefix}_nonpositive_mask"] = torch.tensor(
        diag["per_seed_nonpositive_mask"], dtype=torch.bool
    )
    probe_dict[f"calibrated_thresholds_{prefix}_requires_sanitization"] = torch.tensor(
        diag["requires_sanitization"], dtype=torch.bool
    )
    probe_dict[f"calibrated_thresholds_{prefix}_null_scores"] = torch.tensor(
        diag["per_seed_null_scores"], dtype=torch.float32
    )
    probe_dict[f"calibrated_thresholds_{prefix}_null_scores_pooled"] = torch.tensor(
        diag["pooled_null_scores"], dtype=torch.float32
    )
    probe_dict[f"calibrated_thresholds_{prefix}_null_seed_list"] = torch.tensor(
        diag["null_seed_list"], dtype=torch.long
    )
    probe_dict["calibration_seeds"] = torch.tensor(
        [seed + i for i in range(profile.n_calibration_seeds)], dtype=torch.long
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(probe_dict, output_path)
    verify_induction_probes(probe_dict)
    print(f"[Probe] Saved probe dataset to: {output_path}")
    print(f"[Probe] Mean thresholds: {thresholds_mean.tolist()}")
    print(f"[Probe] Threshold stds : {thresholds_std.tolist()}")
    print(f"[Probe] Threshold rules: {diag['threshold_rules']}")
    print(f"[Probe] Requires sanitization: {diag['requires_sanitization']}")
    return probe_dict


def build_or_load_probe_dataset(
    profile: ExperimentProfile | str,
    paths: RunArtifacts,
    device: str | torch.device = "cpu",
    rebuild: bool = False,
) -> Dict[str, torch.Tensor]:
    """Build the shared probe dataset for a profile or reuse the existing one."""

    profile_obj = get_profile(profile) if isinstance(profile, str) else profile
    ensure_dirs(paths)

    if rebuild and paths.probe_path.exists():
        paths.probe_path.unlink()

    if paths.probe_path.exists():
        probe_dict = torch.load(paths.probe_path, weights_only=True)
        calibration_version = int(
            probe_dict.get("calibration_version", torch.tensor(0, dtype=torch.long)).item()
        )
        if calibration_version != CALIBRATION_VERSION:
            print(
                "[Probe] Existing probe dataset is stale "
                f"(calibration_version={calibration_version}, expected={CALIBRATION_VERSION}). "
                "Rebuilding with current calibration rules..."
            )
            paths.probe_path.unlink()
        else:
            verify_induction_probes(probe_dict)
            return probe_dict

    device_obj = resolve_device(device) if isinstance(device, str) else device
    if profile_obj.dataset_family == "openwebtext_stream":
        return build_probe_dataset(
            output_path=paths.probe_path,
            block_size=profile_obj.block_size,
            seed=0,
            n_general=profile_obj.n_general,
            n_induction=profile_obj.n_induction,
        n_pairs=profile_obj.n_pairs,
        n_general_holdout=profile_obj.n_general_holdout,
        n_induction_holdout=profile_obj.n_induction_holdout,
        n_pairs_holdout=profile_obj.n_pairs_holdout,
        n_calibration_seeds=profile_obj.n_calibration_seeds,
        enable_natural_induction=profile_obj.enable_natural_induction,
    )

    return _build_hf_probe_dataset(
        profile_obj,
        output_path=paths.probe_path,
        seed=0,
        device=device_obj,
    )


def _finalize_best_checkpoint(paths: RunArtifacts) -> Dict[str, Optional[float]]:
    ckpt_files = sorted(paths.ckpt_dir.glob("ckpt_*.pt"))
    best_path: Optional[Path] = None
    best_val: Optional[float] = None
    best_step: Optional[int] = None

    for ckpt_path in ckpt_files:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        val_loss = float(checkpoint.get("val_loss", float("inf")))
        if best_val is None or val_loss < best_val:
            best_val = val_loss
            best_step = int(checkpoint["step"])
            best_path = ckpt_path

    if best_path is not None:
        shutil.copy2(best_path, paths.best_ckpt_path)

    payload = {
        "best_checkpoint": str(best_path) if best_path is not None else None,
        "best_step": best_step,
        "best_val_loss": best_val,
        "checkpoint_count": len(ckpt_files),
    }
    _write_json(paths.train_summary_path, payload)
    return payload


def _train_openwebtext_profile(
    profile: ExperimentProfile,
    paths: RunArtifacts,
    seed: int,
    device: torch.device,
) -> Dict[str, object]:
    set_seed(seed)
    trainer = Trainer(
        config=profile.model_config,
        ckpt_dir=paths.ckpt_dir,
        seed=seed,
        total_steps=profile.total_steps,
        batch_size=profile.batch_size,
        block_size=profile.block_size,
        max_lr=profile.max_lr,
        min_lr=profile.min_lr,
        warmup_steps=profile.warmup_steps,
        weight_decay=profile.weight_decay,
        grad_clip=profile.grad_clip,
        val_every=1,
        val_batches=profile.val_batches,
        device=device,
    )
    trainer.resume_if_possible()

    train_stream = OpenWebTextStream(
        split="train",
        block_size=profile.block_size,
        seed=seed,
    )
    val_stream = OpenWebTextStream(
        split="val",
        block_size=profile.block_size,
        seed=seed,
    )
    history = trainer.train(train_stream, val_stream)
    torch.save(history, paths.train_history_path)
    best = _finalize_best_checkpoint(paths)
    return {
        "history": history,
        **best,
    }


def _train_hf_profile(
    profile: ExperimentProfile,
    paths: RunArtifacts,
    seed: int,
    device: torch.device,
) -> Dict[str, object]:
    from datasets import load_dataset

    set_seed(seed)
    dataset = load_dataset(profile.dataset_name, profile.dataset_config)
    train_tokens = _encode_split_texts(
        _get_hf_split_texts(dataset, profile.train_split, profile.text_column),
        profile.block_size,
    )
    val_tokens = _encode_split_texts(
        _get_hf_split_texts(dataset, profile.validation_split, profile.text_column),
        profile.block_size,
    )

    model = TransformerLM(profile.model_config).to(device)
    decay_params = []
    no_decay_params = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": profile.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=profile.max_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    scheduler = CosineScheduler(
        max_lr=profile.max_lr,
        min_lr=profile.min_lr,
        warmup_steps=profile.warmup_steps,
        total_steps=profile.total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    history: List[Dict[str, float]] = []
    checkpoint_steps = set(profile.checkpoint_steps or [0, profile.total_steps])
    best_val_loss = float("inf")
    best_step = 0
    no_improve_ckpts = 0
    train_loss_window: List[float] = []
    start_time = time.time()

    val_loss_0 = _estimate_tensor_val_loss(
        model,
        val_tokens,
        batch_size=profile.batch_size,
        block_size=profile.block_size,
        n_batches=profile.val_batches,
        device=device,
    )
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        step=0,
        train_loss=float("inf"),
        val_loss=val_loss_0,
        ckpt_dir=paths.ckpt_dir,
        seed=seed,
    )
    shutil.copy2(paths.ckpt_dir / "ckpt_0000000.pt", paths.best_ckpt_path)
    best_val_loss = val_loss_0
    history.append(
        {
            "step": 0,
            "train_loss": float("inf"),
            "val_loss": val_loss_0,
            "lr": scheduler.get_lr(0),
            "elapsed_min": 0.0,
        }
    )
    print(f"step 0 | val_loss={val_loss_0:.4f} | checkpoint saved")

    for step in range(1, profile.total_steps + 1):
        lr = scheduler.set_lr(optimizer, step)
        x, y = _sample_batch(train_tokens, profile.batch_size, profile.block_size, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits, _ = model(x, return_attention=False)
            bsz, seq_len, vocab = logits.shape
            loss = nn.functional.cross_entropy(logits.view(bsz * seq_len, vocab), y.reshape(bsz * seq_len))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), profile.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        train_loss_window.append(float(loss.item()))

        if step not in checkpoint_steps:
            continue

        smooth_train_loss = float(np.mean(train_loss_window[-50:]))
        val_loss = _estimate_tensor_val_loss(
            model,
            val_tokens,
            batch_size=profile.batch_size,
            block_size=profile.block_size,
            n_batches=profile.val_batches,
            device=device,
        )
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            step=step,
            train_loss=smooth_train_loss,
            val_loss=val_loss,
            ckpt_dir=paths.ckpt_dir,
            seed=seed,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_step = step
            no_improve_ckpts = 0
            shutil.copy2(paths.ckpt_dir / f"ckpt_{step:07d}.pt", paths.best_ckpt_path)
        else:
            no_improve_ckpts += 1

        history.append(
            {
                "step": step,
                "train_loss": smooth_train_loss,
                "val_loss": val_loss,
                "lr": lr,
                "elapsed_min": (time.time() - start_time) / 60.0,
            }
        )
        print(
            f"step {step:5d} | "
            f"train_loss={smooth_train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"lr={lr:.2e} | "
            f"elapsed={(time.time() - start_time) / 60.0:.1f} min"
        )
        if (
            profile.early_stopping_patience_ckpts is not None and
            step >= profile.min_steps_before_early_stop and
            no_improve_ckpts >= profile.early_stopping_patience_ckpts
        ):
            print(
                "early stopping triggered | "
                f"step={step} | "
                f"best_step={best_step} | "
                f"best_val_loss={best_val_loss:.4f}"
            )
            break

    torch.save(history, paths.train_history_path)
    summary = {
        "best_checkpoint": str(paths.best_ckpt_path),
        "best_step": best_step,
        "best_val_loss": best_val_loss,
        "checkpoint_count": len(list(paths.ckpt_dir.glob("ckpt_*.pt"))),
    }
    _write_json(paths.train_summary_path, summary)
    print(
        "training complete | "
        f"best_step={best_step} | "
        f"best_val_loss={best_val_loss:.4f} | "
        f"checkpoints={summary['checkpoint_count']}"
    )
    return {
        "history": history,
        **summary,
    }


def run_single_training(
    profile: ExperimentProfile | str,
    seed: int,
    artifact_root: Path | str = Path("artifacts"),
    device: str | torch.device = "auto",
    reset_run: bool = False,
) -> Dict[str, object]:
    """Train one profile/seed pair and save checkpoints locally."""

    profile_obj = get_profile(profile) if isinstance(profile, str) else profile
    paths = resolve_artifacts(profile_obj, seed, artifact_root)
    if reset_run:
        reset_run_artifacts(paths, reset_probe=False)
    ensure_dirs(paths)

    device_obj = resolve_device(device) if isinstance(device, str) else device
    if profile_obj.dataset_family == "openwebtext_stream":
        return _train_openwebtext_profile(profile_obj, paths, seed, device_obj)
    return _train_hf_profile(profile_obj, paths, seed, device_obj)


def run_single_probing(
    profile: ExperimentProfile | str,
    seed: int,
    artifact_root: Path | str = Path("artifacts"),
    device: str | torch.device = "auto",
    batch_size: Optional[int] = None,
    rebuild_probe: bool = False,
) -> Dict:
    """Run the probing pipeline for one trained run."""

    profile_obj = get_profile(profile) if isinstance(profile, str) else profile
    paths = resolve_artifacts(profile_obj, seed, artifact_root)
    device_obj = resolve_device(device) if isinstance(device, str) else device
    build_or_load_probe_dataset(
        profile_obj,
        paths,
        device=device_obj,
        rebuild=rebuild_probe,
    )
    return run_probing_pipeline(
        ckpt_dir=paths.ckpt_dir,
        probe_path=paths.probe_path,
        output_path=paths.results_path,
        ties_log_path=paths.ties_path,
        seed=seed,
        device=device_obj,
        batch_size=batch_size or profile_obj.probe_batch_size,
        resume=True,
    )


def analyze_single_run(
    profile: ExperimentProfile | str,
    seed: int,
    artifact_root: Path | str = Path("artifacts"),
) -> Dict[str, object]:
    """Compute full single-run analysis outputs and save the figure bundle."""

    profile_obj = get_profile(profile) if isinstance(profile, str) else profile
    paths = resolve_artifacts(profile_obj, seed, artifact_root)
    results = load_run_results(paths.results_path)
    global_curves = compute_global_curves([results])
    onset_steps = compute_specialization_onset(global_curves, threshold_frac=0.05)
    onset_cis = compute_onset_bootstrap_cis(
        [results],
        threshold_frac=0.05,
        n_bootstraps=1000,
        random_seed=0,
    )
    learned_onset_steps = compute_specialization_onset(
        global_curves,
        threshold_frac=0.05,
        exclude_positional_init=True,
    )
    mixed_behavior = compute_mixed_behavior_summary([results])
    per_layer_curves = compute_per_layer_curves([results])
    trajectories = compute_head_trajectories(results)
    interesting = find_interesting_trajectories(trajectories, min_type_changes=2)
    change_matrix = compute_type_change_matrix([results])
    sink_persistence = compute_sink_persistence([results])
    hist_data = compute_stability_histogram(change_matrix)
    per_type_stability = compute_per_type_stability([results], change_matrix)
    induction_curve = compute_induction_count_curve([results])
    val_loss_curve = extract_val_loss_curve([results], ckpt_dir=paths.ckpt_dir)
    crossing_steps = find_crossing_steps(induction_curve, fractions=[0.10, 0.25, 0.50])
    discontinuity = compute_discontinuity_score(induction_curve)
    inflection_10 = (
        detect_val_loss_inflection(val_loss_curve, crossing_steps[0.10], window_steps=500)
        if crossing_steps[0.10] is not None
        else {
            "inflection_found": False,
            "inflection_step": None,
            "crossing_step": -1,
            "delta_steps": None,
            "second_deriv": None,
        }
    )
    inflection_25 = (
        detect_val_loss_inflection(val_loss_curve, crossing_steps[0.25], window_steps=500)
        if crossing_steps[0.25] is not None
        else {
            "inflection_found": False,
            "inflection_step": None,
            "crossing_step": -1,
            "delta_steps": None,
            "second_deriv": None,
        }
    )
    threshold_sensitivity = run_threshold_sensitivity([results], scale_factors=[0.8, 1.0, 1.2])

    timeline_path = paths.figures_dir / f"timeline_seed{seed}.png"
    dominant_heatmap_path = paths.figures_dir / f"dominant_type_heatmap_seed{seed}.png"
    specialization_heatmap_path = paths.figures_dir / f"specialization_heatmap_seed{seed}.png"
    stability_path = paths.figures_dir / f"stability_seed{seed}.png"
    trajectories_path = paths.figures_dir / f"trajectories_seed{seed}.png"
    phase_path = paths.figures_dir / f"phase_transition_seed{seed}.png"
    discontinuity_path = paths.figures_dir / f"discontinuity_zoom_seed{seed}.png"

    plot_timeline(
        global_curves,
        timeline_path,
        onset_steps=learned_onset_steps,
        log_x=True,
        show_undiff=True,
        title=f"Head Trajectories — {profile_obj.name} / seed {seed}",
    )
    plot_dominant_type_heatmap(
        per_layer_curves,
        dominant_heatmap_path,
        title=f"Dominant Head Type — {profile_obj.name} / seed {seed}",
    )
    plot_specialization_fraction_heatmap(
        per_layer_curves,
        specialization_heatmap_path,
        title=f"Specialization Fraction — {profile_obj.name} / seed {seed}",
    )
    plot_stability_figure(
        hist_data,
        sink_persistence,
        per_type_stability,
        stability_path,
        title=f"Stability Analysis — {profile_obj.name} / seed {seed}",
    )
    plot_individual_trajectories(
        interesting,
        results["step_index"],
        trajectories_path,
        max_heads=16,
        title=f"Individual Trajectories — {profile_obj.name} / seed {seed}",
    )
    plot_phase_transition(
        induction_curve,
        val_loss_curve,
        crossing_steps,
        inflection_result=inflection_25,
        output_path=phase_path,
        title=f"Induction Emergence vs Val Loss — {profile_obj.name} / seed {seed}",
    )
    plot_discontinuity_comparison(induction_curve, discontinuity_path)

    final_fractions = {
        head_type: float(global_curves["mean"][-1, idx])
        for idx, head_type in enumerate(global_curves["type_names"])
    }
    summary = {
        "profile": profile_obj.name,
        "seed": seed,
        "steps": [int(x) for x in global_curves["steps"].tolist()],
        "onset_steps": onset_steps,
        "onset_cis": onset_cis,
        "learned_onset_steps": learned_onset_steps,
        "final_fractions": final_fractions,
        "mixed_behavior": {
            "fraction_ge2_final": float(mixed_behavior["fraction_ge2_mean"][-1]),
            "fraction_ge3_final": float(mixed_behavior["fraction_ge3_mean"][-1]),
            "mean_dominant_margin_final": float(mixed_behavior["mean_dominant_margin"][-1]),
            "final_top_pairs": mixed_behavior["final_top_pairs"],
            "final_top_triplets": mixed_behavior["final_top_triplets"],
        },
        "crossing_steps": {
            f"{int(frac * 100)}pct": step
            for frac, step in crossing_steps.items()
        },
        "discontinuity_score": float(discontinuity),
        "natural_induction_final_mean": (
            float(np.asarray(results["natural_induction_score_tensor"][-1]).mean())
            if results.get("natural_induction_score_tensor") is not None
            else None
        ),
        "threshold_sensitivity": threshold_sensitivity["robustness_summary"],
        "sink_persistence": {
            "mean_persistence": float(sink_persistence["mean_persistence"]),
            "std_persistence": float(sink_persistence["std_persistence"]),
            "n_ever_sink": int(sink_persistence["n_ever_sink"]),
        },
        "interesting_trajectories": [
            {
                "layer": int(layer),
                "head": int(head),
                "trajectory": [
                    global_curves["type_names"][label]
                    for label in traj
                ],
            }
            for (layer, head), traj in interesting.items()
        ][:16],
        "results_path": str(paths.results_path),
        "figures": {
            "timeline": str(timeline_path),
            "dominant_type_heatmap": str(dominant_heatmap_path),
            "specialization_heatmap": str(specialization_heatmap_path),
            "stability": str(stability_path),
            "trajectories": str(trajectories_path),
            "phase_transition": str(phase_path),
            "discontinuity_zoom": str(discontinuity_path),
        },
    }
    _write_json(paths.summary_path, summary)
    return summary


def analyze_profile_group(
    profile: ExperimentProfile | str,
    seeds: Sequence[int],
    artifact_root: Path | str = Path("artifacts"),
) -> Dict[str, object]:
    """Aggregate multiple completed seeds for one profile."""

    profile_obj = get_profile(profile) if isinstance(profile, str) else profile
    artifact_root_path = Path(artifact_root)

    run_results: List[Dict] = []
    included_seeds: List[int] = []
    for seed in seeds:
        paths = resolve_artifacts(profile_obj, seed, artifact_root_path)
        if not paths.results_path.exists():
            continue
        run_results.append(load_run_results(paths.results_path))
        included_seeds.append(int(seed))

    if not run_results:
        raise FileNotFoundError(
            f"No results found for profile '{profile_obj.name}' in seeds {list(seeds)}."
        )

    global_curves = compute_global_curves(run_results)
    per_layer_curves = compute_per_layer_curves(run_results)
    onset_steps = compute_specialization_onset(global_curves, threshold_frac=0.05)
    onset_cis = compute_onset_bootstrap_cis(
        run_results,
        threshold_frac=0.05,
        n_bootstraps=1000,
        random_seed=0,
    )
    learned_onset_steps = compute_specialization_onset(
        global_curves,
        threshold_frac=0.05,
        exclude_positional_init=True,
    )
    mixed_behavior = compute_mixed_behavior_summary(run_results)

    aggregate_dir = artifact_root_path / profile_obj.name / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    figure_path = aggregate_dir / "timeline_multi_seed.png"
    summary_path = aggregate_dir / "summary_multi_seed.json"

    plot_timeline(
        global_curves,
        figure_path,
        onset_steps=learned_onset_steps,
        log_x=True,
        show_undiff=True,
        title=f"Head Trajectories — {profile_obj.name} / {len(included_seeds)} seeds",
    )

    final_fractions = {
        head_type: float(global_curves["mean"][-1, idx])
        for idx, head_type in enumerate(global_curves["type_names"])
    }
    summary = {
        "profile": profile_obj.name,
        "seeds_requested": [int(seed) for seed in seeds],
        "seeds_included": included_seeds,
        "n_runs": len(run_results),
        "steps": [int(x) for x in global_curves["steps"].tolist()],
        "onset_steps": onset_steps,
        "onset_cis": onset_cis,
        "learned_onset_steps": learned_onset_steps,
        "final_fractions": final_fractions,
        "mixed_behavior": {
            "fraction_ge2_final": float(mixed_behavior["fraction_ge2_mean"][-1]),
            "fraction_ge3_final": float(mixed_behavior["fraction_ge3_mean"][-1]),
            "mean_dominant_margin_final": float(mixed_behavior["mean_dominant_margin"][-1]),
            "final_top_pairs": mixed_behavior["final_top_pairs"],
            "final_top_triplets": mixed_behavior["final_top_triplets"],
        },
        "layer_count": int(per_layer_curves["n_layers"]),
        "timeline_figure": str(figure_path),
        "summary_path": str(summary_path),
    }
    _write_json(summary_path, summary)
    return summary


def _profile_manifest(profile: ExperimentProfile) -> Dict[str, object]:
    return {
        "name": profile.name,
        "description": profile.description,
        "dataset_family": profile.dataset_family,
        "dataset_name": profile.dataset_name,
        "dataset_config": profile.dataset_config,
        "model_size_label": profile.model_size_label,
        "total_steps": profile.total_steps,
        "batch_size": profile.batch_size,
        "block_size": profile.block_size,
        "max_lr": profile.max_lr,
        "min_lr": profile.min_lr,
        "warmup_steps": profile.warmup_steps,
        "weight_decay": profile.weight_decay,
        "grad_clip": profile.grad_clip,
        "val_batches": profile.val_batches,
        "probe_batch_size": profile.probe_batch_size,
        "n_general": profile.n_general,
        "n_induction": profile.n_induction,
        "n_pairs": profile.n_pairs,
        "n_general_holdout": profile.n_general_holdout,
        "n_induction_holdout": profile.n_induction_holdout,
        "n_pairs_holdout": profile.n_pairs_holdout,
        "n_calibration_seeds": profile.n_calibration_seeds,
        "enable_natural_induction": profile.enable_natural_induction,
        "checkpoint_steps": list(profile.checkpoint_steps),
        "model_config": {
            "n_layers": profile.model_config.n_layers,
            "n_heads": profile.model_config.n_heads,
            "d_model": profile.model_config.d_model,
            "d_ffn": profile.model_config.d_ffn,
            "block_size": profile.model_config.block_size,
            "ablation_mode": profile.model_config.ablation_mode,
        },
    }


def run_full_single_experiment(
    profile: ExperimentProfile | str,
    seed: int,
    artifact_root: Path | str = Path("artifacts"),
    device: str | torch.device = "auto",
    reset_run: bool = False,
    rebuild_probe: bool = False,
    skip_train: bool = False,
    skip_probe: bool = False,
    skip_analysis: bool = False,
) -> Dict[str, object]:
    """Execute train -> probe -> analyze for one profile and seed."""

    profile_obj = get_profile(profile) if isinstance(profile, str) else profile
    paths = resolve_artifacts(profile_obj, seed, artifact_root)
    if reset_run:
        reset_run_artifacts(paths, reset_probe=False)
    ensure_dirs(paths)

    manifest: Dict[str, object] = {
        "profile": _profile_manifest(profile_obj),
        "seed": seed,
        "artifacts": {
            "profile_dir": str(paths.profile_dir),
            "probe_path": str(paths.probe_path),
            "seed_dir": str(paths.seed_dir),
            "checkpoint_dir": str(paths.ckpt_dir),
            "best_checkpoint": str(paths.best_ckpt_path),
            "results_path": str(paths.results_path),
            "ties_path": str(paths.ties_path),
            "timeline_figure": str(paths.figure_path),
            "summary_path": str(paths.summary_path),
        },
    }

    print("[Stage] Building or loading probe dataset...")
    build_or_load_probe_dataset(
        profile_obj,
        paths,
        device=device,
        rebuild=rebuild_probe,
    )
    print(f"[Stage] Probe dataset ready: {paths.probe_path}")

    if not skip_train:
        print("[Stage] Starting training...")
        manifest["training"] = run_single_training(
            profile_obj,
            seed=seed,
            artifact_root=artifact_root,
            device=device,
            reset_run=False,
        )
        print("[Stage] Training complete.")

    if not skip_probe:
        print("[Stage] Starting probing...")
        results = run_single_probing(
            profile_obj,
            seed=seed,
            artifact_root=artifact_root,
            device=device,
            batch_size=profile_obj.probe_batch_size,
            rebuild_probe=False,
        )
        manifest["probing"] = {
            "steps": [int(s) for s in results["step_index"]],
            "thresholds": results.get("effective_thresholds", results.get("thresholds")),
            "thresholds_sanitized": bool(results.get("thresholds_sanitized", False)),
        }
        print("[Stage] Probing complete.")

    if not skip_analysis:
        print("[Stage] Starting analysis...")
        manifest["analysis"] = analyze_single_run(
            profile_obj,
            seed=seed,
            artifact_root=artifact_root,
        )
        print("[Stage] Analysis complete.")

    _write_json(paths.manifest_path, manifest)
    print(f"[Stage] Manifest saved: {paths.manifest_path}")
    return manifest


def run_experiment_batch(
    run_specs: Sequence[Dict[str, object] | BatchRunSpec],
    artifact_root: Path | str = Path("artifacts"),
    device: str | torch.device = "auto",
    reset_run: bool = False,
    rebuild_probe: bool = False,
    run_train: bool = True,
    run_probe: bool = True,
    run_analysis: bool = True,
    run_profile_aggregate: bool = True,
) -> Dict[str, object]:
    """Run multiple profiles and seeds sequentially from one notebook/script."""

    normalized_specs = normalize_run_specs(run_specs)
    manifests: List[Dict[str, object]] = []
    aggregate_summaries: List[Dict[str, object]] = []

    for index, spec in enumerate(normalized_specs, start=1):
        profile = spec.profile
        print(f"\n{'=' * 72}")
        print(f"[Batch {index}/{len(normalized_specs)}] profile={profile.name} seeds={list(spec.seeds)}")
        print(f"{'=' * 72}")

        for seed in spec.seeds:
            print(f"\n--- Running profile={profile.name} seed={seed} ---")
            manifest = run_full_single_experiment(
                profile=profile,
                seed=seed,
                artifact_root=artifact_root,
                device=device,
                reset_run=reset_run,
                rebuild_probe=rebuild_probe,
                skip_train=not run_train,
                skip_probe=not run_probe,
                skip_analysis=not run_analysis,
            )
            manifests.append(manifest)

        if run_profile_aggregate and run_analysis:
            aggregate = analyze_profile_group(
                profile=profile,
                seeds=spec.seeds,
                artifact_root=artifact_root,
            )
            aggregate_summaries.append(aggregate)

    return {
        "run_specs": [
            {"profile_name": spec.profile_name, "seeds": list(spec.seeds)}
            for spec in normalized_specs
        ],
        "runs": manifests,
        "aggregates": aggregate_summaries,
    }
