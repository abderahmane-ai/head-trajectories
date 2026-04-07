"""
probing/classifier.py — Statistical head behavior inference from score vectors.

New default methodology:
  1. Compute one-sided empirical p-values from pooled null scores
  2. Apply per-head BH-FDR across the 5 metrics
  3. Treat the surviving metrics as the head's active behavior set
  4. Assign a dominant summary label only if one surviving behavior clears
     a fixed dominance margin in null-relative effect-size space

Legacy compatibility paths remain available only for loading/analyzing
old results and compatibility utilities. They are deprecated.
"""

from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


HEAD_TYPES: List[str] = [
    "WEAK",
    "AMBIGUOUS",
    "SINK",
    "PREV_TOKEN",
    "INDUCTION",
    "POSITIONAL",
    "SEMANTIC",
]

HEAD_TYPE_COLORS: Dict[str, str] = {
    "WEAK": "#D9D9D9",
    "AMBIGUOUS": "#7D8691",
    "SINK": "#E24B4A",
    "PREV_TOKEN": "#378ADD",
    "INDUCTION": "#EF9F27",
    "POSITIONAL": "#1D9E75",
    "SEMANTIC": "#7F77DD",
}

# Legacy threshold defaults remain for diagnostics/legacy loading only (deprecated).
THRESHOLDS: np.ndarray = np.array([0.4, 0.5, 0.3, 0.7, 0.3], dtype=np.float32)
TIE_TOLERANCE: float = 0.05
THRESHOLD_EPSILON: float = 1e-6
DEFAULT_FDR_ALPHA: float = 0.05
DEFAULT_DOMINANCE_MARGIN: float = 0.5
FDR_CORRECTION_SCOPE: str = "per_head"
LABEL_SCHEMA_VERSION: int = 3

LABEL_WEAK: int = 0
LABEL_AMBIGUOUS: int = 1
LABEL_SINK: int = 2
LABEL_PREV: int = 3
LABEL_IND: int = 4
LABEL_POS: int = 5
LABEL_SEM: int = 6

# Legacy alias retained for old callers and legacy report code (deprecated).
LABEL_UNDIFF: int = LABEL_WEAK

BEHAVIOR_NAMES: List[str] = [
    "SINK",
    "PREV_TOKEN",
    "INDUCTION",
    "POSITIONAL",
    "SEMANTIC",
]
BEHAVIOR_TO_LABEL: Dict[int, int] = {
    0: LABEL_SINK,
    1: LABEL_PREV,
    2: LABEL_IND,
    3: LABEL_POS,
    4: LABEL_SEM,
}
NONSPECIALIZED_TYPES = {"WEAK", "AMBIGUOUS"}
_LEGACY_FALLBACK_WARNED = False


@dataclass(frozen=True)
class ClassificationResult:
    label: int
    is_tie: bool
    threshold_flags: np.ndarray
    normalized_scores: np.ndarray
    active_behaviors: np.ndarray
    p_values: np.ndarray
    effect_sizes: np.ndarray
    primary_behavior: int
    runner_up_behavior: int
    dominant_margin: float
    n_behaviors_above_threshold: int
    n_active_behaviors: int


def prepare_thresholds(
    thresholds: np.ndarray | torch.Tensor | None,
    epsilon: float = THRESHOLD_EPSILON,
    warn: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """
    Validate raw thresholds and derive effective thresholds for normalization.

    Thresholds are no longer the primary decision rule, but are retained for:
      - legacy loading
      - reference plotting
      - weak-vs-null diagnostics
    """

    if thresholds is None:
        thresholds = THRESHOLDS
    if isinstance(thresholds, torch.Tensor):
        thresholds = thresholds.detach().cpu().numpy()

    raw = np.asarray(thresholds, dtype=np.float32)
    if raw.shape != (5,):
        raise ValueError(f"thresholds must be shape (5,), got {raw.shape}")
    if not np.all(np.isfinite(raw)):
        raise ValueError(f"thresholds must be finite, got {raw.tolist()}")

    effective = raw.copy()
    sanitization_mask = effective <= 0.0
    was_sanitized = bool(sanitization_mask.any())
    if was_sanitized:
        effective[sanitization_mask] = epsilon
        if warn:
            warnings.warn(
                "Non-positive thresholds detected; using epsilon floor for "
                f"safe normalization at indices {np.where(sanitization_mask)[0].tolist()}",
                RuntimeWarning,
                stacklevel=2,
            )

    return raw, effective, sanitization_mask, was_sanitized


def validate_pooled_null_scores(
    pooled_null_scores: np.ndarray | torch.Tensor,
) -> np.ndarray:
    """Validate pooled null scores used for empirical p-values."""

    if isinstance(pooled_null_scores, torch.Tensor):
        pooled_null_scores = pooled_null_scores.detach().cpu().numpy()
    arr = np.asarray(pooled_null_scores, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 5:
        raise ValueError(
            "pooled_null_scores must have shape (n_null_samples, 5), "
            f"got {arr.shape}"
        )
    if arr.shape[0] < 10:
        raise ValueError(
            f"Need at least 10 pooled null samples for empirical inference, got {arr.shape[0]}"
        )
    finite_per_metric = np.isfinite(arr).sum(axis=0)
    if np.any(finite_per_metric < 10):
        raise ValueError(
            "pooled_null_scores must provide at least 10 finite samples per metric, "
            f"got counts {finite_per_metric.tolist()}"
        )
    return arr


def empirical_p_values(
    scores: np.ndarray | torch.Tensor,
    pooled_null_scores: np.ndarray | torch.Tensor,
) -> np.ndarray:
    """Compute one-sided empirical p-values P(null >= observed) per metric."""

    score_arr = np.asarray(scores, dtype=np.float32)
    null_arr = validate_pooled_null_scores(pooled_null_scores)
    if score_arr.shape != (5,):
        raise ValueError(f"scores must be shape (5,), got {score_arr.shape}")

    p_values = np.ones(5, dtype=np.float32)
    for metric_idx in range(5):
        score = score_arr[metric_idx]
        if not np.isfinite(score):
            p_values[metric_idx] = 1.0
            continue
        finite_null = null_arr[:, metric_idx]
        finite_null = finite_null[np.isfinite(finite_null)]
        exceedances = float((finite_null >= score).sum())
        p_values[metric_idx] = (exceedances + 1.0) / float(finite_null.shape[0] + 1)
    return p_values


def null_effect_sizes(
    p_values: np.ndarray | torch.Tensor,
) -> np.ndarray:
    """Convert empirical p-values into null-relative surprise effect sizes."""

    p_arr = np.asarray(p_values, dtype=np.float32)
    if p_arr.shape != (5,):
        raise ValueError(f"p_values must be shape (5,), got {p_arr.shape}")
    safe = np.clip(p_arr, 1e-12, 1.0)
    return -np.log10(safe)


def bh_fdr_mask(
    p_values: np.ndarray | torch.Tensor,
    alpha: float = DEFAULT_FDR_ALPHA,
) -> np.ndarray:
    """Benjamini-Hochberg mask over the 5 per-head metric p-values."""

    p_arr = np.asarray(p_values, dtype=np.float32)
    if p_arr.shape != (5,):
        raise ValueError(f"p_values must be shape (5,), got {p_arr.shape}")
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")

    m = p_arr.shape[0]
    order = np.argsort(p_arr)
    sorted_p = p_arr[order]
    thresholds = alpha * np.arange(1, m + 1, dtype=np.float32) / float(m)
    passed = sorted_p <= thresholds
    if not np.any(passed):
        return np.zeros_like(p_arr, dtype=bool)
    k = int(np.max(np.where(passed)[0]))
    cutoff = sorted_p[k]
    return p_arr <= cutoff


def classify_head_details(
    scores: Tuple[float, float, float, float, float],
    thresholds: np.ndarray = THRESHOLDS,
    pooled_null_scores: Optional[np.ndarray] = None,
    tie_tolerance: float = TIE_TOLERANCE,
    fdr_alpha: float = DEFAULT_FDR_ALPHA,
    dominance_margin: float = DEFAULT_DOMINANCE_MARGIN,
) -> ClassificationResult:
    """
    Classify a single attention head from its 5 scores.

    New default:
      - empirical p-values from pooled null scores
      - BH-FDR active set
      - dominant summary among surviving metrics by effect size

    Legacy fallback (pooled_null_scores is None):
      - threshold-normalized argmax with tie tolerance
    """

    score_arr = np.array(scores, dtype=np.float32)
    invalid_mask = ~np.isfinite(score_arr)
    raw_thresholds, effective_thresholds, _, _ = prepare_thresholds(
        thresholds,
        warn=False,
    )
    normalized = np.full(5, np.nan, dtype=np.float32)
    finite_mask = ~invalid_mask
    normalized[finite_mask] = score_arr[finite_mask] / effective_thresholds[finite_mask]
    threshold_flags = np.zeros(5, dtype=bool)
    threshold_flags[finite_mask] = score_arr[finite_mask] >= raw_thresholds[finite_mask]
    n_behaviors_above_threshold = int(threshold_flags.sum())

    if pooled_null_scores is None:
        global _LEGACY_FALLBACK_WARNED
        if not _LEGACY_FALLBACK_WARNED:
            warnings.warn(
                "Deprecated classifier fallback in use: pooled_null_scores is missing, "
                "so threshold-normalized legacy classification is applied. "
                "Rebuild probe/results artifacts to use the FDR classifier path.",
                DeprecationWarning,
                stacklevel=2,
            )
            _LEGACY_FALLBACK_WARNED = True
        normalized_for_order = np.where(np.isfinite(normalized), normalized, -np.inf)
        if np.all(~threshold_flags):
            legacy_order = np.argsort(normalized_for_order)[::-1]
            return ClassificationResult(
                label=LABEL_WEAK,
                is_tie=False,
                threshold_flags=threshold_flags,
                normalized_scores=normalized,
                active_behaviors=np.zeros(5, dtype=bool),
                p_values=np.ones(5, dtype=np.float32),
                effect_sizes=np.zeros(5, dtype=np.float32),
                primary_behavior=int(legacy_order[0]),
                runner_up_behavior=int(legacy_order[1]),
                dominant_margin=float(normalized_for_order[legacy_order[0]] - normalized_for_order[legacy_order[1]]),
                n_behaviors_above_threshold=n_behaviors_above_threshold,
                n_active_behaviors=0,
            )

        sort_idx = np.argsort(normalized_for_order)[::-1]
        top1, top2 = float(normalized_for_order[sort_idx[0]]), float(normalized_for_order[sort_idx[1]])
        is_tie = (top1 - top2) < tie_tolerance
        label = LABEL_AMBIGUOUS if is_tie else BEHAVIOR_TO_LABEL[int(sort_idx[0])]
        active_behaviors = threshold_flags.astype(bool)
        return ClassificationResult(
            label=label,
            is_tie=is_tie,
            threshold_flags=threshold_flags,
            normalized_scores=normalized,
            active_behaviors=active_behaviors,
            p_values=np.ones(5, dtype=np.float32),
            effect_sizes=normalized.astype(np.float32),
            primary_behavior=int(sort_idx[0]),
            runner_up_behavior=int(sort_idx[1]),
            dominant_margin=float(top1 - top2),
            n_behaviors_above_threshold=n_behaviors_above_threshold,
            n_active_behaviors=int(active_behaviors.sum()),
        )

    p_values = empirical_p_values(score_arr, pooled_null_scores)
    effect_sizes = null_effect_sizes(p_values)
    active_behaviors = bh_fdr_mask(p_values, alpha=fdr_alpha)
    active_behaviors[invalid_mask] = False
    effect_sizes[invalid_mask] = 0.0
    p_values[invalid_mask] = 1.0
    n_active = int(active_behaviors.sum())

    order = np.argsort(effect_sizes)[::-1]
    primary_behavior = int(order[0])
    runner_up_behavior = int(order[1])
    dominant_margin_value = float(effect_sizes[order[0]] - effect_sizes[order[1]])

    if n_active == 0:
        return ClassificationResult(
            label=LABEL_WEAK,
            is_tie=False,
            threshold_flags=threshold_flags,
            normalized_scores=normalized,
            active_behaviors=active_behaviors,
            p_values=p_values,
            effect_sizes=effect_sizes,
            primary_behavior=primary_behavior,
            runner_up_behavior=runner_up_behavior,
            dominant_margin=dominant_margin_value,
            n_behaviors_above_threshold=n_behaviors_above_threshold,
            n_active_behaviors=0,
        )

    active_idx = np.where(active_behaviors)[0]
    active_effects = effect_sizes[active_idx]
    active_order = active_idx[np.argsort(active_effects)[::-1]]
    primary_behavior = int(active_order[0])
    runner_up_behavior = int(active_order[1]) if len(active_order) > 1 else -1
    runner_effect = float(effect_sizes[runner_up_behavior]) if runner_up_behavior >= 0 else 0.0
    dominant_margin_value = float(effect_sizes[primary_behavior] - runner_effect)
    is_tie = len(active_order) > 1 and dominant_margin_value < dominance_margin

    label = LABEL_AMBIGUOUS if is_tie else BEHAVIOR_TO_LABEL[primary_behavior]
    return ClassificationResult(
        label=label,
        is_tie=is_tie,
        threshold_flags=threshold_flags,
        normalized_scores=normalized,
        active_behaviors=active_behaviors,
        p_values=p_values,
        effect_sizes=effect_sizes,
        primary_behavior=primary_behavior,
        runner_up_behavior=runner_up_behavior,
        dominant_margin=dominant_margin_value,
        n_behaviors_above_threshold=n_behaviors_above_threshold,
        n_active_behaviors=n_active,
    )


def classify_head(
    scores: Tuple[float, float, float, float, float],
    thresholds: np.ndarray = THRESHOLDS,
    pooled_null_scores: Optional[np.ndarray] = None,
    tie_tolerance: float = TIE_TOLERANCE,
    fdr_alpha: float = DEFAULT_FDR_ALPHA,
    dominance_margin: float = DEFAULT_DOMINANCE_MARGIN,
) -> Tuple[int, bool]:
    result = classify_head_details(
        scores=scores,
        thresholds=thresholds,
        pooled_null_scores=pooled_null_scores,
        tie_tolerance=tie_tolerance,
        fdr_alpha=fdr_alpha,
        dominance_margin=dominance_margin,
    )
    return result.label, result.is_tie


class HeadClassifier:
    """
    Classify all heads at every checkpoint and persist both:
      - active-set inference (p-values/effect sizes/FDR mask)
      - dominant summary labels for plots and compatibility
    """

    def __init__(
        self,
        n_checkpoints: int,
        n_layers: int,
        n_heads: int,
        seed: int,
        ties_log_path: Path,
        thresholds: np.ndarray = THRESHOLDS,
        pooled_null_scores: Optional[np.ndarray] = None,
        tie_tolerance: float = TIE_TOLERANCE,
        fdr_alpha: float = DEFAULT_FDR_ALPHA,
        dominance_margin: float = DEFAULT_DOMINANCE_MARGIN,
    ) -> None:
        self.n_checkpoints = n_checkpoints
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.seed = seed
        self.ties_log_path = ties_log_path
        (
            self.raw_thresholds,
            self.effective_thresholds,
            self.threshold_sanitization_mask,
            self.thresholds_sanitized,
        ) = prepare_thresholds(thresholds, warn=True)
        self.thresholds = self.effective_thresholds
        self.tie_tolerance = tie_tolerance
        self.fdr_alpha = fdr_alpha
        self.fdr_correction_scope = FDR_CORRECTION_SCOPE
        self.dominance_margin = dominance_margin
        self.pooled_null_scores = (
            validate_pooled_null_scores(pooled_null_scores)
            if pooled_null_scores is not None
            else None
        )

        self.label_tensor = torch.zeros((n_checkpoints, n_layers, n_heads), dtype=torch.int32)
        self.dominant_label_tensor = torch.zeros((n_checkpoints, n_layers, n_heads), dtype=torch.int32)
        self.score_tensor = torch.zeros((n_checkpoints, n_layers, n_heads, 5), dtype=torch.float32)
        self.threshold_flag_tensor = torch.zeros((n_checkpoints, n_layers, n_heads, 5), dtype=torch.bool)
        self.normalized_score_tensor = torch.zeros((n_checkpoints, n_layers, n_heads, 5), dtype=torch.float32)
        self.active_behavior_tensor = torch.zeros((n_checkpoints, n_layers, n_heads, 5), dtype=torch.bool)
        self.p_value_tensor = torch.ones((n_checkpoints, n_layers, n_heads, 5), dtype=torch.float32)
        self.effect_size_tensor = torch.zeros((n_checkpoints, n_layers, n_heads, 5), dtype=torch.float32)
        self.primary_behavior_tensor = torch.full((n_checkpoints, n_layers, n_heads), -1, dtype=torch.int32)
        self.runner_up_tensor = torch.full((n_checkpoints, n_layers, n_heads), -1, dtype=torch.int32)
        self.dominant_margin_tensor = torch.zeros((n_checkpoints, n_layers, n_heads), dtype=torch.float32)
        self.behavior_count_tensor = torch.zeros((n_checkpoints, n_layers, n_heads), dtype=torch.int32)
        self.natural_induction_score_tensor: Optional[torch.Tensor] = None
        self.semantic_valid_fraction_tensor = torch.zeros((n_checkpoints, n_layers, n_heads), dtype=torch.float32)
        self.semantic_defined_tensor = torch.zeros((n_checkpoints, n_layers, n_heads), dtype=torch.bool)
        self.step_index: List[int] = []
        self._ties: List[Dict] = []

    def record(
        self,
        ckpt_idx: int,
        step: int,
        layer: int,
        head: int,
        scores: Tuple[float, float, float, float, float],
        natural_induction_score: Optional[float] = None,
        semantic_valid_fraction: Optional[float] = None,
        semantic_is_defined: Optional[bool] = None,
    ) -> int:
        result = classify_head_details(
            scores=scores,
            thresholds=self.raw_thresholds,
            pooled_null_scores=self.pooled_null_scores,
            tie_tolerance=self.tie_tolerance,
            fdr_alpha=self.fdr_alpha,
            dominance_margin=self.dominance_margin,
        )

        self.score_tensor[ckpt_idx, layer, head] = torch.tensor(scores, dtype=torch.float32)
        self.label_tensor[ckpt_idx, layer, head] = result.label
        self.dominant_label_tensor[ckpt_idx, layer, head] = result.label
        self.threshold_flag_tensor[ckpt_idx, layer, head] = torch.tensor(result.threshold_flags, dtype=torch.bool)
        self.normalized_score_tensor[ckpt_idx, layer, head] = torch.tensor(result.normalized_scores, dtype=torch.float32)
        self.active_behavior_tensor[ckpt_idx, layer, head] = torch.tensor(result.active_behaviors, dtype=torch.bool)
        self.p_value_tensor[ckpt_idx, layer, head] = torch.tensor(result.p_values, dtype=torch.float32)
        self.effect_size_tensor[ckpt_idx, layer, head] = torch.tensor(result.effect_sizes, dtype=torch.float32)
        self.primary_behavior_tensor[ckpt_idx, layer, head] = result.primary_behavior
        self.runner_up_tensor[ckpt_idx, layer, head] = result.runner_up_behavior
        self.dominant_margin_tensor[ckpt_idx, layer, head] = result.dominant_margin
        self.behavior_count_tensor[ckpt_idx, layer, head] = result.n_active_behaviors
        if semantic_valid_fraction is not None:
            self.semantic_valid_fraction_tensor[ckpt_idx, layer, head] = float(semantic_valid_fraction)
        if semantic_is_defined is not None:
            self.semantic_defined_tensor[ckpt_idx, layer, head] = bool(semantic_is_defined)

        if natural_induction_score is not None:
            if self.natural_induction_score_tensor is None:
                self.natural_induction_score_tensor = torch.zeros(
                    (self.n_checkpoints, self.n_layers, self.n_heads),
                    dtype=torch.float32,
                )
            self.natural_induction_score_tensor[ckpt_idx, layer, head] = natural_induction_score

        if result.is_tie:
            top2 = [idx for idx in [result.primary_behavior, result.runner_up_behavior] if idx >= 0]
            tied = "|".join(BEHAVIOR_NAMES[idx] for idx in top2) if top2 else ""
            self._ties.append(
                {
                    "run_seed": self.seed,
                    "checkpoint_step": step,
                    "layer": layer,
                    "head": head,
                    "types_tied": tied,
                    "dominant_margin": result.dominant_margin,
                    "n_active_behaviors": result.n_active_behaviors,
                    "p_values": "|".join(f"{float(x):.6g}" for x in result.p_values.tolist()),
                }
            )

        return result.label

    def register_step(self, step: int) -> None:
        self.step_index.append(step)

    def flush_ties(self) -> None:
        if not self._ties:
            return
        self.ties_log_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.ties_log_path.exists()
        with open(self.ties_log_path, "a", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "run_seed",
                    "checkpoint_step",
                    "layer",
                    "head",
                    "types_tied",
                    "dominant_margin",
                    "n_active_behaviors",
                    "p_values",
                ],
            )
            if not file_exists:
                writer.writeheader()
            writer.writerows(self._ties)
        self._ties = []

    def save(self, output_path: Path) -> None:
        self.flush_ties()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "label_tensor": self.label_tensor,
                "dominant_label_tensor": self.dominant_label_tensor,
                "score_tensor": self.score_tensor,
                "threshold_flag_tensor": self.threshold_flag_tensor,
                "normalized_score_tensor": self.normalized_score_tensor,
                "active_behavior_tensor": self.active_behavior_tensor,
                "p_value_tensor": self.p_value_tensor,
                "effect_size_tensor": self.effect_size_tensor,
                "primary_behavior_tensor": self.primary_behavior_tensor,
                "runner_up_tensor": self.runner_up_tensor,
                "dominant_margin_tensor": self.dominant_margin_tensor,
                "behavior_count_tensor": self.behavior_count_tensor,
                "semantic_valid_fraction_tensor": self.semantic_valid_fraction_tensor,
                "semantic_defined_tensor": self.semantic_defined_tensor,
                "step_index": self.step_index,
                "seed": self.seed,
                "n_layers": self.n_layers,
                "n_heads": self.n_heads,
                "type_names": HEAD_TYPES,
                "behavior_names": BEHAVIOR_NAMES,
                "label_schema_version": LABEL_SCHEMA_VERSION,
                "thresholds": self.effective_thresholds.tolist(),
                "raw_thresholds": self.raw_thresholds.tolist(),
                "effective_thresholds": self.effective_thresholds.tolist(),
                "threshold_sanitization_mask": self.threshold_sanitization_mask.tolist(),
                "thresholds_sanitized": self.thresholds_sanitized,
                "fdr_alpha": self.fdr_alpha,
                "fdr_correction_scope": self.fdr_correction_scope,
                "dominance_margin": self.dominance_margin,
                "pooled_null_scores": (
                    torch.tensor(self.pooled_null_scores, dtype=torch.float32)
                    if self.pooled_null_scores is not None
                    else None
                ),
                "natural_induction_score_tensor": self.natural_induction_score_tensor,
            },
            output_path,
        )

    @staticmethod
    def load(path: Path) -> Dict:
        if not path.exists():
            raise FileNotFoundError(f"Results file not found: {path}")

        data = torch.load(path, weights_only=True)
        required = {"label_tensor", "score_tensor", "step_index", "seed", "n_layers", "n_heads"}
        missing = required - set(data.keys())
        if missing:
            raise ValueError(f"Results file missing keys: {missing}")

        if "dominant_label_tensor" not in data:
            data["dominant_label_tensor"] = data["label_tensor"]
        if "raw_thresholds" not in data and "thresholds" in data:
            data["raw_thresholds"] = data["thresholds"]
        if "effective_thresholds" not in data and "thresholds" in data:
            data["effective_thresholds"] = data["thresholds"]
        if "threshold_sanitization_mask" not in data:
            data["threshold_sanitization_mask"] = [False] * 5
        if "thresholds_sanitized" not in data:
            data["thresholds_sanitized"] = False
        if "type_names" not in data:
            # Legacy-only: old result files stored a 6-label ontology with an
            # `UNDIFFERENTIATED` bucket. Modern runs use
            # {WEAK, AMBIGUOUS, five behaviors}. Keep this mapping so archived
            # results remain inspectable.
            legacy_types = ["UNDIFFERENTIATED", "SINK", "PREV_TOKEN", "INDUCTION", "POSITIONAL", "SEMANTIC"]
            data["type_names"] = legacy_types
        if "behavior_names" not in data:
            data["behavior_names"] = BEHAVIOR_NAMES
        if "label_schema_version" not in data:
            data["label_schema_version"] = 1
        if "fdr_alpha" not in data:
            data["fdr_alpha"] = DEFAULT_FDR_ALPHA
        if "fdr_correction_scope" not in data:
            data["fdr_correction_scope"] = FDR_CORRECTION_SCOPE
        if "dominance_margin" not in data:
            data["dominance_margin"] = DEFAULT_DOMINANCE_MARGIN
        if "semantic_valid_fraction_tensor" not in data:
            score_tensor = data["score_tensor"]
            semantic_scores = np.asarray(score_tensor[..., 4], dtype=np.float32)
            data["semantic_valid_fraction_tensor"] = torch.where(
                torch.tensor(np.isfinite(semantic_scores), dtype=torch.bool),
                torch.ones_like(torch.tensor(semantic_scores, dtype=torch.float32)),
                torch.zeros_like(torch.tensor(semantic_scores, dtype=torch.float32)),
            )
        if "semantic_defined_tensor" not in data:
            semantic_valid_fraction = np.asarray(data["semantic_valid_fraction_tensor"], dtype=np.float32)
            data["semantic_defined_tensor"] = torch.tensor(semantic_valid_fraction > 0.0, dtype=torch.bool)

        if "active_behavior_tensor" not in data:
            score_tensor = data["score_tensor"]
            n_ckpts, n_layers, n_heads, _ = score_tensor.shape
            thresholds = np.asarray(
                data.get("raw_thresholds", data.get("thresholds", THRESHOLDS)),
                dtype=np.float32,
            )
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
                        result = classify_head_details(
                            tuple(score_tensor[ckpt, layer, head].tolist()),
                            thresholds=thresholds,
                            pooled_null_scores=None,
                            tie_tolerance=TIE_TOLERANCE,
                        )
                        threshold_flag_tensor[ckpt, layer, head] = torch.tensor(result.threshold_flags, dtype=torch.bool)
                        normalized_score_tensor[ckpt, layer, head] = torch.tensor(result.normalized_scores, dtype=torch.float32)
                        active_behavior_tensor[ckpt, layer, head] = torch.tensor(result.active_behaviors, dtype=torch.bool)
                        p_value_tensor[ckpt, layer, head] = torch.tensor(result.p_values, dtype=torch.float32)
                        effect_size_tensor[ckpt, layer, head] = torch.tensor(result.effect_sizes, dtype=torch.float32)
                        primary_behavior_tensor[ckpt, layer, head] = result.primary_behavior
                        runner_up_tensor[ckpt, layer, head] = result.runner_up_behavior
                        dominant_margin_tensor[ckpt, layer, head] = result.dominant_margin
                        behavior_count_tensor[ckpt, layer, head] = result.n_active_behaviors

            data["threshold_flag_tensor"] = threshold_flag_tensor
            data["normalized_score_tensor"] = normalized_score_tensor
            data["active_behavior_tensor"] = active_behavior_tensor
            data["p_value_tensor"] = p_value_tensor
            data["effect_size_tensor"] = effect_size_tensor
            data["primary_behavior_tensor"] = primary_behavior_tensor
            data["runner_up_tensor"] = runner_up_tensor
            data["dominant_margin_tensor"] = dominant_margin_tensor
            data["behavior_count_tensor"] = behavior_count_tensor
            data["legacy_metadata_upgraded"] = True
            data["legacy_path_deprecated"] = True
        else:
            data["legacy_metadata_upgraded"] = False
            data["legacy_path_deprecated"] = False

        if "pooled_null_scores" in data and data["pooled_null_scores"] is not None:
            data["pooled_null_scores"] = validate_pooled_null_scores(data["pooled_null_scores"])

        return data
