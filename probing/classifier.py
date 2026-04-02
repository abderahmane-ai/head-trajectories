"""
probing/classifier.py — Head type classification from score vectors.

Converts the 5 raw scores per head into a discrete type label using
threshold normalization and argmax, with tie-breaking logic and
logging of ambiguous classifications to ties.csv.

The dominant label remains the primary categorical summary, but we also
retain mixed-behavior metadata:
  - threshold flags
  - normalized scores
  - runner-up behavior
  - dominant margin
  - number of behaviors above threshold
"""

import csv
import warnings
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

HEAD_TYPES: List[str] = [
    "UNDIFFERENTIATED",
    "SINK",
    "PREV_TOKEN",
    "INDUCTION",
    "POSITIONAL",
    "SEMANTIC",
]

HEAD_TYPE_COLORS: Dict[str, str] = {
    "UNDIFFERENTIATED": "#AAAAAA",
    "SINK":             "#E24B4A",
    "PREV_TOKEN":       "#378ADD",
    "INDUCTION":        "#EF9F27",
    "POSITIONAL":       "#1D9E75",
    "SEMANTIC":         "#7F77DD",
}

# Thresholds — one per score type (sink, prev, induction, positional, semantic)
THRESHOLDS: np.ndarray = np.array([0.4, 0.5, 0.3, 0.7, 0.3], dtype=np.float32)

# Tie-breaking tolerance — if top two normalized scores are within this,
# assign UNDIFFERENTIATED and log the tie
TIE_TOLERANCE: float = 0.05
THRESHOLD_EPSILON: float = 1e-6
LABEL_SCHEMA_VERSION: int = 2

LABEL_UNDIFF: int = 0
LABEL_SINK:   int = 1
LABEL_PREV:   int = 2
LABEL_IND:    int = 3
LABEL_POS:    int = 4
LABEL_SEM:    int = 5

BEHAVIOR_NAMES: List[str] = HEAD_TYPES[1:]


@dataclass(frozen=True)
class ClassificationResult:
    label: int
    is_tie: bool
    threshold_flags: np.ndarray
    normalized_scores: np.ndarray
    primary_behavior: int
    runner_up_behavior: int
    dominant_margin: float
    n_behaviors_above_threshold: int


# ─────────────────────────────────────────────────────────────────────────────
# Single head classification
# ─────────────────────────────────────────────────────────────────────────────

def prepare_thresholds(
    thresholds: np.ndarray | torch.Tensor | None,
    epsilon: float = THRESHOLD_EPSILON,
    warn: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """
    Validate raw thresholds and derive effective thresholds for normalization.

    Returns:
        (raw_thresholds, effective_thresholds, sanitization_mask, was_sanitized)
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

def classify_head_details(
    scores:           Tuple[float, float, float, float, float],
    thresholds:       np.ndarray = THRESHOLDS,
    tie_tolerance:    float      = TIE_TOLERANCE,
) -> ClassificationResult:
    """
    Classify a single attention head from its 5 scores.

    Classification logic:
        normalized = scores / thresholds
        if all(scores < thresholds):
            label = UNDIFFERENTIATED
        elif top two normalized scores are within tie_tolerance:
            label = UNDIFFERENTIATED  (ambiguous)
            is_tie = True
        else:
            label = argmax(normalized) + 1

    Args:
        scores:        (sink, prev_token, induction, positional, semantic)
        thresholds:    threshold vector, same order as scores
        tie_tolerance: if two highest normalized scores differ by less than
                       this value, classify as UNDIFFERENTIATED

    Returns:
        ClassificationResult with dominant label and mixed-behavior metadata.
    """

    score_arr = np.array(scores, dtype=np.float32)
    raw_thresholds, effective_thresholds, _, _ = prepare_thresholds(
        thresholds,
        warn=False,
    )
    normalized = score_arr / effective_thresholds   # element-wise division
    threshold_flags = score_arr >= raw_thresholds
    n_behaviors_above_threshold = int(threshold_flags.sum())

    # Check if all raw scores are below their respective thresholds
    if np.all(score_arr < raw_thresholds):
        return ClassificationResult(
            label=LABEL_UNDIFF,
            is_tie=False,
            threshold_flags=threshold_flags,
            normalized_scores=normalized,
            primary_behavior=int(np.argmax(normalized)),
            runner_up_behavior=int(np.argsort(normalized)[-2]),
            dominant_margin=float(np.sort(normalized)[-1] - np.sort(normalized)[-2]),
            n_behaviors_above_threshold=n_behaviors_above_threshold,
        )

    # Sort normalized scores descending to check for ties
    sort_idx = np.argsort(normalized)[::-1]
    sorted_norm = normalized[sort_idx]
    top1, top2  = sorted_norm[0], sorted_norm[1]
    dominant_margin = float(top1 - top2)
    primary_behavior = int(sort_idx[0])
    runner_up_behavior = int(sort_idx[1])

    if (top1 - top2) < tie_tolerance:
        return ClassificationResult(
            label=LABEL_UNDIFF,
            is_tie=True,
            threshold_flags=threshold_flags,
            normalized_scores=normalized,
            primary_behavior=primary_behavior,
            runner_up_behavior=runner_up_behavior,
            dominant_margin=dominant_margin,
            n_behaviors_above_threshold=n_behaviors_above_threshold,
        )

    # Assign label = argmax(normalized) + 1
    label = primary_behavior + 1
    return ClassificationResult(
        label=label,
        is_tie=False,
        threshold_flags=threshold_flags,
        normalized_scores=normalized,
        primary_behavior=primary_behavior,
        runner_up_behavior=runner_up_behavior,
        dominant_margin=dominant_margin,
        n_behaviors_above_threshold=n_behaviors_above_threshold,
    )


def classify_head(
    scores:           Tuple[float, float, float, float, float],
    thresholds:       np.ndarray = THRESHOLDS,
    tie_tolerance:    float      = TIE_TOLERANCE,
) -> Tuple[int, bool]:
    result = classify_head_details(scores, thresholds, tie_tolerance)
    return result.label, result.is_tie


# ─────────────────────────────────────────────────────────────────────────────
# Full checkpoint classification
# ─────────────────────────────────────────────────────────────────────────────

class HeadClassifier:
    """
    Classifies all attention heads at every checkpoint and saves:
      - label_tensor: (n_ckpts, n_layers, n_heads)        int
      - score_tensor: (n_ckpts, n_layers, n_heads, 5)     float32
      - threshold_flag_tensor:      (n_ckpts, n_layers, n_heads, 5) bool
      - normalized_score_tensor:    (n_ckpts, n_layers, n_heads, 5) float32
      - primary_behavior_tensor:    (n_ckpts, n_layers, n_heads) int32
      - runner_up_tensor:           (n_ckpts, n_layers, n_heads) int32
      - dominant_margin_tensor:     (n_ckpts, n_layers, n_heads) float32
      - behavior_count_tensor:      (n_ckpts, n_layers, n_heads) int32
      - natural_induction_score_tensor: optional auxiliary metric
      - step_index:   list of training steps

    Also logs all tie-breaking events to ties.csv.
    """

    def __init__(
        self,
        n_checkpoints: int,
        n_layers:      int,
        n_heads:       int,
        seed:          int,
        ties_log_path: Path,
        thresholds:    np.ndarray = THRESHOLDS,
        tie_tolerance: float      = TIE_TOLERANCE,
    ) -> None:
        self.n_checkpoints = n_checkpoints
        self.n_layers      = n_layers
        self.n_heads       = n_heads
        self.seed          = seed
        self.ties_log_path = ties_log_path
        (
            self.raw_thresholds,
            self.effective_thresholds,
            self.threshold_sanitization_mask,
            self.thresholds_sanitized,
        ) = prepare_thresholds(thresholds, warn=True)
        self.thresholds = self.effective_thresholds
        self.tie_tolerance = tie_tolerance

        # Pre-allocate tensors
        self.label_tensor = torch.zeros(
            (n_checkpoints, n_layers, n_heads), dtype=torch.int32
        )
        self.score_tensor = torch.zeros(
            (n_checkpoints, n_layers, n_heads, 5), dtype=torch.float32
        )
        self.threshold_flag_tensor = torch.zeros(
            (n_checkpoints, n_layers, n_heads, 5), dtype=torch.bool
        )
        self.normalized_score_tensor = torch.zeros(
            (n_checkpoints, n_layers, n_heads, 5), dtype=torch.float32
        )
        self.primary_behavior_tensor = torch.full(
            (n_checkpoints, n_layers, n_heads), -1, dtype=torch.int32
        )
        self.runner_up_tensor = torch.full(
            (n_checkpoints, n_layers, n_heads), -1, dtype=torch.int32
        )
        self.dominant_margin_tensor = torch.zeros(
            (n_checkpoints, n_layers, n_heads), dtype=torch.float32
        )
        self.behavior_count_tensor = torch.zeros(
            (n_checkpoints, n_layers, n_heads), dtype=torch.int32
        )
        self.natural_induction_score_tensor: Optional[torch.Tensor] = None
        self.step_index: List[int] = []

        # Tie log buffer
        self._ties: List[Dict] = []

    def record(
        self,
        ckpt_idx:     int,
        step:         int,
        layer:        int,
        head:         int,
        scores:       Tuple[float, float, float, float, float],
        natural_induction_score: Optional[float] = None,
    ) -> int:
        """
        Classify one head at one checkpoint and record the result.

        Args:
            ckpt_idx: index into the checkpoint axis (0-indexed)
            step:     training step of this checkpoint
            layer:    layer index
            head:     head index within layer
            scores:   (sink, prev, induction, positional, semantic)

        Returns:
            label: integer head type label
        """

        result = classify_head_details(
            scores,
            self.raw_thresholds,
            self.tie_tolerance,
        )

        self.score_tensor[ckpt_idx, layer, head] = torch.tensor(
            scores, dtype=torch.float32
        )
        self.label_tensor[ckpt_idx, layer, head] = result.label
        self.threshold_flag_tensor[ckpt_idx, layer, head] = torch.tensor(
            result.threshold_flags, dtype=torch.bool
        )
        self.normalized_score_tensor[ckpt_idx, layer, head] = torch.tensor(
            result.normalized_scores, dtype=torch.float32
        )
        self.primary_behavior_tensor[ckpt_idx, layer, head] = result.primary_behavior
        self.runner_up_tensor[ckpt_idx, layer, head] = result.runner_up_behavior
        self.dominant_margin_tensor[ckpt_idx, layer, head] = result.dominant_margin
        self.behavior_count_tensor[ckpt_idx, layer, head] = (
            result.n_behaviors_above_threshold
        )
        if natural_induction_score is not None:
            if self.natural_induction_score_tensor is None:
                self.natural_induction_score_tensor = torch.zeros(
                    (self.n_checkpoints, self.n_layers, self.n_heads),
                    dtype=torch.float32,
                )
            self.natural_induction_score_tensor[ckpt_idx, layer, head] = (
                natural_induction_score
            )

        if result.is_tie:
            top2_idx = [result.primary_behavior, result.runner_up_behavior]
            self._ties.append({
                "run_seed":       self.seed,
                "checkpoint_step": step,
                "layer":          layer,
                "head":           head,
                "score_1":        float(scores[top2_idx[0]]),
                "score_2":        float(scores[top2_idx[1]]),
                "types_tied":     f"{HEAD_TYPES[top2_idx[0]+1]}|{HEAD_TYPES[top2_idx[1]+1]}",
                "dominant_margin": result.dominant_margin,
                "n_behaviors_above_threshold": result.n_behaviors_above_threshold,
            })

        return result.label

    def register_step(self, step: int) -> None:
        """Append a training step to the step index."""
        self.step_index.append(step)

    def flush_ties(self) -> None:
        """Write all accumulated tie events to ties.csv."""

        if not self._ties:
            return

        self.ties_log_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.ties_log_path.exists()

        with open(self.ties_log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "run_seed", "checkpoint_step", "layer", "head",
                "score_1", "score_2", "types_tied",
                "dominant_margin", "n_behaviors_above_threshold",
            ])
            if not file_exists:
                writer.writeheader()
            writer.writerows(self._ties)

        self._ties = []

    def save(self, output_path: Path) -> None:
        """
        Save label_tensor, score_tensor, and step_index to a single .pt file.

        Args:
            output_path: path to save the results file
        """

        self.flush_ties()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "label_tensor": self.label_tensor,
                "score_tensor": self.score_tensor,
                "threshold_flag_tensor": self.threshold_flag_tensor,
                "normalized_score_tensor": self.normalized_score_tensor,
                "primary_behavior_tensor": self.primary_behavior_tensor,
                "runner_up_tensor": self.runner_up_tensor,
                "dominant_margin_tensor": self.dominant_margin_tensor,
                "behavior_count_tensor": self.behavior_count_tensor,
                "step_index":   self.step_index,
                "seed":         self.seed,
                "n_layers":     self.n_layers,
                "n_heads":      self.n_heads,
                "type_names":   HEAD_TYPES,
                "behavior_names": BEHAVIOR_NAMES,
                "label_schema_version": LABEL_SCHEMA_VERSION,
                "thresholds":   self.effective_thresholds.tolist(),
                "raw_thresholds": self.raw_thresholds.tolist(),
                "effective_thresholds": self.effective_thresholds.tolist(),
                "threshold_sanitization_mask": self.threshold_sanitization_mask.tolist(),
                "thresholds_sanitized": self.thresholds_sanitized,
                "natural_induction_score_tensor": self.natural_induction_score_tensor,
            },
            output_path,
        )

    @staticmethod
    def load(path: Path) -> Dict:
        """
        Load a saved classification results file.

        Returns dict with keys:
            label_tensor, score_tensor, step_index, seed,
            n_layers, n_heads, thresholds
        """

        if not path.exists():
            raise FileNotFoundError(f"Results file not found: {path}")

        data = torch.load(path, weights_only=True)

        required = {
            "label_tensor", "score_tensor", "step_index",
            "seed", "n_layers", "n_heads",
        }
        missing = required - set(data.keys())
        if missing:
            raise ValueError(f"Results file missing keys: {missing}")

        if "raw_thresholds" not in data and "thresholds" in data:
            data["raw_thresholds"] = data["thresholds"]
        if "effective_thresholds" not in data and "thresholds" in data:
            data["effective_thresholds"] = data["thresholds"]
        if "threshold_sanitization_mask" not in data:
            data["threshold_sanitization_mask"] = [False] * 5
        if "thresholds_sanitized" not in data:
            data["thresholds_sanitized"] = False
        if "type_names" not in data:
            data["type_names"] = HEAD_TYPES
        if "behavior_names" not in data:
            data["behavior_names"] = BEHAVIOR_NAMES
        if "label_schema_version" not in data:
            data["label_schema_version"] = 1
        if "threshold_flag_tensor" not in data or "normalized_score_tensor" not in data:
            label_tensor = data["label_tensor"]
            score_tensor = data["score_tensor"]
            n_ckpts, n_layers, n_heads, _ = score_tensor.shape
            threshold_flag_tensor = torch.zeros(
                (n_ckpts, n_layers, n_heads, 5), dtype=torch.bool
            )
            normalized_score_tensor = torch.zeros(
                (n_ckpts, n_layers, n_heads, 5), dtype=torch.float32
            )
            primary_behavior_tensor = torch.full(
                (n_ckpts, n_layers, n_heads), -1, dtype=torch.int32
            )
            runner_up_tensor = torch.full(
                (n_ckpts, n_layers, n_heads), -1, dtype=torch.int32
            )
            dominant_margin_tensor = torch.zeros(
                (n_ckpts, n_layers, n_heads), dtype=torch.float32
            )
            behavior_count_tensor = torch.zeros(
                (n_ckpts, n_layers, n_heads), dtype=torch.int32
            )
            thresholds = np.asarray(
                data.get("raw_thresholds", data.get("thresholds", THRESHOLDS)),
                dtype=np.float32,
            )
            for ckpt in range(n_ckpts):
                for layer in range(n_layers):
                    for head in range(n_heads):
                        result = classify_head_details(
                            tuple(score_tensor[ckpt, layer, head].tolist()),
                            thresholds=thresholds,
                            tie_tolerance=TIE_TOLERANCE,
                        )
                        threshold_flag_tensor[ckpt, layer, head] = torch.tensor(
                            result.threshold_flags, dtype=torch.bool
                        )
                        normalized_score_tensor[ckpt, layer, head] = torch.tensor(
                            result.normalized_scores, dtype=torch.float32
                        )
                        primary_behavior_tensor[ckpt, layer, head] = result.primary_behavior
                        runner_up_tensor[ckpt, layer, head] = result.runner_up_behavior
                        dominant_margin_tensor[ckpt, layer, head] = result.dominant_margin
                        behavior_count_tensor[ckpt, layer, head] = (
                            result.n_behaviors_above_threshold
                        )
                        if int(label_tensor[ckpt, layer, head]) not in range(len(HEAD_TYPES)):
                            label_tensor[ckpt, layer, head] = result.label
            data["threshold_flag_tensor"] = threshold_flag_tensor
            data["normalized_score_tensor"] = normalized_score_tensor
            data["primary_behavior_tensor"] = primary_behavior_tensor
            data["runner_up_tensor"] = runner_up_tensor
            data["dominant_margin_tensor"] = dominant_margin_tensor
            data["behavior_count_tensor"] = behavior_count_tensor
            data["legacy_metadata_upgraded"] = True
        else:
            data["legacy_metadata_upgraded"] = False

        return data
