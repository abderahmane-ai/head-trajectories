"""
probing/classifier.py — Head type classification from score vectors.

Converts the 5 raw scores per head into a discrete type label using
threshold normalization and argmax, with tie-breaking logic and
logging of ambiguous classifications to ties.csv.

Head type labels:
    0 = UNDIFFERENTIATED
    1 = SINK
    2 = PREV_TOKEN
    3 = INDUCTION
    4 = POSITIONAL
    5 = SEMANTIC
"""

import csv
import numpy as np
from pathlib import Path
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

LABEL_UNDIFF: int = 0
LABEL_SINK:   int = 1
LABEL_PREV:   int = 2
LABEL_IND:    int = 3
LABEL_POS:    int = 4
LABEL_SEM:    int = 5


# ─────────────────────────────────────────────────────────────────────────────
# Single head classification
# ─────────────────────────────────────────────────────────────────────────────

def classify_head(
    scores:           Tuple[float, float, float, float, float],
    thresholds:       np.ndarray = THRESHOLDS,
    tie_tolerance:    float      = TIE_TOLERANCE,
) -> Tuple[int, bool]:
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
        (label, is_tie)
        label:  integer head type label [0..5]
        is_tie: True if classification was ambiguous
    """

    score_arr = np.array(scores, dtype=np.float32)
    if isinstance(thresholds, torch.Tensor):
        thresholds = thresholds.detach().cpu().numpy()
    thresholds = np.asarray(thresholds, dtype=np.float32)
    normalized = score_arr / thresholds   # element-wise division

    # Check if all raw scores are below their respective thresholds
    if np.all(score_arr < thresholds):
        return LABEL_UNDIFF, False

    # Sort normalized scores descending to check for ties
    sorted_norm = np.sort(normalized)[::-1]
    top1, top2  = sorted_norm[0], sorted_norm[1]

    if (top1 - top2) < tie_tolerance:
        return LABEL_UNDIFF, True

    # Assign label = argmax(normalized) + 1
    label = int(np.argmax(normalized)) + 1
    return label, False


# ─────────────────────────────────────────────────────────────────────────────
# Full checkpoint classification
# ─────────────────────────────────────────────────────────────────────────────

class HeadClassifier:
    """
    Classifies all attention heads at every checkpoint and saves:
      - label_tensor: (n_ckpts, n_layers, n_heads)        int
      - score_tensor: (n_ckpts, n_layers, n_heads, 5)     float32
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
        if thresholds is None:
            thresholds = THRESHOLDS
        if isinstance(thresholds, torch.Tensor):
            thresholds = thresholds.detach().cpu().numpy()
        thresholds = np.asarray(thresholds, dtype=np.float32)
        if thresholds.shape != (5,):
            raise ValueError(
                f"thresholds must be shape (5,), got {thresholds.shape}"
            )
        self.thresholds    = thresholds
        self.tie_tolerance = tie_tolerance

        # Pre-allocate tensors
        self.label_tensor = torch.zeros(
            (n_checkpoints, n_layers, n_heads), dtype=torch.int32
        )
        self.score_tensor = torch.zeros(
            (n_checkpoints, n_layers, n_heads, 5), dtype=torch.float32
        )
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

        label, is_tie = classify_head(scores, self.thresholds, self.tie_tolerance)

        self.score_tensor[ckpt_idx, layer, head] = torch.tensor(
            scores, dtype=torch.float32
        )
        self.label_tensor[ckpt_idx, layer, head] = label

        if is_tie:
            score_arr  = np.array(scores, dtype=np.float32)
            normalized = score_arr / self.thresholds
            top2_idx   = np.argsort(normalized)[::-1][:2]
            self._ties.append({
                "run_seed":       self.seed,
                "checkpoint_step": step,
                "layer":          layer,
                "head":           head,
                "score_1":        float(scores[top2_idx[0]]),
                "score_2":        float(scores[top2_idx[1]]),
                "types_tied":     f"{HEAD_TYPES[top2_idx[0]+1]}|{HEAD_TYPES[top2_idx[1]+1]}",
            })

        return label

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
                "score_1", "score_2", "types_tied"
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
                "step_index":   self.step_index,
                "seed":         self.seed,
                "n_layers":     self.n_layers,
                "n_heads":      self.n_heads,
                "thresholds":   self.thresholds.tolist(),
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

        return data
