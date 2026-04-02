"""
Unit tests for data/calibration.py — null calibration behavior.
"""

import numpy as np
import pytest
import torch

from data.calibration import (
    SEMANTIC_METRIC_INDEX,
    _shuffle_attention_rows,
    _compute_threshold_statistics,
    _scramble_causal_attention_keys,
    calibrate_thresholds,
)
from probing.scores import sink_score


def _make_perfect_sink_maps(N: int = 4, H: int = 1, T: int = 8) -> torch.Tensor:
    """Construct causal attention maps for a perfect sink head."""
    attn = torch.zeros(N, H, T, T, dtype=torch.float32)
    attn[:, :, :, 0] = 1.0
    return attn


class TestCalibrationNulls:
    """Test calibration null behavior for the sink metric."""

    def test_row_shuffle_leaves_sink_score_unchanged(self):
        """Row shuffling is the wrong null for sink because sink aggregates over rows."""
        attn = _make_perfect_sink_maps()
        shuffled = _shuffle_attention_rows(
            [attn],
            torch.Generator().manual_seed(0),
        )[0]

        original_score = sink_score(attn[:, 0, :, :])
        shuffled_score = sink_score(shuffled[:, 0, :, :])

        assert original_score == pytest.approx(1.0)
        assert shuffled_score == pytest.approx(original_score)

    def test_key_scramble_changes_sink_score(self):
        """Key scrambling should destroy fixed-key anchoring for sink heads."""
        attn = _make_perfect_sink_maps()
        scrambled = _scramble_causal_attention_keys(
            [attn],
            torch.Generator().manual_seed(0),
        )[0]

        original_score = sink_score(attn[:, 0, :, :])
        scrambled_score = sink_score(scrambled[:, 0, :, :])

        assert original_score == pytest.approx(1.0)
        assert scrambled_score < 0.75

    def test_key_scramble_preserves_causality_and_row_sums(self):
        """The new null should keep rows causal and row-stochastic."""
        attn = _make_perfect_sink_maps(N=2, H=2, T=6)
        scrambled = _scramble_causal_attention_keys(
            [attn],
            torch.Generator().manual_seed(0),
        )[0]

        row_sums = scrambled.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)

        future_mask = torch.triu(torch.ones(6, 6, dtype=torch.bool), diagonal=1)
        assert torch.all(scrambled[:, :, future_mask] == 0.0)


class TestCalibrateThresholds:
    """Smoke tests for threshold calibration."""

    def test_semantic_threshold_uses_p99_not_mean_plus_2std(self):
        scores = np.zeros((100, 5), dtype=np.float32)
        scores[:, 0] = np.linspace(0.1, 0.2, 100)
        scores[:, 1] = np.linspace(0.2, 0.3, 100)
        scores[:, 2] = np.linspace(0.3, 0.4, 100)
        scores[:, 3] = np.linspace(0.4, 0.5, 100)
        scores[:, 4] = np.linspace(-0.02, 0.02, 100)

        thresholds, means, stds, _, p99 = _compute_threshold_statistics(scores)

        semantic_mean_plus_2std = means[SEMANTIC_METRIC_INDEX] + 2.0 * stds[SEMANTIC_METRIC_INDEX]
        assert thresholds[SEMANTIC_METRIC_INDEX] == pytest.approx(
            p99[SEMANTIC_METRIC_INDEX], rel=1e-6
        )
        assert thresholds[SEMANTIC_METRIC_INDEX] != pytest.approx(
            semantic_mean_plus_2std, rel=1e-3
        )

    def test_calibration_returns_finite_positive_thresholds(self, small_config):
        T = small_config.block_size
        probe_dict = {
            "general_seqs": torch.randint(0, small_config.vocab_size, (12, T)),
            "induction_seqs": torch.randint(0, small_config.vocab_size, (6, T)),
            "positional_seqs": torch.randint(0, small_config.vocab_size, (6, T)),
            "induction_p1": torch.randint(2, 6, (6,)),
            "induction_p2": torch.randint(10, 14, (6,)),
            "positional_pairs": torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.long),
            "creation_seed": torch.tensor(0, dtype=torch.long),
            "block_size": torch.tensor(T, dtype=torch.long),
        }

        mean, std, per_seed, diagnostics = calibrate_thresholds(
            probe_dict=probe_dict,
            config=small_config,
            device=torch.device("cpu"),
            batch_size=4,
            n_seeds=2,
            return_diagnostics=True,
        )

        assert mean.shape == (5,)
        assert std.shape == (5,)
        assert per_seed.shape == (2, 5)
        assert np.all(np.isfinite(mean))
        assert np.all(mean > 0.0)
        assert diagnostics["per_seed_metric_means"].shape == (2, 5)
        assert diagnostics["per_seed_metric_stds"].shape == (2, 5)
        assert diagnostics["per_seed_metric_p95"].shape == (2, 5)
        assert diagnostics["per_seed_metric_p99"].shape == (2, 5)
        assert diagnostics["per_seed_nonpositive_mask"].shape == (2, 5)
        assert diagnostics["per_seed_null_scores"].shape[0] == 2
        assert diagnostics["per_seed_null_scores"].shape[-1] == 5
        assert diagnostics["pooled_null_scores"].shape[-1] == 5
        assert diagnostics["null_seed_list"].shape == (2,)
        assert diagnostics["threshold_rules"][SEMANTIC_METRIC_INDEX].startswith("quantile_")
