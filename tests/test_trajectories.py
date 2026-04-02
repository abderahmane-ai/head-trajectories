"""
Unit tests for analysis/trajectories.py — trajectory computation.
"""

import pytest
import torch
import numpy as np
from analysis.trajectories import (
    compute_global_curves,
    compute_per_layer_curves,
    compute_head_trajectories,
    find_interesting_trajectories,
    compute_specialization_onset,
    compute_onset_bootstrap_cis,
    compute_mixed_behavior_summary,
)
from probing import LABEL_UNDIFF, LABEL_SINK, LABEL_IND


class TestComputeGlobalCurves:
    """Test compute_global_curves function."""
    
    def test_single_seed(self):
        """Test with a single seed."""
        result = {
            "label_tensor": torch.randint(0, 6, (10, 4, 8)),
            "step_index": list(range(0, 1000, 100)),
            "seed": 42,
            "n_layers": 4,
            "n_heads": 8,
            "type_names": ["UNDIFFERENTIATED", "SINK", "PREV_TOKEN", "INDUCTION", "POSITIONAL", "SEMANTIC"],
        }
        
        curves = compute_global_curves([result])
        
        assert curves["steps"].shape == (10,)
        assert curves["mean"].shape == (10, 6)
        assert curves["std"].shape == (10, 6)
        assert curves["per_seed"].shape == (1, 10, 6)
        assert len(curves["type_names"]) == 6
    
    def test_multiple_seeds(self):
        """Test with multiple seeds."""
        results = []
        for seed in [42, 123, 777]:
            results.append({
                "label_tensor": torch.randint(0, 6, (10, 4, 8)),
                "step_index": list(range(0, 1000, 100)),
                "seed": seed,
                "n_layers": 4,
                "n_heads": 8,
                "type_names": ["UNDIFFERENTIATED", "SINK", "PREV_TOKEN", "INDUCTION", "POSITIONAL", "SEMANTIC"],
            })
        
        curves = compute_global_curves(results)
        
        assert curves["per_seed"].shape == (3, 10, 6)
        # Std should be non-zero with multiple seeds
        assert curves["std"].sum() > 0
    
    def test_fraction_sums_to_one(self):
        """Test that fractions sum to 1.0 at each checkpoint."""
        result = {
            "label_tensor": torch.randint(0, 6, (5, 2, 4)),
            "step_index": [0, 100, 200, 300, 400],
            "seed": 42,
            "n_layers": 2,
            "n_heads": 4,
            "type_names": ["UNDIFFERENTIATED", "SINK", "PREV_TOKEN", "INDUCTION", "POSITIONAL", "SEMANTIC"],
        }
        
        curves = compute_global_curves([result])
        
        # Sum across types should be 1.0 at each checkpoint
        sums = curves["mean"].sum(axis=1)
        assert np.allclose(sums, 1.0), f"Fractions don't sum to 1: {sums}"


class TestComputePerLayerCurves:
    """Test compute_per_layer_curves function."""
    
    def test_per_layer_shapes(self):
        """Test output shapes."""
        results = []
        for seed in [42, 123]:
            results.append({
                "label_tensor": torch.randint(0, 6, (10, 4, 8)),
                "step_index": list(range(0, 1000, 100)),
                "seed": seed,
                "n_layers": 4,
                "n_heads": 8,
                "type_names": ["UNDIFFERENTIATED", "SINK", "PREV_TOKEN", "INDUCTION", "POSITIONAL", "SEMANTIC"],
            })
        
        curves = compute_per_layer_curves(results)
        
        assert curves["steps"].shape == (10,)
        assert curves["n_layers"] == 4
        assert curves["per_layer_mean"].shape == (4, 10, 6)
        assert curves["per_layer_std"].shape == (4, 10, 6)
    
    def test_per_layer_fractions(self):
        """Test that per-layer fractions sum to 1.0."""
        result = {
            "label_tensor": torch.randint(0, 6, (5, 3, 4)),
            "step_index": [0, 100, 200, 300, 400],
            "seed": 42,
            "n_layers": 3,
            "n_heads": 4,
            "type_names": ["UNDIFFERENTIATED", "SINK", "PREV_TOKEN", "INDUCTION", "POSITIONAL", "SEMANTIC"],
        }
        
        curves = compute_per_layer_curves([result])
        
        # For each layer and checkpoint, fractions should sum to 1
        for layer in range(3):
            sums = curves["per_layer_mean"][layer].sum(axis=1)
            assert np.allclose(sums, 1.0), f"Layer {layer} fractions don't sum to 1"


class TestComputeHeadTrajectories:
    """Test compute_head_trajectories function."""
    
    def test_trajectory_extraction(self):
        """Test that trajectories are extracted correctly."""
        n_ckpts, n_layers, n_heads = 10, 2, 4
        result = {
            "label_tensor": torch.randint(0, 6, (n_ckpts, n_layers, n_heads)),
            "step_index": list(range(0, 1000, 100)),
            "seed": 42,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "type_names": ["UNDIFFERENTIATED", "SINK", "PREV_TOKEN", "INDUCTION", "POSITIONAL", "SEMANTIC"],
        }
        
        trajectories = compute_head_trajectories(result)
        
        # Should have one trajectory per head
        assert len(trajectories) == n_layers * n_heads
        
        # Each trajectory should have n_ckpts labels
        for (layer, head), traj in trajectories.items():
            assert len(traj) == n_ckpts
            assert all(0 <= label <= 5 for label in traj)
    
    def test_trajectory_keys(self):
        """Test that trajectory keys are (layer, head) tuples."""
        result = {
            "label_tensor": torch.randint(0, 6, (5, 2, 3)),
            "step_index": [0, 100, 200, 300, 400],
            "seed": 42,
            "n_layers": 2,
            "n_heads": 3,
            "type_names": ["UNDIFFERENTIATED", "SINK", "PREV_TOKEN", "INDUCTION", "POSITIONAL", "SEMANTIC"],
        }
        
        trajectories = compute_head_trajectories(result)
        
        expected_keys = {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)}
        assert set(trajectories.keys()) == expected_keys


class TestFindInterestingTrajectories:
    """Test find_interesting_trajectories function."""
    
    def test_filter_by_changes(self):
        """Test filtering by number of type changes."""
        trajectories = {
            (0, 0): [0, 0, 0, 0, 0],  # No changes
            (0, 1): [0, 1, 1, 1, 1],  # 1 change
            (0, 2): [0, 1, 2, 3, 4],  # 4 changes
            (1, 0): [0, 0, 1, 1, 2],  # 2 changes
        }
        
        interesting = find_interesting_trajectories(trajectories, min_type_changes=2)
        
        # Should include (0, 2) and (1, 0), exclude (0, 0) and (0, 1)
        assert (0, 2) in interesting
        assert (1, 0) in interesting
        assert (0, 0) not in interesting
        assert (0, 1) not in interesting
    
    def test_empty_result(self):
        """Test with no interesting trajectories."""
        trajectories = {
            (0, 0): [1, 1, 1, 1, 1],  # No changes
            (0, 1): [2, 2, 2, 2, 2],  # No changes
        }
        
        interesting = find_interesting_trajectories(trajectories, min_type_changes=1)
        
        assert len(interesting) == 0


class TestComputeSpecializationOnset:
    """Test compute_specialization_onset function."""
    
    def test_onset_detection(self):
        """Test that onset steps are detected correctly."""
        # Create synthetic curves where types appear at different steps
        steps = np.array([0, 100, 200, 300, 400])
        mean = np.zeros((5, 6))
        
        # SINK appears at step 100 (index 1)
        mean[1:, 1] = 0.1  # 10% of heads
        
        # INDUCTION appears at step 300 (index 3)
        mean[3:, 3] = 0.1
        
        global_curves = {
            "steps": steps,
            "mean": mean,
            "std": np.zeros_like(mean),
            "per_seed": mean[np.newaxis, :, :],
            "type_names": ["UNDIFFERENTIATED", "SINK", "PREV_TOKEN", "INDUCTION", "POSITIONAL", "SEMANTIC"],
        }
        
        onset_steps = compute_specialization_onset(global_curves, threshold_frac=0.05)
        
        assert onset_steps["SINK"] == 100
        assert onset_steps["INDUCTION"] == 300
    
    def test_never_appears(self):
        """Test types that never reach threshold."""
        steps = np.array([0, 100, 200])
        mean = np.zeros((3, 6))
        mean[:, 0] = 1.0  # All UNDIFFERENTIATED
        
        global_curves = {
            "steps": steps,
            "mean": mean,
            "std": np.zeros_like(mean),
            "per_seed": mean[np.newaxis, :, :],
            "type_names": ["UNDIFFERENTIATED", "SINK", "PREV_TOKEN", "INDUCTION", "POSITIONAL", "SEMANTIC"],
        }
        
        onset_steps = compute_specialization_onset(global_curves, threshold_frac=0.05)
        
        # All non-UNDIFF types should be None
        assert onset_steps["SINK"] is None
        assert onset_steps["INDUCTION"] is None
        assert onset_steps["SEMANTIC"] is None

    def test_exclude_positional_init(self):
        """Step-0 positional onset can be excluded from learned ordering."""
        steps = np.array([0, 100, 200, 300])
        mean = np.zeros((4, 6))
        mean[:, 0] = 0.7
        mean[:, 4] = np.array([0.1, 0.0, 0.1, 0.1])

        global_curves = {
            "steps": steps,
            "mean": mean,
            "std": np.zeros_like(mean),
            "per_seed": mean[np.newaxis, :, :],
            "type_names": ["UNDIFFERENTIATED", "SINK", "PREV_TOKEN", "INDUCTION", "POSITIONAL", "SEMANTIC"],
        }

        onset_default = compute_specialization_onset(global_curves, threshold_frac=0.05)
        onset_learned = compute_specialization_onset(
            global_curves,
            threshold_frac=0.05,
            exclude_positional_init=True,
        )

        assert onset_default["POSITIONAL"] == 0
        assert onset_learned["POSITIONAL"] == 200


class TestBootstrapAndMixedBehavior:
    def test_bootstrap_onset_cis(self):
        result = {
            "label_tensor": torch.tensor(
                [
                    [[0, 0, 0, 0]],
                    [[1, 1, 0, 0]],
                    [[1, 1, 1, 0]],
                    [[1, 1, 1, 1]],
                ],
                dtype=torch.int32,
            ),
            "step_index": [0, 100, 200, 300],
            "seed": 42,
            "n_layers": 1,
            "n_heads": 4,
            "type_names": ["UNDIFFERENTIATED", "SINK", "PREV_TOKEN", "INDUCTION", "POSITIONAL", "SEMANTIC"],
        }
        cis = compute_onset_bootstrap_cis([result], threshold_frac=0.25, n_bootstraps=50, random_seed=0)
        assert "SINK" in cis
        assert cis["SINK"]["point_estimate"] == 100
        assert cis["SINK"]["n_valid_bootstraps"] > 0

    def test_compute_mixed_behavior_summary(self):
        result = {
            "step_index": [0, 100],
            "behavior_count_tensor": torch.tensor([[[1, 2]], [[2, 3]]], dtype=torch.int32),
            "dominant_margin_tensor": torch.tensor([[[0.5, 0.1]], [[0.4, 0.2]]], dtype=torch.float32),
            "primary_behavior_tensor": torch.tensor([[[0, 1]], [[0, 2]]], dtype=torch.int32),
            "runner_up_tensor": torch.tensor([[[1, 0]], [[1, 0]]], dtype=torch.int32),
            "threshold_flag_tensor": torch.tensor(
                [
                    [[[True, False, False, False, False], [True, True, False, False, False]]],
                    [[[True, True, False, False, False], [True, True, True, False, False]]],
                ],
                dtype=torch.bool,
            ),
        }
        summary = compute_mixed_behavior_summary([result])
        assert summary["steps"].tolist() == [0, 100]
        assert summary["fraction_ge2_mean"][-1] == pytest.approx(1.0)
        assert summary["fraction_ge3_mean"][-1] == pytest.approx(0.5)
        assert summary["final_top_pairs"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
