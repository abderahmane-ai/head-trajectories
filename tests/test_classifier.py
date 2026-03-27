"""
Unit tests for probing/classifier.py — head classification logic.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from probing.classifier import (
    classify_head,
    HeadClassifier,
    prepare_thresholds,
    HEAD_TYPES,
    THRESHOLDS,
    LABEL_UNDIFF,
    LABEL_SINK,
    LABEL_PREV,
    LABEL_IND,
    LABEL_POS,
    LABEL_SEM,
)


class TestClassifyHead:
    """Test the classify_head function."""
    
    def test_all_below_threshold(self):
        """Scores all below threshold should return UNDIFFERENTIATED."""
        scores = (0.1, 0.2, 0.1, 0.3, 0.1)  # All below THRESHOLDS
        label, is_tie = classify_head(scores)
        assert label == LABEL_UNDIFF, f"Should be UNDIFF, got {label}"
        assert not is_tie, "Should not be a tie"
    
    def test_clear_sink(self):
        """Clear sink head should be classified as SINK."""
        scores = (0.9, 0.2, 0.1, 0.3, 0.1)  # Sink dominates
        label, is_tie = classify_head(scores)
        assert label == LABEL_SINK, f"Should be SINK, got {label}"
        assert not is_tie, "Should not be a tie"
    
    def test_clear_induction(self):
        """Clear induction head should be classified as INDUCTION."""
        scores = (0.2, 0.2, 0.8, 0.3, 0.1)  # Induction dominates
        label, is_tie = classify_head(scores)
        assert label == LABEL_IND, f"Should be INDUCTION, got {label}"
        assert not is_tie, "Should not be a tie"
    
    def test_tie_detection(self):
        """Close scores should trigger tie detection."""
        # Sink and prev_token very close after normalization
        scores = (0.42, 0.52, 0.1, 0.3, 0.1)
        # After dividing by thresholds: 0.42/0.4=1.05, 0.52/0.5=1.04
        label, is_tie = classify_head(scores, tie_tolerance=0.05)
        assert label == LABEL_UNDIFF, "Tie should result in UNDIFF"
        assert is_tie, "Should detect tie"
    
    def test_custom_thresholds(self):
        """Test with custom threshold values."""
        scores = (0.35, 0.2, 0.1, 0.3, 0.1)
        custom_thresholds = np.array([0.3, 0.5, 0.3, 0.7, 0.3])
        
        label, is_tie = classify_head(scores, thresholds=custom_thresholds)
        # 0.35 > 0.3, so sink should win
        assert label == LABEL_SINK, f"Should be SINK with custom thresholds, got {label}"
    
    def test_semantic_negative(self):
        """Test that negative semantic scores are handled correctly."""
        scores = (0.2, 0.2, 0.1, 0.3, -0.5)  # Negative semantic
        label, is_tie = classify_head(scores)
        # Negative semantic should not win
        assert label != LABEL_SEM, "Negative semantic should not win"

    def test_nonpositive_thresholds_do_not_crash(self):
        """Non-positive thresholds should be sanitized for normalization only."""
        scores = (0.4, 0.2, 0.1, 0.3, 0.1)
        thresholds = np.array([0.0, 0.5, -0.1, 0.7, 0.3], dtype=np.float32)
        label, is_tie = classify_head(scores, thresholds=thresholds)
        assert label == LABEL_SINK
        assert not is_tie

    def test_nonfinite_thresholds_raise(self):
        """NaN/Inf thresholds should be rejected immediately."""
        scores = (0.4, 0.2, 0.1, 0.3, 0.1)
        with pytest.raises(ValueError):
            classify_head(scores, thresholds=np.array([0.4, np.nan, 0.3, 0.7, 0.3], dtype=np.float32))
        with pytest.raises(ValueError):
            classify_head(scores, thresholds=np.array([0.4, np.inf, 0.3, 0.7, 0.3], dtype=np.float32))


class TestPrepareThresholds:
    """Test threshold validation and sanitization."""

    def test_positive_thresholds_passthrough(self):
        thresholds = np.array([0.4, 0.5, 0.3, 0.7, 0.3], dtype=np.float32)
        raw, effective, mask, sanitized = prepare_thresholds(thresholds)
        assert np.allclose(raw, thresholds)
        assert np.allclose(effective, thresholds)
        assert not sanitized
        assert not mask.any()

    def test_nonpositive_thresholds_are_sanitized(self):
        thresholds = np.array([0.4, 0.0, -0.3, 0.7, 0.3], dtype=np.float32)
        raw, effective, mask, sanitized = prepare_thresholds(thresholds)
        assert np.allclose(raw, thresholds)
        assert sanitized
        assert mask.tolist() == [False, True, True, False, False]
        assert effective[1] > 0.0
        assert effective[2] > 0.0

    def test_nonfinite_thresholds_raise(self):
        with pytest.raises(ValueError):
            prepare_thresholds(np.array([0.4, np.nan, 0.3, 0.7, 0.3], dtype=np.float32))
        with pytest.raises(ValueError):
            prepare_thresholds(np.array([0.4, 0.5, 0.3, np.inf, 0.3], dtype=np.float32))


class TestHeadClassifier:
    """Test the HeadClassifier class."""
    
    def test_initialization(self, workspace_tmpdir):
        """Test classifier initialization."""
        ties_path = workspace_tmpdir / "ties.csv"
        
        classifier = HeadClassifier(
            n_checkpoints=10,
            n_layers=4,
            n_heads=8,
            seed=42,
            ties_log_path=ties_path,
        )
        
        assert classifier.label_tensor.shape == (10, 4, 8)
        assert classifier.score_tensor.shape == (10, 4, 8, 5)
        assert len(classifier.step_index) == 0
    
    def test_record_and_classify(self, workspace_tmpdir):
        """Test recording scores and classification."""
        ties_path = workspace_tmpdir / "ties.csv"
        
        classifier = HeadClassifier(
            n_checkpoints=5,
            n_layers=2,
            n_heads=4,
            seed=42,
            ties_log_path=ties_path,
        )
        
        classifier.register_step(0)
        
        # Record a clear sink head
        scores = (0.9, 0.2, 0.1, 0.3, 0.1)
        label = classifier.record(
            ckpt_idx=0,
            step=0,
            layer=0,
            head=0,
            scores=scores,
        )
        
        assert label == LABEL_SINK
        assert classifier.label_tensor[0, 0, 0] == LABEL_SINK
        assert torch.allclose(
            classifier.score_tensor[0, 0, 0],
            torch.tensor(scores, dtype=torch.float32),
        )
    
    def test_tie_logging(self, workspace_tmpdir):
        """Test that ties are logged correctly."""
        ties_path = workspace_tmpdir / "ties.csv"
        
        classifier = HeadClassifier(
            n_checkpoints=5,
            n_layers=2,
            n_heads=4,
            seed=42,
            ties_log_path=ties_path,
            tie_tolerance=0.05,
        )
        
        classifier.register_step(100)
        
        # Record a tie
        scores = (0.42, 0.52, 0.1, 0.3, 0.1)
        label = classifier.record(
            ckpt_idx=0,
            step=100,
            layer=1,
            head=2,
            scores=scores,
        )
        
        assert label == LABEL_UNDIFF
        assert len(classifier._ties) == 1
        
        # Flush and check file
        classifier.flush_ties()
        assert ties_path.exists()
        
        # Read and verify
        with open(ties_path) as f:
            lines = f.readlines()
            assert len(lines) == 2  # Header + 1 tie
            assert "run_seed" in lines[0]
            assert "42" in lines[1]
            assert "100" in lines[1]
    
    def test_save_and_load(self, workspace_tmpdir):
        """Test saving and loading classifier state."""
        ties_path = workspace_tmpdir / "ties.csv"
        output_path = workspace_tmpdir / "results.pt"
        
        # Create and populate classifier
        classifier = HeadClassifier(
            n_checkpoints=3,
            n_layers=2,
            n_heads=4,
            seed=42,
            ties_log_path=ties_path,
        )
        
        for ckpt in range(3):
            classifier.register_step(ckpt * 100)
            for layer in range(2):
                for head in range(4):
                    scores = (
                        0.5 + 0.1 * head,
                        0.3,
                        0.2,
                        0.4,
                        0.1,
                    )
                    classifier.record(ckpt, ckpt * 100, layer, head, scores)
        
        # Save
        classifier.save(output_path)
        assert output_path.exists()
        
        # Load
        loaded = HeadClassifier.load(output_path)
        
        assert loaded["seed"] == 42
        assert loaded["n_layers"] == 2
        assert loaded["n_heads"] == 4
        assert len(loaded["step_index"]) == 3
        assert loaded["label_tensor"].shape == (3, 2, 4)
        assert loaded["score_tensor"].shape == (3, 2, 4, 5)
        assert "raw_thresholds" in loaded
        assert "effective_thresholds" in loaded
        assert "thresholds_sanitized" in loaded
    
    def test_custom_thresholds(self, workspace_tmpdir):
        """Test classifier with custom thresholds."""
        ties_path = workspace_tmpdir / "ties.csv"
        custom_thresholds = np.array([0.3, 0.4, 0.2, 0.6, 0.2])
        
        classifier = HeadClassifier(
            n_checkpoints=2,
            n_layers=1,
            n_heads=2,
            seed=42,
            ties_log_path=ties_path,
            thresholds=custom_thresholds,
        )
        
        assert np.allclose(classifier.raw_thresholds, custom_thresholds)
        assert np.allclose(classifier.effective_thresholds, custom_thresholds)
        
        classifier.register_step(0)
        scores = (0.35, 0.2, 0.1, 0.3, 0.1)
        label = classifier.record(0, 0, 0, 0, scores)
        
        # With custom thresholds, 0.35 > 0.3 so should be SINK
        assert label == LABEL_SINK

    def test_threshold_sanitization_is_persisted(self, workspace_tmpdir):
        """Sanitized thresholds should be saved with both raw and effective values."""
        ties_path = workspace_tmpdir / "ties.csv"
        output_path = workspace_tmpdir / "results.pt"
        thresholds = np.array([0.4, 0.0, 0.3, -0.2, 0.3], dtype=np.float32)

        classifier = HeadClassifier(
            n_checkpoints=1,
            n_layers=1,
            n_heads=1,
            seed=42,
            ties_log_path=ties_path,
            thresholds=thresholds,
        )
        classifier.register_step(0)
        classifier.record(0, 0, 0, 0, (0.5, 0.1, 0.1, 0.2, 0.1))
        classifier.save(output_path)

        loaded = HeadClassifier.load(output_path)
        assert loaded["thresholds_sanitized"] is True
        assert loaded["raw_thresholds"][1] == pytest.approx(0.0)
        assert loaded["raw_thresholds"][3] == pytest.approx(-0.2)
        assert loaded["effective_thresholds"][1] > 0.0
        assert loaded["effective_thresholds"][3] > 0.0


class TestHeadTypes:
    """Test HEAD_TYPES constants."""
    
    def test_head_types_length(self):
        """Test that HEAD_TYPES has 6 entries."""
        assert len(HEAD_TYPES) == 6
    
    def test_head_types_order(self):
        """Test that HEAD_TYPES is in the correct order."""
        assert HEAD_TYPES[0] == "UNDIFFERENTIATED"
        assert HEAD_TYPES[1] == "SINK"
        assert HEAD_TYPES[2] == "PREV_TOKEN"
        assert HEAD_TYPES[3] == "INDUCTION"
        assert HEAD_TYPES[4] == "POSITIONAL"
        assert HEAD_TYPES[5] == "SEMANTIC"
    
    def test_label_constants(self):
        """Test that label constants match indices."""
        assert LABEL_UNDIFF == 0
        assert LABEL_SINK == 1
        assert LABEL_PREV == 2
        assert LABEL_IND == 3
        assert LABEL_POS == 4
        assert LABEL_SEM == 5


class TestThresholds:
    """Test THRESHOLDS constants."""
    
    def test_thresholds_length(self):
        """Test that THRESHOLDS has 5 entries."""
        assert len(THRESHOLDS) == 5
    
    def test_thresholds_values(self):
        """Test that threshold values are reasonable."""
        assert all(0.0 < t < 1.0 for t in THRESHOLDS)
    
    def test_thresholds_order(self):
        """Test that thresholds correspond to score order."""
        # Order: sink, prev_token, induction, positional, semantic
        assert THRESHOLDS[0] == 0.4  # sink
        assert THRESHOLDS[1] == 0.5  # prev_token
        assert THRESHOLDS[2] == 0.3  # induction
        assert THRESHOLDS[3] == 0.7  # positional
        assert THRESHOLDS[4] == 0.3  # semantic


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
