"""
Test tie tolerance behavior with synthetic score distributions.

This test validates that TIE_TOLERANCE = 0.05 is appropriate by:
1. Creating synthetic score distributions with known gap patterns
2. Testing classification behavior at different tolerances
3. Verifying that 0.05 catches genuine ambiguity without over-inflating UNDIFF
"""

import numpy as np
import pytest
import torch

from probing.classifier import classify_head, LABEL_AMBIGUOUS, LABEL_SINK, LABEL_WEAK


def test_tie_tolerance_clear_winner():
    """Test that clear winners are not affected by tie tolerance."""
    thresholds = np.array([0.4, 0.5, 0.3, 0.7, 0.3], dtype=np.float32)
    
    # Clear SINK winner: z_sink = 2.5, z_prev = 0.5
    scores = (1.0, 0.25, 0.1, 0.2, 0.1)  # sink=1.0/0.4=2.5, prev=0.25/0.5=0.5
    
    # Should classify as SINK regardless of tolerance
    for tolerance in [0.01, 0.05, 0.10, 0.20]:
        label, is_tie = classify_head(scores, thresholds, tie_tolerance=tolerance)
        assert label == LABEL_SINK, f"Expected SINK ({LABEL_SINK}), got {label} with tolerance={tolerance}"
        assert not is_tie


def test_tie_tolerance_genuine_tie():
    """Test that genuine ties are caught by 0.05 tolerance."""
    thresholds = np.array([0.4, 0.5, 0.3, 0.7, 0.3], dtype=np.float32)
    
    # Genuine tie: z_sink = 2.0, z_prev = 1.95 (gap = 0.05)
    scores = (0.8, 0.975, 0.1, 0.2, 0.1)  # sink=0.8/0.4=2.0, prev=0.975/0.5=1.95
    
    # Should be UNDIFFERENTIATED with tolerance >= 0.05
    label, is_tie = classify_head(scores, thresholds, tie_tolerance=0.05)
    assert label == LABEL_AMBIGUOUS, f"Expected AMBIGUOUS, got {label}"
    assert is_tie, "Expected is_tie=True"
    
    # Should classify as SINK with tolerance < 0.05
    label, is_tie = classify_head(scores, thresholds, tie_tolerance=0.04)
    assert label == LABEL_SINK, f"Expected SINK ({LABEL_SINK}), got {label}"
    assert not is_tie


def test_tie_tolerance_near_tie():
    """Test behavior near the tie boundary."""
    thresholds = np.array([0.4, 0.5, 0.3, 0.7, 0.3], dtype=np.float32)
    
    # Near tie: z_sink = 2.0, z_prev = 1.90 (gap = 0.10)
    scores = (0.8, 0.95, 0.1, 0.2, 0.1)  # sink=0.8/0.4=2.0, prev=0.95/0.5=1.90
    
    # Should classify as SINK with tolerance = 0.05 (gap > tolerance)
    label, is_tie = classify_head(scores, thresholds, tie_tolerance=0.05)
    assert label == LABEL_SINK, f"Expected SINK ({LABEL_SINK}), got {label}"
    assert not is_tie
    
    # Should be UNDIFFERENTIATED with tolerance = 0.15 (gap < tolerance)
    label, is_tie = classify_head(scores, thresholds, tie_tolerance=0.15)
    assert label == LABEL_AMBIGUOUS, f"Expected AMBIGUOUS, got {label}"
    assert is_tie


def test_tie_tolerance_distribution_analysis():
    """
    Test with realistic score distributions to see if 0.05 is appropriate.
    
    This simulates what happens in real training:
    - Early: many heads below threshold (UNDIFF)
    - Mid: some clear specialization, some ambiguous
    - Late: mostly clear specialization
    """
    np.random.seed(42)
    thresholds = np.array([0.4, 0.5, 0.3, 0.7, 0.3], dtype=np.float32)
    
    # Simulate 1000 heads with different gap distributions
    n_heads = 1000
    
    # Generate scores where top-2 gaps follow a realistic distribution
    # Most heads have clear winners (gap > 0.2), some are ambiguous (gap < 0.1)
    gaps = np.concatenate([
        np.random.uniform(0.3, 1.0, size=700),   # 70% clear winners
        np.random.uniform(0.1, 0.3, size=200),   # 20% moderate gaps
        np.random.uniform(0.0, 0.1, size=100),   # 10% ambiguous
    ])
    np.random.shuffle(gaps)
    
    # Generate synthetic scores with these gaps
    results_by_tolerance = {}
    
    for tolerance in [0.0, 0.05, 0.10, 0.20]:
        n_undiff = 0
        n_ties = 0
        
        for gap in gaps:
            # Create scores where top-2 have specified gap
            top1_z = np.random.uniform(1.5, 3.0)  # Above threshold
            top2_z = top1_z - gap
            other_z = np.random.uniform(0.0, 0.5, size=3)  # Below threshold
            
            # Convert z-scores back to raw scores
            z_scores = np.array([top1_z, top2_z, *other_z])
            np.random.shuffle(z_scores)  # Randomize which metric is which
            raw_scores = z_scores * thresholds
            
            label, is_tie = classify_head(tuple(raw_scores), thresholds, tie_tolerance=tolerance)
            
            if label == LABEL_WEAK:
                n_undiff += 1
            if label == LABEL_AMBIGUOUS:
                n_ties += 1
        
        results_by_tolerance[tolerance] = {
            'n_undiff': n_undiff,
            'n_ties': n_ties,
            'frac_undiff': n_undiff / n_heads,
            'frac_ties': n_ties / n_heads,
        }
    
    # Print results for inspection
    print("\n" + "="*60)
    print("Tie Tolerance Sensitivity Test")
    print("="*60)
    print(f"{'Tolerance':<12} {'UNDIFF':<10} {'Ties':<10} {'% UNDIFF':<12} {'% Ties':<12}")
    print("-"*60)
    for tol in [0.0, 0.05, 0.10, 0.20]:
        r = results_by_tolerance[tol]
        print(f"{tol:<12.2f} {r['n_undiff']:<10d} {r['n_ties']:<10d} {r['frac_undiff']:<12.1%} {r['frac_ties']:<12.1%}")
    
    # Assertions: 0.05 should catch ~10% as ties (the ambiguous ones)
    # without over-inflating UNDIFF
    r_005 = results_by_tolerance[0.05]
    r_000 = results_by_tolerance[0.0]
    
    # With 0.05 tolerance, we should catch most of the 10% ambiguous cases
    assert 0.05 <= r_005['frac_ties'] <= 0.15, \
        f"Expected ~10% ties with tolerance=0.05, got {r_005['frac_ties']:.1%}"
    
    # The increase in UNDIFF from 0.0 to 0.05 should be modest
    increase = r_005['frac_undiff'] - r_000['frac_undiff']
    assert increase < 0.20, \
        f"Tolerance=0.05 inflates UNDIFF by {increase:.1%}, which is too much"
    
    print("\n✓ Test passed: 0.05 tolerance is appropriate")
    print(f"  - Catches {r_005['frac_ties']:.1%} of heads as genuine ties")
    print(f"  - Increases UNDIFF by only {increase:.1%} vs tolerance=0.0")


def test_tie_tolerance_edge_cases():
    """Test edge cases in tie detection."""
    thresholds = np.array([0.4, 0.5, 0.3, 0.7, 0.3], dtype=np.float32)
    
    # Edge case 1: All scores below threshold
    scores = (0.1, 0.1, 0.1, 0.1, 0.1)
    label, is_tie = classify_head(scores, thresholds, tie_tolerance=0.05)
    assert label == LABEL_WEAK
    assert not is_tie, "Should not be marked as tie if all below threshold"
    
    # Edge case 2: Exact tie (gap = 0.0)
    scores = (0.8, 1.0, 0.1, 0.2, 0.1)  # z_sink=2.0, z_prev=2.0
    label, is_tie = classify_head(scores, thresholds, tie_tolerance=0.05)
    assert label == LABEL_AMBIGUOUS
    assert is_tie
    
    # Edge case 3: Gap exactly at tolerance boundary
    scores = (0.8, 0.975, 0.1, 0.2, 0.1)  # gap = 0.05 exactly
    label, is_tie = classify_head(scores, thresholds, tie_tolerance=0.05)
    assert label == LABEL_AMBIGUOUS, "Gap exactly at boundary should be treated as tie"
    assert is_tie


if __name__ == '__main__':
    # Run the distribution analysis test with verbose output
    test_tie_tolerance_distribution_analysis()
