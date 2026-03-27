#!/usr/bin/env python3
"""
Tie Tolerance Sensitivity Analysis

Tests whether TIE_TOLERANCE = 0.05 is appropriate by analyzing:
1. Distribution of (z_top1 - z_top2) gaps in real data
2. How many heads are affected by different tolerance values
3. Whether 0.05 is a tie-breaker (good) or inflater (bad)

Usage:
    python analysis/tie_tolerance_sensitivity.py <results_path>
    
Example:
    python analysis/tie_tolerance_sensitivity.py pilot_artifacts/production_15m/results/results_production_15m.pt
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from probing.classifier import HeadClassifier, HEAD_TYPES


def analyze_score_gaps(
    score_tensor: torch.Tensor,
    thresholds: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Analyze the distribution of gaps between top-2 normalized scores.
    
    Args:
        score_tensor: (K, L, H, 5) - scores for all heads at all checkpoints
        thresholds: (5,) - calibrated thresholds
        
    Returns:
        Dictionary with gap statistics
    """
    K, L, H, _ = score_tensor.shape
    
    # Normalize scores
    z_scores = score_tensor.numpy() / thresholds.reshape(1, 1, 1, 5)  # (K, L, H, 5)
    
    # For each head at each checkpoint, compute gap between top 2 z-scores
    gaps = []
    top1_values = []
    top2_values = []
    checkpoint_indices = []
    head_coords = []  # (layer, head) tuples
    
    for k in range(K):
        for l in range(L):
            for h in range(H):
                z = z_scores[k, l, h, :]  # (5,)
                sorted_z = np.sort(z)[::-1]  # Descending
                
                top1 = sorted_z[0]
                top2 = sorted_z[1]
                gap = top1 - top2
                
                gaps.append(gap)
                top1_values.append(top1)
                top2_values.append(top2)
                checkpoint_indices.append(k)
                head_coords.append((l, h))
    
    return {
        'gaps': np.array(gaps),
        'top1': np.array(top1_values),
        'top2': np.array(top2_values),
        'checkpoint_idx': np.array(checkpoint_indices),
        'head_coords': head_coords,
    }


def classify_with_tolerance(
    score_tensor: torch.Tensor,
    thresholds: np.ndarray,
    tolerance: float,
) -> Tuple[np.ndarray, int]:
    """
    Classify all heads with a specific tie tolerance.
    
    Returns:
        (labels, n_ties) where labels is (K, L, H) and n_ties is count of tie events
    """
    K, L, H, _ = score_tensor.shape
    labels = np.zeros((K, L, H), dtype=np.int32)
    n_ties = 0
    
    z_scores = score_tensor.numpy() / thresholds.reshape(1, 1, 1, 5)
    
    for k in range(K):
        for l in range(L):
            for h in range(H):
                scores = score_tensor[k, l, h, :].numpy()
                z = z_scores[k, l, h, :]
                
                # Check if all below threshold
                if np.all(scores < thresholds):
                    labels[k, l, h] = 0  # UNDIFFERENTIATED
                    continue
                
                # Check for tie
                sorted_z = np.sort(z)[::-1]
                if (sorted_z[0] - sorted_z[1]) < tolerance:
                    labels[k, l, h] = 0  # UNDIFFERENTIATED (tie)
                    n_ties += 1
                    continue
                
                # Assign argmax
                labels[k, l, h] = int(np.argmax(z)) + 1
    
    return labels, n_ties


def compute_label_distribution(labels: np.ndarray) -> Dict[str, float]:
    """Compute fraction of each label type."""
    total = labels.size
    distribution = {}
    for label_idx, label_name in enumerate(HEAD_TYPES):
        count = (labels == label_idx).sum()
        distribution[label_name] = count / total
    return distribution


def main():
    if len(sys.argv) < 2:
        print("Usage: python analysis/tie_tolerance_sensitivity.py <results_path>")
        print("\nExample:")
        print("  python analysis/tie_tolerance_sensitivity.py pilot_artifacts/production_15m/results/results_production_15m.pt")
        sys.exit(1)
    
    results_path = Path(sys.argv[1])
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        sys.exit(1)
    
    print(f"Loading results from: {results_path}")
    results = HeadClassifier.load(results_path)
    
    score_tensor = results["score_tensor"]  # (K, L, H, 5)
    thresholds = np.asarray(
        results.get("effective_thresholds", results.get("thresholds")),
        dtype=np.float32,
    )
    steps = results["step_index"]           # List of training steps
    
    K, L, H, _ = score_tensor.shape
    print(f"\nDataset: {K} checkpoints, {L} layers, {H} heads/layer")
    print(f"Total head-checkpoint pairs: {K * L * H}")
    print(f"\nCalibrated thresholds: {thresholds}")
    
    # =========================================================================
    # Analysis 1: Distribution of z-score gaps
    # =========================================================================
    print("\n" + "="*80)
    print("ANALYSIS 1: Distribution of (z_top1 - z_top2) gaps")
    print("="*80)
    
    gap_stats = analyze_score_gaps(score_tensor, thresholds)
    gaps = gap_stats['gaps']
    
    print(f"\nGap statistics:")
    print(f"  Mean:       {gaps.mean():.4f}")
    print(f"  Median:     {np.median(gaps):.4f}")
    print(f"  Std:        {gaps.std():.4f}")
    print(f"  Min:        {gaps.min():.4f}")
    print(f"  Max:        {gaps.max():.4f}")
    print(f"\nPercentiles:")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"  {p:2d}th:      {np.percentile(gaps, p):.4f}")
    
    # What fraction of gaps are below various thresholds?
    print(f"\nFraction of gaps below threshold:")
    for tol in [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]:
        frac = (gaps < tol).mean()
        print(f"  < {tol:.2f}:  {frac:.2%}  ({int(frac * len(gaps))} / {len(gaps)})")
    
    # =========================================================================
    # Analysis 2: Sensitivity to tolerance value
    # =========================================================================
    print("\n" + "="*80)
    print("ANALYSIS 2: Sensitivity to tie tolerance")
    print("="*80)
    
    tolerances = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0]
    results_by_tolerance = {}
    
    print(f"\n{'Tolerance':<12} {'Ties':<10} {'UNDIFF %':<12} {'SINK %':<10} {'PREV %':<10} {'IND %':<10} {'POS %':<10} {'SEM %':<10}")
    print("-" * 90)
    
    for tol in tolerances:
        labels, n_ties = classify_with_tolerance(score_tensor, thresholds, tol)
        dist = compute_label_distribution(labels)
        results_by_tolerance[tol] = {
            'labels': labels,
            'n_ties': n_ties,
            'distribution': dist,
        }
        
        print(f"{tol:<12.2f} {n_ties:<10d} ", end="")
        for label_name in HEAD_TYPES:
            print(f"{dist[label_name]:<10.1%} ", end="")
        print()
    
    # =========================================================================
    # Analysis 3: Visualizations
    # =========================================================================
    print("\n" + "="*80)
    print("ANALYSIS 3: Generating visualizations")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Histogram of gaps
    ax = axes[0, 0]
    ax.hist(gaps, bins=100, alpha=0.7, edgecolor='black')
    ax.axvline(0.05, color='red', linestyle='--', linewidth=2, label='Current tolerance (0.05)')
    ax.axvline(np.median(gaps), color='green', linestyle='--', linewidth=2, label=f'Median ({np.median(gaps):.3f})')
    ax.set_xlabel('Gap (z_top1 - z_top2)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Z-Score Gaps')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 2: Cumulative distribution
    ax = axes[0, 1]
    sorted_gaps = np.sort(gaps)
    cumulative = np.arange(1, len(sorted_gaps) + 1) / len(sorted_gaps)
    ax.plot(sorted_gaps, cumulative, linewidth=2)
    ax.axvline(0.05, color='red', linestyle='--', linewidth=2, label='Current tolerance (0.05)')
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Gap (z_top1 - z_top2)')
    ax.set_ylabel('Cumulative Fraction')
    ax.set_title('Cumulative Distribution of Gaps')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 3: UNDIFFERENTIATED fraction vs tolerance
    ax = axes[1, 0]
    undiff_fracs = [results_by_tolerance[tol]['distribution']['UNDIFFERENTIATED'] for tol in tolerances]
    ax.plot(tolerances, undiff_fracs, marker='o', linewidth=2, markersize=8)
    ax.axvline(0.05, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Current (0.05)')
    ax.set_xlabel('Tie Tolerance')
    ax.set_ylabel('Fraction UNDIFFERENTIATED')
    ax.set_title('UNDIFFERENTIATED Fraction vs Tolerance')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 4: Number of ties vs tolerance
    ax = axes[1, 1]
    tie_counts = [results_by_tolerance[tol]['n_ties'] for tol in tolerances]
    ax.plot(tolerances, tie_counts, marker='o', linewidth=2, markersize=8, color='orange')
    ax.axvline(0.05, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Current (0.05)')
    ax.set_xlabel('Tie Tolerance')
    ax.set_ylabel('Number of Tie Events')
    ax.set_title('Tie Events vs Tolerance')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_dir = results_path.parent.parent / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'tie_tolerance_sensitivity.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\nSaved figure to: {output_path}")
    
    # =========================================================================
    # Analysis 4: Interpretation
    # =========================================================================
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    
    median_gap = np.median(gaps)
    frac_below_005 = (gaps < 0.05).mean()
    frac_below_010 = (gaps < 0.10).mean()
    
    current_undiff = results_by_tolerance[0.05]['distribution']['UNDIFFERENTIATED']
    zero_tol_undiff = results_by_tolerance[0.0]['distribution']['UNDIFFERENTIATED']
    
    print(f"\nMedian gap: {median_gap:.4f}")
    print(f"Fraction of gaps < 0.05: {frac_below_005:.2%}")
    print(f"Fraction of gaps < 0.10: {frac_below_010:.2%}")
    
    print(f"\nUNDIFFERENTIATED fraction:")
    print(f"  With tolerance=0.00: {zero_tol_undiff:.2%}")
    print(f"  With tolerance=0.05: {current_undiff:.2%}")
    print(f"  Difference: {(current_undiff - zero_tol_undiff):.2%}")
    
    if median_gap > 0.10:
        print("\n✓ VERDICT: 0.05 is a GOOD tie-breaker")
        print("  - Median gap is large (>0.10), so most heads have clear winners")
        print("  - 0.05 catches only genuinely ambiguous cases")
    elif median_gap > 0.05:
        print("\n✓ VERDICT: 0.05 is REASONABLE")
        print("  - Median gap is moderate (0.05-0.10)")
        print("  - 0.05 catches close calls without being too aggressive")
    elif median_gap < 0.05:
        print("\n⚠ VERDICT: 0.05 may be an INFLATER")
        print("  - Median gap is small (<0.05)")
        print("  - 0.05 may be marking too many heads as UNDIFFERENTIATED")
        print("  - Consider reducing to 0.02 or 0.01")
    
    if frac_below_005 > 0.5:
        print(f"\n⚠ WARNING: {frac_below_005:.1%} of gaps are below 0.05")
        print("  This suggests many heads have ambiguous behaviors")
        print("  0.05 may be appropriate, but verify this is expected")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == '__main__':
    main()
