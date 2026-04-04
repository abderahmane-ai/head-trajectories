"""
visualization/heatmap.py — Layer × checkpoint heatmap of head type specialization.

Produces Figure 2: a grid of (training_step × layer) where each cell is colored
by the dominant head type in that layer at that step. This is the primary
visualization for H3 (layer stratification hypothesis) — if lower layers
specialize earlier, you expect a wave of color propagating upward over time.

Two variants:
  1. Dominant type heatmap — one color per cell = most common non-UNDIFF type
  2. Specialization fraction heatmap — one value per cell = 1 - UNDIFF fraction
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from probing import HEAD_TYPE_COLORS, HEAD_TYPES

matplotlib.rcParams.update({
    "font.family":     "serif",
    "font.size":       10,
    "axes.labelsize":  11,
    "axes.titlesize":  12,
    "figure.dpi":      300,
    "savefig.dpi":     300,
    "savefig.bbox":    "tight",
})

# Colors matching timeline_plot.py — indexed by label (0..5)
LEGACY_TYPE_COLORS_HEX: List[str] = [HEAD_TYPE_COLORS[name] for name in HEAD_TYPES]


def _dominant_type_per_cell(
    per_layer_mean: np.ndarray,
) -> np.ndarray:
    """
    For each (layer, ckpt) cell, return the index of the most common
    non-UNDIFF head type. Returns 0 (UNDIFF) if no non-UNDIFF type
    exceeds 15% of heads in that layer at that step.

    Args:
        per_layer_mean: (n_layers, n_ckpts, 6) — mean fractions

    Returns:
        dominant: (n_layers, n_ckpts) int — type label index
    """

    n_layers, n_ckpts, n_types = per_layer_mean.shape
    dominant = np.zeros((n_layers, n_ckpts), dtype=np.int32)

    for layer in range(n_layers):
        for ckpt in range(n_ckpts):
            fracs = per_layer_mean[layer, ckpt]   # (6,)

    # Mask out non-specialized labels
            non_undiff_fracs        = fracs.copy()
            non_undiff_fracs[0:2]   = -1.0 if len(non_undiff_fracs) > 1 else -1.0

            best_type  = int(np.argmax(non_undiff_fracs))
            best_frac  = fracs[best_type]

            dominant[layer, ckpt] = best_type if best_frac >= 0.15 else 0

    return dominant


def plot_dominant_type_heatmap(
    per_layer_curves: Dict[str, object],
    output_path:      Path,
    title:            str = "Dominant Head Type by Layer and Training Step",
) -> None:
    """
    Plot the dominant head type heatmap.

    X-axis: training step (log scale)
    Y-axis: layer index (0 = bottom)
    Color:  dominant non-UNDIFF head type in that layer at that step

    Args:
        per_layer_curves: output of compute_per_layer_curves
        output_path:      path to save the PNG figure
        title:            figure title
    """

    steps          = per_layer_curves["steps"]           # (n_ckpts,)
    per_layer_mean = per_layer_curves["per_layer_mean"]  # (n_layers, n_ckpts, 6)
    n_layers       = per_layer_curves["n_layers"]
    type_names     = per_layer_curves["type_names"]

    dominant = _dominant_type_per_cell(per_layer_mean)   # (n_layers, n_ckpts)

    # Build a custom colormap from our type colors
    type_colors_hex = [HEAD_TYPE_COLORS.get(name, "#333333") for name in type_names]
    cmap   = mcolors.ListedColormap(type_colors_hex)
    bounds = np.arange(-0.5, len(type_names) + 0.5, 1.0)
    norm   = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(10, 4))

    # pcolormesh with log x-axis — need to compute log-spaced bin edges
    # Use step midpoints for cell centers, extend edges at boundaries
    log_steps = np.log10(np.maximum(steps, 1))
    step_edges = np.concatenate([
        [log_steps[0] - (log_steps[1] - log_steps[0]) / 2],
        (log_steps[:-1] + log_steps[1:]) / 2,
        [log_steps[-1] + (log_steps[-1] - log_steps[-2]) / 2],
    ])

    layer_edges = np.arange(-0.5, n_layers + 0.5)

    # dominant: (n_layers, n_ckpts) — rows = layers, cols = steps
    # pcolormesh expects (rows+1, cols+1) edges
    mesh = ax.pcolormesh(
        step_edges,
        layer_edges,
        dominant,
        cmap=cmap,
        norm=norm,
        shading="flat",
    )

    # X-axis: restore actual step values
    ax.set_xlabel("Training step", fontsize=11)
    ax.set_ylabel("Layer", fontsize=11)
    ax.set_title(title, fontsize=12, pad=8)

    # Tick positions at key training steps
    key_steps = [s for s in [0, 100, 500, 1000, 5000, 10000, 50000, 100000]
                 if s <= steps[-1] and s >= steps[0]]
    tick_pos  = [np.log10(max(s, 1)) for s in key_steps]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([f"{s:,}" for s in key_steps], rotation=30, ha="right")

    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"L{i}" for i in range(n_layers)])

    # Legend patches — one per type
    patches = [
        mpatches.Patch(
            facecolor=type_colors_hex[t],
            label=type_names[t].replace("_", " ").title(),
            edgecolor="#888888",
            linewidth=0.5,
        )
        for t in range(len(type_names))
    ]
    ax.legend(
        handles=patches,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=True,
        framealpha=0.9,
        edgecolor="#CCCCCC",
        fontsize=9,
        title="Head type",
        title_fontsize=9,
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"  [Figure] Dominant type heatmap saved → {output_path}")


def plot_specialization_fraction_heatmap(
    per_layer_curves: Dict[str, object],
    output_path:      Path,
    title:            str = "Fraction of Specialized Heads by Layer and Training Step",
) -> None:
    """
    Plot a continuous heatmap showing 1 - UNDIFF fraction per (layer, step).

    Warmer colors = more specialized. This is a cleaner visualization for
    testing H3 since it shows the wave of specialization propagating through layers.

    Args:
        per_layer_curves: output of compute_per_layer_curves
        output_path:      path to save the PNG figure
        title:            figure title
    """

    steps          = per_layer_curves["steps"]
    per_layer_mean = per_layer_curves["per_layer_mean"]   # (n_layers, n_ckpts, 6)
    n_layers       = per_layer_curves["n_layers"]

    # Specialization = 1 - (WEAK + AMBIGUOUS) fraction for the new schema.
    nonspecialized = per_layer_mean[:, :, :2].sum(axis=-1) if per_layer_mean.shape[-1] >= 2 else per_layer_mean[:, :, 0]
    spec_frac = 1.0 - nonspecialized

    log_steps  = np.log10(np.maximum(steps, 1))
    step_edges = np.concatenate([
        [log_steps[0] - (log_steps[1] - log_steps[0]) / 2],
        (log_steps[:-1] + log_steps[1:]) / 2,
        [log_steps[-1] + (log_steps[-1] - log_steps[-2]) / 2],
    ])
    layer_edges = np.arange(-0.5, n_layers + 0.5)

    fig, ax = plt.subplots(figsize=(10, 3.5))

    mesh = ax.pcolormesh(
        step_edges,
        layer_edges,
        spec_frac,
        cmap="YlOrRd",
        vmin=0.0,
        vmax=1.0,
        shading="flat",
    )

    cbar = fig.colorbar(mesh, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Fraction specialized (non-UNDIFF)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_xlabel("Training step", fontsize=11)
    ax.set_ylabel("Layer", fontsize=11)
    ax.set_title(title, fontsize=12, pad=8)

    key_steps = [s for s in [0, 100, 500, 1000, 5000, 10000, 50000, 100000]
                 if s <= steps[-1] and s >= steps[0]]
    tick_pos  = [np.log10(max(s, 1)) for s in key_steps]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([f"{s:,}" for s in key_steps], rotation=30, ha="right")
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"L{i}" for i in range(n_layers)])

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"  [Figure] Specialization fraction heatmap saved → {output_path}")
