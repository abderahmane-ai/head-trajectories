"""
visualization/timeline_plot.py — Main developmental timeline figure.

Produces the paper's lead figure: dominance curves showing the fraction of all
attention heads classified as each type vs. training step, averaged across
seeds with ±1 std confidence bands.

This is Figure 1 of the paper. It must be publication-quality at 300 DPI,
with proper axis labels, a clean legend, and log-scaled x-axis to show
the dense early checkpoints clearly.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from probing import HEAD_TYPE_COLORS

matplotlib.rcParams.update({
    "font.family":        "serif",
    "font.size":          11,
    "axes.labelsize":     12,
    "axes.titlesize":     13,
    "legend.fontsize":    10,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
})

# Color palette — one per head type (index-aligned with HEAD_TYPES)
# `UNDIFFERENTIATED` is kept only for legacy-result compatibility and is deprecated.
DEEMPHASIZED_TYPES = {"WEAK", "AMBIGUOUS", "UNDIFFERENTIATED"}
TYPE_COLORS: List[str] = [HEAD_TYPE_COLORS.get(name, "#333333") for name in HEAD_TYPE_COLORS]


def plot_timeline(
    global_curves:  Dict[str, np.ndarray],
    output_path:    Path,
    onset_steps:    Optional[Dict[str, Optional[int]]] = None,
    log_x:          bool = True,
    show_undiff:    bool = True,
    title:          str  = "Developmental Trajectories of Attention Heads",
) -> None:
    """
    Plot the main developmental timeline figure.

    Args:
        global_curves:  output of compute_global_curves — contains steps,
                        mean (n_ckpts, n_types), std (n_ckpts, n_types), type_names
        output_path:    path to save the PNG figure
        onset_steps:    optional dict mapping type_name → onset step;
                        if provided, vertical dashed markers are drawn
        log_x:          if True, use log scale on x-axis (recommended)
        show_undiff:    if True, include non-specialized curves (`WEAK`, `AMBIGUOUS`, legacy `UNDIFFERENTIATED`)
        title:          figure title string
    """

    steps      = global_curves["steps"]         # (n_ckpts,)
    mean       = global_curves["mean"]          # (n_ckpts, 6)
    std        = global_curves["std"]           # (n_ckpts, 6)
    type_names = global_curves["type_names"]
    n_seeds    = global_curves["per_seed"].shape[0]

    fig, ax = plt.subplots(figsize=(8, 5))

    for t_idx, type_name in enumerate(type_names):
        if not show_undiff and type_name in DEEMPHASIZED_TYPES:
            continue

        y      = mean[:, t_idx]
        y_std  = std[:, t_idx]
        color  = HEAD_TYPE_COLORS.get(type_name, "#333333")
        ls     = "--" if type_name in DEEMPHASIZED_TYPES else "-"
        lw     = 1.2 if type_name in DEEMPHASIZED_TYPES else (2.2 if type_name == "INDUCTION" else 2.0)
        label  = type_name.replace("_", " ").title()

        ax.plot(steps, y, color=color, lw=lw, ls=ls, label=label, zorder=3)

        # Confidence band (only meaningful with > 1 seed)
        if n_seeds > 1:
            ax.fill_between(
                steps,
                np.clip(y - y_std, 0, 1),
                np.clip(y + y_std, 0, 1),
                color=color,
                alpha=0.12,
                zorder=2,
            )

    # Onset markers — vertical lines at first emergence of each type
    if onset_steps is not None:
        for type_name, step in onset_steps.items():
            if step is None or type_name in DEEMPHASIZED_TYPES:
                continue
            t_idx = type_names.index(type_name) if type_name in type_names else -1
            if t_idx < 0:
                continue
            color = HEAD_TYPE_COLORS.get(type_name, "#333333")
            ax.axvline(
                x=step, color=color, lw=0.8, ls=":", alpha=0.6, zorder=1
            )

    # Axes formatting
    ax.set_xlabel("Training step", fontsize=12)
    ax.set_ylabel("Fraction of all attention heads", fontsize=12)
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_ylim(-0.02, 1.05)

    if log_x and steps[0] > 0:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
        )
    else:
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
        )

    # Legend — place outside plot area to avoid overlap
    legend = ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=True,
        framealpha=0.9,
        edgecolor="#CCCCCC",
        title=f"Head type\n(n={n_seeds} seeds)",
        title_fontsize=9,
    )

    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(axis="y", lw=0.4, alpha=0.4, zorder=0)

    # Annotation: note about confidence bands
    if n_seeds > 1:
        ax.annotate(
            f"Shaded regions: ±1 std across {n_seeds} seeds",
            xy=(0.01, 0.01), xycoords="axes fraction",
            fontsize=8, color="#888888",
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"  [Figure] Timeline plot saved → {output_path}")


def plot_timeline_per_seed(
    global_curves: Dict[str, np.ndarray],
    output_path:   Path,
) -> None:
    """
    Supplementary figure: individual seed curves without averaging,
    one subplot per seed. Shows raw inter-seed variance.

    Args:
        global_curves: output of compute_global_curves
        output_path:   path to save the PNG figure
    """

    steps      = global_curves["steps"]
    per_seed   = global_curves["per_seed"]   # (n_seeds, n_ckpts, 6)
    type_names = global_curves["type_names"]
    n_seeds    = per_seed.shape[0]

    fig, axes = plt.subplots(
        1, n_seeds,
        figsize=(5 * n_seeds, 4),
        sharey=True,
    )
    if n_seeds == 1:
        axes = [axes]

    for s_idx, ax in enumerate(axes):
        for t_idx, type_name in enumerate(type_names):
            if type_name in DEEMPHASIZED_TYPES:
                continue
            y     = per_seed[s_idx, :, t_idx]
            color = HEAD_TYPE_COLORS.get(type_name, "#333333")
            label = type_name.replace("_", " ").title()
            ax.plot(steps, y, color=color, lw=1.8, label=label)

        ax.set_title(f"Seed {s_idx}", fontsize=11)
        ax.set_xlabel("Training step", fontsize=10)
        ax.set_xscale("log")
        ax.set_ylim(-0.02, 1.05)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))
        ax.grid(axis="y", lw=0.4, alpha=0.4)
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
        )

    axes[0].set_ylabel("Fraction of heads", fontsize=10)

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=5,
        frameon=True,
        fontsize=9,
    )

    fig.suptitle("Per-seed developmental trajectories", fontsize=12, y=1.08)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"  [Figure] Per-seed timeline saved → {output_path}")
