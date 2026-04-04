"""
visualization/stability_hist.py — Stability matrix histogram and sink persistence.

Produces Figure 4: two panels.

Panel A — Type-change histogram: distribution of the number of type-label
changes per head across all seeds, layers, and heads. Tests H5 qualitatively:
if most heads have 0 or 1 changes, heads are stable attractors.

Panel B — Sink persistence bar chart: for heads that were ever labeled SINK,
what fraction of their subsequent checkpoints remained SINK? Tests H5 directly.
A mean persistence near 1.0 strongly supports H5.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from typing import Dict, List, Optional

from probing import HEAD_TYPE_COLORS, HEAD_TYPES

matplotlib.rcParams.update({
    "font.family":     "serif",
    "font.size":       11,
    "axes.labelsize":  12,
    "axes.titlesize":  12,
    "figure.dpi":      300,
    "savefig.dpi":     300,
    "savefig.bbox":    "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

TYPE_COLORS_HEX: List[str] = [HEAD_TYPE_COLORS[name] for name in HEAD_TYPES]


def plot_stability_figure(
    hist_data:        Dict[str, np.ndarray],
    sink_persistence: Dict[str, object],
    per_type_stab:    Dict[str, Dict[str, float]],
    output_path:      Path,
    title:            str = "Head Type Stability Analysis (H5)",
) -> None:
    """
    Two-panel stability figure.

    Panel A: histogram of type-change counts per head.
    Panel B: per-type mean changes (bar chart) with sink persistence annotation.

    Args:
        hist_data:        output of compute_stability_histogram
        sink_persistence: output of compute_sink_persistence
        per_type_stab:    output of compute_per_type_stability
        output_path:      path to save the PNG figure
        title:            figure super-title
    """

    fig, (ax_hist, ax_bar) = plt.subplots(1, 2, figsize=(11, 4.5))

    # ── Panel A: Type-change histogram ───────────────────────────────────────
    flat_counts = hist_data["flat_counts"]
    bins        = hist_data["bins"]
    hist        = hist_data["hist"]
    pct_zero    = hist_data["pct_zero"]
    pct_one     = hist_data["pct_one"]
    mean_chg    = hist_data["mean"]

    bar_centers = bins[:-1]
    ax_hist.bar(
        bar_centers, hist,
        color="#7F77DD", alpha=0.75, edgecolor="white", linewidth=0.5,
        width=0.8,
    )

    ax_hist.set_xlabel("Number of type changes per head", fontsize=12)
    ax_hist.set_ylabel("Number of heads", fontsize=12)
    ax_hist.set_title("Type-change distribution", fontsize=12)

    # Integer x-ticks only
    ax_hist.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Annotations
    ax_hist.annotate(
        f"0 changes: {pct_zero * 100:.1f}%\n"
        f"1 change:  {pct_one * 100:.1f}%\n"
        f"Mean: {mean_chg:.2f}",
        xy=(0.62, 0.78), xycoords="axes fraction",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC", alpha=0.8),
    )

    # ── Panel B: Per-type mean changes + sink persistence ────────────────────
    # `UNDIFFERENTIATED` is a deprecated legacy label; exclude non-specialized states.
    non_undiff_types = [t for t in HEAD_TYPES if t not in {"WEAK", "AMBIGUOUS", "UNDIFFERENTIATED"}]
    type_indices     = [HEAD_TYPES.index(t) for t in non_undiff_types]

    means  = [per_type_stab[t]["mean_changes"] for t in non_undiff_types]
    n_vals = [per_type_stab[t]["n_heads"]      for t in non_undiff_types]
    colors = [TYPE_COLORS_HEX[i] for i in type_indices]
    labels = [t.replace("_", "\n").title() for t in non_undiff_types]

    bars = ax_bar.bar(
        range(len(non_undiff_types)), means,
        color=colors, edgecolor="white", linewidth=0.5, alpha=0.82,
    )

    # Label bars with n
    for bar_obj, n, mean_v in zip(bars, n_vals, means):
        ax_bar.text(
            bar_obj.get_x() + bar_obj.get_width() / 2,
            mean_v + 0.02,
            f"n={n}",
            ha="center", va="bottom", fontsize=8, color="#555555",
        )

    ax_bar.set_xticks(range(len(non_undiff_types)))
    ax_bar.set_xticklabels(labels, fontsize=9)
    ax_bar.set_xlabel("Final head type", fontsize=12)
    ax_bar.set_ylabel("Mean type changes during training", fontsize=12)
    ax_bar.set_title("Stability by final type", fontsize=12)

    # Sink persistence annotation box
    mean_p = sink_persistence["mean_persistence"]
    std_p  = sink_persistence["std_persistence"]
    n_sink = sink_persistence["n_ever_sink"]

    verdict_color = "#1D9E75" if mean_p >= 0.8 else (
        "#EF9F27" if mean_p >= 0.6 else "#E24B4A"
    )
    verdict_text  = "H5 supported" if mean_p >= 0.8 else (
        "H5 partial" if mean_p >= 0.6 else "H5 rejected"
    )

    ax_bar.annotate(
        f"Sink persistence:\n"
        f"{mean_p:.3f} ± {std_p:.3f}\n"
        f"({n_sink} ever-SINK heads)\n"
        f"→ {verdict_text}",
        xy=(0.98, 0.97), xycoords="axes fraction",
        fontsize=9,
        ha="right", va="top",
        color=verdict_color,
        bbox=dict(
            boxstyle="round,pad=0.4",
            fc="white",
            ec=verdict_color,
            alpha=0.85,
            linewidth=1.2,
        ),
    )

    fig.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"  [Figure] Stability figure saved → {output_path}")


def plot_individual_trajectories(
    trajectories:  Dict,
    step_index:    List[int],
    output_path:   Path,
    max_heads:     int = 12,
    title:         str = "Individual Head Developmental Trajectories",
) -> None:
    """
    Supplementary figure: plot type-label sequences for individual heads.

    Shows the most "interesting" heads — those with the most type changes.
    Each row is one head; color encodes type label at each checkpoint.

    Args:
        trajectories: output of compute_head_trajectories or
                      find_interesting_trajectories
        step_index:   list of training steps
        output_path:  path to save the PNG figure
        max_heads:    maximum number of heads to show
        title:        figure title
    """

    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches

    # Sort by number of type changes, descending
    sorted_heads = sorted(
        trajectories.items(),
        key=lambda x: sum(1 for i in range(1, len(x[1])) if x[1][i] != x[1][i-1]),
        reverse=True,
    )[:max_heads]

    if not sorted_heads:
        print("  [Figure] No trajectories to plot.")
        return

    n_heads  = len(sorted_heads)
    n_ckpts  = len(step_index)
    steps    = np.array(step_index)

    # Build color matrix
    cmap  = mcolors.ListedColormap(TYPE_COLORS_HEX)
    norm  = mcolors.BoundaryNorm(np.arange(-0.5, 6.5), cmap.N)

    label_matrix = np.zeros((n_heads, n_ckpts), dtype=np.int32)
    row_labels   = []

    for row_idx, ((layer, head), traj) in enumerate(sorted_heads):
        n_changes = sum(
            1 for i in range(1, len(traj)) if traj[i] != traj[i - 1]
        )
        label_matrix[row_idx] = np.array(traj[:n_ckpts], dtype=np.int32)
        row_labels.append(f"L{layer}H{head} ({n_changes}Δ)")

    fig_h = max(3.0, n_heads * 0.4)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    log_steps  = np.log10(np.maximum(steps, 1))
    step_edges = np.concatenate([
        [log_steps[0] - (log_steps[1] - log_steps[0]) / 2],
        (log_steps[:-1] + log_steps[1:]) / 2,
        [log_steps[-1] + (log_steps[-1] - log_steps[-2]) / 2],
    ])
    head_edges = np.arange(-0.5, n_heads + 0.5)

    ax.pcolormesh(
        step_edges, head_edges, label_matrix,
        cmap=cmap, norm=norm, shading="flat",
    )

    ax.set_yticks(range(n_heads))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xlabel("Training step", fontsize=11)
    ax.set_title(title, fontsize=12)

    key_steps = [s for s in [0, 100, 500, 1000, 5000, 10000, 50000, 100000]
                 if s <= steps[-1] and s >= steps[0]]
    tick_pos  = [np.log10(max(s, 1)) for s in key_steps]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([f"{s:,}" for s in key_steps], rotation=30, ha="right")

    # Color legend
    patches = [
        mpatches.Patch(
            facecolor=TYPE_COLORS_HEX[t],
            label=HEAD_TYPES[t].replace("_", " ").title(),
            edgecolor="#888888", linewidth=0.4,
        )
        for t in range(len(HEAD_TYPES))
    ]
    ax.legend(
        handles=patches,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=True, framealpha=0.9,
        edgecolor="#CCCCCC", fontsize=8,
        title="Head type", title_fontsize=8,
    )

    ax.invert_yaxis()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"  [Figure] Individual trajectories saved → {output_path}")
