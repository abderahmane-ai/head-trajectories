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
from typing import Dict, List, Optional, Sequence, Tuple

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


def _pretty_behavior_name(name: str) -> str:
    return name.replace("_", " ").title()


def _format_training_axis(ax: plt.Axes, steps: np.ndarray) -> None:
    """Use a zero-safe quasi-log axis so dense early checkpoints remain visible."""

    if len(steps) == 0:
        return

    has_zero = bool(np.any(np.asarray(steps) <= 0))
    if has_zero:
        ax.set_xscale("symlog", linthresh=1.0, linscale=1.0)
    else:
        ax.set_xscale("log")

    if len(steps) <= 16:
        tick_steps = [int(s) for s in steps.tolist()]
    else:
        candidate_ticks = [0, 50, 100, 200, 400, 800, 1200, 2500, 4000, 6000, 9000, 12000, 50000, 100000]
        lo = int(np.min(steps))
        hi = int(np.max(steps))
        tick_steps = [s for s in candidate_ticks if lo <= s <= hi]
        if not tick_steps:
            tick_steps = [int(steps[0]), int(steps[-1])]

    ax.set_xticks(tick_steps)
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{int(x):,}" if abs(x) >= 1 else "0")
    )


def _plot_curve_panel(
    ax: plt.Axes,
    curves: Dict[str, np.ndarray],
    *,
    include_types: Optional[Sequence[str]] = None,
    onset_steps: Optional[Dict[str, Optional[int]]] = None,
    de_emphasize: Sequence[str] = (),
    panel_title: Optional[str] = None,
    y_label: str = "Fraction of heads",
    log_like_x: bool = True,
) -> None:
    steps = curves["steps"]
    mean = curves["mean"]
    std = curves["std"]
    type_names = list(curves["type_names"])
    n_seeds = curves["per_seed"].shape[0]

    included = set(include_types) if include_types is not None else set(type_names)
    for t_idx, type_name in enumerate(type_names):
        if type_name not in included:
            continue
        y = mean[:, t_idx]
        y_std = std[:, t_idx]
        color = HEAD_TYPE_COLORS.get(type_name, "#333333")
        is_deemphasized = type_name in set(de_emphasize)
        ax.plot(
            steps,
            y,
            color=color,
            lw=1.25 if is_deemphasized else 2.1,
            ls="--" if is_deemphasized else "-",
            alpha=0.75 if is_deemphasized else 1.0,
            label=_pretty_behavior_name(type_name),
            zorder=3,
        )
        if n_seeds > 1:
            ax.fill_between(
                steps,
                np.clip(y - y_std, 0, 1),
                np.clip(y + y_std, 0, 1),
                color=color,
                alpha=0.10 if not is_deemphasized else 0.05,
                zorder=2,
            )

    if onset_steps is not None:
        for type_name, step in onset_steps.items():
            if step is None or type_name not in included:
                continue
            color = HEAD_TYPE_COLORS.get(type_name, "#333333")
            ax.axvline(step, color=color, lw=0.8, ls=":", alpha=0.45, zorder=1)

    if panel_title:
        ax.set_title(panel_title, loc="left", fontsize=11, pad=6)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_ylim(-0.02, 1.05)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(axis="y", lw=0.4, alpha=0.30, zorder=0)
    if log_like_x:
        _format_training_axis(ax, steps)
    else:
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))


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

    fig, ax = plt.subplots(figsize=(8.4, 5.0))

    include_types = [name for name in type_names if show_undiff or name not in DEEMPHASIZED_TYPES]
    _plot_curve_panel(
        ax,
        global_curves,
        include_types=include_types,
        onset_steps=onset_steps,
        de_emphasize=DEEMPHASIZED_TYPES,
        panel_title=None,
        y_label="Fraction of all attention heads",
        log_like_x=log_x,
    )

    # Axes formatting
    ax.set_xlabel("Training step", fontsize=12)
    ax.set_title(title, fontsize=13, pad=10)

    # Legend — place outside plot area to avoid overlap
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=True,
        framealpha=0.9,
        edgecolor="#CCCCCC",
        title=f"Head type\n(n={n_seeds} seeds)",
        title_fontsize=9,
    )

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


def plot_activation_dominance_figure(
    activation_curves: Dict[str, np.ndarray],
    dominance_curves: Dict[str, np.ndarray],
    output_path: Path,
    *,
    activation_onsets: Optional[Dict[str, Optional[int]]] = None,
    dominance_onsets: Optional[Dict[str, Optional[int]]] = None,
    title: str = "Behavior Emergence Over Training",
) -> None:
    """
    Two-panel core figure: active-set emergence (top) and dominant-summary
    trajectories (bottom). This is the cleanest overview of the current
    methodology because it separates detection from summary.
    """

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(9.2, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.18},
    )

    behavior_names = [name for name in activation_curves["type_names"] if name not in DEEMPHASIZED_TYPES]
    dominance_names = list(dominance_curves["type_names"])

    _plot_curve_panel(
        ax_top,
        activation_curves,
        include_types=behavior_names,
        onset_steps=activation_onsets,
        panel_title="A. Activation",
        y_label="Fraction of heads active",
    )
    _plot_curve_panel(
        ax_bottom,
        dominance_curves,
        include_types=dominance_names,
        onset_steps=dominance_onsets,
        de_emphasize=DEEMPHASIZED_TYPES,
        panel_title="B. Dominant Summary",
        y_label="Fraction of heads dominant",
    )

    ax_bottom.set_xlabel("Training step", fontsize=12)
    fig.suptitle(title, fontsize=13, y=0.98)

    handles, labels = ax_bottom.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=min(4, len(labels)),
        frameon=True,
        framealpha=0.95,
        edgecolor="#CCCCCC",
        fontsize=9,
    )

    n_seeds = dominance_curves["per_seed"].shape[0]
    if n_seeds > 1:
        ax_bottom.annotate(
            f"Shaded regions: ±1 std across {n_seeds} seeds",
            xy=(0.01, 0.02),
            xycoords="axes fraction",
            fontsize=8,
            color="#888888",
        )

    plt.tight_layout(rect=[0.0, 0.06, 1.0, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"  [Figure] Activation/dominance overview saved → {output_path}")


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
    seed_labels = global_curves.get("seed_labels", [str(i) for i in range(per_seed.shape[0])])
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

        seed_label = seed_labels[s_idx] if s_idx < len(seed_labels) else str(s_idx)
        ax.set_title(f"Seed {seed_label}", fontsize=11)
        ax.set_xlabel("Training step", fontsize=10)
        ax.set_ylim(-0.02, 1.05)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))
        ax.grid(axis="y", lw=0.4, alpha=0.4)
        _format_training_axis(ax, steps)

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


def plot_mixed_behavior_figure(
    mixed_behavior: Dict[str, np.ndarray],
    output_path: Path,
    *,
    title: str = "Mixed-Behavior Structure",
) -> None:
    """
    Core overlap figure:
      - left: prevalence of heads with >=2 or >=3 active behaviors
      - right: most common dominant/runner-up pairs at the final checkpoint
    """

    steps = mixed_behavior["steps"]
    ge2_mean = mixed_behavior["fraction_ge2_mean"]
    ge2_std = mixed_behavior["fraction_ge2_std"]
    ge3_mean = mixed_behavior["fraction_ge3_mean"]
    ge3_std = mixed_behavior["fraction_ge3_std"]
    top_pairs = mixed_behavior.get("final_top_pairs", [])[:6]
    n_seeds = int(mixed_behavior.get("n_runs", 1))

    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        figsize=(10.4, 4.4),
        gridspec_kw={"width_ratios": [1.35, 1.0], "wspace": 0.28},
    )

    ax_left.plot(steps, ge2_mean, color="#3E7CB1", lw=2.2, label=">=2 active behaviors", zorder=3)
    ax_left.plot(steps, ge3_mean, color="#C97B63", lw=2.0, label=">=3 active behaviors", zorder=3)
    if n_seeds > 1:
        ax_left.fill_between(steps, np.clip(ge2_mean - ge2_std, 0, 1), np.clip(ge2_mean + ge2_std, 0, 1), color="#3E7CB1", alpha=0.12, zorder=2)
        ax_left.fill_between(steps, np.clip(ge3_mean - ge3_std, 0, 1), np.clip(ge3_mean + ge3_std, 0, 1), color="#C97B63", alpha=0.12, zorder=2)
    ax_left.set_title("A. Overlap Prevalence", loc="left", fontsize=11, pad=6)
    ax_left.set_ylabel("Fraction of heads", fontsize=11)
    ax_left.set_xlabel("Training step", fontsize=11)
    ax_left.set_ylim(-0.02, 1.05)
    ax_left.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))
    ax_left.grid(axis="y", lw=0.4, alpha=0.30, zorder=0)
    _format_training_axis(ax_left, steps)
    ax_left.legend(loc="upper left", frameon=True, framealpha=0.95, edgecolor="#CCCCCC", fontsize=9)

    if top_pairs:
        labels = [
            " → ".join(_pretty_behavior_name(part) for part in pair.split(">"))
            for pair, _ in top_pairs
        ][::-1]
        values = [count for _, count in top_pairs][::-1]
        ax_right.barh(np.arange(len(labels)), values, color="#5A6673", alpha=0.88)
        ax_right.set_yticks(np.arange(len(labels)))
        ax_right.set_yticklabels(labels, fontsize=9)
        ax_right.set_xlabel("Final-checkpoint count", fontsize=11)
        ax_right.set_title("B. Dominant/Runner-up Pairs", loc="left", fontsize=11, pad=6)
        ax_right.grid(axis="x", lw=0.4, alpha=0.25)
        for idx, value in enumerate(values):
            ax_right.text(value + max(values) * 0.02, idx, str(value), va="center", fontsize=8, color="#444444")
    else:
        ax_right.text(0.5, 0.5, "No mixed pairs detected", ha="center", va="center", fontsize=10, color="#777777")
        ax_right.set_axis_off()

    fig.suptitle(title, fontsize=13, y=0.98)
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"  [Figure] Mixed-behavior figure saved → {output_path}")
