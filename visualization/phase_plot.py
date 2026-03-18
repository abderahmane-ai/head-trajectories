"""
visualization/phase_plot.py — Phase transition dual-axis plot.

Produces Figure 3: a dual-axis plot showing induction head count (left y-axis)
and validation loss (right y-axis) on the same x-axis (training step).

The key visual claim for H4: a simultaneous discontinuity in the induction
head count curve and an inflection point in the validation loss curve at the
same training step. If H4 holds, these two events visually co-occur.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from typing import Dict, List, Optional

matplotlib.rcParams.update({
    "font.family":     "serif",
    "font.size":       11,
    "axes.labelsize":  12,
    "axes.titlesize":  13,
    "legend.fontsize": 10,
    "figure.dpi":      300,
    "savefig.dpi":     300,
    "savefig.bbox":    "tight",
})

INDUCTION_COLOR = "#EF9F27"
VAL_LOSS_COLOR  = "#378ADD"
MARKER_COLOR    = "#CC3333"


def plot_phase_transition(
    induction_curve:   Dict[str, np.ndarray],
    val_loss_curve:    Dict[str, np.ndarray],
    crossing_steps:    Dict[float, Optional[int]],
    inflection_result: Dict[str, object],
    output_path:       Path,
    title:             str = "Induction Head Emergence and Validation Loss (H4)",
) -> None:
    """
    Dual-axis plot of induction head count vs. validation loss over training.

    Args:
        induction_curve:   output of compute_induction_count_curve
        val_loss_curve:    output of extract_val_loss_curve
        crossing_steps:    output of find_crossing_steps
        inflection_result: output of detect_val_loss_inflection (at 25% crossing)
        output_path:       path to save the PNG figure
        title:             figure title
    """

    steps       = induction_curve["steps"]
    mean_count  = induction_curve["mean_count"]
    std_count   = induction_curve["std_count"]
    final_count = induction_curve["final_count"]
    n_seeds     = induction_curve["per_seed"].shape[0]

    val_steps  = val_loss_curve["steps"]
    val_loss   = val_loss_curve["val_loss"]

    fig, ax1 = plt.subplots(figsize=(8, 4.5))

    # ── Left axis: induction head count ─────────────────────────────────────
    ax1.plot(
        steps, mean_count,
        color=INDUCTION_COLOR, lw=2.2, label="Induction head count", zorder=3,
    )
    if n_seeds > 1:
        ax1.fill_between(
            steps,
            np.clip(mean_count - std_count, 0, None),
            mean_count + std_count,
            color=INDUCTION_COLOR, alpha=0.15, zorder=2,
        )

    ax1.set_xlabel("Training step", fontsize=12)
    ax1.set_ylabel("Induction head count", color=INDUCTION_COLOR, fontsize=12)
    ax1.tick_params(axis="y", labelcolor=INDUCTION_COLOR)
    ax1.set_ylim(bottom=-0.5)

    # ── Right axis: validation loss ──────────────────────────────────────────
    ax2 = ax1.twinx()
    ax2.plot(
        val_steps, val_loss,
        color=VAL_LOSS_COLOR, lw=1.8, ls="--",
        label="Validation loss", zorder=3, alpha=0.85,
    )
    ax2.set_ylabel("Validation loss (cross-entropy)", color=VAL_LOSS_COLOR, fontsize=12)
    ax2.tick_params(axis="y", labelcolor=VAL_LOSS_COLOR)

    # ── Crossing step markers ────────────────────────────────────────────────
    frac_labels = {0.10: "10%", 0.25: "25%", 0.50: "50%"}
    for frac, step in crossing_steps.items():
        if step is None:
            continue
        ax1.axvline(
            x=step, color=INDUCTION_COLOR,
            lw=1.0, ls=":", alpha=0.7, zorder=1,
        )
        ax1.annotate(
            frac_labels.get(frac, f"{int(frac*100)}%"),
            xy=(step, mean_count.max() * 0.05),
            fontsize=7.5,
            color=INDUCTION_COLOR,
            rotation=90,
            va="bottom",
            ha="right",
        )

    # ── Inflection marker ────────────────────────────────────────────────────
    if inflection_result["inflection_found"]:
        inf_step = inflection_result["inflection_step"]
        ax2.axvline(
            x=inf_step, color=MARKER_COLOR,
            lw=1.2, ls="-.", alpha=0.8, zorder=4,
        )
        ax2.annotate(
            f"Val loss\ninflection\n(step {inf_step:,})",
            xy=(inf_step, val_loss.max()),
            xytext=(inf_step * 1.3, val_loss.max() * 0.98),
            fontsize=8,
            color=MARKER_COLOR,
            arrowprops=dict(
                arrowstyle="->",
                color=MARKER_COLOR,
                lw=0.8,
            ),
        )

    # ── X-axis formatting ────────────────────────────────────────────────────
    ax1.set_xscale("log")
    ax1.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )
    ax1.set_title(title, fontsize=13, pad=10)

    # ── Combined legend ──────────────────────────────────────────────────────
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper right",
        frameon=True,
        framealpha=0.9,
        edgecolor="#CCCCCC",
        fontsize=9,
    )

    ax1.grid(axis="x", lw=0.3, alpha=0.3, zorder=0)

    # Annotation: confidence band note
    if n_seeds > 1:
        ax1.annotate(
            f"Shaded: ±1 std across {n_seeds} seeds",
            xy=(0.01, 0.01), xycoords="axes fraction",
            fontsize=8, color="#888888",
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"  [Figure] Phase transition plot saved → {output_path}")


def plot_discontinuity_comparison(
    induction_curve: Dict[str, np.ndarray],
    output_path:     Path,
    window_frac:     float = 0.15,
) -> None:
    """
    Supplementary: zoom into the discontinuity window showing the sharpness
    of the induction head emergence — the core visual claim of H4.

    Args:
        induction_curve: output of compute_induction_count_curve
        output_path:     path to save the PNG figure
        window_frac:     fraction of total training to show around the transition
    """

    steps      = induction_curve["steps"]
    mean_count = induction_curve["mean_count"]
    std_count  = induction_curve["std_count"]
    n_seeds    = induction_curve["per_seed"].shape[0]

    # Find the center of the transition (25% to 75% of final count)
    final      = induction_curve["final_count"]
    if final < 1.0:
        print("  [Figure] Skipping discontinuity zoom — no induction heads detected.")
        return

    t25 = np.where(mean_count >= 0.25 * final)[0]
    t75 = np.where(mean_count >= 0.75 * final)[0]

    if len(t25) == 0 or len(t75) == 0:
        print("  [Figure] Skipping discontinuity zoom — insufficient range.")
        return

    center_idx = (t25[0] + t75[0]) // 2
    half_win   = max(int(len(steps) * window_frac), 5)
    lo         = max(0, center_idx - half_win)
    hi         = min(len(steps), center_idx + half_win)

    zoom_steps = steps[lo:hi]
    zoom_count = mean_count[lo:hi]
    zoom_std   = std_count[lo:hi]

    fig, ax = plt.subplots(figsize=(5, 3.5))

    ax.plot(zoom_steps, zoom_count, color=INDUCTION_COLOR, lw=2.2)
    if n_seeds > 1:
        ax.fill_between(
            zoom_steps,
            np.clip(zoom_count - zoom_std, 0, None),
            zoom_count + zoom_std,
            color=INDUCTION_COLOR, alpha=0.15,
        )

    ax.set_xlabel("Training step", fontsize=11)
    ax.set_ylabel("Induction head count", fontsize=11)
    ax.set_title("Zoomed: Induction Head Emergence", fontsize=12)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(lw=0.3, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"  [Figure] Discontinuity zoom saved → {output_path}")
