"""visualization/ — Publication-quality figures for all analysis results."""

from .timeline_plot import (
    plot_activation_dominance_figure,
    plot_mixed_behavior_figure,
    plot_timeline,
    plot_timeline_per_seed,
    TYPE_COLORS,
)
from .heatmap import (
    plot_dominant_type_heatmap,
    plot_specialization_fraction_heatmap,
)
from .phase_plot import (
    plot_phase_transition,
    plot_discontinuity_comparison,
)
from .stability_hist import (
    plot_stability_figure,
    plot_individual_trajectories,
)

__all__ = [
    "plot_activation_dominance_figure",
    "plot_mixed_behavior_figure",
    "plot_timeline",
    "plot_timeline_per_seed",
    "TYPE_COLORS",
    "plot_dominant_type_heatmap",
    "plot_specialization_fraction_heatmap",
    "plot_phase_transition",
    "plot_discontinuity_comparison",
    "plot_stability_figure",
    "plot_individual_trajectories",
]
