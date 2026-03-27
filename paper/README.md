# Paper: A Methodology for Developmental Trajectory Analysis of Attention Heads

## Compiling the Paper

### Requirements
- LaTeX distribution (TeX Live, MiKTeX, or MacTeX)
- Required packages: amsmath, amssymb, amsthm, graphicx, hyperref, natbib, algorithm, algorithmic, booktabs, geometry

### Compilation

```bash
cd paper
pdflatex head_trajectories.tex
bibtex head_trajectories
pdflatex head_trajectories.tex
pdflatex head_trajectories.tex
```

Or use latexmk for automatic compilation:

```bash
latexmk -pdf head_trajectories.tex
```

## Structure

The paper includes:

1. **Abstract** - Methodological overview
2. **Motivation** - Why developmental analysis matters
3. **Experimental Object** - Checkpoints, probes, and trajectories
4. **Behavioral Metrics** - Formal definitions of the five scores
5. **Calibration and Classification** - Null model, thresholds, ambiguity handling
6. **Trajectory Analysis** - Onset, layer stratification, stability, transitions
7. **Methodological Guarantees and Limits** - What the pipeline measures and what it does not
8. **Discussion** - Extensions and caveats
9. **References** - Core related work

## Key Features

- **Methodology-only scope** - No empirical results or premature claims
- **Professional visual treatment** - modern sans-serif hierarchy, clean front matter, notation tables, and restrained callout styling
- **Complete mathematical specification** - all core objects and metrics formally defined
- **Calibration aligned with code** - causal key-scramble null and threshold diagnostics
- **Conceptual completeness** - covers what the project measures, how, and why

## Scope

This document is intentionally theory-and-methodology only. It should read like a serious methods note: complete enough to reproduce the pipeline conceptually, but disciplined enough to avoid claiming results that have not yet been fully refreshed under the corrected calibration regime.

## Figures

Create a `figures/` subdirectory and add:
- Global type-fraction curves
- Example head trajectories
- Layer-wise specialization heatmaps
- Onset time comparisons
- Phase transition plots (if applicable)

Reference figures in LaTeX:
```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.8\textwidth]{figures/global_curves.pdf}
\caption{Global type-fraction curves showing emergence order.}
\label{fig:global_curves}
\end{figure}
```

## Supplementary Materials

Consider adding supplementary materials for:
- Tie tolerance sensitivity analysis
- Complete probe dataset statistics
- Per-head trajectory visualizations
- Ablation studies
- Additional validation experiments

## Citation

Once published, cite as:

```bibtex
@article{ai2025trajectories,
  title={Developmental Trajectories of Attention Head Specialization in Transformer Language Models},
  author={Ai, Abderahmane},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## Notes

- The paper is currently methodology-only by design
- A future empirical companion can add results and figures once all runs are recomputed under the corrected calibration
- Suitable for submission to: NeurIPS, ICML, ICLR, or specialized interpretability venues
- Consider adding co-authors if collaborators contributed significantly
