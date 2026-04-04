# Developmental Trajectories of Attention Heads

[![Tests](https://github.com/abderahmane-ai/head-trajectories/workflows/Tests/badge.svg)](https://github.com/abderahmane-ai/head-trajectories/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Tracking when and how attention heads specialize during transformer training

Most interpretability work asks *what* attention heads do after training. This project asks *when* they become what they are. We train transformers from scratch with dense checkpointing, probe every head at every checkpoint, and track developmental trajectories from initialization to convergence.

**Current evidence:** The pipeline is stable, and the current methodology now uses **FDR-based multi-behavior inference** rather than heuristic threshold-normalized winner-take-all labeling. The strongest current empirical result remains that final head-type mixes are dataset-sensitive, `PREV_TOKEN` is usually the clearest dominant learned behavior, `SEMANTIC` can appear early and remains calibration-sensitive, and many heads exhibit multiple statistically active behaviors even when the report view assigns one dominant summary label.

## Quick Start

```bash
# Install
pip install -r requirements.txt
python run_tests.py  # exact test count changes over time

# Run a profile-driven experiment locally or in the notebook
python run_single_experiment.py --profile wikitext103_15m_preliminary --seed 42
```

See [docs/QUICKSTART.md](docs/QUICKSTART.md) for detailed setup.
Containerized setup is documented in [docs/CONTAINER.md](docs/CONTAINER.md).

## Method

**Training:** Profile-driven decoder-only transformer runs. The repository currently supports notebook-scale comparison runs on WikiText-103 and LM1B, plus longer OpenWebText runs. Checkpoint schedules are profile-specific, but all keep a dense-early emphasis.

**Probing:** Fixed held-out dataset with probe families:
- General sequences (real text)
- Induction sequences (engineered repeated patterns)
- Natural induction sequences (real repeated subsequences, optional auxiliary comparison and currently disabled by default)
- Positional pairs (same length, different content)

**Scoring:** Five behavioral metrics per head:
- **SINK**: Fixed-position anchoring (causal-normalized)
- **PREV_TOKEN**: Attention to t-1
- **INDUCTION**: Pattern completion via repeated subsequences
- **POSITIONAL**: Content-invariant attention (KL divergence)
- **SEMANTIC**: Alignment with embedding similarity (masked)

**Classification:** Thresholds are still calibrated from a causally scrambled random baseline, but they are now diagnostic/reference quantities rather than the main decision rule. The classifier uses the **pooled empirical null**, computes one-sided empirical p-values per metric, applies **per-head BH-FDR** across the five behaviors, treats the surviving metrics as the head’s **active behavior set**, and then assigns a dominant summary label only if one surviving behavior clears a fixed effect-size margin. Non-specialized states are now split into `WEAK` (no behaviors survive) and `AMBIGUOUS` (multiple survive without a clear winner).

See [docs/METHODOLOGY.md](docs/METHODOLOGY.md) for mathematical specification.

## Results

The repository evaluates five canonical hypotheses:

**H1 (Sink-First Among Learned Types):** Learned sink onset occurs no later than other learned types  
**H2 (Learned Ordered Development):** `SINK ≤ PREV_TOKEN < INDUCTION < SEMANTIC` after separating architectural positional initialization  
**H3 (Layer Stratification):** Lower layers reach specialization earlier than higher layers  
**H4 (Induction Phase Transition):** Induction emergence is abrupt rather than gradual  
**H5 (Sink Persistence):** Heads that become sinks remain sinks for most subsequent checkpoints

**Current status:** Single-seed comparison runs do **not** support strong `H1` or `H2` claims. The clearest current result is that final behavioral mixes differ substantially across datasets, especially in the balance between `PREV_TOKEN` and `SEMANTIC`. The new FDR-based active-set view also confirms widespread mixed-behavior heads, so dominant labels should be read as summaries rather than full identities.

## Repository Structure

```
├── model/              Transformer implementation (RoPE, RMSNorm, SwiGLU)
├── data/               Probe construction + threshold calibration
├── training/           Training loop with checkpoint schedule
├── probing/            Attention extraction + 5 scoring functions
├── analysis/           Trajectory analysis + hypothesis tests
├── visualization/      Figure generation (300 DPI)
├── modal_jobs/         Cloud training scripts (4 runs)
├── tests/              Unit and integration tests
└── docs/               Full documentation
```

## Key Files

- `run_probing.py` — Score all heads at all checkpoints
- `run_analysis.py` — Generate figures + hypothesis verdicts
- `probing/scores.py` — Five behavioral metrics
- `data/calibration.py` — Threshold calibration from random baseline

## Compute Requirements

**Training:** Depends on profile. Notebook-scale 12k-step runs are roughly ~1 hour on an A100-class GPU; 100k-step OpenWebText runs are several hours.  
**Probing:** CPU only, typically minutes per run depending on profile  
**Analysis:** CPU only, <1min

Total cost to reproduce depends on which profiles and how many seeds you run.

## Documentation

- [QUICKSTART.md](docs/QUICKSTART.md) — Installation and execution
- [CONTAINER.md](docs/CONTAINER.md) — Docker runtime for notebook and script workflows
- [METHODOLOGY.md](docs/METHODOLOGY.md) — Mathematical specification
- [ARCHITECTURE.md](docs/ARCHITECTURE.md) — Codebase structure
- [EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md) — Pilot results and findings

## Citation

```bibtex
@misc{abderahmane2025developmental,
  title   = {Developmental Trajectories of Attention Heads},
  author  = {Abderahmane Ainouche},
  year    = {2025},
  url     = {https://github.com/abderahmane-ai/head-trajectories},
  note    = {Independent research, ENSIA Algeria}
}
```

## License

MIT License - see [LICENSE](LICENSE)

---

**Author:** Abderahmane Ainouche | ENSIA Algeria | Academic: `abderahmane.ainouche@ensia.edu.dz` | Contact: `abderahmane.ainouche.ai@gmail.com` | [Prior work: ARU](https://doi.org/10.13140/RG.2.2.18700.58241)
