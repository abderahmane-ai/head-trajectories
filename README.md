# Developmental Trajectories of Attention Heads

[![Tests](https://github.com/abderahmane-ai/head-trajectories/workflows/Tests/badge.svg)](https://github.com/abderahmane-ai/head-trajectories/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Tracking when and how attention heads specialize during transformer training

Most interpretability work asks *what* attention heads do after training. This project asks *when* they become what they are. We train transformers from scratch with dense checkpointing, probe every head at every checkpoint, and track developmental trajectories from initialization to convergence.

**Current evidence:** Results so far suggest an ordered developmental pathway: architectural `positional (RoPE)` structure appears at initialization, and learned specialization often proceeds through `sink → prev-token → induction → semantic`.

## Quick Start

```bash
# Install
pip install -r requirements.txt
python run_tests.py  # suite currently has 83 tests; local temp-dir behavior may vary by environment

# Run full pipeline (requires Modal account for training)
modal run modal_jobs/train_seed42.py        # ~5h on A100, $6
python run_probing.py --seed 42             # ~10min on CPU
python run_analysis.py                      # generates figures/
```

See [docs/QUICKSTART.md](docs/QUICKSTART.md) for detailed setup.

## Method

**Training:** 15M parameter LLaMA-style transformers on OpenWebText with 100 checkpoints per run (dense early: every 50 steps for first 500 steps, sparse late: every 2000 steps after 50K).

**Probing:** Fixed held-out dataset with three probe types:
- General sequences (real text)
- Induction sequences (engineered repeated patterns)
- Positional pairs (same length, different content)

**Scoring:** Five behavioral metrics per head:
- **SINK**: Fixed-position anchoring (causal-normalized)
- **PREV_TOKEN**: Attention to t-1
- **INDUCTION**: Pattern completion via repeated subsequences
- **POSITIONAL**: Content-invariant attention (KL divergence)
- **SEMANTIC**: Alignment with embedding similarity (masked)

**Classification:** Thresholds calibrated from a causally scrambled random baseline (mean + 2σ). Heads are classified by `argmax(scores / thresholds)` with defensive threshold sanitization only as a fallback for pathological calibrations.

See [docs/METHODOLOGY.md](docs/METHODOLOGY.md) for mathematical specification.

## Results

**H1 (Sink-First Among Learned Types):** Learned sink onset occurs no later than other learned types in current runs  
**H2 (Learned Ordered Development):** Current evidence suggests `SINK ≤ PREV_TOKEN < INDUCTION < SEMANTIC` once architectural positional initialization is separated out  
**H3 (Layer Stratification):** Lower layers reach specialization earlier than higher layers in current runs  
**H4 (Induction Phase Transition):** Induction emergence appears abrupt rather than gradual in some runs and scales  
**H5 (Sink Persistence):** Heads that become sinks usually remain sinks for most subsequent checkpoints

**Working finding:** The `SINK → PREV_TOKEN` pathway appears repeatedly in preliminary analyses. Heads may learn fixed-position anchoring before dynamic relative tracking, but that should be treated as an evidence-backed hypothesis rather than a settled law until larger runs and sensitivity checks are complete.

## Repository Structure

```
├── model/              Transformer implementation (RoPE, RMSNorm, SwiGLU)
├── data/               Probe construction + OpenWebText streaming
├── training/           Training loop with checkpoint schedule
├── probing/            Attention extraction + 5 scoring functions
├── analysis/           Trajectory analysis + hypothesis tests
├── visualization/      Figure generation (300 DPI)
├── modal_jobs/         Cloud training scripts (4 runs)
├── tests/              83 unit tests
└── docs/               Full documentation
```

## Key Files

- `run_probing.py` — Score all heads at all checkpoints
- `run_analysis.py` — Generate figures + hypothesis verdicts
- `probing/scores.py` — Five behavioral metrics
- `data/calibration.py` — Threshold calibration from random baseline

## Compute Requirements

**Training:** 4 runs × 5h on A100 = ~$22 total (Modal)  
**Probing:** CPU only, ~10min per run  
**Analysis:** CPU only, <1min

Total cost to reproduce: **~$25**

## Documentation

- [QUICKSTART.md](docs/QUICKSTART.md) — Installation and execution
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
