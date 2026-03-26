# Developmental Trajectories of Attention Heads

[![Tests](https://github.com/abderahmane-ai/head-trajectories/workflows/Tests/badge.svg)](https://github.com/abderahmane-ai/head-trajectories/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Tracking when and how attention heads specialize during transformer training

Most interpretability work asks *what* attention heads do after training. This project asks *when* they become what they are. We train transformers from scratch with dense checkpointing, probe every head at every checkpoint, and track developmental trajectories from initialization to convergence.

**Key finding:** Heads follow a strict developmental pathway: `positional (RoPE) → sink → prev-token → induction → semantic`. Sink heads must form before prev-token heads can emerge—a prerequisite relationship in circuit formation.

## Quick Start

```bash
# Install
pip install -r requirements.txt
python run_tests.py  # 63 tests should pass

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

**Classification:** Thresholds calibrated from random baseline (mean + 2σ). Heads classified by `argmax(scores / thresholds)`.

See [docs/METHODOLOGY.md](docs/METHODOLOGY.md) for mathematical specification.

## Results

**H1 (Sink-First):** ✓ Sinks appear in first 5% of training  
**H2 (Ordered Development):** ✓ Strict ordering: positional → sink → prev-token → induction → semantic  
**H3 (Layer Stratification):** ✓ Lower layers specialize earlier  
**H4 (Phase Transition):** Induction heads emerge discontinuously (scale-dependent)  
**H5 (Sink Persistence):** ✓ Sinks rarely change type once formed

**Novel finding:** SINK → PREV_TOKEN pathway reveals prerequisite structure. Heads learn fixed-position anchoring before dynamic relative tracking. This suggests circuit formation follows strict dependencies, not parallel specialization.

## Repository Structure

```
├── model/              Transformer implementation (RoPE, RMSNorm, SwiGLU)
├── data/               Probe construction + OpenWebText streaming
├── training/           Training loop with checkpoint schedule
├── probing/            Attention extraction + 5 scoring functions
├── analysis/           Trajectory analysis + hypothesis tests
├── visualization/      Figure generation (300 DPI)
├── modal_jobs/         Cloud training scripts (4 runs)
├── tests/              63 unit tests
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
  author  = {Abderahmane},
  year    = {2025},
  url     = {https://github.com/abderahmane-ai/head-trajectories},
  note    = {Independent research, ENSIA Algeria}
}
```

## License

MIT License - see [LICENSE](LICENSE)

---

**Author:** Abderahmane | ENSIA Algeria | [Prior work: ARU](https://doi.org/10.13140/RG.2.2.18700.58241)
