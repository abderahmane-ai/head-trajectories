# Developmental Trajectories of Attention Heads

[![Tests](https://github.com/abderahmane-ai/head-trajectories/workflows/Tests/badge.svg)](https://github.com/abderahmane-ai/head-trajectories/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A mechanistic interpretability research project that tracks **when and in what
order** attention heads in a transformer develop their specialized behavioral
roles during training — from random initialization to convergence.

> **Independent Research** | ENSIA Algeria | 2025

## Scientific Question

We know what attention heads *are* at the end of training (induction heads,
sink heads, positional heads, semantic heads). Nobody has systematically studied
*when* they become what they are.

This project answers that question by training small LLaMA-style models (15M
parameters) from scratch with dense early checkpointing, scoring every head at
every checkpoint on five behavioral metrics, and tracking each head's
developmental trajectory.

## Hypotheses

| ID | Name | Claim |
|----|------|-------|
| H1 | Sink-First | Attention sinks appear in the first 5% of training |
| H2 | Ordered Development | Heads follow: sinks → positional → induction → semantic |
| H3 | Layer Stratification | Lower layers specialize earlier than higher layers |
| H4 | Phase Transition | Induction heads emerge discontinuously |
| H5 | Sink Persistence | Once a head becomes a sink, it rarely changes type |

## Repository Structure

```
trajectories/
├── model/              LLaMA-style transformer (from scratch, no HuggingFace)
├── data/               OpenWebText streaming + probe dataset construction
├── training/           Training loop with non-uniform checkpoint schedule
├── modal_jobs/         Modal A100 training jobs (4 runs)
├── probing/            Attention extraction, 5 scoring functions, classifier
├── analysis/           Trajectory curves, stability, phase transition, controls
├── visualization/      Publication-quality figures (300 DPI)
├── run_probing.py      Entry point: score all checkpoints
└── run_analysis.py     Entry point: produce all figures + hypothesis verdicts
```

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify installation
```bash
python run_tests.py
# Expected: 60 passed in ~4s
```

### 3. Build the probe dataset (once, before training)

The probe dataset is constructed from a held-out split of OpenWebText and
saved as an immutable file. **Never rebuild it after training has started.**

```bash
python -c "
from pathlib import Path
from data import build_probe_dataset, verify_induction_probes
probe = build_probe_dataset(Path('probe/probe_dataset.pt'), seed=0)
verify_induction_probes(probe)
"
```

### 4. Launch training runs on Modal

Run seed 42 first — it builds and commits the shared probe dataset Volume.

```bash
modal run modal_jobs/train_seed42.py
```

Once seed 42 has committed the probe Volume, the remaining runs can launch
in parallel:

```bash
modal run modal_jobs/train_seed123.py
modal run modal_jobs/train_seed777.py
modal run modal_jobs/train_ablation.py
```

Each job resumes automatically if interrupted. Expected cost: ~$20–25 total.

### 5. Download checkpoints from Modal Volumes

```bash
modal volume get trajectories-ckpts-seed42   /checkpoints/seed42
modal volume get trajectories-ckpts-seed123  /checkpoints/seed123
modal volume get trajectories-ckpts-seed777  /checkpoints/seed777
modal volume get trajectories-ckpts-ablation6m /checkpoints/ablation_6m
```

### 6. Run the probing pipeline

Scores all five behavioral metrics for every head at every checkpoint.
Expected runtime: ~5-10 minutes per run (CPU only, no GPU needed) after vectorization optimizations.

```bash
python run_probing.py
```

Run a single seed:
```bash
python run_probing.py --seed 42
```

Dry run (estimate time without processing):
```bash
python run_probing.py --seed 42 --dry_run
```

### 7. Run analysis and produce all figures

```bash
python run_analysis.py
```

Figures are saved to `figures/`. Reports for all 5 hypotheses are printed
to stdout with final verdicts.

## Checkpoint Schedule

The dense-early schedule is the core experimental design decision.
Most developmental action happens in the first 10–20% of training.

| Training phase | Step range | Save every |
|----------------|------------|------------|
| Very early | 0–500 | 50 steps |
| Early | 500–5,000 | 200 steps |
| Mid-early | 5,000–20,000 | 500 steps |
| Mid | 20,000–50,000 | 1,000 steps |
| Late | 50,000–end | 2,000 steps |

Total: ~100 checkpoints per run.

## Head Type Scoring

| Type | Metric | Threshold |
|------|--------|-----------|
| SINK | Max attention weight per row (mean) | Calibrated |
| PREV\_TOKEN | Attention to position t-1 (mean) | Calibrated |
| INDUCTION | Attention to token after first occurrence | Calibrated |
| POSITIONAL | 1 – KL div between same-length seqs | Calibrated |
| SEMANTIC | Pearson corr with cosine embedding sim | Calibrated |

Thresholds are calibrated from random baseline: initialize random models, shuffle
attention rows, compute scores, set threshold = mean + 2*std. This ensures heads
score 2 standard deviations above random noise. Separate calibration for 15M and
6M models (see `data/calibration.py`).

Classification: `argmax(scores / thresholds)` with UNDIFFERENTIATED fallback
if all scores are below threshold, or if the top two normalized scores are
within 0.05 of each other (tie logged to `results/ties.csv`).

## Output Figures

| File | Description | Hypothesis |
|------|-------------|------------|
| `fig1_timeline.png` | Head type fraction vs. training step | H1, H2 |
| `fig1b_timeline_per_seed.png` | Per-seed curves (supplement) | H1, H2 |
| `fig2a_heatmap_dominant.png` | Dominant type by layer and step | H3 |
| `fig2b_heatmap_spec.png` | Specialization wave heatmap | H3 |
| `fig3_phase_transition.png` | Induction count + val loss dual-axis | H4 |
| `fig3b_discontinuity_zoom.png` | Zoomed transition window | H4 |
| `fig4_stability.png` | Type-change histogram + sink persistence | H5 |
| `fig4b_trajectories.png` | Individual head trajectories | H5 |

## Model Configuration

**Primary runs (15M parameters):**
- 8 layers, 8 heads/layer, d\_model=384, d\_ffn=1536
- RoPE positional encoding, RMSNorm, SwiGLU FFN
- Trained on OpenWebText, ~300–500M tokens

**Ablation run (6M parameters):**
- 6 layers, 8 heads/layer, d\_model=256, d\_ffn=1024
- Same hyperparameters, used to test scale-invariance of H3

## Compute Budget

| Run | GPU | Est. time | Est. cost |
|-----|-----|-----------|-----------|
| seed 42 (15M) | A100 | ~5h | ~$6 |
| seed 123 (15M) | A100 | ~5h | ~$6 |
| seed 777 (15M) | A100 | ~5h | ~$6 |
| ablation (6M) | A100 | ~3h | ~$4 |
| **Total** | | **~18h** | **~$22** |

Probing pipeline (CPU): ~5-10 minutes per run after vectorization, no additional GPU cost.

## Citation

If you use this codebase in your research, please cite:

```bibtex
@misc{abderahmane2025developmental,
  title   = {Developmental Trajectories of Attention Heads},
  author  = {Abderahmane},
  year    = {2025},
  url     = {https://github.com/abderahmane-ai/head-trajectories},
  note    = {Independent research, ENSIA Algeria}
}
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenWebText dataset from HuggingFace
- Modal for cloud compute infrastructure
- The mechanistic interpretability community for inspiration

## Author

**Abderahmane**  
4th-year AI Engineering Student, ENSIA Algeria  
Independent Mechanistic Interpretability Research

Prior work: [ARU — Breaking the Zero-Sum Game in RNNs](https://doi.org/10.13140/RG.2.2.18700.58241)

## Contact

- GitHub Issues: For bugs and feature requests
- GitHub: [@abderahmane-ai](https://github.com/abderahmane-ai)

---

**Star ⭐ this repository if you find it useful!**
