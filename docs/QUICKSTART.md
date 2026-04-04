# Quick Start Guide

Get up and running with the Developmental Trajectories project in 10 minutes.

## Prerequisites

- Python 3.10 or higher
- 8GB+ RAM
- (Optional) Modal account for cloud training

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/abderahmane-ai/head-trajectories.git
cd head-trajectories
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify installation
```bash
python run_tests.py
```

You should see the test suite complete successfully. The exact count changes over time, so rely on pass/fail rather than a hardcoded number.

## Quick Demo (Local)

### Build probe dataset
```bash
python -c "
from pathlib import Path
from data import build_probe_dataset, verify_induction_probes

probe = build_probe_dataset(Path('probe/probe_dataset.pt'), seed=0)
verify_induction_probes(probe)
print('✓ Probe dataset built successfully')
"
```

### Test model forward pass
```bash
python -c "
import torch
from model import TransformerLM, ModelConfig

config = ModelConfig.small_15m()
model = TransformerLM(config)
print(f'✓ Model initialized: {model.count_parameters() / 1e6:.1f}M parameters')

# Test forward pass
input_ids = torch.randint(0, config.vocab_size, (2, 16))
logits, attn_maps = model(input_ids, return_attention=True)
print(f'✓ Forward pass successful')
print(f'  Logits shape: {logits.shape}')
print(f'  Attention maps: {len(attn_maps)} layers')
"
```

### Test scoring functions
```bash
python -c "
import torch
from probing.scores import sink_score, semantic_score

# Create fake attention (random softmax)
N, T = 10, 32
attn = torch.softmax(torch.randn(N, T, T), dim=-1)

# Sink score measures fixed-position anchoring (not sharpness)
# Random attention typically scores low (~0.1-0.3)
# A true sink head (all queries attend to j=0) would score ~1.0
score = sink_score(attn)
print(f'✓ Sink score: {score:.4f}')
"
```

## Cloud Training (Modal)

### 1. Install Modal
```bash
pip install modal
modal setup
```

### 2. Launch training
```bash
# First run (builds probe dataset)
modal run modal_jobs/train_seed42.py

# Parallel runs (after seed42 completes)
modal run modal_jobs/train_seed123.py &
modal run modal_jobs/train_seed777.py &
```

These Modal entry points are for the long OpenWebText study. For notebook-scale comparison runs and local orchestration, prefer the profile-based notebook or `run_single_experiment.py`.

## Notebook / Profile-Driven Runs

The main workflow now uses named experiment profiles. Current profiles include:

- `wikitext103_15m_preliminary` — fast current-method sanity run
- `lm1b_15m_comparison` — cross-dataset comparison run
- `openwebtext_15m_main` — main long OpenWebText run
- `openwebtext_6m_ablation` — 6M OpenWebText scale ablation

Example:

```bash
python run_single_experiment.py --profile wikitext103_15m_preliminary --seed 42
```

### 3. Monitor progress
```bash
modal app logs trajectories-seed42
```

### 4. Download checkpoints
```bash
modal volume get trajectories-ckpts-seed42 /checkpoints/seed42
```

## Running the Pipeline

### 1. Probing (after training)
```bash
# Dry run (estimate time)
python run_probing.py --seed 42 --dry_run

# Full run
python run_probing.py --seed 42
```

Expected output:
```
Found N checkpoints to process.
Processing checkpoint 1/N...
[Progress bar]
Probing pipeline complete.
Results saved to: results/results_seed42.pt
```

### 2. Analysis
```bash
python run_analysis.py
```

Expected output:
```
Loading results for 3 seeds...
Computing global trajectories...
Computing per-layer curves...
Generating figures...

Figures saved to figures/:
  - fig1_timeline.png
  - fig2_heatmap_spec.png
  - fig3_mixed_behavior.png
  - fig4_stability.png

Hypothesis verdicts:
  H1 (Sink-First Among Learned Types): supported / not supported / not robust depending on results
  H2 (Learned Ordered Development): supported / not supported / not robust depending on results
  ...
```

## Project Structure

```
head-trajectories/
├── model/           # Transformer implementation
├── data/            # Data loading
├── training/        # Training loop
├── probing/         # Attention probing
├── analysis/        # Scientific analysis
├── visualization/   # Figures
├── tests/           # Unit tests
└── modal_jobs/      # Cloud training
```

## Common Tasks

### Run specific tests
```bash
pytest tests/test_scores.py -v
```

### Check code style
```bash
flake8 probing/ model/ analysis/
```

Style tooling is optional; the repository does not currently require lint-clean output for core execution.

### Generate coverage report
```bash
pytest tests/ --cov=probing --cov=model --cov-report=html
open htmlcov/index.html
```

### Profile performance
```bash
python -m cProfile -o profile.stats run_probing.py --seed 42
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumtime').print_stats(20)"
```

## Troubleshooting

### Import errors
Make sure you're in the project root:
```bash
cd /path/to/head-trajectories
python run_tests.py
```

### Out of memory
Reduce batch size:
```bash
python run_probing.py --seed 42 --batch_size 8
```

### Slow probing
The vectorized semantic_score should be fast. If it's slow:
```bash
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"
```

### Modal authentication
```bash
modal token set --token-id YOUR_TOKEN_ID --token-secret YOUR_TOKEN_SECRET
```

## Next Steps

- Read [ARCHITECTURE.md](ARCHITECTURE.md) for codebase overview
- Read [METHODOLOGY.md](METHODOLOGY.md) for the formal experimental and mathematical specification
- Read [FAQ.md](FAQ.md) for interpretation notes on thresholds, dominant labels, and mixed behaviors
- See [CONTRIBUTING.md](../CONTRIBUTING.md) for development guidelines
- Check [README.md](../README.md) for full documentation
- Explore the test suite in `tests/` for usage examples

## Getting Help

- GitHub Issues: Bug reports and feature requests
- Discussions: Questions and ideas
- Email: [Your contact if provided]

## Resources

- [Modal Documentation](https://modal.com/docs)
- [PyTorch Documentation](https://pytorch.org/docs)
- [Mechanistic Interpretability Resources](https://www.neelnanda.io/mechanistic-interpretability/getting-started)
