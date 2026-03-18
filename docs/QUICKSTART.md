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

You should see: `60 passed in ~4s`

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

# Create fake attention
N, T = 10, 32
attn = torch.softmax(torch.randn(N, T, T), dim=-1)

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
Found 100 checkpoints to process.
Processing checkpoint 1/100...
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
  - fig2a_heatmap_dominant.png
  - fig3_phase_transition.png
  - fig4_stability.png

Hypothesis verdicts:
  H1 (Sink First): SUPPORTED
  H2 (Ordered Development): SUPPORTED
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
