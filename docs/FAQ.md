# Frequently Asked Questions

## General

### What is this project about?
This project studies when and how attention heads in transformers develop their specialized behaviors during training. We track 64 heads across 100 checkpoints to understand their developmental trajectories.

### Is this related to any paper?
This is independent research. A paper may be published based on these results.

### Can I use this code for my own research?
Yes! The code is MIT licensed. Please cite this repository if you use it.

## Technical

### Why build a transformer from scratch?
We need precise control over attention extraction. Using HuggingFace would add unnecessary complexity and make attention map extraction harder.

### Why 100 checkpoints?
The dense-early schedule captures rapid early development while remaining computationally feasible. Most interesting dynamics happen in the first 20% of training.

### Why these five head types?
These are well-established behavioral patterns in the mechanistic interpretability literature: sinks, prev-token, induction, positional, and semantic heads.

### How long does training take?
~5 hours per seed on Modal A100. Total cost for 3 seeds + ablation: ~$22.

### How long does probing take?
After vectorization fixes: ~5-10 minutes per seed on CPU. Previously: 3-5 hours.

### Can I run this without Modal?
Yes, but you'll need a GPU for training. The probing pipeline runs fine on CPU.

## Scientific

### How do you ensure reproducibility?
- Fixed random seeds
- Immutable probe dataset
- Deterministic analysis
- All hyperparameters in code
- Version-controlled thresholds

### What about statistical significance?
We run 3 seeds and report inter-seed agreement. Controls include threshold sensitivity (±20%) and per-seed consistency checks.

### Why not more seeds?
Cost-benefit tradeoff. 3 seeds provide reasonable confidence while keeping compute costs under $25.

### How were thresholds chosen?
Calibrated from random baseline: initialize random models (15M and 6M), shuffle attention rows to destroy structure, compute all 5 scores, set threshold = mean + 2*std. This ensures heads must score 2 standard deviations above random noise to be classified. Calibration runs 3 seeds and reports stability. See `data/calibration.py`.

### What about other head types?
The five types we study are the most well-documented. The framework is extensible - you can add new scoring functions.

## Usage

### Can I use a different dataset?
Yes, but you'll need to rebuild the probe dataset. The code assumes OpenWebText but is adaptable.

### Can I change the model size?
Yes. See `ModelConfig.small_15m()` and `ModelConfig.ablation_6m()` for examples. Adjust `n_layers`, `n_heads`, `d_model`.

### How do I add a new scoring function?
1. Add function to `probing/scores.py`
2. Update `score_head()` to call it
3. Add threshold to `probing/classifier.py`
4. Add tests to `tests/test_scores.py`

### Can I visualize individual head trajectories?
Yes! See `analysis/trajectories.py::compute_head_trajectories()` and `visualization/timeline_plot.py`.

### How do I interpret the results?
See the paper (when published) or the docstrings in `analysis/` modules. Each hypothesis has a clear operationalization.

## Troubleshooting

### Probing is slow
Make sure you're using the vectorized `semantic_score`. Run tests to verify: `pytest tests/test_scores.py::TestSemanticScore::test_vectorization_correctness -v`

### Out of memory during probing
Reduce `--batch_size`: `python run_probing.py --seed 42 --batch_size 8`

### Modal authentication fails
Run `modal setup` and follow the prompts. You'll need a Modal account.

### Tests fail
Ensure you're using Python 3.10+ and have all dependencies: `pip install -r requirements.txt`

### Import errors
Make sure you're running from the project root: `cd /path/to/head-trajectories`

## Contributing

### How can I contribute?
See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines. Bug fixes, documentation, and new features are all welcome.

### I found a bug
Open an issue with a minimal reproducible example. Use the bug report template.

### I have a feature idea
Open an issue to discuss it first. Use the feature request template.

### Can I add my own analysis?
Absolutely! Add a module to `analysis/` and a visualization to `visualization/`. Submit a PR.

## Contact

### How do I get help?
- GitHub Issues: Bug reports and questions
- GitHub Discussions: General questions and ideas
- Email: [If you want to provide contact info]

### Can I collaborate?
Open an issue to discuss collaboration opportunities.

### Where can I learn more about mechanistic interpretability?
- [Neel Nanda's resources](https://www.neelnanda.io/mechanistic-interpretability)
- [Anthropic's interpretability research](https://www.anthropic.com/research)
- [TransformerLens](https://github.com/neelnanda-io/TransformerLens)
