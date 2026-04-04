# Frequently Asked Questions

## General

### What is this project about?
This project studies when and how attention heads in transformers develop their specialized behaviors during training. We track all heads across a profile-specific checkpoint schedule to understand their developmental trajectories.

### Is this related to any paper?
This is independent research. A paper may be published based on these results.

### Can I use this code for my own research?
Yes! The code is MIT licensed. Please cite this repository if you use it.

## Technical

### Why build a transformer from scratch?
We need precise control over attention extraction. Using HuggingFace would add unnecessary complexity and make attention map extraction harder.

### Why dense checkpointing?
The dense-early schedule captures rapid early development while remaining computationally feasible. Most interesting dynamics happen in the first part of training, so the profiles save checkpoints much more densely early on than late.

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
The active methodology is now statistically grounded at the classifier layer: each head/metric score is compared to the pooled empirical null, one-sided empirical p-values are computed, and per-head BH-FDR is applied across the five behaviors. Robustness controls now focus on FDR-alpha sensitivity, null-subsample stability, and inter-seed agreement. Single-seed pilot or comparison runs should still be treated as exploratory rather than conclusive.

### Why not more seeds?
Cost-benefit tradeoff. Multi-seed runs are the right standard for stronger claims, but notebook-scale exploratory runs are often done first to validate methodology, inspect score geometry, and compare datasets cheaply.

### How were thresholds chosen?
Calibration still comes from a random baseline: initialize random models (15M and 6M), causally scramble key positions within each valid attention row to destroy structured behavior while preserving causal row-stochastic attention, compute all 5 scores, and store the resulting empirical null. The repository still stores `mean + 2*std` / `p99` threshold summaries for diagnostics and legacy inspection, but the main classifier now uses the **full pooled null distribution** rather than a single scalar threshold per metric. See `data/calibration.py`.

### What about other head types?
The five types we study are the most well-documented. The framework is extensible - you can add new scoring functions.

## Usage

### Can I use a different dataset?
Yes. The experiment runner is profile-driven and already supports multiple dataset-backed profiles, including WikiText-103, LM1B, and OpenWebText. When you change datasets, rebuild the probe dataset for that profile so calibration and held-out probes remain aligned.

### Can I change the model size?
Yes. See `ModelConfig.small_15m()` and `ModelConfig.ablation_6m()` for examples. Adjust `n_layers`, `n_heads`, `d_model`.

### How do I add a new scoring function?
1. Add function to `probing/scores.py`
2. Update `score_head()` to call it
3. Update the classifier and null-calibration logic to include the new metric in the pooled empirical-null inference
4. Add tests to `tests/test_scores.py`

### Can I visualize individual head trajectories?
Yes! See `analysis/trajectories.py::compute_head_trajectories()` and `visualization/timeline_plot.py`.

### How do I interpret the results?
The canonical hypothesis wording is:

- `H1`: Sink-first among learned types — learned sink onset is no later than prev-token, induction, or semantic onset
- `H2`: Learned ordered development — `SINK <= PREV_TOKEN < INDUCTION < SEMANTIC` after separating architectural positional initialization
- `H3`: Layer stratification — lower layers reach substantial specialization earlier than higher layers
- `H4`: Induction phase transition — induction emergence is abrupt rather than smooth
- `H5`: Sink persistence — heads that become sinks remain sinks for most later checkpoints

Important interpretation notes:

- dominant labels are summaries, not full identities
- the active behavior set is the primary scientific object
- the raw score tensor is preserved, so a head can express multiple statistically active behaviors even if one dominant summary label wins
- onset steps are operational statistics, not immutable truths; they can be sensitive in noisy or single-seed runs
- current single-seed comparison runs do not strongly support `H1` or `H2`

See [METHODOLOGY.md](./METHODOLOGY.md) for the exact operational definitions.

### Is the induction probe dataset too weak?
Current evidence does not suggest that the induction probes are broken. Probe-integrity checks pass, and the induction metric can fire in real runs. The more likely interpretation is that induction is weak, late, or dataset-sensitive under the current small-model / 12k-step comparison settings.

### Are labels the whole story?
No. The classifier now detects an **active behavior set** via empirical p-values and BH-FDR, then assigns a dominant summary label only if one surviving behavior clearly wins. This is why the repository stores active-set tensors, p-values, effect sizes, runner-up behavior, and margins in addition to the dominant label tensor.

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
