# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-19

### Added
- Initial release of Developmental Trajectories research codebase
- LLaMA-style transformer implementation from scratch
- Five behavioral scoring functions (sink, prev_token, induction, positional, semantic)
- Head classification with tie-breaking logic
- Dense-early checkpoint schedule (100 checkpoints per run)
- Modal cloud training integration
- Comprehensive probing pipeline
- Trajectory analysis and visualization
- Scientific controls (legacy threshold sensitivity, inter-seed agreement)
- Phase transition analysis for induction heads
- Publication-quality figures (300 DPI)
- Complete unit test suite (60 tests)
- CI/CD with GitHub Actions

### Performance
- Vectorized semantic_score implementation (30-100× speedup)
- Incremental checkpoint saving with resumption
- Efficient attention extraction with batching

### Documentation
- Comprehensive README with quickstart guide
- Detailed docstrings for all modules
- Test suite documentation
- Contributing guidelines
- Scientific methodology documentation

## [Unreleased]

### Changed
- **BREAKING**: Default classifier replaced with FDR-based multi-behavior inference
  - Computes empirical p-values from pooled null calibration scores
  - Applies per-head BH-FDR across the 5 behavior metrics
  - Stores active behavior sets as the primary state representation
  - Uses effect-size margin to decide dominant summary vs `AMBIGUOUS`
- **BREAKING**: Label ontology migrated from 6-class heuristic schema to 7-class summary schema
  - `UNDIFFERENTIATED` replaced by `WEAK` and `AMBIGUOUS`
  - Dominant labels remain for reporting, but active sets drive inference
- Calibration summaries remain (`mean+2std`, `p99`) as diagnostics/reference only
- Scientific controls now use FDR-alpha sensitivity and null-subsample stability

- **BREAKING**: SINK metric now measures fixed-position anchoring instead of sharpness
  - Uses causal-mask normalization: divides by number of reachable queries per key position
  - Separates true sink heads (score ~1.0) from prev-token heads (score ~0.5)
  - Previous "sharpness" metric conflated sink and prev-token behaviors
- **BREAKING**: SEMANTIC metric now uses exclusion masking to remove confounds
  - Masks out j=0 (sink), j=t-1 (prev-token), j=t (identity) before computing Pearson
  - Requires minimum 6 valid points for stable correlation
  - Prevents structural confounds from inflating semantic scores

### Fixed
- Removed dead `causal_mask` variable in `semantic_score` function

### Added
- Regression test validating SINK vs PREV_TOKEN separation
- Mask validation test for SEMANTIC exclusion masking
- Detailed mathematical documentation in METHODOLOGY.md

### Planned
- Additional visualization options
- Extended analysis for layer stratification (H3)
- Heldout probe set validation
- Performance profiling tools
- Docker containerization
