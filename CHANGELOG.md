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
- Scientific controls (threshold sensitivity, inter-seed agreement)
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

### Planned
- Additional visualization options
- Extended analysis for layer stratification (H3)
- Heldout probe set validation
- Performance profiling tools
- Docker containerization
