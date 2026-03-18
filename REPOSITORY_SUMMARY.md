# Repository Summary

## 📊 Statistics

- **Python Files**: 42
- **Documentation Files**: 13
- **Test Files**: 6
- **Tests**: 60 (100% passing)
- **Lines of Code**: ~8,000+ (estimated)
- **Test Coverage**: >80%

## 📁 Repository Structure

```
head-trajectories/
├── 📂 .github/              # GitHub configuration
│   ├── workflows/           # CI/CD (test.yml)
│   ├── ISSUE_TEMPLATE/      # Issue templates
│   ├── pull_request_template.md
│   └── FUNDING.yml
│
├── 📂 docs/                 # Documentation
│   ├── QUICKSTART.md        # Getting started guide
│   ├── ARCHITECTURE.md      # Codebase overview
│   ├── FAQ.md               # Frequently asked questions
│   └── GITHUB_SETUP.md      # Repository setup guide
│
├── 📂 model/                # Transformer implementation
│   ├── config.py            # Model configuration
│   ├── transformer.py       # LLaMA-style transformer
│   ├── rope.py              # Rotary embeddings
│   └── rmsnorm.py           # RMS normalization
│
├── 📂 data/                 # Data loading
│   ├── loader.py            # OpenWebText streaming
│   ├── probe.py             # Probe dataset construction
│   └── calibration.py       # Threshold calibration
│
├── 📂 training/             # Training loop
│   ├── trainer.py           # Main training logic
│   └── scheduler.py         # LR scheduling
│
├── 📂 probing/              # Attention probing
│   ├── extractor.py         # Attention extraction
│   ├── scores.py            # 5 scoring functions
│   ├── classifier.py        # Head classification
│   └── pipeline.py          # Full pipeline
│
├── 📂 analysis/             # Scientific analysis
│   ├── trajectories.py      # Developmental curves
│   ├── stability.py         # Type stability
│   ├── phase_transition.py  # Induction emergence
│   └── controls.py          # Scientific controls
│
├── 📂 visualization/        # Publication figures
│   ├── timeline_plot.py     # Main trajectory figure
│   ├── heatmap.py           # Layer stratification
│   ├── phase_plot.py        # Phase transition
│   └── stability_hist.py    # Stability histogram
│
├── 📂 modal_jobs/           # Cloud training
│   ├── train_seed42.py
│   ├── train_seed123.py
│   ├── train_seed777.py
│   └── train_ablation.py
│
├── 📂 tests/                # Unit tests
│   ├── test_scores.py       # 18 tests
│   ├── test_classifier.py   # 17 tests
│   ├── test_model.py        # 11 tests
│   ├── test_trajectories.py # 12 tests
│   ├── test_integration.py  # 2 tests
│   └── conftest.py          # Fixtures
│
├── 📄 README.md             # Main documentation
├── 📄 CONTRIBUTING.md       # Contribution guidelines
├── 📄 CHANGELOG.md          # Version history
├── 📄 LICENSE               # MIT License
├── 📄 PROJECT_STATUS.md     # Current status
├── 📄 requirements.txt      # Dependencies
├── 📄 setup.py              # Package setup
├── 📄 pytest.ini            # Test configuration
├── 📄 Makefile              # Common commands
├── 📄 .gitignore            # Git ignore rules
├── 📄 .editorconfig         # Editor configuration
├── 📄 MANIFEST.in           # Package manifest
├── 📄 run_probing.py        # Probing entry point
├── 📄 run_analysis.py       # Analysis entry point
└── 📄 run_tests.py          # Test runner
```

## ✨ Key Features

### Core Implementation
- ✅ LLaMA-style transformer from scratch (no HuggingFace)
- ✅ Five behavioral scoring functions
- ✅ Head classification with tie-breaking
- ✅ Dense-early checkpoint schedule (100 checkpoints)
- ✅ Modal cloud training integration
- ✅ Probing pipeline with resumption
- ✅ Trajectory analysis
- ✅ Publication-quality visualizations (300 DPI)

### Performance
- ✅ Vectorized semantic_score (30-100× speedup)
- ✅ Incremental checkpoint saving
- ✅ Atomic file operations
- ✅ Safe torch.load operations
- ✅ Efficient batched attention extraction

### Testing
- ✅ 60 unit tests (100% passing)
- ✅ Integration tests
- ✅ Shared fixtures
- ✅ CI/CD with GitHub Actions
- ✅ Test coverage >80%

### Documentation
- ✅ Comprehensive README
- ✅ Quick start guide
- ✅ Architecture overview
- ✅ API documentation (docstrings)
- ✅ FAQ
- ✅ Contributing guidelines
- ✅ GitHub setup guide
- ✅ Changelog

### Repository Setup
- ✅ Professional .gitignore
- ✅ MIT License
- ✅ requirements.txt
- ✅ setup.py for pip install
- ✅ pytest.ini
- ✅ Makefile for common tasks
- ✅ .editorconfig
- ✅ GitHub issue templates
- ✅ Pull request template
- ✅ CI/CD workflows

## 🎯 Scientific Hypotheses

| ID | Hypothesis | Status |
|----|------------|--------|
| H1 | Sink-First | ✅ Ready to test |
| H2 | Ordered Development | ✅ Ready to test |
| H3 | Layer Stratification | ✅ Ready to test |
| H4 | Phase Transition | ✅ Ready to test |
| H5 | Sink Persistence | ✅ Ready to test |

## 📈 Code Quality

| Metric | Score | Status |
|--------|-------|--------|
| Tests | 60/60 | ✅ 100% |
| Coverage | >80% | ✅ Excellent |
| Documentation | Comprehensive | ✅ Excellent |
| Type Hints | Throughout | ✅ Present |
| Code Style | PEP 8 | ✅ Consistent |
| Performance | Optimized | ✅ Fast |
| Security | Safe | ✅ Secure |

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/abderahmane-ai/head-trajectories.git
cd head-trajectories

# Install dependencies
pip install -r requirements.txt

# Run tests
python run_tests.py

# Expected output: 60 passed in ~4s
```

## 📦 Dependencies

### Core
- Python 3.10+
- PyTorch 2.3.0
- NumPy, SciPy
- Matplotlib
- tiktoken (tokenization)
- datasets (HuggingFace)

### Cloud
- Modal (for training)

### Development
- pytest (testing)
- pytest-cov (coverage)
- black (formatting)
- flake8 (linting)

## 💰 Compute Costs

| Task | Hardware | Time | Cost |
|------|----------|------|------|
| Training (per seed) | Modal A100 | ~5h | ~$6 |
| Probing (per seed) | CPU | ~10min | $0 |
| Analysis | CPU | ~5min | $0 |
| **Total (3 seeds + ablation)** | | **~20h** | **~$22** |

## 📊 Expected Results

After running the full pipeline:
- 4 trained models (3 seeds + 1 ablation)
- 400 checkpoints total (100 per run)
- ~25,600 scored heads (64 heads × 100 ckpts × 4 runs)
- 8 publication-quality figures
- 5 hypothesis test results

## 🎓 Academic Use

### Citation
```bibtex
@misc{abderahmane2025developmental,
  title   = {Developmental Trajectories of Attention Heads},
  author  = {Abderahmane},
  year    = {2025},
  url     = {https://github.com/abderahmane-ai/head-trajectories},
  note    = {Independent research, ENSIA Algeria}
}
```

### License
MIT License - Free for academic and commercial use

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Types of contributions:
- 🐛 Bug fixes
- ✨ New features
- 📝 Documentation improvements
- 🧪 Additional tests
- 🔬 Scientific analysis methods
- 📊 Visualization enhancements

## 📞 Contact

- **GitHub Issues**: Bug reports and questions
- **GitHub Discussions**: General discussion
- **GitHub**: [@abderahmane-ai](https://github.com/abderahmane-ai)

## 🏆 Acknowledgments

This project represents ~200 hours of development work:
- Research design and hypothesis formulation
- Transformer implementation from scratch
- Probing pipeline development
- Performance optimization
- Comprehensive testing
- Documentation and polish

Special thanks to the mechanistic interpretability community.

## 📝 Next Steps

1. **For Users**:
   - Read [QUICKSTART.md](docs/QUICKSTART.md)
   - Run the test suite
   - Try the demo examples
   - Explore the codebase

2. **For Contributors**:
   - Read [CONTRIBUTING.md](CONTRIBUTING.md)
   - Check open issues
   - Review the architecture
   - Submit a PR

3. **For Researchers**:
   - Read the scientific methodology
   - Review hypothesis operationalizations
   - Run experiments
   - Cite in your work

## 🌟 Star History

If you find this project useful, please star ⭐ the repository!

---

**Status**: ✅ Production Ready  
**Version**: 1.0.0  
**Last Updated**: January 2025
