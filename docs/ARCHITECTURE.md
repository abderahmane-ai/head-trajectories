# Architecture Overview

This document provides a high-level overview of the codebase architecture.

## Directory Structure

```
head-trajectories/
├── model/              # Transformer implementation
│   ├── config.py       # Model configuration
│   ├── transformer.py  # LLaMA-style transformer
│   ├── rope.py         # Rotary positional embeddings
│   └── rmsnorm.py      # RMS normalization
│
├── data/               # Data loading and preprocessing
│   ├── loader.py       # Streaming text loader for long OpenWebText runs
│   ├── probe.py        # Probe dataset construction
│   └── calibration.py  # Threshold calibration
│
├── training/           # Training loop
│   ├── trainer.py      # Main training logic
│   └── scheduler.py    # Learning rate scheduling
│
├── probing/            # Attention probing pipeline
│   ├── extractor.py    # Attention map extraction
│   ├── scores.py       # Five scoring functions
│   ├── classifier.py   # Head type classification
│   └── pipeline.py     # Full probing orchestration
│
├── analysis/           # Scientific analysis
│   ├── trajectories.py # Developmental curves
│   ├── stability.py    # Type stability analysis
│   ├── phase_transition.py  # Induction emergence
│   └── controls.py     # Scientific controls
│
├── visualization/      # Publication figures
│   ├── timeline_plot.py     # Main trajectory figure
│   ├── heatmap.py           # Layer stratification
│   ├── phase_plot.py        # Phase transition
│   └── stability_hist.py    # Stability histogram
│
├── modal_jobs/         # Cloud training jobs
│   ├── train_seed42.py
│   ├── train_seed123.py
│   ├── train_seed777.py
│   └── train_ablation.py
│
├── tests/              # Unit tests
│   ├── test_scores.py
│   ├── test_classifier.py
│   ├── test_model.py
│   ├── test_trajectories.py
│   └── test_integration.py
│
├── run_probing.py      # Entry point: probing pipeline
├── run_analysis.py     # Entry point: analysis + figures
└── run_tests.py        # Entry point: test suite
```

## Data Flow

### Training Phase
```
Dataset/Profile → Tokenization → Batching → Model → Loss → Optimizer
                                                ↓
                                          Checkpoints
                                      (profile-specific)
```

### Probing Phase
```
Checkpoint → Load Model → Extract Attention → Score Heads → Classify
                                                               ↓
                                                          Results.pt
```

### Analysis Phase
```
Results.pt → Compute Trajectories → Generate Figures → Hypothesis Tests
```

## Key Design Decisions

### 1. Attention Extraction
- Uses `return_attention=True` flag in forward pass
- Attention maps detached and moved to CPU immediately
- No gradient computation during probing (eval mode)

### 2. Scoring Functions
- Each function is independent and stateless
- Operates on single head's attention maps
- Returns scalar score in [0, 1] (except semantic: [-1, 1])

### 3. Classification
- Threshold normalization: `scores / thresholds`
- Argmax over normalized scores
- Tie detection with tolerance (0.05)
- UNDIFFERENTIATED fallback

### 4. Checkpoint Schedule
- Profile-specific, but always dense early relative to late
- Short comparison profiles save 14 checkpoints over 12k steps
- Long OpenWebText profiles save many more checkpoints over 100k steps
- Optimizes for early developmental dynamics

### 5. Probe Dataset
- Immutable after construction
- Held-out from training data
- Three types: general, induction, positional
- Saved once per profile / calibration version, then reused across checkpoints for that run configuration

## Module Dependencies

```
model/
  └─ (no internal dependencies)

data/
  └─ model.config (for tokenizer vocab size)

training/
  ├─ model.*
  ├─ data.loader
  └─ training.scheduler

probing/
  ├─ model.*
  ├─ data.probe
  ├─ probing.extractor → probing.scores → probing.classifier
  └─ probing.pipeline (orchestrates all)

analysis/
  ├─ probing.classifier (for HEAD_TYPES, labels)
  └─ analysis.trajectories → analysis.{stability, phase_transition, controls}

visualization/
  └─ analysis.* (consumes analysis outputs)
```

## Extension Points

### Adding a New Scoring Function
1. Add function to `probing/scores.py`
2. Update `score_head()` to call it
3. Update `THRESHOLDS` in `probing/classifier.py`
4. Add corresponding tests in `tests/test_scores.py`

### Adding a New Head Type
1. Add label constant to `probing/classifier.py`
2. Update `HEAD_TYPES` list
3. Update `classify_head()` logic
4. Add color to visualization configs

### Adding a New Analysis
1. Create module in `analysis/`
2. Import trajectory data via `load_run_results()`
3. Compute metrics
4. Add visualization in `visualization/`
5. Integrate into `run_analysis.py`

## Performance Considerations

### Bottlenecks
- ✅ **Fixed**: `semantic_score` was 800M Python ops → now vectorized
- Attention extraction: batched to avoid OOM
- Checkpoint I/O: atomic writes with temp files

### Memory Management
- Attention maps moved to CPU immediately after extraction
- Probing runs on CPU (no GPU needed)
- Batch size configurable for memory constraints

### Parallelization
- Multiple seeds can run in parallel on Modal
- Probing pipeline is sequential per run (by design)
- Analysis can process multiple runs in parallel

## Testing Strategy

### Unit Tests
- Each scoring function tested independently
- Classification logic tested with synthetic data
- Model architecture tested with small configs

### Integration Tests
- End-to-end probing workflow
- Full classification pipeline
- Save/load round-trips

### Fixtures
- `fake_attention`: generates test attention patterns
- `fake_result`: generates synthetic probing results
- `small_config`: fast model config for testing

## Scientific Reproducibility

### Fixed Seeds
- Training: seed per run (42, 123, 777)
- Probe dataset: seed 0 (immutable)
- Analysis: deterministic (no randomness)

### Immutable Data
- Probe dataset is fixed after construction for a given profile and calibration version
- Checkpoints never modified after saving
- Results files append-only (with resumption)

### Version Control
- All hyperparameters in `ModelConfig`
- Thresholds stored in results files
- Git commit hash could be added to checkpoints
