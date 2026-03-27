# Test Suite

Comprehensive unit tests for the Head Trajectories codebase.

## Running Tests

### Run all tests
```bash
python run_tests.py
```

Or directly with pytest:
```bash
pytest tests/ -v
```

### Run specific test files
```bash
pytest tests/test_scores.py -v
pytest tests/test_classifier.py -v
pytest tests/test_model.py -v
```

### Run specific test classes or functions
```bash
pytest tests/test_scores.py::TestSinkScore -v
pytest tests/test_scores.py::TestSinkScore::test_perfect_sink -v
```

### Run with coverage
```bash
pytest tests/ --cov=probing --cov=model --cov=analysis --cov-report=html
```

Then open `htmlcov/index.html` to view the coverage report.

## Test Structure

```
tests/
├── __init__.py              # Package marker
├── conftest.py              # Shared fixtures
├── test_scores.py           # Scoring function tests (5 functions)
├── test_calibration.py      # Calibration null + threshold smoke tests
├── test_classifier.py       # Classification logic tests
├── test_model.py            # Transformer architecture tests
├── test_trajectories.py     # Trajectory analysis tests
└── test_integration.py      # End-to-end workflow tests
```

## Test Coverage

### probing/scores.py
- ✅ `sink_score` - perfect sink, uniform, various shapes
- ✅ `prev_token_score` - perfect prev-token, no prev-token, uniform
- ✅ `induction_score` - perfect induction, no induction, bounds checking
- ✅ `positional_score` - identical patterns, random patterns, content-dependent
- ✅ `semantic_score` - perfect alignment, random attention, vectorization correctness, degenerate cases
- ✅ `score_head` - returns 5 scores, score ranges

### probing/classifier.py
- ✅ `classify_head` - below threshold, clear types, tie detection, custom thresholds
- ✅ `HeadClassifier` - initialization, record/classify, tie logging, save/load, custom thresholds
- ✅ Threshold validation - finite checks, non-positive sanitization, persistence
- ✅ Constants - HEAD_TYPES, THRESHOLDS, label constants

### model/transformer.py
- ✅ `ModelConfig` - default config, head_dim computation, 15M/6M configs
- ✅ `TransformerLM` - initialization, forward pass, attention extraction, causal masking, embedding extraction, parameter count, weight tying, sequence length validation

### analysis/trajectories.py
- ✅ `compute_global_curves` - single/multiple seeds, fraction sums
- ✅ `compute_per_layer_curves` - shapes, per-layer fractions
- ✅ `compute_head_trajectories` - extraction, keys
- ✅ `find_interesting_trajectories` - filter by changes
- ✅ `compute_specialization_onset` - onset detection, never appears

### Integration Tests
- ✅ End-to-end probing workflow (extract → score → classify)
- ✅ Full classification workflow (multiple checkpoints)

### Calibration
- ✅ New causal key-scramble null preserves causal row-stochasticity
- ✅ Regression test showing row shuffling is invalid for the sink metric
- ✅ Calibration smoke test for finite positive thresholds

## Fixtures

Shared fixtures in `conftest.py`:

- `seed` - Sets random seeds for reproducibility
- `small_config` - Small ModelConfig for fast testing
- `fake_attention` - Generates attention patterns (random, sink, prev_token, uniform)
- `fake_result` - Generates fake probing result dicts

## Writing New Tests

### Example test structure:
```python
import pytest
from your_module import your_function

class TestYourFunction:
    """Test your_function."""
    
    def test_basic_case(self):
        """Test basic functionality."""
        result = your_function(input_data)
        assert result == expected_output
    
    def test_edge_case(self):
        """Test edge case handling."""
        result = your_function(edge_case_input)
        assert result is not None
    
    def test_error_handling(self):
        """Test that errors are raised correctly."""
        with pytest.raises(ValueError):
            your_function(invalid_input)
```

### Using fixtures:
```python
def test_with_fixture(fake_attention):
    """Test using a fixture."""
    attn = fake_attention(N=10, T=16, pattern="sink")
    assert attn.shape == (10, 16, 16)
```

## Continuous Integration

To run tests in CI:

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v --cov
```

## Performance

Test suite runs in ~10-30 seconds on a typical laptop.

For faster iteration during development:
```bash
pytest tests/test_scores.py -v  # Run only score tests (~5s)
```

## Troubleshooting

### Import errors
Make sure you're running from the project root:
```bash
cd /path/to/Head\ Trajectories
python run_tests.py
```

### Slow tests
Use `-k` to run specific tests:
```bash
pytest tests/ -k "not slow" -v
```

### Debugging failures
Use `--pdb` to drop into debugger on failure:
```bash
pytest tests/test_scores.py --pdb
```

Or use `-s` to see print statements:
```bash
pytest tests/test_scores.py -s -v
```
