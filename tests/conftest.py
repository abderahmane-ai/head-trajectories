"""
Pytest configuration and shared fixtures.
"""

import shutil
import uuid
from pathlib import Path

import pytest
import torch
import numpy as np


@pytest.fixture
def seed():
    """Set random seeds for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
    return 42


@pytest.fixture
def small_config():
    """Small model config for fast testing."""
    from model import ModelConfig
    return ModelConfig(
        n_layers=2,
        n_heads=4,
        d_model=64,
        d_ffn=256,
        block_size=32,
    )


@pytest.fixture
def fake_attention():
    """Generate fake attention weights for testing."""
    def _make_attention(N, T, pattern="random"):
        if pattern == "random":
            return torch.softmax(torch.randn(N, T, T), dim=-1)
        elif pattern == "sink":
            attn = torch.zeros(N, T, T)
            attn[:, :, 0] = 1.0
            return attn
        elif pattern == "prev_token":
            attn = torch.zeros(N, T, T)
            for i in range(1, T):
                attn[:, i, i-1] = 1.0
            return attn
        elif pattern == "uniform":
            attn = torch.ones(N, T, T)
            attn = torch.tril(attn)
            return attn / attn.sum(dim=-1, keepdim=True)
        else:
            raise ValueError(f"Unknown pattern: {pattern}")
    
    return _make_attention


@pytest.fixture
def fake_result():
    """Generate a fake probing result dict."""
    def _make_result(n_ckpts=10, n_layers=4, n_heads=8, seed=42):
        return {
            "label_tensor": torch.randint(0, 6, (n_ckpts, n_layers, n_heads)),
            "score_tensor": torch.rand(n_ckpts, n_layers, n_heads, 5),
            "step_index": list(range(0, n_ckpts * 100, 100)),
            "seed": seed,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "thresholds": [0.4, 0.5, 0.3, 0.7, 0.3],
        }
    
    return _make_result


@pytest.fixture
def workspace_tmpdir():
    """Create a repo-local temporary directory for filesystem tests."""
    root = Path(__file__).resolve().parent.parent / ".codex_tmp" / "pytest"
    root.mkdir(parents=True, exist_ok=True)
    path = root / str(uuid.uuid4())
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
