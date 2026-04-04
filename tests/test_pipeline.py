from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from data.calibration import CALIBRATION_VERSION
from probing.pipeline import _load_partial_results, run_probing_pipeline


def test_load_partial_results_returns_valid_payload(workspace_tmpdir):
    output_path = workspace_tmpdir / "partial.pt"
    payload = {
        "label_tensor": torch.zeros((2, 1, 1), dtype=torch.int32),
        "score_tensor": torch.zeros((2, 1, 1, 5), dtype=torch.float32),
        "step_index": [0],
        "seed": 42,
        "n_layers": 1,
        "n_heads": 1,
    }
    torch.save(payload, output_path)

    loaded = _load_partial_results(output_path, n_layers=1, n_heads=1, seed=42)

    assert loaded is not None
    assert loaded["step_index"] == [0]


def test_run_probing_pipeline_raises_on_checkpoint_failure_and_saves_partial(
    monkeypatch,
    workspace_tmpdir,
):
    ckpt_dir = workspace_tmpdir / "checkpoints"
    ckpt_dir.mkdir()
    ckpt_path = ckpt_dir / "ckpt_0000000.pt"
    ckpt_path.write_bytes(b"fake")

    probe_path = workspace_tmpdir / "probe_dataset.pt"
    output_path = workspace_tmpdir / "results.pt"
    ties_log_path = workspace_tmpdir / "ties.csv"

    probe_dict = {
        "general_seqs": torch.zeros((2, 8), dtype=torch.long),
        "induction_seqs": torch.zeros((2, 8), dtype=torch.long),
        "induction_p1": torch.zeros((2,), dtype=torch.long),
        "induction_p2": torch.ones((2,), dtype=torch.long),
        "positional_seqs": torch.zeros((2, 8), dtype=torch.long),
        "positional_pairs": torch.tensor([[0, 1]], dtype=torch.long),
        "creation_seed": torch.tensor(0, dtype=torch.long),
        "block_size": torch.tensor(8, dtype=torch.long),
        "calibration_version": torch.tensor(CALIBRATION_VERSION, dtype=torch.long),
        "calibrated_thresholds_15m": torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1], dtype=torch.float32),
    }

    monkeypatch.setattr("data.load_probe_dataset", lambda path: probe_dict)
    monkeypatch.setattr(
        "probing.pipeline.discover_checkpoints",
        lambda path: [ckpt_path],
    )

    fake_model = SimpleNamespace(
        config=SimpleNamespace(n_layers=1, n_heads=1, ablation_mode=False)
    )
    monkeypatch.setattr(
        "probing.extractor.load_model_from_checkpoint",
        lambda path, device: (fake_model, None, None, None),
    )

    def _boom(**kwargs):
        raise RuntimeError("corrupt checkpoint")

    monkeypatch.setattr("probing.pipeline.extract_checkpoint", _boom)

    with pytest.raises(RuntimeError, match="Failed to process checkpoint"):
        run_probing_pipeline(
            ckpt_dir=ckpt_dir,
            probe_path=probe_path,
            output_path=output_path,
            ties_log_path=ties_log_path,
            seed=42,
            device=torch.device("cpu"),
            batch_size=2,
            resume=False,
        )

    assert output_path.exists()
    saved = torch.load(output_path, weights_only=True)
    assert saved["step_index"] == []
    assert tuple(saved["label_tensor"].shape) == (1, 1, 1)


def test_run_probing_pipeline_allows_heldout_without_natural_induction_keys(
    monkeypatch,
    workspace_tmpdir,
):
    ckpt_dir = workspace_tmpdir / "checkpoints"
    ckpt_dir.mkdir()
    ckpt_path = ckpt_dir / "ckpt_0000000.pt"
    ckpt_path.write_bytes(b"fake")

    probe_path = workspace_tmpdir / "probe_dataset.pt"
    output_path = workspace_tmpdir / "results.pt"
    ties_log_path = workspace_tmpdir / "ties.csv"

    probe_dict = {
        "general_seqs": torch.zeros((2, 8), dtype=torch.long),
        "induction_seqs": torch.zeros((2, 8), dtype=torch.long),
        "induction_p1": torch.zeros((2,), dtype=torch.long),
        "induction_p2": torch.ones((2,), dtype=torch.long),
        "positional_seqs": torch.zeros((2, 8), dtype=torch.long),
        "positional_pairs": torch.tensor([[0, 1]], dtype=torch.long),
        "heldout_general_seqs": torch.zeros((2, 8), dtype=torch.long),
        "heldout_induction_seqs": torch.zeros((2, 8), dtype=torch.long),
        "heldout_induction_p1": torch.zeros((2,), dtype=torch.long),
        "heldout_induction_p2": torch.ones((2,), dtype=torch.long),
        "heldout_positional_seqs": torch.zeros((2, 8), dtype=torch.long),
        "heldout_positional_pairs": torch.tensor([[0, 1]], dtype=torch.long),
        "creation_seed": torch.tensor(0, dtype=torch.long),
        "block_size": torch.tensor(8, dtype=torch.long),
        "calibration_version": torch.tensor(CALIBRATION_VERSION, dtype=torch.long),
        "calibrated_thresholds_15m": torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1], dtype=torch.float32),
    }

    monkeypatch.setattr("data.load_probe_dataset", lambda path: probe_dict)
    monkeypatch.setattr("probing.pipeline.discover_checkpoints", lambda path: [ckpt_path])

    fake_model = SimpleNamespace(
        config=SimpleNamespace(n_layers=1, n_heads=1, ablation_mode=False)
    )
    monkeypatch.setattr(
        "probing.extractor.load_model_from_checkpoint",
        lambda path, device: (fake_model, None, None, None),
    )

    def _boom(**kwargs):
        raise RuntimeError("corrupt checkpoint")

    monkeypatch.setattr("probing.pipeline.extract_checkpoint", _boom)

    with pytest.raises(RuntimeError, match="Failed to process checkpoint"):
        run_probing_pipeline(
            ckpt_dir=ckpt_dir,
            probe_path=probe_path,
            output_path=output_path,
            ties_log_path=ties_log_path,
            seed=42,
            device=torch.device("cpu"),
            batch_size=2,
            resume=False,
            use_heldout=True,
        )
