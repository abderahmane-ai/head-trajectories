import numpy as np
import torch

from analysis.controls import _reclassify_result
from analysis.phase_transition import extract_val_loss_curve


def test_reclassify_result_updates_active_behavior_tensor():
    score_tensor = torch.tensor([[[[0.5, 0.0, 0.0, 0.0, 0.0]]]], dtype=torch.float32)
    result = {
        "score_tensor": score_tensor,
        "label_tensor": torch.zeros((1, 1, 1), dtype=torch.int32),
        "dominant_label_tensor": torch.zeros((1, 1, 1), dtype=torch.int32),
        "active_behavior_tensor": torch.zeros((1, 1, 1, 5), dtype=torch.bool),
        "step_index": [0],
        "seed": 42,
        "n_layers": 1,
        "n_heads": 1,
        "raw_thresholds": np.array([0.1] * 5, dtype=np.float32),
        "dominance_margin": 0.5,
        "pooled_null_scores": np.zeros((99, 5), dtype=np.float32),
    }

    reclassified = _reclassify_result(result, alpha=0.10)

    assert bool(reclassified["active_behavior_tensor"][0, 0, 0, 0]) is True
    assert int(reclassified["behavior_count_tensor"][0, 0, 0]) == 1
    assert int(reclassified["label_tensor"][0, 0, 0]) == 2


def test_extract_val_loss_curve_averages_multiple_checkpoint_dirs(workspace_tmpdir):
    seed42_dir = workspace_tmpdir / "seed42"
    seed123_dir = workspace_tmpdir / "seed123"
    seed42_dir.mkdir()
    seed123_dir.mkdir()

    torch.save({"step": 0, "val_loss": 10.0}, seed42_dir / "ckpt_0000000.pt")
    torch.save({"step": 100, "val_loss": 8.0}, seed42_dir / "ckpt_0000100.pt")
    torch.save({"step": 0, "val_loss": 12.0}, seed123_dir / "ckpt_0000000.pt")
    torch.save({"step": 100, "val_loss": 6.0}, seed123_dir / "ckpt_0000100.pt")

    curve = extract_val_loss_curve(
        run_results=[
            {"step_index": [0, 100]},
            {"step_index": [0, 100]},
        ],
        ckpt_dir=[seed42_dir, seed123_dir],
    )

    assert curve["steps"].tolist() == [0, 100]
    assert curve["val_loss"][0] == 11.0
    assert curve["val_loss"][1] == 7.0
