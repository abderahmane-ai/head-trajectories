import torch

from data.calibration import CALIBRATION_VERSION
from experiments.profiles import get_profile, list_profiles
from experiments.runner import (
    build_or_load_probe_dataset,
    ensure_dirs,
    normalize_run_specs,
    reset_run_artifacts,
    resolve_artifacts,
)


def test_profiles_are_available():
    names = [profile.name for profile in list_profiles()]
    assert "wikitext103_15m_preliminary" in names
    assert "lm1b_15m_comparison" in names
    assert "openwebtext_15m_main" in names
    assert "openwebtext_6m_ablation" in names


def test_lm1b_profile_uses_custom_split_layout():
    profile = get_profile("lm1b_15m_comparison")

    assert profile.dataset_family == "huggingface_lm"
    assert profile.dataset_name == "FrankCCCCC/lm1b"
    assert profile.dataset_config is None
    assert profile.text_column == "text"
    assert profile.train_split == "train"
    assert profile.validation_split == "test"
    assert profile.probe_split == "test"
    assert profile.n_general == get_profile("wikitext103_15m_preliminary").n_general


def test_normalize_run_specs_supports_seed_and_seeds():
    normalized = normalize_run_specs([
        {"profile": "wikitext103_15m_preliminary", "seed": 7},
        {"profile_name": "openwebtext_15m_main", "seeds": [42, 123, 777]},
    ])

    assert normalized[0].profile_name == "wikitext103_15m_preliminary"
    assert normalized[0].seeds == (7,)
    assert normalized[1].profile_name == "openwebtext_15m_main"
    assert normalized[1].seeds == (42, 123, 777)


def test_resolve_artifacts_layout(workspace_tmpdir):
    profile = get_profile("wikitext103_15m_preliminary")
    paths = resolve_artifacts(profile, seed=7, artifact_root=workspace_tmpdir)

    assert paths.profile_dir == workspace_tmpdir / profile.name
    assert paths.probe_path == workspace_tmpdir / profile.name / "probe" / "probe_dataset.pt"
    assert paths.ckpt_dir == workspace_tmpdir / profile.name / "seed7" / "checkpoints"
    assert paths.results_path == workspace_tmpdir / profile.name / "seed7" / "results" / "results_seed7.pt"
    assert paths.figure_path == workspace_tmpdir / profile.name / "seed7" / "figures" / "timeline_seed7.png"


def test_reset_run_artifacts_preserves_probe_by_default(workspace_tmpdir):
    profile = get_profile("wikitext103_15m_preliminary")
    paths = resolve_artifacts(profile, seed=11, artifact_root=workspace_tmpdir)
    ensure_dirs(paths)

    paths.probe_path.write_text("probe", encoding="utf-8")
    (paths.seed_dir / "sentinel.txt").write_text("seed", encoding="utf-8")

    reset_run_artifacts(paths, reset_probe=False)

    assert paths.seed_dir.exists() is False
    assert paths.probe_path.exists() is True


def test_build_or_load_probe_dataset_rebuilds_stale_probe(monkeypatch, workspace_tmpdir):
    profile = get_profile("wikitext103_15m_preliminary")
    paths = resolve_artifacts(profile, seed=13, artifact_root=workspace_tmpdir)
    ensure_dirs(paths)

    stale_probe = {
        "general_seqs": torch.zeros((1, profile.block_size), dtype=torch.long),
        "induction_seqs": torch.zeros((1, profile.block_size), dtype=torch.long),
        "induction_p1": torch.zeros((1,), dtype=torch.long),
        "induction_p2": torch.zeros((1,), dtype=torch.long),
        "positional_seqs": torch.zeros((2, profile.block_size), dtype=torch.long),
        "positional_pairs": torch.tensor([[0, 1]], dtype=torch.long),
        "creation_seed": torch.tensor(0, dtype=torch.long),
        "block_size": torch.tensor(profile.block_size, dtype=torch.long),
        "calibration_version": torch.tensor(CALIBRATION_VERSION - 1, dtype=torch.long),
    }
    torch.save(stale_probe, paths.probe_path)

    rebuilt = {"general_seqs": torch.ones((1, profile.block_size), dtype=torch.long)}

    def fake_build(*args, **kwargs):
        return rebuilt

    monkeypatch.setattr("experiments.runner._build_hf_probe_dataset", fake_build)

    loaded = build_or_load_probe_dataset(profile, paths, device="cpu", rebuild=False)

    assert loaded is rebuilt
