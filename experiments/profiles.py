"""Named experiment profiles for notebook and script-based runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

from model import ModelConfig


DatasetFamily = Literal["openwebtext_stream", "huggingface_lm"]


@dataclass(frozen=True)
class ExperimentProfile:
    """Single source of truth for one runnable experiment configuration."""

    name: str
    description: str
    dataset_family: DatasetFamily
    dataset_name: str
    dataset_config: Optional[str]
    model_config: ModelConfig
    total_steps: int
    batch_size: int
    block_size: int
    text_column: str = "text"
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 200
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    val_batches: int = 20
    probe_batch_size: int = 16
    n_general: int = 500
    n_induction: int = 100
    n_pairs: int = 50
    n_general_holdout: int = 100
    n_induction_holdout: int = 20
    n_pairs_holdout: int = 10
    n_calibration_seeds: int = 3
    checkpoint_steps: Tuple[int, ...] = field(default_factory=tuple)
    early_stopping_patience_ckpts: Optional[int] = None
    min_steps_before_early_stop: int = 0

    @property
    def model_size_label(self) -> str:
        return "6m" if self.model_config.ablation_mode else "15m"

    @property
    def dataset_label(self) -> str:
        if self.dataset_config:
            return self.dataset_config.replace("-raw-v1", "").replace("_", "-")
        return self.dataset_name.split("/")[-1]

    @property
    def total_tokens(self) -> int:
        return self.total_steps * self.batch_size * self.block_size


PROFILE_REGISTRY: Dict[str, ExperimentProfile] = {
    "wikitext103_15m_preliminary": ExperimentProfile(
        name="wikitext103_15m_preliminary",
        description="Notebook-friendly preliminary run on WikiText-103 with the primary 15M architecture.",
        dataset_family="huggingface_lm",
        dataset_name="Salesforce/wikitext",
        dataset_config="wikitext-103-raw-v1",
        text_column="text",
        model_config=ModelConfig.small_15m(),
        total_steps=12_000,
        batch_size=64,
        block_size=256,
        max_lr=3e-4,
        min_lr=3e-5,
        warmup_steps=500,
        val_batches=20,
        probe_batch_size=64,
        n_general=240,
        n_induction=64,
        n_pairs=32,
        n_general_holdout=64,
        n_induction_holdout=16,
        n_pairs_holdout=8,
        n_calibration_seeds=3,
        checkpoint_steps=(0, 50, 100, 200, 400, 800, 1200, 1800, 2500, 3200, 4000, 6000, 9000, 12000),
        early_stopping_patience_ckpts=3,
        min_steps_before_early_stop=2500,
    ),
    "ptb_15m_comparison": ExperimentProfile(
        name="ptb_15m_comparison",
        description="Dataset-comparison run on Penn Treebank with the primary 15M architecture.",
        dataset_family="huggingface_lm",
        dataset_name="ptb-text-only/ptb_text_only",
        dataset_config="penn_treebank",
        text_column="sentence",
        model_config=ModelConfig.small_15m(),
        total_steps=12_000,
        batch_size=64,
        block_size=256,
        max_lr=3e-4,
        min_lr=3e-5,
        warmup_steps=500,
        val_batches=20,
        probe_batch_size=64,
        n_general=96,
        n_induction=24,
        n_pairs=12,
        n_general_holdout=24,
        n_induction_holdout=8,
        n_pairs_holdout=4,
        n_calibration_seeds=3,
        checkpoint_steps=(0, 50, 100, 200, 400, 800, 1200, 1800, 2500, 3200, 4000, 6000, 9000, 12000),
        early_stopping_patience_ckpts=3,
        min_steps_before_early_stop=2500,
    ),
    "openwebtext_15m_main": ExperimentProfile(
        name="openwebtext_15m_main",
        description="Main-scale 15M run on OpenWebText using the repo's streaming trainer.",
        dataset_family="openwebtext_stream",
        dataset_name="openwebtext",
        dataset_config=None,
        text_column="text",
        model_config=ModelConfig.small_15m(),
        total_steps=100_000,
        batch_size=32,
        block_size=256,
        max_lr=3e-4,
        min_lr=3e-5,
        warmup_steps=200,
        val_batches=20,
        probe_batch_size=16,
        n_general=500,
        n_induction=100,
        n_pairs=50,
        n_general_holdout=100,
        n_induction_holdout=20,
        n_pairs_holdout=10,
        n_calibration_seeds=3,
    ),
    "openwebtext_6m_ablation": ExperimentProfile(
        name="openwebtext_6m_ablation",
        description="Scale ablation on OpenWebText with the 6M architecture.",
        dataset_family="openwebtext_stream",
        dataset_name="openwebtext",
        dataset_config=None,
        text_column="text",
        model_config=ModelConfig.ablation_6m(),
        total_steps=100_000,
        batch_size=32,
        block_size=256,
        max_lr=3e-4,
        min_lr=3e-5,
        warmup_steps=200,
        val_batches=20,
        probe_batch_size=16,
        n_general=500,
        n_induction=100,
        n_pairs=50,
        n_general_holdout=100,
        n_induction_holdout=20,
        n_pairs_holdout=10,
        n_calibration_seeds=3,
    ),
}


def get_profile(name: str) -> ExperimentProfile:
    """Return a named experiment profile."""

    try:
        return PROFILE_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(PROFILE_REGISTRY))
        raise KeyError(f"Unknown profile '{name}'. Available profiles: {available}") from exc


def list_profiles() -> List[ExperimentProfile]:
    """Return all profiles in deterministic order."""

    return [PROFILE_REGISTRY[name] for name in sorted(PROFILE_REGISTRY)]
