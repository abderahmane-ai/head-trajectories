"""
modal_jobs/train_seed42.py — Modal training job for seed 42 (primary run).

Launches the full training run on a Modal A100 GPU with:
- Persistent checkpoint storage on a named Modal Volume
- Automatic resume from latest checkpoint if interrupted
- Probe dataset construction before training if not already built
- All training hyperparameters fixed for reproducibility

Run with:
    modal run modal_jobs/train_seed42.py
"""

import sys
import time
from pathlib import Path

import modal
from modal.mount import Mount

# ─────────────────────────────────────────────────────────────────────────────
# Modal app and infrastructure
# ─────────────────────────────────────────────────────────────────────────────

APP_NAME    = "trajectories-seed42"
VOLUME_NAME = "trajectories-ckpts-seed42"
PROBE_VOLUME_NAME = "trajectories-probe"

app = modal.App(APP_NAME)

# Persistent volumes — survive between runs
ckpt_volume  = modal.Volume.from_name(VOLUME_NAME,  create_if_missing=True)
probe_volume = modal.Volume.from_name(PROBE_VOLUME_NAME, create_if_missing=True)

# Container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.3.0",
        "numpy",
        "tiktoken",
        "datasets",
        "transformers",   # for dataset loading utilities only
        "huggingface_hub",
        "tqdm",
    )
)

# Mount paths inside the container
CKPT_DIR  = Path("/checkpoints/seed42")
PROBE_DIR = Path("/probe")
CODE_DIR  = Path("/code")

# ─────────────────────────────────────────────────────────────────────────────
# Training hyperparameters (seed 42 — primary 15M model)
# ─────────────────────────────────────────────────────────────────────────────

SEED         = 42
TOTAL_STEPS  = 100_000
BATCH_SIZE   = 32
BLOCK_SIZE   = 256
MAX_LR       = 3e-4
MIN_LR       = 3e-5
WARMUP_STEPS = 200
WEIGHT_DECAY = 0.1
GRAD_CLIP    = 1.0
VAL_EVERY    = 1
VAL_BATCHES  = 20


# ─────────────────────────────────────────────────────────────────────────────
# Modal function
# ─────────────────────────────────────────────────────────────────────────────

@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60 * 8,        # 8 hour maximum (training ~5h, buffer for resume)
    volumes={
        str(CKPT_DIR.parent):  ckpt_volume,
        str(PROBE_DIR):        probe_volume,
    },
    mounts=[
        Mount.from_local_dir(".", remote_path=str(CODE_DIR))
    ],
)
def train() -> None:
    """Full training run for seed 42 on A100."""

    import sys
    sys.path.insert(0, str(CODE_DIR))

    import torch

    from model import ModelConfig, TransformerLM
    from data import (
        OpenWebTextStream,
        build_probe_dataset,
        load_probe_dataset,
        verify_induction_probes,
    )
    from training import Trainer

    # ── Setup ────────────────────────────────────────────────────────────────
    device = torch.device("cuda")

    print(f"\n{'=' * 64}")
    print(f"  Developmental Trajectories — Seed 42 Training Run")
    print(f"  Device : {device} ({torch.cuda.get_device_name(0)})")
    print(f"  PyTorch: {torch.__version__}")
    print(f"{'=' * 64}\n")

    # ── Probe dataset (build once, reuse across all seeds) ───────────────────
    probe_path = PROBE_DIR / "probe_dataset.pt"
    print("  [Probe] Checking probe dataset...")

    probe_dict = build_probe_dataset(
        output_path=probe_path,
        block_size=BLOCK_SIZE,
        seed=0,                  # always seed 0 — immutable
        cache_dir=None,
    )
    verify_induction_probes(probe_dict)

    # Commit probe volume so other seeds can reuse it
    probe_volume.commit()
    print("  [Probe] Probe dataset ready and committed to volume.\n")

    # ── Model config ─────────────────────────────────────────────────────────
    config = ModelConfig.small_15m()
    print(f"  [Model] {config}")

    # ── Data streams ─────────────────────────────────────────────────────────
    print("  [Data] Initializing data streams...")
    train_stream = OpenWebTextStream(
        split="train", block_size=BLOCK_SIZE, seed=SEED
    )
    val_stream = OpenWebTextStream(
        split="val", block_size=BLOCK_SIZE, seed=SEED
    )

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = Trainer(
        config=config,
        ckpt_dir=CKPT_DIR,
        seed=SEED,
        total_steps=TOTAL_STEPS,
        batch_size=BATCH_SIZE,
        block_size=BLOCK_SIZE,
        max_lr=MAX_LR,
        min_lr=MIN_LR,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        grad_clip=GRAD_CLIP,
        val_every=VAL_EVERY,
        val_batches=VAL_BATCHES,
        device=device,
    )

    # ── Resume if checkpoint exists ──────────────────────────────────────────
    trainer.resume_if_possible()

    if trainer.step >= TOTAL_STEPS:
        print(f"  [INFO] Training already complete at step {trainer.step:,}. Exiting.")
        return

    # ── Train ────────────────────────────────────────────────────────────────
    history = trainer.train(train_stream, val_stream)

    # ── Commit checkpoints to volume ─────────────────────────────────────────
    print("  [Volume] Committing checkpoints to Modal Volume...")
    ckpt_volume.commit()
    print(f"  [Volume] Committed. Total checkpoints: {trainer.ckpt_counter}")


@app.local_entrypoint()
def main() -> None:
    """Local entrypoint — dispatches training to Modal."""
    print("Submitting seed-42 training job to Modal...")
    train.remote()
    print("Job submitted.")
