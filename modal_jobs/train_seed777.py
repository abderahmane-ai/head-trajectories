"""
modal_jobs/train_seed777.py — Modal training job for seed 777 (replication run 2).

Third and final primary seed. Same architecture and hyperparameters.
Reuses the shared probe dataset Volume.

Run with:
    modal run modal_jobs/train_seed777.py
"""

from pathlib import Path
import modal
from modal.mount import Mount

APP_NAME          = "trajectories-seed777"
VOLUME_NAME       = "trajectories-ckpts-seed777"
PROBE_VOLUME_NAME = "trajectories-probe"

app = modal.App(APP_NAME)

ckpt_volume  = modal.Volume.from_name(VOLUME_NAME,       create_if_missing=True)
probe_volume = modal.Volume.from_name(PROBE_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.3.0", "numpy", "tiktoken",
        "datasets", "transformers", "huggingface_hub", "tqdm",
    )
)

CKPT_DIR  = Path("/checkpoints/seed777")
PROBE_DIR = Path("/probe")
CODE_DIR  = Path("/code")

SEED         = 777
TOTAL_STEPS  = 100_000
BATCH_SIZE   = 32
BLOCK_SIZE   = 256
MAX_LR       = 3e-4
MIN_LR       = 3e-5
WARMUP_STEPS = 200
WEIGHT_DECAY = 0.1
GRAD_CLIP    = 1.0
VAL_EVERY    = 5
VAL_BATCHES  = 20


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60 * 8,
    volumes={
        str(CKPT_DIR.parent): ckpt_volume,
        str(PROBE_DIR):       probe_volume,
    },
    mounts=[
        Mount.from_local_dir(".", remote_path=str(CODE_DIR))
    ],
)
def train() -> None:
    """Full training run for seed 777 on A100."""

    import sys
    sys.path.insert(0, str(CODE_DIR))

    import torch
    from model import ModelConfig
    from data import OpenWebTextStream, load_probe_dataset, verify_induction_probes
    from training import Trainer

    device = torch.device("cuda")

    print(f"\n{'=' * 64}")
    print(f"  Developmental Trajectories — Seed 777 Training Run")
    print(f"  Device : {device} ({torch.cuda.get_device_name(0)})")
    print(f"{'=' * 64}\n")

    probe_path = PROBE_DIR / "probe_dataset.pt"
    if not probe_path.exists():
        raise FileNotFoundError(
            "Probe dataset not found. Run train_seed42.py first to build it."
        )

    probe_dict = load_probe_dataset(probe_path)
    verify_induction_probes(probe_dict)

    config       = ModelConfig.small_15m()
    train_stream = OpenWebTextStream("train", BLOCK_SIZE, SEED)
    val_stream   = OpenWebTextStream("val",   BLOCK_SIZE, SEED)

    trainer = Trainer(
        config=config,         ckpt_dir=CKPT_DIR,
        seed=SEED,             total_steps=TOTAL_STEPS,
        batch_size=BATCH_SIZE, block_size=BLOCK_SIZE,
        max_lr=MAX_LR,         min_lr=MIN_LR,
        warmup_steps=WARMUP_STEPS, weight_decay=WEIGHT_DECAY,
        grad_clip=GRAD_CLIP,   val_every=VAL_EVERY,
        val_batches=VAL_BATCHES, device=device,
    )

    trainer.resume_if_possible()

    if trainer.step >= TOTAL_STEPS:
        print(f"  [INFO] Training already complete at step {trainer.step:,}.")
        return

    history = trainer.train(train_stream, val_stream)

    ckpt_volume.commit()
    print(f"  [Volume] Committed. Total checkpoints: {trainer.ckpt_counter}")


@app.local_entrypoint()
def main() -> None:
    print("Submitting seed-777 training job to Modal...")
    train.remote()
    print("Job submitted.")
