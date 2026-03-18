"""
modal_jobs/train_ablation.py — Modal training job for the 6M scale ablation.

Uses the smaller ModelConfig.ablation_6m() configuration (6 layers, 256 hidden dim,
~6M parameters) with seed 42. Trains for the same number of steps as the primary
runs to allow direct step-aligned comparison of developmental ordering.

Purpose: test whether the developmental ordering of head types (H2, H3) is
scale-invariant — i.e., does a smaller model follow the same sequence?

Run with:
    modal run modal_jobs/train_ablation.py
"""

from pathlib import Path
import modal
from modal.mount import Mount

APP_NAME          = "trajectories-ablation-6m"
VOLUME_NAME       = "trajectories-ckpts-ablation6m"
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

CKPT_DIR  = Path("/checkpoints/ablation_6m")
PROBE_DIR = Path("/probe")
CODE_DIR  = Path("/code")

# ── Ablation-specific settings ────────────────────────────────────────────
SEED         = 42       # same seed as primary for fair comparison
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
    timeout=60 * 60 * 6,        # 6h — smaller model trains faster
    volumes={
        str(CKPT_DIR.parent): ckpt_volume,
        str(PROBE_DIR):       probe_volume,
    },
    mounts=[
        Mount.from_local_dir(".", remote_path=str(CODE_DIR))
    ],
)
def train() -> None:
    """Scale ablation: 6M parameter model, seed 42, A100."""

    import sys
    sys.path.insert(0, str(CODE_DIR))

    import torch
    from model import ModelConfig
    from data import OpenWebTextStream, load_probe_dataset, verify_induction_probes
    from training import Trainer

    device = torch.device("cuda")

    print(f"\n{'=' * 64}")
    print(f"  Developmental Trajectories — 6M Ablation Run")
    print(f"  Device : {device} ({torch.cuda.get_device_name(0)})")
    print(f"{'=' * 64}\n")

    # ── Load shared probe dataset ────────────────────────────────────────────
    probe_path = PROBE_DIR / "probe_dataset.pt"
    if not probe_path.exists():
        raise FileNotFoundError(
            "Probe dataset not found. Run train_seed42.py first to build it."
        )

    probe_dict = load_probe_dataset(probe_path)
    verify_induction_probes(probe_dict)

    # ── Ablation model config ────────────────────────────────────────────────
    config = ModelConfig.ablation_6m()
    print(f"  [Model] ABLATION CONFIG: {config}")
    print(f"  [Model] This is the 6M scale ablation — NOT the primary 15M model.\n")

    # ── Streams and trainer ──────────────────────────────────────────────────
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
        print(f"  [INFO] Ablation training already complete at step {trainer.step:,}.")
        return

    history = trainer.train(train_stream, val_stream)

    ckpt_volume.commit()
    print(f"  [Volume] Committed. Total checkpoints: {trainer.ckpt_counter}")


@app.local_entrypoint()
def main() -> None:
    print("Submitting 6M ablation training job to Modal...")
    train.remote()
    print("Job submitted.")
