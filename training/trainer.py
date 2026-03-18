"""
training/trainer.py — Full training loop with non-uniform checkpoint schedule.

Implements the complete training loop for the developmental trajectories
experiment. Key responsibilities:

1. Checkpoint saving on the dense-early, sparse-late schedule (NON-NEGOTIABLE)
2. Resume-from-checkpoint logic via Modal Volume inspection
3. Validation loss estimation at each checkpoint
4. Professional training progress logging
5. Seed-specific Volume naming for clean multi-run management

Checkpoint schedule (immutable — this is the core experimental design):
  Steps 0–500:      every 50 steps
  Steps 500–5000:   every 200 steps
  Steps 5000–20000: every 500 steps
  Steps 20000–50000:every 1000 steps
  Steps 50000–end:  every 2000 steps
"""

import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from model import TransformerLM, ModelConfig
from data import OpenWebTextStream, BatchCollator, estimate_val_loss
from .scheduler import CosineScheduler


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint Schedule
# ─────────────────────────────────────────────────────────────────────────────

def should_checkpoint(step: int) -> bool:
    """
    Determine whether a checkpoint should be saved at the given step.

    Schedule:
      Steps 0–500:       every 50
      Steps 500–5000:    every 200
      Steps 5000–20000:  every 500
      Steps 20000–50000: every 1000
      Steps 50000+:      every 2000

    Always checkpoints at step 0 (random initialization baseline).
    """

    if step == 0:
        return True
    if step <= 500:
        return step % 50 == 0
    if step <= 5_000:
        return step % 200 == 0
    if step <= 20_000:
        return step % 500 == 0
    if step <= 50_000:
        return step % 1_000 == 0
    return step % 2_000 == 0


def get_all_checkpoint_steps(total_steps: int) -> List[int]:
    """
    Return a sorted list of all steps where a checkpoint will be saved.
    Used for pre-computing expected checkpoint count.
    """

    return [s for s in range(total_steps + 1) if should_checkpoint(s)]


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint I/O
# ─────────────────────────────────────────────────────────────────────────────

def checkpoint_path(ckpt_dir: Path, step: int) -> Path:
    """Return the path for a checkpoint at the given step."""

    return ckpt_dir / f"ckpt_{step:07d}.pt"


def save_checkpoint(
    model:        TransformerLM,
    optimizer:    torch.optim.Optimizer,
    step:         int,
    train_loss:   float,
    val_loss:     float,
    ckpt_dir:     Path,
    seed:         int,
) -> Path:
    """
    Save a full training checkpoint.

    Checkpoint contains:
      - model state dict
      - optimizer state dict
      - current step
      - train loss at this step
      - val loss at this step
      - training seed
      - model config (for reconstruction without external config file)

    Returns:
        path to the saved checkpoint file
    """

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_path(ckpt_dir, step)

    checkpoint: Dict = {
        "step":        step,
        "seed":        seed,
        "train_loss":  train_loss,
        "val_loss":    val_loss,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "config":      model.config,
    }

    torch.save(checkpoint, path)
    return path


def load_latest_checkpoint(
    ckpt_dir: Path,
    model:    TransformerLM,
    optimizer: torch.optim.Optimizer,
    device:   torch.device,
) -> Tuple[int, float, float]:
    """
    Find and load the latest checkpoint from ckpt_dir.
    If no checkpoint exists, returns (0, inf, inf) — fresh start.

    Returns:
        (step, train_loss, val_loss) at the loaded checkpoint
    """

    if not ckpt_dir.exists():
        print("  [Resume] No checkpoint directory found — starting fresh.")
        return 0, float("inf"), float("inf")

    ckpt_files = sorted(ckpt_dir.glob("ckpt_*.pt"))
    if not ckpt_files:
        print("  [Resume] No checkpoint files found — starting fresh.")
        return 0, float("inf"), float("inf")

    # Latest checkpoint = highest step number
    latest = ckpt_files[-1]
    print(f"  [Resume] Loading checkpoint: {latest.name}")

    try:
        # Register ModelConfig as safe for torch.load
        torch.serialization.add_safe_globals([ModelConfig])
        checkpoint = torch.load(latest, map_location=device, weights_only=True)
    except Exception as e:
        print(f"  [Resume] WARNING: Failed to load {latest.name}: {e}")
        # Try second-to-last checkpoint
        if len(ckpt_files) >= 2:
            fallback = ckpt_files[-2]
            print(f"  [Resume] Trying fallback: {fallback.name}")
            torch.serialization.add_safe_globals([ModelConfig])
            checkpoint = torch.load(fallback, map_location=device, weights_only=True)
        else:
            print("  [Resume] No valid fallback — starting fresh.")
            return 0, float("inf"), float("inf")

    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optim_state"])

    step       = checkpoint["step"]
    train_loss = checkpoint.get("train_loss", float("inf"))
    val_loss   = checkpoint.get("val_loss",   float("inf"))

    print(
        f"  [Resume] Resumed from step {step:,} "
        f"(train_loss={train_loss:.4f}, val_loss={val_loss:.4f})"
    )
    return step, train_loss, val_loss


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    """
    Full training loop for the developmental trajectories experiment.

    Handles:
    - Seed initialization (torch, numpy, random)
    - Model and optimizer construction
    - Checkpoint resume logic
    - Non-uniform checkpoint saving schedule
    - Validation loss estimation
    - Progress logging (formatted, step-aligned)

    Args:
        config:         ModelConfig for the model to train
        ckpt_dir:       directory to save/load checkpoints
        seed:           random seed for full reproducibility
        total_steps:    total training steps
        batch_size:     sequences per batch
        block_size:     tokens per sequence
        max_lr:         peak learning rate
        min_lr:         minimum learning rate
        warmup_steps:   linear warmup duration
        weight_decay:   AdamW weight decay
        grad_clip:      gradient norm clipping value
        val_every:      estimate val loss every N checkpoints (not every N steps)
        val_batches:    number of batches for validation loss estimate
        device:         torch device
        cache_dir:      HuggingFace dataset cache directory
    """

    def __init__(
        self,
        config:       ModelConfig,
        ckpt_dir:     Path,
        seed:         int          = 42,
        total_steps:  int          = 100_000,
        batch_size:   int          = 32,
        block_size:   int          = 256,
        max_lr:       float        = 3e-4,
        min_lr:       float        = 3e-5,
        warmup_steps: int          = 200,
        weight_decay: float        = 0.1,
        grad_clip:    float        = 1.0,
        val_every:    int          = 5,
        val_batches:  int          = 20,
        device:       Optional[torch.device] = None,
        cache_dir:    Optional[Path]         = None,
    ) -> None:
        self.config:       ModelConfig = config
        self.ckpt_dir:     Path        = ckpt_dir
        self.seed:         int         = seed
        self.total_steps:  int         = total_steps
        self.batch_size:   int         = batch_size
        self.block_size:   int         = block_size
        self.max_lr:       float       = max_lr
        self.min_lr:       float       = min_lr
        self.warmup_steps: int         = warmup_steps
        self.weight_decay: float       = weight_decay
        self.grad_clip:    float       = grad_clip
        self.val_every:    int         = val_every
        self.val_batches:  int         = val_batches
        self.cache_dir:    Optional[Path] = cache_dir

        # Device
        self.device: torch.device = device or (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )

        # Set all seeds before any tensor creation
        self._set_seeds(seed)

        # Build model and optimizer
        self.model:     TransformerLM          = TransformerLM(config).to(self.device)
        self.optimizer: torch.optim.AdamW      = self._build_optimizer()
        self.scheduler: CosineScheduler        = CosineScheduler(
            max_lr=max_lr, min_lr=min_lr,
            warmup_steps=warmup_steps, total_steps=total_steps,
        )

        # Training state
        self.step:           int   = 0
        self.last_train_loss: float = float("inf")
        self.last_val_loss:   float = float("inf")
        self.ckpt_counter:    int   = 0   # increments each time we checkpoint

        # Tokens processed counter
        self.tokens_seen: int = 0

        self._print_setup()

    def _set_seeds(self, seed: int) -> None:
        """Set all random seeds for full reproducibility."""

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def _build_optimizer(self) -> torch.optim.AdamW:
        """
        Construct AdamW optimizer with weight decay applied only to
        2D parameters (weight matrices), not to biases or norms.
        """

        decay_params    = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim >= 2:
                decay_params.append(param)
            else:
                no_decay_params.append(param)

        param_groups = [
            {"params": decay_params,    "weight_decay": self.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        return torch.optim.AdamW(
            param_groups,
            lr=self.max_lr,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

    def _print_setup(self) -> None:
        """Print training configuration summary."""

        expected_ckpts = len(get_all_checkpoint_steps(self.total_steps))
        tokens_total   = self.total_steps * self.batch_size * (self.block_size - 1)

        print(f"\n{'=' * 64}")
        print(f"  Developmental Trajectories — Training Setup")
        print(f"{'=' * 64}")
        print(f"  {'Model':<24}: {self.config}")
        print(f"  {'Seed':<24}: {self.seed}")
        print(f"  {'Device':<24}: {self.device}")
        print(f"  {'Total steps':<24}: {self.total_steps:,}")
        print(f"  {'Batch size':<24}: {self.batch_size}")
        print(f"  {'Block size':<24}: {self.block_size}")
        print(f"  {'Max LR':<24}: {self.max_lr:.2e}")
        print(f"  {'Min LR':<24}: {self.min_lr:.2e}")
        print(f"  {'Warmup steps':<24}: {self.warmup_steps:,}")
        print(f"  {'Total tokens':<24}: {tokens_total / 1e6:.1f}M")
        print(f"  {'Expected checkpoints':<24}: {expected_ckpts}")
        print(f"  {'Checkpoint dir':<24}: {self.ckpt_dir}")
        print(f"{'=' * 64}\n")

    def resume_if_possible(self) -> None:
        """
        Attempt to load the latest checkpoint from ckpt_dir.
        Updates self.step and loss trackers if a checkpoint is found.
        """

        step, train_loss, val_loss = load_latest_checkpoint(
            self.ckpt_dir, self.model, self.optimizer, self.device
        )
        self.step            = step
        self.last_train_loss = train_loss
        self.last_val_loss   = val_loss
        self.tokens_seen     = step * self.batch_size * (self.block_size - 1)

        if step > 0:
            # Restore learning rate to the correct value for the resumed step
            self.scheduler.set_lr(self.optimizer, step)

    def _maybe_save_checkpoint(
        self,
        train_loss: float,
        val_loss:   float,
    ) -> Optional[Path]:
        """
        Save a checkpoint if the current step is on the schedule.
        Returns the checkpoint path if saved, else None.
        """

        if not should_checkpoint(self.step):
            return None

        path = save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            step=self.step,
            train_loss=train_loss,
            val_loss=val_loss,
            ckpt_dir=self.ckpt_dir,
            seed=self.seed,
        )
        self.ckpt_counter += 1
        return path

    def _get_val_loss(
        self,
        val_stream: OpenWebTextStream,
    ) -> float:
        """Estimate validation loss. Only called at checkpoint steps."""

        return estimate_val_loss(
            model=self.model,
            val_stream=val_stream,
            batch_size=self.batch_size,
            n_batches=self.val_batches,
            device=self.device,
        )

    def train(
        self,
        train_stream: OpenWebTextStream,
        val_stream:   OpenWebTextStream,
    ) -> Dict[str, List]:
        """
        Execute the full training loop.

        Args:
            train_stream: OpenWebTextStream for training data
            val_stream:   OpenWebTextStream for validation

        Returns:
            history: dict with keys 'steps', 'train_loss', 'val_loss', 'lr'
        """

        history: Dict[str, List] = {
            "steps":      [],
            "train_loss": [],
            "val_loss":   [],
            "lr":         [],
        }

        collator = BatchCollator(
            stream=train_stream,
            batch_size=self.batch_size,
            device=self.device,
        )

        self.model.train()
        data_iter: Iterator = iter(collator)

        # Handle step-0 checkpoint (random initialization baseline)
        if self.step == 0:
            print(f"  Saving step-0 checkpoint (random initialization)...")
            val_loss_0 = self._get_val_loss(val_stream)
            self._maybe_save_checkpoint(train_loss=float("inf"), val_loss=val_loss_0)
            print(f"  Step 0 | val_loss={val_loss_0:.4f} | checkpoint saved\n")

        t_start = time.time()
        t_log   = time.time()
        loss_accumulator: List[float] = []

        print(f"{'─' * 64}")
        print(
            f"  {'Step':>8}  {'Train Loss':>12}  {'Val Loss':>10}  "
            f"{'LR':>10}  {'Tokens':>12}  {'Elapsed':>8}"
        )
        print(f"{'─' * 64}")

        while self.step < self.total_steps:
            self.step += 1

            # Fetch batch
            try:
                input_ids, targets = next(data_iter)
            except StopIteration:
                data_iter = iter(collator)
                input_ids, targets = next(data_iter)

            # Update learning rate
            lr = self.scheduler.set_lr(self.optimizer, self.step)

            # Forward pass
            logits, _ = self.model(input_ids, return_attention=False)

            # Loss computation — reshape for cross_entropy
            B, T, V = logits.shape
            loss = nn.functional.cross_entropy(
                logits.view(B * T, V),
                targets.reshape(B * T),
            )

            # Backward pass
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # Gradient clipping
            grad_norm = nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip
            )

            self.optimizer.step()

            # Tracking
            train_loss = loss.item()
            loss_accumulator.append(train_loss)
            self.tokens_seen += B * T

            # Checkpoint logic
            if should_checkpoint(self.step):
                # Compute smoothed train loss (mean of recent steps)
                smooth_train_loss = float(np.mean(loss_accumulator[-50:]))

                # Estimate val loss only every val_every checkpoints
                if self.ckpt_counter % self.val_every == 0:
                    val_loss = self._get_val_loss(val_stream)
                    self.last_val_loss = val_loss
                else:
                    val_loss = self.last_val_loss

                # Save checkpoint
                ckpt_path = self._maybe_save_checkpoint(smooth_train_loss, val_loss)

                # Log checkpoint line
                elapsed = time.time() - t_start
                print(
                    f"  {self.step:>8,}  {smooth_train_loss:>12.4f}  "
                    f"{val_loss:>10.4f}  {lr:>10.2e}  "
                    f"{self.tokens_seen / 1e6:>10.1f}M  "
                    f"{elapsed:>7.1f}s"
                    + (" ✓" if ckpt_path else "")
                )

                # Update history
                history["steps"].append(self.step)
                history["train_loss"].append(smooth_train_loss)
                history["val_loss"].append(val_loss)
                history["lr"].append(lr)

                self.last_train_loss = smooth_train_loss

            # Periodic progress (non-checkpoint steps, every 1000 steps)
            elif self.step % 1_000 == 0:
                smooth = float(np.mean(loss_accumulator[-100:]))
                elapsed = time.time() - t_start
                print(
                    f"  {self.step:>8,}  {smooth:>12.4f}  "
                    f"{'—':>10}  {lr:>10.2e}  "
                    f"{self.tokens_seen / 1e6:>10.1f}M  "
                    f"{elapsed:>7.1f}s"
                )

        # Final summary
        elapsed_total = time.time() - t_start
        print(f"{'─' * 64}")
        print(f"\n{'=' * 64}")
        print(f"  Training complete")
        print(f"  {'─' * 50}")
        print(f"  {'Total steps':<24}: {self.step:,}")
        print(f"  {'Checkpoints saved':<24}: {self.ckpt_counter}")
        print(f"  {'Tokens processed':<24}: {self.tokens_seen / 1e6:.1f}M")
        print(f"  {'Final train loss':<24}: {self.last_train_loss:.4f}")
        print(f"  {'Final val loss':<24}: {self.last_val_loss:.4f}")
        print(f"  {'Total time':<24}: {elapsed_total / 3600:.2f}h")
        print(f"  {'Checkpoint dir':<24}: {self.ckpt_dir}")
        print(f"{'=' * 64}\n")

        return history