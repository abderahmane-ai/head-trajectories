"""
training/scheduler.py — Cosine learning rate schedule with linear warmup.

Implements the standard warmup + cosine decay schedule used in LLaMA-style
training. The scheduler is stateless — it computes the learning rate for any
given step directly, making it safe to use with checkpoint resumption without
needing to save scheduler state separately.
"""

import math
from typing import Optional
import numpy as np


class CosineScheduler:
    """
    Learning rate scheduler: linear warmup followed by cosine decay to min_lr.

    lr(step) =
        step / warmup_steps * max_lr                    if step < warmup_steps
        min_lr + 0.5 * (max_lr - min_lr) *              if step < total_steps
            (1 + cos(π * decay_progress))
        min_lr                                           if step >= total_steps

    where decay_progress = (step - warmup_steps) / (total_steps - warmup_steps)

    Args:
        max_lr:       peak learning rate after warmup
        min_lr:       floor learning rate at end of cosine decay
        warmup_steps: number of linear warmup steps
        total_steps:  total training steps (warmup + cosine decay)
    """

    def __init__(
        self,
        max_lr:       float = 3e-4,
        min_lr:       float = 3e-5,
        warmup_steps: int   = 200,
        total_steps:  int   = 100_000,
    ) -> None:
        assert warmup_steps < total_steps, (
            f"warmup_steps ({warmup_steps}) must be < total_steps ({total_steps})"
        )
        assert min_lr <= max_lr, (
            f"min_lr ({min_lr}) must be <= max_lr ({max_lr})"
        )

        self.max_lr:       float = max_lr
        self.min_lr:       float = min_lr
        self.warmup_steps: int   = warmup_steps
        self.total_steps:  int   = total_steps

    def get_lr(self, step: int) -> float:
        """
        Compute the learning rate for the given step.

        Args:
            step: current training step (0-indexed)

        Returns:
            learning rate (float)
        """

        # Linear warmup phase
        if step < self.warmup_steps:
            return self.max_lr * (step + 1) / self.warmup_steps

        # Past total steps — stay at min_lr
        if step >= self.total_steps:
            return self.min_lr

        # Cosine decay phase
        decay_steps    = self.total_steps - self.warmup_steps
        elapsed_decay  = step - self.warmup_steps
        decay_progress = elapsed_decay / decay_steps               # 0.0 → 1.0
        cosine_factor  = 0.5 * (1.0 + math.cos(math.pi * decay_progress))

        return self.min_lr + cosine_factor * (self.max_lr - self.min_lr)

    def set_lr(self, optimizer: "torch.optim.Optimizer", step: int) -> float:
        """
        Apply the correct learning rate for the given step to all param groups.

        Args:
            optimizer: AdamW or any optimizer with param_groups
            step:      current training step

        Returns:
            current learning rate (for logging)
        """

        lr = self.get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def get_schedule_array(self) -> np.ndarray:
        """
        Return the full learning rate schedule as a numpy array.
        Useful for plotting and debugging.

        Returns:
            lrs: (total_steps,) float64
        """

        return np.array([self.get_lr(s) for s in range(self.total_steps)])

    def __repr__(self) -> str:
        return (
            f"CosineScheduler("
            f"max_lr={self.max_lr:.2e}, "
            f"min_lr={self.min_lr:.2e}, "
            f"warmup={self.warmup_steps}, "
            f"total={self.total_steps:,})"
        )