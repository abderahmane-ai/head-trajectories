"""
probing/extractor.py — Extract raw attention maps from all heads at a checkpoint.

Loads a saved checkpoint, runs the fixed probe dataset through the model,
and returns raw attention weight tensors (post-softmax, pre-dropout) for
every head in every layer.

The extractor is the bridge between the saved checkpoints and the scoring
pipeline. It is called once per checkpoint during the measurement phase.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from model import TransformerLM, ModelConfig


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model_from_checkpoint(
    ckpt_path: Path,
    device:    torch.device,
) -> Tuple[TransformerLM, int, float, float]:
    """
    Load a TransformerLM from a saved checkpoint file.

    Args:
        ckpt_path: path to checkpoint .pt file
        device:    device to load model onto

    Returns:
        (model, step, train_loss, val_loss)
    """

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Register ModelConfig as safe for torch.load
    torch.serialization.add_safe_globals([ModelConfig])
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)

    config:     ModelConfig = checkpoint["config"]
    step:       int         = checkpoint["step"]
    train_loss: float       = checkpoint.get("train_loss", float("inf"))
    val_loss:   float       = checkpoint.get("val_loss",   float("inf"))

    model = TransformerLM(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model, step, train_loss, val_loss


# ─────────────────────────────────────────────────────────────────────────────
# Attention extraction
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_attention_maps(
    model:      TransformerLM,
    token_ids:  torch.Tensor,
    device:     torch.device,
    batch_size: int = 16,
) -> List[torch.Tensor]:
    """
    Run token_ids through the model and extract raw attention maps
    from all heads in all layers.

    Processes sequences in mini-batches to avoid OOM on large probe sets.

    Args:
        model:      TransformerLM in eval mode
        token_ids:  (N, T) int64 — N sequences of length T
        device:     device for inference
        batch_size: sequences per inference batch

    Returns:
        attn_maps: List of length n_layers, each tensor shape (N, n_heads, T, T)
                   Values are post-softmax attention weights, detached, on CPU.
    """

    model.eval()
    N, T        = token_ids.shape
    n_layers    = model.config.n_layers
    n_heads     = model.config.n_heads

    # Pre-allocate output tensors on CPU
    all_maps: List[List[torch.Tensor]] = [[] for _ in range(n_layers)]

    for start in range(0, N, batch_size):
        end   = min(start + batch_size, N)
        batch = token_ids[start:end].to(device)   # (B, T)

        _, attn_maps = model(batch, return_attention=True)
        # attn_maps: List[(B, n_heads, T, T)] — already on CPU (detached in model)

        if attn_maps is None:
            raise RuntimeError(
                "Model returned None for attention maps. "
                "Ensure return_attention=True is propagating correctly."
            )

        for layer_idx, layer_map in enumerate(attn_maps):
            # layer_map: (B, n_heads, T, T) — CPU tensor
            all_maps[layer_idx].append(layer_map)

    # Concatenate batches along sequence dimension
    # Result: List of (N, n_heads, T, T), one per layer
    result: List[torch.Tensor] = [
        torch.cat(layer_chunks, dim=0)   # (N, n_heads, T, T)
        for layer_chunks in all_maps
    ]

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Full probe extraction at one checkpoint
# ─────────────────────────────────────────────────────────────────────────────

class CheckpointExtraction:
    """
    Container for all attention maps extracted at a single checkpoint.

    Attributes:
        step:               training step of this checkpoint
        train_loss:         training loss at this checkpoint
        val_loss:           validation loss at this checkpoint
        general_maps:       List[(500, n_heads, T, T)] per layer  — general seqs
        induction_maps:     List[(100, n_heads, T, T)] per layer  — induction seqs
        positional_maps:    List[(100, n_heads, T, T)] per layer  — positional seqs
        embedding_matrix:   (vocab_size, d_model) — model's current embeddings
        general_token_ids:  (500, T) — token ids for semantic score computation
        config:             ModelConfig of the checkpoint's model
    """

    def __init__(
        self,
        step:             int,
        train_loss:       float,
        val_loss:         float,
        general_maps:     List[torch.Tensor],
        induction_maps:   List[torch.Tensor],
        positional_maps:  List[torch.Tensor],
        embedding_matrix: torch.Tensor,
        general_token_ids: torch.Tensor,
        config:           ModelConfig,
    ) -> None:
        self.step             = step
        self.train_loss       = train_loss
        self.val_loss         = val_loss
        self.general_maps     = general_maps
        self.induction_maps   = induction_maps
        self.positional_maps  = positional_maps
        self.embedding_matrix = embedding_matrix
        self.general_token_ids = general_token_ids
        self.config           = config

    def __repr__(self) -> str:
        return (
            f"CheckpointExtraction(step={self.step:,}, "
            f"train_loss={self.train_loss:.4f}, "
            f"val_loss={self.val_loss:.4f}, "
            f"n_layers={len(self.general_maps)})"
        )


@torch.no_grad()
def extract_checkpoint(
    ckpt_path:  Path,
    probe_dict: Dict[str, torch.Tensor],
    device:     torch.device,
    batch_size: int = 16,
) -> CheckpointExtraction:
    """
    Load a checkpoint and extract all attention maps needed for scoring.

    Args:
        ckpt_path:  path to checkpoint .pt file
        probe_dict: loaded probe dataset dictionary
        device:     inference device
        batch_size: mini-batch size for inference

    Returns:
        CheckpointExtraction with all attention maps populated
    """

    # Load model
    model, step, train_loss, val_loss = load_model_from_checkpoint(
        ckpt_path, device
    )

    # Extract attention maps for each probe type
    general_seqs    = probe_dict["general_seqs"]    # (500, 256)
    induction_seqs  = probe_dict["induction_seqs"]  # (100, 256)
    positional_seqs = probe_dict["positional_seqs"] # (100, 256)

    general_maps    = extract_attention_maps(model, general_seqs,    device, batch_size)
    induction_maps  = extract_attention_maps(model, induction_seqs,  device, batch_size)
    positional_maps = extract_attention_maps(model, positional_seqs, device, batch_size)

    # Get current embedding matrix — critical: must be from THIS checkpoint
    embedding_matrix = model.get_embedding_matrix()   # (vocab_size, d_model)

    return CheckpointExtraction(
        step=step,
        train_loss=train_loss,
        val_loss=val_loss,
        general_maps=general_maps,
        induction_maps=induction_maps,
        positional_maps=positional_maps,
        embedding_matrix=embedding_matrix,
        general_token_ids=general_seqs,
        config=model.config,
    )
