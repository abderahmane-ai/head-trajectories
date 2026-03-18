"""
data/loader.py — OpenWebText streaming, tokenization, and batch generation.

Streams the OpenWebText dataset via HuggingFace datasets, tokenizes with the
GPT-2 tokenizer (tiktoken), and yields fixed-length token batches for training.
Supports a held-out validation split and a separate held-out probe split
(used exclusively by probe.py — never touched by the training loop).

Split allocation:
  - Training:  first 95% of shuffled documents
  - Validation: next 2.5% of shuffled documents
  - Probe:      final 2.5% of shuffled documents (immutable after construction)
"""

import random
import time
import numpy as np
import torch
from pathlib import Path
from typing import Generator, Iterator, List, Optional, Tuple

import tiktoken
from datasets import load_dataset


# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer
# ─────────────────────────────────────────────────────────────────────────────

def get_tokenizer() -> tiktoken.Encoding:
    """Return the GPT-2 BPE tokenizer. Vocab size: 50257."""

    return tiktoken.get_encoding("gpt2")


def tokenize_text(text: str, enc: tiktoken.Encoding) -> List[int]:
    """
    Tokenize a single document string.
    Appends the EOT token (50256) at the end of each document
    to mark document boundaries in the concatenated token stream.
    """

    return enc.encode_ordinary(text) + [enc.eot_token]


# ─────────────────────────────────────────────────────────────────────────────
# Token Stream
# ─────────────────────────────────────────────────────────────────────────────

class OpenWebTextStream:
    """
    Streams tokenized OpenWebText documents and yields fixed-length
    token chunks for training or validation.

    The dataset is shuffled with a fixed seed for reproducibility,
    then split into train / val / probe portions by document index.

    Args:
        split:      "train", "val", or "probe"
        block_size: number of tokens per chunk (default 256)
        seed:       shuffle seed (default 42)
        cache_dir:  optional path for HuggingFace dataset cache
    """

    TRAIN_FRAC: float = 0.950
    VAL_FRAC:   float = 0.025
    PROBE_FRAC: float = 0.025

    def __init__(
        self,
        split: str,
        block_size: int = 256,
        seed: int = 42,
        cache_dir: Optional[Path] = None,
    ) -> None:
        assert split in ("train", "val", "probe"), (
            f"split must be 'train', 'val', or 'probe', got '{split}'"
        )
        self.split:      str   = split
        self.block_size: int   = block_size
        self.seed:       int   = seed
        self.cache_dir:  Optional[Path] = cache_dir

        self.enc: tiktoken.Encoding = get_tokenizer()
        self._buffer: List[int] = []

        # Load and split dataset
        self._doc_indices: List[int] = self._compute_split_indices()

    def _compute_split_indices(self) -> List[int]:
        """
        Load dataset metadata, shuffle document indices deterministically,
        then slice according to split fractions.
        Returns the list of document indices for this split.
        """

        print(f"  Loading OpenWebText metadata (split={self.split})...")
        dataset = load_dataset(
            "openwebtext",
            split="train",
            cache_dir=str(self.cache_dir) if self.cache_dir else None,
            trust_remote_code=True,
        )
        n_docs = len(dataset)

        # Deterministic shuffle
        rng = random.Random(self.seed)
        indices = list(range(n_docs))
        rng.shuffle(indices)

        # Compute split boundaries
        n_train = int(n_docs * self.TRAIN_FRAC)
        n_val   = int(n_docs * self.VAL_FRAC)

        if self.split == "train":
            selected = indices[:n_train]
        elif self.split == "val":
            selected = indices[n_train : n_train + n_val]
        else:  # probe
            selected = indices[n_train + n_val :]

        print(
            f"  {'=' * 50}\n"
            f"  Split       : {self.split}\n"
            f"  Total docs  : {n_docs:,}\n"
            f"  Split docs  : {len(selected):,}\n"
            f"  Block size  : {self.block_size}\n"
            f"  {'=' * 50}"
        )

        # Store reference to dataset for document access
        self._dataset = dataset
        return selected

    def _doc_generator(self) -> Generator[List[int], None, None]:
        """Yield tokenized documents in split order."""

        for idx in self._doc_indices:
            text = self._dataset[idx]["text"]
            tokens = tokenize_text(text, self.enc)
            if len(tokens) > 0:
                yield tokens

    def __iter__(self) -> Iterator[torch.Tensor]:
        """
        Yield fixed-length token chunks as LongTensors of shape (block_size,).
        Documents are concatenated in order; partial chunks at document
        boundaries are carried over into the next chunk.
        """

        self._buffer = []
        for tokens in self._doc_generator():
            self._buffer.extend(tokens)
            while len(self._buffer) >= self.block_size:
                chunk = self._buffer[:self.block_size]
                self._buffer = self._buffer[self.block_size:]
                yield torch.tensor(chunk, dtype=torch.long)

    def get_raw_tokens(self, n_sequences: int) -> List[List[int]]:
        """
        Return exactly n_sequences raw token lists (each of length block_size).
        Used by probe.py to collect probe sequences from the probe split.
        """

        sequences: List[List[int]] = []
        for chunk in self:
            sequences.append(chunk.tolist())
            if len(sequences) >= n_sequences:
                break
        return sequences


# ─────────────────────────────────────────────────────────────────────────────
# Batch Collator
# ─────────────────────────────────────────────────────────────────────────────

class BatchCollator:
    """
    Wraps an OpenWebTextStream and yields (input_ids, targets) training batches.

    input_ids : (batch_size, block_size)   — tokens 0..T-1
    targets   : (batch_size, block_size)   — tokens 1..T (shifted by 1)

    Args:
        stream:     OpenWebTextStream instance
        batch_size: number of sequences per batch
        device:     torch device to place tensors on
    """

    def __init__(
        self,
        stream: OpenWebTextStream,
        batch_size: int = 32,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.stream:     OpenWebTextStream = stream
        self.batch_size: int               = batch_size
        self.device:     torch.device      = device

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Yield (input_ids, targets) batches indefinitely."""

        batch_buffer: List[torch.Tensor] = []

        while True:
            for chunk in self.stream:
                batch_buffer.append(chunk)
                if len(batch_buffer) == self.batch_size:
                    # Stack into batch: (B, T)
                    batch = torch.stack(batch_buffer, dim=0).to(self.device)
                    # Input is tokens[:-1], target is tokens[1:]
                    # For causal LM: predict next token at every position
                    input_ids = batch[:, :-1]
                    targets   = batch[:, 1:]
                    batch_buffer = []
                    yield input_ids, targets

            # If stream is exhausted, restart it (multiple epochs)
            # This should rarely trigger given ~300M token target
            batch_buffer = []


# ─────────────────────────────────────────────────────────────────────────────
# Validation Loss Helper
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def estimate_val_loss(
    model: torch.nn.Module,
    val_stream: OpenWebTextStream,
    batch_size: int,
    n_batches: int,
    device: torch.device,
) -> float:
    """
    Estimate validation loss over n_batches batches from val_stream.
    Uses a fresh iterator each call so evaluation is consistent.

    Returns:
        mean cross-entropy loss (float)
    """

    model.eval()
    collator = BatchCollator(val_stream, batch_size=batch_size, device=device)
    losses: List[float] = []

    for i, (input_ids, targets) in enumerate(collator):
        if i >= n_batches:
            break
        logits, _ = model(input_ids, return_attention=False)
        # logits: (B, T-1, vocab_size), targets: (B, T-1)
        B, T, V = logits.shape
        loss = torch.nn.functional.cross_entropy(
            logits.view(B * T, V),
            targets.reshape(B * T),
        )
        losses.append(loss.item())

    model.train()
    return float(np.mean(losses))


# ─────────────────────────────────────────────────────────────────────────────
# Quick diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def print_loader_stats(stream: OpenWebTextStream, n_sample: int = 5) -> None:
    """Print sample token chunks and basic stream statistics."""

    print(f"\n{'=' * 60}")
    print(f"  Loader diagnostics — split: {stream.split}")
    print(f"{'=' * 60}")
    enc = stream.enc
    t0 = time.time()
    for i, chunk in enumerate(stream):
        if i >= n_sample:
            break
        decoded = enc.decode(chunk.tolist()[:40])
        print(f"  Chunk {i:2d} | first 40 tokens: {repr(decoded)}")
    print(f"  Elapsed for {n_sample} chunks: {time.time() - t0:.2f}s")
    print(f"{'=' * 60}\n")