from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch

from irisen_model.tokenization import ByteTokenizer


@dataclass(frozen=True)
class TextDataset:
    train_tokens: torch.Tensor
    val_tokens: torch.Tensor


def load_tokens(path: Path, tokenizer: ByteTokenizer) -> torch.Tensor:
    text = path.read_text(encoding="utf-8")
    ids = tokenizer.encode(text)
    if len(ids) < 2:
        raise ValueError(f"{path} must contain at least two UTF-8 bytes")
    return torch.tensor(ids, dtype=torch.long)


def split_tokens(tokens: torch.Tensor, train_fraction: float) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    split_at = max(1, min(len(tokens) - 1, int(len(tokens) * train_fraction)))
    return tokens[:split_at].contiguous(), tokens[split_at:].contiguous()


def load_text_dataset(
    path: Path,
    tokenizer: ByteTokenizer,
    train_fraction: float,
) -> TextDataset:
    tokens = load_tokens(path, tokenizer)
    train_tokens, val_tokens = split_tokens(tokens, train_fraction)
    return TextDataset(train_tokens=train_tokens, val_tokens=val_tokens)


def load_text_dataset_pair(
    train_path: Path,
    val_path: Path,
    tokenizer: ByteTokenizer,
) -> TextDataset:
    return TextDataset(
        train_tokens=load_tokens(train_path, tokenizer),
        val_tokens=load_tokens(val_path, tokenizer),
    )


def ensure_min_length(tokens: torch.Tensor, min_length: int) -> torch.Tensor:
    if len(tokens) >= min_length:
        return tokens
    repeats = math.ceil(min_length / len(tokens))
    return tokens.repeat(repeats)[:min_length].contiguous()


def get_batch(
    tokens: torch.Tensor,
    batch_size: int,
    context_size: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = ensure_min_length(tokens, context_size + 2)
    max_start = len(tokens) - context_size - 1
    starts = torch.randint(0, max_start + 1, (batch_size,))
    x = torch.stack([tokens[start : start + context_size] for start in starts])
    y = torch.stack([tokens[start + 1 : start + context_size + 1] for start in starts])
    return x.to(device), y.to(device)
