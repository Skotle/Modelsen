from __future__ import annotations

import torch
from torch.nn import functional as F


GENERATION_PRESETS = {
    "precise": {"temperature": 0.25, "top_k": 6, "top_p": 0.85, "repetition_penalty": 1.15},
    "balanced": {"temperature": 0.55, "top_k": 18, "top_p": 0.92, "repetition_penalty": 1.08},
    "creative": {"temperature": 0.9, "top_k": 48, "top_p": 0.97, "repetition_penalty": 1.03},
}


def filter_top_k(logits: torch.Tensor, top_k: int | None) -> torch.Tensor:
    if top_k is None:
        return logits
    top_k = min(top_k, logits.size(-1))
    values, _ = torch.topk(logits, top_k)
    return logits.masked_fill(logits < values[:, [-1]], float("-inf"))


def filter_top_p(logits: torch.Tensor, top_p: float | None) -> torch.Tensor:
    if top_p is None or top_p >= 1.0:
        return logits
    if top_p <= 0.0:
        raise ValueError("top_p must be greater than 0")
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = False

    indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
    indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
    return logits.masked_fill(indices_to_remove, float("-inf"))


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated: torch.Tensor,
    repetition_penalty: float,
) -> torch.Tensor:
    if repetition_penalty == 1.0:
        return logits
    if repetition_penalty <= 0.0:
        raise ValueError("repetition_penalty must be greater than 0")

    logits = logits.clone()
    for batch_idx in range(logits.size(0)):
        token_ids = set(generated[batch_idx].tolist())
        for token_id in token_ids:
            if logits[batch_idx, token_id] < 0:
                logits[batch_idx, token_id] *= repetition_penalty
            else:
                logits[batch_idx, token_id] /= repetition_penalty
    return logits


def block_repeated_ngrams(
    logits: torch.Tensor,
    generated: torch.Tensor,
    no_repeat_ngram_size: int | None,
) -> torch.Tensor:
    if no_repeat_ngram_size is None or no_repeat_ngram_size <= 0:
        return logits
    if generated.size(1) + 1 < no_repeat_ngram_size:
        return logits

    logits = logits.clone()
    n = no_repeat_ngram_size
    for batch_idx in range(generated.size(0)):
        tokens = generated[batch_idx].tolist()
        prefix = tuple(tokens[-(n - 1) :]) if n > 1 else tuple()
        banned: set[int] = set()
        for start in range(0, len(tokens) - n + 1):
            ngram = tuple(tokens[start : start + n])
            if n == 1 or ngram[:-1] == prefix:
                banned.add(ngram[-1])
        if banned:
            logits[batch_idx, list(banned)] = float("-inf")
    return logits


@torch.no_grad()
def generate_tokens(
    model,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int | None = None,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be greater than 0")
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.context_size :]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        logits = apply_repetition_penalty(logits, idx, repetition_penalty)
        logits = block_repeated_ngrams(logits, idx, no_repeat_ngram_size)
        logits = logits / temperature
        logits = filter_top_k(logits, top_k)
        logits = filter_top_p(logits, top_p)
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_id), dim=1)
    return idx
