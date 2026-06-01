from __future__ import annotations

import torch
from torch.nn import functional as F


def filter_top_k(logits: torch.Tensor, top_k: int | None) -> torch.Tensor:
    if top_k is None:
        return logits
    top_k = min(top_k, logits.size(-1))
    values, _ = torch.topk(logits, top_k)
    return logits.masked_fill(logits < values[:, [-1]], float("-inf"))


@torch.no_grad()
def generate_tokens(
    model,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be greater than 0")
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.context_size :]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        logits = filter_top_k(logits, top_k)
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_id), dim=1)
    return idx

