from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from irisen_model.data import get_batch, load_tokens
from irisen_model.utils import load_model_from_checkpoint, pick_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate checkpoint loss on a validation text file.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)
    model, tokenizer, metadata = load_model_from_checkpoint(args.checkpoint, device)
    tokens = load_tokens(args.data, tokenizer)

    losses: list[float] = []
    model.eval()
    for _ in range(args.batches):
        x, y = get_batch(tokens, args.batch_size, model.config.context_size, device)
        _, loss = model(x, y)
        assert loss is not None
        losses.append(loss.item())

    mean_loss = sum(losses) / len(losses)
    print(
        f"checkpoint_step={metadata.get('step')} "
        f"tokens={len(tokens)} batches={args.batches} "
        f"loss={mean_loss:.4f} ppl={math.exp(min(mean_loss, 20)):.2f}"
    )


if __name__ == "__main__":
    main()

