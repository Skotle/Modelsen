from __future__ import annotations

import argparse
from pathlib import Path

from irisen_model.serving import TextGenerationEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from an Irisen checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/irisen_tiny.pt"))
    parser.add_argument("--prompt", default="언어 모델은")
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = TextGenerationEngine(args.checkpoint, device=args.device, seed=args.seed)
    text = engine.generate(
        args.prompt,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(text)
    print(f"\n[checkpoint step={engine.metadata.get('step')} loss={engine.metadata.get('loss')}]")


if __name__ == "__main__":
    main()

