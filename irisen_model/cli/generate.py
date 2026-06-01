from __future__ import annotations

import argparse
from pathlib import Path

from irisen_model.generation import GENERATION_PRESETS
from irisen_model.serving import TextGenerationEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from an Irisen checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/irisen_tiny.pt"))
    parser.add_argument("--prompt", default="언어 모델은")
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--preset", choices=sorted(GENERATION_PRESETS), default=None)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--repetition-penalty", type=float)
    parser.add_argument("--no-repeat-ngram-size", type=int)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generation_args = {"temperature": 0.8, "top_k": 40, "top_p": None, "repetition_penalty": 1.0}
    if args.preset is not None:
        generation_args.update(GENERATION_PRESETS[args.preset])
    if args.temperature is not None:
        generation_args["temperature"] = args.temperature
    if args.top_k is not None:
        generation_args["top_k"] = args.top_k
    if args.top_p is not None:
        generation_args["top_p"] = args.top_p
    if args.repetition_penalty is not None:
        generation_args["repetition_penalty"] = args.repetition_penalty

    engine = TextGenerationEngine(args.checkpoint, device=args.device, seed=args.seed)
    text = engine.generate(
        args.prompt,
        max_new_tokens=args.tokens,
        temperature=generation_args["temperature"],
        top_k=generation_args["top_k"],
        top_p=generation_args["top_p"],
        repetition_penalty=generation_args["repetition_penalty"],
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )
    print(text)
    print(f"\n[checkpoint step={engine.metadata.get('step')} loss={engine.metadata.get('loss')}]")


if __name__ == "__main__":
    main()
