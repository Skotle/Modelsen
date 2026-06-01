from __future__ import annotations

import argparse
from pathlib import Path
from collections.abc import Sequence

from irisen_model.generation import GENERATION_PRESETS
from irisen_model.serving import TextGenerationEngine


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
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
    parser.add_argument("--stop", action="append", help="Stop printing at this text. Can be passed multiple times.")
    parser.add_argument("--completion-only", action="store_true", help="Print only newly generated text.")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of different samples to generate.")
    parser.add_argument("--seed", type=int, help="Set for reproducible output. Omit for varied output.")
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


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

    if args.num_samples < 1:
        raise ValueError("--num-samples must be at least 1")

    engine = TextGenerationEngine(args.checkpoint, device=args.device, seed=args.seed)
    for sample_idx in range(args.num_samples):
        text = engine.generate(
            args.prompt,
            max_new_tokens=args.tokens,
            temperature=generation_args["temperature"],
            top_k=generation_args["top_k"],
            top_p=generation_args["top_p"],
            repetition_penalty=generation_args["repetition_penalty"],
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            stop=args.stop,
            return_full_text=not args.completion_only,
        )
        if args.num_samples > 1:
            print(f"[sample {sample_idx + 1}/{args.num_samples}]")
        print(text)
        if args.num_samples > 1 and sample_idx != args.num_samples - 1:
            print()
    print(f"\n[checkpoint step={engine.metadata.get('step')} loss={engine.metadata.get('loss')}]")


if __name__ == "__main__":
    main()
