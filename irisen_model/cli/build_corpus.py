from __future__ import annotations

import argparse
import json
from pathlib import Path

from irisen_model.data.synthetic_corpus import CorpusBuildConfig, build_synthetic_corpus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic synthetic train/validation data.")
    parser.add_argument("--out-dir", type=Path, default=Path("data/build"))
    parser.add_argument("--train-examples", type=int, default=12_000)
    parser.add_argument("--val-examples", type=int, default=1_500)
    parser.add_argument("--eval-examples", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260530)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_synthetic_corpus(
        CorpusBuildConfig(
            output_dir=args.out_dir,
            train_examples=args.train_examples,
            val_examples=args.val_examples,
            eval_examples=args.eval_examples,
            seed=args.seed,
        )
    )
    print(json.dumps(manifest["artifacts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

