from __future__ import annotations

import argparse
import json
from pathlib import Path

from irisen_model.data.labeled_corpus import LabeledCorpusBuildConfig, build_labeled_corpus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic labeled train/validation data.")
    parser.add_argument("--out-dir", type=Path, default=Path("data/labeled"))
    parser.add_argument("--target-tokens", type=int, default=1_200_000)
    parser.add_argument("--validation-fraction", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=20260601)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_labeled_corpus(
        LabeledCorpusBuildConfig(
            output_dir=args.out_dir,
            target_tokens=args.target_tokens,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )
    )
    print(json.dumps({"totals": manifest["totals"], "artifacts": manifest["artifacts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

