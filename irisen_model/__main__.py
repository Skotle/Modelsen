from __future__ import annotations

import textwrap


def main() -> None:
    print(
        textwrap.dedent(
            """
            irisen_model

            Build data:
              python scripts/build_corpus.py --out-dir data/build

            Train:
              python -m irisen_model.train --config configs/irisen-tiny.json --data data/tiny_korean.txt

            Generate:
              python -m irisen_model.generate --checkpoint runs/irisen_tiny.pt --prompt "언어 모델은"
            """
        ).strip()
    )


if __name__ == "__main__":
    main()
