from __future__ import annotations

from collections.abc import Iterable


class ByteTokenizer:
    """UTF-8 byte tokenizer with a stable 256-token vocabulary."""

    vocab_size = 256
    name = "byte"

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, token_ids: Iterable[int]) -> str:
        values = bytes(int(token_id) % 256 for token_id in token_ids)
        return values.decode("utf-8", errors="replace")

    def to_dict(self) -> dict[str, int | str]:
        return {"type": self.name, "vocab_size": self.vocab_size}

    @classmethod
    def from_dict(cls, values: dict[str, object] | None = None) -> "ByteTokenizer":
        values = values or {}
        if values.get("type", cls.name) != cls.name:
            raise ValueError(f"Unsupported tokenizer type: {values.get('type')}")
        return cls()

