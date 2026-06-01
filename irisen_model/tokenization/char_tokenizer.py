from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


class CharTokenizer:
    """Unicode character tokenizer that keeps Korean text valid during generation."""

    name = "char"
    unk_token = "<unk>"

    def __init__(self, vocab: Iterable[str]) -> None:
        chars = sorted(set(vocab))
        if self.unk_token in chars:
            chars.remove(self.unk_token)
        self.itos = [self.unk_token, *chars]
        self.stoi = {char: idx for idx, char in enumerate(self.itos)}
        self.vocab_size = len(self.itos)

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(text)

    @classmethod
    def from_files(cls, paths: Iterable[Path]) -> "CharTokenizer":
        text = "".join(path.read_text(encoding="utf-8") for path in paths)
        return cls.from_text(text)

    def encode(self, text: str) -> list[int]:
        unk_id = self.stoi[self.unk_token]
        return [self.stoi.get(char, unk_id) for char in text]

    def decode(self, token_ids: Iterable[int]) -> str:
        chars: list[str] = []
        for token_id in token_ids:
            idx = int(token_id)
            if idx <= 0 or idx >= len(self.itos):
                continue
            chars.append(self.itos[idx])
        return "".join(chars)

    def to_dict(self) -> dict[str, object]:
        return {"type": self.name, "vocab": self.itos}

    @classmethod
    def from_dict(cls, values: dict[str, object] | None = None) -> "CharTokenizer":
        values = values or {}
        if values.get("type", cls.name) != cls.name:
            raise ValueError(f"Unsupported tokenizer type: {values.get('type')}")
        vocab = values.get("vocab")
        if not isinstance(vocab, list) or not all(isinstance(item, str) for item in vocab):
            raise ValueError("CharTokenizer checkpoint is missing a valid vocab")
        tokenizer = cls([])
        tokenizer.itos = vocab
        tokenizer.stoi = {char: idx for idx, char in enumerate(vocab)}
        tokenizer.vocab_size = len(vocab)
        return tokenizer

