from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class IrisenConfig:
    """Architecture configuration for the Irisen causal language model."""

    model_type: str = "irisen"
    vocab_size: int = 256
    context_size: int = 128
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int | None = None
    dropout: float = 0.1
    bias: bool = True
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.context_size < 2:
            raise ValueError("context_size must be at least 2")
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "IrisenConfig":
        return cls(**values)


ModelConfig = IrisenConfig
