from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class TrainConfig:
    batch_size: int = 16
    steps: int = 500
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 100
    eval_batches: int = 10
    train_fraction: float = 0.9
    seed: int = 1337
    device: str = "auto"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TrainConfig":
        return cls(**values)

