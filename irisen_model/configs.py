from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .modeling.configuration_irisen import IrisenConfig
from .training.config import TrainConfig


@dataclass
class ExperimentConfig:
    model: IrisenConfig
    training: TrainConfig

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ExperimentConfig":
        return cls(
            model=IrisenConfig.from_dict(values.get("model", {})),
            training=TrainConfig.from_dict(values.get("training", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.to_dict(),
            "training": self.training.to_dict(),
        }


def load_experiment_config(path: Path | None) -> ExperimentConfig:
    if path is None:
        return ExperimentConfig(model=IrisenConfig(), training=TrainConfig())
    raw = json.loads(path.read_text(encoding="utf-8"))
    return ExperimentConfig.from_dict(raw)


def save_experiment_config(config: ExperimentConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(config.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

