from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from irisen_model.modeling import IrisenConfig, IrisenForCausalLM
from irisen_model.tokenization import Tokenizer, tokenizer_from_dict
from irisen_model.training.config import TrainConfig


CHECKPOINT_SCHEMA_VERSION = 1


def save_checkpoint(
    path: Path,
    model: IrisenForCausalLM,
    tokenizer: Tokenizer,
    train_config: TrainConfig,
    step: int,
    loss: float | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "model_type": model.config.model_type,
            "model_config": model.config.to_dict(),
            "tokenizer": tokenizer.to_dict(),
            "train_config": train_config.to_dict(),
            "model_state": model.state_dict(),
            "step": step,
            "loss": loss,
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device | str) -> dict[str, Any]:
    return torch.load(path, map_location=device)


def load_model_from_checkpoint(
    path: Path,
    device: torch.device | str,
) -> tuple[IrisenForCausalLM, Tokenizer, dict[str, Any]]:
    checkpoint = load_checkpoint(path, device)
    config = IrisenConfig.from_dict(checkpoint["model_config"])
    model = IrisenForCausalLM(config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    tokenizer = tokenizer_from_dict(checkpoint.get("tokenizer"))
    return model, tokenizer, checkpoint
