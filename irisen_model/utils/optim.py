from __future__ import annotations

import torch

from irisen_model.modeling import IrisenForCausalLM
from irisen_model.training.config import TrainConfig


def build_optimizer(model: IrisenForCausalLM, config: TrainConfig) -> torch.optim.Optimizer:
    decay_params: list[torch.nn.Parameter] = []
    nodecay_params: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2 and not name.endswith("token_embedding.weight"):
            decay_params.append(param)
        else:
            nodecay_params.append(param)

    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": config.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
        betas=(0.9, 0.95),
    )

