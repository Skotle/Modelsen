from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

from irisen_model.checkpoints import save_checkpoint
from irisen_model.data import TextDataset, get_batch
from irisen_model.modeling import IrisenForCausalLM
from irisen_model.tokenization import ByteTokenizer
from irisen_model.utils.optim import build_optimizer

from .config import TrainConfig


@dataclass(frozen=True)
class TrainMetrics:
    step: int
    train_loss: float
    val_loss: float
    saved_checkpoint: bool


class Trainer:
    def __init__(
        self,
        model: IrisenForCausalLM,
        tokenizer: ByteTokenizer,
        dataset: TextDataset,
        config: TrainConfig,
        device: torch.device,
        output_path: Path,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.config = config
        self.device = device
        self.output_path = output_path
        self.optimizer = build_optimizer(model, config)

    @torch.no_grad()
    def estimate_loss(self) -> dict[str, float]:
        self.model.eval()
        out: dict[str, float] = {}
        splits = {"train": self.dataset.train_tokens, "val": self.dataset.val_tokens}
        for split, tokens in splits.items():
            losses = []
            for _ in range(self.config.eval_batches):
                x, y = get_batch(tokens, self.config.batch_size, self.model.config.context_size, self.device)
                _, loss = self.model(x, y)
                assert loss is not None
                losses.append(loss.item())
            out[split] = sum(losses) / len(losses)
        self.model.train()
        return out

    def train(self) -> list[TrainMetrics]:
        return self.fit()

    def fit(self, on_metrics: Callable[[TrainMetrics], None] | None = None) -> list[TrainMetrics]:
        history: list[TrainMetrics] = []
        best_val_loss: float | None = None
        self.model.train()

        for step in range(1, self.config.steps + 1):
            x, y = get_batch(
                self.dataset.train_tokens,
                self.config.batch_size,
                self.model.config.context_size,
                self.device,
            )
            _, loss = self.model(x, y)
            assert loss is not None

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()

            should_eval = step == 1 or step % self.config.eval_interval == 0 or step == self.config.steps
            if should_eval:
                losses = self.estimate_loss()
                saved_checkpoint = best_val_loss is None or losses["val"] < best_val_loss
                metrics = TrainMetrics(
                    step=step,
                    train_loss=losses["train"],
                    val_loss=losses["val"],
                    saved_checkpoint=saved_checkpoint,
                )
                if best_val_loss is None or metrics.val_loss < best_val_loss:
                    best_val_loss = metrics.val_loss
                    save_checkpoint(
                        self.output_path,
                        self.model,
                        self.tokenizer,
                        self.config,
                        step,
                        best_val_loss,
                    )
                history.append(metrics)
                if on_metrics is not None:
                    on_metrics(metrics)
        return history

