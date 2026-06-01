from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import torch

from irisen_model.checkpoints import load_model_from_checkpoint
from irisen_model.generation import generate_tokens
from irisen_model.utils import pick_device, set_seed


class TextGenerationEngine:
    """Small serving-style wrapper around checkpoint loading and decoding."""

    def __init__(self, checkpoint: Path, device: str = "auto", seed: int | None = None) -> None:
        if seed is not None:
            set_seed(seed)
        self.device = pick_device(device)
        self.model, self.tokenizer, self.metadata = load_model_from_checkpoint(checkpoint, self.device)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int | None = 40,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int | None = None,
        stop: Iterable[str] | None = None,
        return_full_text: bool = True,
    ) -> str:
        prompt_ids = self.tokenizer.encode(prompt) or [10]
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        out = generate_tokens(
            self.model,
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        text = self.tokenizer.decode(out[0].tolist())
        if not return_full_text:
            decoded_prompt = self.tokenizer.decode(prompt_ids)
            if text.startswith(decoded_prompt):
                text = text[len(decoded_prompt) :]
        for stop_text in stop or []:
            stop_at = text.find(stop_text)
            if stop_at >= 0:
                text = text[:stop_at]
        return text
