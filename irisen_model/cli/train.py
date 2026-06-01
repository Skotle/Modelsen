from __future__ import annotations

import argparse
from pathlib import Path

from irisen_model.configs import load_experiment_config
from irisen_model.data import load_text_dataset, load_text_dataset_pair
from irisen_model.modeling import IrisenConfig, IrisenForCausalLM
from irisen_model.tokenization import ByteTokenizer
from irisen_model.training import TrainConfig, Trainer
from irisen_model.utils import pick_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an Irisen causal language model.")
    parser.add_argument("--config", type=Path, default=Path("configs/irisen-tiny.json"))
    parser.add_argument("--data", type=Path, default=Path("data/tiny_korean.txt"))
    parser.add_argument("--val-data", type=Path)
    parser.add_argument("--out", type=Path, default=Path("runs/irisen_tiny.pt"))

    parser.add_argument("--context-size", type=int)
    parser.add_argument("--d-model", type=int)
    parser.add_argument("--layers", type=int)
    parser.add_argument("--heads", type=int)
    parser.add_argument("--dropout", type=float)

    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--grad-clip", type=float)
    parser.add_argument("--eval-interval", type=int)
    parser.add_argument("--eval-batches", type=int)
    parser.add_argument("--train-fraction", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    return parser.parse_args()


def apply_model_overrides(config: IrisenConfig, args: argparse.Namespace) -> IrisenConfig:
    values = config.to_dict()
    mapping = {
        "context_size": args.context_size,
        "d_model": args.d_model,
        "n_layers": args.layers,
        "n_heads": args.heads,
        "dropout": args.dropout,
    }
    for key, value in mapping.items():
        if value is not None:
            values[key] = value
    return IrisenConfig.from_dict(values)


def apply_train_overrides(config: TrainConfig, args: argparse.Namespace) -> TrainConfig:
    values = config.to_dict()
    mapping = {
        "batch_size": args.batch_size,
        "steps": args.steps,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "eval_interval": args.eval_interval,
        "eval_batches": args.eval_batches,
        "train_fraction": args.train_fraction,
        "seed": args.seed,
        "device": args.device,
    }
    for key, value in mapping.items():
        if value is not None:
            values[key] = value
    return TrainConfig.from_dict(values)


def main() -> None:
    args = parse_args()
    experiment_config = load_experiment_config(args.config)
    model_config = apply_model_overrides(experiment_config.model, args)
    train_config = apply_train_overrides(experiment_config.training, args)

    set_seed(train_config.seed)
    device = pick_device(train_config.device)
    tokenizer = ByteTokenizer()
    if args.val_data is None:
        dataset = load_text_dataset(args.data, tokenizer, train_config.train_fraction)
    else:
        dataset = load_text_dataset_pair(args.data, args.val_data, tokenizer)
    model = IrisenForCausalLM(model_config).to(device)

    print(
        f"device={device} train_tokens={len(dataset.train_tokens)} "
        f"val_tokens={len(dataset.val_tokens)} parameters={sum(p.numel() for p in model.parameters())}"
    )
    trainer = Trainer(model, tokenizer, dataset, train_config, device, args.out)

    def log(metrics) -> None:
        suffix = f" saved={args.out}" if metrics.saved_checkpoint else ""
        print(
            f"step={metrics.step} train_loss={metrics.train_loss:.4f} "
            f"val_loss={metrics.val_loss:.4f}{suffix}"
        )

    trainer.fit(on_metrics=log)


if __name__ == "__main__":
    main()
