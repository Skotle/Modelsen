import tempfile
import unittest
from pathlib import Path

import torch

from irisen_model.checkpoints import load_model_from_checkpoint, save_checkpoint
from irisen_model.modeling import IrisenConfig, IrisenForCausalLM
from irisen_model.serving import TextGenerationEngine
from irisen_model.tokenization import ByteTokenizer
from irisen_model.training import TrainConfig


class CheckpointAndServingTest(unittest.TestCase):
    def test_checkpoint_round_trip_and_engine(self) -> None:
        torch.manual_seed(7)
        tokenizer = ByteTokenizer()
        config = IrisenConfig(context_size=8, d_model=16, n_layers=1, n_heads=4, dropout=0.0)
        model = IrisenForCausalLM(config)
        train_config = TrainConfig(steps=1, batch_size=2, eval_batches=1)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            save_checkpoint(path, model, tokenizer, train_config, step=1, loss=1.23)

            loaded_model, loaded_tokenizer, metadata = load_model_from_checkpoint(path, "cpu")
            self.assertEqual(metadata["schema_version"], 1)
            self.assertEqual(loaded_model.config.context_size, 8)
            self.assertEqual(loaded_tokenizer.decode(tokenizer.encode("검증")), "검증")

            engine = TextGenerationEngine(path, device="cpu", seed=7)
            out = engine.generate("A", max_new_tokens=2, temperature=1.0, top_k=10)
            self.assertGreaterEqual(len(out), 1)


if __name__ == "__main__":
    unittest.main()

