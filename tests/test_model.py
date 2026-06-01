import unittest

import torch

from irisen_model import LanguageModel, ModelConfig


class LanguageModelTest(unittest.TestCase):
    def test_forward_loss_and_generate_shapes(self) -> None:
        torch.manual_seed(1)
        config = ModelConfig(context_size=8, d_model=32, n_layers=2, n_heads=4, dropout=0.0)
        model = LanguageModel(config)
        x = torch.randint(0, 256, (2, 8))
        y = torch.randint(0, 256, (2, 8))

        logits, loss = model(x, y)
        self.assertEqual(logits.shape, (2, 8, 256))
        self.assertIsNotNone(loss)
        self.assertEqual(loss.dim(), 0)

        generated = model.generate(x[:, :3], max_new_tokens=5, temperature=1.0, top_k=20)
        self.assertEqual(generated.shape, (2, 8))

    def test_context_limit_is_enforced(self) -> None:
        config = ModelConfig(context_size=4, d_model=16, n_layers=1, n_heads=4, dropout=0.0)
        model = LanguageModel(config)
        with self.assertRaises(ValueError):
            model(torch.randint(0, 256, (1, 5)))


if __name__ == "__main__":
    unittest.main()

