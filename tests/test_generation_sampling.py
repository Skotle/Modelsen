import unittest

import torch

from irisen_model import IrisenConfig, IrisenForCausalLM
from irisen_model.generation import filter_top_p, generate_tokens


class GenerationSamplingTest(unittest.TestCase):
    def test_top_p_keeps_at_least_one_candidate(self) -> None:
        logits = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
        filtered = filter_top_p(logits, 0.5)
        self.assertTrue(torch.isfinite(filtered[0, 0]))
        self.assertTrue(torch.isneginf(filtered[0, -1]))

    def test_generate_accepts_creativity_controls(self) -> None:
        torch.manual_seed(11)
        model = IrisenForCausalLM(
            IrisenConfig(context_size=8, vocab_size=32, d_model=16, n_layers=1, n_heads=4, dropout=0.0)
        )
        idx = torch.randint(0, 32, (1, 4))
        out = generate_tokens(
            model,
            idx,
            max_new_tokens=4,
            temperature=0.8,
            top_k=8,
            top_p=0.9,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
        )
        self.assertEqual(out.shape, (1, 8))


if __name__ == "__main__":
    unittest.main()

