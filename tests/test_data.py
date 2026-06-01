import unittest

import torch

from irisen_model.data import get_batch, split_tokens


class DataTest(unittest.TestCase):
    def test_split_keeps_train_and_validation(self) -> None:
        train, val = split_tokens(torch.arange(10), 0.8)
        self.assertGreaterEqual(len(train), 1)
        self.assertGreaterEqual(len(val), 1)

    def test_get_batch_repeats_short_inputs(self) -> None:
        x, y = get_batch(torch.tensor([1, 2, 3]), batch_size=4, context_size=8, device="cpu")
        self.assertEqual(x.shape, (4, 8))
        self.assertEqual(y.shape, (4, 8))


if __name__ == "__main__":
    unittest.main()

