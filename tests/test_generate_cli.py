import unittest

from irisen_model.cli.generate import parse_args


class GenerateCliTest(unittest.TestCase):
    def test_seed_is_optional_and_num_samples_defaults_to_one(self) -> None:
        args = parse_args([])
        self.assertIsNone(args.seed)
        self.assertEqual(args.num_samples, 1)

    def test_seed_can_be_set_for_reproducibility(self) -> None:
        args = parse_args(["--seed", "77", "--num-samples", "3"])
        self.assertEqual(args.seed, 77)
        self.assertEqual(args.num_samples, 3)


if __name__ == "__main__":
    unittest.main()
