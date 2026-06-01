import unittest

from irisen_model import ByteTokenizer


class ByteTokenizerTest(unittest.TestCase):
    def test_utf8_round_trip(self) -> None:
        tokenizer = ByteTokenizer()
        text = "안녕, tiny model! 123"
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_vocab_size_is_fixed(self) -> None:
        tokenizer = ByteTokenizer()
        self.assertEqual(tokenizer.vocab_size, 256)


if __name__ == "__main__":
    unittest.main()

