import unittest

from irisen_model.tokenization import CharTokenizer, tokenizer_from_dict


class CharTokenizerTest(unittest.TestCase):
    def test_korean_round_trip_without_utf8_breakage(self) -> None:
        tokenizer = CharTokenizer.from_text("언어 모델은 간결하게 설명한다.")
        text = "언어 모델은"
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_checkpoint_metadata_round_trip(self) -> None:
        tokenizer = CharTokenizer.from_text("abc 가나다")
        restored = tokenizer_from_dict(tokenizer.to_dict())
        self.assertEqual(restored.decode(restored.encode("가나다")), "가나다")

    def test_korean_alphabet_preserves_unseen_hangul_prompt_chars(self) -> None:
        tokenizer = CharTokenizer.from_text("언어 모델", alphabet="korean")
        text = "비유를 섞어 창의적으로"
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)


if __name__ == "__main__":
    unittest.main()
