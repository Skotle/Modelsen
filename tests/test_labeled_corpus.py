import tempfile
import unittest
from pathlib import Path

from irisen_model.data import LabeledCorpusBuildConfig, build_labeled_corpus


class LabeledCorpusTest(unittest.TestCase):
    def test_build_labeled_corpus_reaches_token_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = build_labeled_corpus(
                LabeledCorpusBuildConfig(
                    output_dir=Path(tmp),
                    target_tokens=10_000,
                    validation_fraction=0.2,
                    seed=777,
                )
            )
            self.assertGreaterEqual(manifest["totals"]["char_tokens"], 10_000)
            self.assertGreater(manifest["totals"]["train_examples"], 0)
            self.assertGreater(manifest["totals"]["validation_examples"], 0)
            for item in manifest["artifacts"].values():
                path = Path(item["path"])
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 0)
                self.assertEqual(len(item["sha256"]), 64)

    def test_response_format_ends_answers_before_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = build_labeled_corpus(
                LabeledCorpusBuildConfig(
                    output_dir=Path(tmp),
                    target_tokens=5_000,
                    validation_fraction=0.2,
                    seed=778,
                    format="response",
                )
            )
            self.assertGreaterEqual(manifest["totals"]["char_tokens"], 5_000)
            train_text = Path(manifest["artifacts"]["train_text"]["path"]).read_text(encoding="utf-8")
            self.assertIn("<|response|>", train_text)
            self.assertIn("<|end|>", train_text)
            self.assertNotIn("검증키워드:", train_text)


if __name__ == "__main__":
    unittest.main()
