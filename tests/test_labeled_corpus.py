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


if __name__ == "__main__":
    unittest.main()

