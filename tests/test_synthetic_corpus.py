import tempfile
import unittest
from pathlib import Path

from irisen_model.data import CorpusBuildConfig, build_synthetic_corpus


class SyntheticCorpusTest(unittest.TestCase):
    def test_build_synthetic_corpus_outputs_manifest_and_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = build_synthetic_corpus(
                CorpusBuildConfig(
                    output_dir=Path(tmp),
                    train_examples=16,
                    val_examples=8,
                    eval_examples=5,
                    seed=123,
                )
            )

            artifacts = manifest["artifacts"]
            self.assertEqual(artifacts["train"]["examples"], 16)
            self.assertEqual(artifacts["validation"]["examples"], 8)
            self.assertEqual(artifacts["eval_prompts"]["examples"], 5)
            for item in artifacts.values():
                path = Path(item["path"])
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 0)
                self.assertEqual(len(item["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()

