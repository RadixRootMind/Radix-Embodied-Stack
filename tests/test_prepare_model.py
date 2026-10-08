"""Exercise checkpoint preparation against temporary JSON files only."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "prepare_model.py"


class PrepareModelTests(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory()
        self.addCleanup(self.workspace.cleanup)
        self.model = Path(self.workspace.name) / "checkpoint with spaces"
        self.model.mkdir()
        self.config = self.model / "config.json"
        self.preprocessor = self.model / "policy_preprocessor.json"
        self.original_config = {
            "type": "pi05",
            "device": "cuda",
            "compile_model": True,
            "output_features": {"action": {"shape": [6]}},
            "metadata": {"label": "测试 checkpoint"},
        }
        self.original_preprocessor = {
            "steps": [
                {"config": {"device": "cuda"}},
                {"config": {"tokenizer_name": "original/tokenizer"}},
            ],
            "nested": {"items": [{"device": "cuda", "keep": 17}]},
            "metadata": "retain this value",
        }

    def write_checkpoint(self):
        self.config.write_text(json.dumps(self.original_config), encoding="utf-8")
        self.preprocessor.write_text(json.dumps(self.original_preprocessor), encoding="utf-8")

    def run_prepare(self, mode, tokenizer=None):
        command = [sys.executable, str(SCRIPT), str(self.model), "--mode", mode]
        if tokenizer is not None:
            command.extend(["--tokenizer", tokenizer])
        return subprocess.run(command, capture_output=True, text=True, timeout=10)

    def test_malformed_preprocessor_preserves_both_files(self):
        for mode in ("export", "runtime"):
            with self.subTest(mode=mode):
                self.write_checkpoint()
                self.preprocessor.write_bytes(b"{ invalid preprocessor JSON\n")
                before = (self.config.read_bytes(), self.preprocessor.read_bytes())
                result = self.run_prepare(mode)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("JSONDecodeError", result.stderr)
                self.assertEqual((self.config.read_bytes(), self.preprocessor.read_bytes()), before)

    def test_malformed_config_preserves_both_files(self):
        for mode in ("export", "runtime"):
            with self.subTest(mode=mode):
                self.write_checkpoint()
                self.config.write_bytes(b"{ invalid config JSON\n")
                before = (self.config.read_bytes(), self.preprocessor.read_bytes())
                result = self.run_prepare(mode)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("JSONDecodeError", result.stderr)
                self.assertEqual((self.config.read_bytes(), self.preprocessor.read_bytes()), before)

    def test_valid_checkpoint_updates_only_requested_values(self):
        for mode, tokenizer in (("export", None), ("runtime", "./models/自定义 tokenizer")):
            with self.subTest(mode=mode):
                self.write_checkpoint()
                result = self.run_prepare(mode, tokenizer)
                self.assertEqual(result.returncode, 0, result.stderr)
                config = json.loads(self.config.read_text(encoding="utf-8"))
                expected = dict(self.original_config)
                expected.update(
                    type="pi05" if mode == "export" else "hm_pi05",
                    device="cpu",
                    compile_model=False,
                )
                self.assertEqual(config, expected)
                preprocessor = json.loads(self.preprocessor.read_text(encoding="utf-8"))
                self.assertEqual(preprocessor["steps"][0]["config"]["device"], "cpu")
                self.assertEqual(
                    preprocessor["steps"][1]["config"]["tokenizer_name"],
                    tokenizer if tokenizer is not None else "./models/paligemma-3b-pt-224",
                )
                self.assertEqual(preprocessor["nested"], {"items": [{"device": "cpu", "keep": 17}]})
                self.assertEqual(preprocessor["metadata"], self.original_preprocessor["metadata"])
                self.assertTrue(self.config.read_bytes().endswith(b"\n"))
                self.assertTrue(self.preprocessor.read_bytes().endswith(b"\n"))

    def test_missing_input_preserves_remaining_file(self):
        for missing in (self.config, self.preprocessor):
            with self.subTest(missing=missing.name):
                self.write_checkpoint()
                remaining = self.preprocessor if missing == self.config else self.config
                before = remaining.read_bytes()
                missing.unlink()
                result = self.run_prepare("runtime")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Checkpoint must contain", result.stderr)
                self.assertEqual(remaining.read_bytes(), before)
                self.assertFalse(missing.exists())

    def test_repeated_mode_switches_are_idempotent(self):
        self.write_checkpoint()
        for mode in ("export", "runtime", "export"):
            with self.subTest(mode=mode):
                result = self.run_prepare(mode)
                self.assertEqual(result.returncode, 0, result.stderr)
                before = (self.config.read_bytes(), self.preprocessor.read_bytes())
                result = self.run_prepare(mode)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual((self.config.read_bytes(), self.preprocessor.read_bytes()), before)


if __name__ == "__main__":
    unittest.main()
