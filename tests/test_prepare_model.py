"""Run the checkpoint patching CLI against temporary copies of config files."""

import json
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "so101_patch_model_config.py"
LLM_CONFIGS = (
    Path("config/pi0/llm/pi05_gemma_2b_xh2a_2k_so101_mask.py"),
    Path("config/pi0/llm/pi05_gemma_expert_300m_xh2a_2k_so101_mask.py"),
)


class PrepareModelTests(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory()
        self.addCleanup(self.workspace.cleanup)
        self.root = Path(self.workspace.name)
        self.script = self.root / SCRIPT.name
        shutil.copyfile(SCRIPT, self.script)
        self.model = self.root / "checkpoint with spaces"
        self.model.mkdir()
        self.config = self.model / "config.json"
        self.preprocessor = self.model / "policy_preprocessor.json"
        self.postprocessor = self.model / "policy_postprocessor.json"
        self.original_config = {
            "type": "pi05", "device": "cuda", "compile_model": True,
            "n_action_steps": 50, "pretrained_path": "original/base",
            "pretrained_revision": "old-revision", "use_relative_actions": False,
            "relative_exclude_joints": [], "action_feature_names": ["joint"],
            "output_features": {"action": {"shape": [6]}},
            "metadata": {"label": "测试 checkpoint"},
        }
        preprocessor = {
            "steps": [
                {"registry_name": "device_processor", "config": {"device": "cuda"}},
                {"registry_name": "tokenizer_processor", "config": {"tokenizer_name": "original/tokenizer"}},
                {"registry_name": "rename_observations_processor", "config": {"rename_map": {"keep": "keep"}}},
                {"registry_name": "other_processor", "config": {"device": "cuda", "keep": 17}},
            ],
            "metadata": "retain this value",
        }
        postprocessor = {
            "steps": [{"registry_name": "device_processor", "config": {"device": "cuda"}}],
            "metadata": "retain postprocessor metadata",
        }
        self.original_files = {
            path: json.dumps(data, ensure_ascii=False).encode("utf-8")
            for path, data in ((self.config, self.original_config),
                               (self.preprocessor, preprocessor), (self.postprocessor, postprocessor))
        }
        for relative in LLM_CONFIGS:
            self.original_files[self.root / relative] = (ROOT / relative).read_bytes()
        self.restore_files()

    def restore_files(self):
        for path, content in self.original_files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)

    def snapshot(self):
        return {path: path.read_bytes() if path.exists() else None for path in self.original_files}

    def run_prepare(self, mode, no_backup=True):
        command = [sys.executable, str(self.script), "--model-dir", str(self.model),
                   "--mode", mode, "--tokenizer-path", "custom/tokenizer with spaces",
                   "--base-model-path", "custom/base", "--n-action-steps", "30"]
        if no_backup:
            command.append("--no-backup")
        return subprocess.run(command, capture_output=True, text=True, timeout=10)

    def assert_invalid_json_preserves_files(self, payload, message):
        for path in (self.config, self.preprocessor, self.postprocessor):
            for mode in ("export", "infer"):
                with self.subTest(file=path.name, mode=mode):
                    self.restore_files()
                    path.write_bytes(payload)
                    before = self.snapshot()
                    result = self.run_prepare(mode, no_backup=False)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(message, result.stderr)
                    self.assertEqual(self.snapshot(), before)
                    self.assertEqual(list(self.root.rglob("*.bak_*")), [])

    def test_malformed_json_preserves_all_files(self):
        self.assert_invalid_json_preserves_files(b"{ invalid JSON\n", "JSONDecodeError")

    def test_non_object_json_preserves_all_files(self):
        self.assert_invalid_json_preserves_files(b"[]\n", "is not a JSON object")

    def test_valid_checkpoint_preserves_existing_patch_behavior(self):
        for mode in ("export", "infer"):
            with self.subTest(mode=mode):
                self.restore_files()
                result = self.run_prepare(mode)
                self.assertEqual(result.returncode, 0, result.stderr)
                expected = dict(self.original_config)
                for key in ("pretrained_revision", "use_relative_actions", "relative_exclude_joints", "action_feature_names"):
                    expected.pop(key)
                expected.update(type="pi05" if mode == "export" else "hm_pi05", device="cpu",
                                compile_model=False, n_action_steps=30, pretrained_path="custom/base")
                self.assertEqual(json.loads(self.config.read_text()), expected)
                preprocessor = json.loads(self.original_files[self.preprocessor])
                preprocessor["steps"][0]["config"]["device"] = "cpu"
                preprocessor["steps"][1]["config"]["tokenizer_name"] = "custom/tokenizer with spaces"
                preprocessor["steps"][2]["config"]["rename_map"]["observation.images.top"] = "observation.images.front"
                self.assertEqual(json.loads(self.preprocessor.read_text()), preprocessor)
                postprocessor = json.loads(self.original_files[self.postprocessor])
                postprocessor["steps"][0]["config"]["device"] = "cpu"
                self.assertEqual(json.loads(self.postprocessor.read_text()), postprocessor)
                for relative, sequence in zip(LLM_CONFIGS, (712, 50)):
                    config = runpy.run_path(str(self.root / relative))
                    self.assertEqual(config["hf_model_dir"], str(self.model))
                    self.assertEqual(config["config_dir"], "custom/tokenizer with spaces")
                    self.assertEqual(config["model"]["wrap_cfg"]["max_sequence_length"], 1024)
                    self.assertEqual(config["model"]["wrap_cfg"]["input_sequence_length"], sequence)
                for path in (self.config, self.preprocessor, self.postprocessor):
                    self.assertTrue(path.read_bytes().endswith(b"\n"))
                self.assertEqual(list(self.root.rglob("*.bak_*")), [])

    def test_backups_preserve_original_files(self):
        result = self.run_prepare("infer", no_backup=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        for path, original in self.original_files.items():
            backups = list(path.parent.glob(path.name + ".bak_*"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), original)

    def test_missing_input_preserves_remaining_files(self):
        for missing in self.original_files:
            with self.subTest(file=missing.name):
                self.restore_files()
                missing.unlink()
                before = self.snapshot()
                result = self.run_prepare("infer", no_backup=False)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Required file missing", result.stderr)
                self.assertEqual(self.snapshot(), before)
                self.assertEqual(list(self.root.rglob("*.bak_*")), [])

    def test_repeated_mode_switches_are_idempotent(self):
        for mode in ("export", "infer", "export"):
            with self.subTest(mode=mode):
                result = self.run_prepare(mode)
                self.assertEqual(result.returncode, 0, result.stderr)
                before = self.snapshot()
                result = self.run_prepare(mode)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(self.snapshot(), before)


if __name__ == "__main__":
    unittest.main()
