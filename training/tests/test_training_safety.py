import json
import tempfile
import unittest
from pathlib import Path

from jet_training.export import adapter_provenance, conversion_command
from jet_training.train import validate_resume


class TrainingSafetyTests(unittest.TestCase):
    def test_text_only_conversion_excludes_missing_mtp_layer(self):
        command = conversion_command(
            Path("merged"), Path("model.gguf"), "q8_0", Path("convert_hf_to_gguf.py")
        )
        self.assertIn("--no-nextn", command)

    def make_checkpoint(self, root, identity):
        output = root / "run"
        checkpoint = output / "checkpoint-2"
        checkpoint.mkdir(parents=True)
        for name in (
            "trainer_state.json",
            "adapter_config.json",
            "optimizer.pt",
            "scheduler.pt",
            "rng_state.pth",
        ):
            (checkpoint / name).write_text("{}", encoding="utf-8")
        (output / "run.json").write_text(json.dumps({"identity": identity}), encoding="utf-8")
        return output, checkpoint

    def test_existing_run_is_not_overwritten_without_resume(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaisesRegex(ValueError, "already exists"),
        ):
            validate_resume(Path(directory), None, {})

    def test_resume_checks_scheduler_horizon_and_data_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            identity = {"model": "pinned", "train_sha256": "old", "max_steps": 10}
            output, checkpoint = self.make_checkpoint(Path(directory), identity)
            validate_resume(output, checkpoint, identity)
            for changed in ({**identity, "max_steps": 20}, {**identity, "train_sha256": "new"}):
                with self.assertRaisesRegex(ValueError, "differ from original"):
                    validate_resume(output, checkpoint, changed)

    def test_resume_rejects_incomplete_or_external_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, checkpoint = self.make_checkpoint(root, {})
            (checkpoint / "optimizer.pt").unlink()
            with self.assertRaisesRegex(ValueError, "optimizer.pt"):
                validate_resume(output, checkpoint, {})
            with self.assertRaisesRegex(ValueError, "direct child"):
                validate_resume(output, root / "another-run" / "checkpoint-2", {})

    def test_export_rejects_adapter_for_another_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory)
            model = {"id": "Qwen/Qwen3.5-4B", "revision": "a" * 40}
            (adapter / "jet_training.json").write_text(
                json.dumps({"identity": {"model": model}}), encoding="utf-8"
            )
            (adapter / "adapter_config.json").write_text(
                json.dumps({"base_model_name_or_path": model["id"], "revision": "b" * 40}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "base revision"):
                adapter_provenance(adapter, {"model": model})


if __name__ == "__main__":
    unittest.main()
