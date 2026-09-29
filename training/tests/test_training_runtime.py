"""Opt-in, offline tests of real Qwen3.5/PEFT/Trainer kernels and checkpoints.

Set JET_RUN_TRAINING_RUNTIME=1. A CUDA GPU with BF16 is required for training;
checkpoint mapping itself runs on CPU. No pretrained model is downloaded.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

from jet_training.tokenization import AnswerOnlyCollator, TokenizedDataset
from jet_training.train import add_lora, load_text_model, validate_resume


@unittest.skipUnless(
    os.environ.get("JET_RUN_TRAINING_RUNTIME") == "1", "opt-in real Qwen3.5 runtime test"
)
class TrainingRuntimeTests(unittest.TestCase):
    def make_checkpoint(self, path):
        from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration, Qwen3_5TextConfig

        text = Qwen3_5TextConfig(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            linear_conv_kernel_dim=4,
            layer_types=["linear_attention", "full_attention"],
            max_position_embeddings=128,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0,
                "mrope_section": [2, 3, 3],
            },
            pad_token_id=0,
            eos_token_id=1,
        )
        config = Qwen3_5Config(
            text_config=text.to_dict(),
            vision_config={
                "depth": 1,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_heads": 2,
                "out_hidden_size": 32,
                "num_position_embeddings": 16,
                "patch_size": 2,
                "temporal_patch_size": 2,
                "spatial_merge_size": 2,
            },
        )
        model = Qwen3_5ForConditionalGeneration(config)
        model.save_pretrained(path)
        return model

    def config(self, path):
        return {
            "model": {"id": str(path), "revision": "a" * 40, "attn_implementation": "sdpa"},
            "lora": {
                "r": 4,
                "alpha": 8,
                "dropout": 0.0,
                "target_modules": [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                    "in_proj_qkv",
                    "in_proj_z",
                    "in_proj_b",
                    "in_proj_a",
                    "out_proj",
                ],
            },
        }

    def test_nested_multimodal_checkpoint_loads_exact_text_weights(self):
        import torch

        with tempfile.TemporaryDirectory() as directory:
            original = self.make_checkpoint(Path(directory))
            loaded = load_text_model(self.config(Path(directory)))
            self.assertFalse(any("visual" in name for name, _ in loaded.named_parameters()))
            for key, value in original.model.language_model.state_dict().items():
                torch.testing.assert_close(
                    loaded.model.state_dict()[key], value.to(torch.bfloat16), rtol=0, atol=0
                )
            torch.testing.assert_close(
                loaded.lm_head.weight, original.lm_head.weight.to(torch.bfloat16), rtol=0, atol=0
            )

    def test_bf16_lora_updates_resume_and_merge_preserve_predictions(self):
        import torch
        from peft import PeftModel
        from transformers import (
            Qwen3_5ForCausalLM,
            Trainer,
            TrainerCallback,
            TrainingArguments,
            set_seed,
        )

        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            self.skipTest("CUDA BF16 GPU required")
        set_seed(42)
        rows = [
            {
                "input_ids": [2, 3, 4, 5, 6, 7, 8],
                "attention_mask": [1] * 7,
                "labels": [-100] * 4 + [6, 7, 8],
            },
            {
                "input_ids": [2, 9, 10, 5, 11],
                "attention_mask": [1] * 5,
                "labels": [-100] * 3 + [5, 11],
            },
        ]
        dataset = TokenizedDataset(rows)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_path, output = root / "base", root / "run"
            self.make_checkpoint(base_path)
            config = self.config(base_path)
            model = add_lora(load_text_model(config), config)
            before = {
                name: value.detach().clone()
                for name, value in model.named_parameters()
                if "lora_B" in name
            }

            class StopAfterTwoSteps(TrainerCallback):
                def on_step_end(self, args, state, control, **kwargs):
                    if state.global_step == 2:
                        control.should_training_stop = True
                    return control

            def trainer_for(candidate, callbacks=None):
                return Trainer(
                    model=candidate,
                    args=TrainingArguments(
                        output_dir=str(output),
                        max_steps=3,
                        per_device_train_batch_size=2,
                        gradient_accumulation_steps=1,
                        learning_rate=0.01,
                        bf16=True,
                        gradient_checkpointing=True,
                        gradient_checkpointing_kwargs={"use_reentrant": False},
                        save_steps=1,
                        save_total_limit=2,
                        logging_steps=1,
                        report_to="none",
                        remove_unused_columns=False,
                        prediction_loss_only=True,
                        dataloader_pin_memory=False,
                    ),
                    train_dataset=dataset,
                    data_collator=AnswerOnlyCollator(0),
                    callbacks=callbacks,
                )

            trainer = trainer_for(model, callbacks=[StopAfterTwoSteps()])
            result = trainer.train()
            self.assertEqual(trainer.state.global_step, 2)
            self.assertTrue(torch.isfinite(torch.tensor(result.training_loss)))
            self.assertTrue(
                any(
                    not torch.equal(value.cpu(), before[name].cpu())
                    for name, value in model.named_parameters()
                    if name in before
                )
            )
            interrupted_weights = {
                name: value.detach().cpu().clone()
                for name, value in model.named_parameters()
                if "lora_B" in name
            }
            identity = {"synthetic_test": True, "max_steps": 3}
            (output / "run.json").write_text(json.dumps({"identity": identity}), encoding="utf-8")
            checkpoint = output / "checkpoint-2"
            validate_resume(output, checkpoint, identity)
            resumed = trainer_for(add_lora(load_text_model(config), config))
            resumed.train(resume_from_checkpoint=str(checkpoint))
            self.assertEqual(resumed.state.global_step, 3)
            self.assertEqual(resumed.lr_scheduler.last_epoch, 3)
            self.assertTrue(
                any(
                    entry["learning_rate"] > 0
                    for entry in resumed.state.log_history
                    if entry.get("step") == 3 and "learning_rate" in entry
                )
            )
            self.assertTrue(
                any(
                    not torch.equal(value.detach().cpu(), interrupted_weights[name])
                    for name, value in resumed.model.named_parameters()
                    if name in interrupted_weights
                )
            )
            adapter = root / "adapter"
            resumed.save_model(str(adapter))
            reloaded = PeftModel.from_pretrained(load_text_model(config), adapter).to("cuda").eval()
            inputs = AnswerOnlyCollator(0)(rows)
            inputs = {key: value.to("cuda") for key, value in inputs.items() if key != "labels"}
            with torch.no_grad():
                expected = reloaded(**inputs).logits.float()
                merged = reloaded.merge_and_unload(safe_merge=True).eval()
                actual = merged(**inputs).logits.float()
            # BF16 merging rounds the materialized weights once.
            torch.testing.assert_close(actual, expected, rtol=0.04, atol=0.01)
            merged_path = root / "merged"
            merged.save_pretrained(merged_path)
            restored = (
                Qwen3_5ForCausalLM.from_pretrained(merged_path, dtype=torch.bfloat16)
                .to("cuda")
                .eval()
            )
            with torch.no_grad():
                torch.testing.assert_close(
                    restored(**inputs).logits.float(), actual, rtol=0, atol=0
                )


if __name__ == "__main__":
    unittest.main()
