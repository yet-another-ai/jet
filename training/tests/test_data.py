"""Leakage boundaries and Jet candidate fidelity without ML dependencies."""

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path

from jet_training.data import (
    DEFAULT_LOCK,
    collect_examples,
    invalid_reason,
    load_exclusions,
    make_example,
    read_lock,
    semantic_key,
    training_records,
)


def source(name, splits=("train", "validation", "test")):
    return {
        "id": name,
        "repository": f"fixture/{name}",
        "revision": "a" * 40,
        "config": "default",
        "files": [
            {"split": split, "path": f"{split}.parquet", "sha256": "b" * 64} for split in splits
        ],
    }


def snli(premise, label=0):
    return {"premise": premise, "hypothesis": "A person moves.", "label": label}


class DataTests(unittest.TestCase):
    def test_lock_is_complete_and_immutable(self):
        lock = read_lock(DEFAULT_LOCK)
        self.assertEqual(len(lock["sources"]), 4)
        for item in lock["sources"]:
            self.assertIn("/by-sa/", item["license_url"])

    def test_label_blind_normalized_dedup_across_arc_subsets(self):
        easy = {
            "question": " Which  THING? ",
            "choices": {"label": ["A", "B"], "text": ["Ａ cat", "A dog"]},
            "answerKey": "A",
        }
        challenge = {
            "question": "which thing?",
            "choices": {"label": ["1", "2"], "text": ["a dog", "a cat"]},
            "answerKey": "1",
        }
        self.assertEqual(semantic_key("arc_easy", easy), semantic_key("arc_challenge", challenge))

    def test_heldout_priority_and_unsampled_test_rows_block_training(self):
        spec = source("snli")
        rows = {
            ("snli", "test"): [snli("first test"), snli("second test")],
            ("snli", "validation"): [snli("FIRST TEST", 1), snli("validation only")],
            ("snli", "train"): [
                snli("first test", 2),
                snli("second test"),
                snli("validation only", 1),
                snli("train only"),
            ],
        }
        with contextlib.redirect_stdout(io.StringIO()):
            selected, counts = collect_examples([spec], rows, seed=17, limit_per_source=1)
        self.assertEqual(selected["train"][0]["request"]["state"]["premise"], "train only")
        self.assertEqual(counts["snli/train"]["dropped"]["duplicate_content"], 3)
        self.assertEqual(counts["snli/validation"]["dropped"]["duplicate_content"], 1)
        self.assertEqual(counts["snli/test"]["dropped"]["sampling_limit"], 1)

    def test_boolq_validation_is_final_test(self):
        spec = source("boolq", ("train", "validation"))
        row = {"passage": "A cat sits.", "question": "Does it sit?", "answer": True}
        rows = {("boolq", "validation"): [row], ("boolq", "train"): [dict(row, answer=False)]}
        with contextlib.redirect_stdout(io.StringIO()):
            selected, counts = collect_examples([spec], rows, seed=1)
        self.assertEqual(len(selected["test"]), 1)
        self.assertEqual(selected["train"], [])
        self.assertEqual(selected["validation"], [])
        self.assertEqual(selected["test"][0]["gold_key"], "true")
        self.assertEqual(counts["boolq/train"]["dropped"]["duplicate_content"], 1)

    def test_frozen_mmlu_request_blocks_reordered_arc_training(self):
        spec = source("arc_easy")
        arc = {
            "question": " The QUESTION? ",
            "choices": {"text": ["first", "second"], "label": ["1", "2"]},
            "answerKey": "2",
        }
        request = {
            "state": {"dataset": "mmlu", "id": "mmlu:test:0", "question": "the question?"},
            "questions": {"answer": {"type": "choice", "criteria": {"A": "SECOND", "B": "first"}}},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mmlu.jsonl"
            path.write_text(json.dumps(request) + "\n", encoding="utf-8")
            keys, files = load_exclusions([path])
        rows = {
            ("arc_easy", "train"): [arc],
            ("arc_easy", "test"): [],
            ("arc_easy", "validation"): [],
        }
        with contextlib.redirect_stdout(io.StringIO()):
            selected, counts = collect_examples([spec], rows, seed=1, excluded_keys=keys)
        self.assertEqual(selected["train"], [])
        self.assertEqual(counts["arc_easy/train"]["dropped"]["external_holdout_overlap"], 1)
        self.assertEqual(files[0]["rows"], 1)
        self.assertEqual(len(files[0]["sha256"]), 64)

    def test_snli_invalid_labels_are_excluded(self):
        self.assertEqual(invalid_reason("snli", snli("premise", -1)), "invalid_label")
        self.assertIsNone(invalid_reason("snli", snli("premise", 2)))

    def test_choice_permutation_preserves_gold_semantics(self):
        spec = source("arc_easy")
        row = {
            "question": "Which is correct?",
            "choices": {
                "label": ["1", "2", "3", "4"],
                "text": ["wrong1", "correct", "wrong2", "wrong3"],
            },
            "answerKey": "2",
        }
        positions = set()
        for seed in range(10):
            example = make_example(spec, "train", 0, row, spec["files"][0], seed)
            criteria = example["request"]["questions"]["answer"]["criteria"]
            self.assertEqual(criteria[example["gold_key"]], "correct")
            positions.add(example["gold_key"])
        self.assertGreater(len(positions), 1)

    def test_bounded_selection_is_repeatable_and_seed_sensitive(self):
        spec = source("snli")
        rows = {(spec["id"], split): [] for split in ("test", "validation", "train")}
        rows[("snli", "train")] = [snli(f"person {index}") for index in range(40)]
        with contextlib.redirect_stdout(io.StringIO()):
            first, _ = collect_examples([spec], rows, seed=1, snli_train_limit=5)
            repeat, _ = collect_examples([spec], rows, seed=1, snli_train_limit=5)
            changed, _ = collect_examples([spec], rows, seed=2, snli_train_limit=5)
        self.assertEqual(first, repeat)
        self.assertEqual(len(first["train"]), 5)
        self.assertNotEqual(first, changed)

    def test_records_use_exact_rust_target_and_reject_missing_gold(self):
        spec = source("boolq", ("train", "validation"))
        example = make_example(
            spec,
            "train",
            0,
            {
                "passage": "A cat sits.",
                "question": "Does it sit?",
                "answer": True,
            },
            spec["files"][0],
            1,
        )
        target = '"Yes\\n\\"quoted\\""'
        payload = {
            "questions": [
                {
                    "question_id": "answer",
                    "messages": [
                        {"role": "system", "content": "from Rust"},
                        {"role": "user", "content": "question"},
                    ],
                    "candidates": [
                        {"key": "true", "target": target},
                        {"key": "false", "target": '"No"'},
                    ],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.jsonl"
            path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            records = list(training_records([example], path))
            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(record["target"], target)
            self.assertEqual(record["messages"], payload["questions"][0]["messages"])
            self.assertEqual(record["provenance"]["source_split"], "train")
            bad = copy.deepcopy(payload)
            bad["questions"][0]["candidates"] = [{"key": "false", "target": '"No"'}]
            path.write_text(json.dumps(bad) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "gold candidate"):
                list(training_records([example], path))
            path.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "fewer"):
                list(training_records([example], path))
            path.write_text((json.dumps(payload) + "\n") * 2, encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "extra"):
                list(training_records([example], path))


if __name__ == "__main__":
    unittest.main()
