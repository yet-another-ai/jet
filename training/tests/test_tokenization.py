import copy
import json
import tempfile
import unittest
from pathlib import Path

from jet_training.tokenization import load_tokenized, pad_examples, tokenize_record


class CharacterTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        return "|user|" + messages[-1]["content"] + "|assistant|"

    def __call__(self, text, **kwargs):
        assert kwargs == {"add_special_tokens": False}
        return {"input_ids": [ord(character) for character in text]}


def record(target='"Yes"'):
    return {
        "id": "unit:1",
        "dataset": "unit",
        "messages": [
            {"role": "system", "content": "Choose one"},
            {"role": "user", "content": "Question"},
        ],
        "target": target,
        "gold_key": "true",
        "candidates": [{"key": "false", "target": '"No"'}, {"key": "true", "target": target}],
    }


class TokenizationTests(unittest.TestCase):
    def test_only_exact_candidate_is_supervised_without_eos(self):
        example = tokenize_record(record('"中文\\nanswer"'), CharacterTokenizer(), 1000)
        supervised = [token for token in example["labels"] if token != -100]
        self.assertEqual(supervised, [ord(character) for character in '"中文\\nanswer"'])
        self.assertEqual(example["input_ids"][-len(supervised) :], supervised)
        self.assertEqual(
            example["labels"][: -len(supervised)],
            [-100] * (len(example["labels"]) - len(supervised)),
        )

    def test_padding_does_not_supervise_prompt_or_pad_tokens(self):
        examples = [
            tokenize_record(record(value), CharacterTokenizer(), 1000)
            for value in ('"Yes"', '"Much longer answer"')
        ]
        batch = pad_examples(examples, 0)
        padding = len(batch["input_ids"][0]) - len(examples[0]["input_ids"])
        self.assertGreater(padding, 0)
        self.assertEqual(batch["labels"][0][-padding:], [-100] * padding)
        self.assertEqual(batch["attention_mask"][0][-padding:], [0] * padding)
        self.assertEqual(batch["input_ids"][0][-padding:], [0] * padding)

    def test_unstable_tokenizer_boundary_fails(self):
        class MergingTokenizer(CharacterTokenizer):
            def __call__(self, text, **kwargs):
                result = super().__call__(text, **kwargs)
                if text.endswith('"Yes"'):
                    result["input_ids"][0] += 1
                return result

        with self.assertRaisesRegex(ValueError, "boundary is unstable"):
            tokenize_record(record(), MergingTokenizer(), 1000)

    def test_overlength_drops_complete_answer_and_reports_count(self):
        exact = tokenize_record(record(), CharacterTokenizer(), 1000)
        self.assertIsNone(
            tokenize_record(record(), CharacterTokenizer(), len(exact["input_ids"]) - 1)
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text(json.dumps(record()) + "\n", encoding="utf-8")
            examples, report = load_tokenized(path, CharacterTokenizer(), 1)
        self.assertEqual(examples, [])
        self.assertEqual(report["dropped_overlength"], 1)
        self.assertEqual(report["dropped_ids"], ["unit:1"])
        self.assertEqual(report["truncated"], 0)

    def test_label_key_cannot_replace_semantic_target(self):
        invalid = copy.deepcopy(record())
        invalid["target"] = "true"
        with self.assertRaisesRegex(ValueError, "gold candidate"):
            tokenize_record(invalid, CharacterTokenizer(), 1000)


if __name__ == "__main__":
    unittest.main()
