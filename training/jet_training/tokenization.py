"""Answer-only tokenization matching JET's complete candidate continuations."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def validate_record(record: dict[str, Any]) -> None:
    for field in ("id", "dataset", "target", "gold_key"):
        if not isinstance(record.get(field), str) or not record[field]:
            raise ValueError(f"record requires nonempty string {field!r}")
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("record requires messages")
    if [message.get("role") for message in messages] != ["system", "user"]:
        raise ValueError("JET records require exactly system and user messages")
    if any(not isinstance(message.get("content"), str) for message in messages):
        raise ValueError("only text message content is supported")
    candidates = record.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("record requires candidates")
    keys = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise TypeError("candidates must be objects")
        key, target = candidate.get("key"), candidate.get("target")
        if not isinstance(key, str) or not key or not isinstance(target, str):
            raise ValueError("candidate key and target must be strings")
        try:
            semantic_text = json.loads(target)
        except (ValueError, TypeError) as error:
            raise ValueError("candidate target must be a JSON-quoted string") from error
        if not isinstance(semantic_text, str) or not semantic_text.strip():
            raise ValueError("candidate target must quote nonempty semantic text")
        keys.append(key)
    if len(set(keys)) != len(keys):
        raise ValueError("candidate keys must be unique")
    gold = [item for item in candidates if item["key"] == record["gold_key"]]
    if len(gold) != 1 or gold[0]["target"] != record["target"]:
        raise ValueError("target must equal the gold candidate's semantic target")


def tokenize_record(record: dict[str, Any], tokenizer: Any, max_length: int) -> dict | None:
    """Return an unpadded example, or drop the whole overlength example.

    Do not append EOS: JET scores the quoted continuation without an end token.
    Tokenizing the joined text detects token merges across the prompt boundary.
    """
    validate_record(record)
    prompt = tokenizer.apply_chat_template(
        record["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    joined_ids = tokenizer(prompt + record["target"], add_special_tokens=False)["input_ids"]
    if not prompt_ids or joined_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(f"{record['id']}: tokenizer boundary is unstable")
    target_ids = joined_ids[len(prompt_ids) :]
    if not target_ids:
        raise ValueError(f"{record['id']}: target has no tokens")
    if len(joined_ids) > max_length:
        return None
    return {
        "input_ids": joined_ids,
        "attention_mask": [1] * len(joined_ids),
        "labels": [-100] * len(prompt_ids) + target_ids,
    }


def load_tokenized(path: Path, tokenizer: Any, max_length: int) -> tuple[list[dict], dict]:
    examples = []
    ids: set[str] = set()
    kept: Counter = Counter()
    dropped: Counter = Counter()
    dropped_ids = []
    kept_ids = []
    target_tokens = 0
    longest = 0
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                validate_record(record)
                if record["id"] in ids:
                    raise ValueError(f"duplicate id {record['id']!r}")
                ids.add(record["id"])
                example = tokenize_record(record, tokenizer, max_length)
            except (ValueError, TypeError, KeyError, AttributeError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            if example is None:
                dropped[record["dataset"]] += 1
                dropped_ids.append(record["id"])
                continue
            kept[record["dataset"]] += 1
            kept_ids.append(record["id"])
            examples.append(example)
            target_tokens += sum(value != -100 for value in example["labels"])
            longest = max(longest, len(example["input_ids"]))
    report = {
        "input_records": len(ids),
        "kept": len(examples),
        "dropped_overlength": len(dropped_ids),
        "dropped_ids": dropped_ids,
        "kept_by_dataset": dict(kept),
        "dropped_by_dataset": dict(dropped),
        "target_tokens": target_tokens,
        "longest_kept_tokens": longest,
        "max_length": max_length,
        "truncated": 0,
        "kept_ids": kept_ids,
    }
    return examples, report


def pad_examples(examples: list[dict], pad_token_id: int) -> dict[str, list[list[int]]]:
    if not examples:
        raise ValueError("cannot collate an empty batch")
    width = max(len(example["input_ids"]) for example in examples)
    batch: dict[str, list[list[int]]] = {"input_ids": [], "attention_mask": [], "labels": []}
    for example in examples:
        missing = width - len(example["input_ids"])
        for key, padding in (("input_ids", pad_token_id), ("attention_mask", 0), ("labels", -100)):
            batch[key].append(example[key] + [padding] * missing)
    return batch


class AnswerOnlyCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples: list[dict]) -> dict:
        import torch

        return {
            key: torch.tensor(value, dtype=torch.long)
            for key, value in pad_examples(examples, self.pad_token_id).items()
        }


class TokenizedDataset:
    def __init__(self, examples: list[dict]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        return self.examples[index]
