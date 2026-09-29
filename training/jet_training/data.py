"""Prepare pinned public text data using Jet's actual candidate prompt exporter.

Run with ``python -m jet_training.data --jet-binary PATH --output-dir PATH``.
Only official training rows enter train/. BoolQ validation remains a final test.
All held-out content is reserved before bounded deterministic training sampling.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import re
import subprocess
import sys
import unicodedata
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

DEFAULT_LOCK = Path(__file__).resolve().parents[1] / "datasets.lock.json"
SPLIT_PRIORITY = ("test", "validation", "train")
SNLI_LABELS = ("Entailment", "Neutral", "Contradiction")


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(text: str) -> str:
    """Label-blind comparison only; preserve original text in actual prompts."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def semantic_key(source_id: str, row: dict[str, Any]) -> str:
    if source_id == "boolq":
        fields = ["reading", normalize(row["passage"]), normalize(row["question"])]
    elif source_id in ("arc_easy", "arc_challenge"):
        # ARC subsets share a namespace; reordered choices and labels cannot hide leakage.
        fields = [
            "multiple_choice",
            normalize(row["question"]),
            sorted(normalize(value) for value in row["choices"]["text"]),
        ]
    elif source_id == "snli":
        fields = ["inference", normalize(row["premise"]), normalize(row["hypothesis"])]
    else:
        raise ValueError(f"unsupported source {source_id!r}")
    return hashlib.sha256(json_bytes(fields)).hexdigest()


def output_split(source_id: str, source_split: str) -> str:
    # The existing BoolQ benchmark must not become a checkpoint selection dataset.
    return "test" if source_id == "boolq" and source_split == "validation" else source_split


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def invalid_reason(source_id: str, row: dict[str, Any]) -> str | None:
    if source_id == "boolq":
        valid = (
            _text(row.get("passage"))
            and _text(row.get("question"))
            and isinstance(row.get("answer"), bool)
        )
    elif source_id in ("arc_easy", "arc_challenge"):
        choices = row.get("choices", {})
        labels, texts = choices.get("label", []), choices.get("text", [])
        valid = (
            _text(row.get("question"))
            and len(labels) == len(texts)
            and 2 <= len(labels) <= 26
            and len(set(labels)) == len(labels)
            and all(_text(item) for item in labels + texts)
            and row.get("answerKey") in labels
        )
        if valid and len({normalize(item) for item in texts}) != len(texts):
            return "duplicate_candidate_text"
    elif source_id == "snli":
        if row.get("label") not in (0, 1, 2):
            return "invalid_label"
        valid = _text(row.get("premise")) and _text(row.get("hypothesis"))
    else:
        raise ValueError(f"unsupported source {source_id!r}")
    return None if valid else "malformed_row"


def rank(seed: int, namespace: str, value: Any) -> bytes:
    return hashlib.sha256(json_bytes([seed, namespace, value])).digest()


def make_example(
    source: dict[str, Any],
    source_split: str,
    index: int,
    row: dict[str, Any],
    source_file: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    source_id = source["id"]
    identifier = f"{source_id}:{source_split}:{index}"
    split = output_split(source_id, source_split)
    state: dict[str, Any] = {"dataset": source_id, "id": identifier}
    if source_id == "boolq":
        state.update(passage=row["passage"], question=row["question"])
        question = {"type": "noul", "instructions": "Answer the question using only the passage."}
        gold: bool | str = row["answer"]
        gold_key = str(gold).lower()
    else:
        if source_id == "snli":
            state.update(premise=row["premise"], hypothesis=row["hypothesis"])
            texts = list(SNLI_LABELS)
            correct = row["label"]
            instructions = (
                "Determine the relationship of the hypothesis to the premise. "
                "Entailment means the hypothesis must be true given the premise; "
                "Contradiction means it must be false; Neutral means neither follows."
            )
        else:
            state["question"] = row["question"]
            texts = row["choices"]["text"]
            correct = row["choices"]["label"].index(row["answerKey"])
            instructions = "Choose the single correct answer."
        order = list(range(len(texts)))
        if split == "train":
            order.sort(key=lambda item: rank(seed, identifier, item))
        criteria = {chr(65 + position): texts[item] for position, item in enumerate(order)}
        gold = gold_key = chr(65 + order.index(correct))
        question = {"type": "choice", "instructions": instructions, "criteria": criteria}
    provenance = {
        "repository": source["repository"],
        "revision": source["revision"],
        "config": source["config"],
        "source_split": source_split,
        "row_index": index,
        "source_file": source_file["path"],
        "source_file_sha256": source_file["sha256"],
        "row_sha256": hashlib.sha256(json_bytes(row)).hexdigest(),
    }
    return {
        "request": {"state": state, "questions": {"answer": question}},
        "gold": {
            "dataset": source_id,
            "id": identifier,
            "gold": gold,
            "task_type": question["type"],
        },
        "gold_key": gold_key,
        "provenance": provenance,
        "content_sha256": semantic_key(source_id, row),
        "split": split,
    }


def collect_examples(
    sources: list[dict[str, Any]],
    rows: dict[tuple[str, str], Iterable[dict]],
    *,
    seed: int,
    limit_per_source: int | None = None,
    snli_train_limit: int = 10_000,
    excluded_keys: set[str] | None = None,
) -> tuple[dict[str, list[dict]], dict[str, Any]]:
    """Scan full held-out sets before sampling; keep memory bounded for SNLI train."""
    result: dict[str, list[dict]] = {split: [] for split in SPLIT_PRIORITY}
    counts: dict[str, Any] = {}
    excluded = excluded_keys or set()
    seen: set[str] = set()
    for split in SPLIT_PRIORITY:
        for source in sources:
            source_id = source["id"]
            for source_file in source["files"]:
                source_split = source_file["split"]
                if output_split(source_id, source_split) != split:
                    continue
                key = f"{source_id}/{source_split}"
                counter: Counter = Counter()
                counts[key] = {"output_split": split, "dropped": {}, "counts": counter}
                dropped: Counter = Counter()
                limit = snli_train_limit if split == "train" and source_id == "snli" else None
                if limit_per_source is not None:
                    limit = min(limit, limit_per_source) if limit is not None else limit_per_source
                # Heap contains the smallest deterministic ranks, independent of Python RNG.
                selected: list[tuple[int, int, dict]] = []
                for index, row in enumerate(rows[(source_id, source_split)]):
                    counter["input_rows"] += 1
                    reason = invalid_reason(source_id, row)
                    if reason:
                        dropped[reason] += 1
                        continue
                    content_key = semantic_key(source_id, row)
                    if content_key in excluded:
                        dropped["external_holdout_overlap"] += 1
                        continue
                    if content_key in seen:
                        dropped["duplicate_content"] += 1
                        continue
                    seen.add(content_key)
                    counter["eligible_rows"] += 1
                    priority = int.from_bytes(rank(seed, key, index), "big")
                    if limit is not None and len(selected) >= limit and priority >= -selected[0][0]:
                        continue
                    example = make_example(source, source_split, index, row, source_file, seed)
                    item = (-priority, index, example)
                    if limit is not None and len(selected) >= limit:
                        heapq.heapreplace(selected, item)
                    else:
                        heapq.heappush(selected, item)
                examples = [entry[2] for entry in sorted(selected, key=lambda item: item[1])]
                counter["selected_rows"] = len(examples)
                dropped["sampling_limit"] = counter["eligible_rows"] - len(examples)
                counts[key]["dropped"] = dict(sorted(dropped.items()))
                counts[key]["counts"] = dict(counter)
                result[split].extend(examples)
                print(
                    f"{key}: {len(examples)} selected of {counter['input_rows']} rows", flush=True
                )
    # Interleave datasets for training while keeping test/validation source order inspectable.
    result["train"].sort(key=lambda item: rank(seed, "train-mixture", item["gold"]["id"]))
    return result, counts


def load_exclusions(paths: list[Path]) -> tuple[set[str], list[dict[str, Any]]]:
    """Reserve frozen benchmark request content without reading its gold labels."""
    keys: set[str] = set()
    metadata = []
    for path in paths:
        count = 0
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                request = json.loads(line)
                state = request.get("state", {})
                questions = request.get("questions", {})
                question = questions.get("answer", {})
                criteria = question.get("criteria", {})
                if (
                    question.get("type") == "choice"
                    and _text(state.get("question"))
                    and isinstance(criteria, dict)
                    and criteria
                    and all(_text(item) for item in criteria.values())
                ):
                    key = semantic_key(
                        "arc_easy",
                        {
                            "question": state["question"],
                            "choices": {"text": list(criteria.values())},
                        },
                    )
                elif _text(state.get("passage")) and _text(state.get("question")):
                    key = semantic_key("boolq", state)
                elif _text(state.get("premise")) and _text(state.get("hypothesis")):
                    key = semantic_key("snli", state)
                else:
                    raise ValueError(f"unsupported exclusion request at {path}:{line_number}")
                keys.add(key)
                count += 1
        metadata.append({"path": str(path.resolve()), "sha256": sha256_file(path), "rows": count})
    return keys, metadata


def read_lock(path: Path) -> dict[str, Any]:
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("schema_version") != 1:
        raise ValueError("unsupported dataset lock schema")
    expected = {"boolq", "arc_easy", "arc_challenge", "snli"}
    sources = lock.get("sources", [])
    if {source["id"] for source in sources} != expected or len(sources) != len(expected):
        raise ValueError(
            "dataset lock must contain BoolQ, ARC Easy/Challenge, and SNLI exactly once"
        )
    for source in sources:
        if not re.fullmatch(r"[0-9a-f]{40}", source["revision"]):
            raise ValueError(f"{source['id']}: revision must be an immutable commit SHA")
        splits = [item["split"] for item in source["files"]]
        expected_splits = (
            {"train", "validation"} if source["id"] == "boolq" else set(SPLIT_PRIORITY)
        )
        if set(splits) != expected_splits or len(splits) != len(expected_splits):
            raise ValueError(f"{source['id']}: expected one parquet per official split")
        for item in source["files"]:
            if (
                not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
                or not item["path"].endswith(".parquet")
                or item["bytes"] <= 0
            ):
                raise ValueError(f"{source['id']}: invalid source file metadata")
    return lock


def load_sources(
    lock: dict[str, Any], cache_dir: Path, offline: bool
) -> dict[tuple[str, str], Any]:
    # Set these before importing libraries so --offline performs no HTTP requests.
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    try:
        from datasets import load_dataset
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError(
            "Install the training data dependencies first (see training/README.md)."
        ) from error
    result = {}
    for source in lock["sources"]:
        for item in source["files"]:
            path = Path(
                hf_hub_download(
                    repo_id=source["repository"],
                    repo_type="dataset",
                    filename=item["path"],
                    revision=source["revision"],
                    cache_dir=str(cache_dir / "hub"),
                    local_files_only=offline,
                )
            )
            if path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
                raise RuntimeError(
                    f"Source checksum mismatch: {source['repository']}/{item['path']}"
                )
            # Built-in parquet reader only: never import code from a dataset repository.
            result[(source["id"], item["split"])] = load_dataset(
                "parquet",
                data_files={item["split"]: str(path)},
                split=item["split"],
                revision=source["revision"],
                cache_dir=str(cache_dir / "datasets"),
            )
    return result


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("wb") as stream:
        for row in rows:
            stream.write(json_bytes(row) + b"\n")


def training_records(examples: list[dict], exported_path: Path) -> Iterator[dict]:
    with exported_path.open(encoding="utf-8") as stream:
        for index, example in enumerate(examples):
            line = stream.readline()
            if not line:
                raise RuntimeError(f"prompt exporter returned fewer than {len(examples)} requests")
            exported = json.loads(line)
            questions = exported.get("questions", [])
            if len(questions) != 1 or questions[0].get("question_id") != "answer":
                raise RuntimeError(f"unexpected exported question at line {index + 1}")
            question = questions[0]
            candidates = question["candidates"]
            matches = [item["target"] for item in candidates if item["key"] == example["gold_key"]]
            if len(matches) != 1:
                raise RuntimeError(f"missing or ambiguous gold candidate at line {index + 1}")
            yield {
                "id": example["gold"]["id"],
                "dataset": example["gold"]["dataset"],
                "split": example["split"],
                "messages": question["messages"],
                "target": matches[0],
                "candidates": candidates,
                "gold_key": example["gold_key"],
                "provenance": example["provenance"],
                "content_sha256": example["content_sha256"],
            }
        if stream.read().strip():
            raise RuntimeError("prompt exporter returned extra requests")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jet-binary",
        type=Path,
        required=True,
        help="Jet binary exposing the model-free export-prompts command",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="New or empty output directory"
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_LOCK.parent / "data" / "cache")
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--seed", type=int, default=20_260_929)
    parser.add_argument("--snli-train-limit", type=int, default=10_000)
    parser.add_argument(
        "--limit-per-source",
        type=int,
        help="Smoke-test cap per source/split; still scans all rows to prevent leakage",
    )
    parser.add_argument(
        "--offline", action="store_true", help="Require cached, checksum-verified sources"
    )
    parser.add_argument(
        "--exclude-requests",
        type=Path,
        action="append",
        default=[],
        help="Frozen benchmark Jet requests JSONL to exclude by content (repeatable)",
    )
    args = parser.parse_args(argv)
    if args.snli_train_limit <= 0 or (
        args.limit_per_source is not None and args.limit_per_source <= 0
    ):
        parser.error("row limits must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    binary = args.jet_binary.resolve(strict=True)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory must be empty: {output}")
    lock = read_lock(args.lock)
    excluded, exclusion_files = load_exclusions(args.exclude_requests)
    rows = load_sources(lock, args.cache_dir.resolve(), args.offline)
    splits, counts = collect_examples(
        lock["sources"],
        rows,
        seed=args.seed,
        limit_per_source=args.limit_per_source,
        snli_train_limit=args.snli_train_limit,
        excluded_keys=excluded,
    )
    output.mkdir(parents=True, exist_ok=True)
    files = {}
    for split in ("train", "validation", "test"):
        directory = output / split
        directory.mkdir()
        examples = splits[split]
        write_jsonl(directory / "requests.jsonl", (item["request"] for item in examples))
        write_jsonl(directory / "gold.jsonl", (item["gold"] for item in examples))
        subprocess.run(
            [
                str(binary),
                "export-prompts",
                "--input",
                str(directory / "requests.jsonl"),
                "--output",
                str(directory / "prompts.jsonl"),
            ],
            check=True,
        )
        write_jsonl(
            directory / "records.jsonl", training_records(examples, directory / "prompts.jsonl")
        )
        for path in sorted(directory.iterdir()):
            files[path.relative_to(output).as_posix()] = {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "rows": len(examples),
            }
    manifest = {
        "schema_version": 1,
        "status": "ready",
        "seed": args.seed,
        "limit_per_source": args.limit_per_source,
        "snli_train_limit": args.snli_train_limit,
        "dataset_lock_sha256": sha256_file(args.lock),
        "sources": lock["sources"],
        "jet_binary_sha256": sha256_file(binary),
        "counts": counts,
        "files": files,
        "external_holdouts": {"files": exclusion_files, "unique_content_keys": len(excluded)},
        "split_counts": {key: len(value) for key, value in splits.items()},
        "deduplication": {
            "normalization": "Unicode NFKC, casefold, collapse whitespace; excludes gold labels and IDs",
            "priority": list(SPLIT_PRIORITY),
            "stage": "all valid source rows before sampling",
            "arc_key": "question and sorted candidate texts; shared namespace across ARC configurations",
            "scope": "exact normalized source content; no fuzzy or base-pretraining contamination guarantee",
        },
        "held_out_policy": "BoolQ validation is test only; ARC/SNLI validation selects checkpoints; "
        "official tests are final evaluation only. MMLU is never a training source; "
        "frozen benchmarks are excluded when supplied via --exclude-requests.",
        "transformations": "Reformat official text as Jet requests; drop invalid labels/duplicate "
        "content; deterministically cap and mix training; permute train choice order.",
    }
    (output / "manifest.json").write_bytes(
        json.dumps(manifest, ensure_ascii=False, indent=2).encode() + b"\n"
    )
    print(json.dumps(manifest["split_counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"jet-training-data: {error}", file=sys.stderr)
        raise SystemExit(1) from error
