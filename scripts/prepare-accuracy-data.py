#!/usr/bin/env python3
"""Download pinned BoolQ/MMLU sources and build Jet accuracy JSONL files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable


BOOLQ_ROWS_URL = "https://datasets-server.huggingface.co/rows"
BOOLQ_ROWS = 3_270
BOOLQ_SHA256 = "0477ed7c7519c25ad2633c53581ed370f3bd1857dfac177af559b658969b0577"
MMLU_URL = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"
MMLU_SHA256 = "bec563ba4bac1d6aaf04141cd7d1605d7a5ca833e38f994051e818489592989b"
SOURCE_REVISIONS = {
    "boolq_repository": "google-research-datasets/boolean-questions@90af34107399cc7a446b373dc4ee35b8001da7c2",
    "boolq_huggingface": "google/boolq@35b264d03638db9f4ce671b711558bf7ff0f80d5",
    "mmlu_repository": "hendrycks/test@4450500f923c49f1fb1dd3d99108a0bd9717b660",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("tests/accuracy/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("tests/accuracy/generated"))
    parser.add_argument("--boolq-limit", type=int, default=256)
    parser.add_argument("--mmlu-per-subject", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20_260_921)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, expected: str) -> bool:
    return path.is_file() and sha256(path) == expected


def replace_checked(temporary: Path, destination: Path, expected: str) -> None:
    actual = sha256(temporary)
    if actual != expected:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"SHA-256 mismatch for {destination.name}: expected {expected}, got {actual}"
        )
    temporary.replace(destination)


def download_boolq(destination: Path) -> None:
    if verify(destination, BOOLQ_SHA256):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=destination.parent, delete=False
    ) as output:
        temporary = Path(output.name)
        for offset in range(0, BOOLQ_ROWS, 100):
            length = min(100, BOOLQ_ROWS - offset)
            query = urllib.parse.urlencode(
                {
                    "dataset": "google/boolq",
                    "config": "default",
                    "split": "validation",
                    "offset": offset,
                    "length": length,
                }
            )
            with urllib.request.urlopen(f"{BOOLQ_ROWS_URL}?{query}") as response:
                payload = json.load(response)
            rows = payload.get("rows", [])
            if len(rows) != length:
                raise RuntimeError(
                    f"BoolQ returned {len(rows)} rows at offset {offset}; expected {length}"
                )
            for item in rows:
                output.write(
                    json.dumps(item["row"], ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
    replace_checked(temporary, destination, BOOLQ_SHA256)


def download_file(url: str, destination: Path, expected: str) -> None:
    if verify(destination, expected):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as output:
        temporary = Path(output.name)
        with urllib.request.urlopen(url) as response:
            shutil.copyfileobj(response, output, length=1024 * 1024)
    replace_checked(temporary, destination, expected)


def extract_mmlu_tests(archive: Path, destination: Path) -> None:
    marker = destination / ".source-sha256"
    if marker.is_file() and marker.read_text(encoding="utf-8").strip() == MMLU_SHA256:
        return
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    count = 0
    with tarfile.open(archive) as source:
        for member in source.getmembers():
            path = Path(member.name)
            if (
                not member.isfile()
                or len(path.parts) != 3
                or path.parts[:2] != ("data", "test")
                or not path.name.endswith("_test.csv")
            ):
                continue
            extracted = source.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"could not read {member.name} from MMLU archive")
            with (destination / path.name).open("wb") as output:
                shutil.copyfileobj(extracted, output)
            count += 1
    if count != 57:
        raise RuntimeError(f"expected 57 MMLU test subjects, extracted {count}")
    marker.write_text(f"{MMLU_SHA256}\n", encoding="utf-8")


def rank(seed: int, namespace: str, index: int) -> bytes:
    return hashlib.sha256(f"{seed}\0{namespace}\0{index}".encode()).digest()


def select_indices(
    indices: Iterable[int], limit: int, seed: int, namespace: str
) -> list[int]:
    return sorted(
        sorted(indices, key=lambda index: rank(seed, namespace, index))[:limit]
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def boolq_examples(path: Path, limit: int, seed: int) -> list[tuple[dict, dict]]:
    rows = read_jsonl(path)
    if not 0 < limit <= len(rows):
        raise ValueError(f"--boolq-limit must be between 1 and {len(rows)}")
    false_indices = [index for index, row in enumerate(rows) if not row["answer"]]
    true_indices = [index for index, row in enumerate(rows) if row["answer"]]
    false_limit = min(limit // 2, len(false_indices))
    true_limit = min(limit - false_limit, len(true_indices))
    false_limit = min(limit - true_limit, len(false_indices))
    if false_limit + true_limit != limit:
        raise RuntimeError(f"could not select {limit} BoolQ examples")
    selected = select_indices(false_indices, false_limit, seed, "boolq:false")
    selected += select_indices(true_indices, true_limit, seed, "boolq:true")
    examples = []
    for index in sorted(selected):
        row = rows[index]
        example_id = f"boolq:validation:{index}"
        state = {
            "dataset": "boolq",
            "id": example_id,
            "passage": row["passage"],
            "question": row["question"],
        }
        if row.get("title"):
            state["title"] = row["title"]
        request = {
            "state": state,
            "questions": {
                "answer": {
                    "type": "noul",
                    "instructions": "Answer the question using only the passage.",
                }
            },
        }
        gold = {"dataset": "boolq", "id": example_id, "gold": row["answer"]}
        examples.append((request, gold))
    return examples


def mmlu_examples(directory: Path, per_subject: int, seed: int) -> list[tuple[dict, dict]]:
    if per_subject <= 0:
        raise ValueError("--mmlu-per-subject must be positive")
    examples = []
    labels = ("A", "B", "C", "D")
    for path in sorted(directory.glob("*_test.csv")):
        subject = path.name.removesuffix("_test.csv")
        with path.open(encoding="utf-8", newline="") as source:
            rows = list(csv.reader(source))
        if per_subject > len(rows):
            raise ValueError(
                f"--mmlu-per-subject={per_subject} exceeds {subject} size {len(rows)}"
            )
        selected = select_indices(
            range(len(rows)), per_subject, seed, f"mmlu:{subject}"
        )
        for index in selected:
            row = rows[index]
            if len(row) != 6 or row[5] not in labels:
                raise RuntimeError(f"malformed MMLU row {subject}:{index}")
            example_id = f"mmlu:test:{subject}:{index}"
            request = {
                "state": {
                    "dataset": "mmlu",
                    "id": example_id,
                    "subject": subject.replace("_", " "),
                    "question": row[0],
                },
                "questions": {
                    "answer": {
                        "type": "choice",
                        "instructions": "Choose the single correct answer.",
                        "criteria": dict(zip(labels, row[1:5])),
                    }
                },
            }
            gold = {
                "dataset": "mmlu",
                "id": example_id,
                "subject": subject,
                "gold": row[5],
            }
            examples.append((request, gold))
    if len(list(directory.glob("*_test.csv"))) != 57:
        raise RuntimeError("MMLU test directory does not contain 57 subjects")
    return examples


def write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for value in values:
            output.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> int:
    args = parse_args()
    downloads = args.data_dir / "downloads"
    boolq_path = downloads / "boolq-validation.jsonl"
    mmlu_archive = downloads / "mmlu-data.tar"
    mmlu_test = args.data_dir / "mmlu-test"

    download_boolq(boolq_path)
    download_file(MMLU_URL, mmlu_archive, MMLU_SHA256)
    extract_mmlu_tests(mmlu_archive, mmlu_test)

    examples = boolq_examples(boolq_path, args.boolq_limit, args.seed)
    boolq_count = len(examples)
    examples.extend(mmlu_examples(mmlu_test, args.mmlu_per_subject, args.seed))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "requests.jsonl", (item[0] for item in examples))
    write_jsonl(args.output_dir / "gold.jsonl", (item[1] for item in examples))

    manifest = {
        "seed": args.seed,
        "sources": {
            "boolq": {
                "rows_endpoint": BOOLQ_ROWS_URL,
                "split": "validation",
                "normalized_sha256": BOOLQ_SHA256,
            },
            "mmlu": {"url": MMLU_URL, "archive_sha256": MMLU_SHA256},
            **SOURCE_REVISIONS,
        },
        "selection": {
            "boolq": {"examples": boolq_count, "stratified_by": "answer"},
            "mmlu": {
                "examples": len(examples) - boolq_count,
                "subjects": 57,
                "per_subject": args.mmlu_per_subject,
            },
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {len(examples)} requests "
        f"({boolq_count} BoolQ, {len(examples) - boolq_count} MMLU) "
        f"to {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, urllib.error.URLError) as error:
        print(f"prepare-accuracy-data: {error}", file=sys.stderr)
        sys.exit(1)
