#!/usr/bin/env python3
"""Evaluate Jet JSONL responses against generated BoolQ/MMLU gold labels."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    correct = sum(row["correct"] for row in rows)
    return {
        "examples": total,
        "correct": correct,
        "accuracy": correct / total if total else None,
        "mean_gold_nll": (
            sum(row["gold_nll"] for row in rows) / total if total else None
        ),
        "mean_confidence": (
            sum(row["confidence"] for row in rows) / total if total else None
        ),
        "gold_counts": dict(
            sorted(Counter(str(row["gold"]).lower() for row in rows).items())
        ),
        "prediction_counts": dict(
            sorted(Counter(str(row["prediction"]).lower() for row in rows).items())
        ),
    }


def main() -> int:
    args = parse_args()
    gold = read_jsonl(args.gold)
    responses = read_jsonl(args.responses)
    if len(gold) != len(responses):
        raise RuntimeError(
            f"line count mismatch: {len(gold)} gold labels, {len(responses)} responses"
        )

    evaluated = []
    failures = []
    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line_number, (expected, response) in enumerate(zip(gold, responses), 1):
        if "error" in response:
            failures.append(
                {"line": line_number, "id": expected["id"], "error": response["error"]}
            )
            continue
        answer = response.get("answers", {}).get("answer")
        if not isinstance(answer, dict):
            raise RuntimeError(f"line {line_number} has no answers.answer object")
        dataset = expected["dataset"]
        if dataset == "boolq":
            probability_true = answer.get("noul")
            if not isinstance(probability_true, (int, float)):
                raise RuntimeError(f"line {line_number} has no noul probability")
            prediction = probability_true >= 0.5
            probability_gold = probability_true if expected["gold"] else 1 - probability_true
            confidence = max(probability_true, 1 - probability_true)
        elif dataset == "mmlu":
            prediction = answer.get("choice")
            probabilities = answer.get("probabilities")
            if not isinstance(probabilities, dict) or expected["gold"] not in probabilities:
                raise RuntimeError(f"line {line_number} has malformed choice probabilities")
            probability_gold = probabilities[expected["gold"]]
            confidence = max(probabilities.values())
        else:
            raise RuntimeError(f"line {line_number} has unknown dataset {dataset!r}")
        probability_gold = max(float(probability_gold), sys.float_info.min)
        row = {
            "dataset": dataset,
            "id": expected["id"],
            "gold": expected["gold"],
            "prediction": prediction,
            "correct": prediction == expected["gold"],
            "gold_probability": probability_gold,
            "gold_nll": -math.log(probability_gold),
            "confidence": confidence,
        }
        evaluated.append(row)
        if dataset == "mmlu":
            by_subject[expected["subject"]].append(row)

    grouped = {
        dataset: summarize([row for row in evaluated if row["dataset"] == dataset])
        for dataset in ("boolq", "mmlu")
    }
    report = {
        "overall": summarize(evaluated),
        "datasets": grouped,
        "mmlu_by_subject": {
            subject: summarize(rows) for subject, rows in sorted(by_subject.items())
        },
        "failures": failures,
    }
    for dataset, summary in grouped.items():
        accuracy = summary["accuracy"]
        nll = summary["mean_gold_nll"]
        if accuracy is not None and nll is not None:
            print(
                f"{dataset}: {summary['correct']}/{summary['examples']} "
                f"accuracy={accuracy:.4f} mean_gold_nll={nll:.4f}"
            )
    print(f"failures: {len(failures)}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"evaluate-accuracy: {error}", file=sys.stderr)
        sys.exit(1)
