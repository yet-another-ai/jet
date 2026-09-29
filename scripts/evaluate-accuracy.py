#!/usr/bin/env python3
"""Evaluate Jet noul/choice responses against labelled decision datasets."""

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
    parser.add_argument("--requests", type=Path,
                        help="Match choice labels with identical candidate text")
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
        "mean_brier": sum(row["brier"] for row in rows) / total if total else None,
        "ece_10_bins": calibration_error(rows) if total else None,
        "gold_counts": dict(
            sorted(Counter(str(row["gold"]).lower() for row in rows).items())
        ),
        "prediction_counts": dict(
            sorted(Counter(str(row["prediction"]).lower() for row in rows).items())
        ),
    }


def calibration_error(rows: list[dict[str, Any]]) -> float:
    bins: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        bins[min(9, int(row["confidence"] * 10))].append(row)
    return sum(
        abs(sum(row["confidence"] - row["correct"] for row in bucket))
        for bucket in bins.values()
    ) / len(rows)


def probability(value: Any, line_number: int) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value <= 1):
        raise RuntimeError(f"line {line_number} has invalid probability {value!r}")
    return float(value)


def evaluate(gold: list[dict[str, Any]], responses: list[dict[str, Any]],
             requests: list[dict[str, Any] | None] | None = None) -> dict[str, Any]:
    if requests is None:
        requests = [None] * len(gold)
    if len(gold) != len(responses):
        raise RuntimeError(
            f"line count mismatch: {len(gold)} gold labels, {len(responses)} responses"
        )
    if len(requests) != len(gold):
        raise RuntimeError(
            f"line count mismatch: {len(gold)} gold labels, {len(requests)} requests"
        )

    evaluated = []
    failures = []
    duplicate_candidates = []
    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line_number, (expected, response, request) in enumerate(
        zip(gold, responses, requests), 1
    ):
        dataset = expected["dataset"]
        task_type = expected.get("task_type", {"boolq": "noul", "mmlu": "choice"}.get(dataset))
        if task_type not in ("noul", "choice"):
            raise RuntimeError(f"line {line_number} has unknown task_type {task_type!r}")
        if task_type == "noul" and not isinstance(expected["gold"], bool):
            raise RuntimeError(f"line {line_number} requires a boolean gold label")
        criteria = None
        if request is not None:
            state = request.get("state")
            if not isinstance(state, dict) or state.get("id") != expected["id"]:
                raise RuntimeError(f"line {line_number} request/gold ID mismatch")
            if request.get("questions", {}).get("answer", {}).get("type") != task_type:
                raise RuntimeError(f"line {line_number} request/gold task type mismatch")
            if task_type == "choice":
                criteria = request.get("questions", {}).get("answer", {}).get("criteria")
                if not isinstance(criteria, dict) or not all(
                    isinstance(value, str) for value in criteria.values()
                ) or expected["gold"] not in criteria:
                    raise RuntimeError(f"line {line_number} has malformed choice criteria")
                repeated = {
                    text: sorted(key for key, value in criteria.items() if value == text)
                    for text in set(criteria.values())
                    if sum(value == text for value in criteria.values()) > 1
                }
                if repeated:
                    duplicate_candidates.append({
                        "line": line_number,
                        "id": expected["id"],
                        "equivalent_labels": sorted(repeated.values()),
                    })
        if "error" in response:
            failures.append(
                {"line": line_number, "id": expected["id"], "error": response["error"]}
            )
            continue
        answer = response.get("answers", {}).get("answer")
        if not isinstance(answer, dict):
            raise TypeError(f"line {line_number} has no answers.answer object")
        if task_type == "noul":
            probability_true = probability(answer.get("noul"), line_number)
            prediction = probability_true >= 0.5
            probability_gold = probability_true if expected["gold"] else 1 - probability_true
            confidence = max(probability_true, 1 - probability_true)
            # Sum-of-squares Brier convention for both binary and multiclass tasks.
            brier = 2 * (probability_true - float(expected["gold"])) ** 2
        else:
            prediction = answer.get("choice")
            probabilities = answer.get("probabilities")
            if (not isinstance(probabilities, dict)
                    or expected["gold"] not in probabilities
                    or prediction not in probabilities
                    or (criteria is not None and prediction not in criteria)):
                raise RuntimeError(f"line {line_number} has malformed choice probabilities")
            accepted = ([key for key, value in criteria.items()
                         if value == criteria[expected["gold"]]]
                        if criteria is not None else [expected["gold"]])
            if any(key not in probabilities for key in accepted):
                raise RuntimeError(f"line {line_number} has missing equivalent probabilities")
            probabilities = {key: probability(value, line_number)
                             for key, value in probabilities.items()}
            if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-5):
                raise RuntimeError(f"line {line_number} probabilities do not sum to one")
            if criteria is not None and set(criteria) != set(probabilities):
                raise RuntimeError(f"line {line_number} candidate/probability keys differ")
            probability_gold = min(1.0, sum(probabilities[key] for key in accepted))
            confidence = (min(1.0, sum(probabilities[key] for key, value in criteria.items()
                                       if value == criteria[prediction]))
                          if criteria is not None else probabilities[prediction])
            semantic_probabilities: dict[str, float] = defaultdict(float)
            for key, value in probabilities.items():
                semantic_probabilities[criteria[key] if criteria is not None else key] += value
            gold_semantic = criteria[expected["gold"]] if criteria is not None else expected["gold"]
            brier = sum((value - float(key == gold_semantic)) ** 2
                        for key, value in semantic_probabilities.items())
        probability_gold = max(float(probability_gold), sys.float_info.min)
        row = {
            "dataset": dataset,
            "id": expected["id"],
            "gold": expected["gold"],
            "prediction": prediction,
            "correct": (prediction in accepted if task_type == "choice"
                        else prediction == expected["gold"]),
            "gold_probability": probability_gold,
            "gold_nll": -math.log(probability_gold),
            "confidence": confidence,
            "brier": brier,
        }
        evaluated.append(row)
        if dataset == "mmlu":
            by_subject[expected["subject"]].append(row)

    grouped = {
        dataset: summarize([row for row in evaluated if row["dataset"] == dataset])
        for dataset in sorted({"boolq", "mmlu"} | {row["dataset"] for row in gold})
    }
    report = {
        "overall": summarize(evaluated),
        "datasets": grouped,
        "mmlu_by_subject": {
            subject: summarize(rows) for subject, rows in sorted(by_subject.items())
        },
        "failures": failures,
        "duplicate_candidates": duplicate_candidates,
    }
    return report


def main() -> int:
    args = parse_args()
    report = evaluate(read_jsonl(args.gold), read_jsonl(args.responses),
                      read_jsonl(args.requests) if args.requests else None)
    for dataset, summary in report["datasets"].items():
        accuracy = summary["accuracy"]
        nll = summary["mean_gold_nll"]
        if accuracy is not None and nll is not None:
            print(
                f"{dataset}: {summary['correct']}/{summary['examples']} "
                f"accuracy={accuracy:.4f} mean_gold_nll={nll:.4f}"
            )
    print(f"failures: {len(report['failures'])}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"evaluate-accuracy: {error}", file=sys.stderr)
        sys.exit(1)
