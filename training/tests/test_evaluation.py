"""Regression coverage for evaluating the added decision datasets."""

import importlib.util
import math
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "evaluate-accuracy.py"
SPEC = importlib.util.spec_from_file_location("jet_accuracy", SCRIPT)
assert SPEC and SPEC.loader
accuracy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(accuracy)


class EvaluationTests(unittest.TestCase):
    def test_generic_choice_semantic_equivalence_and_calibration(self):
        gold = [{"id": "arc:1", "dataset": "arc_easy", "task_type": "choice", "gold": "A"}]
        requests = [
            {
                "state": {"id": "arc:1"},
                "questions": {
                    "answer": {
                        "type": "choice",
                        "criteria": {"A": "same", "B": "same", "C": "other"},
                    }
                },
            }
        ]
        responses = [
            {
                "answers": {
                    "answer": {"choice": "B", "probabilities": {"A": 0.2, "B": 0.5, "C": 0.3}}
                }
            }
        ]
        report = accuracy.evaluate(gold, responses, requests)
        result = report["datasets"]["arc_easy"]
        self.assertEqual(result["correct"], 1)
        self.assertAlmostEqual(result["mean_gold_nll"], -math.log(0.7))
        self.assertAlmostEqual(result["mean_brier"], 0.18)
        self.assertAlmostEqual(result["ece_10_bins"], 0.3)

    def test_existing_boolq_and_mmlu_without_task_type(self):
        gold = [
            {"id": "b", "dataset": "boolq", "gold": True},
            {"id": "m", "dataset": "mmlu", "subject": "test", "gold": "B"},
        ]
        responses = [
            {"answers": {"answer": {"noul": 0.8}}},
            {"answers": {"answer": {"choice": "B", "probabilities": {"A": 0.1, "B": 0.9}}}},
        ]
        report = accuracy.evaluate(gold, responses)
        self.assertEqual(report["overall"]["correct"], 2)
        self.assertEqual(report["mmlu_by_subject"]["test"]["correct"], 1)

    def test_invalid_probabilities_fail_instead_of_good_metrics(self):
        gold = [{"id": "b", "dataset": "boolq", "gold": True}]
        for value in (float("nan"), float("inf"), -0.1, 1.1, True):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                accuracy.evaluate(gold, [{"answers": {"answer": {"noul": value}}}])

    def test_failure_stays_visible(self):
        report = accuracy.evaluate(
            [{"id": "snli:1", "dataset": "snli", "task_type": "choice", "gold": "neutral"}],
            [{"error": {"message": "failed"}}],
        )
        self.assertEqual(len(report["failures"]), 1)
        self.assertEqual(report["overall"]["examples"], 0)
        self.assertIsNone(report["datasets"]["snli"]["accuracy"])


if __name__ == "__main__":
    unittest.main()
