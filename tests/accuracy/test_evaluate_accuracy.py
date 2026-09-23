"""Regression checks for ambiguous answer labels in MMLU evaluation."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVALUATOR = ROOT / "scripts/evaluate-accuracy.py"


class EquivalentAnswersTest(unittest.TestCase):
    def test_equivalent_gold_label_counts_and_combines_probability(self) -> None:
        criteria = {"A": "same", "B": "same", "C": "other", "D": "last"}
        cases = [
            ("first", "A", "B", {"A": 0.2, "B": 0.2, "C": 0.5, "D": 0.1}),
            ("second", "D", "A", {"A": 0.3, "B": 0.3, "C": 0.2, "D": 0.2}),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            requests = [
                {"state": {"id": identity}, "questions": {"answer": {
                    "type": "choice", "instructions": "Choose", "criteria": criteria}}}
                for identity, _, _, _ in cases
            ]
            gold = [
                {"dataset": "mmlu", "id": identity, "subject": "fixture", "gold": label}
                for identity, label, _, _ in cases
            ]
            responses = [
                {"answers": {"answer": {"type": "choice", "choice": prediction,
                                          "probabilities": probabilities}}}
                for _, _, prediction, probabilities in cases
            ]
            for name, rows in (("requests", requests), ("gold", gold),
                               ("responses", responses)):
                (directory / f"{name}.jsonl").write_text(
                    "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
                )
            report_path = directory / "report.json"
            result = subprocess.run(
                [sys.executable, str(EVALUATOR), "--gold", str(directory / "gold.jsonl"),
                 "--requests", str(directory / "requests.jsonl"),
                 "--responses", str(directory / "responses.jsonl"),
                 "--report", str(report_path)],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["overall"]["examples"], 2)
            self.assertEqual(report["overall"]["correct"], 1)
            self.assertEqual(len(report["duplicate_candidates"]), 2)
            self.assertEqual(report["duplicate_candidates"][0]["equivalent_labels"],
                             [["A", "B"]])
            self.assertAlmostEqual(report["overall"]["mean_gold_nll"],
                                   (-math.log(0.4) - math.log(0.2)) / 2)
            self.assertAlmostEqual(report["overall"]["mean_confidence"], 0.5)


if __name__ == "__main__":
    unittest.main()
