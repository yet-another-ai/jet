"""CPU-only checks for publishing complete, reproducible paper measurements."""

from collections import defaultdict
from contextlib import contextmanager
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("paper_results", ROOT / "paper/update_results.py")
PUBLISHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PUBLISHER)
RUNNER = PUBLISHER.load_module("fixture_runner", ROOT / "scripts/rerun-paper-4090.py")
EVALUATOR = PUBLISHER.load_module("fixture_evaluator", ROOT / "scripts/evaluate-accuracy.py")


def save_lines(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@contextmanager
def replaced_json(path, value):
    original = path.read_bytes()
    PUBLISHER.write_json(path, value)
    try:
        yield
    finally:
        path.write_bytes(original)


class PairedComparisonTests(unittest.TestCase):
    def response(self, choice, probabilities=None):
        return {"answers": {"answer": {"choice": choice, "probabilities": probabilities or
                                      {"A": 0.3, "B": 0.3, "C": 0.4}}}}

    def test_equivalent_labels_do_not_count_as_corrections_or_regressions(self):
        criteria = {"A": "same", "B": "same", "C": "different"}
        requests = [{"questions": {"answer": {"criteria": criteria}}}] * 4
        gold = [{"gold": "A", "subject": subject} for subject in ("one", "one", "two", "two")]
        result = PUBLISHER.paired(
            [self.response(choice) for choice in ("A", "C", "B", "C")],
            [self.response(choice) for choice in ("B", "B", "C", "B")], requests, gold)
        self.assertEqual(result["changed"], 4)
        self.assertEqual(result["semantic_changed"], 3)
        self.assertEqual(result["corrected"], 2)
        self.assertEqual(result["regressed"], 1)
        self.assertEqual(result["gain"], 25.0)
        self.assertEqual(result["subjects-improved"], 1)
        self.assertEqual(result["subjects-tied"], 1)

    def test_probability_differences_are_distinct_from_top_one_changes(self):
        request = {"questions": {"answer": {"criteria": {"A": "a", "B": "b", "C": "c"}}}}
        result = PUBLISHER.paired(
            [self.response("C")], [self.response("C", {"A": 0.2, "B": 0.3, "C": 0.5})],
            [request], [{"gold": "C", "subject": "one"}])
        self.assertEqual(result["changed"], 0)
        self.assertEqual(result["probability_changed"], 1)
        self.assertAlmostEqual(result["max_probability_difference"], 0.1)

    def test_empty_and_misaligned_comparisons_are_rejected(self):
        with self.assertRaises(ValueError):
            PUBLISHER.paired([], [], [], [])
        with self.assertRaises(ValueError):
            PUBLISHER.paired([], [], [], [{"gold": "A", "subject": "one"}])


class CompleteSuiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.suite = Path(cls.temporary.name)
        suite = cls.suite
        requests = [{"state": {"id": f"row-{i}"}, "questions": {"answer": {
            "type": "choice", "criteria": {"A": "answer", "B": "other"}}}} for i in range(14042)]
        gold = [{"id": f"row-{i}", "dataset": "mmlu", "subject": f"subject-{i % 57}", "gold": "A"}
                for i in range(14042)]
        subjects = defaultdict(list)
        for index, row in enumerate(gold):
            subjects[row["subject"]].append((hashlib.sha256(row["id"].encode()).hexdigest(), index))
        indices = sorted(i for group in subjects.values() for _, i in sorted(group)[:10])
        response = {"answers": {"answer": {"type": "choice", "choice": "A",
                                           "probabilities": {"A": 0.75, "B": 0.25}}}}
        reports = {}
        for dataset, selection in (("full", range(14042)), ("control", indices)):
            folder = suite / dataset
            folder.mkdir()
            selected_requests, selected_gold = [requests[i] for i in selection], [gold[i] for i in selection]
            save_lines(folder / "requests.jsonl", selected_requests)
            save_lines(folder / "gold.jsonl", selected_gold)
            save_lines(folder / "responses.jsonl", [response] * len(selection))
            reports[dataset] = EVALUATOR.evaluate(selected_gold, [response] * len(selection), selected_requests)
        (suite / "jet.exe").write_bytes(b"synthetic fixture; never executed")
        (suite / "model.gguf").write_bytes(b"synthetic fixture; never loaded")
        (suite / "source.patch").write_bytes(b"fixture source patch")
        scripts = {}
        for filename in ("benchmark-model.py", "evaluate-accuracy.py", "rerun-paper-4090.py"):
            shutil.copy2(ROOT / "scripts" / filename, suite / filename)
            scripts[filename] = RUNNER.metadata(suite / filename)
        models = {model: {**RUNNER.metadata(suite / "model.gguf"), "model_id": info[1]}
                  for model, info in RUNNER.MODEL_FILES.items()}
        manifest = {"scoring_method": "mean_token_log_probability", "binary": RUNNER.metadata(suite / "jet.exe"),
                    "models": models, "scripts": scripts, "common_flags": RUNNER.COMMON_FLAGS,
                    "git_commit": "0" * 40, "source_patch_sha256": RUNNER.digest(suite / "source.patch"),
                    "control_indices": indices, "cases": RUNNER.matrix(), "data": {
                        f"{dataset}/{name}": RUNNER.metadata(suite / dataset / name)
                        for dataset in ("full", "control") for name in ("requests.jsonl", "gold.jsonl")}}
        results = []
        for case in manifest["cases"]:
            folder = suite / case["id"]
            folder.mkdir()
            run_folder = folder / "run-01"
            run_folder.mkdir()
            dataset, model = case["dataset"], models[case["model"]]
            source_files = {
                "benchmark_input": ("requests.jsonl", manifest["data"][f"{dataset}/requests.jsonl"]),
                "benchmark_gold": ("gold.jsonl", manifest["data"][f"{dataset}/gold.jsonl"]),
                "harness": ("benchmark-model.py", scripts["benchmark-model.py"]),
                "evaluator": ("evaluate-accuracy.py", scripts["evaluate-accuracy.py"]),
            }
            for filename, item in source_files.values():
                (folder / filename).hardlink_to(item["path"])
            (run_folder / "responses.jsonl").hardlink_to(suite / dataset / "responses.jsonl")
            count = 14042 if dataset == "full" else 570
            run = {"success": True, "response_count": count, "request_count": count,
                   "overall": reports[dataset]["overall"], "failures": 0, "returncode": 0,
                   "evaluation_returncode": 0, "wall_seconds": 100.0,
                   "started_at_utc": "2026-09-30T00:00:00+00:00",
                   "responses_sha256": RUNNER.digest(run_folder / "responses.jsonl")}
            PUBLISHER.write_json(folder / "summary.json", {
                "completed_runs": 1, "attempted_runs": 1, "request_count_per_run": count, "runs": [run]})
            PUBLISHER.write_json(run_folder / "report.json", reports[dataset])
            provenance = {"binary": manifest["binary"], "backend": case["backend"], "model_id": model["model_id"],
                          "model_files": [{key: model[key] for key in ("path", "size_bytes", "mtime_ns")}],
                          "batch_requests": 8, "extra_args": manifest["common_flags"],
                          "environment_overrides": case["environment"], "runs_requested": 1, "request_count": count,
                          **{key: item for key, (_, item) in source_files.items()}}
            PUBLISHER.write_json(folder / "provenance.json", provenance)
            results.append(RUNNER.completed_case(folder, case, manifest))
        cls.manifest = manifest
        cls.status = {"state": "complete", "completed_cases": 31, "total_cases": 31,
                      "results": results, "updated_at_utc": "2026-09-30T01:00:00+00:00"}
        PUBLISHER.write_json(suite / "manifest.json", manifest)
        PUBLISHER.write_json(suite / "status.json", cls.status)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @contextmanager
    def fixture_hashes(self):
        with mock.patch.object(RUNNER, "EXPECTED_INPUT", self.manifest["data"]["full/requests.jsonl"]["sha256"]), \
                mock.patch.object(RUNNER, "EXPECTED_GOLD", self.manifest["data"]["full/gold.jsonl"]["sha256"]):
            yield

    def validate(self):
        with self.fixture_hashes():
            return PUBLISHER.validated_suite(self.suite, RUNNER)

    def test_complete_synthetic_suite_is_recomputed_and_accepted(self):
        _, _, result_map, responses, summaries, _ = self.validate()
        self.assertEqual(len(result_map), 31)
        self.assertEqual(len(responses["full-4b-lora-cuda"]), 14042)
        self.assertEqual(summaries["full-4b-lora-cuda"]["overall"]["correct"], 14042)

    def test_duplicate_completed_ids_are_rejected(self):
        status = copy.deepcopy(self.status)
        status["results"][-1] = status["results"][0]
        with replaced_json(self.suite / "status.json", status), self.assertRaisesRegex(ValueError, "result IDs"):
            self.validate()

    def test_incomplete_suite_is_rejected(self):
        with replaced_json(self.suite / "status.json", {**self.status, "state": "running"}), \
                self.assertRaisesRegex(ValueError, "All 31"):
            self.validate()

    def test_changed_frozen_binary_is_rejected(self):
        path = self.suite / "jet.exe"
        original = path.read_bytes()
        try:
            path.write_bytes(b"changed binary")
            with self.assertRaisesRegex(ValueError, "Frozen artifact changed"):
                self.validate()
        finally:
            path.write_bytes(original)

    def test_forged_matching_summary_and_report_are_recomputed(self):
        folder = self.suite / self.manifest["cases"][0]["id"]
        summary, report = PUBLISHER.read(folder / "summary.json"), PUBLISHER.read(folder / "run-01/report.json")
        summary["runs"][0]["overall"]["correct"] = 0
        report["overall"]["correct"] = 0
        status = copy.deepcopy(self.status)
        status["results"][0]["overall"]["correct"] = 0
        with replaced_json(folder / "summary.json", summary), \
                replaced_json(folder / "run-01/report.json", report), \
                replaced_json(self.suite / "status.json", status), \
                self.assertRaisesRegex(ValueError, "recomputed response metrics"):
            self.validate()

    def test_pending_and_complete_publication_emit_all_metrics_without_a770_points(self):
        with tempfile.TemporaryDirectory() as temporary:
            figures = Path(temporary)
            shutil.copy2(ROOT / "paper/figures/generate_mmlu_chart.py", figures / "generate_mmlu_chart.py")
            original_load_module = PUBLISHER.load_module
            def load_module(name, path):
                return RUNNER if name == "paper_runner" else original_load_module(name, path)
            for arguments, expected_state in ((["--pending"], "pending"), (["--suite", str(self.suite)], "complete")):
                with mock.patch.object(PUBLISHER, "FIGURES", figures), \
                        mock.patch.object(PUBLISHER, "load_module", side_effect=load_module), \
                        mock.patch.object(sys, "argv", ["update_results.py", *arguments]), self.fixture_hashes():
                    PUBLISHER.main()
                macros = (figures / "rtx4090-results.tex").read_text(encoding="utf-8")
                for case in RUNNER.matrix():
                    for metric in PUBLISHER.METRICS:
                        self.assertIn(f"jetresult@{case['id']}@{metric}", macros)
                rows = PUBLISHER.read(figures / "mmlu-results.json")
                self.assertEqual(len(rows), 14)
                self.assertTrue(all(row.get("deployment") != "a770" for row in rows))
                self.assertEqual(PUBLISHER.read(figures / "rtx4090-evidence.json")["state"], expected_state)
                if expected_state == "pending":
                    self.assertTrue(all(row["accuracy"] is None for row in rows if "case" in row))
                else:
                    self.assertNotIn("{pending}", macros)


if __name__ == "__main__":
    unittest.main()
