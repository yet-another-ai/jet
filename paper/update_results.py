"""Publish only verified mean-token RTX 4090 results into the paper's figures and macros."""

from __future__ import annotations

import argparse
from collections import defaultdict
import datetime as dt
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "paper/figures"
MODELS = {"08b": "Qwen3.5-0.8B", "2b": "Qwen3.5-2B", "4b": "Qwen3.5-4B",
          "9b": "Qwen3.5-9B", "27b": "Qwen3.5-27B", "35b": "Qwen3.6-35B-A3B"}
METRICS = ("correct", "accuracy", "wall", "rate", "nll", "brier")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load validation module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validated_suite(suite, runner):
    manifest, status = read(suite / "manifest.json"), read(suite / "status.json")
    expected_cases = runner.matrix()
    if (len(expected_cases) != 31 or manifest["cases"] != expected_cases
            or status.get("state") != "complete" or status.get("completed_cases") != 31
            or status.get("total_cases") != 31 or len(status.get("results", [])) != 31):
        raise ValueError("All 31 cases in the exact rerun matrix must complete before publishing")
    result_map = {result["id"]: result for result in status["results"]}
    if len(result_map) != 31 or set(result_map) != {case["id"] for case in expected_cases}:
        raise ValueError("Suite result IDs do not match the complete matrix")
    if (manifest["scoring_method"] != "mean_token_log_probability"
            or manifest["common_flags"] != runner.COMMON_FLAGS):
        raise ValueError("Unexpected scoring method or common execution settings")
    for item in [manifest["binary"], *manifest["data"].values(), *manifest["scripts"].values()]:
        if runner.digest(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"Frozen artifact changed: {item['path']}")
    if runner.digest(suite / "source.patch") != manifest["source_patch_sha256"]:
        raise ValueError("Frozen source patch changed")
    if runner.digest(Path(runner.__file__)) != manifest["scripts"]["rerun-paper-4090.py"]["sha256"]:
        raise ValueError("The validator differs from the frozen runner")
    for item in manifest["models"].values():
        runner.verify_model(item)
    for filename, expected in (("requests.jsonl", runner.EXPECTED_INPUT),
                               ("gold.jsonl", runner.EXPECTED_GOLD)):
        if runner.digest(suite / "full" / filename) != expected:
            raise ValueError(f"Unexpected full-MMLU {filename} hash")
    datasets = {
        name: (read_lines(suite / name / "requests.jsonl"), read_lines(suite / name / "gold.jsonl"))
        for name in ("full", "control")
    }
    for name, count in (("full", 14042), ("control", 570)):
        if any(len(rows) != count for rows in datasets[name]):
            raise ValueError(f"Unexpected {name} request/gold row count")
    indices = manifest["control_indices"]
    if (len(indices) != 570 or len(set(indices)) != 570
            or any(not isinstance(i, int) or not 0 <= i < 14042 for i in indices)):
        raise ValueError("Invalid control projection indices")
    subjects = defaultdict(list)
    for index, row in enumerate(datasets["full"][1]):
        subjects[row["subject"]].append((hashlib.sha256(row["id"].encode()).hexdigest(), index))
    expected_indices = sorted(index for group in subjects.values() for _, index in sorted(group)[:10])
    if len(subjects) != 57 or indices != expected_indices:
        raise ValueError("Control projection differs from the declared per-subject hash selection")
    for full, control in zip(datasets["full"], datasets["control"], strict=True):
        if [full[i] for i in indices] != control:
            raise ValueError("Control requests/gold differ from the declared full-set projection")
    evaluator = load_module("paper_evaluator", Path(manifest["scripts"]["evaluate-accuracy.py"]["path"]))
    responses, summaries = {}, {}
    for case in expected_cases:
        folder = Path(result_map[case["id"]]["directory"])
        verified = runner.completed_case(folder, case, manifest)
        if verified is None or verified != result_map[case["id"]]:
            raise ValueError(f"Case summary does not match the verified completed run: {case['id']}")
        summaries[case["id"]] = read(folder / "summary.json")["runs"][0]
        responses[case["id"]] = read_lines(folder / "run-01/responses.jsonl")
        requests, gold = datasets[case["dataset"]]
        recomputed = evaluator.evaluate(gold, responses[case["id"]], requests)
        if (recomputed["failures"] or recomputed != read(folder / "run-01/report.json")):
            raise ValueError(f"Case report differs from recomputed response metrics: {case['id']}")
    return manifest, status, result_map, responses, summaries, datasets


def rows():
    result = []
    for model, label in MODELS.items():
        for backend in ("vulkan", "cuda"):
            result.append({"case": f"full-{model}-{backend}", "group": f"JET / {label}",
                           "label": f"RTX 4090, full MMLU, {backend.upper() if backend == 'cuda' else 'Vulkan'}",
                           "figure_label": "RTX 4090", "backend": "CUDA" if backend == "cuda" else "Vulkan",
                           "deployment": "4090", "scope": "mmlu", "requests": 14042,
                           "accuracy": None, "correct": None, "wall_s": None, "req_s": None,
                           "scoring_method": "mean_token_log_probability", "status": "pending",
                           "show_in_figure": True})
        if model == "4b":
            result.append({**result[-1], "case": "full-4b-lora-cuda", "fine_tuned": True,
                           "label": "RTX 4090, LoRA, full MMLU, CUDA", "figure_label": "RTX 4090, LoRA"})
    result.append({"group": "Jev 1.13", "label": "Native decision interface (API)",
                   "accuracy": 89.06, "req_s": 2.86, "wall_s": None, "requests": 14042,
                   "scope": "api", "show_in_figure": True, "figure_label": "Hosted API"})
    return result


def paired(left, right, requests, gold):
    if not gold:
        raise ValueError("A paired comparison requires at least one example")
    corrected = regressed = changed = 0
    semantic_changed = probability_changed = 0
    by_subject = {}
    max_probability_difference = 0.0
    for a, b, request, expected in zip(left, right, requests, gold, strict=True):
        aa, bb = a["answers"]["answer"], b["answers"]["answer"]
        choices = request["questions"]["answer"]["criteria"]
        gold_text = choices[expected["gold"]]
        correct_a = choices[aa["choice"]] == gold_text
        correct_b = choices[bb["choice"]] == gold_text
        corrected += not correct_a and correct_b
        regressed += correct_a and not correct_b
        changed += aa["choice"] != bb["choice"]
        semantic_changed += choices[aa["choice"]] != choices[bb["choice"]]
        by_subject[expected["subject"]] = by_subject.get(expected["subject"], 0) + int(correct_b) - int(correct_a)
        if set(aa["probabilities"]) != set(choices) or set(bb["probabilities"]) != set(choices):
            raise ValueError("Paired comparison candidate probability keys differ")
        if any(not math.isfinite(value) or not 0 <= value <= 1
               for answer in (aa, bb) for value in answer["probabilities"].values()):
            raise ValueError("Paired comparison contains invalid probabilities")
        difference = max(abs(aa["probabilities"][key] - bb["probabilities"][key]) for key in choices)
        probability_changed += difference > 0
        max_probability_difference = max(max_probability_difference, difference)
    return {"changed": changed, "corrected": corrected, "regressed": regressed,
            "gain": 100 * (corrected - regressed) / len(gold),
            "subjects-improved": sum(value > 0 for value in by_subject.values()),
            "subjects-declined": sum(value < 0 for value in by_subject.values()),
            "subjects-tied": sum(value == 0 for value in by_subject.values()),
            "semantic_changed": semantic_changed, "probability_changed": probability_changed,
            "max_probability_difference": max_probability_difference}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--suite", type=Path, help="A complete 31-case rerun directory")
    group.add_argument("--pending", action="store_true", help="Remove superseded numbers and mark reruns pending")
    args = parser.parse_args()
    runner = load_module("paper_runner", ROOT / "scripts/rerun-paper-4090.py")
    chart = rows()
    values = {(case["id"], metric): "pending" for case in runner.matrix() for metric in METRICS}
    comparison_metrics = ("changed", "semantic_changed", "probability_changed", "max_probability_difference")
    derived = {"paired-lora": ("gain", "corrected", "regressed", "subjects-improved", "subjects-declined", "subjects-tied"),
               "vulkan-27b-vs-35b": ("correct-difference", "accuracy-difference", "rate-ratio"),
               "full-35b-backends": comparison_metrics, "best": ("model", "accuracy", "gap-to-jev"),
               "suite": ("date", "binary", "source"),
               **{f"control-{model}-backends": comparison_metrics for model in ("08b", "2b", "35b")}}
    values.update({(case, key): "pending" for case, keys in derived.items() for key in keys})
    evidence = {"state": "pending", "scoring_method": "mean_token_log_probability",
                "a770": "Separate full-MMLU campaign completed with the current method; see evidence.md. Cross-host values do not isolate hardware effects."}
    if args.suite:
        suite = args.suite.resolve()
        manifest, status, result_map, responses, summaries, datasets = validated_suite(suite, runner)
        for case, result in summaries.items():
            metric = result["overall"]
            values.update({(case, key): value for key, value in {
                "correct": f"{metric['correct']:,}", "accuracy": f"{100 * metric['accuracy']:.2f}",
                "wall": f"{result['wall_seconds']:,.2f}", "rate": f"{metric['examples'] / result['wall_seconds']:.2f}",
                "nll": f"{metric['mean_gold_nll']:.4f}", "brier": f"{metric['mean_brier']:.4f}"}.items()})
        for row in chart:
            if "case" not in row:
                continue
            result = summaries[row["case"]]
            metric = result["overall"]
            row.update({"accuracy": 100 * metric["accuracy"], "correct": metric["correct"],
                        "wall_s": result["wall_seconds"], "req_s": 14042 / result["wall_seconds"],
                        "mean_gold_nll": metric["mean_gold_nll"], "mean_brier": metric["mean_brier"],
                        "status": "complete", "source": str(Path(result_map[row["case"]]["directory"]) / "summary.json"),
                        "binary_sha256": manifest["binary"]["sha256"]})
        full_requests, full_gold = datasets["full"]
        control_requests, control_gold = datasets["control"]
        comparisons = {"paired-lora": paired(responses["full-4b-cuda"], responses["full-4b-lora-cuda"], full_requests, full_gold),
                       "full-35b-backends": paired(responses["full-35b-cuda"], responses["full-35b-vulkan"], full_requests, full_gold)}
        for model in ("08b", "2b", "35b"):
            comparisons[f"control-{model}-backends"] = paired(responses[f"control-{model}-cuda"], responses[f"control-{model}-vk-default"], control_requests, control_gold)
        for case, comparison in comparisons.items():
            for key in derived[case]:
                values[case, key] = (f"{comparison[key]:.2f}" if key == "gain" else
                                     f"{comparison[key]:.6g}" if key == "max_probability_difference" else
                                     f"{comparison[key]:,}")
        chart_by_case = {row["case"]: row for row in chart if "case" in row}
        a, b = chart_by_case["full-27b-vulkan"], chart_by_case["full-35b-vulkan"]
        values["vulkan-27b-vs-35b", "correct-difference"] = f"{a['correct'] - b['correct']:,}"
        values["vulkan-27b-vs-35b", "accuracy-difference"] = f"{a['accuracy'] - b['accuracy']:.2f}"
        values["vulkan-27b-vs-35b", "rate-ratio"] = f"{b['req_s'] / a['req_s']:.2f}"
        best = max(chart_by_case.values(), key=lambda row: row["accuracy"])
        values["best", "model"] = best["group"].removeprefix("JET / ") + (" LoRA" if best.get("fine_tuned") else "") + f" ({best['backend']})"
        values["best", "accuracy"] = f"{best['accuracy']:.2f}"
        values["best", "gap-to-jev"] = f"{89.06 - best['accuracy']:.2f}"
        start_date = min(run["started_at_utc"][:10] for run in summaries.values())
        end_date = status["updated_at_utc"][:10]
        values["suite", "date"] = start_date if start_date == end_date else f"{start_date}--{end_date}"
        values["suite", "binary"] = manifest["binary"]["sha256"][:12]
        values["suite", "source"] = manifest["git_commit"][:12]
        # Each Vulkan option is compared with the same model's default run, without
        # inferring that one option alone explains differences across GPUs.
        for case in manifest["cases"]:
            if case["dataset"] == "control" and case["environment"]:
                comparisons[case["id"]] = paired(responses[f"control-{case['model']}-vk-default"], responses[case["id"]], control_requests, control_gold)
        for backend in ("cuda", "vk-default"):
            comparisons[f"repeat-08b-{backend}"] = paired(responses[f"control-08b-{backend}"], responses[f"control-08b-{backend}-repeat"], control_requests, control_gold)
        for model in ("08b", "2b", "35b"):
            for backend, variant in (("cuda", "cuda"), ("vulkan", "vk-default")):
                projected = [responses[f"full-{model}-{backend}"][i] for i in manifest["control_indices"]]
                comparisons[f"projection-{model}-{backend}"] = paired(projected, responses[f"control-{model}-{variant}"], control_requests, control_gold)
        evidence.update({"state": "complete", "suite": str(suite), "binary_sha256": manifest["binary"]["sha256"],
                         "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                         "cases": status["results"], "paired_comparisons": comparisons})
    evidence["comparison_definitions"] = {
        "changed": "Number of differing top-1 label keys, including semantically equivalent labels.",
        "semantic_changed": "Number of differing top-1 answer texts.",
        "probability_changed": "Number of rows with any nonzero candidate probability difference.",
        "max_probability_difference": "Maximum absolute difference across candidate label probabilities.",
        "corrected/regressed": "Correctness compares answer text, accepting all equivalent gold labels.",
    }
    # All validations and calculations complete before any paper file is changed.
    write_json(FIGURES / "mmlu-results.json", chart)
    write_json(FIGURES / "rtx4090-evidence.json", evidence)
    lines = ["% Generated by paper/update_results.py; do not edit numeric values by hand.",
             r"\newcommand{\jetresult}[2]{\csname jetresult@#1@#2\endcsname}"]
    for (case, metric), value in sorted(values.items()):
        lines.append(r"\expandafter\def\csname jetresult@" + case + "@" + metric + r"\endcsname{" + value + "}")
    (FIGURES / "rtx4090-results.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    subprocess.run([sys.executable, str(FIGURES / "generate_mmlu_chart.py")], check=True)


if __name__ == "__main__":
    main()
