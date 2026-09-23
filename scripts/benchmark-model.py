#!/usr/bin/env python3
"""Benchmark a Jet model against the fixed accuracy workload, preserving artifacts."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "target/release/jet")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New directory; existing directories are never overwritten")
    parser.add_argument("--input", type=Path,
                        default=ROOT / "tests/accuracy/generated/requests.jsonl")
    parser.add_argument("--gold", type=Path,
                        default=ROOT / "tests/accuracy/generated/gold.jsonl")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--batch-requests", type=int, default=8)
    parser.add_argument("--backend", choices=("cpu", "vulkan"), default="vulkan")
    parser.add_argument("--limit", type=int, help="Use only the first N requests and matching labels")
    parser.add_argument("--model-provenance", type=Path,
                        help="Optional JSON download metadata, copied without hashing large weights")
    parser.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[],
                        help="Additional Jet judge flags; must be the final script option")
    args = parser.parse_args()
    for name in ("runs", "batch_requests", "limit"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    managed = {"--model-path", "--model-id", "--input", "--output", "--timings",
               "--batch-requests", "--backend"}
    for value in args.extra_args:
        if value.split("=", 1)[0] in managed:
            parser.error(f"{value} is managed by the benchmark; use its script option")
    return args


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_metadata(path: Path, *, hash_file: bool = False) -> dict[str, Any]:
    stat = path.stat()
    result = {"path": str(path.resolve()), "size_bytes": stat.st_size,
              "mtime_ns": stat.st_mtime_ns}
    if hash_file:
        result["sha256"] = sha256(path)
    return result


def model_files(path: Path) -> list[Path]:
    # A split GGUF is loaded by passing its first shard to llama.cpp.
    match = re.fullmatch(r"(.+)-(\d{5})-of-(\d{5})\.gguf", path.name)
    if not match:
        return [path]
    stem, index, total = match.groups()
    if int(index) != 1:
        raise ValueError("--model-path must name the first GGUF shard")
    return [path.with_name(f"{stem}-{part:05d}-of-{int(total):05d}.gguf")
            for part in range(1, int(total) + 1)]


def read_json_lines(path: Path) -> list[str]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for line in lines:
        json.loads(line)
    return lines


def git_output(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"git {' '.join(args)}: {result.stderr.strip()}")
    return result.stdout


def make_summary(results: list[dict[str, Any]], request_count: int) -> dict[str, Any]:
    successful = [run for run in results if run.get("success")]
    summary: dict[str, Any] = {
        "request_count_per_run": request_count,
        "completed_runs": len(successful),
        "attempted_runs": len(results),
        "notes": [
            "Each run starts a fresh process; no explicit warmup or cache eviction is performed.",
            "Wall time includes startup, model load, JSONL processing, and shutdown.",
            "Seconds per request is amortized whole-run time, not single-request latency or P95.",
            "Stage durations are cumulative engine wall times, not GPU kernel timings.",
            "Model identity uses file metadata and optional supplied provenance; weights are not rehashed.",
        ],
        "runs": results,
    }
    if successful:
        mean_wall = statistics.mean(run["wall_seconds"] for run in successful)
        summary.update({
            "mean_wall_seconds": mean_wall,
            "mean_amortized_seconds_per_request": mean_wall / request_count,
            "requests_per_second": request_count / mean_wall,
            "mean_stage_seconds_per_request": {
                key.removesuffix("_ms"): statistics.mean(run["stages"][key] for run in successful)
                / 1000 / request_count
                for key in successful[0]["stages"] if key.endswith("_ms")
            },
        })
    return summary


def run_once(args: argparse.Namespace, output: Path, index: int,
             request_count: int) -> dict[str, Any]:
    run_dir = output / f"run-{index:02d}"
    run_dir.mkdir()
    command = [str(args.binary), "judge", "--backend", args.backend,
               "--model-path", str(args.model_path), "--model-id", args.model_id,
               "--batch-requests", str(args.batch_requests),
               "--input", str(output / "requests.jsonl"),
               "--output", str(run_dir / "responses.jsonl"),
               "--timings", str(run_dir / "stages.json")]
    if not any(value.split("=", 1)[0] == "--thinking" for value in args.extra_args):
        command += ["--thinking", "disabled"]
    command += args.extra_args
    result: dict[str, Any] = {
        "run": index, "directory": str(run_dir), "command": command,
        "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "request_count": request_count, "success": False,
    }
    write_json(run_dir / "timing.json", result)
    print(json.dumps({"event": "start", "run": index, "command": command}), flush=True)
    start = time.perf_counter()
    try:
        with (run_dir / "engine.log").open("w", encoding="utf-8") as log:
            process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        result["wall_seconds"] = time.perf_counter() - start
        result["returncode"] = process.returncode
        result["amortized_seconds_per_request"] = result["wall_seconds"] / request_count
        stages_path = run_dir / "stages.json"
        if stages_path.exists():
            result["stages"] = json.loads(stages_path.read_text(encoding="utf-8"))
            result["stage_seconds_per_request"] = {
                key.removesuffix("_ms"): value / 1000 / request_count
                for key, value in result["stages"].items() if key.endswith("_ms")
            }
        responses_path = run_dir / "responses.jsonl"
        if responses_path.exists():
            result["responses_sha256"] = sha256(responses_path)
            result["response_count"] = sum(bool(line.strip()) for line in
                                            responses_path.read_text(encoding="utf-8").splitlines())
            evaluation_command = [sys.executable, str(output / "evaluate-accuracy.py"),
                                  "--gold", str(output / "gold.jsonl"),
                                  "--requests", str(output / "requests.jsonl"),
                                  "--responses", str(responses_path),
                                  "--report", str(run_dir / "report.json")]
            result["evaluation_command"] = evaluation_command
            with (run_dir / "evaluation.log").open("w", encoding="utf-8") as log:
                evaluated = subprocess.run(evaluation_command, cwd=ROOT, stdout=log,
                                           stderr=subprocess.STDOUT)
            result["evaluation_returncode"] = evaluated.returncode
            if (run_dir / "report.json").exists():
                report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
                result["overall"] = report["overall"]
                result["datasets"] = report["datasets"]
                result["failures"] = len(report["failures"])
        if process.returncode:
            raise RuntimeError(f"Jet exited with status {process.returncode}; see {run_dir / 'engine.log'}")
        if "stages" not in result:
            raise RuntimeError("Jet did not produce stage timings")
        if result.get("evaluation_returncode") != 0:
            raise RuntimeError(f"Accuracy evaluation failed; see {run_dir / 'evaluation.log'}")
        result["success"] = True
    except (OSError, ValueError, RuntimeError, KeyboardInterrupt) as error:
        result.setdefault("wall_seconds", time.perf_counter() - start)
        result["error"] = str(error) or type(error).__name__
    write_json(run_dir / "timing.json", result)
    print(json.dumps({"event": "finish", "run": index, "success": result["success"],
                      "wall_seconds": result["wall_seconds"],
                      "amortized_seconds_per_request": result.get("amortized_seconds_per_request"),
                      "error": result.get("error")}), flush=True)
    return result


def main() -> int:
    args = parse_args()
    for name in ("binary", "model_path", "input", "gold", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if not os.access(args.binary, os.X_OK):
        raise ValueError(f"Binary is missing or not executable: {args.binary}")
    model_metadata = [file_metadata(path) for path in model_files(args.model_path)]
    requests, gold = read_json_lines(args.input), read_json_lines(args.gold)
    if not requests or len(requests) != len(gold):
        raise ValueError(f"Input/gold must be nonempty with equal lengths: {len(requests)}/{len(gold)}")
    if args.limit:
        requests, gold = requests[:args.limit], gold[:args.limit]
    supplied_provenance = (json.loads(args.model_provenance.read_text(encoding="utf-8"))
                           if args.model_provenance else None)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    for filename, lines in (("requests.jsonl", requests), ("gold.jsonl", gold)):
        (output / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")
    shutil.copy2(__file__, output / "benchmark-model.py")
    shutil.copy2(ROOT / "scripts/evaluate-accuracy.py", output / "evaluate-accuracy.py")
    git_status = git_output("status", "--short")
    (output / "working-tree.patch").write_text(git_output("diff", "--binary", "HEAD"), encoding="utf-8")
    provenance = {
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "repo_root": str(ROOT), "git_commit": git_output("rev-parse", "HEAD").strip(),
        "git_status": git_status, "submodules": git_output("submodule", "status"),
        "binary": file_metadata(args.binary, hash_file=True),
        "model_id": args.model_id, "model_files": model_metadata,
        "model_provenance_supplied_unverified": supplied_provenance,
        "original_input": file_metadata(args.input, hash_file=True),
        "original_gold": file_metadata(args.gold, hash_file=True),
        "benchmark_input": file_metadata(output / "requests.jsonl", hash_file=True),
        "benchmark_gold": file_metadata(output / "gold.jsonl", hash_file=True),
        "harness": file_metadata(output / "benchmark-model.py", hash_file=True),
        "evaluator": file_metadata(output / "evaluate-accuracy.py", hash_file=True),
        "request_count": len(requests), "runs_requested": args.runs,
        "backend": args.backend, "batch_requests": args.batch_requests,
        "extra_args": args.extra_args, "platform": platform.platform(),
        "environment_overrides": {
            key: value for key, value in os.environ.items()
            if key.startswith(("JET_", "GGML_", "LLAMA_", "VK_"))
            and not any(secret in key.upper() for secret in ("TOKEN", "SECRET", "PASSWORD", "KEY"))
        },
    }
    write_json(output / "provenance.json", provenance)
    results: list[dict[str, Any]] = []
    for index in range(1, args.runs + 1):
        result = run_once(args, output, index, len(requests))
        results.append(result)
        write_json(output / "summary.json", make_summary(results, len(requests)))
        if not result["success"]:
            print(f"benchmark-model: {result['error']}", file=sys.stderr)
            return 1
    print(f"Benchmark artifacts: {output}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"benchmark-model: {error}", file=sys.stderr)
        sys.exit(1)
