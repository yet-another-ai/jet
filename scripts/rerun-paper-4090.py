#!/usr/bin/env python3
"""Run the paper's RTX 4090 matrix sequentially with frozen inputs and resumable artifacts."""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
MODEL_FILES = {
    "08b": ("models/Qwen3.5-0.8B-Q8_0.gguf", "qwen/qwen3.5-0.8b-q8_0"),
    "2b": ("models/Qwen3.5-2B-Q8_0.gguf", "qwen/qwen3.5-2b-q8_0"),
    "4b": ("models/Qwen3.5-4B-Q8_0.gguf", "qwen/qwen3.5-4b-q8_0"),
    "4b-lora": ("training/exports/qwen35-4b-lora-text-Q8_0.gguf", "jet/qwen35-4b-lora-q8_0"),
    "9b": ("models/Qwen3.5-9B-Q8_0.gguf", "qwen/qwen3.5-9b-q8_0"),
    "27b": ("models/Qwen3.5-27B-Q4_K_M.gguf", "qwen/qwen3.5-27b-q4_k_m"),
    "35b": ("models/Qwen3.6-35B-A3B-Q4_K_M.gguf", "qwen/qwen3.6-35b-a3b-q4_k_m"),
}
EXPECTED_INPUT = "5393a9f78039744f999e7c403a0bac0fd7b26931f513e3f9e8dc8ef34d69cf7c"
EXPECTED_GOLD = "6f0e61960c7459f6738805a4c2e2f89b9287171d798760ea140f8b3bb57cf209"
COMMON_FLAGS = ["--max-sequences", "2", "--micro-batch", "256", "--max-output-rows",
                "256", "--threads", "8", "--no-mmap", "--thinking", "disabled"]
VK_CASES = {
    "vk-default": {},
    "vk-no-cm2": {"GGML_VK_DISABLE_COOPMAT2": "1"},
    "vk-no-cm": {"GGML_VK_DISABLE_COOPMAT2": "1", "GGML_VK_DISABLE_COOPMAT": "1"},
    "vk-no-cm-f32": {"GGML_VK_DISABLE_COOPMAT2": "1", "GGML_VK_DISABLE_COOPMAT": "1",
                     "GGML_VK_DISABLE_F16": "1"},
    "vk-no-cm-no-intdot": {"GGML_VK_DISABLE_COOPMAT2": "1", "GGML_VK_DISABLE_COOPMAT": "1",
                           "GGML_VK_DISABLE_INTEGER_DOT_PRODUCT": "1"},
    "vk-no-cm-f32-no-intdot": {"GGML_VK_DISABLE_COOPMAT2": "1", "GGML_VK_DISABLE_COOPMAT": "1",
                               "GGML_VK_DISABLE_F16": "1", "GGML_VK_DISABLE_INTEGER_DOT_PRODUCT": "1"},
}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def metadata(path):
    stat = path.stat()
    item = {"path": str(path.resolve()), "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "sha256": digest(path)}
    verify_model(item)
    return item


def verify_model(item, *, hash_weights=False):
    path = Path(item["path"])
    stat = path.stat()
    if stat.st_size != item["size_bytes"] or stat.st_mtime_ns != item["mtime_ns"]:
        raise ValueError(f"Model or artifact changed since suite preparation: {path}")
    if hash_weights and digest(path) != item["sha256"]:
        raise ValueError(f"Model checksum changed since suite preparation: {path}")
    if hash_weights:
        verify_model(item)


def matrix():
    cases = []
    for backend in ("cuda", "vulkan"):
        for model in MODEL_FILES:
            if model == "4b-lora" and backend != "cuda":
                continue
            cases.append({"id": f"full-{model}-{backend}", "model": model,
                          "backend": backend, "dataset": "full", "environment": {}})
    for model in ("08b", "2b", "35b"):
        variants = ["cuda", "vk-default", "vk-no-cm2", "vk-no-cm"]
        if model == "08b":
            variants += ["vk-no-cm-f32", "vk-no-cm-no-intdot", "vk-no-cm-f32-no-intdot",
                         "cpu", "cuda-repeat", "vk-default-repeat"]
        for variant in variants:
            base_variant = variant.removesuffix("-repeat")
            backend = "vulkan" if variant.startswith("vk-") else base_variant
            cases.append({"id": f"control-{model}-{variant}", "model": model,
                          "backend": backend, "dataset": "control",
                          "environment": VK_CASES.get(base_variant, {})})
    return cases


def contain_windows_processes():
    """Keep descendants in a kill-on-close job so interruption cannot orphan a GPU run."""
    if os.name != "nt":
        return
    import ctypes
    from ctypes import wintypes

    class BasicLimits(ctypes.Structure):
        _fields_ = [("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64),
                    ("flags", wintypes.DWORD), ("min_working_set", ctypes.c_size_t),
                    ("max_working_set", ctypes.c_size_t), ("active_processes", wintypes.DWORD),
                    ("affinity", ctypes.c_size_t), ("priority", wintypes.DWORD),
                    ("scheduling", wintypes.DWORD)]

    class IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in
                    ("read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]

    class ExtendedLimits(ctypes.Structure):
        _fields_ = [("basic", BasicLimits), ("io", IoCounters),
                    ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t),
                    ("peak_process_memory", ctypes.c_size_t), ("peak_job_memory", ctypes.c_size_t)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel.SetInformationJobObject.restype = wintypes.BOOL
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    job = kernel.CreateJobObjectW(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    limits = ExtendedLimits()
    limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not kernel.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
        error = ctypes.get_last_error()
        kernel.CloseHandle(job)
        raise ctypes.WinError(error)
    if not kernel.AssignProcessToJobObject(job, kernel.GetCurrentProcess()):
        error = ctypes.get_last_error()
        kernel.CloseHandle(job)
        raise ctypes.WinError(error)
    # Deliberately keep the sole, non-inherited handle open until process exit.


@contextmanager
def suite_lock(output):
    """OS-owned lock is released even if this process is terminated."""
    with (output / "suite.lock").open("a+b") as lock:
        lock.seek(0)
        if os.name == "nt":
            import msvcrt
            if lock.read(1) == b"":
                lock.write(b"0")
                lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == "nt":
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def prepare(args):
    output = args.output_dir
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = read(manifest_path)
        if manifest["cases"] != matrix():
            raise ValueError("The saved matrix differs from this runner; use a new output directory")
        if digest(Path(__file__)) != manifest["scripts"]["rerun-paper-4090.py"]["sha256"]:
            raise ValueError("Runner changed since suite preparation; restore it or use a new output directory")
        for item in [manifest["binary"], *manifest["data"].values(), *manifest["scripts"].values()]:
            if digest(Path(item["path"])) != item["sha256"]:
                raise ValueError(f"Frozen artifact changed: {item['path']}")
        if digest(output / "source.patch") != manifest["source_patch_sha256"]:
            raise ValueError("Frozen source patch changed")
        for model, item in manifest["models"].items():
            print(json.dumps({"event": "verify_model", "model": model}), flush=True)
            verify_model(item, hash_weights=True)
        return manifest
    if any(p.name != "suite.lock" for p in output.iterdir()):
        raise ValueError("Incomplete suite preparation: use a fresh output directory")
    if not args.binary:
        raise ValueError("--binary is required when creating a suite")
    source = args.input_dir
    for filename, expected in (("requests.jsonl", EXPECTED_INPUT), ("gold.jsonl", EXPECTED_GOLD)):
        if digest(source / filename) != expected:
            raise ValueError(f"Unexpected full-MMLU {filename} hash")
    requests = (source / "requests.jsonl").read_text(encoding="utf-8").splitlines()
    gold = (source / "gold.jsonl").read_text(encoding="utf-8").splitlines()
    if len(requests) != 14042 or len(gold) != 14042:
        raise ValueError("Expected exactly 14,042 requests and gold records")
    subjects = defaultdict(list)
    for index, line in enumerate(gold):
        row = json.loads(line)
        subjects[row["subject"]].append((hashlib.sha256(row["id"].encode()).hexdigest(), index))
    indices = sorted(index for group in subjects.values() for _, index in sorted(group)[:10])
    if len(subjects) != 57 or len(indices) != 570:
        raise ValueError("Control selection must contain 10 rows from each of 57 subjects")
    for dataset, selection in (("full", range(len(gold))), ("control", indices)):
        folder = output / dataset
        folder.mkdir()
        for filename, rows in (("requests.jsonl", requests), ("gold.jsonl", gold)):
            (folder / filename).write_text("\n".join(rows[i] for i in selection) + "\n", encoding="utf-8")
    binary = output / args.binary.name
    shutil.copy2(args.binary, binary)
    for filename in ("benchmark-model.py", "evaluate-accuracy.py", "rerun-paper-4090.py"):
        shutil.copy2(ROOT / "scripts" / filename, output / filename)
    patch = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=ROOT)
    (output / "source.patch").write_bytes(patch)
    model_metadata = {}
    for model, (filename, _) in MODEL_FILES.items():
        print(json.dumps({"event": "hash_model", "model": model}), flush=True)
        model_metadata[model] = metadata(ROOT / filename)
        model_metadata[model]["model_id"] = MODEL_FILES[model][1]
    manifest = {
        "created_at_utc": now(), "scoring_method": "mean_token_log_probability",
        "binary": metadata(binary), "models": model_metadata,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_patch_sha256": digest(output / "source.patch"),
        "source_files_sha256": {str(p.relative_to(ROOT)): digest(p)
                                for p in sorted((ROOT / "crates").rglob("*.rs"))},
        "scripts": {filename: metadata(output / filename)
                    for filename in ("benchmark-model.py", "evaluate-accuracy.py", "rerun-paper-4090.py")},
        "data": {f"{dataset}/{filename}": metadata(output / dataset / filename)
                 for dataset in ("full", "control") for filename in ("requests.jsonl", "gold.jsonl")},
        "control_selection": "10 smallest SHA256(gold.id) per subject, original order; no response-based selection",
        "control_indices": indices, "cases": matrix(), "common_flags": COMMON_FLAGS,
        "nvidia_smi": subprocess.check_output(["nvidia-smi"], text=True),
        "notes": ["GPU cases run sequentially; one fresh process per case.",
                  "Wall time includes model loading and shutdown; no explicit warmup or cache eviction.",
                  "Existing LoRA weights are reevaluated, not retrained.",
                  "Arc A770 experiments require separate reruns on that machine."],
    }
    save(manifest_path, manifest)
    return manifest


def completed_case(folder, case, manifest):
    summary_path = folder / "summary.json"
    if not summary_path.exists():
        return None
    try:
        summary = read(summary_path)
    except json.JSONDecodeError:
        # The harness writes this file last, but not atomically. An interrupted
        # write is an unfinished attempt, not a reason to prevent a fresh retry.
        return None
    expected_count = 14042 if case["dataset"] == "full" else 570
    if summary.get("completed_runs") != 1 or summary.get("attempted_runs") != 1:
        return None
    if not isinstance(summary.get("runs"), list) or len(summary["runs"]) != 1:
        raise ValueError(f"Invalid completed-run summary: {folder}")
    run = summary["runs"][0]
    provenance = read(folder / "provenance.json")
    model = manifest["models"][case["model"]]
    expected_model = {key: model[key] for key in ("path", "size_bytes", "mtime_ns")}
    expected_files = {
        "benchmark_input": ("requests.jsonl", manifest["data"][f"{case['dataset']}/requests.jsonl"]),
        "benchmark_gold": ("gold.jsonl", manifest["data"][f"{case['dataset']}/gold.jsonl"]),
        "harness": ("benchmark-model.py", manifest["scripts"]["benchmark-model.py"]),
        "evaluator": ("evaluate-accuracy.py", manifest["scripts"]["evaluate-accuracy.py"]),
    }
    valid = (run.get("success") and run.get("response_count") == expected_count
             and summary.get("request_count_per_run") == expected_count
             and run.get("request_count") == expected_count
             and run.get("overall", {}).get("examples") == expected_count
             and run.get("failures") == 0
             and run.get("returncode") == 0 and run.get("evaluation_returncode") == 0
             and isinstance(run.get("wall_seconds"), (int, float))
             and math.isfinite(run["wall_seconds"]) and run["wall_seconds"] > 0
             and provenance.get("binary", {}).get("sha256") == manifest["binary"]["sha256"]
             and provenance.get("backend") == case["backend"]
             and provenance.get("model_id") == model["model_id"]
             and provenance.get("model_files") == [expected_model]
             and provenance.get("batch_requests") == 8
             and provenance.get("extra_args") == manifest["common_flags"]
             and provenance.get("environment_overrides") == case["environment"]
             and provenance.get("runs_requested") == 1
             and provenance.get("request_count") == expected_count)
    valid = valid and all(
        provenance.get(key, {}).get("sha256") == item["sha256"]
        and digest(folder / filename) == item["sha256"]
        for key, (filename, item) in expected_files.items()
    )
    run_folder = folder / "run-01"
    valid = valid and digest(run_folder / "responses.jsonl") == run.get("responses_sha256")
    if valid:
        report = read(run_folder / "report.json")
        valid = report.get("overall") == run["overall"] and report.get("failures") == []
    if not valid:
        raise ValueError(f"Completed case failed validation: {folder}")
    return {"id": case["id"], "directory": str(folder), "wall_seconds": run["wall_seconds"],
            "requests_per_second": expected_count / run["wall_seconds"], "overall": run["overall"],
            "responses_sha256": run["responses_sha256"]}


def execute(args, manifest):
    output = args.output_dir
    results = []
    base_env = {key: value for key, value in os.environ.items()
                if not key.startswith(("GGML_", "JET_", "LLAMA_", "VK_", "CUDA_VISIBLE_DEVICES"))}
    if args.cuda_bin:
        base_env["PATH"] = str(args.cuda_bin.resolve()) + os.pathsep + base_env["PATH"]
    for case in manifest["cases"]:
        attempts = sorted(output.glob(case["id"] + "-attempt-*"))
        completed = [result for folder in attempts
                     if (result := completed_case(folder, case, manifest)) is not None]
        if completed:
            results.append(completed[-1])
            continue
        attempt_number = max((int(path.name.rsplit("-", 1)[1]) for path in attempts), default=0) + 1
        folder = output / f"{case['id']}-attempt-{attempt_number:02d}"
        save(output / "status.json", {"state": "running", "updated_at_utc": now(), "pid": os.getpid(),
                                       "active_case": case["id"], "active_directory": str(folder),
                                       "completed_cases": len(results), "total_cases": len(manifest["cases"]),
                                       "results": results})
        # The harness resolves its repository from its source location. Keep a frozen
        # audit copy and reject edits before each launch instead of relocating it.
        for filename in ("benchmark-model.py", "evaluate-accuracy.py", "rerun-paper-4090.py"):
            if digest(ROOT / "scripts" / filename) != manifest["scripts"][filename]["sha256"]:
                raise ValueError(f"Benchmark source changed while suite was running: {filename}")
        for item in [manifest["binary"],
                     manifest["data"][f"{case['dataset']}/requests.jsonl"],
                     manifest["data"][f"{case['dataset']}/gold.jsonl"]]:
            if digest(Path(item["path"])) != item["sha256"]:
                raise ValueError(f"Frozen artifact changed before launch: {item['path']}")
        verify_model(manifest["models"][case["model"]])
        command = [sys.executable, str(ROOT / "scripts/benchmark-model.py"), "--binary", manifest["binary"]["path"],
                   "--model-path", manifest["models"][case["model"]]["path"],
                   "--model-id", manifest["models"][case["model"]]["model_id"],
                   "--input", str(output / case["dataset"] / "requests.jsonl"),
                   "--gold", str(output / case["dataset"] / "gold.jsonl"),
                   "--output-dir", str(folder), "--runs", "1", "--backend", case["backend"],
                   "--batch-requests", "8", "--extra-args", *manifest["common_flags"]]
        env = base_env | case["environment"]
        print(json.dumps({"event": "case_start", "case": case["id"], "at": now()}), flush=True)
        process = subprocess.run(command, cwd=ROOT, env=env)
        if process.returncode:
            save(output / "status.json", {"state": "failed", "updated_at_utc": now(),
                                           "active_case": case["id"], "active_directory": str(folder),
                                           "returncode": process.returncode, "results": results})
            return process.returncode
        verify_model(manifest["models"][case["model"]])
        result = completed_case(folder, case, manifest)
        if result is None:
            raise ValueError(f"Benchmark did not complete: {folder}")
        results.append(result)
        save(output / "results.json", {"scoring_method": manifest["scoring_method"], "results": results})
    save(output / "results.json", {"scoring_method": manifest["scoring_method"], "results": results})
    save(output / "status.json", {"state": "complete", "updated_at_utc": now(),
                                   "completed_cases": len(results), "total_cases": len(manifest["cases"]),
                                   "results": results})
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, help="Newly built mean-token scoring executable")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "tests/accuracy/generated-mmlu-full-ascii")
    parser.add_argument("--cuda-bin", type=Path, help="CUDA runtime DLL directory on Windows")
    parser.add_argument("--plan-only", action="store_true", help="Freeze and validate the suite without running inference")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with suite_lock(args.output_dir):
        phase = "preparation"
        try:
            contain_windows_processes()
            manifest = prepare(args)
            print(json.dumps({"event": "suite_ready", "cases": len(manifest["cases"]),
                              "binary_sha256": manifest["binary"]["sha256"]}), flush=True)
            phase = "execution"
            return 0 if args.plan_only else execute(args, manifest)
        except (Exception, KeyboardInterrupt) as error:
            status_path = args.output_dir / "status.json"
            try:
                status = read(status_path) if status_path.exists() else {}
            except (OSError, ValueError):
                status = {}
            status.update({"state": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                           "updated_at_utc": now(), "pid": os.getpid(), "phase": phase,
                           "error": str(error) or type(error).__name__})
            try:
                save(status_path, status)
            except OSError as status_error:
                print(f"Could not record suite failure: {status_error}", file=sys.stderr)
            print(f"rerun-paper-4090: {status['error']}", file=sys.stderr)
            return 130 if isinstance(error, KeyboardInterrupt) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"rerun-paper-4090: {error}", file=sys.stderr)
        sys.exit(1)
