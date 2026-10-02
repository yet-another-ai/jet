# CPU CI

CI follows Heron's separate reusable Test and Build workflows, with one final
aggregate `Gate` in the entry workflow. It runs on every pull
request, main push, or manual dispatch. All required jobs must succeed, including
every matrix leg. Failures, cancellations and unexpected job skips fail the gate.
Linux-only format/vision steps are intentionally skipped on other test hosts.
Superseded runs are cancelled; no repository settings are changed by the workflow.

## Tests

- Linux x64, Windows x64 and macOS arm64 run locked CPU-only Cargo Clippy and
  workspace tests. Linux also checks formatting and the CPU `vision` feature's
  validation/unit tests. Model-dependent tests keep their existing `#[ignore]`.
- Linux checks Python syntax/undefined names with Ruff's `E4,E7,E9,F` rules,
  installs only the locked Python `dev` dependency group (Ruff), then runs
  offline data/tokenization/training-safety/evaluation helper tests and evaluator
  fixture tests. The opt-in training runtime tests remain skipped.
- No CUDA, Metal or Vulkan feature/toolkit, GPU runner, pretrained model, dataset
  preparation/download, training, inference benchmark or accuracy evaluation is
  enabled. Offline Hugging Face flags and `JET_RUN_TRAINING_RUNTIME=0` reinforce
  that boundary. There are no JavaScript/Electron deliverables or checks in Jet.

## Packaging

Native CPU hosts produce these CLI archives, plus SHA-256 sidecars, retained for
14 days as Actions artifacts:

| Host | Artifact |
| --- | --- |
| Windows x64 (`windows-2025`) | `jet-cpu-windows-x64.zip` |
| Linux x64 (`ubuntu-24.04`) | `jet-cpu-linux-x64.tar.gz` |
| Linux arm64 (`ubuntu-24.04-arm`) | `jet-cpu-linux-arm64.tar.gz` |
| macOS arm64 (`macos-15`) | `jet-cpu-macos-arm64.tar.gz` |

`scripts/package-cpu.ps1` builds the actual release CLI with `--locked` and no
optional features; the release optimization profile is preserved. It includes
Jet/llama.cpp licenses, README, small JSONL fixtures and source/build metadata.
It extracts each archive and checks `--version`, `--help`, and model-free
`export-prompts` JSONL output. It never publishes a release or signs an artifact.
Linux packages use the host glibc/C++ runtime (Ubuntu 24.04 baseline), macOS uses
system libraries, and Windows statically links the MSVC runtime. These are CPU packages, not
GPU runtime bundles. GPU packaging scripts remain available for local use.

`JET_CPU_PORTABLE=1` disables build-host CPU tuning and AVX/AVX2/FMA/F16C for
portable x64 artifacts, and sets the ARM CPU baseline to `armv8-a`. This trades
performance for portability; local builds retain normal native tuning unless
explicitly opted in. Run `mise run package:cpu` with PowerShell 7 and the native
platform toolchain installed. Existing archives are never silently overwritten.

## Dependencies, permissions and caches

Mise 2026.10.0 is pinned separately from action revisions; the already-merged
tooling PR #1 supplies the minimum version. Only Rust/CMake or uv are installed
with `--locked` from `mise.lock`. Cargo uses `Cargo.lock`; Ruff is installed with
`uv sync --locked --only-group dev` and uses `training/.python-version`. No lockfile
is regenerated. Actions are pinned to immutable commit SHAs.

All workflows have `contents: read`; checkout disables persisted credentials.
PR setup receives an explicitly empty token, with no downstream token default.
Non-PR setup can use the read-only workflow token for tool downloads. No secrets
are inherited, and `pull_request_target` is not used.

Mise caches include tool version, OS/architecture, install subset and config/lock
hashes. Cargo caches separate test/release configurations and include OS,
architecture, runner image version, tool/config/lock/build-script hashes and the
pinned llama.cpp gitlink; restore prefixes never cross those boundaries. The
commit suffix allows fresh successful main builds to refresh native build trees.
Python's uv cache is keyed by OS/architecture and tool/Python/dependency locks.
PRs restore caches but cannot publish them. Build output archives are not cached.
