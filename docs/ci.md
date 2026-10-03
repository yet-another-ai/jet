---
title: Continuous integration
description: Jet's CPU test, package, dependency, permission, and cache policies.
---

# Continuous integration

CI uses separate reusable Test, Build, and Documentation workflows, with one final
aggregate `Gate` in the entry workflow. It runs on every pull
request, main push, or manual dispatch. All required jobs must succeed, including
every matrix leg. Failures, cancellations and unexpected job skips fail the gate.
Linux-only format/vision steps are intentionally skipped on other test hosts.
Superseded runs are cancelled; no repository settings are changed by the workflow.

## Tests

- Linux x64, Windows x64 and macOS arm64 run locked CPU-only Cargo Clippy and
  workspace tests. Linux also checks formatting and the CPU `vision` feature's
  validation/unit tests. Model-dependent tests keep their existing `#[ignore]`.
- macOS supports Apple Silicon ARM64 only. Both component workflows verify
  `runner.arch` and `uname -m` before any checkout/build/test. There are no Intel
  macOS matrix entries or archives; the native packager also rejects that host.
- Linux checks Python syntax/undefined names with Ruff's `E4,E7,E9,F` rules,
  installs only the locked Python `dev` dependency group (Ruff), then runs
  offline data/tokenization/training-safety/evaluation helper tests and evaluator
  fixture tests. The opt-in training runtime tests remain skipped.
- No CUDA, Metal or Vulkan feature/toolkit, GPU runner, pretrained model, dataset
  preparation/download, training, inference benchmark or accuracy evaluation is
  enabled. Offline Hugging Face flags and `JET_RUN_TRAINING_RUNTIME=0` reinforce
  that boundary. There is no Electron deliverable.

## Documentation

The Documentation workflow installs the locked Node and pnpm tools, restores the
frozen dependency graph, and builds the VitePress site. The build validates page
rendering and internal links, then uploads the static output as a GitHub Pages
artifact. Pull requests must pass this build. A successful main-branch Gate deploys
the same artifact to GitHub Pages.

## Packaging

Native CPU hosts produce these CLI archives, retained for
14 days as Actions artifacts:

| Host | Artifact |
| --- | --- |
| Windows x64 (`windows-latest`) | `jet-cpu-windows-x64.zip` |
| Linux x64 (`ubuntu-26.04`) | `jet-cpu-linux-x64.tar.gz` |
| Linux arm64 (`ubuntu-26.04-arm`) | `jet-cpu-linux-arm64.tar.gz` |
| macOS arm64 (`macos-latest`) | `jet-cpu-macos-arm64.tar.gz` |

The [official runner image list](https://github.com/actions/runner-images#available-images)
was checked on 2026-10-02: `macos-latest` maps to macOS 26 ARM64, and
`windows-latest` maps to Windows Server 2025. Ubuntu's newest available image is
26.04; `ubuntu-latest` still maps to 24.04, so both Linux architectures use the
newer explicit 26.04 labels. The single Gate and Python checks also use 26.04.

`cargo xtask package` builds the actual release CLI with `--locked` and no
optional features; the release optimization profile is preserved. It includes
Jet/llama.cpp licenses, README and small JSONL fixtures.
It extracts each archive and checks `--version`, `--help`, and model-free
`export-prompts` JSONL output. It never publishes a release or signs an artifact.
Linux packages use the host glibc/C++ runtime (Ubuntu 26.04 baseline), macOS uses
system libraries, and Windows statically links the MSVC runtime. These are CPU packages, not
GPU runtime bundles. GPU packaging scripts remain available for local use.

`JET_CPU_PORTABLE=1` disables build-host CPU tuning and SSE4.2/AVX/AVX2/BMI2/FMA/F16C for
portable x64 artifacts, and sets the ARM CPU baseline to `armv8-a`. This trades
performance for portability; local builds retain normal native tuning unless
explicitly opted in. Run `cargo xtask package` (or `mise run package:cpu`) with
the native platform toolchain installed; CI uses the same Rust entry point with
`--platform` to check the expected host. Rust libraries generate and extract both
archive formats without PowerShell, Bash or external archivers. Running the task
again replaces its CPU archive.

## Dependencies, permissions and caches

Mise 2026.10.0 is pinned separately from action revisions; the already-merged
tooling PR #1 supplies the minimum version. Only Rust/CMake or uv are installed
with `--locked` from `mise.lock`. Cargo uses `Cargo.lock`; Ruff is installed with
`uv sync --locked --only-group dev` and uses `training/.python-version`. No lockfile
is regenerated in CI. Cargo.lock also locks the xtask archive dependencies.
Actions are pinned to immutable commit SHAs.

All workflows have `contents: read`; checkout disables persisted credentials.
PR setup receives an explicitly empty token, with no downstream token default.
Non-PR setup can use the read-only workflow token for tool downloads. No secrets
are inherited, and `pull_request_target` is not used.

Mise caches include tool version, OS/architecture, install subset and config/lock
hashes. Cargo caches separate test/release configurations and include OS,
architecture, runner image version, tool/dependency locks and the pinned
llama.cpp gitlink. These identify the native toolchain and dependency inputs;
Cargo/CMake track changed source files. The run ID lets successful trusted runs
refresh build caches without hashing workflow/source files. Python's uv cache
uses OS/architecture and `uv.lock`; uv tracks wheel/interpreter compatibility.
PRs restore caches but cannot publish them. Build output archives are not cached.

Run `apm install` to restore the project-local Stop That Shit skill from
`lennney/stop-that-shit/skills/stop-that-shit` at its locked revision.
