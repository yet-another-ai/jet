# Agent Guide

This file is the entry point for AI agents and automation working in this
repository. Keep project and agent-facing documentation under `agents/docs/`,
the VitePress website and user guide under `docs/`, and reusable agent skills
under `.agents/skills/`.

## Project Shape

- Product: local structured decision engine. Given context and a fixed set of
  candidate answers, Jet scores candidates with teacher forcing and returns a
  structured result instead of free-form generation.
- Interfaces: JSONL CLI and `jet serve` (`jet-cli`), Actix HTTP service
  (`jet-server`), and Rust API (`jet-core` protocol types, `jet-engine` runtime).
- Crates:
  - `jet-core` — request/response protocol, errors, and score normalization.
  - `jet-engine` — llama.cpp-backed scoring, prompts, backends, optional vision.
  - `jet-llama-sys` — FFI and build glue for the vendored llama.cpp submodule.
  - `jet-server` — HTTP Decisions adapter, bounded scheduling, and server lifecycle.
  - `jet-cli` — command-line front end.
  - `xtask` — packaging and related repository tasks.
- Backends: CPU by default; optional CUDA, Metal, and Vulkan features for local
  GPU offload. CI exercises CPU-only builds and packages.
- Toolchain: `mise` with locked Rust, CMake, Node, pnpm, uv, and related tools
  in `mise.toml` / `mise.lock`. Prefer `mise exec -- …` / `mise run …` so the
  repository-managed versions are used.
- Native dependency: `vendor/llama.cpp` is a pinned git submodule. Initialize
  submodules before building.
- Documentation boundary: public user docs live in `docs/` (VitePress).
  Architecture, engineering standards, ADR policy, and agent workflows live
  under `agents/docs/`. Do not hand-edit APM-installed skills under
  `.agents/skills/`; update `apm.yml` instead.
- Maintain current design and behavior in the existing owning documents by
  default. Agents MUST NOT create an ADR or mark one Accepted without an
  explicit maintainer request for that action; authorization to implement a
  change does not imply either. See `agents/docs/adr/README.md`.

## Common Commands

```sh
mise install
git submodule update --init --recursive
mise exec -- cargo fmt --all -- --check
mise exec -- cargo clippy --locked --workspace --all-targets --no-default-features -- -D warnings
mise exec -- cargo test --locked --workspace --no-default-features
mise run package:cpu
mise exec -- pnpm docs:build
# Optional: embed the WebGPU playground into the docs build
# mise run playground:wllama && mise exec -- pnpm docs:build
```

Python tooling (offline helpers under `training/` and `tests/accuracy/`):

```sh
uv sync --directory training --locked --only-group dev
uv run --directory training --no-sync ruff check --select E4,E7,E9,F \
  jet_training tests ../scripts ../tests/accuracy/test_evaluate_accuracy.py
```

Always run project commands from a login shell so startup files activate the
repository-managed toolchain. On Windows, use PowerShell 7 (`pwsh`).

## Supporting Notes

- [Repository overview](README.md)
- [User guide](docs/guide/getting-started.md)
- [HTTP server](docs/guide/http-server.md)
- [Architecture](agents/docs/architecture.md)
- [Architecture decision records](agents/docs/adr/README.md)
- [Engineering standards](agents/docs/engineering-standards.md)
- [Continuous integration](docs/ci.md)
- [Agent development notes](agents/docs/README.md)
- [Agent skill dependencies](apm.yml)

## Documentation Boundary

Keep repository-wide agent instructions in `AGENTS.md`. The `docs/` workspace is
the public VitePress site: landing page, user guide, results, and advanced
usage. Place architecture, engineering standards, ADR policy, CI policy detail
for agents, and implementation checklists under `agents/docs/`. Treat
`.agents/skills/` as APM-managed content derived from `apm.yml` and
`apm.lock.yaml`. Do not place temporary agent notes or generated output in
either documentation directory.
