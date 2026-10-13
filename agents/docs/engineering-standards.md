# Engineering standards

These rules apply to hand-authored production code, build tooling, tests, and
architecture changes in this repository. `MUST`, `MUST NOT`, `SHOULD`, and
`MAY` are normative. A rule that can be checked mechanically should become a
repository check. A rule that requires judgment belongs in the review
checklist.

The standards optimize first for the maintainer, then for AI agents and external
contributors. Prefer explicit ownership and checkable constraints over a clever
abstraction whose safe use depends on oral history.

## Change discipline

- A change MUST have one stated user, product, maintenance, or reliability
  outcome. Incidental cleanup must remain reviewable.
- New behavior MUST define success, unavailable, failed, and recovery states
  where applicable (especially around model load, backends, and I/O).
- A bug fix MUST first add an automated reproduction that fails for the reported
  behavior when feasible. When automation is genuinely impossible, the change
  description MUST state why and give a repeatable manual check.
- Implementation, tests, agent-facing documentation, and shipped user
  documentation MUST change together when their shared behavior changes.
- `TODO` and `FIXME` comments MUST link to a tracked issue. Comments without an
  owner or durable tracking reference are forbidden.
- Do not edit APM-installed files under `.agents/skills/`; update `apm.yml` and
  regenerate through the locked APM workflow.

## Ownership and dependency direction

Keep the crate structure described in [architecture.md](architecture.md). Split
modules by feature or runtime ownership, not by arbitrary categories such as
`types`, `helpers`, or `utils`.

- `jet-core` MUST remain free of llama.cpp, filesystem CLI concerns, and
  packaging logic.
- Protocol and error types that cross the public API MUST live in `jet-core`
  (or a deliberately shared crate), not duplicated in the CLI.
- `unsafe` blocks MUST be minimized, documented, and kept near the FFI boundary
  (`jet-llama-sys` and narrow engine adapters).
- Optional backends and vision MUST stay behind Cargo features. Default CPU
  paths MUST NOT require GPU toolkits or network model downloads.

## Errors and API boundaries

- Public Rust APIs SHOULD return `jet_core::Result` / `JetError` rather than
  panicking on expected failure modes (missing model, unsupported backend,
  invalid request).
- Clippy lints that deny `unwrap`/`expect`/`panic`/`todo` in workspace code are
  part of the contract; do not weaken them without maintainer direction.
- CLI process exit codes MUST remain meaningful: nonzero when any JSONL line
  fails.

## Tests

- Tests MUST prove Jet-owned behavior at the narrowest useful boundary.
- Prefer locked, offline, CPU-default tests in CI. Model-dependent tests that
  need downloaded weights SHOULD stay `#[ignore]` or otherwise opt-in unless
  the fixture is vendored and cheap.
- Python helper tests under `training/` and `tests/accuracy/` MUST remain
  offline in default CI (`HF_*_OFFLINE`, no training runtime unless explicitly
  enabled).
- Packaging smoke tests (`cargo xtask package`) MUST keep archive extract and
  model-free CLI checks; they MUST NOT publish releases or sign artifacts from
  the default Gate path.

## Documentation

- Update the existing owning document for current behavior. Do not create a
  parallel “decision” note to bypass [ADR governance](adr/README.md).
- User-visible CLI/API changes update `docs/`. Agent and architecture rules
  update `agents/docs/` and `AGENTS.md` as needed.
- CI policy that users and contributors need lives in `docs/ci.md`.

## Exceptions

Policy exceptions and deferred correctness work MUST link to a scoped issue.
Authorization to land a change does not by itself authorize an ADR.
