# Contributing to Jet

Thank you for contributing. Jet is under active development; thoughtful patches,
reviews, bug reports, and design discussion all help.

## Getting started

Use the project-managed toolchain:

```sh
mise install
git submodule update --init --recursive
mise exec -- cargo test --locked --workspace --no-default-features
```

See [README.md](README.md) and [AGENTS.md](AGENTS.md) for build, package, and
documentation commands.

## Normative project rules

All contributions must follow the repository governance index and the rules
applicable to the changed scope:

- [Governance index and precedence](agents/docs/README.md#adoption-and-precedence)
- [Engineering standards](agents/docs/engineering-standards.md)
- [Architecture](agents/docs/architecture.md)
- [Architecture decision records](agents/docs/adr/README.md)
- [Agent guide](AGENTS.md)

Review every document that applies; a documentation-only or isolated internal
change need not pretend to affect every domain. Update the existing owning
documents for current design and behavior, and explain the change and its
validation in the PR. Maintainers decide when a durable choice warrants an ADR;
changes to crate boundaries, protocols, backends, or packaging do not
automatically require one. Agents may create an ADR or mark one Accepted only
when explicitly requested by a maintainer for that action. Follow
[ADR governance](agents/docs/adr/README.md). Policy exceptions and deferred
correctness work must link to an issue.

## Pull requests

Fill in the repository pull request template, and:

- Use a [Conventional Commits](https://www.conventionalcommits.org/) title
  (for example `feat(engine): …`, `fix(cli): …`, `chore(ci): …`).
- Prefer small, reviewable changes with a clear purpose.
- Complete the governance review in the pull request template, using `N/A` only
  with a short reason.
- Match existing style and architecture; do not drive-by reformat unrelated code.
- Keep optional GPU backends (`cuda`, `metal`, `vulkan`) and the `vision` feature
  behind their feature flags; do not force GPU or model downloads into default
  CPU CI paths.
- Ensure submissions are compatible with the
  [Apache License 2.0](LICENSE). Do not paste code of unknown origin or with a
  conflicting license.
- Avoid low-signal volume: mass drive-by refactors, speculative cleanup sweeps,
  or batches of weakly related PRs are likely to be declined.

Maintainers may ask for tests, a narrower diff, or a short explanation of why a
change preserves protocol and backend invariants.

## AI-assisted and AI-generated contributions

Jet **welcomes** AI-assisted and AI-generated code. Used well, AI can raise both
delivery speed and implementation quality.

The project does **not** reject contributions because they were produced with
AI. The same quality bar applies as for fully human-authored work.

### Human responsibility

Any commit or pull request that AI authored or helped produce must have a human
owner who:

1. Understands the change well enough to explain it in review.
2. Can fix defects the AI introduced.
3. Accepts full responsibility for the result—as if they had written every line.

“The model wrote it” is not a reason to merge unsafe, incorrect, or
unmaintainable code, and it is not a reason to leave bugs unfixed later.

### Review

AI-involved changes require substantive human review by at least one person who
meets the bar above. Rubber-stamp approval is not enough: the reviewer should
read for intent, boundaries, and failure modes, and should be able to repair
mistakes.

Prefer a second reviewer who knows the affected subsystem before merge—
especially for high-risk areas:

- `unsafe` Rust, FFI, and `jet-llama-sys` / llama.cpp boundaries
- Scoring math and protocol serialization
- Backend feature flags and packaging
- CI, cache, and permission changes

Tests generated or extended by AI must be checked by a human. Green CI is not
sufficient if assertions miss the real requirement or quietly weaken coverage.

### Attribution and disclosure

- The git committer should be the responsible human. Co-Author trailers that
  name an AI tool are **not** required.
- Mentioning AI use in a pull request description is **optional**. Maintainers
  must not reject a change solely because AI was or was not disclosed.

### Maintainer discretion and newcomer space

Some AI-involved pull requests may still be declined or deferred. That is not a
ban on AI; it protects review bandwidth and leaves room for newcomers. Reasons
include, without limitation:

- Work reserved for newcomers
- Diffs that are too large to review carefully, or that ignore documented
  architecture constraints
- Duplicate or already-decided work
- Low-signal or spam-like contribution patterns

Maintainer judgment is final in these cases.

## Issues

Prefer a clear reproduction, expected versus actual behavior, and platform /
backend details when relevant. AI may help draft issues; the report itself should
still be accurate and actionable. Do not open floods of speculative or duplicate
tickets.

## License

By contributing, you agree that your contributions are licensed under the
[Apache License 2.0](LICENSE).
