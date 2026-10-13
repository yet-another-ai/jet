# Agent Development Notes

This directory holds project and agent-facing documentation that is too detailed
for the repository-level `AGENTS.md`.

Use it for architecture, engineering standards, ADR policy, agent workflows, and
implementation checklists. Manage reusable agent skills through `apm.yml` rather
than documenting or editing installed copies under `.agents/skills/`.

## Notes

- [Architecture](architecture.md)
- [Architecture decision records](adr/README.md)
- [Engineering standards](engineering-standards.md)
- [Continuous integration (public)](../../docs/ci.md)

## Authority

`AGENTS.md` is the concise repository entry point. These documents are the
normative detail behind it. Architecture defines current ownership and design;
engineering standards govern code and tests; ADRs preserve the rationale for
durable choices. Creation, acceptance, and revisions follow
[ADR governance](adr/README.md).

The public `docs/` workspace describes behavior that is already available to
users. Do not use public documentation to announce unimplemented roadmap scope.

## Adoption and precedence

These rules are adopted for all new work and all materially changed code or
behavior. Existing code is not evidence that a conflicting pattern is still
permitted. When a change touches non-conforming code, either bring the affected
scope into compliance or link a narrowly scoped tracking issue or policy
exception.

Apply the documents by responsibility:

1. [Engineering standards](engineering-standards.md) govern implementation,
   testing, review evidence, errors, module cohesion, and exceptions.
2. [Architecture](architecture.md) governs current crate ownership, dependency
   direction, protocol boundaries, and backend feature assumptions.
3. [Accepted ADRs](adr/README.md) preserve durable decisions and explain why
   the current architecture rule exists.

A proposed ADR is not authority. Maintain current rules in the owning documents
as part of authorized work. Historical implementation details in an ADR do not
independently freeze the current design; clarify misleading details with a dated
amendment and a link as described in [ADR governance](adr/README.md). Genuine
conflicts about durable commitments require resolution within the maintainer's
direction. Raise unresolved substantive conflicts and continue unaffected work.
ADR creation is not a routine approval stage.

Judgment-based rules are recorded in the pull-request governance review.
Exceptions and deferred correctness work require a linked issue; no separate
waiver process is required at the current project scale.
