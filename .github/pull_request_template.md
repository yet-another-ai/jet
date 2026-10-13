<!--
Title: use Conventional Commits, e.g.
  feat(engine): normalize multimodal usage fields
  fix(cli): report backend feature when unavailable
  chore(ci): keep Gate as the single required check
-->

## Summary

<!-- State the user, product, maintenance, or reliability outcome. Prefer purpose over a file list. -->

-

## Tracking

<!-- Link the issue when one exists. Every exception, TODO/FIXME, or deferred correctness item requires one. -->

- Issue: <!-- Closes #... / Relates to #... / None -->

## Governance review

<!-- Check each line after reviewing the linked rule. Use the notes below for evidence or explain N/A. -->

- [ ] [Engineering standards](https://github.com/yet-another-ai/jet/blob/main/agents/docs/engineering-standards.md): ownership, failure states, tests, documentation, and source cohesion are addressed.
- [ ] [Architecture](https://github.com/yet-another-ai/jet/blob/main/agents/docs/architecture.md): crate boundaries, protocol ownership, backend features, and FFI constraints are preserved or updated.
- [ ] [ADR governance](https://github.com/yet-another-ai/jet/blob/main/agents/docs/adr/README.md): current documentation is updated where needed; any new ADR or acceptance follows explicit maintainer direction.
- [ ] Exceptions, TODOs/FIXMEs, and deferred correctness work are absent or linked to a scoped issue.

<!--
Engineering evidence / N/A reason:
Architecture impact / N/A reason:
Documentation updates; maintainer ADR direction if applicable; exception link:
-->

## Test plan

<!-- How you validated this. Check boxes as you go. -->

- [ ] `mise exec -- cargo test --locked --workspace --no-default-features` (or narrower checks noted below)
- [ ]
- [ ]

## Notes for reviewers

<!-- Risk areas, highest-risk behavior, FFI/backend impact, and issue-linked follow-ups. -->
