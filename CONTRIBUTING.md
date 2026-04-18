# Contributing to hermes-lcm

Thanks for contributing.

This project is small, review-driven, and correctness-first. Keep changes scoped, tested, and easy to reason about.

## Workflow

Preferred flow:

1. Open or reference an issue when the change affects behavior, architecture, or public tooling.
2. Create a focused branch from `main`.
3. Add or update tests with the change.
4. Run local validation before opening the PR.
5. Open a PR with a clear summary, rationale, and validation section.

Typical branch names:

- `fix/...`
- `feat/...`
- `docs/...`
- `refactor/...`
- `test/...`

## Issues

Use issues for:

- bugs
- behavior regressions
- architectural direction
- follow-up work that should not be buried in PR comments

When filing a bug, include:

- expected behavior
- actual behavior
- minimal repro steps
- relevant logs, stack traces, or failing tests
- version / branch context when relevant

If a report is speculative, say so. If it is directional rather than a concrete bug, label it clearly in the issue body.

## Commits

Prefer clear, conventional-style subjects:

- `fix: ...`
- `feat: ...`
- `docs: ...`
- `refactor: ...`
- `test: ...`

Keep commits focused. Avoid mixing unrelated cleanup into the same change.

## Pull Requests

Open small PRs when possible. Large PRs are harder to review and easier to get wrong.

PR titles should be descriptive and usually follow the same style as commit subjects.

PR bodies should use this shape:

```md
## Summary
- what changed
- what changed

## Why
- why this change is needed
- important constraints or tradeoffs

## Validation
- exact commands you ran
- relevant outputs if useful

## Notes
- follow-ups, caveats, or non-blocking context
```

Good PRs are:

- accurate about what is actually implemented
- honest about scope
- explicit about tradeoffs
- backed by tests

Do **not** claim behavior that is only partially implemented. If a filter, feature, or fix only applies to one path, say that clearly.

## Validation

Default validation for code changes:

```bash
pytest tests/test_lcm_core.py tests/test_lcm_engine.py -q
pytest -q
python -m compileall -q .
git diff --check
```

If your PR only touches a narrow surface area, include the focused command too. Example:

```bash
pytest tests/test_lcm_command.py -q
```

If you skip part of the default validation, explain why in the PR body.

## Testing expectations

- behavior changes should come with tests
- bug fixes should include a regression test when practical
- command/output changes should verify the rendered text, not just internal helpers
- keep tests readable; avoid clever fixtures when simple setup is enough

## Review expectations

Before requesting review:

- rebase or merge `main` so the branch is current
- resolve conflicts locally
- make sure the PR description matches the branch exactly
- ensure CI is expected to pass from the current head

Reviewers will check:

- correctness
- edge cases
- test coverage
- whether the implementation matches the claimed behavior
- whether the change is appropriately scoped

## Scope guidelines

Priority order:

1. correctness
2. regressions
3. operator safety
4. maintainability
5. new features

Backwards-compatible, well-tested changes are preferred. Destructive or risky workflows should be backup-first and clearly labeled.

## Documentation

Update docs when you change:

- user-facing commands
- tool schemas
- configuration flags
- expected operator workflows

If a new feature needs explanation for contributors or operators, document it in the same PR.

## Questions

If you are unsure whether something should be an issue first, open the issue. It is cheaper than reviewing the wrong PR.
