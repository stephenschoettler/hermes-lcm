# Upstream the Five Local LCM Commits as Feature Pull Requests

**Goal:** Publish the five local `hermes-lcm` commits as three focused upstream pull requests with source-backed rationale, complete behavior, tests, and clean public authorship.

**Status:** Plan only. No issue, fork, branch push, or pull request has been created.

## Repository state

- Repository: `/Users/grantjordan/.hermes/plugins/hermes-lcm`
- Upstream: `stephenschoettler/hermes-lcm`
- Inspected upstream base: `6b7dbb13f6013042b831cdd4790066a8e376a2d9`
- Verified local tip: `5f0832bcac9ad509481a3a32b79677cc566e2a57`
- Range: five commits, zero behind the inspected upstream base
- Planning worktree: `/private/tmp/hermes-lcm-upstream-pr-planning-20260804`
- Planning branch: `plan/lcm-upstream-prs-20260804`
- Publication route: Grant's public fork, because Grant lacks upstream push permission

## Evidence and root constraint

The five commits contain three independent features:

1. Noncritical maintenance enters synchronous preflight before the first model call, even below the context threshold.
2. Every cached agent clone opens another SQLite helper set, multiplying descriptors and initialization work.
3. Summary and expansion calls need separate reasoning budgets, but upstream exposes only model routing.

Follow-up commits are not separate features:

- `0dd5063` corrects the reasoning payload to use Hermes `reasoning_config`.
- `5f0832b` removes a duplicate lifecycle import.
- `b4dde57` adds three-line test isolation against ambient provider registration.

Upstream overlap:

- Shared SQLite storage: issue #463 describes the exact descriptor problem. Upstream has related locks and lifecycle pieces, but no shared reference-counted bundle.
- Preflight deferral: upstream tracks maintenance debt and critical pressure, but noncritical debt still enters synchronous preflight.
- Reasoning controls: upstream has no summary/expansion effort settings or Hermes-authoritative effort override.

## Defect found during planning

The local tip defines and displays `summary_reasoning_effort`. The escalation layer accepts the setting. The two real engine summary call sites do not pass the configured value.

Expansion reasoning is wired end to end. Summary reasoning is not.

The reasoning pull request must add a failing consumption-level test and wire both summary sites. Publishing the current partial implementation unchanged is prohibited.

## Scope

### In

- Create or reuse public `grantjayy/hermes-lcm`.
- Base every pull request on current upstream `main`.
- Open two new upstream design issues and reuse issue #463.
- Produce three independent pull requests.
- Split mixed commit `17df35d` by hunk.
- Repair summary-reasoning consumption before publication.
- Rewrite unpublished commits to Grant's verified GitHub noreply identity.
- Run focused, default, full, release, lint, compilation, and diff checks.
- Read back every issue, fork branch, and pull request.

### Out

- No pull-request merge.
- No gateway restart or runtime update.
- No Hermes core edits.
- No force-push to existing shared branches.
- No deletion of the installed checkout or existing worktrees.
- No broad asynchronous-compaction redesign.
- No absorption of pull request #470 or #486.
- No unrelated test repair beyond `b4dde57`.

## Publication order

1. Reasoning controls: no upstream equivalent and low collision risk.
2. Shared SQLite storage: highest measured impact and existing issue #463, but larger lifecycle surface.
3. Preflight deferral: smallest diff, but needs clear differentiation from existing deferred maintenance.

All three branches must remain independently applicable to upstream `main`.

## Pull request 1: Separate reasoning controls

### New issue

**Title:** `feat: configure summary and expansion reasoning effort separately`

**Why:** Routine summaries and user-requested expansion answers have different cost and quality needs. Hermes has one authoritative per-call `reasoning_config`; provider-specific reasoning fields can conflict with host routing.

Request:

- independent `summary_reasoning_effort` and `expansion_reasoning_effort`;
- YAML and environment precedence matching existing config;
- rejection of unsupported nonempty values;
- status diagnostics;
- propagation through every real summary and expansion call;
- Hermes `reasoning_config` as the only effort override;
- empty settings preserving normal task defaults.

### Pull request

**Title:** `feat: add separate summary and expansion reasoning controls`

**Source:** reasoning hunks from `17df35d`, all of `0dd5063`, and `b4dde57`.

**Files:**

- Modify `README.md`.
- Modify `config.py`.
- Modify `engine.py` for the two missing summary-call arguments.
- Modify `escalation.py`.
- Modify `model_routing.py`.
- Modify `tools.py`.
- Modify `tests/test_lcm_core.py` for test isolation.
- Modify `tests/test_lcm_engine.py` for consumption tests.
- Modify `tests/test_reasoning_routing.py`.

### Test-driven steps

1. Add a failing test that configures summary effort and captures the real auxiliary request through an engine summary path.
2. Confirm failure occurs because the engine omits the setting.
3. Pass `reasoning_effort=self._config.summary_reasoning_effort` at leaf and condensation summary calls.
4. Preserve expansion proof that `reasoning_config` reaches Hermes and `extra_body.reasoning` never appears.
5. Preserve empty-setting and strict-validation tests.
6. Keep the three-line provider-registry isolation beside its existing routing test.

### Acceptance

- Summary and expansion efforts can differ.
- Both reach real outbound Hermes call boundaries.
- `none` disables reasoning through `reasoning_config`.
- Empty values preserve defaults.
- Unsupported nonempty values fail loudly.
- Status reports both effective settings.
- No provider-specific reasoning alias is injected.

## Pull request 2: Shared SQLite clone storage

### Existing issue

Link #463: `fix: agent clones multiply SQLite file descriptors`.

Close #463 only if the final pull request covers the issue's full accepted direction. The local clone-family bundle does not yet prove cross-bundle construction locking for independently created engines that share one database path. Either add that behavior and its concurrency tests, obtain maintainer agreement that the narrower clone-family fix resolves the issue, or leave #463 open with the remaining work stated clearly.

Re-run a checked-in or pull-request-attached clone benchmark against both the pinned upstream base and the final feature branch. The benchmark must measure retained-clone file descriptors, median clone setup, ten-clone setup, and initial startup. Record its exact command, environment, raw output path, and both commit SHAs.

Historical private results provide targets, not publishable claims:

- Before the prototype, 12 retained clones added about 72 file descriptors.
- With the prototype, 12 retained clones added no physical SQLite descriptors.
- Median clone setup improved from 5.90 ms to 0.19 ms.
- Ten-clone setup improved from 91.63 ms to 1.96 ms.
- Initial plugin startup stayed effectively flat.

Publish only measurements reproduced by the final benchmark run. Do not copy the historical private timings into the issue or pull request unless the new receipt confirms them.

### Pull request

**Title:** `fix: share SQLite storage across LCM engine clones`

**Source:** storage, locking, lifecycle, and clone-test hunks from `17df35d`. Do not replay `5f0832b`; the commit only repairs import pollution introduced by the mixed local commit.

**Files:**

- Modify `engine.py`.
- Modify `store.py`.
- Modify `dag.py`.
- Modify `tests/test_lcm_core.py`.

Keep the net `lifecycle_state.py` diff empty unless current upstream analysis proves a required storage behavior change. Do not carry the unused `functools.wraps` import or unrelated comment deletions from the local range.

### Behavior

- One clone family shares one helper bundle for message storage, summary DAG, lifecycle state, assertions, and query views.
- Each engine keeps independent session, cursor, model, provider, and runtime state.
- Each clone owns one lease.
- Shutdown releases one lease and closes helpers after the final lease.
- Shared SQLite operations are serialized across concurrent clones.
- Clone-construction failure rolls back its lease.
- Abandoned clones release through finalization.
- Final cleanup attempts every helper close and reports combined failures.

### Coordination

- Pull request #470 is complementary. Final clone cleanup must release a lease rather than duplicate bundle finalization.
- Pull request #486 handles cross-process FTS bootstrap locking. Shared in-process locking must not claim to replace it.
- Inspect current upstream helper ownership before replay. Every helper from `_bind_storage` must be bundle-owned or explicitly engine-local.

### Acceptance

- Retained clones do not linearly increase LCM SQLite descriptors.
- Owner-first shutdown leaves clones operational.
- Final release closes each helper once.
- Concurrent clone writes and DAG operations remain correct.
- Runtime state stays clone-local.
- Model and provider metadata remain preserved.
- Cleanup failures are visible.

## Pull request 3: Defer noncritical preflight maintenance

### New issue

**Title:** `perf: keep noncritical LCM maintenance off the pre-model path`

**Why:** LCM can request synchronous compression before the first model call while below threshold. Routine debt should wait until threshold, overflow, explicit compression, or configured critical pressure requires work.

Differentiate the request from:

- #25 maintenance-debt catch-up;
- #151 critical-pressure escape;
- #440 session-start rollup deferral;
- open #287 asynchronous compaction.

### Pull request

**Title:** `perf: defer noncritical LCM preflight maintenance`

**Source:** all of `b0fb0d3`.

**Files:**

- Modify `compaction.py`.
- Modify `tests/test_lcm_engine.py`.

### Behavior

- Under-threshold eligible backlog records debt without synchronous compression.
- Under-threshold deferred debt compresses only at configured critical pressure.
- Threshold, overflow, cleanup, ignored-backlog, and explicit required paths stay synchronous.
- Disclose that a disabled critical-pressure ratio can postpone debt until another required trigger.

### Acceptance

- Real preflight returns false for noncritical under-threshold backlog.
- Required sibling branches remain true.
- Debt remains recorded.
- No summary model call occurs in the deferred case.

## Clean public submission procedure

For each feature:

1. Open or reuse the governing upstream issue before branch creation and record its number.
2. Require the planned commits and pull request body to contain `Refs #<number>`; use `Closes` only when the final scope fully resolves the accepted issue.
3. Fetch and pin current upstream `main` immediately before branch creation.
4. Create a fresh isolated worktree from that exact SHA.
5. Configure Grant's verified public identity locally.
6. Replay only approved feature hunks; never cherry-pick mixed `17df35d` wholesale.
7. Use test-driven development for missing summary consumption.
8. For shared storage, create and run the named clone descriptor and latency benchmark against the pinned base and final branch.
9. Verify commit count, files, cumulative diff, and whitespace.
10. Rewrite only unpublished commits with `--reset-author` when needed.
11. Require equal cumulative patch IDs before and after metadata-only rewriting.
12. Create or verify the public fork while preserving upstream identity.
13. Push only the focused branch to Grant's fork.
14. Require local and remote SHAs to match.
15. Open the upstream pull request with a body file and explicit repository, base, and head.
16. Read back state, base, head owner, head SHA, commits, files, issue references, and checks.

## Validation for every pull request

Run from its clean worktree:

- Named feature-focused tests.
- `pytest tests/test_lcm_core.py tests/test_lcm_engine.py tests/test_packaging_install.py -q`.
- `pytest -q`.
- `scripts/validate_release.sh --full --keep-going --output /tmp/hermes-lcm-release-validation-<topic>`.
- `python -m compileall` over changed Python files.
- Repository-configured Ruff validation.
- `git diff --check <pinned-base>...HEAD`.
- `git diff --check`.
- `git diff --cached --check`.

Classify broad failures against pristine worktrees at the same upstream SHA. Never call a branch green while a branch-caused failure remains.

Monitor current-head checks and review threads after publication. Repair only reproducing current-head findings. Never merge.

## Issue and pull-request writing requirements

Use upstream's template: Summary, Why, Validation, Notes, and governing issue reference.

Every pull request body must contain a `Hermes-Session` provenance line for this contribution session.

Do not claim behavior, measurements, or test success without receipts from the final branch.

## Hard stops

Stop and return to Grant if:

- upstream `main` changes the behavior or produces a material conflict;
- issue discussion changes the proposed behavior;
- #470 lands and changes finalization ownership;
- shared storage needs an architecture choice not settled by #463;
- clone-local runtime state cannot be preserved;
- summary effort cannot be proven at the real call boundary;
- the account cannot create a public fork or push branches;
- a verified public author identity cannot be established with current credentials;
- a branch introduces failures absent from pristine upstream;
- destructive rewriting, merging, gateway restart, or deployment becomes necessary.

## Authority

Hermes may decide:

- mechanical hunk placement within the three settled features;
- test names and fixture details;
- branch names following upstream conventions;
- wording edits preserving the reasoning above;
- repairs for review findings that do not change behavior or scope.

Grant must decide:

- material behavior, architecture, or scope changes requested upstream;
- whether to proceed without verified public identity;
- destructive rewrites of published history;
- whether and when any pull request is merged.

## Routing

- `reasoning_mode`: adversarial
- `execution_topology`: durable
- `gjc_profile`: adversarial
- `gjc_workflow`: ultragoal
- `capability_evidence`: Mixed commits cross SQLite ownership, concurrent cleanup, pre-model latency, and provider request contracts. Public history adds identity and review risks.
- `topology_evidence`: Three independent pull requests have separate issues, branches, tests, review loops, and GitHub receipts. Work may span upstream discussion and CI cycles.
- `escalation_triggers`: Upstream drift, lifecycle ownership changes, public API changes, unverified identity, branch-caused failures, destructive history work, or scope-changing review requests.

## Done

- Two new issues and existing #463 govern the three contributions.
- Three focused pull requests are open against current upstream `main` from Grant's public fork.
- Each pull request contains only its feature and required tests/docs.
- Reasoning controls prove both summary and expansion consumption.
- All branch-caused checks are green.
- No unresolved current-head review blocker remains.
- Issue, branch, pull request, commit, file, identity, and check receipts are recorded.
- No pull request is merged and no runtime is changed.
