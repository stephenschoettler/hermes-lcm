# PR #182 bot-finding verdicts

Reviewed PR: `100yenadmin/hermes-lcm#182`

Reviewed head: `95d20d0384db203a3a673b6890bce6939b4f829c`

Base: `cb92bf40c1d4c862cb56090792113b69d88660d4`

Mode: validate-then-fix, address existing feedback only. The fail-closed
migration posture is binding. No commit, push, PR reply, or thread resolution
was performed.

## 1. Mixed legacy digests with an active profile

- Source: CodeRabbit discussion `3673283174`
- Reported priority: Major / digest summary High
- Verdict: verified; fixed in the local working tree.
- Validation: a legacy table with active profile `A` and stored digests
  `{A, B}` migrated without raising and rewrote every row to `A`.
  `validate-before-fix.xml` records the failing regression.
- Fix: migration now adopts the active digest only when every stored legacy
  digest matches it. Any foreign digest fails closed before table replacement.
- Proof: `test_legacy_state_embedding_migration_refuses_mixed_active_rows`
  passes and verifies the legacy primary-key table remains intact after the
  rejected migration.

## 2. Ambiguous migration recovery text

- Source: Codex discussion `3673282194`
- Reported priority: P2 / digest summary Medium
- Verdict: verified; fixed in the local working tree.
- Validation: schema migration blocks `build_state_semantic_index()` before a
  rebuild can start, while the old error said only to rebuild.
  `validate-before-fix.xml` records the message mismatch.
- Fix: the fail-closed error now documents the recovery explicitly: discard
  the ambiguous legacy state embeddings and run a fresh backfill.
- Proof:
  `test_legacy_state_embedding_migration_refuses_ambiguous_inactive_rows`
  requires the fresh-backfill recovery text and verifies rollback preserves the
  legacy primary-key table.

## 3. Rebuild-guard TOCTOU

- Source: CodeRabbit discussion `3673283186`
- Reported priority: Minor / digest summary Medium
- Verdict: verified; fixed in the local working tree.
- Validation: the guard probe observed `Connection.in_transaction == False`
  on the reviewed head. `validate-before-fix.xml` records the failure.
- Fix: the active-profile read and same-profile rebuild guard now run under
  `self._lock`, after `BEGIN IMMEDIATE`. A rejection follows the existing
  exception path and rolls back the transaction.
- Proof:
  `test_forced_same_profile_rebuild_refuses_to_mutate_serving_rows` observes
  the guard inside the transaction, confirms the transaction is closed after
  rejection, and confirms serving rows and rankings remain unchanged.

## 4. Test helper staging-table name reuse

- Source: CodeRabbit discussion `3673283168`
- Reported priority: Trivial / digest summary Low
- Verdict: verified; fixed in the local working tree.
- Validation: `_downgrade_state_embeddings_to_legacy_primary_key` used the
  production migration table name
  `lcm_trajectory_state_embeddings_profile_scoped`.
- Fix: the test helper now uses
  `lcm_trajectory_state_embeddings_legacy_stage` for rename, copy, and drop.
- Proof: the complete state-semantic migration test file passes.

## Validation

- Failing-first evidence: `validate-before-fix.xml` — 3 expected failures.
- Bot regression subset: `focused-bot-regressions.xml` — 3 passed.
- Migration regressions: `migration-regressions.xml` — 30 passed.
- Full CI replica: `full-suite.xml` — 2711 passed, 35 failed, 1 skipped,
  12 xfailed. The 35 failure names exactly match the prior lane baseline:
  zero new and zero missing.
- Ruff on both changed source files: passed.
- `git diff --check`: passed.

Reproduce the checked-in validation evidence with:

```bash
python3 -m pytest tests/test_trajectory_state_semantic_expansion.py -q
python3 -m pytest tests/test_trajectory_store*.py -q
ruff check trajectory_store.py tests/test_trajectory_state_semantic_expansion.py
git diff --check
```

Raw command logs are retained off-repo by the operator.

Proof boundary: this proves the local working-tree fixes and exact baseline
parity in the named CI-replica environment. It does not prove a pushed head,
current-head remote CI, merge readiness, merge, release, deployment, or runtime
adoption.

## Round 2

Reviewed head: `5a9259b5f53f04bd91b747c8be9bf420aec30b7c`

| Raw comment ID | Terminal disposition | Fix | Reproducible validation evidence |
| --- | --- | --- | --- |
| `3674826213` | Verified; fixed in the local working tree. | The `resume=False` same-profile guard derives the requested digest from the active state profile's dimension under `BEGIN IMMEDIATE`, before any provider probe. The refusal test makes query probing fail if reached and asserts zero query calls. | `python3 -m pytest tests/test_trajectory_state_semantic_expansion.py -q` |
| `3674826226` | Verified; fixed in the local working tree. | Cutover now flips predecessor and target flags with one ordered upsert statement: the predecessor is cleared before the target is activated, preserving the one-active unique index. The distinct-profile test traces the cutover and requires exactly one active-flag mutation statement. | `python3 -m pytest tests/test_trajectory_state_semantic_expansion.py -q` |
| `3674838648` | Verified; fixed in the local working tree. | Added the positive distinct-profile `resume=False` test. It observes the prior staged row set deleted before fresh persistence, the new profile active, and the predecessor inactive with its rows unchanged. | `python3 -m pytest tests/test_trajectory_state_semantic_expansion.py -q` |
| `3674880065` | Verified; fixed in the local working tree. | Removed the machine-local evidence path, published the exact regeneration commands, and recorded that raw logs remain off-repo. | `rg -n '/Vo[l]umes' FINDINGS-VERDICTS-BOTS.md` |

Round 2 validation:

```bash
python3 -m pytest tests/test_trajectory_state_semantic_expansion.py -q
# 31 passed
python3 -m pytest tests/test_trajectory_store*.py -q
# 15 passed
ruff check trajectory_store.py tests/test_trajectory_state_semantic_expansion.py
# All checks passed
git diff --check
# no output
```

The local full-suite CI-replica command could not reach test execution because
the checkout does not contain CI's generated `agent.context_engine` stub; it
stopped during collection with 22 import errors. The acceptance fallback above
therefore uses the focused file plus `tests/test_trajectory_store*.py`.

Proof boundary: Round 2 proves the four binding fixes in the local working tree
and the named focused fallback checks. It does not prove a pushed head, remote
CI, merge readiness, merge, release, deployment, or runtime adoption.

## Round 3

Reviewed head: `367e9b4`

| Raw comment ID | Terminal disposition | Fix | Reproducible validation evidence |
| --- | --- | --- | --- |
| `3676895819` | Verified; fixed in the local working tree. | `_load_state_semantic_matrix()` now holds the store lock across its freshness and row reads, so a same-connection reader waits for the legacy table replacement transaction instead of observing the table between `DROP` and rename. The regression pauses migration immediately after the legacy table is dropped and requires the reader to return the expected serving state IDs only after migration resumes. | `python3 -m pytest tests/test_trajectory_state_semantic_expansion.py -q` |
| `3676895821` | Verified; fixed in the local working tree. | Each embedding persistence transaction now rechecks that its target profile is inactive before upserting. The overlapping-builder regression holds a slow builder after embedding, lets a fast builder activate the shared target, then requires the slow builder to abort without changing any serving row. | `python3 -m pytest tests/test_trajectory_state_semantic_expansion.py -q` |

Round 3 validation:

```bash
python3 -m pytest tests/test_trajectory_state_semantic_expansion.py -q
# 33 passed
python3 -m pytest tests/test_trajectory_store*.py -q
# 15 passed
ruff check trajectory_store.py tests/test_trajectory_state_semantic_expansion.py
# All checks passed!
git diff --check
# no output
```

Proof boundary: Round 3 proves the two reported concurrency defects are fixed
in the local working tree, the Round 2 pre-probe refusal and single-statement
cutover tests remain green, and the named focused fallback checks pass. It does
not prove a pushed head, remote CI, merge readiness, merge, release, deployment,
or runtime adoption.

## Round 4 — two codex P2 on head 9750790

| Row | Priority | Raw comment IDs | Terminal disposition | Validation and result |
|---:|---|---|---|---|
| R4-1 | P2 | 3677048438 | Fixed now | The pre-probe guard prefers the caller's resolved `dim` and infers from the serving profile only when the requested dimension is unknown — a distinct-dimension rebuild for the same provider/model is no longer falsely refused. Focused suite green. |
| R4-2 | P2 | 3677048435 | Declined in-PR; filed as a fork issue | Full-backfill leasing hardens against two same-declared-identity providers returning DIFFERENT vectors — a provider-contract violation — on a default-off path. The round-3 recheck already prevents post-cutover rewrites (the serving-data hazard). Leasing is real hardening but out of proportion for this train; tracked as its own issue. |
