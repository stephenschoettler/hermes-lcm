# V2 re-baseline batch report

- Branch: `batch/v2-rebaseline`
- Baseline: `e5acbbf26715a1f2e721abcdf7d180c5dd1d8d32`
- Binding spec: `SPEC-V2-REBASELINE-BATCH.md`
- Binding architecture decisions: `fork/docs/program-architecture:bench/DECISIONS-R3-UPSTREAM-ARCH.md`
- Upstream follow-up sources: `a53276cf896280fab183cfcc811c9fd5bb49c522`, `ae0f06847247aeb8abb5b7a055bdd14bc0039ff7`
- Git mutation: none; all changes remain unstaged in the working tree.

## Validate-then-port table

Each upstream hunk was checked against the current `fork/main` baseline before editing. No fix was already present or superseded in full.

| Source | Fix | Current-main validation | Disposition |
|---|---|---|---|
| `a53276c` | `evidence_compiler.py`: finite coverage counts distinct grounded `exact_ref` values | Current code still used `len(validated)`, so duplicate claims over one grounded ref could certify coverage. | **ported** |
| `a53276c` | `query_view_store.py`: hit CAS checks generation/rowcount and re-reads readiness | Current hit update had no generation predicate or post-update readiness confirmation. | **ported** |
| `a53276c` | `adaptive_retrieval.py`: persisted slot refs intersect the selected evidence set | Current manifest persisted every resolved slot ref, including unselected evidence. | **ported** |
| `a53276c` | `trajectory_store.py`: reject an existing incompatible trajectory schema before repair DDL | Current constructor ran `_init_schema()` before validating the stored trajectory schema version. | **ported** |
| `a53276c` | `vector_store.py`: deadline-expired paths do not start `COUNT(*)` | The original sites were still missing the fix, while #184 had added resident and streaming paths with the same invariant. | **ported**, adapted to current resident, binary-prescreen, candidate-enumeration, streaming, summary, and chunk paths |
| `ae0f068` | `adaptive_retrieval.py`: `mark_failed` cleanup is best-effort | Current exception handler allowed a cleanup failure to replace the original build failure. | **ported** |

## Unit results

All commands below use repository-relative test paths and run from the repository
root. `PYTHONPATH=.` makes the commands portable in a clean checkout without
operator-local environment variables.

### Unit 1: six upstream follow-up fixes

Command:

```text
PYTHONPATH=. python3 -m pytest -q tests/test_adaptive_retrieval.py::test_persisted_slot_refs_include_only_selected_evidence tests/test_adaptive_retrieval.py::test_query_view_cleanup_cannot_replace_the_build_failure tests/test_evidence_compiler.py::test_duplicate_grounded_ref_cannot_certify_finite_coverage tests/test_query_view_store.py::test_hit_confirmation_rechecks_generation_after_source_mutation tests/test_trajectory_store.py::test_newer_trajectory_schema_is_rejected_before_fts_repair tests/test_vector_store.py::test_full_scan_budget_stops_early_and_reports_bounded tests/test_vector_store.py::test_full_scan_budget_includes_candidate_enumeration tests/test_vector_store.py::test_full_scan_absolute_deadline_stops_between_batches tests/test_vector_store.py::test_resident_deadline_does_not_start_a_count_query tests/test_prescreen_flip_blackout.py::test_deadline_bounds_a_synced_binary_summary_prescreen tests/test_int8_two_stage_knn.py::test_chunk_deadline_bounds_a_synced_binary_prescreen
```

Result: **11 passed**.

### Unit 2: D-ARCH-1 event-dedupe certification

Command:

```text
PYTHONPATH=. python3 -m pytest -q tests/test_evidence_contract.py::test_finite_enumeration_distinguishes_same_entity_events_by_date tests/test_evidence_contract.py::test_finite_enumeration_returns_dated_and_undated_count_uncertified tests/test_evidence_contract.py::test_finite_enumeration_collapses_repeated_undated_mentions
```

Result: **3 passed**.

### Unit 3: D-ARCH-2 adjacency-reserve backfill

Command:

```text
PYTHONPATH=. python3 -m pytest -q tests/test_trajectory_store.py::test_unused_adjacency_reserve_backfills_the_full_ranked_limit tests/test_trajectory_store.py::test_partial_adjacency_reserve_backfills_in_rank_order tests/test_trajectory_store.py::test_full_adjacency_reserve_keeps_the_existing_composition
```

Result: **3 passed**.

### Unit 4: D-ARCH-3 date-anchor trust boundary

Command:

```text
PYTHONPATH=. python3 -m pytest -q tests/test_reasoning.py::test_compute_accepts_caller_anchor_only_when_it_agrees_with_sidecar tests/test_reasoning.py::test_compute_sidecar_overrides_disagreeing_caller_anchor tests/test_reasoning.py::test_compute_without_sidecar_marks_temporal_result_low_trust
```

Result: **3 passed**.

### Unit 5: batch-2 `how long ago` operand fix

Command:

```text
PYTHONPATH=. python3 -m pytest -q tests/test_reasoning.py::test_planner_uses_explicit_cardinality_and_interval_units
```

Result: **1 passed**. The regression proves that `how long ago` plans one evidence-date operand plus the question-date anchor.

## Affected-file validation

Spec acceptance files:

```text
PYTHONPATH=. python3 -m pytest -q tests/test_trajectory_store.py tests/test_evidence_contract.py tests/test_reasoning.py
```

Result: **81 passed**.

Upstream-port affected test files:

```text
PYTHONPATH=. python3 -m pytest -q tests/test_adaptive_retrieval.py tests/test_evidence_compiler.py tests/test_query_view_store.py tests/test_vector_store.py tests/test_prescreen_flip_blackout.py tests/test_int8_two_stage_knn.py
```

Result: **166 passed**.

Changed-file lint:

```text
ruff check adaptive_retrieval.py evidence_compiler.py query_view_store.py reasoning.py requirements_compiler.py tools.py trajectory_store.py vector_store.py tests/test_adaptive_retrieval.py tests/test_evidence_compiler.py tests/test_evidence_contract.py tests/test_int8_two_stage_knn.py tests/test_prescreen_flip_blackout.py tests/test_query_view_store.py tests/test_reasoning.py tests/test_trajectory_store.py tests/test_vector_store.py
```

Result: **all checks passed**.

## Acceptance-count chronology

The acceptance suite is the same three-file command throughout this report; its
count grew only when review rounds added regressions to those files:

- Initial batch and the post-CI-fix rerun: **81 passed**.
- Round 2: **85 passed** (**+4** Round-2 regression tests).
- Round 3: **91 passed** (**+6** Round-3 regression tests).
- Round 4: **92 passed** (**+1 net** acceptance test: the prior all-unavailable
  case remains covered under its updated semantics, and a new mixed
  available/unavailable case was added).
- Round 5: **94 passed** (**+2** acceptance tests: explicit-date certification
  without a sidecar and collision resistance for distinct overlong same-date
  event keys).
- Round 6: **99 passed** (**+5** acceptance cases: one low-trust finite-count
  certification regression, three `calendar` unit variants, and one bounded
  caller-session-date schema/note regression).
- Round 6.2 pre-Round-7 head: **100 passed** (**+1** explicitly negated
  finite-event regression).
- Round 7: **105 passed** (**+5** cases: missing-`observed_at` fallback,
  event-bound negation, observed-available/future-event exclusion, and two
  leap-day anniversary dates; the open-cardinality disclosure test was
  strengthened without adding a case).

The Round-4 final-batch deadline regression is in `tests/test_vector_store.py`,
so it increases the upstream-affected suite from **166** to **167** without
changing the three-file acceptance-suite count. The Round-5 no-NumPy
final-batch regression and corrected query-view stale-branch regression increase
that suite to **168**.

## PR #190 CI fix

The reported CI command was reproduced locally before editing. It failed with
the expected six failures: one chunk-vector coverage-total regression, four
stale composition-policy expectations, and one stale state-semantic expansion
expectation.

- `vector_store.py`: scalar and vectorized scans now return deadline expiry
  separately from other bounded-stop causes. Message and chunk paths skip
  `COUNT(*)` only after actual deadline/budget expiry; an unscorable live vector
  remains bounded while still reporting the corpus total. The existing
  `test_exact_scan_does_not_overstate_coverage_for_unscorable_live_vector`
  contract was not modified.
- `tests/test_trajectory_composition_policies.py`: the default, Policy A,
  Policy D, and hybrid tests now cite D-ARCH-2 and assert that default backfill
  re-admits the lexical winner while each policy still promotes or protects it
  through its own mechanism.
- `tests/test_trajectory_state_semantic_expansion.py`: the full-pool test now
  cites D-ARCH-2 and pins the deterministic 14-state composition: incumbent
  rank order is preserved and the newly admitted semantic state fills the
  unused reserve.

Focused CI command:

```text
PYTHONPATH=. python3 -m pytest tests/test_chunk_vector_store.py tests/test_trajectory_composition_policies.py tests/test_trajectory_state_semantic_expansion.py -q
```

Result after the fix: **60 passed**.

The batch acceptance command above was rerun after the fix: **81 passed**.
The upstream-port affected-file command above was also rerun: **166 passed**.

CI-fix lint:

```text
ruff check vector_store.py tests/test_trajectory_composition_policies.py tests/test_trajectory_state_semantic_expansion.py
```

Result: **all checks passed**.

## Round 2 review dispositions

Review mode: **address**, Round 2 delta only. Candidate identity:
PR `#190`, branch `batch/v2-rebaseline`, local base/head before this unstaged
batch `e5acbbf26715a1f2e721abcdf7d180c5dd1d8d32` /
`cdc95c13e688cfa1b7af714c4d3a9661a65cf0a4`. No Git mutation or GitHub write
was performed.

| Comment ID | Priority | Disposition |
|---|---|---|
| `3679976787` | P1 | **fixed** — finite-scan candidates are availability-checked before event-date handling, real host `observed_at` takes precedence over a sidecar as the availability boundary, and grounding always receives the historical `as_of` when present. Regression: an undated event observed after the question date is excluded and cannot produce a count. |
| `3680021268` | Major | **fixed** — `_validate_existing_schema_version` returns only when `lcm_trajectory_corpora` is absent; malformed existing tables propagate `sqlite3.OperationalError`. Regression proves a missing `schema_version` column raises before any FTS rebuild. |
| `3679973891` | P2 | **fixed** — the grounding/operand pairing now uses `zip(..., strict=True)`, so cardinality drift fails instead of truncating silently. |
| `3679976780` | P2 | **fixed** — `_finite_event_key` reserves the 300-character budget for the date suffix. Regression proves an over-300-character base yields distinct keys for two dates and preserves both suffixes. |
| `3679976783` | P2 | **fixed** — temporal certification now requires the sidecar anchor to parse successfully. A malformed sidecar uses D-ARCH-3's absent/unusable-sidecar cell: `anchor_trust=low_trust`, `temporal_certified=false`; it does not retain `engine_sidecar` trust. |
| `3679973895` | P3 | **documented only** — comments at both certification sites name caller-evidence `exact_ref` uniqueness and engine finite-scan `dedupe_key` uniqueness and state that the surfaces intentionally differ. |
| `3680021253` | Minor | **superseded by the Round-6 portability fix** — all commands now use repository-relative `PYTHONPATH=.` and require no operator-local environment variable. |

Round-2 proof:

- Focused Round-2 regressions: **8 passed**.
- Exact CI slice: **60 passed**.
- Batch acceptance suite: **85 passed**.
- Upstream-affected suite: **166 passed**.
- Changed-file `ruff check`: **all checks passed**.
- `git diff --check`: **clean**.

Delivery checkpoint: **COMPLETE** for the named local Round-2 disposition gate;
**ADVANCE** to the orchestrator handoff. This does not claim remote exact-head
CI, merge readiness, merge, release, or runtime proof for the unstaged delta.

## Round 3 review dispositions

Review mode: **address**, binding Round-3 disposition batch. Candidate identity:
PR `#190`, branch `batch/v2-rebaseline`, local base/head before this unstaged
batch `e5acbbf26715a1f2e721abcdf7d180c5dd1d8d32` /
`0c8bea9a19e032b3b1261a8ff3d4931458228d92`. Each finding body was read through
the requested per-comment `gh api` route before implementation. No Git mutation
or GitHub write was performed.

| Comment ID | Priority | Disposition |
|---|---|---|
| `3680071381` | P1 | **fixed** — an uncertified finite count now injects an explicit `UNCERTIFIED` disclosure that says an undated contributor prevents exhaustive certification for the requested window and instructs the answerer to preserve that disclosure. Regression proves a dated+undated mix carries the warning while a fully dated set does not. |
| `3680071370` | P2 | **fixed** — adjacency-reserve backfill now consults and updates the same per-trajectory count used by nucleus and adjacency selection. Regression creates a backfill candidate that would become the third hit under `diversity_cap=2` and proves it is not admitted. |
| `3680071376` | P2 | **fixed** — `how long ago` planning extracts explicitly requested days, weeks, months, or years; execution supports complete years; and a question with no unit uses the coarsest non-zero complete unit. Answer-level regressions cover all four explicit units and the no-unit path. |
| `3680077589` | P2 | **fixed** — the effective scan stop remains bounded by `budget_s`, but `deadline_expired` now identifies only the caller's absolute deadline. Summary and chunk scan paths preserve `total` after budget-only scan or enumeration expiry, while absolute-deadline paths still avoid `COUNT(*)`. The F44 change is limited to stop-cause classification and the two mirrored consumers. |
| `3679973901` | P3 | **documented and verified** — the selector-stage write site now names the `None` = not applicable, `False` = uncertified, `True` = certified contract and forbids bool coercion. Repo-wide consumer grep found only direct JSON serialization; no dashboard or serializer coerces the value. Regression proves the non-temporal selector stage serializes `None` as JSON `null`, distinct from the existing uncertified `false` coverage. |

Round-3 focused regressions:

```text
PYTHONPATH=. python3 -m pytest -q tests/test_evidence_contract.py::test_finite_enumeration_distinguishes_same_entity_events_by_date tests/test_evidence_contract.py::test_finite_enumeration_returns_dated_and_undated_count_uncertified tests/test_trajectory_store.py::test_adjacency_reserve_backfill_preserves_diversity_cap tests/test_reasoning.py::test_how_long_ago_answer_honors_explicit_unit tests/test_reasoning.py::test_how_long_ago_answer_without_unit_uses_coarsest_fit tests/test_reasoning.py::test_public_compute_tool_reports_stages_and_discards_mutated_candidate tests/test_vector_store.py::test_full_scan_budget_stops_early_and_reports_bounded tests/test_vector_store.py::test_full_scan_budget_includes_candidate_enumeration
```

Result: **11 passed**.

Round-3 acceptance proof:

- Exact CI slice above: **60 passed**.
- Batch acceptance suite above: **91 passed**.
- Upstream-affected suite above: **166 passed**.
- Changed-file `ruff check` (the explicit command above) plus the Round-6/CI-exercised test files (`tests/test_chunk_vector_store.py`, `tests/test_trajectory_composition_policies.py`, `tests/test_trajectory_state_semantic_expansion.py`, verified separately): **all checks passed**.
- `git diff --check`: **clean**.

Delivery checkpoint: **COMPLETE** for the named local Round-3 disposition gate;
**ADVANCE** to the orchestrator handoff. This does not claim remote exact-head
CI for the unstaged delta, merge readiness, merge, release, or runtime proof.

## Round 4 review dispositions

Review mode: **address**, binding Round-4 disposition batch. Candidate identity:
PR `#190`, branch `batch/v2-rebaseline`, local base/head before this unstaged
batch `e5acbbf26715a1f2e721abcdf7d180c5dd1d8d32` /
`76ba0f695f37e55ee73047a957d6814da33ff29c`. Each finding body was read through
the requested per-comment `gh api` route before implementation. No Git mutation
or GitHub write was performed.

| Comment ID | Priority | Disposition |
|---|---|---|
| `3680166653` | P2 | **fixed** — `_scan_vectorized_ranked` now marks caller-deadline expiry only when the scan is genuinely truncated. A deadline that trips on the final batch leaves the scan complete; summary and chunk streaming consumers report the completed `scanned` and `total` values. Regression proves a two-batch corpus returns `coverage=full`, `scanned=4`, and `total=4` when the clock crosses the deadline during the final batch. |
| `3680166656` | P2 | **fixed** — unavailable-as-of clauses remain excluded before event dating, but no longer abort the remaining finite enumeration. The remaining clauses are counted and certified under D-ARCH-1; `unavailable_as_of_clauses` records the separate exclusion. Regressions prove a mixed corpus counts only the available event and an all-unavailable corpus returns no computation while preserving the exclusion count. |
| `3680154851` | Major | **fixed in this report** — the acceptance chronology now states the exact per-round progression: 81 initial/post-CI, 85 after Round 2, 91 after Round 3, and 92 after Round 4. |

Round-4 focused regressions:

```text
PYTHONPATH=. python3 -m pytest -q tests/test_vector_store.py::test_full_scan_deadline_on_final_batch_reports_complete_total tests/test_evidence_contract.py::test_finite_enumeration_counts_available_and_excludes_postdated_event tests/test_evidence_contract.py::test_finite_enumeration_all_unavailable_returns_no_count
```

Result: **3 passed**.

Round-4 acceptance proof:

- Exact CI slice above: **60 passed**.
- Batch acceptance suite above: **92 passed**.
- Upstream-affected suite above: **167 passed**.
- Changed-file `ruff check` (the explicit command above) plus the Round-6/CI-exercised test files (`tests/test_chunk_vector_store.py`, `tests/test_trajectory_composition_policies.py`, `tests/test_trajectory_state_semantic_expansion.py`, verified separately): **all checks passed**.
- `git diff --check`: **clean**.

Delivery checkpoint: **COMPLETE** for the named local Round-4 disposition gate;
**ADVANCE** to the orchestrator handoff. This does not claim remote exact-head
CI for the unstaged delta, merge readiness, merge, release, or runtime proof.

## Round 5 review dispositions

Review mode: **address**, binding Round-5 disposition batch. Candidate identity:
PR `#190`, branch `batch/v2-rebaseline`, local base/head before this unstaged
batch `e5acbbf26715a1f2e721abcdf7d180c5dd1d8d32` /
`430a48df17acfa27436d3a4e7b4c001c0e51f7ae`. Each finding body was read through
the requested per-comment `gh api` route before implementation. No commit,
push, PR write, or other Git/GitHub mutation was performed.

| Comment ID | Priority | Disposition |
|---|---|---|
| `3680300339` | P1 | **fixed** — answer-ready recall leaves `occurrence_time` in the exact pre-batch compute-operand shape and publishes `anchor_trust`, `temporal_certified`, `session_date_overridden`, and any `trust_note` in a sibling `temporal_trust` block. The wire regression passes that exact recalled occurrence object into `lcm_compute`, proves a computed result, and proves the object is unchanged across the seam. |
| `3680300336` | P2 | **fixed** — trust classification now follows the resolved occurrence type. Explicit dates are certified without a sidecar; relative resolutions use the D-ARCH-3 valid, absent, and invalid sidecar cells; an unrelated/unknown occurrence is not certified merely because a sidecar exists. Regressions cover explicit-date/no-sidecar certification and relative/no-sidecar low trust. |
| `3680300341` | P2 | **fixed** — the scalar no-NumPy summary and chunk scan consumers now mirror the vectorized completed-scan provenance: when the deadline crosses during the final scored batch, coverage remains `full` with `scanned` and `total`. The forced-no-NumPy regression proves `full`, `4/4`. |
| `3680283571` | P2 | **fixed in the test** — the query-view regression now advances corpus generation through `MessageStore.append` after positive-dependency validation and immediately before the confirmation snapshot. It asserts the exact negative-space watermark stale reason, `delta_required`, stale view state, unchanged published generation, and zero hits. |
| `3680283573` | P3 | **fixed** — overlong finite-event key bases retain a bounded readable prefix plus a SHA-256 prefix before the date suffix. Distinct long-named events on the same date no longer collide; the 300-character key and date suffix contracts remain intact. |

Round-5 focused regressions:

```text
PYTHONPATH=. python3 -m pytest -q tests/test_lcm_recall.py::test_recalled_occurrence_round_trips_as_unchanged_compute_operand tests/test_reasoning.py::test_explicit_occurrence_without_sidecar_is_certified tests/test_reasoning.py::test_compute_without_sidecar_marks_temporal_result_low_trust tests/test_reasoning.py::test_malformed_sidecar_date_is_low_trust_and_uncertified tests/test_vector_store.py::test_full_scan_deadline_on_final_batch_without_numpy_reports_complete_total tests/test_query_view_store.py::test_hit_confirmation_rechecks_generation_after_source_mutation tests/test_evidence_contract.py::test_finite_event_key_hashes_distinct_overlong_bases_on_same_date
```

Result: **7 passed**.

Round-5 acceptance proof:

- Exact CI slice above: **60 passed**.
- Batch acceptance suite above: **94 passed**.
- Upstream-affected suite above: **168 passed**.
- Full `tests/test_lcm_recall.py` affected suite: **102 passed**.
- Changed-file `ruff check` (the explicit command above) plus the Round-6/CI-exercised test files (`tests/test_chunk_vector_store.py`, `tests/test_trajectory_composition_policies.py`, `tests/test_trajectory_state_semantic_expansion.py`, verified separately): **all checks passed**.
- `git diff --check`: **clean**.

Delivery checkpoint: **COMPLETE** for the named local Round-5 disposition gate;
**ADVANCE** to the orchestrator handoff. This does not claim remote exact-head
CI for the unstaged delta, merge readiness, merge, release, or runtime proof.

## Round 6 final full-round dispositions

Review mode: **address**, binding Round-6 final full-round disposition batch.
Candidate identity: PR `#190`, branch `batch/v2-rebaseline`, local/remote head
before this unstaged batch `849a26be61119e105515bba27c944a75df7f8c41`.
Every finding body was read through the requested per-comment `gh api` route.
No commit, push, PR write, review reply/resolution, or other Git/GitHub mutation
was performed.

| Comment ID | Priority | Disposition |
|---|---|---|
| `3680255017` | P1 | **fixed** — finite enumeration now passes the engine trust sidecar into grounding and requires both a dated and non-low-trust operand set for certification. A relative event resolved only from host `observed_at` remains counted but sets `every_counted_event_trusted=false`, `finite_coverage=false`, and the existing uncertified reason/disclosure. |
| `3680255014` | P2 | **fixed** — `how long ago, in calendar days/weeks/months` preserves the explicitly requested unit. The answer-level parameterization covers all three calendar forms. |
| `3680255021` | P2 | **fixed** — persisted computation traces allow and store the bounded `temporal_trust` sibling. The low-trust compute → publish → warm-replay regression proves the cached view exposes the same `{status, certified, notes}` block as the fresh compute. |
| `3680255026` | P2 | **fixed** — caller `occurrence_time.session_date` text is truncated to 64 characters before inclusion in a trust note, and the compute schema now declares `maxLength: 64`. |
| `3680273764` | Minor | **fixed in this report** — every replay command uses repository-relative `PYTHONPATH=.` and no operator environment variable. |
| `3680273767` | Major | **fixed** — all six message/chunk scan result sites now use one `_scanned_knn_result` helper for deadline-aware `scanned`/`total` gating; the existing bounded/full/deadline suites plus the new chunk mirror preserve behavior. |
| `3680378478`, `3680438106` | P2 + Major | **fixed** — lookup now takes a fresh corpus snapshot inside the hit-confirmation write transaction before incrementing the hit. The regression mutates only after the initial negative-space snapshot, asserts a branch-entry marker, reaches the second snapshot, and receives the branch-specific `corpus advanced during hit confirmation` result with a stale view and zero hits. |
| `3680378485` | P2 | **fixed** — the unconditional bare assert is gone. Calendar endpoint validation is an explicit fallback scoped to `auto`, `month`, and `year`; day/week execution retains the prior narrow path and is safe under `python -O`. |
| `3680378489` | P3 | **fixed** — one `temporal_trust_wire` helper emits the same bounded `{status, certified, notes}` shape for `lcm_compute`, `lcm_recall`, and persisted replay. The recall-to-compute wire test compares the two public blocks directly. |
| `3680438117` | Trivial | **fixed** — both summary final-batch regressions use an explicit `AssertionError`-raising COUNT stub, and the mirrored `knn_chunks` regression proves a completed two-batch deadline crossing reports `full`, `scanned=4`, `total=4` without calling COUNT. |

Round-6 focused regressions:

```text
PYTHONPATH=. python3 -m pytest -q tests/test_evidence_contract.py::test_relative_finite_event_without_sidecar_is_counted_but_uncertified tests/test_reasoning.py::test_how_long_ago_answer_honors_explicit_unit tests/test_reasoning.py::test_caller_session_date_is_bounded_in_schema_and_trust_note tests/test_adaptive_retrieval.py::test_low_trust_computation_persists_and_replays_temporal_trust tests/test_lcm_recall.py::test_recalled_occurrence_round_trips_as_unchanged_compute_operand tests/test_query_view_store.py::test_hit_confirmation_rechecks_generation_after_source_mutation tests/test_vector_store.py::test_full_scan_deadline_on_final_batch_reports_complete_total tests/test_vector_store.py::test_full_scan_deadline_on_final_batch_without_numpy_reports_complete_total tests/test_chunk_vector_store.py::test_chunk_deadline_on_final_batch_reports_total_without_count
```

Result: **15 passed**.

Round-6 acceptance proof:

- Exact CI slice: **61 passed**.
- Batch acceptance suite: **99 passed**.
- Upstream-affected suite: **169 passed**.
- Full `tests/test_lcm_recall.py` suite: **102 passed**.
- Changed-file `ruff check` (the explicit command above) plus the Round-6/CI-exercised test files (`tests/test_chunk_vector_store.py`, `tests/test_trajectory_composition_policies.py`, `tests/test_trajectory_state_semantic_expansion.py`, verified separately): **all checks passed**.
- `git diff --check`: **clean**.

Delivery checkpoint: **COMPLETE** for the named local Round-6 final full-round
disposition gate; **ADVANCE** to the orchestrator handoff. All ten deduplicated
findings are fixed locally. This does not claim remote exact-head CI for the
unstaged delta, merge readiness, merge, release, deployment, or runtime proof.

## Round 7 review dispositions

Review mode: **address**, binding Round-7 disposition batch. Candidate identity:
PR `#190`, branch `batch/v2-rebaseline`, local/remote head before this unstaged
batch `cbdd4cc2b158c099407e5737c9011e7e11ecb574`. Every finding body was read
through the requested per-comment `gh api` route. No commit, push, PR write,
review reply/resolution, checkout, or other Git/GitHub mutation was performed.

| Comment ID | Priority | Disposition |
|---|---|---|
| `3680940804`, `3680911089`, `3680911881` | P1 + Major + P2 | **fixed together** — negation is scoped to the counted event assertion. `no`/`none`/`zero`/`without` reject only when bound to the requested unit or its event action; `not`/`never` and the full `didn't`/`don't`/`doesn't`/`haven't`/`hasn't` forms (including curly apostrophes and apostrophe-stripped `didn t`-style forms) reject only when directly attached to an event action. Bare `don`/`haven`/other stems are gone. Regressions preserve rejection of the Round-6.2 negated event and admit `Don`, unrelated `no`/`without` details, `Havre`, and `Haven`. |
| `3680911885` | P3 | **fixed** — a relative event with neither `observed_at` nor a benchmark session date now falls through the pre-Round-2 availability semantics. It remains counted under D-ARCH-1 and is explicitly uncertified; a present future `observed_at` still excludes through the epoch path. |
| `3680911888` | P3 | **fixed** — the resolved `event_day` future-date guard now runs before every availability-signal path can return. A regression gives a future event an already-available `observed_at` and proves the event is excluded. |
| `3680911886` | P3 | **fixed in the test** — the open-cardinality regression now asserts the uncertified reason, both certificate trust/date flags, `finite_coverage=false`, and the Round-3 `UNCERTIFIED` injected-context disclosure. |
| `3680940808` | P2 | **fixed** — completed-year comparison clamps the anniversary day to the last valid day of the target February. `2020-02-29` to both `2021-02-28` and `2021-03-01` returns one completed year. |

Round-7 focused regressions:

```text
PYTHONPATH=. python3 -m pytest -q tests/test_evidence_contract.py::test_open_cardinality_returns_one_event_count_uncertified tests/test_evidence_contract.py::test_relative_finite_event_without_observed_at_is_counted_but_uncertified tests/test_evidence_contract.py::test_source_event_negation_binds_to_event_action_or_counted_unit tests/test_evidence_contract.py::test_finite_enumeration_excludes_future_event_with_available_observed_at tests/test_evidence_contract.py::test_finite_enumeration_rejects_explicitly_negated_undated_events tests/test_evidence_contract.py::test_finite_enumeration_counts_available_and_excludes_postdated_event tests/test_reasoning.py::test_how_long_ago_years_clamps_leap_day_anniversary
```

Result: **8 passed**.

Round-7 acceptance proof:

- Exact CI slice: **61 passed**.
- Batch acceptance suite: **105 passed**.
- Upstream-affected suite: **169 passed** with the isolated checkout's
  `/Volumes/LEXAR/hermes-work/ci-stub` added to `PYTHONPATH`; the literal
  `PYTHONPATH=.` command could not collect `test_adaptive_retrieval.py` or
  `test_query_view_store.py` because this checkout has no sibling `agent`
  package. No symlink or other harness file was created.
- Full `tests/test_lcm_recall.py` suite under the same CI-stub import
  prerequisite: **102 passed**.
- Changed/acceptance/CI-file `ruff check`: **all checks passed**.
- `git diff --check`: **clean**.

Delivery checkpoint: **COMPLETE** for the named local Round-7 disposition gate;
**ADVANCE** to the orchestrator handoff. All seven comment IDs have the binding
local fix or test disposition. This does not claim remote exact-head CI for the
unstaged delta, merge readiness, merge, release, deployment, or runtime proof.

## Decision fidelity

- D-ARCH-1: dated candidates dedupe by the existing event key plus resolved date; undated candidates retain the existing collapse. Counts with any undated contributor are returned with `finite_coverage=false`.
- D-ARCH-2: adjacency remains first priority for its reserve; unused slots return to ranked candidates in rank order.
- D-ARCH-3: explicit dates are self-anchoring and certified without a sidecar; relative occurrences use the engine occurrence-date sidecar as the trusted anchor, disagreement overrides the caller and is noted, and absence or invalidity returns a low-trust, uncertified result. Finite counts still include low-trust relative events but cannot certify them. Trust metadata is a sibling of the operand-shaped `occurrence_time` object and has the same public wire shape on compute, recall, and replay.
- Batch-2: `how long ago` uses one evidence-date operand and the question-date anchor.
- Deviations from the binding D-ARCH text: **none**.

## Proof boundary

This report proves the unstaged working-tree implementation, focused unit behavior, affected test-file results, and changed-file lint only. It does not prove merge, remote CI, paired V2 benchmark results, V1 sanity-slice results, release, deployment, or runtime adoption.
