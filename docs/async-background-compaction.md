# Opt-in async/background compaction with atomic publish

Design spike for preparing old stable chunks off the turn-critical path while keeping current LCM behavior unchanged unless explicitly enabled.

Refs:

- Lossless Claw #807 — prepare incremental summaries in the background and publish atomically
- Lossless Claw #942 — live config must beat stale persisted thresholds/debt
- Lossless Claw #902 — summary failure/backoff must not wedge compaction debt forever

## Problem

Today `LCMEngine.compress()` does the expensive work synchronously: ingest, select the oldest raw backlog outside the fresh tail, call the summarizer, write canonical DAG nodes, optionally condense, then assemble the active context. That preserves the important cache-friendly property: active context changes only at threshold/full-sweep boundaries. The downside is foreground latency, especially with slow local summarizers or serial leaf chains.

The safe target is not “write summaries in another thread and flip a boolean.” The target is a two-phase lifecycle:

1. **Prepare** non-canonical summaries for old, stable raw-message chunks in the background.
2. **Promote** a complete, still-valid batch atomically when normal foreground compaction would have run.

Until promotion, active context, search, recall, expansion, transcript GC, and doctor integrity checks behave as if the prepared summaries do not exist.

## Non-goals for the first slice

- No default behavior change.
- No mandatory background thread in normal installs.
- No prebuilt condensation layers in the first slice. Leaf-only promotion is already useful and is easier to prove atomic.
- No replacement for foreground compaction. If prepared work is absent, incomplete, stale, or invalid, foreground compaction falls back to today’s path.
- No persisted threshold override that can win over live config. Persisted metadata is evidence to validate against live policy, not policy itself.

## Proposed flags

Add config fields, all disabled by default:

| Field | Env | Default | Meaning |
| --- | --- | ---: | --- |
| `async_background_compaction_enabled` | `LCM_ASYNC_BACKGROUND_COMPACTION_ENABLED` | `false` | Enables the feature surface. |
| `async_background_compaction_worker_enabled` | `LCM_ASYNC_BACKGROUND_COMPACTION_WORKER_ENABLED` | `false` | Allows automatic background preparation. Tests and hosts may still call one-shot prep manually when the feature is enabled. |
| `async_background_compaction_max_batches` | `LCM_ASYNC_BACKGROUND_COMPACTION_MAX_BATCHES` | `2` | Backpressure cap per conversation. |
| `async_background_compaction_retry_backoff_seconds` | `LCM_ASYNC_BACKGROUND_COMPACTION_RETRY_BACKOFF_SECONDS` | `300` | Cooldown after summary failures. |

The enable flag should guard all writes to the new tables and all promotion attempts. Reader filters must still be robust if old pending rows exist after the flag is later disabled.

## Storage model

Add tables separate from canonical `summary_nodes`:

### `compaction_batches`

One row per prepared generation.

Suggested columns:

- `batch_id TEXT PRIMARY KEY`
- `conversation_id TEXT NOT NULL`
- `session_id TEXT NOT NULL`
- `state TEXT NOT NULL` — `pending`, `preparing`, `ready`, `promoting`, `promoted`, `rejected`, `failed`, `superseded`
- `frontier_start_store_id INTEGER NOT NULL`
- `frontier_end_store_id INTEGER NOT NULL`
- `fresh_tail_count INTEGER NOT NULL`
- `leaf_chunk_tokens INTEGER NOT NULL`
- `policy_fingerprint TEXT NOT NULL`
- `summary_route_fingerprint TEXT NOT NULL`
- `source_coverage_hash TEXT NOT NULL`
- `expected_leaf_count INTEGER NOT NULL`
- `prepared_leaf_count INTEGER NOT NULL DEFAULT 0`
- `failure_count INTEGER NOT NULL DEFAULT 0`
- `next_retry_at REAL`
- `last_error TEXT`
- `created_at REAL NOT NULL`
- `updated_at REAL NOT NULL`
- `promoted_at REAL`
- `rejected_reason TEXT`

Indexes:

- `(conversation_id, state, created_at)`
- `(session_id, state, created_at)`
- `(next_retry_at, state)`

### `pending_summary_nodes`

Prepared but non-canonical leaf summaries.

Suggested columns:

- `pending_id TEXT PRIMARY KEY`
- `batch_id TEXT NOT NULL REFERENCES compaction_batches(batch_id)`
- `conversation_id TEXT NOT NULL`
- `session_id TEXT NOT NULL`
- `depth INTEGER NOT NULL DEFAULT 0`
- `summary TEXT NOT NULL`
- `token_count INTEGER NOT NULL`
- `source_token_count INTEGER NOT NULL`
- `source_ids TEXT NOT NULL`
- `source_identity_hashes TEXT NOT NULL`
- `source_range_start_store_id INTEGER NOT NULL`
- `source_range_end_store_id INTEGER NOT NULL`
- `previous_pending_ids TEXT NOT NULL DEFAULT '[]'`
- `created_at REAL NOT NULL`
- `earliest_at REAL`
- `latest_at REAL`
- `expand_hint TEXT DEFAULT ''`

Indexes:

- `(batch_id, source_range_start_store_id)`
- `(conversation_id, state)` is intentionally on the batch table; pending nodes should not have independent canonical state.

Do **not** add pending rows to `summary_nodes`. Reusing the canonical table with a lifecycle flag would make every reader, FTS query, and integrity check a footgun. Keeping pending rows in a separate table gives active-only defaults naturally.

## Fingerprints and validation inputs

A prepared batch is valid only for the exact policy and source frontier it was created for.

### Policy fingerprint

Hash a normalized JSON object of compaction policy inputs that affect chunking or active-context semantics:

- schema/protocol version, e.g. `async_compaction_protocol_v1`
- `fresh_tail_count`
- `leaf_chunk_tokens`
- `context_threshold` / effective preflight threshold policy
- `dynamic_leaf_chunk_enabled`
- `dynamic_leaf_chunk_max`
- `ignore_message_patterns` plus their source
- sensitive-pattern config that changes stored/summarized text
- large-output externalization settings that change serializer input
- `custom_instructions`
- `l2_budget_ratio`, `l3_truncate_tokens`
- `incremental_max_depth` only if the batch later supports pending condensation

### Summary route fingerprint

Hash the effective summarizer contract:

- `summary_model`
- `summary_fallback_models`
- provider/model route after parsing, if available
- summarizer timeout class only if it changes produced summaries or failure policy
- plugin version/protocol version

A model/config change should not necessarily delete pending rows immediately, but promotion must reject rows whose fingerprints no longer match live config.

### Source coverage hash

For every source row in the prepared range, hash a canonical tuple:

```text
store_id | session_id | conversation_id | role | content_sha256 | tool_call_id | tool_calls_sha256 | tool_name | timestamp
```

Promotion validates both:

- the ordered set of `store_id`s is exactly what the batch claims;
- the identity hash for every row still matches.

This catches transcript reconciliation, late ingest of missing rows inside the range, externalization/GC rewrites, and accidental ordinal drift.

## Preparation lifecycle

Background preparation should operate only on raw messages that are outside the fresh tail at preparation time.

1. Resolve live config and compute the compactable prefix using the same filtering as foreground compaction.
2. Choose the oldest chunk(s) up to the current publishable frontier.
3. Create or resume a `pending`/`preparing` batch for that frontier.
4. Generate leaf summaries into `pending_summary_nodes`, optionally using previous pending summaries from the same batch as continuity context.
5. Mark the batch `ready` only when `prepared_leaf_count == expected_leaf_count` and every expected range is covered exactly once.

Preparation must not mutate:

- `summary_nodes`
- active replay context
- compaction count
- frontier markers
- transcript GC state
- generated placeholder ordinals

## Atomic promotion

Foreground compaction remains the owner of canonical changes. When `should_compress_preflight()` says compaction is needed, `compress()` may attempt prepared promotion before doing synchronous summarization.

Promotion runs in one SQLite transaction on a single connection, using `BEGIN IMMEDIATE` so foreground/background writers serialize.

The promotion path should not call existing helpers that perform their own commits on separate SQLite connections. Canonical node inserts, lifecycle frontier updates, batch state changes, and superseding older batches must share one transaction boundary. In-process markers such as `_last_compacted_store_id` should update only after the transaction commits.

Validation inside the transaction:

1. Feature flag still enabled.
2. Batch is `ready` for this `conversation_id` and `session_id`.
3. Live policy fingerprint equals batch policy fingerprint.
4. Live summary route fingerprint equals batch summary route fingerprint.
5. Current lifecycle frontier equals the batch’s expected start frontier.
6. Fresh-tail boundary still permits promoting the full prepared range. If the boundary moved such that only a prefix is safe, reject the batch for v1 rather than partially publish.
7. Source rows for the claimed range still exist, are ordered, and match every source identity hash.
8. Pending nodes cover the full source range without gaps or overlaps.
9. No canonical `summary_nodes` already cover the same source IDs. This handles a foreground compaction race that won before promotion.

Publish steps in the same transaction:

1. Insert canonical `summary_nodes` copied from pending rows.
2. Advance lifecycle frontier to the promoted end store id.
3. Mark the batch `promoted` with `promoted_at`.
4. Mark older pending/ready batches for the same conversation `superseded`.
5. Commit.

Only after commit may the engine assemble active context from canonical summaries. If any validation step fails, mark the batch `rejected` with a reason in a transaction that does **not** insert canonical summaries or advance the frontier, then fall back to today’s foreground compaction path.

## Race semantics

### Foreground compaction wins first

If a synchronous foreground pass inserts canonical nodes and advances the frontier while a background batch is pending, later promotion sees either frontier mismatch or canonical source overlap. It rejects/supersedes the stale batch and leaves the foreground result intact.

### Background ready wins first

Foreground promotion takes `BEGIN IMMEDIATE`, validates current source/frontier/fingerprints, publishes, then assembles. A concurrent background preparer trying to write the same batch waits or fails with normal SQLite busy behavior and must reload batch state before continuing.

### Background failure/backoff

Summary failure increments `failure_count`, stores a compact `last_error`, and sets `next_retry_at`. It must not create compaction debt that blocks foreground recovery. Foreground compaction can always ignore failed/pending work and use the current synchronous path.

### Restart

On startup/session bind:

- `promoting` from a crashed transaction should not be visible as canonical unless the transaction committed. If the batch row says `promoting` but no canonical nodes/frontier were advanced, mark it `rejected` or return it to `ready` after validation.
- `preparing` older than a timeout becomes `pending` for retry, or `failed` if backoff policy says so.
- `ready` batches are left ready, but promotion still revalidates live config and source coverage.
- Pending rows are never used by active context during recovery.

SQLite transaction atomicity should mean there is no half-published active state. Recovery still needs explicit cleanup of stale lifecycle labels so status/doctor is trustworthy.

## Reader and diagnostics rules

Active readers default to canonical rows only:

- `lcm_grep` summary search ignores pending rows.
- `lcm_expand(node_id=...)` cannot expand pending IDs through the canonical node path.
- `lcm_describe` active DAG overview excludes pending rows.
- transcript GC eligibility ignores pending rows.
- doctor active-context integrity checks ignore pending rows unless checking async health specifically.

Add an explicit async section to status/doctor instead:

```json
"async_compaction": {
  "enabled": false,
  "worker_enabled": false,
  "pending_batches": 0,
  "preparing_batches": 0,
  "prepared_batches": 0,
  "promoted_batches": 0,
  "rejected_batches": 0,
  "failed_batches": 0,
  "superseded_batches": 0,
  "pending_summaries": 0,
  "oldest_pending_age_seconds": null,
  "last_rejected_reason": null,
  "last_error": null
}
```

Doctor should warn, not fail, for normal disabled state. It should warn on:

- stale `preparing` batches beyond recovery timeout;
- ready batches whose live policy fingerprint no longer matches;
- failed batches whose backoff has expired but no worker has retried;
- pending rows whose batch is missing.

## Implementation sequence

1. **Acceptance tests first** for stale rejection, config change rejection, foreground/background race, summary failure/backoff, restart recovery, successful atomic promotion, pending invisibility, and status/doctor counts.
2. Schema only: create the two tables and reader filters, with feature disabled and no behavior change.
3. Manual one-shot preparer behind the flag, no automatic worker yet.
4. Atomic promotion path in `compress()` before foreground summarization, with fallback on any reject.
5. Status/doctor async counts.
6. Optional worker loop with backpressure and retry policy.
7. Later: pending condensed layers, if leaf-only promotion leaves too much foreground work.

## Test matrix

These are mirrored in `tests/test_async_background_compaction_design.py` as xfailed RED spike tests until the implementation exists.

| Test | Proves |
| --- | --- |
| `test_pending_summaries_are_invisible_until_atomic_promotion` | Pending rows do not affect active assembly/search/status counters except async diagnostics. |
| `test_atomic_promotion_rejects_stale_source_identity` | Transcript reconciliation or row rewrite invalidates the batch without canonical mutation. |
| `test_atomic_promotion_rejects_live_config_change` | Live config/policy wins over persisted prepared metadata. |
| `test_foreground_compaction_race_supersedes_pending_batch` | Foreground compaction and background promotion cannot double-compact or mix generations. |
| `test_summary_failure_backoff_does_not_wedge_foreground_compaction` | Failed background prep backs off but foreground synchronous compaction remains available. |
| `test_restart_recovers_or_discards_pending_batches_safely` | Restart never makes pending rows canonical and cleans stale lifecycle states. |
| `test_default_disabled_async_compaction_is_inert` | Default-off config performs no background preparation and reports zero async counts. |
| `test_atomic_promotion_rejects_summary_route_change` | Model/route changes reject stale prepared summaries before canonical mutation. |
| `test_atomic_promotion_rejects_live_threshold_policy_change` | Live threshold policy changes beat persisted prepared metadata. |
| `test_successful_atomic_promotion_is_all_or_nothing` | Canonical node insert, frontier advance, and batch promotion commit together. |
| `test_atomic_promotion_rolls_back_partial_publish_failure` | A mid-promotion failure leaves no canonical node/frontier/batch half-state. |
| `test_status_and_doctor_report_async_compaction_counts` | Operators see pending/prepared/promoted/rejected/failed counts. |

## Open questions

- Should v1 reject a ready batch when only a prefix is still publishable, or support prefix promotion? Recommendation: reject in v1. Prefix promotion makes continuity and expected leaf counts more complex.
- Should automatic workers live inside `LCMEngine`, a plugin lifecycle helper, or a host-managed scheduler? Recommendation: start with a manual one-shot preparer and make the automatic worker a later slice.
- Should route fingerprint include fallback model order? Recommendation: yes. Different fallback order can change output after partial failures.
- Should summary timeout changes reject prepared work? Recommendation: no unless timeout policy changes the output contract; include route/model/policy version, not operational timing knobs.
