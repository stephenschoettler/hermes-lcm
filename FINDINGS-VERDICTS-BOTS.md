# PR #184 bot findings — validation and verdicts

Scope: the 34 raw bot comments in `pr184_comments.json`, deduplicated into the
20 semantic rows required by `SPEC-BOTS.md`. Every raw comment ID is mapped
below. The nine High rows are 1–9.

Binding contract: `coverage` continues to describe candidate reach. A resident
scan that scores every candidate therefore remains `coverage="full"`.
`KNNResult.scoring` is additive: resident/quantized scoring reports
`int8_quantized`; exact float32 scoring reports `float32_exact`.

| Row | Priority | Raw comment IDs | Terminal disposition | Validation and result |
|---:|---|---|---|---|
| 1 | High | 3673526888 | Fixed now | Reproduced that resident summary loads bypassed suppression. Resident count/load queries now join `summary_nodes` and exclude `suppressed_at IS NOT NULL` when that column exists; a warm-cache suppression regression test passes. |
| 2 | High | 3673526894 | Fixed now | Reproduced the preallocated count/load overrun seam. Resident construction now rejects `end > count`, falls back to exact scoring, and rechecks profile version plus live count before publishing. |
| 3 | High | 3673526904 | Fixed now | Reproduced orphan chunk reach after deleting its message. Resident count/load queries now join `messages`; warm resident entries are invalidated when live cardinality changes. |
| 4 | High | 3674122396, 3674137273 | Fixed now | Applied the binding coverage/scoring decision to `KNNResult`, summary KNN, chunk KNN, docstrings, tests, and benchmark output. No coverage value was added or changed for resident reach. |
| 5 | High | 3674426257, 3674426269, 3674426671, 3674426674 | Fixed now | Validated budget aliasing and cross-budget eviction. Registry keys include database path, data version, identity, and budget; stale purge/LRU enforcement is confined to the matching budget partition and cannot evict the protected warm key. |
| 6 | High | 3674137283, 3674465153 | Fixed now | Validated that array bytes alone undercount resident memory and concurrent cold callers duplicate work. Entry sizing now includes Python list/item metadata, and a per-registry-key build lock with post-wait recheck produces one cold build. |
| 7 | High | 3674426659, 3674465147 | Fixed now | Reproduced the shared `:memory:` registry namespace. Every in-memory store now receives a UUID-backed private namespace and drops its entries/build locks on close. |
| 8 | High | 3673536547, 3673536580, 3674465157 | Fixed now | Reproduced loss of candidate recency order during streamed loads. An ordered temporary candidate table carries `candidate_ordinal`, and vector rows are emitted in that order; truncated scans now retain newest-first semantics. |
| 9 | High | 3673536556, 3674465164 | Fixed now | Reproduced the host-parameter failure with SQLite's variable limit lowered to 8. Candidate IDs are inserted into the temporary table with bounded `executemany`; no corpus-sized `IN (...)` clause remains. |
| 10 | Non-High | 3673526909, 3674426278 | Fixed now | Cold resident construction no longer reports loaded-but-unscored rows on a deadline; it falls back to exact deadline-aware scoring. Warm resident ranking reports completed batch boundaries. Existing int8 deadline and pooled float32 cache behaviors both pass. |
| 11 | Non-High | 3673536562 | Already fixed at starting HEAD | Validated that data-version changes purge stale resident entries before lookup/publication. No additional change was required. |
| 12 | Non-High | 3673536506, 3673536587 | Fixed now | Removed the inert resident-budget config from the test that overrides bytes directly. The duplicate-block refactor comment was already addressed and did not reproduce a supported-path failure, so no further refactor was added. |
| 13 | Non-High | 3673505461 | Fixed now | Added legacy int8 decode parity proof: resident and retained exact arithmetic return identical IDs and scores within `1e-7`. |
| 14 | Non-High | 3673505456 | Fixed now | Benchmark seeding now writes `archived=0` explicitly and output distinguishes first-query, page-cache-hot timing. The script was run successfully against both float32 and int8 profiles. |
| 15 | Non-High | 3674122408, 3674426264 | Fixed now | Restored gold-vector exclusion assertions for max-row, time-budget, and absolute-deadline truncation tests. |
| 16 | Non-High | 3673526897, 3674161057 | Fixed now | Benchmark setup removes the database plus WAL/SHM before seeding, making the same work directory rerunnable; two consecutive smoke runs completed. |
| 17 | Non-High | 3673536473 | Fixed now | Renamed the benchmark's misleading `cold_ms` metric to `first_query_ms` and documents that filesystem page cache may already be warm. |
| 18 | Non-High | 3673536476 | Fixed now | Benchmark CLI rejects `warm_runs < 1`; first and warm calls must report full reach, the requested result count, and matching top-k IDs. |
| 19 | Non-High | 3673536512 | Fixed now | Added exact top-result assertions to the resident summary test and exact/resident identity and score parity checks for int8. |
| 20 | Non-High | 3673536501, 3674161048 | Fixed now | Benchmark metadata now records git SHA, Python/NumPy versions, warm runs, batch/scan settings, resident use, and scoring; crossover sizes include 2,499 and 2,500 rows. |

Parity result: 20/20 semantic rows and 34/34 raw comment IDs have a terminal
disposition. Focused contract tests pass. The CI-replica suite, with one
unrelated concurrency test run separately, has zero new or missing failure
names relative to the recorded 35-failure baseline.
