# Retrieval tools reference

Use this page when you need the exact LCM tool contract or archive-migration notes. For install, activation, and runtime configuration, start with [Operator guide](operator-guide.md).

## Agent Tools

Hermes-LCM's bundled skill and active recall policy route current-session,
cross-conversation, and time-bounded questions through these tools. Use
`session_search` for Hermes-tracked history that is not present in `lcm.db`.

Recommended escalation:

- current compacted conversation: `lcm_grep` -> `lcm_describe` ->
  `lcm_expand_query`;
- cross-conversation LCM memory: `lcm_recall` -> the returned
  `lcm_load_session` or exact `lcm_expand` hint;
- recent/time-bounded recall: `lcm_recent` or time-bounded `lcm_grep`;
- exact supported operations: bounded exact evidence ->
  `lcm_evidence_pack`/`lcm_compute`.

`lcm_expand` is known-handle drill-down, not broad first-step discovery. The
canonical runtime policy is
`skills/hermes-lcm/references/recall-policy.md` and is injected only while LCM
is the active context engine.

| Tool | Use |
|------|-----|
| `lcm_grep` | Search current-session raw messages and summaries. `mode='full_text'` is the byte-compatible default; `mode='semantic'` searches embedded summaries; `mode='hybrid'` combines full-text and semantic ranks with RRF. Opt into `content_scope='externalized'` or `'both'` for bounded literal search over recoverable payload prefixes owned by the active session. Opt into `session_scope='all'` or `session_scope='session'` (with `session_id`) for bounded archive recovery over rows already present in `lcm.db`, including externally backfilled rows that may carry source strings such as `openclaw-lcm:*`; broader scopes return raw-message hits only in full-text mode and cannot search externalized payloads. Raw-message filters `role`, `time_from`, `time_to`, `source`, and `conversation_id` are pushed into the full-text query; when any is supplied, externalized payload results are omitted, and summary hits are omitted for the role/time filters so the filter contract stays exact. Use `session_search` for earlier separate sessions or broad cross-session recall. |
| `lcm_recall` | Search the agent's entire memory across ALL conversations and all time by meaning. Runs three arms over the whole local database — full-text raw messages, embedded summary KNN, and verbatim chunk KNN (all cross-session, no filter) — fuses them with RRF, dedupes chunk hits against FTS by `store_id`, and applies a soft prior `final_score = rank_score × (1 + scope_bias × is_current_conversation) × recency_boost(half_life=30d, floor 0.5)`. `scope_bias` (0..1, default 0.5) and recency are ranking BOOSTS, never filters. `include` selects `all`/`summaries`/`verbatim`. An optional voyage `rerank-2.5-lite` stage (`LCM_RERANK_ENABLED`, default off) reorders the top window of candidates AFTER the scope/recency prior, as a pure rank-reorder (voyage relevance is never spliced onto the RRF scale); any failure skips silently to RRF order. When embeddings are disabled or the vector corpora are empty the tool degrades to the full-text arm — including for `include='summaries'`, whose only vector arm is dead in that state, so a summaries request still returns full-text hits rather than nothing. Each hit carries an `expand_hint`: verbatim/current-session hits get an `lcm_expand(...)` handle, while cross-session summary hits get an `lcm_load_session(...)` handle (`lcm_expand`'s `node_id` mode is current-session only). Use `lcm_grep(mode='full_text')` for exact text in a known range and `lcm_load_session` for full transcripts. |
| `lcm_query_state` | Query the feature-flagged same-DB V4 assertion sidecar by canonical subject, optional predicate/kind/scope/speaker, and optional as-of boundary. Returns typed lifecycle state with exact message store IDs, character spans, hashes, and quotes. It preserves unresolved conflicts and never treats recency alone as supersession. |
| `lcm_compute` | Execute a question-derived, provider-neutral date/count/sum/difference/order/latest-state operation over exact raw spans or assertion IDs. Values, units, labels, keys, dates, operand order, completeness, and final wording are validated; unsupported or ambiguous inputs return an evidence-only fallback. |
| `lcm_compile_evidence` | Turn one bounded semantic proposal into an exact-source-grounded evidence brief. The proposal may name facets and operands, but product code validates refs, quotes, spans, entities, dates, values, units, distinct keys, roles, and sources; finite coverage is never accepted from the proposal alone. |
| `lcm_evidence_pack` | Build a bounded, same-`lcm.db` packet from baseline exact refs. It resolves only unique quote spans inside declared windows, validates facets and occurrence time, deduplicates refs, and may emit an immutable canonical computation trace. It returns no prose and never accepts caller-asserted open-cardinality completeness as proof. |
| `lcm_retrieve` | With `LCM_ADAPTIVE_RETRIEVAL_ENABLED=true`, coordinate one bounded retrieval episode inside the existing answerer turn. Named evidence requirements close only against exact observed refs; at most three calls to `lcm_recall`, `lcm_recent`, `lcm_query_state`, `lcm_load_session`, or `lcm_expand` are allowed. Warm reuse validates exact positive dependencies and the corpus coverage watermark. Final prose is never cached. |
| `lcm_recent` | Retrieve recent summaries with natural UTC periods. Ready temporal rollups are preferred; missing, stale, disabled, and sub-day windows transparently use leaf summaries instead. |
| `lcm_load_session` | Load one ordered raw-message transcript page for an explicit `session_id`. This is not search: it returns raw rows in `store_id` order, bounded by `limit`, with per-message content bounded by `max_content_chars`, and continues with `after_store_id` from `next_cursor`. Set `include_exact_ref=true` when rows will feed exact citation or computation; the default response stays byte-compatible. |
| `lcm_describe` | Inspect the current-session DAG or preview an `externalized_ref` without loading full content. |
| `lcm_expand` | Recover source messages, child summaries, or externalized payloads with pagination. Use `store_id` to fetch a single raw message regardless of session, suitable for drilling into a cross-session `lcm_grep` result. In `store_id` mode, `include_exact_ref=true` adds the exact returned slice without changing default bytes. |
| `lcm_expand_query` | Answer a question using expanded current-session LCM context while returning a bounded answer. |
| `lcm_status` | Show runtime health, context pressure, config, source lineage, and lifecycle stats. |
| `lcm_inspect` | Read-only operator inventory for current-session lineage, message/frontier metadata, fresh tail, externalized refs/readability, compaction skip/no-op reasons, and matched ignore/stateless patterns. It returns metadata only; use `lcm_load_session`/`lcm_expand` when you need content. |
| `lcm_doctor` | Run database, FTS, lifecycle, config, and context-pressure diagnostics. |

### Retrieval contract

LCM retrieval tools default to current-session scope. `lcm_grep` accepts
`session_scope='all'` or `session_scope='session'` as an explicit opt-in for
bounded archive search over rows already present in `lcm.db` (raw-message hits
only). Once a session id is known, `lcm_load_session` can enumerate that session's
raw transcript in chronological `store_id` pages without a search query. Use
Hermes `session_search` for broad cross-session history outside the LCM database.

`lcm_retrieve` is a policy envelope around those existing tools, not another
answering model. Its controller logic has no provider client; a dispatched
`lcm_recall` still uses its configured embedding provider and returns that stage's
bounded provenance. The answerer starts an episode with a typed identity and named
evidence slots, calls `search` only for an open slot, and explicitly assigns
returned exact citations to slots before `finish`. The controller stops after
three rounds, after any no-progress call, at 40 candidate refs, at 7,500 context
tokens, or at its bounded character limit. Unsupported tool arguments and
benchmark/reference metadata fail closed. `finish` can pass selector-supplied
operands through `lcm_compute`; values, units, citations, and final wording are
then verified by the pure engine. Enabling adaptive retrieval also binds the
same-`lcm.db` query-view store. Warm views reuse evidence and immutable
computation traces only when typed intent, requirement cardinality, exact source
dependencies, and negative-space coverage still match.

Within the current session, `source` filters raw rows directly and filters
summary nodes by descendant raw-message source lineage. `unknown` is a real
source value, not a wildcard. Legacy blank-source rows are treated as `unknown`.
`role`, `time_from`, and `time_to` are raw-message filters and are applied in the
message search query before result limiting. `time_from` and `time_to` accept Unix
seconds or timezone-aware ISO 8601 strings; naive ISO strings are rejected so the
same query means the same thing across machines. When a raw-message filter is
active, `lcm_grep` returns raw rows only and reports `summary_results_omitted`.

### Full-text, semantic, and hybrid modes

`lcm_grep` defaults to `mode='full_text'`. Omitting `mode` or setting it to
`full_text` uses the historical FTS path without changing its serialized result.
The existing `sort` argument still controls ordering inside that full-text arm;
it is separate from the retrieval `mode`.

`mode='semantic'` embeds the query with the configured provider's query path
(preserving query/document input asymmetry), then searches the current embedding
profile with cosine KNN. Semantic hits are summary nodes and retain the normal
`node_id`, depth, session, time-window, `expand_hint`, and bounded 300-character
snippet provenance. Each hit adds `score`/`cosine_score` and a
`confidence`/`confidence_band` value:

| Cosine score | Confidence |
|---:|---|
| `>= 0.65` | `high` |
| `>= 0.50` | `medium` |
| `>= 0.35` | `low` |
| `< 0.35` | `noise` |

The response-level `coverage` value comes directly from the vector store:
`full` means the complete current profile was scanned, `bounded` means the
dependency-free bounded-scan fallback was used, `full_approx` means the
two-stage binary-prescreen path reached the whole corpus but its stage-1
Hamming prescreen kept only the closest `LCM_KNN_PRESCREEN_MULTIPLIER × k`
survivors before the exact rescore — so, unlike exact-scan `full`, its top-k is
an approximate result — and `none` means no usable profile/vector coverage was
available. See the operator guide's
[vector storage scale options](operator-guide.md#vector-storage-scale-options-v3)
for how `LCM_EMBEDDING_BINARY_PRESCREEN` and the other v3 storage knobs produce
`full_approx` coverage.

Semantic and hybrid requests run under one absolute wall-clock deadline from
`embedding_query_timeout_s` (3 seconds by default), started at `lcm_grep` entry.
The same deadline covers provider resolution, query embedding, optional NumPy
import, bounded KNN, result hydration, FTS fallback, both hybrid arms, and
fusion. A disabled/missing provider or transient semantic failure degrades to
the existing full-text result when one is available, adding
`degraded_to_fts: true`, `degraded_reason`, and `coverage: 'none'`. Fallback
uses independent read-only SQLite connections with progress interruption. If
the absolute deadline is exhausted before a usable result exists, the tool
returns an explicit `timeout` error and does not start another fallback or
hybrid arm. Hybrid may still return FTS results computed before its semantic
arm timed out; this starts no new I/O after expiry. Provider authentication
failures also remain operator-readable rather than being silently hidden.

`source` first uses the SQL-bounded candidate window, then verifies descendant
source lineage within that window before ranking. This avoids corpus-sized
lineage enumeration but means source-filtered semantic coverage is explicitly
`bounded`, not a claim that source was applied before the global corpus bound.
The recursive lineage walk has its own hard work cap. If that cap is exceeded,
or a legacy DB lacks the `messages.source` column, provenance is
`unverifiable_provenance` and the filter **fails closed** rather than treating
"can't check" as "all allowed". The remaining raw-message filters are
governed by the advertised contract: `role`, `time_from`, `time_to`,
`conversation_id`, and broader `session_scope` values (`all`/`session`) all
return raw-message hits only. Because a summary node has no single role/lane and
is cross-session, the semantic arm degrades to the raw full-text path (which
enforces those filters at the message-row level and reports `degraded_to_fts`)
whenever any of them is supplied, rather than emitting summary hits that would
violate the contract. The semantic arm therefore produces summary hits only for
a current-session query with no role/time/conversation filter (`source` still
allowed).

`mode='hybrid'` runs both arms and deduplicates shared summary nodes by
`node_id`. It fuses ranks with reciprocal-rank fusion only:

```text
rrf_score = sum(1 / (60 + rank))
```

Hybrid hits surface `fts_rank` and/or `semantic_rank`, plus `rrf_score`.
Semantic hits also retain `semantic_score` and `confidence`. Both arms inspect
`min(500, max(50, limit * 3))` candidates before the public result limit is
applied. If the semantic arm fails or times out after FTS completed, hybrid
returns those already-computed FTS results with the same degradation fields;
it does not start a new fallback operation. If the FTS arm itself returns no
usable result before the deadline, the semantic arm is not started and an
explicit timeout is returned. Fusion expiry also returns an explicit timeout.
No external reranker is called.

### Weighted RRF fusion (lcm_recall)

`lcm_recall` fuses three arms — full-text raw messages, summary-vector KNN, and
chunk-vector KNN — with reciprocal-rank fusion, but each arm carries a tunable
weight so the fused ranking is never dragged below its best arm:

```text
rrf_score = sum(weight_arm / (60 + rank))
```

Naive equal-weight fusion measured **−21 R@5** against pure summary vectors on
LongMemEval: the weak FTS arm got equal say and pulled strong vector matches
down. Down-weighting FTS restores the vector-best identity to the top of a
strong-vector/weak-FTS corpus.

Weights come from `LCM_RECALL_ARM_WEIGHTS`, a lenient `arm=weight` list:

```bash
LCM_RECALL_ARM_WEIGHTS="fts=0.5,summary=1.0,chunk=1.0"
```

Defaults are `fts=0.5, summary=1.0, chunk=1.0` (conservative; the LongMemEval
harness tunes from here). Unknown arm names, malformed pairs, and
non-finite/non-numeric weights are ignored, and any arm left unspecified keeps
its default — a fully unparsable value falls back to the defaults rather than
erroring the tool. `lcm_grep`'s hybrid RRF is unaffected: it keeps implicit
`1.0` weights (byte-identical to the unweighted formula above). The weights
actually applied to the arms that ran are echoed back under
`provenance.arm_weights`.

If the summary or chunk arm ran under `coverage='bounded'` (recency-truncated
candidate scan) or `coverage='full_approx'` (two-stage binary-prescreen KNN,
see [vector storage scale options](operator-guide.md#vector-storage-scale-options-v3)),
`lcm_recall` surfaces that in its response `degraded_reason`, naming the arm and
the caveat — the same disclosure mechanism as `lcm_grep`'s `degraded_reason` —
so a caller never mistakes a truncated or approximate arm's contribution to the
fused ranking for an exhaustive exact one.

### Deterministic recall smoke evaluation

The committed offline harness builds a fixed-seed, 60-summary synthetic corpus
and evaluates 30 labeled queries across exact-term, paraphrase, and
multi-hop-ish strata. Its mock embedder produces hash-based unit vectors with
topic clustering, so it requires no network, model download, or credential:

```bash
python3 scripts/eval_retrieval_recall.py
```

The command prints recall@5 and recall@10 per mode and stratum as a Markdown
table. `--json` emits stable machine-readable output. These values are a
deterministic smoke proof, not a claim about production-provider quality.

Real-provider numbers will differ. A live rerun is deliberately double-gated
because it can make network calls and incur provider cost: configure
`LCM_EMBEDDING_PROVIDER` and `LCM_EMBEDDING_MODEL` (plus the provider credential
or local endpoint), then run:

```bash
LCM_RECALL_EVAL_ALLOW_LIVE_PROVIDER=1 \
  python3 scripts/eval_retrieval_recall.py --live-provider
```

Carried-over summary nodes can become current-session content after `/new`, but
their source eligibility still comes from the descendant raw messages. Expanding
a carried-over current-session node recovers those original raw message sources
even when the sources still belong to the previous session.

### Natural-time retrieval

`lcm_recent` accepts `today`, `yesterday`, `Nd`, `week`, `month`,
`date:YYYY-MM-DD`, and `last Nh`. All periods are normalized to UTC `[start,
end)` windows. Results are newest-first, limited to 200 sections, and bounded to
the same 20,000-character response ceiling used by the retrieval tools.

```json
{"period": "today"}
```

```json
{"period": "7d", "limit": 20}
```

```json
{"period": "date:2026-07-15", "scope": "global"}
```

When temporal rollups are enabled and `ready` rollups cover the **entire**
requested window, the response includes their ids and `ready` status in
`provenance.rollups`. If any day in the window lacks a ready rollup (missing or
stale), the feature flag is off, or the request uses a sub-day `last Nh` window,
the tool falls back for the whole window to the existing read-only leaf-summary
retrieval over the same time bounds — including retained higher-depth and
carry-forward summaries, not only current-session depth-0 leaves. This fallback
is a successful retrieval, including for an empty window; `provenance.fallback`
is `true` and no LLM call is made while serving either path. See the
[operator guide's temporal rollup operations](operator-guide.md#temporal-rollup-operations)
for enablement, tuning, status inspection, and bounded rebuild commands.

### Lossless raw recovery contract

Tool responses are bounded so one retrieval call cannot flood the main context.
Lossless recovery means raw content is stored with stable source lineage and can
be recovered in deterministic pages.

- `lcm_expand(node_id=...)` pages immediate sources with `source_offset` and `source_limit`
- `lcm_load_session(session_id=...)` pages ordered raw session rows with `after_store_id` and `next_cursor`; each row includes bounded content plus truncation metadata, and large individual rows can be recovered with `lcm_expand(store_id=...)` using `content_offset`
- oversized raw messages continue with `content_offset`
- `lcm_expand(externalized_ref=...)` pages payload content with `content_offset`
- `lcm_expand_query` uses `context_max_tokens` for auxiliary context and reports truncation/pagination hints when needed

### Searching externalized payloads

Externalized-payload search is opt-in so the default `lcm_grep` cost and result
shape remain unchanged:

```text
lcm_grep(query="distinctive error", content_scope="externalized")
lcm_grep(
  query="distinctive error",
  content_scope="both",
  externalized_refs=["20260714_...json"],
)
```

Version 1 is deliberately narrow:

- it performs case-insensitive literal matching inside payload content; history
  keeps its existing FTS5 query behavior
- it searches only payloads whose stored session id exactly matches the active
  session, so every returned ref remains recoverable with `lcm_expand`
- it scans at most 256 files and the first 512,000 encoded content-field bytes
  per file, and caps externalized result material at 64,000 response characters
- a result includes the ref, tool-call id, bounded snippet, line and byte
  position, original content byte/character sizes, and `scan_truncated`
- `externalized_refs` is optional; explicit invalid, missing, symlinked, or
  foreign-session refs are rejected rather than silently widened

Cross-session externalized search remains unsupported. Search the historical
row first, load the intended session, or recover a known current-session ref
directly with `lcm_expand`.

## Strong retrieval references V1 foundation

The V1 foundation is additive and intentionally narrower than the existing
legacy locators. It does not change any public tool schema or emit references
from tools yet; message, summary-node, sidecar, and cursor adapters are later
packets.

A migrated database has one canonical UUIDv4 authority in `metadata` under
`database_uuid`. The `database_uuid_v1` named migration creates it exactly once
for a fresh or pre-V1 database and fails closed if a completed migration later
finds a missing or malformed value. The numeric core schema remains v5.

The named `retrieval_references_v1` registry stores only the non-null SHA-256 hex
digest of a random `rh1_` + 256-bit opaque token. Target, scope, and revision
bindings are canonical JSON held server-side. V1 kinds are `message`,
`summary_node`, `externalized_payload`, `source_cursor`, `content_cursor`, and
`session_cursor`; consistency is `live`, `high_watermark`, or `snapshot`.
Every issue, resolve, revoke, and explicit-clone operation read-validates both
named migration markers plus the exact registry schema; an unmarked, orphaned,
or damaged authority fails closed rather than being adopted or repaired by an
operation.

The wire envelope is:

```json
{"v":1,"kind":"message","database_uuid":"...","token":"rh1_..."}
```

Resolution requires trusted host context and two explicit authorization boundaries. A
preauthorization callback is invoked after parsing/version validation but before
the registry row is read, so an unauthenticated or categorically unauthorized
caller cannot probe target, scope, revision, existence, or replacement content.
After lookup, a separate scope callback receives only the canonical stored scope
and must authorize that binding before target or revision JSON is decoded or
returned. Omitting either callback fails closed. `expected_scope` remains an
additional trusted-adapter constraint; it is not a substitute for host scope
authorization. Before stored-scope authorization succeeds, missing, malformed,
wrong-kind, revoked, and expired registry states are normalized to
`reference_forbidden`; detailed lifecycle and kind errors are available only
after the caller is authorized for that scope. Target and revision columns are
fetched only after those checks. Revocation uses the same two gates and does not
decode or return target/revision data. Issuance requires a separate trusted
operation callback over canonical kind, target, scope, and revision before
registry publication. The caller must not emit the envelope before its
transaction is durable. Explicit clone rotation requires its own administrative
callback and is an operator/database-boundary operation, not a user reference
action. The stable
strong error codes are `invalid_request`, `reference_invalid`,
`reference_unsupported_version`, `reference_database_mismatch`,
`reference_kind_mismatch`, `reference_scope_mismatch`, `reference_forbidden`,
`reference_not_found`, and `reference_stale`; structured failures retain a
human-readable `error` field.

Ordinary restart and byte-for-byte restore preserve the authority UUID and
registry. A replacement database has a different UUID. A raw byte copy remains
indistinguishable from a restore until an explicit clone operation rotates the
UUID and atomically revokes copied V1 rows. This foundation does not claim that
legacy scalar IDs are safe, and it provides no principal/tenant, signing/MAC,
key-management, target loading, revision adapter, or pagination behavior.

### lossless-claw/OpenClaw import utility

`hermes-lcm` includes an opt-in operator script for backfilling raw message rows from a lossless-claw/OpenClaw LCM SQLite database into the local hermes-lcm SQLite store:

```bash
python scripts/import_lossless_claw.py \
  --source-db ~/.openclaw/path/to/lcm.db \
  --target-db ~/.hermes/lcm.db \
  --agent sammy
```

The script is intentionally conservative:

- dry-run is the default; pass `--apply` to write
- run it against an explicit target DB path, preferably while Hermes is stopped for that profile
- writes create a timestamped target DB backup first when the target already exists
- only raw messages are imported; summary DAG import is out of scope
- imported rows keep explicit provenance in `session_id` and `source`, for example `openclaw-lcm:agent:sammy:<source-session>`
- the default provenance identity is the concrete source `conversations.session_id`, preserving source session boundaries even when many conversations share one `session_key`
- pass `--session-identity session_key` only when you intentionally want conversations with the same source session key grouped into one imported LCM session
- reruns are idempotent for the same `--import-id`; the default `import_id` is path-derived, so pass a stable `--import-id` if you may import the same copied DB from different paths
- changing `--agent`, `--namespace`, or `--session-identity` under the same `--import-id` is treated as the same import and will skip already-tracked source messages; use a new `--import-id` for a different mapping
- no OpenClaw config or separate secret tables are imported, but raw transcripts and tool payloads are imported and may contain sensitive user data

This is a local archive migration path. It does not make LCM a general memory provider, and it does not change the current-session retrieval contract for agent tools.

## Message retrieval-reference V1 adapter

The narrow `hermes_lcm.message_references` adapter binds one existing
`messages.store_id` to a V1 opaque `kind="message"` envelope. It accepts a
caller-supplied canonical scope and computes a SHA-256 semantic revision over
normalized session/source/conversation/role/content/tool metadata and the
finite timestamp. `pinned` and `token_estimate` are operational metadata and do
not affect the revision.

Use `issue_message_reference(...)` with the trusted host
`authorization_context`, mandatory `authorize_target` callback, and
`authorize_issue` callback. Target preauthorization runs before any transaction
or `messages` lookup, so denied existing and missing IDs are indistinguishable;
foundation issue authorization then validates the loaded target/revision before
registry insertion. Use `resolve_message_reference(...)` with both foundation
authorization callbacks. Resolution authorizes the opaque envelope and stored
scope before selecting the message row, and verifies the current semantic
revision in one stable read snapshot. Issuance uses one `BEGIN IMMEDIATE`
transaction when the caller has no active transaction; caller-owned transactions
remain rollbackable. The adapter returns only the foundation's closed V1 error
codes and does not add public tool emission, pagination, session cursors,
summary-node/sidecar references, principal/tenant policy, signing, or schema
changes.

## Related references

- [Embeddings setup — free and local options](./embeddings-setup.md) — provider configuration for the `semantic` / `hybrid` modes

- [Operator guide](operator-guide.md)
- [Architecture notes](architecture.md)
- [Benchmarking and stress checks](../benchmarks/README.md)
