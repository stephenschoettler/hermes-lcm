# Operator guide

This page holds the detailed install, activation, configuration, diagnostics, and slash-command reference that used to live in the top-level README. The README stays focused on first-run adoption; this file is the operator reference.

## Requirements

- Hermes Agent with the pluggable context engine slot ([PR #7464](https://github.com/NousResearch/hermes-agent/pull/7464))
- Python 3.11+
- No required third-party runtime dependencies. `tiktoken` is used if available; otherwise LCM falls back to character-based token estimates. `regex` is used if available to apply timeouts to message ignore patterns; if it is not installed, message-level regex filtering is disabled with a warning rather than running unbounded stdlib `re` matches.

## Install

Canonical install path: clone `hermes-lcm` as a general user plugin.

```bash
git clone https://github.com/stephenschoettler/hermes-lcm \
  ~/.hermes/plugins/hermes-lcm
```

For a profile-specific install:

```bash
git clone https://github.com/stephenschoettler/hermes-lcm \
  ~/.hermes/profiles/myprofile/plugins/hermes-lcm
```

From an existing checkout, install a symlink:

```bash
./scripts/install.sh
# Optional profile-aware install:
HERMES_PROFILE=myprofile ./scripts/install.sh
```

The installer exposes both the plugin checkout and the bundled
`skills/hermes-lcm` package in the matching global/profile skill tree. It is
safe to run from a checkout already cloned into the canonical plugin path and
refuses a conflicting plugin or skill path before creating either link.

## Activate

The plugin has two names:

- plugin manifest name: `hermes-lcm`
- runtime context engine name: `lcm`

Both must be configured:

```yaml
plugins:
  enabled:
    - hermes-lcm

context:
  engine: lcm
```

Restart Hermes after changing plugin or context-engine config.

## Update

If you cloned directly into the plugin directory:

```bash
cd ~/.hermes/plugins/hermes-lcm && git pull --ff-only
```

For a profile-specific install:

```bash
cd ~/.hermes/profiles/myprofile/plugins/hermes-lcm && git pull --ff-only
```

If you installed a symlink from a separate checkout:

```bash
./scripts/update.sh
```

Restart Hermes after updating.

## Upgrade from v0.20.0 to v0.21.0-rc2

1. While the old runtime is running, run `/lcm backup`. If Hermes or any other
   SQLite writer may still be running, this is the only supported online backup
   path.
2. Alternatively, stop Hermes and every other process that can write the
   database. After all writers are fully stopped, copy the profile's `lcm.db`
   plus any existing `lcm.db-wal` and `lcm.db-shm` companions together as one
   quiescent snapshot. Do not copy these files separately while a writer is
   live.
3. Update the plugin checkout to the RC and restart Hermes.
4. Send one normal message, then confirm `lcm_status` reports plugin version
   `0.21.0-rc2` and the expected database path.
5. For a migration-shape audit, query that database with
   `SELECT value FROM metadata WHERE key = 'schema_version';`; the expected
   result is `5`.

No manual core migration, data import, or embedding backfill is required. An
existing v0.20 database opens in place and remains on core schema version 5.
The 0.21 assertion, query-view, and trajectory families use additive named
feature markers and create their tables only when the corresponding store or
workflow is invoked. A stock/default-off upgrade therefore creates none of
those optional tables. If you later enable a 0.21-only store, treat the
pre-upgrade backup as the downgrade path rather than opening that modified
database with an older plugin.

Temporal rollup settings do not change. When rollups are enabled, maintenance
now runs through bounded eventual background work instead of blocking session
start. A busy or full maintenance queue defers work to a later opportunity;
`lcm_recent` continues to fall back to bounded leaf summaries while rollups are
missing or stale.

## Verify

Run:

```bash
hermes plugins
```

Expected signals:

- plugin list includes `hermes-lcm`
- selected context engine is `lcm`
- tool list includes all 15 schemas: `lcm_grep`, `lcm_recall`,
  `lcm_query_state`, `lcm_compute`, `lcm_compile_evidence`,
  `lcm_evidence_pack`, `lcm_retrieve`, `lcm_recent`, `lcm_load_session`,
  `lcm_describe`, `lcm_expand`, `lcm_expand_query`, `lcm_status`, `lcm_inspect`,
  and `lcm_doctor`
- ordinary skill discovery includes `hermes-lcm`; plugin-qualified explicit
  loading is `hermes-lcm:hermes-lcm` on hosts that support plugin skills

Typical output:

```text
Plugins (1):
  ✓ hermes-lcm v0.21.0-rc2 (15 tools)

Provider Plugins:
  Context Engine: lcm
```

For source checkouts, `lcm_status`, `/lcm status`, `lcm_inspect`,
`lcm_doctor`, and `/lcm doctor` also report the loaded plugin path and
best-effort git identity:
`plugin_git_commit`, `plugin_git_branch`, and `plugin_git_dirty`.

The product-owned recall policy is injected through the host's ephemeral
`pre_llm_call` user-context seam only after an LCM engine is bound to the
session. It does not modify the system prompt or activate when another context
engine is serving the turn. Its canonical file and digest source is
`skills/hermes-lcm/references/recall-policy.md`.

## Troubleshooting

### `hermes plugins` shows `lcm (not found)` but LCM tools exist

If `plugins.enabled` contains `hermes-lcm`, `context.engine: lcm` is set, and
the runtime exposes LCM tools, LCM is loaded. The `lcm (not found)` line is a
Hermes host discovery/status mismatch, not an LCM storage or compaction failure.

### Startup log mentions `context-engine schemas` or `Path B fallback`

This is expected on older Hermes hosts that do not advertise
`context_engine_tool_handlers_receive_messages`, including Hermes Agent v0.16.
LCM tools are still available through the context-engine schema/dispatch path
(Path B). The plugin intentionally avoids standalone plugin-registry tool
registration (Path A) on those hosts because Path A would shadow Path B and lose
current-turn ingest.

Healthy signals are the same as above: selected context engine `lcm`, all 15
`lcm_*` tools in the live tool list, and `lcm_status` / `lcm_inspect` / `lcm_doctor` responding
after one normal message initializes the session.

### `/lcm status` looks unbound after restart

After a fresh Hermes restart, `/lcm status` may show `session_id: (unbound)` or
`threshold_tokens: (uninitialized)`. Send one normal Hermes message first, then
run `lcm_status` or `/lcm status` again for live per-session fields.

## Configuration

Most installs only need `plugins.enabled` and `context.engine: lcm`. Useful
environment variables:

| Variable | Default | Use |
|----------|---------|-----|
| `LCM_CONTEXT_THRESHOLD` | `0.35` | Fraction of the context window that triggers LCM compaction |
| `LCM_ABSOLUTE_THRESHOLD_TOKENS` | `0` | If `> 0`, force compaction at this absolute prompt-token count instead of `context_length × LCM_CONTEXT_THRESHOLD`. Cross-model context-health setpoint (common coding default: `130000`) so large windows do not delay compaction and degrade recall |
| `LCM_FRESH_TAIL_COUNT` | `32` | Recent messages protected from compaction |
| `LCM_FRESH_TAIL_MAX_TOKENS` | `0` | Optional token cap for the protected fresh tail (`0` disables it); always retains the newest message and complete assistant/tool-result groups |
| `LCM_INCREMENTAL_MAX_DEPTH` | `3` | Max DAG condensation depth (`-1` = unlimited, `0` = leaf only); enables hierarchical summarization |
| `LCM_LEAF_CHUNK_TOKENS` | `20000` | Raw-backlog floor before leaf compaction; with dynamic chunking enabled, the base chunk target |
| `LCM_DYNAMIC_LEAF_CHUNK_ENABLED` | `false` | Enable chunk-sized leaf compaction passes instead of compacting the whole non-tail raw backlog per pass |
| `LCM_DYNAMIC_LEAF_CHUNK_MAX` | `40000` | Upper bound for dynamic leaf chunk targets |
| `LCM_THRESHOLD_FULL_SWEEP_ENABLED` | `false` | At threshold, opt into one synchronous bounded sweep that drains chunked raw history before publishing one new active context |
| `LCM_SUMMARY_PREFIX_TARGET_TOKENS` | `0` | Sweep-only summary-frontier target; `0` derives one `LCM_LEAF_CHUNK_TOKENS` budget |
| `LCM_NEW_SESSION_RETAIN_DEPTH` | `2` | DAG depth retained after manual `/new` (`-1` all, `0` none) |
| `LCM_IGNORE_SESSION_PATTERNS` | empty | Comma-separated session globs excluded from LCM storage |
| `LCM_STATELESS_SESSION_PATTERNS` | empty | Comma-separated session globs kept read-only |
| `LCM_IGNORE_MESSAGE_PATTERNS` | empty | Comma-separated regex patterns; matching message content (plain text, extracted text parts for structured/multimodal content, or normalized JSON fallback when no text parts exist) is excluded from LCM storage |
| `LCM_SENSITIVE_PATTERNS_ENABLED` | `false` | Opt in to deterministic redaction before LCM storage, FTS indexing, summarization, active replay, and externalized ingest payloads |
| `LCM_SENSITIVE_PATTERNS` | `api_key,bearer_token,password_assignment,private_key` | Comma-separated named sensitive pattern catalog entries to apply when redaction is enabled |
| `LCM_LARGE_OUTPUT_EXTERNALIZATION_ENABLED` | `false` | Store oversized ingest payloads, including tool results, media blocks, and generic raw content, in plugin-managed JSON files |
| `LCM_LARGE_OUTPUT_EXTERNALIZATION_THRESHOLD_CHARS` | `12000` | Externalization threshold for normalized payload text |
| `LCM_LARGE_OUTPUT_ACTIVE_REPLAY_STUBBING_ENABLED` | `false` | Replace token-heavy textual tool results with recoverable externalized refs in active replay; current-turn ingest is immediate and historical assembly respects the protected fresh tail; requires large-output externalization |
| `LCM_LARGE_OUTPUT_ACTIVE_REPLAY_STUB_THRESHOLD_TOKENS` | `25000` | Token-aware threshold for active-replay tool-result stubbing |
| `LCM_LARGE_OUTPUT_TRANSCRIPT_GC_ENABLED` | `false` | Rewrite already-externalized summarized tool rows to compact placeholders |
| `LCM_CRITICAL_BUDGET_PRESSURE_RATIO` | `0.0` | Disabled at `0.0`; when set, permits critical-pressure bypasses for bounded deferred catch-up and cache-friendly follow-on condensation only |
| `LCM_SUMMARY_MODEL` | auxiliary | Override summarization model |
| `LCM_SUMMARY_FALLBACK_MODELS` | empty | Comma-separated summarization models tried after `LCM_SUMMARY_MODEL` or the auxiliary task default fails |
| `LCM_SUMMARY_CIRCUIT_BREAKER_FAILURE_THRESHOLD` | `2` | Consecutive failed summarization calls before a route is skipped temporarily |
| `LCM_SUMMARY_CIRCUIT_BREAKER_COOLDOWN_SECONDS` | `300` | Seconds to skip an open summary route before retrying it |
| `LCM_EXPANSION_MODEL` | summary model / auxiliary | Override `lcm_expand_query` synthesis model |
| `LCM_EXPANSION_CONTEXT_TOKENS` | `32000` | Context budget used by the auxiliary LLM for `lcm_expand_query` |
| `LCM_SUMMARY_TIMEOUT_MS` | `60000` | Timeout for one summarization call |
| `LCM_TEMPORAL_ROLLUPS_ENABLED` | `false` | Enable derived UTC day/week/month summary rollups and their maintenance hooks |
| `LCM_ROLLUP_DAILY_TARGET_TOKENS` | `5000` | Target size for daily rollup summarization |
| `LCM_ROLLUP_DAILY_MAX_TOKENS` | `15000` | Hard token ceiling for a daily rollup |
| `LCM_ROLLUP_AGGREGATE_MAX_TOKENS` | `20000` | Hard token ceiling for weekly and monthly rollups |
| `LCM_ROLLUP_BUILDS_PER_PASS` | `2` | Maximum rollups built by one automatic pass or `/lcm rollups rebuild` command |
| `LCM_EXPANSION_TIMEOUT_MS` | `120000` | Timeout for one `lcm_expand_query` synthesis call |
| `LCM_DATABASE_PATH` | auto | SQLite database path. Empty config resolves to `HERMES_HOME/lcm.db`; plugin installs or operators may set this env var to another profile-scoped path such as `~/.hermes/hermes-lcm.db`. |
| `LCM_FTS_INTEGRITY_CHECK_INTERVAL_HOURS` | `24` | Minimum hours between startup FTS5 deep integrity-checks (O(index size)). `0` checks every startup (previous behavior); a negative value never checks on startup. Structural checks always run regardless. |
| `LCM_ENABLE_SLASH_COMMAND` | `false` | Enable the optional `/lcm` operator command surface |
| `LCM_EMBEDDINGS_ENABLED` | `false` | Opt in to embedding warmup, backfill, and semantic retrieval storage |
| `LCM_EMBEDDING_PROVIDER` | empty | Embedding provider: `voyage`, `ollama`, or `fastembed` |
| `LCM_EMBEDDING_MODEL` | empty | Provider model identifier registered by `/lcm embed warmup` |
| `LCM_EMBEDDING_STORAGE_DTYPE` | `float32` | Vector storage dtype for newly-registered embedding profiles: `float32` (byte-identical legacy path) or `int8` (per-vector quantization plus a sign-bit prescreen, a distinct profile identity). See [Vector storage scale options (v3)](#vector-storage-scale-options-v3) |
| `LCM_EMBEDDING_STORE_DIM` | `0` | Optional Matryoshka truncation dimension for newly-registered profiles (`0` = full profile dim); truncated vectors are renormalized and are also a distinct profile identity |
| `LCM_EMBEDDING_BINARY_PRESCREEN` | `false` | Write the sign-bit prescreen for float32 identities too (int8 identities always write it), unlocking the full-corpus two-stage KNN; flipping it on an already-populated identity mints a new, distinct identity rather than mutating the existing one |
| `LCM_KNN_PRESCREEN_MULTIPLIER` | `4` | Stage-1 prescreen breadth for two-stage KNN: `M = multiplier × k` lowest-Hamming-distance survivors are exact-rescored |
| `LCM_PROACTIVE_RECALL_ENABLED` | `false` | Opt in to proactive memory injection: at assembly, embed the newest user message and inject one budget-capped "relevant memories" block (needs `LCM_EMBEDDINGS_ENABLED`). Default-off keeps assembly byte-identical |
| `LCM_PROACTIVE_RECALL_MIN_SCORE` | `0.01` | Relevance floor for an injected memory. RRF-scale by default (a top-of-arm hit is ~0.016); with `LCM_RERANK_ENABLED` the score is a `[0,1]` cross-encoder relevance, so raise this (e.g. `0.3`) for a strict semantic gate |
| `LCM_PROACTIVE_RECALL_BUDGET_TOKENS` | `500` | Hard token budget for the single injected block (1-3 items) |
| `LCM_PROACTIVE_RECALL_PROVIDER` | empty | Embedding-provider override for the injection query only (e.g. keep a local `fastembed` provider offline even when search uses `voyage`). Empty reuses the main provider. The override provider must have embedded the corpus for its arms to return hits |
| `LCM_DOCTOR_CLEAN_APPLY_ENABLED` | `false` | Permit destructive `/lcm doctor clean apply` in trusted operator contexts |
| `LCM_EMPTY_LIFECYCLE_GC_ENABLED` | `true` | Master toggle for automatic pruning of lifecycle rows for sessions that never ingested any messages or summary nodes |
| `LCM_EMPTY_LIFECYCLE_GC_THRESHOLD` | `200` | Number of lifecycle rows at which the GC pass fires (default 200 so fresh installs skip the work) |
| `LCM_EMPTY_LIFECYCLE_GC_MAX_AGE_HOURS` | `24` | Automatic GC only deletes empty lifecycle rows at least this old; set `0` only in trusted/test environments that intentionally want immediate empty-row pruning |

### Evidence and adaptive retrieval (0.21 RC)

Hermes exposes all 15 LCM tool schemas whenever LCM is the active context
engine. Exposure is not activation. On a stock install:

- `lcm_compute`, `lcm_compile_evidence`, and `lcm_evidence_pack` are bounded,
  provider-neutral operations over caller-supplied exact refs. Calling them does
  not enable a store, run an extractor, or activate an answering model.
- `lcm_query_state` returns `status: disabled` until
  `LCM_ASSERTIONS_ENABLED=true` creates/binds the rebuildable assertion sidecar.
- `lcm_retrieve` returns `status: disabled` until
  `LCM_ADAPTIVE_RETRIEVAL_ENABLED=true`. The controller itself has no model or
  provider client, but retrieval calls it dispatches retain their existing
  embedding-provider behavior.

| Variable | Default | Use |
|----------|---------|-----|
| `LCM_ASSERTIONS_ENABLED` | `false` | Create and bind the same-DB assertion sidecar so `lcm_query_state` can return typed, source-cited state. This alone performs no extraction or backfill. |
| `LCM_ASSERTION_EXTRACTION_ENABLED` | `false` | With assertions enabled, run bounded structured extraction over exact persisted rows before compaction. This may send source text to the configured extraction/summary model. |
| `LCM_ASSERTION_EXTRACTION_MODEL` | empty | Extraction-model override; otherwise uses `LCM_EXTRACTION_MODEL`, then the summary model. |
| `LCM_ASSERTION_EXTRACTION_MAX_SOURCES_PER_PASS` | `4` | Maximum exact source rows per extraction pass; runtime clamps to 1-8. |
| `LCM_ASSERTION_EXTRACTION_TIMEOUT_SECONDS` | `30` | Timeout for each exact-source extraction call; runtime clamps to 0.1-120 seconds. |
| `LCM_QUERY_VIEWS_ENABLED` | `false` | Create and bind demand-shaped evidence views without invoking a model or retrieval provider. |
| `LCM_ADAPTIVE_RETRIEVAL_ENABLED` | `false` | Enable `lcm_retrieve` and bind query views for evidence reuse. Episodes are bounded to existing retrieval tools and store evidence/traces, never final prose. |
| `LCM_PREANSWER_EVIDENCE_ENABLED` | `false` | Enable the automatic pre-answer evidence hook. Disabled preserves the ordinary hook context and performs no retrieval or computation. |
| `LCM_PREANSWER_EVIDENCE_MODE` | empty | When the master flag is enabled, empty selects legacy selective behavior; explicit values are `off`, `legacy_selective`, or `requirements_v1`. |
| `LCM_SELECTIVE_COMPILER_ENABLED` | `false` | Separately opt into the semantic selector for code-derived closed operations. Disabling the selective compiler does not prevent the pre-answer hook from retrieving a baseline. |
| `LCM_SELECTIVE_COMPILER_MODEL` | empty | Optional model override for the selective compiler. |

When no caller-supplied baseline is available, routed pre-answer turns may call
`lcm_recall` to build one. If embeddings are enabled, this retrieval inherits
the configured embedding provider and may send the current question to that
provider. Disabling the selective compiler prevents its selector and answering
model calls; it does not disable this retrieval path.

Privacy boundary: assertion and query-view records live in the selected
profile's existing `lcm.db` and may include exact quotes, spans, and dependency
refs. Provider-neutral evidence tools do not upload them by themselves.
Embedding-backed retrieval, including automatic pre-answer baseline retrieval,
assertion extraction, and the optional selective compiler can send configured
content to their selected providers. Review those provider and redaction
settings before opting in; sensitive-pattern redaction is also default-off and
is forward-only.

Advanced compaction, assembly, and extraction knobs are defined in `config.py`.

### Temporal rollup operations

Temporal rollups are opt-in. Set `LCM_TEMPORAL_ROLLUPS_ENABLED=true`, tune the
four `LCM_ROLLUP_*` controls above if needed, and restart Hermes. Rollup periods
are UTC calendar periods. Enabling the feature creates its tables lazily; a
disabled install creates no rollup tables and leaves the core schema untouched,
so a base build still opens the database.

Automatic maintenance marks a day (and its containing week and month) stale when
a **summary node covering that day is published** — publication, not raw
ingest, is the signal a rollup consumes — and rebuilds at most
`LCM_ROLLUP_BUILDS_PER_PASS` rows per pass, with daily rollups ahead of
aggregates. A week or month is only published `ready` once every day in the
period that has content has a `ready` daily rollup; while any content day is
missing, stale, or building the aggregate stays stale with a recorded reason and
`lcm_recent` falls back to daily/leaf summaries for the whole window. Rebuilding
a daily re-stales its containing week and month so aggregates never remain
`ready` against an outdated day.

**Scope and rotation boundary.** Rollups are scoped to the LCM session id.
Summary nodes carry no conversation-family key at this layer, so a rollup does
not automatically span sessions across a `/new` rotation; after a rotation,
retained higher-depth summaries are carried into the new session and remain
retrievable, but per-period rollup rows are rebuilt under the new session scope.
Build-cursor state is tracked per `(period_kind, scope)` so multiple scopes
sharing one database never share a cursor.

With `LCM_ENABLE_SLASH_COMMAND=true`, operators can inspect the current
foreground session and request a bounded synchronous rebuild:

```text
/lcm rollups
/lcm rollups rebuild day 2026-07-15
/lcm rollups rebuild week 2026-07-15
/lcm rollups rebuild month 2026-07-15
/lcm rollups rebuild all 2026-07-15
```

The optional date defaults to the current UTC date. Week targets normalize to
Monday and month targets to the first day. `all` targets the containing day,
week, and month in that order. The command first **durably seeds a `stale` row
for every requested target** (creating one if it does not yet exist), then
attempts no more than the configured per-pass limit and prints an outcome for
every target. Targets beyond the bound are reported `stale (bounded; not
attempted)` and remain as durable `stale` rows, so later automatic maintenance
builds them. Builds run now and may invoke the summary model. They use the same
summary circuit breaker, fallback routes, timeout, and spend guard as normal LCM
summarization.

`lcm_inspect` always includes a `temporal_rollups` block with the enabled flag,
ready/stale/building/failed counts for each period kind, oldest stale age,
last-build cursors, and the last error. `/lcm rollups` renders the same data as a
table. Both paths are read-only and make no LLM calls. When the feature is off
or its tables are empty, the block remains present with zero counts and null
age/cursor/error values so monitoring consumers do not need a second schema.

Sensitive-pattern handling is disabled by default so ordinary LCM storage and
`lcm_expand` remain lossless. When `LCM_SENSITIVE_PATTERNS_ENABLED=true`, matched
secret values are replaced with metadata-only placeholders before SQLite, FTS,
summaries, active replay, and externalized payload JSON receive the content. This
is intentionally not lossless for matching values: the raw matched secret is
unrecoverable after redaction.

Supported named catalog entries are:

- `api_key`: `api_key`, `api_token`, `access_token`, `secret_key`, and
  `client_secret` assignments or JSON keys.
- `bearer_token`: `Bearer ...` strings and token-like JSON keys.
- `password_assignment`: `password`, `passwd`, `pwd`, and `passphrase`
  assignments or JSON keys, including quoted values with spaces.
- `private_key`: PEM private-key blocks.

Redaction is forward-only. Enabling it does not rewrite existing SQLite rows,
FTS shadow tables, DAG summaries, or externalized payload JSON that were written
before the setting was enabled. Non-password placeholders include a short
truncated SHA-256 digest for correlation. `password_assignment` placeholders omit
the digest to avoid making password-like values easier to dictionary-check.
`lcm_status`, `lcm_inspect`, and `lcm_doctor` expose the enabled state, configured pattern names,
unknown names, source, and placeholder format without exposing raw secret values.

### Cache policy boundary

LCM is **cache-friendly**, not fully cache-aware. It may avoid some follow-on
condensation churn, but current cache usage counters are retrospective status
data only; they do not tell the plugin whether the next prompt mutation will
break a hot provider cache. For that reason LCM does **not** implement
provider/model TTL heuristics, provider-family detection, cache-touch/cache-break
tracking, TTL delays, or full cache-aware deferred compaction.

`LCM_CRITICAL_BUDGET_PRESSURE_RATIO` is a narrow escape hatch. It is disabled by
default (`0.0`). When set, LCM compares prompt pressure against the context
window and only at or above that ratio may bypass the existing polite gates for
bounded deferred maintenance catch-up and cache-friendly follow-on condensation.
Below that pressure, simpler existing behavior is preserved. Revisit full
cache-aware deferred compaction only after Hermes core exposes reliable cache
state / cache-break signals.

### Threshold ownership

When `context.engine: lcm` is active, `LCM_CONTEXT_THRESHOLD` is the compaction
threshold LCM uses. Hermes core `compression.threshold` belongs to the built-in
compressor. Hermes core `compression.enabled` is still the global gate that
allows compaction, so leave it enabled when using LCM.

If `LCM_ABSOLUTE_THRESHOLD_TOKENS` is set to a positive integer, it overrides the
ratio-derived trigger after window math runs. Use this when you want a fixed
context-health setpoint across model switches (for example `130000` for coding
agents) so a larger window does not silently delay compaction, lower recall, or
let long sessions accumulate more noise before LCM intervenes. Leave it at `0`
to keep ratio-based behavior. When the absolute override is active, Codex
GPT-5.5 ratio auto-raise is suppressed so the absolute setpoint stays pinned.

If startup/status output shows a host-side compression percentage that disagrees
with LCM, trust live LCM status after a normal message has initialized the
session.

### Preset inspection and dry-run suggestions

Model-aware presets are inspectable, but they are not automatic live config
mutations. With `LCM_ENABLE_SLASH_COMMAND=true`, use:

```text
/lcm preset show codex_gpt_long_context
/lcm preset show codex_spark_context
/lcm preset suggest
/lcm preset apply codex_gpt_long_context --dry-run
/lcm preset apply codex_spark_context --dry-run
```

`/lcm preset show` reports the shipped preset metadata, benchmark provenance,
policy file, policy version, and metric summary. `/lcm preset suggest` chooses a
safe shipped suggestion for the current context window when one exists and emits
confidence reasons. The confidence is `benchmark-backed-route` only when host
metadata identifies an OpenAI Codex/GPT route; otherwise it stays `context-only`
and tells the operator to confirm provider/model family before applying any env
changes. `/lcm preset apply ... --dry-run` previews env-var settings only; it does not
write files, change process state, or override explicit parseable
preset-managed `LCM_*` environment variables. Invalid preset-managed env values
are reported as invalid instead of being treated as active runtime overrides.

The current runtime preset dry-run previews are:

```text
# codex_gpt_long_context: near-272k GPT/Codex routes
LCM_CONTEXT_THRESHOLD=0.75
LCM_FRESH_TAIL_COUNT=24
LCM_LEAF_CHUNK_TOKENS=8000

# codex_spark_context: near-128k GPT-5.3 Codex Spark routes
LCM_CONTEXT_THRESHOLD=0.75
LCM_FRESH_TAIL_COUNT=16
LCM_LEAF_CHUNK_TOKENS=8000
```

`target_after_compaction=0.55` is still benchmark provenance, not a runtime
setting, because the engine does not expose that live knob yet.

For dashboards and agents, `lcm_status` also exposes the same information under
`preset_suggestion` as read-only JSON: suggested preset name/family, selection
reason, match confidence, confidence reasons, provenance, explicit override diagnostics, invalid
override diagnostics, unsupported benchmark-only fields, and the dry-run delta.
This surface is safe to consume without scraping slash-command text; it does not
write files, mutate process state, expose local benchmark run paths, or apply
`LCM_*` environment changes.

### FAQ: tuning LCM for large context windows

Long-context models change the tuning problem. A 1M-token model does not mean
you always want to spend 750k prompt tokens before LCM starts compacting. Start
with the active prompt budget you are willing to pay for, then tune the threshold
around that budget.

The basic calculation is:

```text
compaction trigger = effective context window * LCM_CONTEXT_THRESHOLD
```

Or, working backwards:

```text
LCM_CONTEXT_THRESHOLD = desired compaction trigger / effective context window
```

Examples, as math rather than universal recommendations:

| Effective context window | Desired trigger | Threshold |
|--------------------------|-----------------|-----------|
| `128000` | `96000` | `0.75` |
| `200000` | `140000` | `0.70` |
| `400000` | `240000` | `0.60` |
| `1000000` | `250000` | `0.25` |
| `1000000` | `400000` | `0.40` |
| `1000000` | `600000` | `0.60` |

If your Hermes config caps the model's effective `context_length`, tune against
that effective value instead of the provider's advertised maximum. For example,
a 1M-token provider with an effective `context_length` of `400000` and
`LCM_CONTEXT_THRESHOLD=0.60` starts compaction around `240000` prompt tokens.

A reasonable first pass for a true 1M effective window is:

| Goal | Desired trigger | Threshold | Notes |
|------|-----------------|-----------|-------|
| Lower spend / earlier DAG building | `200000` to `300000` | `0.20` to `0.30` | Good when cost and latency matter more than maximum live context |
| Balanced large-context use | `350000` to `500000` | `0.35` to `0.50` | Good starting point for many long-running agents |
| Keep more raw context active | `600000+` | `0.60+` | Higher token burn, later compaction |

What the main knobs do:

- `LCM_CONTEXT_THRESHOLD` decides when compaction starts. Lower values build the
  DAG earlier and reduce active prompt burn, but compact more often.
- `LCM_FRESH_TAIL_COUNT` protects recent messages from compaction. Raise it if
  your agent often needs the last few tool calls or planning turns verbatim.
- `LCM_LEAF_CHUNK_TOKENS` is the raw-backlog floor before a leaf compaction pass
  starts. With the default `LCM_DYNAMIC_LEAF_CHUNK_ENABLED=false`, the pass
  compacts the whole non-tail raw backlog, not only a chunk of this size.
- `LCM_DYNAMIC_LEAF_CHUNK_ENABLED=true` changes leaf passes into chunk-sized
  work. In that mode `LCM_LEAF_CHUNK_TOKENS` is the base target and
  `LCM_DYNAMIC_LEAF_CHUNK_MAX` is the upper bound for a dynamic chunk target.
- `LCM_THRESHOLD_FULL_SWEEP_ENABLED=true` makes a threshold-triggered invocation
  keep draining oldest raw chunks outside the protected tail, even after prompt
  pressure falls below the trigger. It always uses the configured working leaf
  size, then condenses a too-large summary frontier toward
  `LCM_SUMMARY_PREFIX_TARGET_TOKENS` (`0` means one leaf budget). The whole
  invocation is bounded to 12 summary calls and 120 seconds between calls,
  persists each completed DAG pass, and publishes one active context at the end.
  It remains synchronous and does not enable deferred/background maintenance.
- `LCM_EXPANSION_CONTEXT_TOKENS` controls how much recovered material
  `lcm_expand_query` may feed to the auxiliary model. It does not change what
  LCM stores.
- `LCM_LARGE_OUTPUT_EXTERNALIZATION_ENABLED=true` helps when large tool outputs,
  logs, media payloads, or raw JSON blobs dominate token pressure.

Common questions:

**Should I leave the default threshold on a 1M-token model?**

Not always. The default `0.35` means compaction starts around `350000` prompt
tokens on a true 1M effective window. That leaves far more headroom for new
content but compacts more aggressively, which can shorten recall of older
details.

**Should I change leaf chunk settings first?**

Usually no. Start with `LCM_CONTEXT_THRESHOLD`, `LCM_FRESH_TAIL_COUNT`, and large
output externalization. Only tune leaf chunking after checking `lcm_status` and
understanding whether your workload is dominated by huge raw backlog passes. If
you want chunk-sized leaf passes, enable dynamic leaf chunking explicitly.

**Does compacting earlier hurt recall?**

It changes the tradeoff. More content moves from the live prompt into summaries
earlier, but raw messages are still stored and recoverable through LCM tools. If
you need exact details later, use `lcm_grep`, `lcm_describe`, `lcm_expand`, or
`lcm_expand_query` instead of relying only on the active prompt.

**How do I know whether my settings are working?**

After a normal message has initialized the session, check `lcm_status` for the
effective context length, threshold tokens, prompt pressure, compression count,
raw rows, and summary nodes. Run `lcm_doctor` when behavior looks surprising.

### Session pattern syntax

Pattern matching checks multiple keys: raw `session_id`, `platform`, and
`platform:session_id`.

- `*` matches within one colon-delimited segment
- `**` can span across colons

Example: `cron:*` can match Hermes cron sessions, while exact raw session IDs
still work.

### Noise suppression

LCM offers two layers of noise filtering, sized to two different shapes of
noise:

- **Session-level filters** (`LCM_IGNORE_SESSION_PATTERNS`,
  `LCM_STATELESS_SESSION_PATTERNS`) catch the case where the noisy traffic
  arrives as its own session or platform, for example a dedicated `cron:*`
  session. Match keys cover the session id, the platform, and
  `platform:session_id`.
- **Message-level patterns** (`LCM_IGNORE_MESSAGE_PATTERNS`) catch the case
  where cron alerts or other noise are injected into a normal Telegram or
  WhatsApp conversation as ordinary user-visible messages. From LCM's
  perspective the session/platform is `telegram` or `whatsapp`, not `cron`,
  so only the message content is distinctive.

Message-level patterns are Python regex strings, comma-separated, compiled
once at engine start. They run against plain message text. For structured
multimodal payloads, LCM matches against concatenated text parts first, so
anchored patterns bind to the text an operator sees. If a structured payload
contains no text parts, matching falls back to the normalized JSON form that
LCM would have written to the store. Matching messages are skipped before
storage, so new matching rows do not enter the messages table or FTS index.
Filtering is role-agnostic by default, since cron alerts can be re-emitted
under any role depending on the gateway.

Example operator config:

```
LCM_IGNORE_MESSAGE_PATTERNS=^Cronjob Response:,^>>>Cronjob Response<<<:
```

Invalid regex entries are logged at warning level and dropped; the
surviving patterns in the same list still take effect, so a misconfigured
entry never crashes ingest. Pattern matching uses a 50 ms per-pattern timeout
when the optional `regex` package is installed. If `regex` is not installed,
LCM logs a warning and disables message-level regex filtering rather than
running unbounded stdlib `re` matches in the ingest path.

One operator-facing limitation to know about:

- **Compaction-window edge.** The filter runs at ingest time. When a
  matching message is part of the chunk being summarized in the same
  turn it arrived, the message's text may appear inside the resulting
  summary node text. In long-running sessions where compaction triggers
  every several dozen turns, this can affect multiple summary nodes per
  day rather than only happening rarely. The summary node's `source_ids`
  will not reference the filtered message (it was never written to the
  store), so DAG lineage stays clean; only the serialized summary text
  can carry it. Closing this window is tracked as follow-up work.

`lcm_status` surfaces the full filter contract under `session_filters`, including
`ignore_session_patterns`, `stateless_session_patterns`, `ignore_message_patterns`,
their `*_source` fields (`default` or `env`), the current session's `ignored` and
`stateless` booleans, and a process-lifetime `ignored_message_count` so operators
can confirm their patterns are loaded and watch how often message filters fire.
The counter resets on engine restart.

Ignored/stateless sessions are a storage ownership boundary, not a context-window
opt-out. LCM does not ingest raw messages or create DAG nodes for sessions that
match `LCM_IGNORE_SESSION_PATTERNS`, `LCM_STATELESS_SESSION_PATTERNS`, or the
in-process auxiliary/thread stateless marker. If those sessions cross the normal
context threshold, LCM delegates the compaction call to Hermes' native
`ContextCompressor` so the active request is still bounded before model overflow.
If the native compressor is unavailable, LCM falls back to a deterministic
head/tail trim as a last-resort safety net, still without writing the bypassed
session to `lcm.db`.

### Large tool-output handling

Externalization for ordinary large tool output is opt-in. When enabled,
oversized tool results are written to plugin-managed JSON files and referenced
from summaries. They remain inspectable later through
`lcm_describe(externalized_ref=...)` and `lcm_expand(externalized_ref=...)`.

Active-replay stubbing is separately opt-in and requires ordinary large-output
externalization. Newly ingested textual tool results above the token threshold
are durably externalized and immediately replaced in provider-visible replay,
including results in the protected fresh tail. Preflight adopts that replay
change even if no leaf compaction is eligible. A second historical assembly
pass replaces older eligible results before evaluating the assembly budget and
respects the protected fresh tail. Tool roles, `tool_call_id` values, and
compatible structured text block types/keys are retained; the historical pass
does not rewrite raw SQLite/DAG lineage. Structured image/media tool results
remain inline, preserving the provider-replay contract established by PR #226.
Failure to durably externalize is fail-open: the provider receives the original
inline payload. Results from `lcm_describe` and `lcm_expand` also stay inline so
recovery does not recursively create another drilldown step.

`lcm_grep` keeps history-only behavior by default. Operators and agents may opt
into bounded active-session payload search with
`content_scope='externalized'|'both'`; optional `externalized_refs` narrows the
scan to known refs. The payload path rejects symlinks and foreign-session refs,
scans at most 256 files and 512,000 encoded content bytes per file, and returns
only bounded snippets plus recovery metadata. See the
[retrieval tools reference](retrieval-tools.md#searching-externalized-payloads)
for the exact contract.

The storage-boundary payload guard is separate from that opt-in. LCM always
scans messages at the store boundary before writing `messages.content` or
`messages.tool_calls` to SQLite. Inline `data:*;base64,...` payloads and
conservative long base64-looking runs are replaced with compact placeholders and
written to the same plugin-managed externalized-payload directory. This is a
safer default for media-ish payloads: LCM still preserves lossless recovery via
the placeholder `ref` and `lcm_expand(externalized_ref=...)`, but it does not
duplicate those payload bytes into `lcm.db`, FTS shadow tables, WAL files, or
ordinary SQLite backups. If externalization fails, LCM logs a warning and leaves
the original text inline rather than dropping data.

`lcm_doctor` reports the effective SQLite database path, core schema-table
presence, SQLite `journal_mode`, `quick_check`, database/WAL sizes, the largest
content/tool-call rows, suspicious inline `data:*;base64` rows, suspicious long
base64-looking rows, and aggregate externalized-payload stats.
Doctor output is metadata-only for these scans; it intentionally does not print
raw payload previews.

`lcm_doctor` JSON includes a top-level `guidance` array for every warning or
failure. The slash-command text also reports `triage_guidance` for the warning
and failure classes surfaced in the command output, using the same operator
action vocabulary. Each item maps a warning class to one of three operator
actions:

- `safe/ignore`: informational operating state; leave it alone unless it is
  crowding useful recall or repeatedly surprising operators.
- `inspect`: read the named rows/session IDs/config before making changes.
- `backup-first cleanup`: run the read-only preview command, create `/lcm backup`,
  then run the explicit apply command only if the preview still matches intent.

Warning-only classes should not auto-clean state: `summary_quality`, broad
`lifecycle_fragmentation`, payload-storage suspicion, and `context_pressure` are
evidence for review, not proof that mutation is safe.

This guard is scoped to LCM's own `lcm.db` write boundary. It does not prevent
Hermes core, or any other host layer, from writing inline payloads to Hermes
`state.db`; that is upstream/outside LCM scope. It also does not rewrite
historical rows already present in `lcm.db`. Cleaning older suspicious rows
requires a separate backup-first cleanup or migration flow.

Transcript GC is separate and also opt-in. It only rewrites already-externalized,
already-summarized tool-role rows to compact placeholders. It keeps the same
`store_id`, keeps payload files, skips pinned messages, and preserves lossless
recovery through `externalized_ref`. After GC, `lcm_grep` will not match the
original giant tool blob text directly; search summaries or refs instead.

## Slash Commands

Slash commands are disabled by default. Enable them only in trusted operator
contexts:

```bash
export LCM_ENABLE_SLASH_COMMAND=1
```

Available commands:

- `/lcm` or `/lcm status` - current runtime/session status
- `/lcm doctor` - read-only health checks
- `/lcm doctor clean` - read-only scan for obvious junk/noise session candidates
- `/lcm doctor clean apply` - backup-first cleanup for safe pattern-matched candidates; requires `LCM_DOCTOR_CLEAN_APPLY_ENABLED=true`
- `/lcm doctor clean lifecycle` - read-only scan for lifecycle rows with zero messages/nodes
- `/lcm doctor clean lifecycle apply` - backup-first cleanup of empty lifecycle rows; requires `LCM_DOCTOR_CLEAN_APPLY_ENABLED=true`
- `/lcm doctor repair` - read-only SQLite/FTS repair diagnostics
- `/lcm doctor repair apply` - backup-first SQLite/FTS repair
- `/lcm doctor source` - read-only scan for legacy blank-source rows
- `/lcm doctor source apply` - backup-first normalization of legacy blank-source rows to `unknown`
- `/lcm doctor retention` - read-only retention analysis
- `/lcm backup` - timestamped SQLite backup
- `/lcm rotate` - read-only preview of an in-place tail-preserving compact of the active session
- `/lcm rotate apply` - backup-first rotate that advances the lifecycle frontier past pre-tail raw messages
- `/lcm embed warmup` - explicitly prepare the configured provider/model and register its vector dimension
- `/lcm embed backfill [--limit N] [--corpus summary|chunks|both] [--policy conversational|heads|full]` - preview pending embeddings, token use, batches, and estimated cost for a corpus
- `/lcm embed backfill --apply [--limit N] [--corpus summary|chunks|both] [--policy ...] [--confirm-raw-text]` - populate a bounded set of pending embeddings for a corpus
- `/lcm help` - command help

`--corpus` selects which corpus to backfill (default `summary`); `--policy` chooses the chunking
policy and applies only to the chunk corpus; `--confirm-raw-text` acknowledges that the chunk corpus
sends raw verbatim text to a cloud provider (required for `--corpus chunks|both --apply` on a cloud
provider — see *Embedding backfill* below).

Apply paths are intentionally narrow and backup-first. Start with diagnostics
before cleanup or repair.

### Rotate: in-place compact without changing session identity

`/lcm rotate` lets an operator compact a long-running session in place without
changing `session_id` or `conversation_id`. It is the in-session counterpart to
Hermes-level `/new` (which starts a new session) and to `/lcm doctor clean`
(which prunes whole junk sessions).

What rotate does:

- preserves the live tail (`LCM_FRESH_TAIL_COUNT` most-recent messages)
- advances the lifecycle frontier marker past every raw message before the tail,
  so subsequent bootstrap stops replaying them into the active prompt
- writes a rolling `*-rotate-latest.sqlite3` backup under the same backup
  directory as `/lcm backup`, overwriting the previous rotate slot atomically
  so disk usage stays bounded across repeated rotates

What rotate does not do:

- it does not delete raw messages — pre-tail rows remain in the SQLite store
  and stay recoverable through `lcm_load_session` and `lcm_expand`, preserving
  the lossless raw recovery contract
- it does not invoke the summarization model — rotate is a frontier and backup
  operation, not a synthesis step. Pre-tail content that was not yet covered by
  a summary node remains accessible only as raw rows; trigger normal compaction
  first if you want the pre-tail range covered by summary nodes before rotating
- it does not change session or conversation identity, run on stateless
  sessions (`LCM_STATELESS_SESSION_PATTERNS`), or run on ignored sessions
  (`LCM_IGNORE_SESSION_PATTERNS`) — rotate refuses on both with a clear reason

`lcm_status` and `/lcm status` surface `last_rotate_at` (epoch float, or `null`
when no rotate has happened) and the rolling backup path so operators can see
when rotate last ran. The mtime of the rolling backup file is the source of
truth — no new lifecycle column is added by this feature.

Re-running `/lcm rotate apply` on a session whose frontier is already at or
ahead of the target boundary reports `status: noop` and is safe to retry.
A no-op apply does not write a new rolling backup, so the previous
known-good `*-rotate-latest.sqlite3` snapshot survives idempotent retries.

## Embedding backfill

Embedding backfill is opt-in and dry-run-first. Configure and warm the model
before applying any work:

```bash
export LCM_EMBEDDINGS_ENABLED=true
export LCM_EMBEDDING_PROVIDER=ollama   # voyage or fastembed are also supported
export LCM_EMBEDDING_MODEL=nomic-embed-text

/lcm embed warmup
/lcm embed backfill
/lcm embed backfill --apply
```

The default invocation previews up to 200 newest pending depth-0 summaries. It
reports the total pending count, selected count, estimated input tokens,
provider batches, estimated cost, remaining work, and duration. It makes no
provider call and opens SQLite read-only, so it performs no database write.
Use `--limit N` to choose a smaller bounded invocation before adding `--apply`.

Local Ollama and FastEmbed estimates are `$0`. Voyage estimates use the known
per-token rate for the configured model (or a conservative generic Voyage
rate when the model is not in the built-in table); treat the line as a planning
estimate and verify current provider pricing before a large run. The estimate
uses gross list price and does not subtract account-specific free tokens.

Apply mode is safe to resume. Rows already embedded for the current registered
profile are skipped by the discovery query, and a run that stops leaves its
unwritten rows pending for the next invocation. The command serializes apply
runs with a single-flight claim; a crashed claim becomes eligible for takeover
after 10 minutes. Provider calls occur before per-row SQLite writes, and each
row is committed independently, so one malformed row does not roll back the
rest of a successful provider batch.

Voyage authentication failures abort immediately because later batches would
fail the same way. Transient provider failures are reported for the affected
rows and later batches continue; rerun the command to retry anything still
pending. Documents rejected by a provider token cap are listed under
`skipped_overcap` and also remain pending. The claim is released on normal,
provider-error, and row-write-error exit paths.

### Corpora: summary vs chunks

`--corpus` selects what gets embedded:

- `summary` (default) — the generated leaf-summary embeddings described above.
- `chunks` — the **raw-history chunk corpus**: verbatim message text chunked by
  `--policy` (`conversational` | `heads` | `full`), used for verbatim/chunk-KNN
  recall. This is a **separate corpus with its own backfill run** — a
  summary-only backfill leaves it empty, and verbatim/chunk recall returns
  nothing beyond FTS until you run `--corpus chunks --apply` as well.
- `both` — runs the summary backfill, then the chunk backfill, in one command.

```bash
/lcm embed backfill --corpus chunks              # dry-run preview for the chunk corpus
/lcm embed backfill --corpus chunks --apply --confirm-raw-text
/lcm embed backfill --corpus both --apply --confirm-raw-text
```

**Raw-text consent gate.** Unlike summaries, the chunk corpus sends **raw,
verbatim message text** — including tool-result and error/traceback content — to
the embedding provider. When the provider is a cloud provider (e.g. Voyage),
`--corpus chunks|both --apply` **refuses** unless you also pass
`--confirm-raw-text`. Local providers (fastembed/ollama) never transmit text
off-box and are exempt. Note that `LCM_SENSITIVE_PATTERNS_ENABLED` redaction runs
at **ingest**, so it does not retro-redact history already stored — that older
raw text is still sent during a chunk backfill. See
[embeddings-setup.md](embeddings-setup.md) for the full discussion.

### Contextualized chunk grouping

Voyage's `voyage-context-*` chunk models (the chunk-corpus default for the
`voyage` provider, see `default_chunk_model`) are contextualized-embedding
models: one message's chunks are sent to the provider's
`/v1/contextualizedembeddings` endpoint as a single grouped document — an inner
list of that message's chunks — instead of as independent inputs, so the model
actually conditions each chunk's embedding on its sibling chunks from the same
message. Non-context providers and models (fastembed, ollama, plain Voyage
embedding models) are unaffected and continue to embed chunks independently.
Grouped requests are bounded by the provider's per-document and per-request
caps (32K tokens per chunk, 120K tokens per document, 120K tokens and 16,000
chunks per request); a document over the per-document budget is split into
contiguous sub-documents so no single request is rejected. The retry path for
uncertain chunks (`/lcm embed backfill`'s retry-uncertain pass) selects rows
out of store_id order, so it stably re-sorts the selected documents by
`(store_id, chunk_index)` before batching — otherwise interleaved retry rows
would collapse into singleton groups and silently defeat contextualization for
that pass.

### Vector storage scale options (v3)

The default embedding storage path is float32 at full provider dimension, with
an exact scan bounded by `LCM_EMBEDDING_BOUNDED_SCAN_ROWS` once a corpus
outgrows that bound (see the `coverage` contract in the
[retrieval tools reference](retrieval-tools.md#full-text-semantic-and-hybrid-modes)).
That path is unchanged from pre-v3 installs. At real-archive scale — tens of
thousands of vectors and up — a full brute-force scan gets slow and
memory-heavy, and the recency-bounded fallback stops reaching the whole corpus.
Four v3 environment variables, all opt-in, additive, and default-off, trade
some exactness for full-corpus reach at much lower query cost:

- `LCM_EMBEDDING_STORAGE_DTYPE` (default `float32`) selects the vector storage
  dtype for **newly-registered** embedding profiles. `int8` stores each vector
  as a per-vector symmetrically-quantized signed-byte array plus a
  little-endian float32 scale in the same blob, and always writes a
  companion sign-bit prescreen. `dtype` is part of the profile identity hash,
  so an int8 identity never mixes with an existing float32 one.
- `LCM_EMBEDDING_STORE_DIM` (default `0`, meaning the full profile dimension)
  applies an optional Matryoshka truncation to newly-registered profiles. When
  set to a value less than the provider's native dimension, vectors are
  truncated to that many leading dimensions and renormalized before
  storage/quantization. The stored dimension is also part of the profile
  identity hash, so truncated vectors never mix with full-dimension ones.
- `LCM_EMBEDDING_BINARY_PRESCREEN` (default `false`) writes the sign-bit
  Hamming prescreen for float32 identities too (int8 identities always write
  it), unlocking the two-stage KNN described below.
- `LCM_KNN_PRESCREEN_MULTIPLIER` (default `4`) sets the two-stage KNN's stage-1
  prescreen breadth: `M = multiplier × k` lowest-Hamming-distance survivors are
  loaded and exact-rescored in stage 2. Larger values widen the approximate
  prescreen toward exact recall at more cost.

**Two-stage KNN.** With a sign-bit prescreen present, `knn` packs the query's
sign bits, Hamming-XORs against every corpus row's prescreen bits (the whole
corpus, not a recency-bounded slice), keeps the `M = LCM_KNN_PRESCREEN_MULTIPLIER
× k` lowest-Hamming survivors, then loads only those survivors' full vectors and
ranks them by exact cosine. The response reports `coverage='full_approx'`: the
whole corpus was reached, but stage-1 keeps only the closest `M` candidates
before the exact rescore, so the result is an approximate top-k rather than the
exhaustive top-k that exact-scan `coverage='full'` gives.

**Safety design (flipping the prescreen flag never silently truncates a
corpus).** `LCM_EMBEDDING_BINARY_PRESCREEN` growing a companion sign-bit table
means a prescreen corpus must never be read as if it were a legacy
binary-free float32 corpus. Flipping the flag on an already-populated float32
identity therefore mints a **new, distinct profile identity** (folded into the
identity hash) rather than adding sign-bits to the existing one in place. As a
second, independent guard, the two-stage path only engages when the sign-bit
table is a **complete** mirror of the vector table for that identity — every
stored vector has a matching sign-bit row. If the two tables disagree (a
partially-populated prescreen, or none at all), `knn` falls back to the
existing exact bounded/full scan and reports honest `bounded`/`full` coverage
instead of silently dropping rows the sign-bit table doesn't cover while still
claiming `coverage='full'`.

**Measured trade-offs (C1 bench, 92,997-vector real chunk archive, Voyage
`voyage-context-3`, dim 1024):**

| Path | Coverage | Query latency (warm) | Peak RSS | recall@10 vs. exact |
|---|---|---|---|---|
| float32 full-scan (default) | `full` | ~195ms | ~3.3GB | — (exact) |
| float32 + binary prescreen, two-stage | `full_approx` | ~67ms | ~279MB | 0.96 |
| int8, two-stage | `full_approx` | ~65ms | ~279MB | 0.84 |

The int8 path trades additional recall for roughly a quarter of the disk
footprint of stored vectors (quantized bytes vs. float32), making it the option
for disk-constrained installs; the float32+binary path keeps full-precision
vectors on disk and gives up less recall for the same query-time and RAM win.

**Current limitation.** Enabling `LCM_EMBEDDING_BINARY_PRESCREEN` (or switching
to `int8`) on an archive that is already fully embedded requires a full
re-backfill under the newly-minted identity — there is no shipped operator path
yet to derive the prescreen bits (or quantized vectors) locally from the
existing float32 corpus without re-registering and re-running
`/lcm embed backfill --apply` against the new identity. A local-derive
migration path (computing sign-bits/quantization from already-stored float32
vectors, no provider re-call) is planned but not yet built.

## Import and backfill

### Historical tool-output sidecars

`scripts/backfill_externalized_tool_outputs.py` pre-creates Hermes-native
externalized-payload sidecars for large textual tool rows already in an LCM
database. It opens SQLite read-only and never rewrites messages or summaries.
The default is a dry run:

```bash
python scripts/backfill_externalized_tool_outputs.py \
  --database ~/.hermes/lcm.db \
  --hermes-home ~/.hermes \
  --manifest ./externalization-backfill.json

# Create only eligible sidecars after reviewing the scrubbed manifest.
python scripts/backfill_externalized_tool_outputs.py \
  --database ~/.hermes/lcm.db \
  --hermes-home ~/.hermes \
  --manifest ./externalization-backfill.json \
  --apply
```

The manifest contains references, digests, counts, sizes, and token estimates,
not raw payload content, session ids, or tool-call ids. Repeated apply runs are
idempotent. Rollback is also dry-run by default and accepts only an applied
manifest; `--apply` deletes a manifest-owned sidecar only when its content still
matches the recorded digest and neither a message nor a summary references it:
Sidecar retention and redaction: before a historical tool result is written to
the externalized-payload directory the command applies the currently enabled
`LCM_SENSITIVE_PATTERNS` policy to its content, exactly as live ingest does, so no
un-redacted secret is copied onto the new retention surface. Every manifest digest,
ref, and provenance proof is derived from the redacted content that is actually
stored. Each sidecar mirrors the live externalized-payload shape (kind, role,
session id, tool-call id, and the redacted content) so recovery through
`lcm_expand(externalized_ref=...)` keeps working. The manifest records the active
redaction policy and refuses to resume a journal written under a different policy.

The manifest is a durable ownership journal containing references, digests,
provenance proofs, target-identity hashes, the redaction policy, counts, sizes, and
token estimates, not raw payload content, session ids, or tool-call ids; the refs
carry a `historical-backfill` marker rather than a tool-call stub. Reusing the same
manifest path preserves every sidecar created by earlier apply runs. An interrupted apply
can be rerun with the same path to recover its pending journal entries. The
journal is bound to one database file and one externalized-payload storage root;
the command refuses reuse or rollback against another target. Manifest files and
sidecars are opened without following their final symlink and rollback rechecks
file identity before deletion. A new manifest is published with a no-clobber
link; an existing journal is updated with an atomic name exchange and restored
if the displaced inode is not the validated manifest. A regular file or symlink
that races into the manifest leaf is therefore preserved and the command fails
closed. Existing-journal updates require atomic name-exchange support from the
host OS and filesystem. Apply and rollback require the storage directory to be
owned by the current user and not writable by group or other users. They also
hold an advisory lock on the opened directory so concurrent invocations of this
script cannot mutate it together.

The command also refuses database schemas newer than this build before scanning
rows or writing sidecars. Apply failures are recorded in `counts.failed` and
`failed_paths` and cause a nonzero exit status. Rollback is dry-run by default
and accepts only a complete, applied ownership journal for the historical-
externalization operation; `--apply` deletes a sidecar only when its backfill
provenance binds it to that journal, its content still matches the recorded
digest, and no message content, nested `messages.tool_calls` value, or summary
references it:

```bash
python scripts/backfill_externalized_tool_outputs.py \
  --database ~/.hermes/lcm.db \
  --hermes-home ~/.hermes \
  --rollback ./externalization-backfill.json
```

Stop the profile that owns the target database before an operator apply or
rollback. Sidecar creation is additive, but a quiescent profile makes the
reviewed manifest and reference checks a stable operator boundary.
Stop the profile that owns the target database and every other process running
as that account that can write the externalized-payload directory before an
operator apply or rollback. The directory lock coordinates this script only;
writers that ignore it are outside the supported integrity boundary. POSIX has
no unlink-by-open-file-descriptor primitive, so rollback moves an owned sidecar
to a random quarantine name, checks that inode again immediately before the
name-based unlink, and fails closed with the quarantine entry retained if a
replacement is detected. This quiescent, owner-controlled directory is the
precondition that makes the final unlink safe.

### OpenClaw/lossless-claw history

`scripts/import_lossless_claw.py` is the local, dry-run-by-default operator path
for moving OpenClaw history into a Hermes-LCM `lcm.db`. It supports two source
families:

- `--source-db <path>`: import from an existing lossless-claw/OpenClaw SQLite
  `lcm.db`. Use `--include-summaries` when you also want compatible source
  summaries imported into Hermes `summary_nodes`.
- `--source-jsonl <path>` / `--source-jsonl-dir <path>`: import OpenClaw JSONL
  session exports when there is no source SQLite database, for example fresh
  installs, plugin-off catch-up, or one-off session migrations. JSONL import is
  raw-message-only because the session files do not contain a summary DAG.

Examples:

```bash
# SQLite source, dry-run
python scripts/import_lossless_claw.py \
  --source-db ~/.openclaw/path/to/lcm.db \
  --target-db ~/.hermes/lcm.db \
  --agent sammy \
  --json

# JSONL source directory, dry-run
python scripts/import_lossless_claw.py \
  --source-jsonl-dir ~/.openclaw/agents/sammy/sessions \
  --target-db ~/.hermes/lcm.db \
  --agent sammy \
  --json

# Apply only after reviewing the report
python scripts/import_lossless_claw.py \
  --source-jsonl-dir ~/.openclaw/agents/sammy/sessions \
  --target-db ~/.hermes/lcm.db \
  --agent sammy \
  --import-id sammy-jsonl-2026-07 \
  --apply
```

Safety and reconciliation behavior:

- dry-run is default; writes require `--apply`
- apply mode backs up an existing target DB before writing
- `--json` reports `scanned`, `eligible`, `would_import`, `imported`,
  `skipped_existing`, `skipped_empty`, `invalid_rows`, `warnings`, and summary
  counters
- reruns are idempotent for the same `--import-id`; pass a stable explicit
  import id if the same source files may be copied to different paths
- JSONL imports preserve session id, role, content, timestamp, tool call/result
  metadata, and provenance in target `session_id` / `source`
- no OpenClaw config or separate secret tables are imported; raw transcripts,
  summaries, and tool payloads may still contain sensitive user data

## Related references

- [Retrieval tools reference](retrieval-tools.md)
- [Architecture notes](architecture.md)
- [Benchmarking and stress checks](../benchmarks/README.md)
- [Release validation](release-validation.md)
- [Packaging and distribution posture](packaging.md)
