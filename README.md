<p align="center">
  <img src="banner.png" alt="HERMES-LCM" width="800">
</p>

[![CI](https://github.com/stephenschoettler/hermes-lcm/actions/workflows/ci.yml/badge.svg)](https://github.com/stephenschoettler/hermes-lcm/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/stephenschoettler/hermes-lcm)](https://github.com/stephenschoettler/hermes-lcm/releases)

**Lossless Context Management plugin for [Hermes Agent](https://github.com/NousResearch/hermes-agent)**

> Bounded context, unbounded memory. Nothing is ever lost.

Based on the [LCM paper](https://papers.voltropy.com/LCM) by Ehrlich & Blackman (Voltropy PBC, Feb 2026).
Inspired by [lossless-claw](https://github.com/martian-engineering/lossless-claw) for OpenClaw.

---

## The Problem

When active context fills up, agents usually replace older turns with a flat
summary. Details can fall out of the prompt, and recovery depends on a separate
history path the model may not use.

<p align="center">
  <img src="docs/standard_compression.png" alt="Standard compression" width="700">
</p>

## The Fix

Persist the conversation, compact old context into a hierarchical summary DAG,
and give the agent tools to drill back into the exact material that was
compacted.

<p align="center">
  <img src="docs/lcm_compression.png" alt="LCM compression" width="700">
</p>

<p align="center">
  <img src="docs/architecture.png" alt="Architecture" width="700">
</p>

## What It Does

- **SQLite message store** - preserves raw messages by default before compaction
- **Summary DAG** - compacts older context into depth-aware summary nodes
- **Bounded recovery** - pages raw messages, child summaries, and externalized payloads without flooding the main context
- **Agent tools** - `lcm_grep`, `lcm_describe`, `lcm_expand`, and `lcm_expand_query`
- **Source-aware retrieval** - filters raw rows and summaries by descendant source lineage
- **Session controls** - ignore noisy sessions or keep sessions read-only with glob patterns
- **Large payload controls** - optional ingest-time externalization for oversized tool/media/raw payloads, plus transcript GC for already-externalized tool results
- **Sensitive-pattern controls** - optional named redaction of API keys, bearer tokens, passwords, and private keys before LCM stores or summarizes them
- **Storage-boundary payload guard** - media-ish `data:*;base64` and long base64-looking strings are externalized before LCM writes them to SQLite
- **Diagnostics** - `lcm_status`, `lcm_doctor`, and optional `/lcm` slash commands

## LCM vs built-in compression

Hermes core may persist original conversation history in `state.db` before
built-in compression rewrites the active prompt. Built-in compression can still
be lossy in the active context, but previous content may be recoverable later
through host-level history tools such as `session_search`.

`hermes-lcm` is different because recall is part of the active context engine:

- plugin-local store and DAG built specifically for drill-down
- current-session retrieval through LCM tools, not an auxiliary cross-session search step
- explicit source-lineage and session-boundary rules

Position LCM around retrieval quality, autonomy, and drill-down behavior. Do not
claim that Hermes core has no persisted record of pre-compression history.

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

## Verify

Run:

```bash
hermes plugins
```

Expected signals:

- plugin list includes `hermes-lcm`
- selected context engine is `lcm`
- tool list includes `lcm_grep`, `lcm_load_session`, `lcm_describe`, `lcm_expand`, `lcm_expand_query`, `lcm_status`, and `lcm_doctor`

Typical output:

```text
Plugins (1):
  ✓ hermes-lcm v0.13.0 (7 tools)

Provider Plugins:
  Context Engine: lcm
```

For source checkouts, `lcm_status`, `/lcm status`, `lcm_doctor`, and
`/lcm doctor` also report the loaded plugin path and best-effort git identity:
`plugin_git_commit`, `plugin_git_branch`, and `plugin_git_dirty`.

## Troubleshooting

### `hermes plugins` shows `lcm (not found)` but LCM tools exist

If `plugins.enabled` contains `hermes-lcm`, `context.engine: lcm` is set, and
the runtime exposes LCM tools, LCM is loaded. The `lcm (not found)` line is a
Hermes host discovery/status mismatch, not an LCM storage or compaction failure.

### `/lcm status` looks unbound after restart

After a fresh Hermes restart, `/lcm status` may show `session_id: (unbound)` or
`threshold_tokens: (uninitialized)`. Send one normal Hermes message first, then
run `lcm_status` or `/lcm status` again for live per-session fields.

## Configuration

Most installs only need `plugins.enabled` and `context.engine: lcm`. Useful
environment variables:

| Variable | Default | Use |
|----------|---------|-----|
| `LCM_CONTEXT_THRESHOLD` | `0.75` | Fraction of the context window that triggers LCM compaction |
| `LCM_FRESH_TAIL_COUNT` | `64` | Recent messages protected from compaction |
| `LCM_LEAF_CHUNK_TOKENS` | `20000` | Raw-backlog floor before leaf compaction; with dynamic chunking enabled, the base chunk target |
| `LCM_DYNAMIC_LEAF_CHUNK_ENABLED` | `false` | Enable chunk-sized leaf compaction passes instead of compacting the whole non-tail raw backlog per pass |
| `LCM_DYNAMIC_LEAF_CHUNK_MAX` | `40000` | Upper bound for dynamic leaf chunk targets |
| `LCM_NEW_SESSION_RETAIN_DEPTH` | `2` | DAG depth retained after manual `/new` (`-1` all, `0` none) |
| `LCM_IGNORE_SESSION_PATTERNS` | empty | Comma-separated session globs excluded from LCM storage |
| `LCM_STATELESS_SESSION_PATTERNS` | empty | Comma-separated session globs kept read-only |
| `LCM_IGNORE_MESSAGE_PATTERNS` | empty | Comma-separated regex patterns; matching message content (plain text, extracted text parts for structured/multimodal content, or normalized JSON fallback when no text parts exist) is excluded from LCM storage |
| `LCM_SENSITIVE_PATTERNS_ENABLED` | `false` | Opt in to deterministic redaction before LCM storage, FTS indexing, summarization, active replay, and externalized ingest payloads |
| `LCM_SENSITIVE_PATTERNS` | `api_key,bearer_token,password_assignment,private_key` | Comma-separated named sensitive pattern catalog entries to apply when redaction is enabled |
| `LCM_LARGE_OUTPUT_EXTERNALIZATION_ENABLED` | `false` | Store oversized ingest payloads, including tool results, media blocks, and generic raw content, in plugin-managed JSON files |
| `LCM_LARGE_OUTPUT_EXTERNALIZATION_THRESHOLD_CHARS` | `12000` | Externalization threshold for normalized payload text |
| `LCM_LARGE_OUTPUT_TRANSCRIPT_GC_ENABLED` | `false` | Rewrite already-externalized summarized tool rows to compact placeholders |
| `LCM_CRITICAL_BUDGET_PRESSURE_RATIO` | `0.0` | Disabled at `0.0`; when set, permits critical-pressure bypasses for bounded deferred catch-up and cache-friendly follow-on condensation only |
| `LCM_SUMMARY_MODEL` | auxiliary | Override summarization model |
| `LCM_SUMMARY_FALLBACK_MODELS` | empty | Comma-separated summarization models tried after `LCM_SUMMARY_MODEL` or the auxiliary task default fails |
| `LCM_SUMMARY_CIRCUIT_BREAKER_FAILURE_THRESHOLD` | `2` | Consecutive failed summarization calls before a route is skipped temporarily |
| `LCM_SUMMARY_CIRCUIT_BREAKER_COOLDOWN_SECONDS` | `300` | Seconds to skip an open summary route before retrying it |
| `LCM_EXPANSION_MODEL` | summary model / auxiliary | Override `lcm_expand_query` synthesis model |
| `LCM_EXPANSION_CONTEXT_TOKENS` | `32000` | Context budget used by the auxiliary LLM for `lcm_expand_query` |
| `LCM_SUMMARY_TIMEOUT_MS` | `60000` | Timeout for one summarization call |
| `LCM_EXPANSION_TIMEOUT_MS` | `120000` | Timeout for one `lcm_expand_query` synthesis call |
| `LCM_DATABASE_PATH` | auto | SQLite database path, profile-scoped by default |
| `LCM_ENABLE_SLASH_COMMAND` | `false` | Enable the optional `/lcm` operator command surface |
| `LCM_DOCTOR_CLEAN_APPLY_ENABLED` | `false` | Permit destructive `/lcm doctor clean apply` in trusted operator contexts |

Advanced compaction, assembly, and extraction knobs are defined in `config.py`.

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
`lcm_status` and `lcm_doctor` expose the enabled state, configured pattern names,
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

If startup/status output shows a host-side compression percentage that disagrees
with LCM, trust live LCM status after a normal message has initialized the
session.

### Preset inspection and dry-run suggestions

Model-aware presets are inspectable, but they are not automatic live config
mutations. With `LCM_ENABLE_SLASH_COMMAND=true`, use:

```text
/lcm preset show codex_gpt_long_context
/lcm preset suggest
/lcm preset apply codex_gpt_long_context --dry-run
```

`/lcm preset show` reports the shipped preset metadata, benchmark provenance,
policy file, policy version, and metric summary. `/lcm preset suggest` chooses a
safe shipped suggestion for the current context window when one exists, and
labels the current selector as `context-only` when the provider/model family is
not available to the plugin. `/lcm preset apply ... --dry-run` previews env-var
settings only; it does not
write files, change process state, or override explicit parseable
preset-managed `LCM_*` environment variables. Invalid preset-managed env values
are reported as invalid instead of being treated as active runtime overrides.

The current `codex_gpt_long_context` dry-run preview is:

```text
LCM_CONTEXT_THRESHOLD=0.75
LCM_FRESH_TAIL_COUNT=24
LCM_LEAF_CHUNK_TOKENS=8000
```

`target_after_compaction=0.55` is still benchmark provenance, not a runtime
setting, because the engine does not expose that live knob yet.

For dashboards and agents, `lcm_status` also exposes the same information under
`preset_suggestion` as read-only JSON: suggested preset name/family, selection
reason, match confidence, provenance, explicit override diagnostics, invalid
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
- `LCM_EXPANSION_CONTEXT_TOKENS` controls how much recovered material
  `lcm_expand_query` may feed to the auxiliary model. It does not change what
  LCM stores.
- `LCM_LARGE_OUTPUT_EXTERNALIZATION_ENABLED=true` helps when large tool outputs,
  logs, media payloads, or raw JSON blobs dominate token pressure.

Common questions:

**Should I leave the default threshold on a 1M-token model?**

Not always. The default `0.75` means compaction may wait until roughly `750000`
prompt tokens on a true 1M effective window. That can be intentional if you want
maximum live context, but it is expensive and delays DAG construction.

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

### Large tool-output handling

Externalization for ordinary large tool output is opt-in. When enabled,
oversized tool results are written to plugin-managed JSON files and referenced
from summaries. They remain inspectable later through
`lcm_describe(externalized_ref=...)` and `lcm_expand(externalized_ref=...)`.

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

## Agent Tools

Use these tools for current-session recall after compaction. Use `session_search`
for earlier separate sessions or broad cross-session history.

| Tool | Use |
|------|-----|
| `lcm_grep` | Search current-session raw messages and summaries. Opt into `session_scope='all'` or `session_scope='session'` (with `session_id`) for bounded archive recovery over rows already present in `lcm.db`, including externally backfilled rows that may carry source strings such as `openclaw-lcm:*`; broader scopes return raw-message hits only. Raw-message filters `role`, `time_from`, and `time_to` are pushed into the search query; when any of them is supplied, summary hits are omitted so the filter contract stays exact. Use `session_search` for earlier separate sessions or broad cross-session recall. |
| `lcm_load_session` | Load one ordered raw-message transcript page for an explicit `session_id`. This is not search: it returns raw rows in `store_id` order, bounded by `limit`, with per-message content bounded by `max_content_chars`, and continues with `after_store_id` from `next_cursor`. |
| `lcm_describe` | Inspect the current-session DAG or preview an `externalized_ref` without loading full content. |
| `lcm_expand` | Recover source messages, child summaries, or externalized payloads with pagination. Use `store_id` to fetch a single raw message regardless of session, suitable for drilling into a cross-session `lcm_grep` result. |
| `lcm_expand_query` | Answer a question using expanded current-session LCM context while returning a bounded answer. |
| `lcm_status` | Show runtime health, context pressure, config, source lineage, and lifecycle stats. |
| `lcm_doctor` | Run database, FTS, lifecycle, config, and context-pressure diagnostics. |

### Retrieval contract

LCM retrieval tools default to current-session scope. `lcm_grep` accepts
`session_scope='all'` or `session_scope='session'` as an explicit opt-in for
bounded archive search over rows already present in `lcm.db` (raw-message hits
only). Once a session id is known, `lcm_load_session` can enumerate that session's
raw transcript in chronological `store_id` pages without a search query. Use
Hermes `session_search` for broad cross-session history outside the LCM database.

Within the current session, `source` filters raw rows directly and filters
summary nodes by descendant raw-message source lineage. `unknown` is a real
source value, not a wildcard. Legacy blank-source rows are treated as `unknown`.
`role`, `time_from`, and `time_to` are raw-message filters and are applied in the
message search query before result limiting. `time_from` and `time_to` accept Unix
seconds or timezone-aware ISO 8601 strings; naive ISO strings are rejected so the
same query means the same thing across machines. When a raw-message filter is
active, `lcm_grep` returns raw rows only and reports `summary_results_omitted`.

Carried-over summary nodes can become current-session content after `/new`, but
their source eligibility still comes from the descendant raw messages. Expanding
a carried-over current-session node recovers those original raw message sources
even when the sources still belong to the previous session.

### Lossless raw recovery contract

Tool responses are bounded so one retrieval call cannot flood the main context.
Lossless recovery means raw content is stored with stable source lineage and can
be recovered in deterministic pages.

- `lcm_expand(node_id=...)` pages immediate sources with `source_offset` and `source_limit`
- `lcm_load_session(session_id=...)` pages ordered raw session rows with `after_store_id` and `next_cursor`; each row includes bounded content plus truncation metadata, and large individual rows can be recovered with `lcm_expand(store_id=...)` using `content_offset`
- oversized raw messages continue with `content_offset`
- `lcm_expand(externalized_ref=...)` pages payload content with `content_offset`
- `lcm_expand_query` uses `context_max_tokens` for auxiliary context and reports truncation/pagination hints when needed

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
- `/lcm doctor repair` - read-only SQLite/FTS repair diagnostics
- `/lcm doctor repair apply` - backup-first SQLite/FTS repair
- `/lcm doctor source` - read-only scan for legacy blank-source rows
- `/lcm doctor source apply` - backup-first normalization of legacy blank-source rows to `unknown`
- `/lcm doctor retention` - read-only retention analysis
- `/lcm backup` - timestamped SQLite backup
- `/lcm rotate` - read-only preview of an in-place tail-preserving compact of the active session
- `/lcm rotate apply` - backup-first rotate that advances the lifecycle frontier past pre-tail raw messages
- `/lcm help` - command help

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

## How It Works

1. **Ingest** - persist each message in SQLite with FTS metadata
2. **Compact** - summarize older messages outside the fresh tail into D0 leaf nodes
3. **Condense** - merge same-depth nodes into higher-depth summaries
4. **Escalate** - shrink oversize summaries from detailed to bullets to deterministic truncate
5. **Assemble** - combine system prompt, highest-depth summaries, and fresh tail
6. **Retrieve** - use LCM tools to drill into compacted history or synthesize from expanded context

## Development

Important files:

```text
plugin.yaml      manifest
__init__.py      plugin registration and optional slash-command registration
engine.py        LCMEngine main orchestrator
store.py         SQLite message store and FTS
dag.py           summary DAG and FTS
config.py        env var defaults and overrides
command.py       /lcm command handlers
tools.py         lcm_grep, lcm_load_session, lcm_describe, lcm_expand, lcm_expand_query
schemas.py       tool schemas shown to the model
tests/           standalone pytest coverage
```

Run tests:

```bash
pip install pytest
python -m pytest tests/ -v
```

No Hermes Agent checkout is required for the test suite; tests include a
lightweight ABC stub.

## Contributing

Issues and PRs welcome. Bug fixes and correctness improvements are highest
priority. New features should be scoped, backwards-compatible, and tested.

See [CONTRIBUTING.md](CONTRIBUTING.md) for branch, validation, and PR guidance.
See the [releases page](https://github.com/stephenschoettler/hermes-lcm/releases) for changelogs.

## License

MIT

## Star History

<a href="https://www.star-history.com/?repos=stephenschoettler%2Fhermes-lcm&type=timeline&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=stephenschoettler/hermes-lcm&type=timeline&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=stephenschoettler/hermes-lcm&type=timeline&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=stephenschoettler/hermes-lcm&type=timeline&legend=top-left" />
 </picture>
</a>
