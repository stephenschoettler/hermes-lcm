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
  ✓ hermes-lcm v0.18.1 (7 tools)

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

### Startup log mentions `context-engine schemas` or `Path B fallback`

This is expected on older Hermes hosts that do not advertise
`context_engine_tool_handlers_receive_messages`, including Hermes Agent v0.16.
LCM tools are still available through the context-engine schema/dispatch path
(Path B). The plugin intentionally avoids standalone plugin-registry tool
registration (Path A) on those hosts because Path A would shadow Path B and lose
current-turn ingest.

Healthy signals are the same as above: selected context engine `lcm`, the seven
`lcm_*` tools in the live tool list, and `lcm_status` / `lcm_doctor` responding
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
| `LCM_FRESH_TAIL_COUNT` | `32` | Recent messages protected from compaction |
| `LCM_INCREMENTAL_MAX_DEPTH` | `3` | Max DAG condensation depth (`-1` = unlimited, `0` = leaf only); enables hierarchical summarization |
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
| `LCM_DATABASE_PATH` | auto | SQLite database path. Empty config resolves to `HERMES_HOME/lcm.db`; plugin installs or operators may set this env var to another profile-scoped path such as `~/.hermes/hermes-lcm.db`. |
| `LCM_FTS_INTEGRITY_CHECK_INTERVAL_HOURS` | `24` | Minimum hours between startup FTS5 deep integrity-checks (O(index size)). `0` checks every startup (previous behavior); a negative value never checks on startup. Structural checks always run regardless. |
| `LCM_ENABLE_SLASH_COMMAND` | `false` | Enable the optional `/lcm` operator command surface |
| `LCM_DOCTOR_CLEAN_APPLY_ENABLED` | `false` | Permit destructive `/lcm doctor clean apply` in trusted operator contexts |
| `LCM_EMPTY_LIFECYCLE_GC_ENABLED` | `true` | Master toggle for automatic pruning of lifecycle rows for sessions that never ingested any messages or summary nodes |
| `LCM_EMPTY_LIFECYCLE_GC_THRESHOLD` | `200` | Number of lifecycle rows at which the GC pass fires (default 200 so fresh installs skip the work) |
| `LCM_EMPTY_LIFECYCLE_GC_MAX_AGE_HOURS` | `24` | Automatic GC only deletes empty lifecycle rows at least this old; set `0` only in trusted/test environments that intentionally want immediate empty-row pruning |

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

## Related references

- [Retrieval tools reference](retrieval-tools.md)
- [Architecture notes](architecture.md)
- [Benchmarking and stress checks](../benchmarks/README.md)
- [Release validation](release-validation.md)
- [Packaging and distribution posture](packaging.md)
