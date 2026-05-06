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
- **Large output controls** - optional externalization and transcript GC for oversized tool results
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
- No required third-party runtime dependencies. `tiktoken` is used if available; otherwise LCM falls back to character-based token estimates.

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
- tool list includes `lcm_grep`, `lcm_describe`, `lcm_expand`, `lcm_expand_query`, `lcm_status`, and `lcm_doctor`

Typical output:

```text
Plugins (1):
  ✓ hermes-lcm v0.8.0 (6 tools)

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
| `LCM_LEAF_CHUNK_TOKENS` | `20000` | Token floor for leaf compaction chunks |
| `LCM_NEW_SESSION_RETAIN_DEPTH` | `2` | DAG depth retained after manual `/new` (`-1` all, `0` none) |
| `LCM_IGNORE_SESSION_PATTERNS` | empty | Comma-separated session globs excluded from LCM storage |
| `LCM_STATELESS_SESSION_PATTERNS` | empty | Comma-separated session globs kept read-only |
| `LCM_LARGE_OUTPUT_EXTERNALIZATION_ENABLED` | `false` | Store oversized tool outputs in plugin-managed JSON files |
| `LCM_LARGE_OUTPUT_EXTERNALIZATION_THRESHOLD_CHARS` | `12000` | Externalization threshold for tool output text |
| `LCM_LARGE_OUTPUT_TRANSCRIPT_GC_ENABLED` | `false` | Rewrite already-externalized summarized tool rows to compact placeholders |
| `LCM_SUMMARY_MODEL` | auxiliary | Override summarization model |
| `LCM_EXPANSION_MODEL` | summary model / auxiliary | Override `lcm_expand_query` synthesis model |
| `LCM_EXPANSION_CONTEXT_TOKENS` | `32000` | Context budget used by the auxiliary LLM for `lcm_expand_query` |
| `LCM_SUMMARY_TIMEOUT_MS` | `60000` | Timeout for one summarization call |
| `LCM_EXPANSION_TIMEOUT_MS` | `120000` | Timeout for one `lcm_expand_query` synthesis call |
| `LCM_DATABASE_PATH` | auto | SQLite database path, profile-scoped by default |
| `LCM_ENABLE_SLASH_COMMAND` | `false` | Enable the optional `/lcm` operator command surface |
| `LCM_DOCTOR_CLEAN_APPLY_ENABLED` | `false` | Permit destructive `/lcm doctor clean apply` in trusted operator contexts |

Advanced compaction, assembly, and extraction knobs are defined in `config.py`.

### Threshold ownership

When `context.engine: lcm` is active, `LCM_CONTEXT_THRESHOLD` is the compaction
threshold LCM uses. Hermes core `compression.threshold` belongs to the built-in
compressor. Hermes core `compression.enabled` is still the global gate that
allows compaction, so leave it enabled when using LCM.

If startup/status output shows a host-side compression percentage that disagrees
with LCM, trust live LCM status after a normal message has initialized the
session.

### Session pattern syntax

Pattern matching checks multiple keys: raw `session_id`, `platform`, and
`platform:session_id`.

- `*` matches within one colon-delimited segment
- `**` can span across colons

Example: `cron:*` can match Hermes cron sessions, while exact raw session IDs
still work.

### Large tool-output handling

Externalization is opt-in. When enabled, oversized tool results are written to
plugin-managed JSON files and referenced from summaries. They remain inspectable
later through `lcm_describe(externalized_ref=...)` and
`lcm_expand(externalized_ref=...)`.

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
| `lcm_grep` | Search current-session raw messages and summaries. Opt into `session_scope='all'` or `session_scope='session'` (with `session_id`) for bounded archive recovery over rows already present in `lcm.db`, including externally backfilled rows that may carry source strings such as `openclaw-lcm:*`; broader scopes return raw-message hits only. Use `session_search` for earlier separate sessions or broad cross-session recall. |
| `lcm_describe` | Inspect the current-session DAG or preview an `externalized_ref` without loading full content. |
| `lcm_expand` | Recover source messages, child summaries, or externalized payloads with pagination. Use `store_id` to fetch a single raw message regardless of session, suitable for drilling into a cross-session `lcm_grep` result. |
| `lcm_expand_query` | Answer a question using expanded current-session LCM context while returning a bounded answer. |
| `lcm_status` | Show runtime health, context pressure, config, source lineage, and lifecycle stats. |
| `lcm_doctor` | Run database, FTS, lifecycle, config, and context-pressure diagnostics. |

### Retrieval contract

LCM retrieval tools default to current-session scope. `lcm_grep` accepts
`session_scope='all'` or `session_scope='session'` as an explicit opt-in for
bounded archive search over rows already present in `lcm.db` (raw-message hits
only); use Hermes `session_search` for broad cross-session history outside the
LCM database.

Within the current session, `source` filters raw rows directly and filters
summary nodes by descendant raw-message source lineage. `unknown` is a real
source value, not a wildcard. Legacy blank-source rows are treated as `unknown`.

Carried-over summary nodes can become current-session content after `/new`, but
their source eligibility still comes from the descendant raw messages.

### Lossless raw recovery contract

Tool responses are bounded so one retrieval call cannot flood the main context.
Lossless recovery means raw content is stored with stable source lineage and can
be recovered in deterministic pages.

- `lcm_expand(node_id=...)` pages immediate sources with `source_offset` and `source_limit`
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
- `/lcm help` - command help

Apply paths are intentionally narrow and backup-first. Start with diagnostics
before cleanup or repair.

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
tools.py         lcm_grep, lcm_describe, lcm_expand, lcm_expand_query
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
