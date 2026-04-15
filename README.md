<p align="center">
  <img src="banner.png" alt="HERMES-LCM" width="800">
</p>

[![CI](https://github.com/stephenschoettler/hermes-lcm/actions/workflows/ci.yml/badge.svg)](https://github.com/stephenschoettler/hermes-lcm/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/stephenschoettler/hermes-lcm)](https://github.com/stephenschoettler/hermes-lcm/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Lossless Context Management plugin for [Hermes Agent](https://github.com/NousResearch/hermes-agent)**

> Bounded context, unbounded memory. Nothing is ever lost.

Based on the [LCM paper](https://papers.voltropy.com/LCM) by Ehrlich & Blackman (Voltropy PBC, Feb 2026).
Inspired by [lossless-claw](https://github.com/martian-engineering/lossless-claw) for OpenClaw.

---

## The Problem

When context fills up, agents replace your conversation with a flat lossy summary.
Details get lost. The model confidently misremembers. No way to go back.

```
Standard compression:

  Turn 1-50:  "You discussed database schema changes and deployment plans"
  Turn 51+:   (live conversation)

  You: "What column type did we decide on for the user_events table?"
  Agent: "I believe we discussed using JSONB" ← wrong, it was an integer array
```

## The Fix

Every message persisted. Hierarchical DAG summaries. Agent tools to drill back
into anything that was compacted.

```
LCM compression:

  D2 summary: project overview, key decisions (weeks of context)
  D1 summary: database schema discussion, deployment checklist (hours)
  D0 summary: specific column decisions, migration commands (minutes)
  Fresh tail: last 64 messages (live, never compacted)

  You: "What column type did we decide on for the user_events table?"
  Agent: [lcm_grep: "user_events column type"] → integer[] with GIN index
```

```
  Active Context                    Summary DAG
 ┌────────────────────┐
 │  System Prompt     │            ┌──────────┐
 │  (with LCM note)   │            │ D2 node  │ ← weeks
 ├────────────────────┤            ├──────────┤
 │  DAG Summaries     │ ─────────  │ D1 node  │ ← hours
 │  (highest depth)   │            │ D1 node  │
 ├────────────────────┤            ├──────────┤
 │  Fresh Tail        │            │ D0 node  │ ← minutes
 │  (last 64 msgs)    │            │ D0 node  │
 │  (never compacted) │            │ D0 node  │
 └────────────────────┘            └──────────┘
                                        │
                                        ▼
                              ┌────────────────────┐
                              │ Immutable Store    │
                              │ (SQLite + FTS5)    │
                              │ Every msg verbatim │
                              └────────────────────┘
```

## Why This Plugin

~3,200 lines of Python. Zero external dependencies. 105 tests that run standalone in under a second. Full lossless context management — immutable store, hierarchical DAG, agent retrieval tools, guided compression, assembly guardrails, session filtering — in a single lightweight plugin that drops into Hermes with no build step, no runtime overhead, and nothing to configure beyond `context.engine: lcm`.

## What It Does

- **Immutable store** — every message persisted in SQLite, never modified
- **Summary DAG** — hierarchical compaction (D0 minutes → D1 hours → D2 days)
- **3-level escalation** — L1 detailed → L2 bullets → L3 deterministic truncate (guaranteed convergence)
- **Agent tools** — `lcm_grep`, `lcm_describe`, `lcm_expand`, `lcm_expand_query` for structured retrieval
- **Current-turn search** — live messages ingested before tool execution
- **Session filtering** — exclude noisy sessions entirely or mark them read-only with glob patterns
- **Profile-scoped** — separate DB per Hermes profile

## Requirements

- Hermes Agent with the **pluggable context engine slot** ([PR #7464](https://github.com/NousResearch/hermes-agent/pull/7464))
- Python 3.11+
- No additional dependencies (uses Hermes auxiliary LLM for summarization)

## Install

```bash
# Clone into the context engine plugin directory
git clone https://github.com/stephenschoettler/hermes-lcm \
  ~/.hermes/hermes-agent/plugins/context_engine/lcm

# Or for a specific profile
git clone https://github.com/stephenschoettler/hermes-lcm \
  ~/.hermes/profiles/myprofile/hermes-agent/plugins/context_engine/lcm
```

> **Note:** Context engines must be installed under `plugins/context_engine/<name>/`,
> not `plugins/<name>/`. The general `~/.hermes/plugins/` directory is for tools,
> hooks, and CLI extensions — context engines are discovered separately.

Restart Hermes. Activate the engine — either via the interactive UI or config file:

**Option A — `hermes plugins` UI:**

```
hermes plugins
```

The composite plugins screen shows provider categories at the bottom.
Select **Context Engine** and pick `lcm` from the radiolist.

**Option B — config.yaml:**

```yaml
context:
  engine: lcm
```

Verify with `hermes plugins`:

```
Plugins (1):
  ✓ hermes-lcm v0.2.1 (6 tools)

Provider Plugins:
  Context Engine: lcm
```

## Configuration

Environment variables (all optional):

| Variable | Default | Description |
|----------|---------|-------------|
| `LCM_FRESH_TAIL_COUNT` | `64` | Recent messages protected from compaction |
| `LCM_LEAF_CHUNK_TOKENS` | `20000` | Token threshold for leaf compaction |
| `LCM_CONTEXT_THRESHOLD` | `0.75` | Fraction of context window triggering compaction |
| `LCM_INCREMENTAL_MAX_DEPTH` | `1` | Max condensation depth (`0` = disabled, `-1` = unlimited) |
| `LCM_CONDENSATION_FANIN` | `4` | Same-depth nodes needed to trigger condensation |
| `LCM_IGNORE_SESSION_PATTERNS` | *(empty)* | Comma-separated glob patterns for sessions to exclude from LCM storage entirely |
| `LCM_STATELESS_SESSION_PATTERNS` | *(empty)* | Comma-separated glob patterns for sessions that stay read-only (`platform:session_id` matching supported) |
| `LCM_SUMMARY_MODEL` | *(auxiliary)* | Override model for summarization |
| `LCM_EXPANSION_MODEL` | *(summary model / auxiliary)* | Override model for `lcm_expand_query` synthesis |
| `LCM_SUMMARY_TIMEOUT_MS` | `60000` | Timeout for a single model-backed summarization call |
| `LCM_EXPANSION_TIMEOUT_MS` | `120000` | Timeout for `lcm_expand_query` answer synthesis |
| `LCM_DATABASE_PATH` | `~/.hermes/lcm.db` | SQLite database path (auto profile-scoped) |
| `LCM_NEW_SESSION_RETAIN_DEPTH` | `2` | DAG depth retained after `/new` (`-1` = all, `0` = none, `2` = keep d2+) |

Pattern syntax matches `lossless-claw`:
- `*` matches within one colon-delimited segment
- `**` can span across colons

Hermes currently matches each pattern against multiple candidate keys for flexibility:
- raw `session_id`
- `platform`
- `platform:session_id`

That means patterns like `cron:*` can catch Hermes cron sessions today, while plain raw session-id matching still works if you know the exact IDs you want to target.

## Agent Tools

| Tool | Description |
|------|-------------|
| `lcm_grep` | Search raw messages AND summaries across all depths. FTS5 syntax. |
| `lcm_describe` | Inspect DAG structure — token counts, children, expand hints. No node_id = session overview. |
| `lcm_expand` | Recover original content from a summary node. Token-budgeted. |
| `lcm_expand_query` | Answer a question from expanded LCM context using either a query or explicit node_ids. Uses the expansion path/model instead of the summarization path. |
| `lcm_status` | Quick health overview — compression count, store size, DAG depth distribution, context usage, and active config. |
| `lcm_doctor` | Run diagnostics — database integrity, FTS index sync, orphaned nodes, config validation, context pressure. |

## Gateway Slash Commands

When Hermes host support for plugin slash commands is available, `hermes-lcm` also exposes a `/lcm` operator surface for quick read-only diagnostics from chat:

- `/lcm` or `/lcm status` — current session/runtime status
- `/lcm doctor` — SQLite + FTS health checks and store/node totals
- `/lcm doctor clean` — best-effort read-only scan for obvious junk/noise sessions matched from stored session keys
- `/lcm backup` — create a timestamped SQLite backup before any future cleanup workflow

`/lcm doctor clean apply` is intentionally not implemented yet. This slice is diagnostics-first and backup-first.

## How It Works

1. **Ingest** — every message persisted verbatim in an immutable SQLite store
2. **Compact** — when context pressure builds, older messages outside the fresh tail are summarized into D0 leaf nodes
3. **Condense** — when enough D0 nodes accumulate, they're condensed into D1 nodes (and so on up)
4. **Escalate** — if a summary is too long, escalate: L1 detailed → L2 bullets → L3 deterministic truncate
5. **Assemble** — active context = system prompt + highest-depth summaries + fresh tail
6. **Retrieve** — agent uses `lcm_grep`/`lcm_describe`/`lcm_expand`/`lcm_expand_query` to drill into compacted history or synthesize answers from expanded context

## Architecture

```
hermes-lcm/
├── plugin.yaml      # manifest
├── __init__.py      # register(ctx) → ctx.register_context_engine()
├── engine.py        # LCMEngine(ContextEngine) — main orchestrator
├── store.py         # immutable SQLite message store (FTS5)
├── dag.py           # summary DAG with depth-aware nodes (FTS5)
├── escalation.py    # L1 → L2 → L3 guaranteed convergence
├── config.py        # LCMConfig + env var overrides
├── command.py       # /lcm slash command handlers for gateway diagnostics
├── tokens.py        # tiktoken with char-based fallback
├── schemas.py       # tool schemas (what the LLM sees)
├── tools.py         # tool handlers (lcm_grep, lcm_describe, lcm_expand, lcm_expand_query)
└── tests/           # standalone pytest coverage
```

**Running tests:**

```bash
pip install pytest
python -m pytest tests/ -v
```

No Hermes Agent checkout required — the test suite includes a lightweight ABC stub so it runs standalone.

## Context Engine Slot

Requires the **pluggable context engine slot** — an ABC (`ContextEngine`) in
hermes-agent core that makes the `ContextCompressor` swappable via the plugin
system. Config-driven selection via `context.engine` in config.yaml, with a
`plugins/context_engine/` discovery directory. Same pattern as OpenClaw's
`contextEngine` slot + `lossless-claw`.

- **PR:** [NousResearch/hermes-agent#7464](https://github.com/NousResearch/hermes-agent/pull/7464) (supersedes [#6126](https://github.com/NousResearch/hermes-agent/pull/6126), [#5700](https://github.com/NousResearch/hermes-agent/pull/5700))
- **Issue:** [NousResearch/hermes-agent#5701](https://github.com/NousResearch/hermes-agent/issues/5701) (closed by #7464)
- **Paper:** [papers.voltropy.com/LCM](https://papers.voltropy.com/LCM)

## Contributing

Issues and PRs welcome. This project has active community contributors and CI runs on every push and PR.

- **Bug fixes** and **correctness improvements** are always top priority
- **New features** should be scoped and backwards-compatible
- **Tests required** — run `python -m pytest tests/ -v` before submitting

See the [releases page](https://github.com/stephenschoettler/hermes-lcm/releases) for changelogs.

## License

MIT
