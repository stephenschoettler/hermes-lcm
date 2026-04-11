```
██   ██ ███████ ██████  ███    ███ ███████ ███████       ██       ██████ ███    ███
██   ██ ██      ██   ██ ████  ████ ██      ██            ██      ██      ████  ████
███████ █████   ██████  ██ ████ ██ █████   ███████ █████ ██      ██      ██ ████ ██
██   ██ ██      ██   ██ ██  ██  ██ ██           ██       ██      ██      ██  ██  ██
██   ██ ███████ ██   ██ ██      ██ ███████ ███████       ███████  ██████ ██      ██
```

**Lossless Context Management plugin for [Hermes Agent](https://github.com/NousResearch/hermes-agent)**

> Bounded context, unbounded memory. Nothing is ever lost.

Based on the [LCM paper](https://papers.voltropy.com/LCM) by Ehrlich & Blackman (Voltropy PBC, Feb 2026).
Inspired by [lossless-claw](https://github.com/martian-engineering/lossless-claw) for OpenClaw.

---

## The Problem

When context fills up, agents replace your conversation with a flat lossy summary.
Details get lost. The model confidently misremembers. No way to go back.

## The Fix

Every message persisted. Hierarchical DAG summaries. Agent tools to drill back
into anything that was compacted.

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

## What It Does

- **Immutable store** — every message persisted in SQLite, never modified
- **Summary DAG** — hierarchical compaction (D0 minutes → D1 hours → D2 days)
- **3-level escalation** — L1 detailed → L2 bullets → L3 deterministic truncate (guaranteed convergence)
- **Agent tools** — `lcm_grep`, `lcm_describe`, `lcm_expand` for structured retrieval
- **Current-turn search** — live messages ingested before tool execution
- **Profile-scoped** — separate DB per Hermes profile

## Requirements

- Hermes Agent with the **pluggable context engine slot** ([PR #7464](https://github.com/NousResearch/hermes-agent/pull/7464))
- Python 3.11+
- No additional dependencies (uses Hermes auxiliary LLM for summarization)

## Install

```bash
# Clone into your Hermes plugins directory
git clone https://github.com/stephenschoettler/hermes-lcm ~/.hermes/plugins/hermes-lcm

# Or for a specific profile
git clone https://github.com/stephenschoettler/hermes-lcm ~/.hermes/profiles/myprofile/plugins/hermes-lcm
```

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
  ✓ hermes-lcm v0.1.0 (3 tools, 0 hooks)

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
| `LCM_SUMMARY_MODEL` | *(auxiliary)* | Override model for summarization |
| `LCM_DATABASE_PATH` | `~/.hermes/lcm.db` | SQLite database path (auto profile-scoped) |
| `LCM_NEW_SESSION_RETAIN_DEPTH` | `2` | DAG depth retained after `/new` (`-1` = all, `0` = none, `2` = keep d2+) |

## Agent Tools

| Tool | Description |
|------|-------------|
| `lcm_grep` | Search raw messages AND summaries across all depths. FTS5 syntax. |
| `lcm_describe` | Inspect DAG structure — token counts, children, expand hints. No node_id = session overview. |
| `lcm_expand` | Recover original content from a summary node. Token-budgeted. |

## How It Works

1. **Ingest** — every message persisted verbatim in an immutable SQLite store
2. **Compact** — when context pressure builds, older messages outside the fresh tail are summarized into D0 leaf nodes
3. **Condense** — when enough D0 nodes accumulate, they're condensed into D1 nodes (and so on up)
4. **Escalate** — if a summary is too long, escalate: L1 detailed → L2 bullets → L3 deterministic truncate
5. **Assemble** — active context = system prompt + highest-depth summaries + fresh tail
6. **Retrieve** — agent uses `lcm_grep`/`lcm_describe`/`lcm_expand` to drill into compacted history

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
├── tokens.py        # tiktoken with char-based fallback
├── schemas.py       # tool schemas (what the LLM sees)
├── tools.py         # tool handlers (lcm_grep, lcm_describe, lcm_expand)
└── tests/           # 35 tests
```

Requires the **pluggable context engine slot** — an ABC (`ContextEngine`) in
hermes-agent core that makes the `ContextCompressor` swappable via the plugin
system. Config-driven selection via `context.engine` in config.yaml, with a
`plugins/context_engine/` discovery directory. Same pattern as OpenClaw's
`contextEngine` slot + `lossless-claw`.

- **PR:** [NousResearch/hermes-agent#7464](https://github.com/NousResearch/hermes-agent/pull/7464) (supersedes [#6126](https://github.com/NousResearch/hermes-agent/pull/6126), [#5700](https://github.com/NousResearch/hermes-agent/pull/5700))
- **Issue:** [NousResearch/hermes-agent#5701](https://github.com/NousResearch/hermes-agent/issues/5701) (closed by #7464)
- **Paper:** [papers.voltropy.com/LCM](https://papers.voltropy.com/LCM)

## License

MIT
