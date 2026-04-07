# hermes-lcm — Lossless Context Management for Hermes Agent

A context engine plugin that replaces Hermes Agent's built-in compressor with
DAG-based summarization. Every message is preserved, every detail recoverable.

Based on the [LCM paper](https://papers.voltropy.com/LCM) by Ehrlich & Blackman
(Voltropy PBC, Feb 2026).

## What it does

- **Persists every message** in an immutable SQLite store
- **Summarizes** older messages into a hierarchical DAG (depth 0–N)
- **3-level escalation** guarantees convergence (L1 detailed → L2 bullets → L3 truncate)
- **Agent tools** (`lcm_grep`, `lcm_describe`, `lcm_expand`) let the agent search
  and drill into compacted history
- **Nothing is ever lost** — raw messages stay in the store, summaries link back
  to sources

## Requirements

- Hermes Agent with the **context engine plugin slot** (PR pending)
- Python 3.11+
- No additional dependencies (uses Hermes auxiliary LLM for summarization)

## Install

```bash
# Copy to Hermes plugins directory
cp -r hermes-lcm ~/.hermes/plugins/hermes-lcm

# Or symlink for development
ln -s /path/to/hermes-lcm ~/.hermes/plugins/hermes-lcm
```

Restart Hermes. You should see `hermes-lcm` in `/plugins`.

## Configuration

Environment variables (all optional):

| Variable | Default | Description |
|----------|---------|-------------|
| `LCM_FRESH_TAIL_COUNT` | `64` | Recent messages protected from compaction |
| `LCM_LEAF_CHUNK_TOKENS` | `20000` | Token threshold for leaf compaction |
| `LCM_CONTEXT_THRESHOLD` | `0.75` | Fraction of context window triggering compaction |
| `LCM_INCREMENTAL_MAX_DEPTH` | `1` | Max condensation depth (0=leaf only, -1=unlimited) |
| `LCM_CONDENSATION_FANIN` | `4` | Same-depth nodes needed to trigger condensation |
| `LCM_SUMMARY_MODEL` | *(auxiliary)* | Override model for summarization |
| `LCM_DATABASE_PATH` | `~/.hermes/lcm.db` | SQLite database path |
| `LCM_NEW_SESSION_RETAIN_DEPTH` | `2` | DAG depth retained after `/new` |

## Agent Tools

| Tool | Description |
|------|-------------|
| `lcm_grep` | Search raw messages AND summaries across all depths |
| `lcm_describe` | Inspect DAG structure — token counts, children, expand hints |
| `lcm_expand` | Recover original content from a summary node |

## How it works

```
Fresh Tail (protected)          Summary DAG
┌─────────────────────┐        ┌──────────┐
│ Recent N messages    │        │ D2 node  │ ← condensation of D1
│ (always in context)  │        │          │
└─────────────────────┘        ├──────────┤
                               │ D1 node  │ ← condensation of D0
                               │ D1 node  │
                               ├──────────┤
                               │ D0 node  │ ← summary of raw messages
                               │ D0 node  │
                               │ D0 node  │
                               └──────────┘
                                    ↓
                           Immutable Store (SQLite)
                           Every message verbatim
```

Active context sent to the LLM:
1. System prompt (with LCM note)
2. Highest-depth uncondensed summaries
3. Fresh tail messages

## License

MIT
