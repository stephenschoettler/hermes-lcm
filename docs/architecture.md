# Architecture notes

This page keeps the implementation model and product-positioning nuance outside the quickstart README while preserving the details new operators and reviewers need.

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

## Related references

- [Operator guide](operator-guide.md)
- [Retrieval tools reference](retrieval-tools.md)
- [Opt-in async/background compaction design](async-background-compaction.md)
- [Benchmarking and stress checks](../benchmarks/README.md)
