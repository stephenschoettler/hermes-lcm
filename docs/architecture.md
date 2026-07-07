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
- **Diagnostics** - `lcm_status`, `lcm_inspect`, `lcm_doctor`, and optional `/lcm` slash commands

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

Important files, grouped by concern:

```text
Plugin surface
  plugin.yaml            manifest
  __init__.py            plugin registration and optional slash-command registration
  schemas.py             tool schemas shown to the model
  tools.py               lcm_grep, lcm_load_session, lcm_describe, lcm_expand, lcm_expand_query
  command.py             /lcm command handlers
  config.py              env var defaults and overrides
  presets.py             operator config presets

Engine
  engine.py              LCMEngine orchestrator (composes the mixins below)
  compaction.py          CompactionMixin — should_compress / compress leaf pipeline
  reconcile.py           ReconcileMixin — post-restart ingest-cursor reconciliation and replay identity
  aux_session.py         AuxiliarySessionMixin — auxiliary session id / context metadata helpers
  placeholder_ledger.py  PlaceholderLedgerMixin — ignored-active-replay placeholder bookkeeping
  engine_registry.py     process-wide active-clone registry (resolve_active_lcm_engine)
  runtime_identity.py    plugin/git identity for status and doctor
  codex_routing.py       Codex OAuth route detection and effective context caps
  sqlite_util.py         SQLite lock-contention / busy-timeout helpers
  message_analysis.py    tool-call-id pairing and synthetic-noise detection
  sanitize.py            active-context sanitizers
  maintenance.py         backup / rotate maintenance ops

Storage and lifecycle
  store.py               SQLite message store and FTS
  dag.py                 summary DAG and FTS
  lifecycle_state.py     lifecycle/frontier state store
  db_bootstrap.py        schema bootstrap and migrations
  diagnostics.py         state-db path containment and doctor helpers

Ingest, content and retrieval
  ingest_protection.py   redaction, externalized-payload and persisted-output protection
  externalize.py         large-payload externalization
  extraction.py          pre-compaction content extraction
  escalation.py          summarize-with-escalation auxiliary routing
  model_routing.py       LCM auxiliary model-override routing
  tokens.py              token counting
  message_content.py     content normalization helpers
  message_patterns.py    message-pattern matching
  session_patterns.py    session-pattern matching
  search_query.py        retrieval query parsing

tests/                   standalone pytest coverage
```

### Engine decomposition

`engine.py` is being decomposed into cohesive, behaviour-preserving modules.
Cohesive groups of stateful `LCMEngine` methods are lifted verbatim into
`*Mixin` classes in their own files (`compaction.py`, `reconcile.py`,
`aux_session.py`, `placeholder_ledger.py`, …) and mixed back into `LCMEngine`.
Because the methods still run bound to the engine instance (`self` is the
`LCMEngine`), they read and write the same runtime state and call the same
sibling helpers through normal attribute lookup — so the split changes file
layout only, not behaviour or the public surface. Pure/helper groups are
extracted as plain module functions instead of mixins (`engine_registry.py`,
`codex_routing.py`, `sqlite_util.py`, `runtime_identity.py`,
`message_analysis.py`).

**Mixin ordering (MRO).** LCM decomposition mixins are listed *before*
`ContextEngine` in the bases — `class LCMEngine(SomeMixin, …, ContextEngine)`.
`ContextEngine` defines the protocol methods `compress`, `should_compress`, and
`should_compress_preflight`; `CompactionMixin` overrides them, so it must precede
`ContextEngine` to win the MRO. The other mixins hold only private methods with no
`ContextEngine` counterpart, so their position is not load-bearing, but the same
"mixins first" convention is kept for all of them so the bases line stays correct
under any merge order.

**Test note.** Some engine tests patch a module-level function on
`hermes_lcm.engine` (for example `summarize_with_escalation` or the `count_*`
token helpers) to intercept a call. When a method that calls such a function is
moved to a mixin module, the test must patch that name on the mixin's module
instead — the moved method resolves the name from its own module. Methods whose
tests rely on this (notably `_summarize_leaf_chunk_with_rescue`) are deliberately
kept on `LCMEngine` and reached from the mixin via `self`.

**Status.** The decomposition is partial and ongoing: the compaction, reconcile,
placeholder-ledger, auxiliary-session and active-engine-registry concerns have
been extracted; larger clusters that remain on `LCMEngine` — session lifecycle
(`on_session_start` / `on_session_end` / rollover), context assembly, the ingest
pipeline, and the LCM-bypass host-fallback path — are candidates for later,
equally behaviour-preserving extraction.

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
