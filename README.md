<p align="center">
  <img src="docs/banner.png" alt="HERMES-LCM" width="800">
</p>

[![CI](https://github.com/stephenschoettler/hermes-lcm/actions/workflows/ci.yml/badge.svg)](https://github.com/stephenschoettler/hermes-lcm/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/stephenschoettler/hermes-lcm)](https://github.com/stephenschoettler/hermes-lcm/releases)

**Lossless Context Management plugin for [Hermes Agent](https://github.com/NousResearch/hermes-agent)**

> Bounded context, unbounded memory. Nothing is ever lost.

Based on the [LCM paper](https://papers.voltropy.com/LCM) by Ehrlich & Blackman (Voltropy PBC, Feb 2026). Inspired by [lossless-claw](https://github.com/martian-engineering/lossless-claw) for OpenClaw.

---

## Why this exists

When active context fills up, agents usually replace older turns with a flat summary. Details can fall out of the prompt, and recovery depends on a separate history path the model may not use.

`hermes-lcm` persists the conversation, compacts older context into a hierarchical summary DAG, and gives the agent tools to drill back into the exact material that was compacted.

<p align="center">
  <img src="docs/standard_compression.png" alt="Standard compression" width="700">
</p>

<p align="center">
  <img src="docs/lcm_compression.png" alt="LCM compression" width="700">
</p>

## What you get

- SQLite message store for raw-message preservation before compaction
- Summary DAG for depth-aware long-session condensation
- Recovery tools: `lcm_grep`, `lcm_load_session`, `lcm_describe`, `lcm_expand`, `lcm_expand_query`
- Health tools: `lcm_status`, `lcm_doctor`, plus optional `/lcm` slash commands
- Source-aware retrieval, session controls, large-payload externalization, and optional sensitive-pattern redaction
- Storage-boundary payload guard for media-ish `data:*;base64` values and oversized base64-like payloads before SQLite writes

## LCM vs built-in compression

Hermes core may persist original conversation history in `state.db` before built-in compression rewrites the active prompt. Built-in compression can still be lossy in the active context, but previous content may be recoverable later through host-level history tools such as `session_search`.

`hermes-lcm` is different because recall is part of the active context engine: it uses a plugin-local store and DAG built specifically for drill-down, current-session retrieval through LCM tools, and explicit source-lineage/session-boundary rules. Position LCM around retrieval quality, autonomy, and drill-down behavior, not around the false claim that Hermes core has no persisted pre-compression record.

## Retrieval and storage contracts

LCM tool recall defaults to current-session recall. `lcm_grep` can explicitly opt into bounded archive recovery with `session_scope='all'` for every session in the local LCM database, or `session_scope='session'` with a `session_id` for one known session. Broader LCM scopes only cover historical rows already present in `lcm.db`; use Hermes `session_search` for Hermes-tracked transcript history outside the plugin-local database.

Lossless raw recovery contract: use `lcm_load_session` to enumerate a known session in chronological pages with `after_store_id`, then use `lcm_expand` with `source_offset` or `content_offset` to recover bounded original detail. `lcm_expand_query` uses the fresh context budget configured by `LCM_EXPANSION_CONTEXT_TOKENS` when expanding enough evidence for synthesis.

Storage-boundary payload guard: LCM externalizes media-ish `data:*;base64` payloads and long base64-looking blobs before they cross the SQLite storage boundary. The guard covers raw message `messages.content` and assistant `messages.tool_calls` payloads so large binary-ish data does not get embedded directly in `lcm.db` rows. Doctor output is metadata-only: payload-boundary diagnostics report counts/paths, not raw payload bytes. If bytes already landed in Hermes `state.db`, that is upstream/outside LCM scope; use backup-first cleanup or migration procedures before mutating historical host rows.

## Requirements

- Hermes Agent with the pluggable context engine slot ([PR #7464](https://github.com/NousResearch/hermes-agent/pull/7464))
- Python 3.11+
- No required third-party runtime dependencies. `tiktoken` and `regex` are optional accelerators/guards.

## Quickstart

Clone as a general user plugin:

```bash
git clone https://github.com/stephenschoettler/hermes-lcm \
  ~/.hermes/plugins/hermes-lcm
```

Or install a symlink from an existing checkout:

```bash
./scripts/install.sh
# Optional profile-aware install:
HERMES_PROFILE=myprofile ./scripts/install.sh
```

Enable both the plugin manifest name and the runtime context engine name:

```yaml
plugins:
  enabled:
    - hermes-lcm

context:
  engine: lcm
```

Restart Hermes after changing plugin or context-engine config.

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
  ✓ hermes-lcm v0.18.0 (7 tools)

Provider Plugins:
  Context Engine: lcm
```

For a live session check, send one normal Hermes message after restart, then run `lcm_status` or `/lcm status`. For database/config health, run `lcm_doctor` or `/lcm doctor`.

If startup logs say LCM tools are available through `context-engine schemas` or mention the `Path B fallback`, that is expected on older Hermes hosts such as Hermes Agent v0.16. The seven `lcm_*` tools remain available through the context-engine path; standalone plugin-registry registration is not required there.

## Update

If you cloned directly into the plugin directory:

```bash
cd ~/.hermes/plugins/hermes-lcm && git pull --ff-only
```

If you installed a symlink from a separate checkout:

```bash
./scripts/update.sh
```

Restart Hermes after updating.

## Common first-run checks

- `hermes plugins` shows `hermes-lcm` and context engine `lcm`
- `lcm_status` reports runtime identity, plugin path, git commit/branch/dirty state, and live session fields after the first message
- `lcm_doctor` reports database path, SQLite health, FTS state, lifecycle stats, and payload-boundary diagnostics
- Benchmark entrypoint: `python scripts/lcm_benchmark.py --fixture benchmarks/fixtures/long_history_canaries.json --output /tmp/hermes-lcm-benchmark-smoke --allow-external-output --json`
- Stress/release smoke entrypoint: `python scripts/lcm_stress_check.py --output /tmp/hermes-lcm-stress-smoke --tier smoke --json`

## Documentation map

- [Operator guide](docs/operator-guide.md) - install, update, verification, troubleshooting, config, slash commands, rotate
- [Retrieval tools reference](docs/retrieval-tools.md) - tool contracts, pagination, source filters, lossless raw recovery, OpenClaw import utility
- [Architecture notes](docs/architecture.md) - feature model, LCM-vs-built-in-compression nuance, internals, development files
- [Benchmarking and stress checks](benchmarks/README.md) - deterministic replay, preset provenance, scrubbed exports, release stress checks
- [Release validation](docs/release-validation.md) - local offline validation entrypoint and reusable checklist artifact shape
- [Packaging and distribution posture](docs/packaging.md) - current clone/symlink decision and when pip-style packaging should be revisited
- [Changelog](CHANGELOG.md) - recent release arc and post-v0.18 follow-up state
- [Contributing](CONTRIBUTING.md) - branch, PR, and validation expectations

## Development

Run tests:

```bash
pip install pytest
python -m pytest tests/ -v
```

No Hermes Agent checkout is required for the test suite; tests include a lightweight ABC stub.

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
