# Changelog

This repo also publishes GitHub Releases. This file is the repo-root release surface for operators who want the recent release arc without leaving the checkout.

## Unreleased

- Began a behaviour-preserving decomposition of the ~9k-line `engine.py` into cohesive modules: stateful method clusters became `*Mixin` classes (`compaction.py`, `reconcile.py`, `aux_session.py`, `placeholder_ledger.py`) mixed back into `LCMEngine`, and pure/helper groups became plain modules (`engine_registry.py`, `codex_routing.py`, `sqlite_util.py`, `runtime_identity.py`, `message_analysis.py`). No behaviour or public-surface change; documented in `docs/architecture.md`.

## v0.18.1 - 2026-06-30

Release focus: compaction privacy, clone/hook integrity, doctor signal accuracy, and model-context safety.

- Excluded ignored backlog and stripped injected context before compaction, preventing ignored or synthetic context from entering LCM summaries. (#283, #282)
- Preserved Discord lane metadata, active LCM clone resolution, and context metadata through cloned engines and post hooks. (#292, #293, #289)
- Hardened runtime identity, raw tool call integrity refs, payload integrity checks, and doctor path/lifecycle diagnostics. (#281, #278, #279, #291, #273, #280)
- Updated Codex OAuth effective context window safety defaults. (#274, #276)
- Completed focus-topic demotion behavior and preserved raw session ownership across compression rollover. (#268, #269)
- Refreshed operator docs, community-health files, and release-validation guidance. (#272)

## v0.18.0 - 2026-06-18

Release focus: retrieval depth, durability, status provenance, and long-session correctness.

- Added recursive evidence support for `lcm_expand_query`, improving synthesized answers from expanded LCM context. (#266)
- Hardened externalized payload durability. (#265)
- Avoided duplicate ingest protection work on hot paths. (#262)
- Aggregated DAG status stats for cheaper health surfaces. (#264)
- Preserved source lineage after long sessions. (#263)
- Surfaced LCM config provenance in runtime status. (#261)
- Fixed per-turn ingest for WebUI sessions and batch timestamp deduplication. (#260)

## v0.17.0 - 2026-06-14

Release focus: automatic focus-topic derivation and lifecycle hygiene.

- Added auto-derived focus topics during compression.
- Added empty lifecycle-row garbage collection to prevent unbounded accumulation. (#256)
- Improved runtime context indicators.

## v0.16.x - 2026-06

Release focus: engine isolation, WAL durability, database-path clarity, and startup cost control.

- Isolated LCM engine state per agent. (#247)
- Preferred bound sessions on sibling chains when the host has zero DAG.
- Tuned compaction defaults and clarified context-threshold ownership. (#245)
- Clarified `LCM_DATABASE_PATH` override behavior. (#249)
- Hardened WAL durability and graceful-close checkpoints. (#237)
- Throttled startup FTS integrity checks to reduce launch time. (#236)

## Links

- GitHub Releases: https://github.com/stephenschoettler/hermes-lcm/releases
- Release workflow: [`.github/workflows/release.yml`](.github/workflows/release.yml)
- Validation expectations: [`CONTRIBUTING.md`](CONTRIBUTING.md)
