# Changelog

This repo also publishes GitHub Releases. This file is the repo-root release surface for operators who want the recent release arc without leaving the checkout.

## Unreleased

- Preserved raw session ownership across compression rollover after `/new`, so carried-over summaries can still expand back to original raw rows. (#269)
- Current post-v0.18 follow-up state: adoption and operator-readiness hardening. No broad feature lane is active from this changelog entry.

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
