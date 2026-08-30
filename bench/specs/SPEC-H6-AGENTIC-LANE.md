# SPEC — H6: Official AGENTIC lane for hermes-lcm (LongMemEval-V2 agentic tier) — DRAFT, execution OWNER-GATED

_Authored 07-24 ~05:50 after harness recon. Proposed to owner in the 05:40 reply; this spec makes "go" a one-word
dispatch. No execution before owner go (new lane; full-run spend needs sizing)._

## Why this lane

- The published ladder splits at agency: static ceiling 51.0 (RAG+notes) vs agentic 69.9 (Codex) / 74.9 (AgentRunbook-C).
- The leaderboard metric is **LAFS Gain — a Pareto frontier over (accuracy, query-latency)**. Published points:
  Codex 69.9 @ 177.2s · AgentRunbook-C 74.9 @ 108.3s · RAG+notes 51.0 @ 0.2s (small track; medium track differs
  slightly). Two winnable slots: high-acc agentic AND low-latency compact-static. This lane targets the first.
- **The experiment IS the mission**: same coding agent, vanilla file memory (official `codex` baseline) vs
  hermes-lcm store — does our memory make the SAME agent better? Clean causal read on "agent memory for agents."

## What the official machinery already provides (recon receipts, adapter worktree)

**PATH PIN (cold-start):** all files below live in the official-benchmark adapter worktree
`/Volumes/LEXAR/hermes-work/wt-bench-h1-v2adapter` (branch `bench/h1-v2-adapter`, org fork
100yenadmin/LongMemEval-V2) — NOT in the hermes-lcm repo. This is the authoritative checkout; ignore other
longmemeval clones on disk. Full P0 protocol pin (file:line cites): comment on issue #145.

- `memory_modules/` includes the official `codex` baseline ("vanilla Codex coding-agent memory") and `agentrunbook_c`;
  runner `evaluation/scripts/run_codex.sh`; env contract `CODEX_BINARY` (pin v0.117.0 per README), `CODEX_MODEL`
  (README example: gpt-5.4-mini), `CODEX_REASONING_EFFORT`; needs `rg`/`find` on PATH.
- Scoring identical to static lane (`qa_eval_metrics.py` matchers/checkers + combining rules) → numbers comparable
  by construction. `leaderboard/compute_lafs.py` computes the frontier locally.

## Phases (each gated before the next)

**P0 — protocol conformance read (Fable, ~1h):** read the `codex` module + `run_codex.sh` end-to-end: how insert()
materializes the agent workspace, how query() invokes the agent per question, what latency is measured, what model
the PUBLISHED 69.9 used (must match or disclose divergence). Deliverable: 1-page protocol pin in this file's addendum.

**P1 — vanilla-Codex baseline repro, 60q stratified slice:** validates our binary/env/latency/cost before any
hermes work. Gate: slice accuracy within ±8pts of 69.9's slice-expectation AND per-question cost measured.
Deliverable: cost/latency table → sizes the full-run spend ask for the owner. (Slice uses the existing codex/OpenAI
account lane; FULL 451q agentic run = explicit owner spend sign-off with the P1-measured number.)

**P2 — `hermes_lcm_agentic` memory module:** insert() = existing static-lane ingest (reuse, clone-fast-path);
query() = agent workspace where the hermes-lcm store is queryable (CLI surface over the same store the static lane
built; the agent explores via tools instead of receiving one bulk context). Same CODEX_BINARY/model pins as P1.
Tests: module contract vs official Memory ABC; golden 5q smoke.

**P3 — paired slice (60q): vanilla-Codex vs hermes-agentic, same qids, same agent+model.** Predeclared gate
(frozen now): hermes-agentic must beat vanilla-Codex slice accuracy by ≥+5 points to justify the full run.
Latency recorded for LAFS honesty (expect ours faster: indexed store vs raw-file grep).

**P4 — full 451 (owner-gated on P1 cost + P3 gate)** → integrity gate (same step-0 as W2d) → LAFS placement →
submission decision (owner).

## Rails
- All artifacts under session-notes hermes-benchprog-h1/artifacts/H6-*; snapshot-first; manifests frozen+sha256
  before launches; no re-ingest (clone); Sol-stream ceiling respected; every run's model/effort/binary pinned in
  the run manifest. Published-number comparisons always cite tier + consumer + latency.
