# M8 — THE PROGRAM HAS BEEN OPTIMIZING THE WRONG OBJECTIVE (LAFS is accuracy × latency, and latency was never targeted)

_Found 2026-07-25 by the fresh-eyes audit workflow, then independently re-verified by the orchestrator (Opus 5)
by executing the benchmark's OWN scorer: `/Volumes/LEXAR/hermes-work/lme-v2-official/leaderboard/compute_lafs.py`,
`fixed_frontier_points["small"]`, T_MIN=1.0, T_MAX=200.0. Every number below is computed, not estimated._

## The finding

LAFS Gain — the leaderboard's ranking metric — is the **area between our submission's Pareto frontier and a fixed
reference frontier, over a latency budget sweep from 1s to 200s**. A point that is dominated on BOTH axes by an
existing reference point contributes **exactly zero**. The program has optimized accuracy for three days and has
never once treated latency as an objective.

**Every result we have banked, and every submission trigger we predeclared, scores 0.0000:**

| point | LAFS Gain |
|---|---|
| static 27.7% @ 0.109s (banked) | **0.0000** |
| agentic 66.1% @ 196.9s (banked) | **0.0000** |
| both submitted together | **0.0000** |
| static trigger 42.8% (#151) | **0.0000** |
| agentic trigger 69.9% (#151) | **0.0000** |
| M7 projection 72.1% @ 196.9s | **0.0000** |

## What actually scores (same accuracy, no new mechanism)

| change | LAFS Gain |
|---|---|
| agentic 66.1% @ 197s → **100s** | 0.11 |
| agentic 66.1% @ 197s → **50s** | **1.09** |
| agentic 66.1% @ 197s → **20s** | **2.82** |
| agentic 66.1% @ 197s → 5s | 6.77 |
| static @0.109s: 51.0% | 0.0000 (hard cliff) |
| static @0.109s: **51.5%** | 0.31 |
| static @0.109s: **55%** | **2.49** |
| static @0.109s: 58.6% | 4.72 |
| **static 55% + agentic 66.1%@50s (both)** | **3.58** |

**Decision-grade comparison:** getting static from 27.7% → 51.5% (+23.8 accuracy points, months of work) is worth
**0.31**. Cutting agentic latency 197s → 20s with *zero* new accuracy work is worth **2.82** — ~9× more, for
strictly less effort. M7's +6 accuracy points are worth **0.0000** at today's latency and **1.97** at 50s: latency
is a **multiplier on every accuracy gain**, not an alternative to it.

## Where the 197s actually goes (measured, so we know the lever)

`H6-P4-FULL-METRICS.json`: 5,005,656 agent output tokens / 451 questions = **11,099 output tokens per question**,
÷ 196.9s = **56.4 tok/s**. The latency is ~100% the pinned `gpt-5.4-mini @ xhigh` agent generating its own
reasoning tokens — **not** hermes store round-trips (our store's own retrieval is 0.109s mean in the static lane).
No store optimization can touch it. The only levers are agent-side: reasoning effort, turn cap, search fan-out —
all of which the §2d comparability pin currently forbids.

## The two corrections

1. **§2d's agent pin has served its purpose and must stop constraining the leaderboard point.** The pin exists to
   keep our number comparable to the published Codex-69.9 config and to our own vanilla A/B. That A/B is DONE and
   banked (P4: hermes 66.1% @197s beats vanilla 63.3% @256s — a real, citable capability result). The leaderboard
   imposes no such constraint. Going forward the leaderboard submission point may vary agent config freely
   (effort/turns/fan-out); the A/B result stays banked, separately labeled, unchanged.
2. **The static lane's real bar is >51.0%, not 42.8%.** 42.8 is worth zero because RAG+notes (51.0 @0.2s) dominates
   it everywhere. But static's measured latency (0.109s) is *cheaper than the reference point itself*, which makes
   it the **steepest marginal slope in the program above the cliff** (~0.62 LAFS per accuracy point). Static is
   all-or-nothing: worthless below 51.0, extremely valuable above it. The current roadmap projects ~47% max — so
   static needs a fundamentally bigger jump than wave-3.5 provides, or it should not be the funded lane.

## Corrected priority order

1. **Agentic latency reduction** (new Lane-A item, ahead of everything): 197s → <108.3s crosses onto the frontier;
   →50s ≈ 1.09; →20s ≈ 2.82. Dev-loop: 60q slices varying agent effort/turn-cap, gated on (accuracy not down) ×
   (latency down). This is the only path to a non-zero score that requires no new mechanism.
2. **M7 negative-evidence disclosure** — keep, but sequence AFTER latency; it is a multiplier that pays 1.97 at
   50s and 0.00 at 197s.
3. **Static** — re-aim at >51.0 or defund; also needs its decoding-parity regression fixed (see M9/arm-E void).
4. **Submission trigger (#151) must be rewritten** from accuracy thresholds to `lafs_gain_for_submission(...) > 0`,
   computed with the repo's scorer. Both current triggers would fire on results worth exactly zero.
