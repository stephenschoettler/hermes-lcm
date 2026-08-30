# HERMES-LCM BENCHMARK PROGRAM — PLAN v5 (rewritten 2026-07-25 after the M8 objective correction)

**Durable copies of everything below live on fork main (100yenadmin/hermes-lcm):**
`bench/PROGRAM-ARCHITECTURE.md` (binding strategy + decision records) · `bench/OPUS-DRIVE-LOOP.md` (execution
loop) · `bench/ENDGAME-PLAYBOOK.md` (landing protocols) · `bench/FINDING-M7-*.md`, `bench/FINDING-M8-*.md` ·
`bench/specs/`. Tracker of record = GitHub issues; hub #107; takeover entry #152.
**This file is the human-readable state-of-the-program. The repo docs are authoritative for gates.**

---

## 0. THE CORRECTION THAT REFRAMES EVERYTHING (M8, 07-25)

The leaderboard metric (**LAFS Gain**) is the **area between our accuracy-vs-latency curve and a fixed reference
frontier, swept over latency budgets 1s→200s**. A result that is beaten on *both* accuracy and speed by an
existing entry scores **exactly zero**. Verified by running the benchmark's own scorer with our numbers:

| result | accuracy | latency | LAFS |
|---|---|---|---|
| static (banked) | 27.7% | 0.109s | **0.0000** |
| agentic P4 (banked) | 66.1% | 196.9s | **0.0000** |
| old static trigger | 42.8% | 0.109s | **0.0000** |
| old agentic trigger | 69.9% | 196.9s | **0.0000** |
| M7 projection | 72.1% | 196.9s | **0.0000** |

**What actually scores:** agentic 66.1% at 100s = 0.11 · at 50s = **1.09** · at 20s = **2.82** (no new accuracy
work). Static needs **>51.0%** (51.5% = 0.31, 55% = **2.49**) — worthless below that cliff, steepest slope in the
program above it. Both together (static 55% + agentic 66.1%@50s) = **3.58**.

**Consequences:** (1) **latency is now a first-class objective and a multiplier on every accuracy gain** —
M7 is worth 0.00 at 197s and 1.97 at 50s; (2) the §2d agent-config pin has served its purpose (the vanilla A/B is
banked) and no longer constrains the leaderboard point; (3) static's real bar is 51.0%, not 42.8%; (4) the
submission trigger is now `lafs_gain_for_submission(...) > 0`, **computed, never inferred from headline accuracy**.

---

## 1. WHERE WE ARE — BANKED AND VERIFIED

| item | result | status |
|---|---|---|
| Official static baseline | **125/451 = 27.7%** @0.109s | banked, integrity-gated, tagged |
| Official agentic (P4) | **298/451 = 66.1%** @196.9s | banked, tag `bench-H6-P4-298` |
| Controlled A/B (the capability claim) | hermes **66.1%@197s** vs vanilla-Codex **63.3%@256s**, same agent/reader/judge, both pinned | **valid — our headline capability result** |
| memorybench V1 | **444/500** | closed by its own gate |
| Corpus ceiling | **1/451** gold absent from store | all ~72pts of headroom is read-time, not ingest |
| Answerable-only accuracy (agentic) | **249/323 = 77.1%** | higher than AgentRunbook-C's 74.9% *overall* |
| R1 release | published | fork |
| Wave-1 upstream PR | **#436 open**, 5 review findings fixed test-first | awaiting maintainer |
| Instrument fixes | SpendGuard, comparator, CI date-bomb, malformed-response guard | all landed |
| Mechanisms built (default-off) | diversity-cap, adaptive-excerpt, state-semantic-quota, antiboilerplate, title-boost | **code valid**; see §2 for measurement status |

**Findings ledger:** M1 compactness-bound consumer · M2 recall reachability · M3 agentic answers via the same 9B
reader (curation IS the benchmark) · M4 hub-crowding + web excerpt truncation · M5 corpus complete ·
**M7 abstention/false-premise = biggest untouched loss mass** · **M8 LAFS = accuracy × latency** ·
**M9 run-config parity rule**.

---

## 2. WHAT IS BROKEN / VOIDED (be honest about this)

- **Static dev arms E/F/G/H measurements are CONFOUNDED** — they ran with unpinned decoding (provider defaults)
  while the baseline pins temperature 0.6 / top_p 0.95 / top_k 20. Arms A/B/C are pinned and valid.
  ⇒ **arm E's 36.7% is not established; armC (33.3% / 23.3% spurious) is the last VALIDATED static candidate.**
- **The arm-E full-451 promotion was killed at ~20%** for the same reason (saved ~3h + ~$4). Partial preserved as
  `FULL-RUN-*-VOIDED-unpinned-decoding`.
- **Zero capability promotions have been banked to date.** The 119→205→bugs-fixed arc was *measurement recovery*,
  not capability. P4 is the first genuine capability result; it is not yet a leaderboard-scoring one.

---

## 3. ROADMAP — the path to a non-zero score, in priority order

### P1 · AGENTIC LATENCY (running now — the only path needing no new invention)
Sweep agent reasoning effort {high, medium, low} on the frozen 60q manifest, decoding pinned, measuring the
accuracy/latency trade curve; **plus L4, a concurrency-1 contention probe** (12 fixed questions) because P4's
197s was measured with 6 agents competing on one machine — distribution (min 67s, p10 101s, median 166s) suggests
the honest uncontended number is materially lower. **Target: ≤108.3s at ≈66% ⇒ on the frontier; 50s ⇒ ~1.09.**
Gate: accuracy not materially down × latency down; LAFS computed by the orchestrator with the repo's scorer.

### P2 · M7 NEGATIVE-EVIDENCE DISCLOSURE (#157) — the accuracy multiplier
52% of agentic losses and 34% of static losses are abstention/false-premise questions. The reader **asserts** on
122/128 of them (73 hallucinations) because the curated evidence pack has no channel for "I searched and found
nothing" — only 29% of failing packs contain any absence language. Fix is content-only (§2b-legal) and is the
owner's multi-call thesis: make the agent report its null results. Abstention 38%→60% = 72.1% ⇒ **1.97 LAFS once
latency lands** (0.00 before it). Pilot: one 60q slice per lane, joint gate (abstention up AND answerable not down).

### P3 · STATIC — re-aim at >51.0% or defund
Needs +23.3 points from 27.7%; the current roadmap (armC + wave-3.5 + M7-static) projects ~47% — short of the
cliff. Prerequisite regardless: re-measure arms E/F with pinned decoding (~$0.30) to learn which mechanisms are
real. Decision point after P1/P2 land: fund a bigger static jump, or keep static as merge-worthy code only.

### P4 · WAVE-3.5 (#155) — delivery seat-selection family
Spec exists and is sound, but it targets answerable-question spurious-unknowns: a smaller mass than M7, and worth
0.00 without latency. **Demoted below P1/P2.**

### P5 · OWNER-GATED / PARKED
H6-P5 frontier-consumer ceiling run (#156) · H7 compaction-replay own-benchmark (#149) · portfolio secondaries
STATE-Bench + PersonaMem-v2 (#153) · leaderboard submission (#151, now `lafs > 0`) · R2 fork release (agreed: publish;
sequence after P1 so it carries the latency result) · wave-2 upstream PR (code, after mechanisms validate).

---

## 4. STANDING RULES (added this cycle)

- **M9 run-config parity:** before any run is used in a gate, diff its `run_args.json` (decoding params, models,
  store path, question set) against the comparison baseline's. Part of the pre-launch probe. *A number is not a
  measurement until its config matches what it is compared to.*
- **Compute LAFS, never infer it.** Any "is this worth submitting?" question is answered by running
  `leaderboard/compute_lafs.py`, not by comparing headline accuracies.
- **Latency is measured and reported with its concurrency condition.** Never publish a latency figure without
  stating whether it was measured contended or uncontended.
- Prior rules unchanged: snapshot-first · manifests frozen + achievability-checked · predeclared gates never
  relaxed at one-short · one-primary law · file-keyed foreground polling · durable scheduling · assert outward
  actions · artifacts to LEXAR session-notes · locate a run's log via `lsof` before declaring it stalled.

---

## 5. OWNER DECISIONS OUTSTANDING
1. **H6-P5 frontier run** (#156) — new spend; measures the deployable ceiling rather than the leaderboard's
   weak-consumer question. Recommended after P1.
2. **Static lane disposition** (§3 P3) — fund the jump to 51%+, or keep as code-only. Recommend deciding after
   P1/P2 numbers land.
3. Submission and R2 timing — both agreed in principle (hold submission until `lafs > 0`; publish R2), sequencing
   after the latency result.

---

## 6. HISTORY (condensed)
Day 1: program takeover after a 5-day stall → SpendGuard root cause → comparator bug (both-sides rescore, +86
corrected to +24) → honest Sol baseline 205 → H2 attribution → H3.1 quarantined → H5b structural no-go →
V1 gate chain → 444 banked. Day 2: official protocol became the primary instrument (125/451) → R1 published →
wave-1 upstream PR → wave-3 dev loop (25.0 → 33.3 armC) → H6 lane P0→P4 (66.1%) → V1 closed at 444.
Day 3 (07-25): corpus ceiling proved read-time headroom → M7 abstention discovery → adversarial fresh-eyes audit
→ **M8 objective correction + M9 parity rule** → arm-E promotion voided → latency sweep launched.
