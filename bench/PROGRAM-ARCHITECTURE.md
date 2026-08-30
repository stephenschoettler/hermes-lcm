# HERMES-LCM BENCHMARK PROGRAM — MASTER ARCHITECTURE v1 (frozen 2026-07-24 ~06:10 +07)

_Authored by the program architect (Fable) under owner grant of full authority (07-24 ~05:55: "You have full
capability and authority. You're the architect/designer/and release product manager"). This document + the
ROADMAP issues + OPUS-DRIVE-LOOP.md are the complete operating state: a continuation agent (Opus 4.8) must be
able to execute from these without access to this session's context. Strategy changes require either the owner
or a documented DECISION-RECORD addendum here — never silent drift._

## 0. Mission and the number ladder

Make agent memory better for all agents, proven publicly on LongMemEval-V2 (primary) and memorybench-V1
(secondary). The public ladder (leaderboard page, verified 07-24; leaderboard still EMPTY — first-mover slot open):

| Tier | Published points (small track) | Our position |
|---|---|---|
| Reader-only floor | 1.3% | — |
| Static (fixed reader) | RAG 42.8 @ ~0.2s · RAG+notes 51.0 @ 0.2s · AgentRunbook-R 58.6 @ 26.9s | **ours: 27.7 (125/451)** |
| Agentic (own agent) | Codex 69.9 @ 177.2s · AgentRunbook-C 74.9 @ 108.3s | not yet entered |

Leaderboard metric = **LAFS Gain — the AREA between our accuracy-vs-latency curve and the fixed reference
frontier, swept over latency budgets 1s..200s** (leaderboard/compute_lafs.py). A point beaten on BOTH axes by a
reference point scores exactly 0. **★ SUPERSEDED TARGETS (M8, 07-25): the old "static >=42.8 -> >=51.0 ->
agentic >=69.9 -> >=74.9" ladder is INVALID as a goal set — 42.8% scores 0.0000 and 69.9% at our 197s scores
0.0000.** Verified with the repo's own scorer.

**CORRECT TARGETS (computed):**
| target | LAFS |
|---|---|
| agentic 66.1% (already banked) at <=108.3s | >0 — on the frontier |
| agentic 66.1% at 50s / 20s | 1.09 / 2.82 |
| static >51.0% at our 0.109s (51.5 / 55 / 58.6) | 0.31 / 2.49 / 4.72 |
| static 42.8%, agentic 69.9%@197s, M7 72.1%@197s | **0.0000 each** |

Latency is a FIRST-CLASS objective and a multiplier on all accuracy work (M7 = 0.00 at 197s, 1.97 at 50s).
Priority order: agentic latency -> M7 abstention -> static re-aimed at >51.0 or defunded. Full analysis:
bench/FINDING-M8-LAFS-METRIC-ERROR.md; §2f carries the decision record.

## 1. The two load-bearing measurements (why this architecture)

**M1 — The official static run (125/451, integrity-gated, OFFICIAL-RESULTS.md):** the dominant loss class is
spurious-unknown — the official Qwen3.5-9B reader answers "unknown" on 95/323 (29%) of answerable questions
while holding full ~22k-token untruncated contexts (medians identical for answered vs unknown; zero truncation).
**The static score is bounded by context COMPACTNESS at the consumer, not recall volume.** Enterprise is worst
(64/95 spurious-unknowns; 20.9% vs web 33.8%).

**M2 — H5(b) adjacency no-go (H5B-SWEEP-REPORT.md):** pool-entry ceiling 4/30 vs gate ≥8/30, proven structural —
25/30 recall-miss targets sit on zero-pooled-state trajectories that same-source adjacency can never seed.
**The recall floor is reachability: states need direct semantic addressability (embedding backfill), not
neighborhood expansion.**

**M3 — H6-P0 protocol pin (07-24, file:line-verified):** the agentic tier ALSO answers through the same fixed
Qwen3.5-9B reader — the coding agent only CURATES evidence (memory_markdown + ≤20 trajectory spans) which the
reader consumes; scoring path identical to static. **Codex-69.9 vs our 27.7 is therefore entirely context-curation
quality into the same weak consumer.** The static/agentic ladder split is curation-by-pipeline vs curation-by-agent.

M1+M2+M3 converge: **reach the right states, then deliver them small — curation quality IS the benchmark.**
That is wave-3 (pipeline curation) and H6 (agent curation over our store); they share the same thesis.

**M4 — W3c enterprise classification (07-24, 20/20 sampled, #144):** REFINES M1 for the enterprise subset —
the needle was ABSENT from the delivered context in 20/20 sampled enterprise spurious-unknowns (the reader
abstained CORRECTLY). Mechanism: 3-4 generic hub trajectories (incident-list/search/creation flows) crowd out
the topic-specific trajectory in retrieval regardless of question topic (hub 96131e7b in 19/20 contexts).
Same magnet-family pathology as H3.1. CONSEQUENCE (recorded before any W3b build): W3b must lead with a
retrieval-diversity mechanism (hub-trajectory cap / per-source diversity quota) — repacking cannot surface
what retrieval never returned — and W3a state-level embeddings attack the same root (topic-specific states
become directly addressable instead of riding hub-trajectory coarse vectors). WEB refinement (10-case sample,
same method): 7/10 needle-absent (hub-crowding generalizes, weaker) + 3/10 needle-PRESENT but truncated by the
fixed ~2.5-2.9K-char per-state AXTree excerpt window — a web-only delivery bug (W3b component 2: adaptive/
needle-aware excerpt windows). Compactness remains the live thesis for the static→agentic gap (M3).

**M5 — Corpus-coverage ceiling (deterministic all-451 audit, 07-25, no model calls; CORPUS-COVERAGE-CEILING.md):**
only **1/451 (0.22%)** questions have gold material ABSENT from the stored state text (enterprise c9568c9b,
"risk level"); web 0/240. Of the ~72-point gap between official 27.7% and 100%, **at most ~0.4 points is a real
corpus/ingest limit — the entire remaining ~72 points is retrieval + ranking + delivery + reasoning headroom.**
This VALIDATES the whole program direction and the owner's lossless-raw thesis: storage is essentially complete;
there is no ingest work to do; every point is won in the read-time path (composition/delivery/consumer). It also
means the recall floor (H5 family) is NOT corpus-bounded — states exist, they must be reached and delivered.
(Refines the shadow-audit's "4 absent cases" to a rigorous 1; the naive audit's 40 were 39 method false-positives
— MC options are always echoed in the prompt.)

## 2. DECISION RECORD — the official protocol becomes the primary static instrument (NEW, 07-24)

The full 451-question official run cost **≈$1.1 and ~3.5h wall** (parallelizable to <1h in 4 batches). The Sol
internal protocol costs more and measures a frontier reader that masks the exact failure mode (bulk-context
tolerance) the official reader punishes. Therefore, effective immediately:
- **Static candidates iterate against the OFFICIAL protocol directly** (60q stratified slices for dev loops at
  ~$0.15, full 451 for gates), using the frozen batch machinery at
  `/Volumes/LEXAR/Codex/session-notes/2026-07-23/hermes-benchprog-h4/artifacts/phase3-openrouter/` (harness
  worktree: `/Volumes/LEXAR/hermes-work/wt-bench-h1-v2adapter`).
- Sol protocol is demoted to a dev-loop probe and for V1 (whose harness is Sol-native). Sol numbers remain
  non-official-labeled always.
- The 205/451 Sol baseline remains the tagged historical reference; the official 125/451 is the number of record.
Predeclared instrument rules carry over: snapshot-first, per-category integrity check on every full run,
both-sides rescoring if any scorer changes.

## 2b. DECISION RECORD — we do NOT own the reader/answer prompt (wave-3.5 design workflow, 07-25; CODE-VERIFIED)
The official LME-V2 harness system/answer prompt is FIXED and shared across ALL leaderboard systems
(evaluation/harness.py DOMAIN_SYSTEM_PROMPTS, ~L72-89: "output exactly \\boxed{UNKNOWN}... Do not guess"),
and hermes-lcm's own adapter DELIBERATELY ships an EMPTY prompt override (qa-harness .../hermes-lcm/prompts.ts:
"deliberately ships NO custom answer or judge prompt... a bespoke tuned prompt would inflate its numbers
relative to providers scored under the defaults"). **CONSEQUENCE (binding on all future work): any
'reader-contract' mechanism may only shape the `memory_context` CONTENT we return — never the instructions.**
Editing the answer prompt would be BOTH leaderboard-invalid (non-comparable) AND a self-inflated number. This
killed the M4 "calibrated-inference/analogy-license" proposal in wave-3.5 (it fought the fixed prompt =
textbook design-for-Sol). Reader-contract work = evidence-embedded scaffolding at the compilation stage only.

## 2c. NEXT FAMILY — wave-3.5 spec exists (bench/specs/SPEC-W35-FAMILY.md, 07-25)
Synthesized+adversarially-critiqued design for the successor family AFTER arm-E promotion banks. Center of mass
= DELIVERY seat-selection (C4 flagship; targets the 6/8 arm-E residuals that die at final seat-selection to
boilerplate), NOT recall or reader-text (0/30 sampled unknowns were format-hostility). Components: C4 (seat
precision) · C5 (bounded local cross-encoder rerank, smoke-gated, cut-if-no-composed-gain) · QD (recall
widening for the 2/8 retrieval residuals) · RC (reader-contract scaffolding, smallest/last). Gated as a §6c
FAMILY on composed whole-funnel net vs promoted arm-E. Sequenced: does NOT block/compete with the arm-E
promotion run. Full spec + cut-list in the file.

## 2d. DECISION RECORD — H6 agent model is FIXED at the published-point config (why gpt-5.4-mini, owner Q 07-25)
The H6 agentic experiment is a CONTROLLED A/B: hold the official protocol fixed, vary ONLY the memory. The
official protocol fixes reader=Qwen3.5-9B (the consumer, shared by all leaderboard systems), judge=gpt-5.2, and
— for the agentic tier — agent=gpt-5.4-mini @ xhigh, which is the config the PUBLISHED Codex-69.9 point used
(official repo defaults codex.py:39-40 + README; P0 recon on #145). gpt-5.4-mini is deliberately NOT frontier:
a weak agent makes the MEMORY's contribution visible and keeps the number comparable to both the published 69.9
AND our own vanilla-Codex baseline (which must use the same agent). Using Sol/frontier here would break both
comparisons. Labeled "reproduced-with-defaults" + version-divergence disclosed (codex-cli 0.144.6 vs repo v0.117.0).
This measures COMPETITIVENESS on the leaderboard's terms — NOT the product ceiling.

## 2e. H6-P5 (proposed, owner-gated) — frontier-consumer product-ceiling variant
P4 answers "competitive on the board's terms." It does NOT answer the frontier-future thesis (§6b): how good is
the memory when a STRONG model consumes it. H6-P5 = hermes-agentic with a FRONTIER agent (Sol/latest) curating,
optionally + a frontier reader — EXPLICITLY non-leaderboard-comparable, a product-ceiling measurement. Partial
frontier signal already exists (Sol-internal static 45.5% vs 9B 27.7% = the consumer-capacity gap). P5 is
owner-gated (new frontier-agent spend). Both truths wanted: P4=public standing, P5=deployable ceiling.

## 2f. ★ DECISION RECORD — LAFS is accuracy X latency; latency is now a FIRST-CLASS objective (M8, 07-25)
Verified by running the benchmark's own scorer (leaderboard/compute_lafs.py, fixed small frontier, T=1..200s):
**every banked result and BOTH predeclared submission triggers score exactly 0.0000** (static 27.7%@0.109s;
agentic 66.1%@196.9s; triggers 42.8% and 69.9%; even M7's 72.1%@197s). A point dominated on both axes by a
reference point contributes zero area. What scores: agentic 66.1% at 50s = 1.09, at 20s = 2.82 (NO new accuracy
work); static >51.0% at our 0.109s (51.5%=0.31, 55%=2.49). Getting static +23.8 accuracy points = 0.31; cutting
agentic latency 197s->20s = 2.82 (~9x, less work). Latency is a MULTIPLIER on accuracy work, not an alternative
(M7 = 0.00 at 197s, 1.97 at 50s). Measured cause of the 197s: 11,099 agent output tokens/question at 56.4 tok/s
= the pinned agent's own reasoning; our store's retrieval is 0.109s. CONSEQUENCES: (1) the §2d agent pin has
served its purpose (the vanilla A/B is banked) and no longer constrains the LEADERBOARD point — agent
effort/turn-cap/fan-out may now be varied for the submission point, with the A/B result kept separately labeled;
(2) static's real bar is >51.0%, not 42.8% — worthless below the cliff, steepest slope in the program above it
(~0.62 LAFS/point); (3) #151's trigger is rewritten to lafs_gain_for_submission(...) > 0, computed, never
inferred from headline accuracies. PRIORITY ORDER: agentic latency reduction FIRST, then M7, then static
re-aimed at >51.0 or defunded. Full analysis: bench/FINDING-M8-LAFS-METRIC-ERROR.md.

## 2g. STANDING RULE — run-config parity check before any gate (M9, 07-25)
The arm-E full-451 promotion was killed at ~20% because it ran with unpinned decoding (temperature/top_p/top_k
= null -> provider defaults) while its 125 baseline pins 0.6/0.95/20. Scope: dev iterations 1-2 (arms A/B/C) are
pinned and valid; **iterations 3-4 (arms E/F/G/H) are ALL unpinned**, so arm E's 36.7% was never validly
comparable and armC (33.3%/23.3%) is the last VALIDATED static candidate. RULE: before any run is used in a gate,
diff its run_args.json (decoding params, reader/judge model, store path, question set) against the comparison
baseline's; make it part of the pre-launch probe. A number is not a measurement until its config matches what it
is compared to.

## 3. Lane architecture

### Lane S — static compactness (wave-3; epic issue W3)
Goal: official static 125 → ≥193 (42.8-parity) → ≥230 (51.0-parity). Mechanisms, in dependency order:
- **W3a — H5(a) state-level embedding backfill** (recall floor). Backfill Voyage embeddings for pool states so
  zero-pooled-state trajectories are directly reachable. Gate summary (the ISSUE #142 body is the sole complete
  source of truth): pool-entry ≥8/30 on the frozen H5 target set
  (`/Volumes/LEXAR/Codex/session-notes/2026-07-23/hermes-benchprog-h1/artifacts/h5-targets.json`); plus the
  delivered-recall, preservation/golden, and latency conditions per #142. Spend: size first (W3a-0, expect low
  single-digit $; hard cap $10 without owner ping; re-meter at 50% of the backfill before continuing).
- **W3b — compact delivery** (the spurious-unknown killer). Replace 22k bulk contexts with budgeted sharp
  contexts (target ≤4k tokens) via the selective/evidence-contract path (the R1 subsystem exists for exactly
  this). Development gate: 60q official slice, primary metric = spurious-unknown rate on answerable questions
  (currently 29%) + accuracy; promotion gate: paired full-451 official vs the 125 baseline, net ≥+15,
  category-integrity pass. NOTE: compact delivery also collapses query latency → LAFS position improves on both
  axes.
- **W3c — enterprise diagnosis**: read 20 enterprise spurious-unknown cases (artifacts have full contexts),
  classify (format? entity density? question style?), feed findings into W3b templates.
Sequencing: W3a and W3c parallel; W3b consumes both; one promotion gate at the end (one-primary law).

### Lane A — agentic (H6; spec SPEC-H6-AGENTIC-LANE.md; epic issue H6)
The controlled experiment: SAME coding agent, vanilla file memory vs hermes-lcm store. Phases P0 (protocol
read — dispatched 07-24 ~06:05) → P1 (vanilla-Codex 60q repro; validates env/cost/latency; measures per-q cost
for the full-run spend ask) → P2 (`hermes_lcm_agentic` memory module: insert = existing ingest; query = agent
workspace with store-backed search CLI) → P3 (paired 60q, gate: hermes ≥ vanilla +5pts) → P4 (full 451 +
integrity + LAFS). Model/effort/binary pinned per run manifest; published-point parity checked in P0/P1.

### Lane P — product-truth compaction-replay harness (H7; NEW — owner's concept, 07-24 ~05:55)
The owner's design intuition, architected: benchmarks above measure ingest-then-query; the PRODUCT reality is
a live agent whose context fills and must compact without losing operational memory. H7 harness:
1. Replay a long conversation/work stream into a live agent session (LongMemEval-V2 histories are the replay
   corpus; their questions are the probes).
2. At a context threshold (parameter; owner suggested ~230k tokens), trigger hermes-lcm compaction (ingest the
   evicted span into the store; keep the compact residue in-context).
3. After each compaction cycle, run the probe subset answerable from evicted material; compare three arms:
   (a) naive truncation, (b) summary-only compaction, (c) hermes-lcm compaction+store (agent may query the store).
4. Metrics: retention curve vs compaction count; probe accuracy per arm; residue token cost.
This measures "compactions provide memory relief AND keep data findable" — the product claim, directly. H7 is
sequenced AFTER H6-P3 (it reuses H6's agent-with-store machinery; building it first would duplicate work).
Design doc to be expanded in the H7 epic issue before any build.

### Lane V — memorybench V1 (cycle-2; spec SPEC-W2A-CYCLE2.md; epic issue W2A-C2)
Fixes 5 diagnosed classes; gate chain predeclared in the spec (paired ≥+8 AND ≥+3 vs cycle-1; redesigned
achievability-verified blind, losses ≤2 & net ≥0; ONE full ≥450, MISS 445-449 ⇒ bank+close). Priority: below
lanes S and A (V1 is secondary; 444 is already banked and respectable).

### Lane C — community/release (release-PM authority granted)
- R1 release (program-r1): PUBLISH (fork-local; owner asked for tagging/release push on 07-23).
- Wave-1 upstream PR: POST from `upstream-wave-1@0cb7f37` with the completed body (official number filled,
  CI story resolved). Shepherd per PR-shepherding standard; expect slow maintainer, Tosko4 reviews well.
- Leaderboard submission: **HOLD** (release-PM decision, standing): 27.7 as the first public number frames the
  project badly; submit when Lane S crosses ≥42.8 (RAG-parity) or Lane A produces a ≥69.9 entry. Revisit rule:
  any lane crossing its threshold triggers the submission packet (validator + integrity gate + GEX divergence
  rule; GEX cross-check continues meanwhile).
- #423 / #434 / harness PR #2: continue shepherding (respond-to-review autonomy per established pattern).

## 4. Standing discipline (unchanged, binding on all agents)
Snapshot-first before ANY subsequent run · manifests frozen+sha256 BEFORE launch, with achievability verified
at freeze (blind-R2 lesson: compute the control slice at freeze time) · predeclared gates, never relaxed
at one-short, revisions only as documented addenda BEFORE numbers · one-primary law per candidate · no re-ingest
of clonable state · file-keyed foreground polling, never process-exit or phantom background watches · durable
scheduling for anything outliving a session, and execute-inline when a cron misses · assert outward-action
success (ls-remote / re-read after push) · artifacts to LEXAR session-notes, never /tmp · Sol numbers labeled
non-official, always · routing ledger line per dispatch · pre-digest >200-line outputs.

## 5. Risk register (top 5, with mitigations)
1. **W3b compact contexts lose recall** (sharp but wrong) → dev gate tracks accuracy AND spurious-unknown
   jointly; paired full-451 promotion gate catches net harm.
2. **H6 P1 baseline repro misses published 69.9 badly** → P0 pins model/effort; if repro diverges >8pts,
   STOP the lane and file a protocol-parity issue rather than tuning toward a number (instrument-first rule).
3. **Agentic full-run cost blows up** → P1 measures per-question cost; P4 is explicitly owner-gated on that number.
4. **Continuation-agent drift** (Opus re-litigating settled decisions) → this doc's decision records are binding;
   OPUS-DRIVE-LOOP.md defines what Opus may decide alone vs park.
5. **Upstream PR stalls** (slow maintainer) → wave-1 is additive-only + fork-local value is already banked;
   stall costs nothing on the critical path; keep fork releases flowing.

## 6a. DECISION RECORD — benchmark portfolio & the frontier-consumer position (owner dialogue, 07-24 ~05:3x)

Owner input recorded: (i) authority now explicitly extends to release publishing, upstream PR waves, and
leaderboard submissions — all release-PM calls are the orchestrator's; (ii) the critical rule is
CHECK-THE-ANSWERS forensics — ">90% of the time it's bugs in the harness, not what the agent built"
(this program's own history: SpendGuard, comparator LaTeX, fixture date-bomb, hub-crowding all found by
reading answers); (iii) owner is nervous about LongMemEval-V2 (brand new, nobody has posted, hard to build
for) and asks whether AgentArena / MemoryArena / BEAM / AMA-Bench / PersonaMem / LongMemEval-v1 fit the
mission (agents as employees / chief-of-staff) better; (iv) the future-is-frontier thesis: tiny local readers
are not the agentic future (GLM-5.2-local ≈ the floor going forward; multimodal + hybrid memory beyond that),
while acknowledging the purist counter-theory (a memory system so good no-API-call retrieval alone scores).

**Decisions (standing until revised by a documented addendum):**
1. **LongMemEval-V2 stays the PRIMARY public target.** The empty leaderboard is a first-mover asset, our
   instrumentation there is now cheap (~$1/full run) and battle-tested, and the harness-risk the owner fears
   is exactly what our check-the-answers discipline converts into wins (M4). The three tiers map cleanly onto
   the three theories: static lane = the purist no-API-calls theory test · agentic lane (H6) = the
   frontier-consumer future · H7 = the product truth. The portfolio position is a HEDGE built into one bench.
2. **Portfolio — RESOLVED 07-24 (#153 decision comment = the record):** 15-candidate survey completed.
   PRIMARY: LongMemEval-V2 (only bench whose haystack categorically exceeds frontier windows — immune to
   context-stuffing false positives). SECONDARY 1: STATE-Bench (microsoft/STATE-Bench, stateful DB-mutating
   agent loop, "bring your own memory") — readiness spike gated on H6-P3. SECONDARY 2: PersonaMem-v2
   (implicit-preference inference, 37-48% frontier headroom) — adapter gated on W3b promotion attempt 1.
   BEAM parked (scale-regime peer; revisit when a slot opens). Excluded: LongBench V2, CL-Bench, LoCoMo
   (mission grounds). AgentLongBench unconfirmed (no primary source). #138 closed into this.
3. **LongMemEval-v1: do NOT invest in crushing it** (saturated, >95% public repos = low marginal signal).
   Instead MINE the winners: competitor-technique survey (new issue) extracting their retrieval/curation
   mechanics for W3b/H6-P2. Revisit only if the portfolio spike ranks a v1 number as cheap credibility.
4. **Frontier-local reader arm (deferred, noted):** GLM-5.2-local on GEX44 as an H6-class consumer arm once
   P1-P3 land — tests the owner's "minimum future standard" directly. Not scheduled yet.
5. **Submission authority:** now fully delegated; the #151 trigger rule stands as the orchestrator's own
   standing decision (submit on static ≥193 or agentic ≥69.9 — the first public number frames the project).

## 6b. DECISION RECORD — the product thesis: lossless raw + read-time intelligence (owner, 07-24 evening)

Owner articulated the product philosophy; the week's evidence supports it; it is now binding design doctrine:
1. **Store raw, losslessly. Think at read time.** Write-time summarization freezes today's model's judgment
   into the permanent record and multiplies data; raw records let every future (better) model re-read the
   original. THE RULE: never store at write time what you could not regenerate from the raw record; never
   let a derived artifact answer when the raw record is reachable.
2. **Write-time INDEXES are not summaries** — FTS, embeddings, titles/labels, timestamps are regenerable
   pointers INTO raw data and are encouraged (the 'Last Updated At' miss was an index gap; iteration-4 knob H
   is more lossless index). Lossy derivations as sources of truth are rejected.
3. **Multi-call, sub-agent retrieval is the product path** (LCM-GREP/LCM-XPAND lineage): if one call lacks
   confidence, fire another; orchestrators park retrieval on sub-agents while the conversation moves.
   Evidence: same 9B reader — one-shot static 27.7% vs agent-curated multi-call 63-67% on identical stored
   data; and the store made the agent FASTER than filesystem grep (216s vs 256s) — lossless ≠ slow when
   indexes are good. Caveat baked in from P3R: each call must be precision-first (undisciplined
   ranking/fusion made the agent WORSE); discipline per call, iteration across calls.
4. **Scale favors this thesis**: at LME-V2's 25-115M-token regime no summary fits anyway — write-time
   distillation competitors dominate only on in-window benchmarks. The earlier 'write-time distillation'
   W4 candidate (Arch briefing) is WITHDRAWN accordingly.
5. **Own benchmark = H7 sharpened**: score task-completion × latency × cost (the owner's three factors) on
   live replay with compaction — external benchmarks stay useful but subordinate ("did the agent accomplish
   the goal" outranks any leaderboard number).

## 6c. DECISION RECORD — gate mechanism FAMILIES, not single mechanisms (shadow-audit, 07-25)

A shadow review named the program's load-bearing self-contradiction and it is CORRECT: the multiplicative-funnel
finding (a recovery needs candidate AND delivery AND reader-contract to all succeed) is logically incompatible
with the inherited codex-era doctrine of "one mechanism per release, net ≥+N alone." A single-stage fix
STRUCTURALLY cannot clear a whole-funnel net gate when most failures have >1 blocked stage — it is set up to
quarantine even when it perfectly does its job (H3.1: +9/−2 on its target cohort, quarantined on whole-funnel
net anyway = a false negative induced by a gate mismatched to the problem's causal shape).

**Corrected doctrine (binding):**
1. Gate mechanism FAMILIES (recall + delivery + reader-contract), each COMPONENT carrying its own
   component-level evidence, against the whole-funnel net. Do NOT require a single mechanism to clear the
   whole-funnel bar alone.
2. This is ALREADY LIVE on the static lane: wave-3 arm E (diversity-cap + adaptive-excerpt + state-quota) is
   a composed family gated on the composed net (36.7%/18.6% dev-loop), not three separate ≥+N gates. §6c makes
   the principle explicit rather than contextual.
3. **H3.1 rehoming:** the quarantined V2 anti-magnet composition (lexical-floor + arm-quota, +9/−2 core
   validated) is NOT dead — it is a CANDIDATE COMPONENT for the wave-3 family. BUT (funnel logic cuts both
   ways, proven by iteration-4 arm G: adding anti-boilerplate composition to arm E REGRESSED accuracy
   36.7→30.0 by demoting needle-bearing states) re-homing is EMPIRICAL: test H3.1's mechanism as a knob,
   gate on composed net, keep only if it improves the composed metric. Do not auto-promote a validated
   component — compose-and-measure.
4. **Re-validate cross-judge before trusting H3.1's number:** its +9/−2 was measured under gpt-5.6-sol as
   BOTH answerer and judge (a self-preference gradient — audit is right this was never flagged). The live
   wave-3 dev loop already uses the official split (Qwen reader / gpt-5.2 judge — the sol-both criticism does
   NOT apply to the current static loop), so any H3.1 re-home is measured under the official pairing, not
   trusted from its sol-both origin.

## 6d. PROCESS LESSONS banked from the same audit (binding)
- **Deterministic cleanup precedes expensive LLM attribution.** H2's 54-agent/6.7M-token attribution ran on
  the un-rescored 317-failure set; the comparator rescore (zero model calls) later proved ~71 were
  scorer-mangled correct answers. Always run the free deterministic pass first, hand the LLM a clean set.
- **Sober framing:** honest 3-day ledger = V1 +3 real, V2 +24 real instrument delta, ZERO new-capability
  promotions yet. The 119→205 was ~+62 comparator-correction + SpendGuard, i.e. measurement recovery, not
  capability. Arm E is the first live shot at a real promotion and is NOT banked until the full-451 net-gate
  clears. Do not narrate measurement wins as capability wins.
- **Budget the EXPENSIVE lane too:** OpenRouter tracked to the cent ($/$15) while Sol/codex internal runs
  (dozens of 120–500q passes + a 6.7M-token workflow) went unbudgeted. Keep a rough internal-lane tally.
- **A number told to the owner must survive the rescore BEFORE it is told** (the +86→+24 reached check-in #1
  first). **Outward actions need a completion record, not faith** (the tag-push casualty). **Durable crons are
  the default** for anything outliving a session (the checkpoint-cron death). All now standing rules.
- **Bound the corpus ceiling before spending on recall:** a deterministic all-451 gold-vs-stored-text audit
  (dispatched 07-25, → CORPUS-COVERAGE-CEILING.md) tells us how much headroom is real vs ingest-capped —
  instrument-first, one layer deeper, before more H5-class work.
- **H5 is mostly WIRING not research** (audit): the v1–v3 chunk-embedding + int8 two-stage KNN stack is
  already in-tree post-R1 = state-granularity retrieval. Any H5 follow-up dispatch says "evaluate wiring the
  recall stack already shipped," not "explore from scratch" — VERIFY in-tree before dispatching.

## 6. Pointers
GATE AUTHORITY NOTE: every gate summary in this document is an abbreviation — the GitHub ISSUE BODY carries the
complete, binding gate text; score against the issue verbatim.
Specs: SPEC-H6-AGENTIC-LANE.md · SPEC-W2A-CYCLE2.md · SPEC-H5b (superseded, archived) · H7 epic (#149).
Evidence: hermes-benchprog-h4/artifacts/OFFICIAL-RESULTS.md (+OFFICIAL-FULL-RAW) · H5B-SWEEP-REPORT.md ·
W2A-* artifacts · check-in #2 (#107 comment). Ops: bench/RUNBOOK.md (fork) · ~/.claude/runbooks/
hermes-benchmark-ops.md · OPUS-DRIVE-LOOP.md (continuation operating system). Tracker: #107 map + the
wave-3/H6/H7 milestones+issues posted 07-24.
