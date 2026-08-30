# SPEC — W2a CYCLE 2 (V1 answer-layer repair, iteration 2) — DRAFT, execution OWNER-GATED

_Authored 07-24 ~05:15 by the orchestrator, BEFORE any cycle-2 work exists. Gates below are predeclared now;
they do not change after numbers land. Prereq: owner selects option (A) in check-in #2 item 4._

## Diagnosis classes to fix (all with named qids, from cycle-1 evidence)

1. **Over-caution / refusal-where-evidence-suffices** — 37f165cf (refused the 440+416 cross-time sum r1 computed),
   af082822 (refused "2 weeks ago"). Contract's evidence bar reads too strict for cross-time aggregation.
2. **Terse abstention** — 80ec1f4f_abs (abstained correctly but dropped the "no museum visits in December ⇒ 0"
   elaboration the judge credits). Abstentions must state the derived conclusion, not bare "I don't know."
3. **Premise-check answer-through** — gpt4_93159ced_abs (engaged the contradiction NovaTech≠Google, then answered
   the calculation anyway instead of stopping at "you haven't started at Google").
4. **C2 transport** — selector quotes failing exact `includes` (suspected CLI-transport unicode; normalize NFKC +
   quote/dash folding before match) and `inferOperation` misreading "compared to my first X" as ordinal.
5. **Comparator selection under recency-first** — efc3f7c2 (picked 7:30 instead of 6:30 as the other-weekday wake
   time). For compare-across-states questions, the NON-current side of the comparison must come from the state the
   question anchors, not the most recent mention.

## Regression set (all must hold; flaps get 2/2 same-wording probes, documented)

- The 6 card-caused losses (07741c45, dad224aa, ba358f49, a2f3aa27, 7405e8b1, 31ff4165)
- The 11 cycle-1 gains + 13 C2 cases (threshold 11/13, unchanged)
- NEW: the 5 class-exemplar qids above (37f165cf, af082822, 80ec1f4f_abs, gpt4_93159ced_abs, efc3f7c2)
- Cards byte-stability 10/10 · bun tests green · tsc/prettier clean

## Gate chain (PREDECLARED — note the blind redesign and why it is legitimate now)

1. **Paired-120** (manifest caa16e03, control frozen): must project **≥+8 vs control** (unchanged bar) AND
   **net ≥+3 vs cycle-1 candidate on the same 120** (new bar: cycle-2 must beat cycle-1, not just control —
   cycle-1 showed +11-vs-control can hide net-0-vs-predecessor).
2. **Blind-100 R3 — REDESIGNED (documented gate revision, made before any cycle-2 numbers exist):**
   Cycle-1's blind failed structurally: the exclusion-union stripped all r3-control failures, making "positive net"
   unpassable (control 100/100 on the draw). Root cause: r3-control is wrong on only ~59/500; fresh-forever draws
   have no gain-room by construction. **New design:** stratified fresh draw of 100 from qids NEVER used in any
   regression/repair replay (paired-120 and blind-R1/R2 qids ARE eligible for the strata only if never inspected
   per-qid in tuning — the 17 repair qids and all per-qid-diagnosed qids are permanently excluded), composed of
   ~15 control-wrong + ~85 control-right (proportional to the pool), WITH the control slice computed AT FREEZE TIME
   and the achievable-range documented in the manifest. **Gate: losses ≤2 AND net ≥0** (non-regression form —
   correctness-level, achievable by construction, verified achievable at freeze). Freeze + sha256 before launch.
3. **ONE full-500 primary** only if 1+2 green. Bar **≥450** unchanged; MISS branch 445–449 ⇒ bank tag + close V1-L1
   (no cycle-3 without owner). Clone-ingest fast path; no re-ingest.

## Execution notes
- Branch off bench/v1l1-repair @8abae5f (wt-v1l1). Contract wording changes replay-ablated as in cycle-1 (the
  4-iteration wording table protocol worked; keep it).
- Two Sol streams max; snapshot-first everywhere; artifacts under hermes-benchprog-h1/artifacts/W2A-C2-*.
- Dispatch lane: coder-class agent for the fix build; fast-worker for launches; Fable scores all gates.
- Budget: ONE cycle. If cycle-2 misses its paired bar, V1-L1 closes at 444 pending owner.
