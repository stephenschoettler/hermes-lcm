# SPEC — Wave-3.5 mechanism family (synthesized 07-25 opus judge; wave35-family-design workflow)

Ground truth confirmed. The W3B-ARME artifact settles the central dispute in all three critiques: **6/8 of arm-E's current live residuals die at delivery seat-selection, not recall or reader-framing.** That reorders the whole family. Writing the spec.

---

# SPEC-W35-FAMILY.md — Wave-3.5 reader-funnel mechanism family (static lane, hermes-lcm)

**Author:** program architect (Opus) · **Date:** 2026-07-24 · **Status:** predeclared, pre-build
**Inputs synthesized:** reader-contract proposal (M1–M4) · W3.5-QD retrieval proposal (QD1–QD4) · C4 precision-rerank proposal (tier0/tier1/guardrail) + all three adversarial critiques
**Ground-truth files read for this spec:** `PROGRAM-ARCHITECTURE.md` §6b/§6c/§6d · `artifacts/W3B-ARME-UNKNOWNS-REDUCIBILITY.md` (both at `/Volumes/LEXAR/Codex/session-notes/2026-07-23/hermes-benchprog-h1/`)

---

## §0. The anchor fact that reorders every proposal (read first)

All three proposals were drafted against the pre-arm-E **W3c** sample (211-row official-enterprise-run). The **only current, post-arm-E ground truth** is `W3B-ARME-UNKNOWNS-REDUCIBILITY.md`, and it says something the proposals under-weight:

> **0/8 irreducible · 2/8 REDUCIBLE-retrieval (762d34b8, 2975030d) · 6/8 REDUCIBLE-delivery** (ee68431d, 1bfd2eac, 32d04f31, 7e32e4a2, b54161f8, c0696888). "The dominant residual failure mode has moved **downstream of retrieval**… gold state reaching the candidate pool (sometimes surviving the C1 diversity cap) and getting cut at final composition/token-budget selection, often losing to low-information boilerplate (task-goal headers, adjacent catalog items, page furniture)."

Its own verdict names the wave-3.5 work: **"tightening the diversity-cap's per-trajectory state-selection scoring and the final compilation/token-budget step that is now the dominant loss point."**

Consequences, all binding on this spec:

1. **The family's center of mass is DELIVERY seat-selection, not recall (QD) and not reader-contract text.** The precision leg (C4) is promoted to flagship; QD and reader-contract are narrower, later, ablatable knobs.
2. **Every proposal's headline case evidence is potentially stale.** 2d4e08b9 / 882965b7 / 1cf04706 (cited by QD and reader-contract) are **not in arm-E's current 8-residual set** — their live status is *unverified*, not merely unproven. Stage 0 re-verification is a hard prerequisite before any component gets promotion credit.
3. **M1's flagship case (c0696888) is a delivery-selection loss, not a delivery-text-framing loss.** The decisive state "lost the diversity cap to two low-value task-boilerplate states… the task GOAL text, repeated." Tagging already-delivered boilerplate as "background" cannot restore a state that was never delivered. This kills M1-as-designed and re-points its true target (see cut list).
4. **The larger enterprise mass (64/95, 20/20 sampled = needle-ABSENT hub-crowding) is a RECALL failure explicitly out of scope for this family** — it needs a C1-internal anti-hub cap, which is armG-adjacent and demands its own first-principles design. Parked as **wave-4** (§5), not smuggled into wave-3.5.

The §6c doctrine governing the whole spec: **gate FAMILIES on the composed whole-funnel net, each component carrying its own component-level evidence, compose-and-measure, keep a component only if it improves the composed metric** (arm G's 36.7→30.0 regression proves composition can demote needle-bearing states — the funnel cuts both ways). Official pairing only: **Qwen3.5-9B reader / gpt-5.2 judge, never sol-both** (§6c point 4, the H3.1 self-preference-gradient lesson).

---

## §1. Component set to build (default-off knobs, dependency order)

Naming follows the live W3b convention: **C1** = diversity cap, **C2** = adaptive excerpt, **C3** = budgeted sharp compilation / final token-budget fill. Wave-3.5 adds C4/C5 (delivery), QD (recall, pre-C1), RC (reader-contract, at render inside C3). Every knob ships **default-off, byte-identical output when off**, proven on the golden-451/451 on both W3A-dbwork DBs before any promotion attempt — the same proof bar every prior W3b component met.

### Stage 0 — Prerequisites (no new mechanism; unblocks honest measurement)

Build these first; they are the fix for the systemic flaw (evidentiary staleness) all three critiques independently flagged.

- **0a. Freeze the promoted arm-E baseline.** Wave-3.5 composes onto the *promoted* arm-E (cap2 + adaptive-excerpt + state-quota32), snapshot + sha256 manifest frozen before any number is produced (§6d snapshot-first). This is the 36.7%/18.6% dev-loop / full-451 reference for every gate below.
- **0b. Fix the `harness.py:923` malformed-provider-response crash** (`TypeError` on `response.choices[0].message`) that blocked *both* prior official Knob-H enterprise attempts (arm H, iteration 4). Until this is fixed there is **no completed official accuracy number for the title-boost lever on any qid set** — so nothing downstream may claim it "works live."
- **0c. Re-verification replay grid + zero-generation smoke harness.** Re-run the frozen arm-E baseline and record, per qid, `delivered_evidence_refs` / `survivor_state_ids` / pre-cap pool (the exact telemetry W3B-ARME used). Output: the **current** live-residual classification for every case any component claims to fix — no component gets promotion credit for a case not confirmed still-broken under the frozen baseline.

### Component C4 — Delivery seat-selection precision (**flagship**; targets 6/8 live residuals)

Two default-off sub-knobs, both operating **strictly after C1(cap)+C2(excerpt) produce their survivor pool and before/at C3's final ≤4000-token budget fill.** Neither reaches into C1's MMR scoring function — that is the exact boundary arm G crossed (it penalized states by lexical Jaccard-to-siblings, demoting needle-bearing states that structurally resemble neighbors). C4 conditions on **(question, candidate) relevance**, never on redundancy-to-siblings, so it structurally cannot repeat arm G's mechanism.

- **C4a — verbatim title/label reserved-seat guarantee.**
  - *What:* extend the existing (default-off) `HERMES_LCM_TITLE_BOOST` from a candidate-order bias into a small reserved allocation at C3 final fill — reserve **at most 1–2 seats** for a state whose title/field-label contains a verbatim question phrase, so a literal match can never be pushed out by ordinary fill order. **Step-1 (evidence-only, no new code): replay Knob H alone against arm-E's 6 REDUCIBLE-delivery residuals** (once 0b lands) to see how many it fixes for free before writing the reserved-seat guarantee.
  - *Targets which failure class:* the label/title-shaped subset of the 6/8 delivery losses (b54161f8 "Microsoft Powerpoint"/"Siebel Client"; the procedure-title family W3c-1cf04706 *if 0c confirms it is still live*).
  - *Why the 9B benefits:* the fixed reader does no search of its own and is honest about gaps (states "not visible" when material is absent) — so precision in *what is handed to it* moves its output predictably (M1/M4 finding), unlike volume (M1: compactness-bound, medians identical answered-vs-unknown). A literal anchor is exactly the cue a weak reader needs for label-shaped questions and costs it nothing to consume.
  - *Knob:* `HERMES_LCM_TITLE_BOOST` (exists) + new sub-flag `HERMES_LCM_TITLE_RESERVED_SEATS` (int, default 0; range 1–2).
  - *Risk:* verbatim n-gram can be incidental noise (W3c web false-positives: "My Account" hitting a site-wide nav link; bare "B" corpus noise). Mitigate by capping the reserved allocation at 1–2 and requiring sufficiently long/distinctive phrases — never an unconditional override of C1/C3 order.

- **C4b — de-boilerplate the final seat-selection (question-conditioned).**
  - *What:* at C3 final fill, down-weight the specific low-information state shapes W3B-ARME names as the recurring winners — **task-GOAL-header states (repeated goal text), adjacent-catalog / page-furniture states** — using a **(question, candidate) relevance score**, not sibling-similarity. This directly implements the W3B-ARME verdict's "final compilation/token-budget step" half.
  - *Targets which failure class:* the core delivery-loss mechanism in c0696888 (needle lost the seat to two repeated task-GOAL states), 32d04f31, 7e32e4a2 (needle cut for other catalog items' pages), b54161f8.
  - *Why the 9B benefits:* removes the exact confound the reader cannot see past — boilerplate rendered in the identical 7-line structural block as genuine content (`_rendered_hit_text` today emits pure provenance with zero salience signal).
  - *Knob:* `HERMES_LCM_C3_DEBOILERPLATE` (default off).
  - *Risk (armG-shaped, take seriously):* a boilerplate proxy misfires whenever a genuinely needle-bearing state resembles page furniture (repeated field labels, similar form pages). This is *the* funnel-cuts-both-ways risk. Mitigation is not rhetorical ("descriptive wording") — it is the **per-case delta gate** (§3): C4b is kept only if it improves the *composed* net AND does not demote any confirmed-live needle in the residual set.

### Component C5 — Bounded local cross-encoder rerank (delivery-stage reorder; **default-off, smoke-gated, cut-if-no-composed-gain**)

The task explicitly requires a rerank component; this is it, positioned as the *mechanized generalization* of C4b's intuition (score "does this passage answer this question" directly) and gated hard per critique-3's caution.

- *What:* a small **local** cross-encoder (ms-marco-MiniLM-L-6-v2-ONNX class ~80MB CPU, or bge-reranker-v2-m3 if AXTree/DOM-length text needs a longer-passage model) scores (question, candidate-state) pairs over C1's *modestly over-provisioned* survivor pool; C3 walks the reranked order to commit its budget. C1's MMR is untouched.
- *Targets which failure class:* same 6/8 delivery residuals as C4, as an alternative/complementary ordering signal.
- *Why the 9B benefits:* a cross-encoder makes upstream the exact relevance judgment the reader needs made for it, versus arm G's novelty-vs-siblings proxy.
- *Knob:* `HERMES_LCM_CROSS_RERANK` (default 0; active = top-N depth, e.g. 10–40; omitted from adapter call when off).
- *Hard gate before any official spend (§6d "deterministic cleanup precedes expensive LLM attribution"):* a **zero-model-generation smoke test** — score the reranker's own top-20 against the **already-labeled** arm-E-8 + W3c-30 gold-state-id sets and confirm it ranks known-gold near the top. **If it does not clear the smoke test, C5 is cut before it costs one official-protocol token.** Explicitly cut if it does not improve the composed net over C4-alone (avoid a new ML dependency for no composed gain — critique 3's central objection).
- *Risk:* domain mismatch (MS-MARCO short-passage training vs long DOM dumps); smoke-test **circularity** (validating on the same labeled set it is later scored against) — mitigate by holding out a fresh sample in the smoke set. Latency stays on the LAFS fast side (small local model, top-N cap, no API round-trip).

### Component QD — Recall candidate widening (pre-C1; targets the 2/8 retrieval-gating residuals + multi-entity)

Additive, upstream, zero-LLM-call. Deliberately **narrowed** per both QD critiques. Realistic ceiling is **0–1 full-451 points, not 1–3** — because 6/8 live residuals are downstream of where QD operates, QD cannot reach them by its own scoping.

- **QD3 (build first — mandatory architectural rail, no accuracy claim).** Pin the decomposer to a pure regex/rule extractor (comma/semicolon split, quoted-string, temporal-token lexicon, **capitalized multi-word proper-noun chunking** — the latter added to catch un-punctuated titles like "Data Management Delete Job" that a naïve comma-splitter silently no-ops on). Zero model calls → byte-identical sub-query sets on frozen inputs; batched embeddings + parallel ANN. Enum `query_decomposition.decomposer_backend {regex_v1 (default, locked), llm_fallback (disabled, requires a documented latency-budget addendum)}`. This forecloses the one demonstrated failure in this class — AgentRunbook-R's LLM-controller multi-query cost.
- **QD1 (entity fan-out).** Up to K=4 entity sub-queries per enumerated question, unioned into the pre-cap pool. `query_decomposition.entity_fanout_enabled` (default false), `entity_fanout_max_k` (default 4). **Predeclared credit condition (tightened):** Stage-0c must confirm 2d4e08b9 / 882965b7 are *still* spurious-unknown under the frozen arm-E baseline before any "fixed by QD1" credit; the only *confirmed-current* target is **762d34b8** (and even it is mixed — one sub-facet already reaches the pool and still dies at delivery, a preview of QD's ceiling).
- **QD4 (literal/exact-phrase sub-query, procedure-category only).** BM25/substring sub-query for long distinctive quoted or multi-token proper-noun phrases. `query_decomposition.literal_boost_enabled` (default false), scoped to procedure-category questions. **Re-verify 1cf04706 (or an equivalent procedure-title case) is still live under the frozen baseline before crediting.** Restrict to sufficiently long/distinctive phrases (the "My Account"/bare-"B" false-positive discipline).
- **CUT: QD2 (time/action facet).** Zero case evidence anywhere in the 30-case classified sample or the 8-case current residual set — the proposal's own text concedes "a hypothesis, not yet demonstrated." Do not build, do not gate; revisit only if a facet-driven miss actually surfaces after QD1 lands.

### Component RC — Reader-contract text scaffolding (at render inside C3; smallest, last)

- **RC-M2 (fast-abstention / no-speculation scaffold) — ship, but scored on the LATENCY axis only.** One short synthetic stop-condition item prepended to `memory_context`. Targets the F-class 20K-token self-doubt loop (85251f36, e9fde9fe, 762d34b8-web, 609acb91; verified `finish_reason: length` with `\boxed{UNKNOWN}` mid-ramble). **These already score UNKNOWN by parser fallback → RC-M2 does NOT move accuracy/spurious; it moves LAFS query-latency/cost.** `HERMES_LCM_FAST_ABSTAIN` (default 0). **Load-bearing guard (promotion-blocking, not optional): a paired answered-rate check** — weak readers often need *more* reasoning tokens to land correct, so verify RC-M2 does not cut off tokens that would have reached a correct `\boxed{}`.
- **RC-M1 (salience tagging) — HELD, contingent on a population that does not yet exist in evidence.** Do NOT tag every rendered item off a ranking-derived heuristic (that is a soft arm G: a weak 9B reader treats an explicit `[relevance: background]` tag as quasi-directive). Its two cited cases don't support the pathway (c0696888 = selection loss; hub-crowding = reader already correctly recognizing non-needle). **Gate to build:** Stage-0c must first *find and quantify* the population where needle + boilerplate are **co-delivered** and the reader visibly conflates them. If that population is empty or tiny, RC-M1 is not built. If built: `HERMES_LCM_SALIENCE_TAGS` (default 0), descriptive-not-directive wording, and a required test against the actual 9B's behavior on **deliberately mistagged** content before promotion.

### Cut list (explicit, with rationale)

| Cut | From | Why |
|---|---|---|
| **M4 (calibrated-inference / analogy license)** | reader-contract | Fights the FIXED, harness-owned system prompt ("Do not guess. Never attempt to guess…"); evidence 2/38 against a verified **20/20 correct-abstention** population; asks a weak 9B to hold a fine "generalize-vs-guess" distinction it is worst at → highest odds of a net-negative hallucination surprise, worse-shaped than arm G. Textbook design-for-Sol. |
| **M3 (multi-entity decomposition labeling)** | reader-contract | 2/3 of its own cited cases (2d4e08b9, 882965b7) are root-Class-C **needle-absent** — nothing delivered to group. Its effective retrieval-side value is already covered by QD1; the reader-facing labeling half has a ~1-case reachable population. |
| **QD2 (time/action facet)** | QD | Zero case evidence; author-conceded hypothesis. |
| **C4-tier1 as an unconditional build** | C4 proposal | Kept only as **C5, behind the smoke-gate + composed-net cut** — not an automatic build. |
| **RC-M1 as-designed (tag-everything)** | reader-contract | Held pending an empirically-identified co-delivery population; redesigned, not shipped as proposed. |

---

## §2. Dev-loop protocol (identical discipline to wave-3)

1. **Slices:** the 60q official dev-loop slice, both domains, run under the **official Qwen3.5-9B reader / gpt-5.2 judge** split — never sol-both (§6c point 4; H3.1's contaminated +9/−2 is the cited precedent).
2. **Joint primary metric (both must improve):** **accuracy AND spurious-unknown rate**, tracked together, against the frozen arm-E baseline (36.7%/18.6% on this slice). **No accuracy number is reportable without its paired spurious number** — the exact rule that would have caught arm G at the dev-loop stage instead of after an official run.
3. **Two mandatory secondary axes** (not credited to the accuracy gate, but promotion-blocking if they regress):
   - **Delivered-token cost per item / total context tokens.** Non-negotiable because M1 says this reader is *compactness-bound*; every RC/label mechanism adds bytes to an already ~19–25K-token context that itself drives the F-class loop. A scaffold that improves selection but bloats the context can net-regress.
   - **LAFS query-latency / completion-token spend** — the axis RC-M2 is scored on, and the guardrail C5/QD3 protect (must stay on the fast side of the Pareto frontier; no LLM in the retrieval path).
4. **Ablation matrix (compose-and-measure, never auto-promote):** each knob measured (a) alone on the frozen baseline, (b) composed, (c) leave-one-out from the full composed set. A component is **kept only if the leave-one-out shows it improves the composed net.** This is the operational form of the arm G lesson.
5. **Order of spend (§6d — deterministic before expensive):** golden-451 byte-identical off-proof → provider-free replay-grid deltas (pool / Delivered-Recall@16 / preservation-retained-vs-disturbed / residual-8 per-case) → C5 zero-generation smoke test → only then the 60q official LLM slice. Snapshot raw outputs to session-notes `artifacts/` *before* any subsequent run (volatile-score rule).

---

## §3. Predeclared family promotion gate (frozen before numbers)

W3.5 promotes as a **family** on the **composed full-451 official protocol**, scored against the GitHub issue body verbatim (gate-authority note: the issue text is binding, this doc abbreviates). All checks predeclared; none skippable:

1. **Default-off proof:** golden-451/451 byte-identical with all W3.5 knobs off, both W3A-dbwork DBs.
2. **Composed full-451 net vs frozen promoted arm-E baseline** — the composed net must clear a **net non-regression + positive-net** bar (net accuracy ≥ arm-E, no category regressed below arm-E). One-primary law applied to the **composed family**, not to any single knob.
3. **Spurious strictly improved** vs the arm-E baseline's spurious rate (the §6c recalibration — spurious-strictly-improved replaces the pre-recalibration hard ≤15% dev-slice condition; scored on full-451, not the 60q slice).
4. **Category-integrity pass** (Lane S's standing W3b promotion requirement): no official category (enterprise/web × static split) regresses.
5. **Per-case delta gate (the arm-G backstop, and it is per-case, not aggregate):** against the confirmed-live residual set from Stage-0c **plus a fresh held-out sample**, no component may demote a confirmed-live needle even while the aggregate rate improves — the hidden-wash failure arm G proved is possible. Aggregate-only gates can pass while masking a swap; this closes that hole.
6. **Secondary-axis non-regression:** delivered-token cost and LAFS latency do not regress (RC-M2 credited only on latency; C5/QD must not push retrieval latency off the fast frontier).
7. **Reader/judge provenance:** every promotion number produced under Qwen3.5-9B / gpt-5.2, survives a rescore before it is reported (§6d "a number told to the owner must survive the rescore before it is told").

---

## §4. Sequencing (explicit — this is wave-3.5, not another arm-E iteration)

1. **arm-E promotion banks FIRST** under the recalibrated §6c gate (spurious strictly improved 27.9%→18.6% + composed full-451 net + category-integrity). Wave-3.5 does **not** block, gate, or compete with that run — it is cheaper, already sequenced, and defines the frozen baseline W3.5 sits on. *(Note the live tension surfaced by the critiques: `W3B-ARME` records arm-E's 18.6% as above the pre-recalibration ≤15% dev-slice gate with a "continue engineering" verdict. Under the §6c recalibration the promotion gate is "spurious strictly improved," which arm-E clears — so it banks. If the program instead holds the hard ≤15%, then C4 below is literally arm-E's iteration-4 delivery fix and must land first. Either way C4 is the next real work; the sequencing decision affects only the label, not the build order.)*
2. **Stage 0** (baseline freeze + `harness.py:923` fix + re-verification/smoke harness) — the gate that converts stale case evidence into confirmed-live targets. Nothing downstream gets promotion credit until 0c reclassifies its target cases.
3. **Build in dependency order, attacking the dominant bottleneck first:**
   **C4 (delivery seat-selection) → C5 (rerank, smoke-gated) → QD3→QD1→QD4 (recall widening; re-tune C4/C5 selection against the widened pool, since QD changes the candidate statistics they were calibrated on) → RC-M2 (latency) → RC-M1 (only if 0c finds the co-delivery population).**
4. **Compose-and-measure at each addition;** drop any knob whose leave-one-out doesn't improve the composed net. Promote the surviving family through the §3 gate as one unit.
5. **Out of scope / parked to wave-4:** the recall-side **anti-hub-crowding cap** for the 64/95 needle-ABSENT enterprise mass — the single largest unaddressed headroom, and a differentiator (no competitor in TECHNIQUE-SURVEY-154 ships an anti-hub cap). It is C1-internal, armG-adjacent, and needs its own first-principles design + gate; explicitly not smuggled into wave-3.5.

---

## §5. Risks + the one mechanism most likely to move the number

**The one mechanism most likely to move the fixed-9B number: C4 (delivery seat-selection precision), specifically C4b's question-conditioned de-boilerplating of the final fill.** This is where **6/8 of arm-E's confirmed-current live residuals die** (needle reaches the pool, sometimes survives the cap, then loses the last seats to repeated task-GOAL headers / adjacent-catalog / page-furniture). It is the exact stage the `W3B-ARME` verdict names as "now the dominant loss point," it operates on the 9B's real bottleneck (compactness/precision, not volume), and it does so without touching C1's MMR (so it cannot repeat arm G's mechanism). Honest headroom: closing most of the 6/8 could move the 60q slice spurious from 18.6% toward ~5–7% (the swing that clears the recalibrated gate with margin); on full-451 a more conservative **+2 to +6 static points** on top of promoted arm-E. QD contributes ~0–1; RC-M2 contributes on latency only; C5 is upside-if-it-clears-its-smoke-gate.

**Top risks, ranked:**

1. **C4b / C5 repeat arm G's *shape* with a different mechanism** — a boilerplate proxy or a domain-mismatched cross-encoder rewards topically-plausible hub states (heavy shared vocabulary) over the true needle, regressing the composed net while looking locally reasonable. *Mitigation:* the §3.5 **per-case delta gate** (not aggregate-only) + C5's zero-generation smoke gate + leave-one-out ablation. The proposals' claim of "structural immunity" is overstated — C4/C5 avoid arm G's *exact code path* but not its *failure shape*; the gate, not the architecture, is the real protection.
2. **Evidentiary staleness misattributes credit** — a case arm-E already solved gets credited to QD1/C4a. *Mitigation:* Stage-0c reclassification is a hard precondition for credit.
3. **Context-bloat backfire** — RC/label components add bytes to a compactness-bound reader, feeding the F-class loop. *Mitigation:* delivered-token-cost is a promotion-blocking secondary axis (§2.3).
4. **F-class ceiling** — a residual slice of unknowns (2–4 of ~30 sampled) is a decoding-loop pathology **no context-composition mechanism can fix**; RC-M2 converts them to fast clean UNKNOWNs (latency win) but not to correct answers. This caps the family's reachable accuracy headroom and must be stated plainly rather than narrated away.
5. **Strategic mis-scoping** — the largest headroom (64/95 needle-absent hub-crowding) is *recall*, explicitly out of this family. Wave-3.5 is a precision/delivery mop-up on the arm-E residual, not the big lever; treating it as "the lever" would overstate its weight against the ~68-point RAG-42.8-parity gap. That lever is wave-4's anti-hub cap.

**Net honest expectation:** a disciplined delivery-led family whose credible movement on the official 9B number is dominated by C4, bounded to low-single-digit full-451 points on top of promoted arm-E, gated so that the arm-G failure mode is caught at the dev-loop/per-case stage rather than after an official run.
