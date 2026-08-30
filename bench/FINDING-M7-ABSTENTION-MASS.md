# M7 — THE ABSTENTION/FALSE-PREMISE CLASS IS THE PROGRAM'S BIGGEST UNTOUCHED LOSS MASS

_Found 2026-07-25 by the orchestrator (Opus 5) from the raw per-question data of BOTH completed full runs.
Written to ~/.claude because /Volumes/LEXAR became inaccessible (macOS removable-volume permission reset at the
app upgrade) — move to the program packet artifacts dir once access is restored._

## The numbers (raw per_question.jsonl, both lanes — computed, not estimated)

| lane | answerable | abstention (128 q) | abstention share of ALL losses |
|---|---|---|---|
| STATIC (125/451) | 109/323 = 33.7% | **16/128 = 12.5%** | 112 losses = **34%** |
| AGENTIC (298/451) | 249/323 = 77.1% | **49/128 = 38.3%** | 79 losses = **52%** |

**The agentic lane's ANSWERABLE accuracy is already 77.1%** — higher than AgentRunbook-C's 74.9% *overall*.
Nearly the entire remaining gap to the published points is the abstention class.

## The mechanism (verified by reading the answers, not inferred)

On the 128 abstention questions the agentic reader **asserted an answer 122 times (95%)** and said UNKNOWN only 6
times. Of the 122 assertions, **73 were wrong — hallucinations against a false premise.** These are
false-premise questions: "what SECOND field decides duration besides risk level?" (there is no second field);
"which TWO fields appear on completion?" (only one does); "what fieldset appears between those sections?" (none).
The reader names a plausible field ("Impact", "Approval History") because **the curated evidence pack carries no
signal of absence** — only 21/73 (29%) of failing packs contain any negative-evidence phrasing at all. The agent
reports what it FOUND and silently drops what it searched for and did NOT find. Curation over-persuades: a tidy
evidence story makes a weak reader confident.

The 6 cases where the reader DID say UNKNOWN scored 0/6 — the official abstention checker requires asserting the
specific negative conclusion ("there is no second field"), not a bare "I don't know". Same lesson V1's W2a repair
learned ("state the derived zero/none conclusion").

## Why this is the mirror-image of what the program has been fighting

All night the static lane fought **spurious-unknowns** (reader abstains when it should answer — 29% of answerable).
The agentic lane's dominant failure is the **exact opposite** (reader asserts when it should refuse the premise —
95% of abstention questions). Same fixed 9B reader, opposite failures, determined by how evidence is presented:
bulk 22k context → paralysis; tidy curated pack → overconfidence.

## Proposed mechanism: NEGATIVE-EVIDENCE DISCLOSURE (content-only, §2b-compliant)

When the curation stage searches for the question's key entity/field/relation and the store returns nothing (or
nothing matching), the evidence pack MUST record it explicitly: `searched "<term>": no matching state found`.
This shapes `memory_context` CONTENT (allowed) — NOT the fixed answer prompt (forbidden per §2b). It is also
exactly the owner's multi-call thesis (§6b): the agent already issues several searches; we simply require it to
REPORT null results instead of dropping them. Applies to both lanes — the agentic curation contract
(INSTRUCTION.md + memory_module_output schema) and the static compilation stage (a deterministic absence line
when a question-derived key term has zero pool hits).

## Arithmetic — this is the path over both thresholds

- **AGENTIC** needs +17 q for the published 69.9 point (315/451). Abstention 38.3%→50% = +15 q → 69.4%;
  →60% = +27 q → **72.1% (beats Codex 69.9)**; →70% → 74.9% (= AgentRunbook-C).
- **STATIC** needs +68 q for RAG-parity (193/451). Abstention 12.5%→50% = **+48 q** alone → 173/451; plus arm-E's
  projected ~+40 on answerable → ~213/451 = **47.2%, past the 42.8 RAG line.**

No other single class in either lane offers this much.

## Roadmap consequence

Wave-3.5 (#155, delivery seat-selection) targets spurious-unknowns on ANSWERABLE questions — real, but a smaller
mass and in the wrong lane for the published-point chase. **Negative-evidence disclosure should become the next
mechanism family, ahead of or parallel to wave-3.5.** Cheap to test: one 60q slice per lane; metric = abstention
accuracy up AND answerable not down (§6c joint family gate).

## Caveats / falsifiers (test before building at scale)

1. The abstention checker is LLM-judged (`llm_abstention_checker`, gpt-5.2) — confirm on a sample that correct
   negative assertions actually score, i.e. the class is winnable and not judge-capped.
2. Negative-evidence lines could *induce* spurious-unknowns on answerable questions (the opposite failure) —
   the dev-loop metric must be JOINT per §6c.
3. "Absence" must be derived from an actually-executed search returning zero, never asserted speculatively —
   otherwise it is a hallucinated negative.
