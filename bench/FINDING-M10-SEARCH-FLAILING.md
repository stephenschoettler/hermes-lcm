# M10 — SEARCH-FLAILING UNIFIES M7 AND M8: accuracy and latency are the SAME problem, and it is a memory problem

_Found 2026-07-25 by the orchestrator (Opus 5) from the P4 agentic run's 455 per-question agent traces
(`query_traces/*/attempt_*/summary.json`: `duration_seconds` + `semantic_status_counts` = number of store
searches the agent issued). All numbers computed, not estimated._

## The measurement

**Correlation between number of searches the agent issues and its latency: r = 0.710.**

| searches issued | n | mean latency | **accuracy** | agent output tokens |
|---|---|---|---|---|
| 1–2 | 27 | 113.2s | **81.5%** | 5,997 |
| 3–4 | 91 | 122.9s | 72.5% | 6,752 |
| 5–7 | 135 | 168.6s | 74.1% | 9,567 |
| **8+** | **202 (45%)** | **256.5s** | **55.9%** | 14,543 |

**Searching more is a symptom of failure, not of effort.** When the store answers on the first or second search,
the agent is both *fastest* (113s) and *most accurate* (81.5%). When it has to search 8+ times it is slowest
(257s) and least accurate (55.9%) — it is flailing, and it usually still gets the answer wrong.

**This means accuracy and latency are not a trade-off here. They are the same variable, driven by retrieval
quality on the first search — which is exactly what a memory system is for.**

## The unification with M7

| class | n | searches (mean) | latency | accuracy |
|---|---|---|---|---|
| abstention / false-premise | 130 | **11.1** | **250.1s** | **38.5%** |
| answerable | 325 | 7.5 | 173.2s | 77.2% |

Abstention questions are **41% of the entire 8+ search bucket** but only **7% of the 1–2 bucket**. The agent
hunts for a field that does not exist, searches ~11 times, burns 250s, and then hallucinates an answer anyway.

**M7's negative-evidence disclosure is therefore also the biggest latency mechanism in the program.** Telling
the agent "searched X: no matching state found" lets it stop early instead of flailing. One mechanism, both axes:
- accuracy: abstention 38.5% → target ~60% (M7's arithmetic: +27 questions)
- latency: if abstention questions behaved like fast answerable ones (~120s), overall mean drops
  **195.2s → 158.0s** with no other change — and that is *before* the effort knob or a tail cap.

## Why this reframes the whole program

The old framing: "our memory makes the reader more accurate." The measured framing:
**our memory's job is to answer the agent's question on the FIRST search — and doing so wins accuracy AND
latency simultaneously, which is precisely what LAFS scores.** Retrieval quality is a latency lever. That is a
far stronger product story than "+3 questions right," and it is the story the leaderboard's own metric rewards.

## Stacking estimate (to be measured, not assumed)

- baseline: 66.1% @ 195.2s → LAFS 0.0000
- + M7 (abstention stops flailing): ~73% @ ~158s → still 0.0000 (above the 108.3s wall)
- + effort=high (measured −37% latency, −5 acc): ~68% @ ~100s → **inside window A (<108.3s, >58.6%) ⇒ non-zero**
- + tail cap (mean is tail-dragged: median 104.5 vs mean 126.3 at effort=high) → further
The order matters: M7 first (it *removes* work rather than truncating it), then effort, then cap.

## Falsifiers / next tests
1. Does the search count actually drop when negative evidence is disclosed? Instrument `searches_per_question`
   as a first-class metric in the M7 pilot — it is the leading indicator.
2. Causality caution: 8+ searches may partly *mark* intrinsically hard questions rather than cause the loss.
   The M7 pilot is the clean test — it changes the evidence, not the question.
3. Confirm on the static lane too: does the same flailing show up as more retrieval rounds / larger contexts?
