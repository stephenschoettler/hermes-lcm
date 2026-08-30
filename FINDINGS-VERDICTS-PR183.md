# PR #183 bot findings — verdicts

- Round 1 reviewed head: `418db28bcc85f24581cec6de75d6e8e91073f3b0`
- Round 2 source head: `af0f14226ee61fc9c18a2abfbab8cf4acbdb8516`
- Base: `cb92bf40c1d4c862cb56090792113b69d88660d4`
- Mode: validate-then-fix; two Highs first; no push
- Review source commands: `gh api repos/stephenschoettler/hermes-lcm/pulls/183/comments --paginate`
  and `gh api repos/stephenschoettler/hermes-lcm/pulls/183/reviews --paginate`
- Raw review JSON and command logs are retained off-repo by the operator.

## 1. High — injected `OR` glue becomes a scored term

Verdict: **Not applicable as stated; invariant proof strengthened.**

At the reviewed head, `extract_search_terms()` already rejects every
case-insensitive member of `_BOOLEAN_OPERATORS`, so the injected uppercase
`OR` tokens do not reach fetch-limit or directness scoring. The precheck
produced terms `["dog", "vet", "appointment"]` and a directness score of `0.0`
for `"or hall or"`. Added
`test_search_prose_mode_operator_glue_contributes_zero_score` to make that
ranking invariant explicit. CodeRabbit comment `3673449204` is a duplicate
test-coverage suggestion and is included in this disposition.

Evidence: `high1-precheck.log`, `highs-fixed.log`.

## 2. High — Unicode-symbol LIKE fallback escapes the flag

Verdict: **Fixed.**

Both `MessageStore.search()` and `SummaryDAG.search()` routed Unicode symbols
to LIKE while `fts_prose_mode=False`. Failing-first tests reproduced both
routes. Symbol preservation is now passed to `requires_like_fallback()` only
when prose mode is enabled (and, for messages, the caller did not compose FTS
operators). Added flag-off route regressions for both backends and moved the
symbol-preservation assertions onto the flag-on path.

The original byte probe covered only an ASCII prose question. Its documented
blind spot was any Unicode-symbol query that distinguishes the historical FTS
route from the new LIKE route. The corpus now includes `licensed ©` plus a
symbol-bearing and symbol-free row.

Evidence: `highs-failing-first.log` (2 expected failures), `highs-fixed.log`.

## 3. Medium — trailing punctuation breaks symbol LIKE terms

Verdict: **Fixed.**

`sanitize_like_query("Find ©?")` previously yielded the term `©?`. Edge
question/exclamation punctuation is now removed when the token retains other
content, while punctuation-only fallback queries such as `???` remain intact.
The parameterized symbol search now exercises `©?`, `€?`, and `™?`.

Evidence: `medium-low-precheck.log`, `medium-low-postcheck.log`,
`punctuation-delta.log`.

## 4. Medium — lowercase `and`/`or` disappear before prose classification

Verdict: **Fixed.**

Classification now uses word tokens that retain ordinary lowercase
conjunctions; scoring extraction still excludes boolean-looking operator
tokens. `cats and dogs are common pets today` is now classified as prose and
builds `cats OR dogs OR common OR pets OR today`.

Evidence: `medium-low-precheck.log`, `medium-low-postcheck.log`,
`medium-low-fixed.log`.

## 5. Medium — conversational lead words leak into the disjunction

Verdict: **Fixed.**

Added `find`, `please`, `recall`, and `remember` to the prose stoplist while
retaining them as classification lead words. Parameterized regressions prove
each form reduces to the requested signal; for example,
`Can you remember my PIN?` now builds `PIN`.

Evidence: `medium-low-precheck.log`, `medium-low-postcheck.log`,
`medium-low-fixed.log`.

## 6. Medium — term-cap assertion proves formatting, not the bound

Verdict: **Fixed.**

The test now extracts actual search terms and asserts their count is bounded by
the module's `_PROSE_TERM_LIMIT`; it no longer hard-codes `12` via
`split(" OR ")`.

Evidence: `medium-low-fixed.log`.

## 7. Low — ranking test contains a tautological assertion

Verdict: **Fixed.**

The no-op truthy-set assertion was replaced with explicit ordered evidence:
the target must rank first, and the remaining ranked IDs must be exactly the
two seeded distractors.

Evidence: `medium-low-fixed.log`, `focused-parity-rerun.log`.

## 8. Low — balanced precision guard ignores curly quotes

Verdict: **Fixed.**

The prose classifier now treats a retained ASCII quoted phrase or a balanced
smart-quoted phrase (`“…”`) as a precision signal. The regression proves
smart-quoted prose remains conjunctive instead of becoming an OR query.

Evidence: `medium-low-precheck.log`, `medium-low-postcheck.log`,
`medium-low-fixed.log`.

## Round 1 final gates

- Focused parity: **411 passed** (`focused-parity-rerun.log`).
- Full isolated CI replica, Python 3.11 and `ulimit -n 1024`:
  **2755 passed, 1 skipped, 12 xfailed** (`full-suite-final.log`,
  `full-suite-final.xml`).
- Ruff: **passed** (`ruff.log`).
- Compileall, script py-compile, shell syntax, and `git diff --check`:
  **passed** (`compileall.log`, `py-compile.log`, `bash-syntax.log`,
  `git-diff-check.log`).
- Final flag-off base/candidate outputs: **byte-identical**, 2,104 bytes each,
  SHA-256
  `d9cf3621ed51669eeaff13642b8805d393e45838e351fd1bf236defd0b9e3219`
  (`flag-off-byte-identity-final.log`).
- No commit, push, PR reply, thread resolution, merge, deploy, or release was
  performed.

## Round 2

Validated against source head `af0f14226ee61fc9c18a2abfbab8cf4acbdb8516`
before editing. All seven raw findings remained applicable.

| Raw comment ID | Terminal disposition | Change and reproducible validation |
|---|---|---|
| `3674835828` | **Fixed.** | A question mark now requires a stopword, lead word, or at least six terms before prose routing. Compact technical questions remain conjunctive. Run: `python3 -m pytest tests/test_lcm_core.py -q -k "compact_question_keywords or conversational_question_disjunctive"` |
| `3674839099` | **Fixed with `3674835828`.** | The four named compact-query boundaries and the conversational positive boundary are covered by the same classifier fix. Run: `python3 -m pytest tests/test_lcm_core.py -q -k "compact_question_keywords or conversational_question_disjunctive"` |
| `3674835822` | **Fixed.** | Stopword-empty prose normalization retains the last non-lead subject candidate, so `will` remains searchable instead of restoring the full framing query. Run: `python3 -m pytest tests/test_lcm_core.py -q -k "stoplisted_subject_signal"` |
| `3674835811` | **Fixed.** | Message and summary Unicode-symbol LIKE routes now share bounded prose term extraction, including stopword/lead-word filtering and the term cap. Run: `python3 -m pytest tests/test_lcm_core.py -q -k "preserves_non_ascii_symbol_signal or filters_summary_symbol_like_terms or caps_disjunctive_terms"` |
| `3674839106` | **Fixed.** | `resolve_prose_sort()` owns implicit relevance promotion and is used by tools, message-store, and summary-DAG search. Run: `python3 -m pytest tests/test_lcm_core.py -q -k "prose_sort_promotion_is_centralized"` |
| `3674839116` | **Fixed.** | Bare possessive `s` tokens are excluded from both classifier counts while extraction behavior is unchanged. Run: `python3 -m pytest tests/test_lcm_core.py -q -k "possessive_s_tokens"` |
| `3674829341` | **Fixed.** | Machine-local artifact paths were replaced by reproducible `gh api` commands; raw logs are explicitly retained off-repo. Run: `python3 -c 'from pathlib import Path; text = Path("FINDINGS-VERDICTS-BOTS.md").read_text(); assert chr(47) + "Volumes" + chr(47) not in text'` |

### Round 2 validation

- Focused prose cases: **18 passed**.
  Command: `python3 -m pytest tests/test_lcm_core.py -q -k "prose"`.
- Focused prose/search/grep selector over the directly relevant modules:
  **20 passed, 400 deselected**.
  Command: `python3 -m pytest tests/test_lcm_core.py tests/test_lcm_recall.py -q -k "prose or search_query or grep"`.
- Full focused core and recall parity: **420 passed**.
  Command: `python3 -m pytest tests/test_lcm_core.py tests/test_lcm_recall.py -q`.
- Flag-off regression corpus: **byte-identical to the recorded source-head
  baseline**, 2,104 bytes, SHA-256
  `d9cf3621ed51669eeaff13642b8805d393e45838e351fd1bf236defd0b9e3219`.
  Command:
  `probe_dir="$(mktemp -d)"; python3 tests/probes/fts_prose_flag_off_probe.py . "$probe_dir/probe.db" "$probe_dir/output.json"`.
  The unchanged zero-score operator-glue regression also passed in the
  420-test focused parity run.
- Changed-file Ruff: **passed**.
  Command:
  `python3 -m ruff check search_query.py store.py dag.py tools.py tests/test_lcm_core.py`.
- The broader command `python3 -m pytest tests/ -q -k "prose or search_query or grep"`
  cannot collect in this standalone checkout because unrelated test modules
  require the absent host `agent` package/exported `LCMEngine`. It produced
  22 collection errors before selecting tests; this is not claimed green.
- Delivery checkpoint: **COMPLETE** for the local Round 2 disposition-batch
  gate; primary verdict **ADVANCE** to the orchestrator-owned Git/current-head
  CI lane. This does not claim PR merge readiness or current-head remote CI.
- No commit, push, PR reply, thread resolution, merge, deploy, or release was
  performed.

## Round 3 — re-review findings on head d073426 (4 semantic rows, 4 raw IDs)

| Row | Priority | Raw comment IDs | Terminal disposition | Validation and result |
|---:|---|---|---|---|
| R3-1 | P2 | 3676882648 | Documented (no code change) | The classifier input is already canonical on both routes: `should_use_fts_prose_mode` sanitizes via the FTS form internally, so FTS and LIKE paths classify identically by construction; LIKE-route EXTRACTION deliberately stays on the LIKE-sanitized form so unindexable characters remain searchable (the round-1 emoji lesson). Invariant now stated in comments at both `_search_like` prose branches. |
| R3-2 | P3 | 3676882656 | Fixed now | Both `getattr(engine._config, "fts_prose_mode", False)` reads replaced with direct attribute access; a missing field now fails loudly instead of silently disabling prose routing. |
| R3-3 | P3 | 3676882661 | Fixed now | `build_fts5_match_query` restores the pre-refactor guard: an empty prose-term list falls back to the sanitized form, so an empty disjunction can never reach FTS5 MATCH. |
| R3-4 | P2 | 3676904019 | Fixed now | `preserve_unicode_symbols` is now gated on the prose CLASSIFICATION (not the flag alone) in both `MessageStore.search` and `SummaryDAG.search`; a compact classifier-negative symbol query keeps the historical FTS route flag-on. Regression test `test_search_flag_on_compact_symbol_query_keeps_flag_off_route` added. |

Round-3 validation: focused suite 89 passed
(`python3 -m pytest tests/test_lcm_core.py -q -k "prose or search or grep"`);
flag-off probe byte-identical (2,104 bytes, SHA-256 `d9cf3621ed51669eeaff13642b8805d393e45838e351fd1bf236defd0b9e3219`).

## Round 3.5 — two evaos P3s on head ac21666

| Row | Priority | Raw comment IDs | Terminal disposition | Validation and result |
|---:|---|---|---|---|
| R35-1 | P3 | 3676937640 | Declined behavior change; contract documented | A lone curly quote inside a prose-shaped query is overwhelmingly a typing accident; forcing conjunctive mode on it would degrade the common case. The balanced-only contract (both quote styles) is now stated at the classifier's precision-signal comment, as the finding offered. |
| R35-2 | P3 | 3676937641 | Fixed now | `_lcm_grep_full_text` passes `allow_operators=False` explicitly and documents the invariant that grep-level and store/dag-level promotion must agree. No behavior change (downstream kwargs already defaulted False). |

## Round 3.6 — codex P2 on head edbc3ff

| Row | Priority | Raw comment IDs | Terminal disposition | Validation and result |
|---:|---|---|---|---|
| R36-1 | P2 | 3676960744 | Fixed now | `SummaryDAG._search_like` no longer truncates after the first unordered batch when `source` is unset: it scans the bounded candidate set (`candidate_cap`) before applying `limit`. Regression test `test_search_summary_like_scans_past_first_batch_for_best_match` (20 decoys + target at row 21, limit=1). Disclosure: this also corrects the same first-batch truncation on the pre-existing flag-off LIKE fallback — a strict recall improvement bounded by `candidate_cap`; the pinned flag-off regression corpus is byte-identical (same SHA-256 `d9cf3621…e3219`). |

## Round 4 — five findings on head 00b5afd

| Row | Priority | Raw comment IDs | Terminal disposition | Validation and result |
|---:|---|---|---|---|
| R4-1 | P2 | 3677032230 | Fixed now | The term cap bounds ORDINARY terms; an unindexable Unicode symbol (the LIKE route's routing signal) now survives the cap. The round-2 cap test is amended to state this contract; new test `test_search_prose_term_cap_preserves_unicode_symbol_signal`. |
| R4-2 | P2 | 3677032233 | Fixed now | New `_PROSE_FRAGMENTS` set (possessive/contraction debris incl. the closed n't-stem set) filtered from disjunctions AND excluded from classification counts. `What didn't we deploy?` extracts `[deploy]`; test added. |
| R4-3 | P2 | 3677042003 | Refuted in part; disclosure stands | The claimed polarity is inverted: removing `not source` changed the source-UNSET branch (previously truncated at the first batch — the defect); the source-SET branch always paged and is unchanged. The valid kernel — the pinned corpus does not exercise deep paging — was already stated in R36-1: the sha is a no-regression check on the pinned corpus, not a proof of no-change on the fixed branch; the behavior delta was disclosed as a strict, cap-bounded recall improvement. |
| R4-4 | P3 | 3677042008 | Refuted | The expand path passes `sort=None`, and the store/dag backends' internal `resolve_prose_sort` promotes exactly this case — the "direct caller" scenario those blocks exist for (as the round-3 sort-helper disposition documented). Both entry points therefore rank identically; no divergence exists. |
| R4-5 | P3 | 3677042011 | Fixed now | The subject-fallback may return a subject-capable stopword (``will``) but never a pure framing token: new `_PROSE_NEVER_SUBJECT` set; when nothing but framing survives, extraction returns no terms and `build_fts5_match_query` falls back to the sanitized conjunctive form. Test `test_search_prose_all_framing_query_falls_back_to_conjunctive`. |

Round-4 validation: focused 93 passed + recall 90 passed; flag-off probe byte-identical
(2,104 bytes, SHA-256 `d9cf3621…e3219`).

## Round 5 — nine raw findings on head a7986ae

| Raw comment ID | Terminal disposition | Change and validation |
|---|---|---|
| `3677176251` | **Fixed now** | Trailing sentence-punctuation normalization is prose-extraction-only. Default extraction retains the merge-base literal `🚀?`; the flag-off byte-identity probe remains the acceptance gate. |
| `3677176246` | **Fixed now** | Total prose terms are capped at `_PROSE_TERM_LIMIT`, with up to `_PROSE_SYMBOL_SLOTS = 2` reserved for the first unindexable routing-signal symbols. |
| `3677148375` | **Fixed with `3677176246`** | The pathological 1,100-symbol query is bounded to the reserved symbol slots and completes without SQLite's expression-tree error. |
| `3677176241` | **Fixed now** | Raw unindexable Unicode symbols count as subject terms for prose classification, so `Find ©?` reaches LIKE and retrieves `copyright © archive`; compact symbol-free queries retain their prior classification. |
| `3677176257` | **Fixed now** | Message-store and summary-DAG prose LIKE relevance count each distinct term at most once, so distinct coverage outranks repetition; flag-off scoring is unchanged. |
| `3677148380` | **Fixed now** | Summary-DAG LIKE batch pagination now uses explicit `ORDER BY rowid`. |
| `3677148370` | **Fixed now** | Added the DAG end-to-end FTS-route analog proving flag-on prose disjunction recovers a node that implicit AND misses. |
| `3677176262` | **Declined in-PR; filed as fork issue** | fork issue (rank fusion) |
| `3677148385` | **Refuted** | Refuted — contradicts the round-3 R3-2 disposition the same reviewer requested; LCMConfig defines the field with a default (config.py:656); loud failure on a malformed config is the contract. |

Round-5 validation: focused core selector 100 passed; recall module 90 passed;
flag-off probe byte-identical (2,104 bytes, SHA-256
`d9cf3621ed51669eeaff13642b8805d393e45838e351fd1bf236defd0b9e3219`);
changed-file Ruff passed. No Git mutation was performed.
