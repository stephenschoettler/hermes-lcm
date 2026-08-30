# hermes-lcm deterministic benchmarks

This directory contains deterministic replay fixtures and policy files for benchmark-driven LCM preset work. For the retrieval-quality and judged-QA-accuracy benchmarks (LongMemEval harnesses, fairness rules, reproduction, results index), see [`METHODOLOGY.md`](METHODOLOGY.md).

The benchmark harness is offline by default:

- no live provider calls
- deterministic summarization stub
- no live Hermes config mutation
- writes isolated to the requested output directory

## Active tool-result stubbing benchmark

The focused active-replay benchmark builds ten deterministic synthetic tool
results near 30K tokens each, protects the newest two pairs as the fresh tail,
and compares provider-visible assembly with stubbing disabled and enabled. It
verifies that every emitted ref recovers byte-identical normalized content and
prints aggregate-only JSON:

```bash
python benchmarks/benchmark_active_tool_stubbing.py \
  --output /tmp/hermes-lcm-active-tool-stubbing.json
```

The result measures provider-visible prompt tokens and local assembly latency.
It is not a provider billing-cost measurement and contains no raw payload text.

## Threshold full-sweep benchmark

For the opt-in threshold full-sweep policy, compare one ordinary incremental
invocation with one sweep using an offline synthetic workload:

```bash
python scripts/benchmark_threshold_full_sweep.py
```

The JSON report compares prompt-prefix publication count, summary-call count,
compression ratio, latency, and retained/recoverable synthetic facts. It never
includes message contents, temporary paths, or session identifiers.

## Run the default replay suite

```bash
python scripts/lcm_benchmark.py \
  --fixture benchmarks/fixtures/long_history_canaries.json \
  --fixture benchmarks/fixtures/repeated_compaction_chatter.json \
  --fixture benchmarks/fixtures/summary_timeout_probe.json \
  --fixture benchmarks/fixtures/summary_refusal_probe.json \
  --fixture benchmarks/fixtures/scrubbed_operator_coding_tool_heavy.json \
  --fixture benchmarks/fixtures/scrubbed_operator_chatter_repeated_compaction.json \
  --output benchmarks/runs/local-smoke \
  --json
```

Use `--allow-external-output` when writing outside the repository:

```bash
python scripts/lcm_benchmark.py \
  --fixture benchmarks/fixtures/repeated_compaction_chatter.json \
  --output /tmp/hermes-lcm-benchmark \
  --allow-external-output \
  --json
```

When no `--policy` is supplied, the harness loads built-in policies:

- `baseline_272k`, current long-context baseline
- `codex_gpt_long_context`, initial GPT/Codex long-context benchmark candidate
- `codex_spark_context`, GPT-5.3 Codex Spark / 128k benchmark candidate
- `pressure_smoke`, a deliberately small benchmark-only policy that proves pressure/chatter metrics trigger compaction

The committed policy files in `benchmarks/policies/` are the canonical benchmark inputs. Compare the GPT/Codex candidate against baseline with committed fixtures:

```bash
python scripts/lcm_benchmark.py \
  --fixture benchmarks/fixtures/long_history_canaries.json \
  --fixture benchmarks/fixtures/repeated_compaction_chatter.json \
  --policy benchmarks/policies/baseline.yaml \
  --policy benchmarks/policies/codex_gpt_long_context.yaml \
  --output benchmarks/runs/codex-gpt-long-context \
  --json
```

For a large deterministic pressure probe without committing a huge transcript fixture, generate a synthetic fixture inline:

```bash
python scripts/lcm_benchmark.py \
  --synthetic-fixture codex_pressure_probe:42:4:1000 \
  --policy benchmarks/policies/baseline.yaml \
  --policy benchmarks/policies/codex_gpt_long_context.yaml \
  --output benchmarks/runs/codex-gpt-pressure \
  --json
```

The 128k Spark preset uses the same pressure-probe shape with a smaller fresh tail to preserve post-compaction headroom under the lower trigger:

```bash
python scripts/lcm_benchmark.py \
  --synthetic-fixture spark_pressure_probe:42:4:1000 \
  --policy benchmarks/policies/codex_spark_context.yaml \
  --output benchmarks/runs/codex-spark-pressure \
  --json
```

Synthetic fixture specs use `name:pairs:canaries:filler_words` and are deterministic. They are bounded to 250 message pairs and 2,000 filler words so typos do not create huge benchmark outputs. Benchmark output directories should be fresh or cleaned between runs because the harness refuses to reuse non-empty per-run directories.

The committed `summary_timeout_probe` and `summary_refusal_probe` fixtures are small pilot fixtures for summary-provider failure-mode accounting. Their `benchmark_profile` records `summary_level` and `summary_failure_mode` metadata so reports can group timeout/refusal fallback scenarios without embedding provider calls or secrets in fixture content.

The committed scrubbed operator-shape fixtures extend the suite beyond pure synthetic pressure probes without leaking local transcripts:

- `scrubbed_operator_coding_tool_heavy.json` models a long coding lane with repeated tool output, patch/test loops, and old canaries.
- `scrubbed_operator_chatter_repeated_compaction.json` models a repeated-chatter lane with a compaction-prone recent tail.

These fixtures use bounded `benchmark_repeat` markers to expand scrubbed shape messages at load time. The marker is removed before replay/storage, keeping the committed JSON small while preserving the pressure profile needed to compare baseline and candidate policies.

`codex_gpt_long_context` and `codex_spark_context` are benchmark candidates and now have inspectable dry-run preset surfaces. `pressure_smoke` is not a runtime preset recommendation. It is a control policy for validating benchmark signals.

## Output files

The harness writes:

- `metrics.jsonl`, one serialized replay result per fixture/policy pair
- `summary.json`, aggregate provenance, metric summary, and ranked policy comparison
- per-run `metrics.json` files under fixture/policy-version output directories, for example `fixture__policy__v1/metrics.json`

Summary metadata includes:

- `benchmark_version`
- `generated_at_utc`
- `fixture_suite`
- `policy_versions`
- `metric_summary` (including `summary_failure_modes` and `summary_level_runs` when summary-failure profiles are present)
- `policy_comparison`

The comparison score is intentionally conservative. It rewards canary recall and stability, then penalizes failures, repeated-compaction risk, and excessive fresh-tail pressure. Treat it as a harness signal, not as proof that a policy is ready to become `preset: auto`.

## Scrubbed community exports

Use `--export` to write a shareable benchmark result JSON without raw transcript contents or local state paths. The export path follows the same repo-containment policy as `--output`; pass `--allow-external-output` when writing either path outside the repository.

```bash
python scripts/lcm_benchmark.py \
  --synthetic-fixture codex_pressure_probe:42:4:1000 \
  --policy benchmarks/policies/baseline.yaml \
  --policy benchmarks/policies/codex_gpt_long_context.yaml \
  --output benchmarks/runs/codex-gpt-pressure \
  --export benchmarks/runs/codex-gpt-pressure-export.json \
  --provider openai-codex \
  --model gpt-5.5
```

Only the file written by `--export` is the scrubbed community artifact. If you also pass `--json`, stdout prints the full local benchmark summary, including per-run diagnostic paths, and should not be shared as the community export.

The export contract is intentionally aggregate-only:

- `schema_version`
- `benchmark_version`
- `generated_at_utc`
- `provider` and `model` labels supplied by the operator
- `transcript_contents_included: false`
- `fixture_suite`
- `fixtures`
- `policies`
- `policy_versions`
- `policy_settings`
- `metric_summary`
- `policy_comparison`

The export omits per-run `metrics` rows because they can include local `database_path` and `hermes_home` values. Raw transcript content is never included by default.

## Stress release checks

Use the deterministic stress check before release cuts or risky context-engine changes. It is offline by default, patches summarization in-process, writes all SQLite and payload artifacts under the requested output directory, and exits non-zero when any scenario records a failure.

```bash
python scripts/lcm_stress_check.py \
  --output /tmp/hermes-lcm-stress-$(date +%Y%m%d-%H%M%S) \
  --tier release \
  --json
```

For a quick local smoke pass:

```bash
python scripts/lcm_stress_check.py \
  --output /tmp/hermes-lcm-stress-smoke \
  --tier smoke \
  --json
```

For a longer manual lifecycle soak pass, use the `soak` tier. It is intentionally not a default CI gate:

```bash
python scripts/lcm_stress_check.py \
  --output /tmp/hermes-lcm-stress-soak-$(date +%Y%m%d-%H%M%S) \
  --tier soak \
  --scenario lifecycle_soak_and_profile_rebinds \
  --json
```

The stress runner currently covers:

- multi-cycle compaction with planted canary recall through `lcm_grep` and `lcm_expand`
- sensitive-pattern redaction plus large-output externalization boundary checks
- current/all/explicit session scope and `lcm_load_session` pagination
- punctuation/unicode/FTS-hostile query fuzzing with bounded fallback behavior
- concurrent reader/writer smoke while compaction is active
- lifecycle soak across `/new` rollover, restart/rebind, Hermes home profile rebinding, SQLite WAL growth checks, and externalized-payload accumulation

Generated artifacts:

- `results/stress-results.json`, full machine-readable case output
- `stress-summary.md`, concise operator summary
- `sandbox/`, isolated Hermes home, SQLite databases, and externalized payload files

Hard gates for release use: `failure_count == 0`, no live profile writes, no raw configured secrets in SQLite rows/file bytes or externalized payload files, all planted non-secret canaries retrievable according to their scope, `lcm_doctor` healthy after stress, and artifact hashes recorded in `stress-results.json`. The JSON records a canonical hash for `stress-results.json` with the self-referential `artifact_hashes` field excluded, plus direct hashes for non-self-referential artifacts such as `stress-summary.md`.

## Preset provenance and dry-run surface

The shipped preset catalog is inspectable from the `/lcm` command surface when slash commands are enabled:

```text
/lcm preset show codex_gpt_long_context
/lcm preset suggest
/lcm preset apply codex_gpt_long_context --dry-run
```

Current `codex_gpt_long_context` / `codex_spark_context` provenance from the fresh-main validation suite:

- policy file: `benchmarks/policies/codex_gpt_long_context.yaml`
- policy version: `1`
- benchmark version: `2`
- fixture suite: committed baseline/chatter/failure fixtures, two scrubbed operator-shape fixtures, plus `codex_pressure_probe:42:4:1000` and `spark_pressure_probe:42:4:1000`
- aggregate candidate score: `92.941` vs `82.941` for `baseline_272k`
- retrieval canary recall: `1.0`
- repeated-compaction risk: candidate `0`, baseline `4`
- Spark minimum post-compaction headroom: `26,432` tokens in the validation suite

The dry-run apply surface previews env-var changes only:

```text
LCM_CONTEXT_THRESHOLD=0.75
LCM_FRESH_TAIL_COUNT=24
LCM_LEAF_CHUNK_TOKENS=8000
```

Explicit parseable preset-managed operator config wins. If `LCM_FRESH_TAIL_COUNT` or another supported preset-managed `LCM_*` knob is already set to a value the runtime can parse, `/lcm preset suggest` and `/lcm preset apply ... --dry-run` report that value as kept rather than overwritten. Invalid env values are reported separately, and the preview shows the preset value that would replace them. Runtime `target_after_compaction` is still benchmark-only metadata because the engine does not yet expose that as a live config field.

## Metrics added for preset research

Each replay records:

- `post_compaction_headroom_tokens`
- `post_compaction_headroom_ratio`
- `fresh_tail_tokens`
- `fresh_tail_pressure_ratio`
- `estimated_next_turn_tokens`
- `repeated_compaction_risk`
- `active_canary_recall`
- `retrieval_canary_recall`

These are the first benchmark-quality signals for issue #189. Runtime `preset: auto`, live-provider tuning, and automatic config edits remain out of scope.

## Symptom-to-knob tuning guide

Use benchmark output and `lcm_status`, not guesswork:

| Symptom | First knob to inspect | Direction |
|---------|-----------------------|-----------|
| Compaction happens nearly every turn | `post_compaction_headroom_tokens`, `repeated_compaction_risk`, `LCM_CONTEXT_THRESHOLD` | Lower the trigger or target more headroom before considering runtime auto-preset behavior |
| Fresh tail dominates the active prompt | `fresh_tail_pressure_ratio`, `fresh_tail_tokens`, `LCM_FRESH_TAIL_COUNT` | Lower the protected tail for long-context GPT/Codex-style routes; keep it high only when recent tool turns must stay verbatim |
| Leaf passes are huge and slow | `LCM_LEAF_CHUNK_TOKENS`, `LCM_DYNAMIC_LEAF_CHUNK_ENABLED` | Reduce chunk size or enable dynamic chunking after confirming raw backlog is the pressure source |
| Old facts are not in the active prompt but are retrievable | `active_canary_recall`, `retrieval_canary_recall` | Do not overfit for active recall; train usage toward `lcm_grep`, `lcm_expand`, and `lcm_expand_query` |
| Old facts are not retrievable | `retrieval_canary_recall`, failures, fixture coverage | Treat as a correctness bug or fixture gap before changing preset thresholds |
| Large tool outputs dominate token pressure | externalization status, payload sizes | Enable large-output externalization before tuning compaction thresholds |

Hard gates for promoting a preset: no replay failures, no raw transcript leakage in exports, stable retrieval recall, explainable fixture/provenance metadata, and no conflict with explicit operator config.

## LongMemEval retrieval harness

`scripts/lcm_longmemeval.py` measures retrieval quality (recall@k / NDCG@10)
on **LongMemEval_S** (Wu et al., ICLR 2025) for the LCM retrieval arms —
`fts` (raw-message FTS5), `summary_vectors` (summary embeddings), `hybrid_rrf`
(reciprocal-rank fusion, k=60), `hybrid_rerank` (a reranker over the fused pool,
see below), `chunk_vectors` (raw-chunk KNN), `hybrid_rrf3` (FTS + summary +
chunk fusion), and `lcm_recall` (the **production tool users actually call** — see
below). There is **no LLM judge**: the dataset labels the evidence
session(s) per question (`answer_session_ids`), so recall is computable offline.
It ingests each question's history into a fresh temporary LCM store (reusing the
`store`/`dag`/`vector_store` APIs directly, no live Hermes host), builds one
deterministic summary per session, optionally backfills embeddings, then scores
each arm against the labeled evidence.

### Session-level and turn-level scoring

Every arm reports metrics at **two granularities**. Session-level recall/NDCG ask
"did the arm retrieve the evidence *session*?" Turn-level recall/NDCG ask "did it
retrieve the evidence *turns*?" — a hit is a retrieved item whose `(session,
turn-range)` intersects the labeled evidence turns (`has_answer` markers). Raw
FTS message hits and raw chunk hits localize to a single turn via `store_id →
(session, turn_index)`. A **summary** covers a whole session and cannot localize a
turn, so summary hits score at **session granularity** — a retrieved evidence-
session summary credits every evidence turn of that session at once. Arms whose
ranking contains summary items are flagged with an asterisk (`*`) in the markdown
table and `session_granularity: true` in the JSON, so the coarse turn-level number
is never mistaken for exact localization. The `t*` columns in the table are the
turn-level metrics.

### The rerank arm (`hybrid_rerank`)

By default the rerank arm is a **deterministic embedding-cosine reranker** over
the fused candidate pool (`rerank_mode: "placeholder-cosine"`) — clearly labeled
as a placeholder for a real cross-encoder, and safe offline. With `--provider
voyage --rerank`, the arm instead calls **`VoyageProvider.rerank`
(rerank-2.5-lite)** on the top `RERANK_CANDIDATE_WINDOW` (=20) fused sessions in a
single API call under an absolute `RERANK_TIMEOUT_S` (=10s) budget; the remaining
fused tail is appended unchanged. Any provider error falls back to the placeholder.
The mode actually used is recorded in the JSON (`rerank.mode`) and the markdown
header so no one mistakes a placeholder run for a real-reranker run.

### The production arm (`lcm_recall`)

The other arms measure retrieval *primitives* — the harness reimplements each
arm's ranking (its own FTS query builder, its own RRF fusion). `lcm_recall`
instead scores the **actual `tools.lcm_recall` tool** end-to-end: weighted RRF
over the FTS + summary + chunk arms (`retrieval_core.rrf_fuse` with the
`LCM_RECALL_ARM_WEIGHTS` down-weighting of the FTS arm), the scope/recency prior,
chunk-vs-FTS dedup by `store_id`, and `include`-filtering — the full path a caller
gets, none of which the per-arm numbers exercise. It is invoked per question
against the same per-question temp store via a `SimpleNamespace` engine (the proven
smoke-test stand-in), with the warmed harness embedder injected through the tool's
provider cache so no second model load or network call occurs.

Two honesty notes on how the production behavior shows up in these numbers:

- **Scope prior is neutral, recency prior is not.** The probe engine uses a
  *fresh* current-session id that is disjoint from the dataset's sessions (see
  `fresh_recall_session_id`), so the scope prior — which boosts hits from the
  *current conversation* — never fires on a dataset session. The **recency prior
  still applies** to every hit (newer hits are boosted by a half-life multiplier);
  that is the real production behavior and is deliberately left in rather than
  stubbed out, so the number reflects the tool as shipped.
- **`limit` is clamped to the production ceiling.** `lcm_recall` caps its response
  at `_LCM_RECALL_LIMIT_CAP` (=25) hits, so its session ranking is only as deep as
  the tool will ever surface; recall@10 is measured over the deduped sessions of
  those top hits. Its per-question latency is also the *real* tool cost (thread
  pool, read-only connection setup, provider resolution, KNN pooling), so it is
  much higher than the reimplemented arms' microbenchmark timings.

### Ingest batching (F7)

Two ingest optimizations (measured per-question in `ingest.per_question_ms`):
each question's session summaries are embedded in **one** batched `embed_documents`
call instead of one call per session (sub-batched at `EMBED_BATCH_SIZE`=64 only to
guard against a pathologically large haystack), and a single pre-migrated SQLite
**template DB is cloned per question** rather than re-running the schema bootstrap
500×. Pass `--no-db-template` to disable template reuse (for A/B measurement).

The summary-call collapse is the win that matters for **network/live providers**
(e.g. Voyage), where per-call round-trip latency dominates: tens of summary calls
per question become one, and a full-haystack run's embed round-trips drop by the
session count — comfortably past the ≥3× F7 target on the live path. Raw chunks
are deliberately **not** batched: for a **local** ONNX provider (fastembed),
`embed_documents` pads every text in a batch to the batch's longest sequence, so
batching hundreds of variable-length chunks is *slower* than embedding them one at
a time (measured ~0.4× on `bge-small`). Local ONNX ingest is compute-bound, so the
portable local win is the DB-template reuse, not embed batching.

The dataset is downloaded **once** by an explicit operator command, never during
a run. The canonical source is the Hugging Face dataset `xiaowu0162/longmemeval`,
with small file `longmemeval_s` (~278 MB, 500 questions) and medium file
`longmemeval_m`, both pinned to revision
`2ec2a557f339b6c0369619b1ed5793734cc87533`:

```bash
python scripts/lcm_longmemeval.py fetch --output /path/to/longmemeval-data
```

The medium tier uses the streaming prepared format so the multi-gigabyte corpus
is never fully materialized in memory. Fetch, prepare, then run it as three
explicit operator steps. The prepare step requires `ijson`; the command below
supplies it with `uv run --with ijson`, or you can first run
`python -m pip install ijson` and invoke the script with `python` instead:

```bash
python scripts/lcm_longmemeval.py fetch \
  --dataset-label m \
  --output /path/to/longmemeval-data
uv run --with ijson python scripts/lcm_longmemeval.py prepare \
  --dataset /path/to/longmemeval-data/longmemeval_m \
  --dataset-label m \
  --prepared-dir /path/to/longmemeval-prepared \
  --allow-external-output
python scripts/lcm_longmemeval.py run \
  --prepared-dir /path/to/longmemeval-prepared \
  --dataset-label m \
  --provider stub --limit 5 \
  --output benchmarks/runs/longmemeval-medium-stub
```

Deterministic plumbing proof (offline, `<60s`, scores are meaningless with the
hash-based stub embedder):

```bash
python scripts/lcm_longmemeval.py run \
  --dataset /path/to/longmemeval-data/longmemeval_s \
  --provider stub --limit 5 \
  --output benchmarks/runs/longmemeval-stub
```

CI-grade local run with the deterministic FastEmbed provider (the model is
downloaded once into the FastEmbed cache; the query path is local thereafter).
`fastembed` is an optional dependency — install it into a virtualenv, and point
its model cache at a roomy volume with `LCM_LONGMEMEVAL_FASTEMBED_CACHE`:

```bash
python -m venv .venv-fastembed
.venv-fastembed/bin/pip install fastembed
LCM_LONGMEMEVAL_FASTEMBED_CACHE=/path/to/fastembed-cache \
  .venv-fastembed/bin/python scripts/lcm_longmemeval.py run \
    --dataset /path/to/longmemeval-data/longmemeval_s \
    --provider fastembed --model BAAI/bge-small-en-v1.5 --limit 25 \
    --output benchmarks/runs/longmemeval-fastembed
```

`--provider voyage --model <voyage-model>` is allowed for an explicit
live-provider run (network + spend); add `--rerank` to exercise the real
`VoyageProvider.rerank` arm (see "The rerank arm" above). Use `--limit N` to bound
cost; the full 500-question run over all six arms with `bge-small` takes on the
order of minutes on a laptop.

Output is **aggregate-only**, matching the export hygiene of
`scripts/lcm_benchmark.py`: `longmemeval_metrics.json` (per-arm and per-category
recall@1/5/10, NDCG@10 at both session and turn granularity — the latter under a
`turn` block with a `session_granularity` flag — per-arm latency percentiles, plus
top-level `rerank` mode/window/budget and `ingest` timing/provenance) plus a
`longmemeval_metrics.md` table. It contains no transcript content, session ids,
or local paths. Abstention questions (`question_id` ending in `_abs`) have no
evidence session and are excluded from recall (`abstention_excluded` counts
them). Categories: single-session-user / -assistant / -preference,
multi-session, temporal (temporal-reasoning), and knowledge-update.

**Honest caveat — this is our configuration, not a universal verdict.** MemDelta
(arXiv:2606.29914) shows that memory-benchmark rankings **flip** with the choice
of embedding model and base model: an arm that wins under one embedder can lose
under another. So these numbers gate the LCM rerank and embed-policy defaults
*for the precise configuration recorded in the metrics JSON* (provider, model,
dataset revision) and must not be read as an absolute claim that one arm is
better than another. Re-run with your intended production embedder before
trusting the ordering, and always publish the exact configuration alongside the
scores.
