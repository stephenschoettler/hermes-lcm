# hermes-lcm deterministic benchmarks

This directory contains deterministic replay fixtures and policy files for benchmark-driven LCM preset work.

The benchmark harness is offline by default:

- no live provider calls
- deterministic summarization stub
- no live Hermes config mutation
- writes isolated to the requested output directory

## Run the default replay suite

```bash
python scripts/lcm_benchmark.py \
  --fixture benchmarks/fixtures/long_history_canaries.json \
  --fixture benchmarks/fixtures/repeated_compaction_chatter.json \
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

Synthetic fixture specs use `name:pairs:canaries:filler_words` and are deterministic. They are bounded to 250 message pairs and 2,000 filler words so typos do not create huge benchmark outputs. Benchmark output directories should be fresh or cleaned between runs because the harness refuses to reuse non-empty per-run directories.

`codex_gpt_long_context` is a benchmark candidate, not an automatically selected runtime preset yet. `pressure_smoke` is not a runtime preset recommendation. It is a control policy for validating benchmark signals.

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
- `metric_summary`
- `policy_comparison`

The comparison score is intentionally conservative. It rewards canary recall and stability, then penalizes failures, repeated-compaction risk, and excessive fresh-tail pressure. Treat it as a harness signal, not as proof that a policy is ready to become `preset: auto`.

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

These are the first benchmark-quality signals for issue #189. Runtime preset selection, `/lcm preset suggest`, `/lcm preset apply`, live-provider tuning, and automatic config edits remain out of scope.
