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
  --fixture benchmarks/fixtures/summary_timeout_probe.json \
  --fixture benchmarks/fixtures/summary_refusal_probe.json \
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

The committed `summary_timeout_probe` and `summary_refusal_probe` fixtures are small pilot fixtures for summary-provider failure-mode accounting. Their `benchmark_profile` records `summary_level` and `summary_failure_mode` metadata so reports can group timeout/refusal fallback scenarios without embedding provider calls or secrets in fixture content.

`codex_gpt_long_context` is a benchmark candidate and now has an inspectable dry-run preset surface. `pressure_smoke` is not a runtime preset recommendation. It is a control policy for validating benchmark signals.

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

The stress runner currently covers:

- multi-cycle compaction with planted canary recall through `lcm_grep` and `lcm_expand`
- sensitive-pattern redaction plus large-output externalization boundary checks
- current/all/explicit session scope and `lcm_load_session` pagination
- punctuation/unicode/FTS-hostile query fuzzing with bounded fallback behavior
- concurrent reader/writer smoke while compaction is active

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

Current `codex_gpt_long_context` provenance:

- policy file: `benchmarks/policies/codex_gpt_long_context.yaml`
- policy version: `1`
- benchmark version: `2`
- fixture suite: `codex_pressure_probe:42:4:1000`
- pressure score: `92.5` vs `72.5` for `baseline_272k`
- retrieval canary recall: `1.0`
- repeated-compaction risk: candidate `0`, baseline `1`

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
