"""Tests for model-aware benchmark data structures and policies."""

import json

from benchmarking.policies import builtin_policies, load_policy
from benchmarking.types import Canary, LCMPolicy, ReplayFixture, ReplayMetrics, SummaryFailureMode


def test_policy_round_trips_through_json_mapping():
    policy = LCMPolicy(
        name="codex_gpt_272k",
        context_length=272_000,
        context_threshold=0.75,
        fresh_tail_count=24,
        leaf_chunk_tokens=8_000,
        target_after_compaction=0.55,
        notes="dry-run candidate",
    )

    restored = LCMPolicy.from_dict(json.loads(json.dumps(policy.to_dict())))

    assert restored == policy


def test_fixture_round_trips_nested_canaries():
    fixture = ReplayFixture(
        name="long_history_canaries",
        messages=[{"role": "user", "content": "CANARY_ALPHA = VALUE_ALPHA"}],
        canaries=[Canary(id="CANARY_ALPHA", value="VALUE_ALPHA", expected_query="CANARY_ALPHA")],
        tags=["exact_recall"],
    )

    restored = ReplayFixture.from_dict(json.loads(json.dumps(fixture.to_dict())))

    assert restored == fixture
    assert restored.canaries[0].expected_query == "CANARY_ALPHA"


def test_fixture_round_trips_benchmark_profile_metadata():
    fixture = ReplayFixture(
        name="summary_timeout_probe",
        messages=[{"role": "user", "content": "CANARY_TIMEOUT = VALUE_TIMEOUT"}],
        canaries=[Canary(id="CANARY_TIMEOUT", value="VALUE_TIMEOUT", expected_query="CANARY_TIMEOUT")],
        tags=["summary_failure"],
        benchmark_profile={
            "summary_level": 3,
            "summary_failure_mode": "llm_timeout_then_truncate",
        },
    )

    restored = ReplayFixture.from_dict(json.loads(json.dumps(fixture.to_dict())))

    assert restored == fixture
    assert restored.benchmark_profile["summary_level"] == 3


def test_metrics_round_trips_summary_failure_profile():
    metrics = ReplayMetrics(
        policy_name="baseline_272k",
        fixture_name="summary_timeout_probe",
        prompt_tokens_before=100,
        prompt_tokens_after=80,
        threshold_tokens=75,
        compression_count=1,
        compaction_attempts=1,
        post_compaction_headroom_tokens=-5,
        active_canaries_found=1,
        retrieval_canaries_found=2,
        total_canaries=2,
        summary_level=3,
        summary_failure_mode=SummaryFailureMode.LLM_TIMEOUT_THEN_TRUNCATE,
        failures=["TimeoutError: summary provider timed out"],
    )

    restored = ReplayMetrics.from_dict(json.loads(json.dumps(metrics.to_dict())))

    assert restored == metrics
    assert restored.summary_failure_mode == SummaryFailureMode.LLM_TIMEOUT_THEN_TRUNCATE


def test_metrics_round_trips_with_failure_list():
    metrics = ReplayMetrics(
        policy_name="baseline_272k",
        fixture_name="long_history_canaries",
        prompt_tokens_before=100,
        prompt_tokens_after=80,
        threshold_tokens=75,
        compression_count=1,
        compaction_attempts=1,
        post_compaction_headroom_tokens=-5,
        active_canaries_found=1,
        retrieval_canaries_found=2,
        total_canaries=2,
        failures=["headroom below zero"],
    )

    restored = ReplayMetrics.from_dict(json.loads(json.dumps(metrics.to_dict())))

    assert restored == metrics


def test_builtin_policies_include_baseline_codex_policy_and_pressure_smoke():
    policies = {policy.name: policy for policy in builtin_policies()}

    assert policies["baseline_272k"].context_length == 272_000
    assert policies["baseline_272k"].fresh_tail_count == 64
    assert policies["codex_gpt_long_context"].context_length == 272_000
    assert policies["codex_gpt_long_context"].fresh_tail_count == 24
    assert policies["codex_gpt_long_context"].leaf_chunk_tokens == 8_000
    assert policies["codex_gpt_long_context"].target_after_compaction == 0.55
    assert policies["codex_gpt_long_context"].policy_version == "1"
    assert policies["pressure_smoke"].context_length < 1_000
    assert policies["pressure_smoke"].policy_version == "1"


def test_committed_codex_gpt_long_context_policy_matches_builtin_policy():
    policy = load_policy("benchmarks/policies/codex_gpt_long_context.yaml")
    builtin = {item.name: item for item in builtin_policies()}["codex_gpt_long_context"]

    assert policy == builtin
    assert "benchmark candidate" in policy.notes


def test_committed_policy_files_match_builtin_policies():
    builtins = {item.name: item for item in builtin_policies()}
    policy_paths = [
        "benchmarks/policies/baseline.yaml",
        "benchmarks/policies/codex_gpt_long_context.yaml",
        "benchmarks/policies/pressure_smoke.yaml",
    ]

    loaded = [load_policy(path) for path in policy_paths]

    assert {policy.name for policy in loaded} == set(builtins)
    assert loaded == [builtins[policy.name] for policy in loaded]


def test_load_policy_accepts_json_and_minimal_yaml(tmp_path):
    json_path = tmp_path / "policy.json"
    json_path.write_text(json.dumps({
        "name": "json_policy",
        "context_length": 1000,
        "context_threshold": 0.7,
        "fresh_tail_count": 8,
        "leaf_chunk_tokens": 100,
    }))
    yaml_path = tmp_path / "policy.yaml"
    yaml_path.write_text("""
name: yaml_policy
context_length: 2000
context_threshold: 0.8
fresh_tail_count: 12
leaf_chunk_tokens: 250
dynamic_leaf_chunk_enabled: false
target_after_compaction: 0.55
""".strip())

    assert load_policy(json_path).name == "json_policy"
    yaml_policy = load_policy(yaml_path)
    assert yaml_policy.name == "yaml_policy"
    assert yaml_policy.dynamic_leaf_chunk_enabled is False
    assert yaml_policy.target_after_compaction == 0.55


def test_load_policy_resolves_repo_relative_paths_from_external_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    policy = load_policy("benchmarks/policies/baseline.yaml")

    assert policy.name == "baseline_272k"
