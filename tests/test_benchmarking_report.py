"""Tests for benchmark summary provenance and policy comparison."""

import json

from benchmarking.policies import builtin_policies
from benchmarking.report import build_community_export, compare_policies, summarize_metrics
from benchmarking.types import ReplayMetrics


def _metrics(
    *,
    policy_name: str,
    policy_version: str = "1",
    active_canaries_found: int = 1,
    retrieval_canaries_found: int = 1,
    total_canaries: int = 1,
    repeated_compaction_risk: bool = False,
    fresh_tail_pressure_ratio: float = 0.10,
    failures: list[str] | None = None,
) -> ReplayMetrics:
    return ReplayMetrics(
        policy_name=policy_name,
        policy_version=policy_version,
        fixture_name="repeated_compaction_chatter",
        fixture_tags=["compaction_chatter", "synthetic"],
        prompt_tokens_before=1_000,
        prompt_tokens_after=500,
        threshold_tokens=800,
        compression_count=1,
        compaction_attempts=1,
        post_compaction_headroom_tokens=300,
        post_compaction_headroom_ratio=0.375,
        fresh_tail_message_count=2,
        fresh_tail_tokens=80,
        fresh_tail_pressure_ratio=fresh_tail_pressure_ratio,
        estimated_next_turn_tokens=120,
        repeated_compaction_risk=repeated_compaction_risk,
        active_canaries_found=active_canaries_found,
        retrieval_canaries_found=retrieval_canaries_found,
        total_canaries=total_canaries,
        active_canary_recall=active_canaries_found / total_canaries,
        retrieval_canary_recall=retrieval_canaries_found / total_canaries,
        failures=failures or [],
    )


def test_compare_policies_ranks_stable_recall_above_chattery_policy():
    rows = [
        _metrics(policy_name="baseline", repeated_compaction_risk=True, fresh_tail_pressure_ratio=0.80),
        _metrics(policy_name="candidate", policy_version="2"),
    ]

    comparison = compare_policies(rows)

    assert [row["policy_name"] for row in comparison] == ["candidate", "baseline"]
    assert comparison[0]["policy_version"] == "2"
    assert comparison[0]["repeated_compaction_risk_count"] == 0
    assert comparison[1]["repeated_compaction_risk_count"] == 1
    assert comparison[1]["fresh_tail_pressure_events"] == 1
    assert comparison[0]["score"] > comparison[1]["score"]


def test_compare_policies_keeps_policy_versions_separate():
    rows = [
        _metrics(policy_name="candidate", policy_version="1", repeated_compaction_risk=True),
        _metrics(policy_name="candidate", policy_version="2"),
    ]

    comparison = compare_policies(rows)

    assert [(row["policy_name"], row["policy_version"]) for row in comparison] == [
        ("candidate", "2"),
        ("candidate", "1"),
    ]
    assert [row["runs"] for row in comparison] == [1, 1]


def test_summarize_metrics_includes_versioned_provenance_and_comparison():
    rows = [
        _metrics(policy_name="baseline", repeated_compaction_risk=True, fresh_tail_pressure_ratio=0.80),
        _metrics(policy_name="candidate", policy_version="2"),
    ]

    summary = summarize_metrics(rows)

    assert summary["benchmark_version"] == "2"
    assert summary["generated_at_utc"].endswith("Z")
    assert summary["policy_versions"] == {"baseline": "1", "candidate": "2"}
    assert summary["fixture_suite"] == [
        {"name": "repeated_compaction_chatter", "runs": 2, "tags": ["compaction_chatter", "synthetic"]}
    ]
    assert summary["metric_summary"]["repeated_compaction_risk_count"] == 1
    assert [row["policy_name"] for row in summary["policy_comparison"]] == ["candidate", "baseline"]


def test_build_community_export_is_scrubbed_and_includes_policy_settings():
    rows = [
        _metrics(policy_name="baseline_272k", repeated_compaction_risk=True, fresh_tail_pressure_ratio=0.80),
        _metrics(policy_name="codex_gpt_long_context"),
    ]
    rows[0].database_path = "/tmp/private/lcm.db"
    rows[0].hermes_home = "/home/w0lf/.hermes/profiles/turing"
    summary = summarize_metrics(rows)
    policies = builtin_policies()

    export = build_community_export(
        summary,
        policies=policies,
        provider="openai-codex",
        model="gpt-5.5",
    )
    serialized = json.dumps(export, sort_keys=True)

    assert export["schema_version"] == "1"
    assert export["benchmark_version"] == "2"
    assert export["provider"] == "openai-codex"
    assert export["model"] == "gpt-5.5"
    assert export["transcript_contents_included"] is False
    assert export["policy_settings"]["codex_gpt_long_context@1"]["fresh_tail_count"] == 24
    assert export["policy_settings"]["codex_gpt_long_context@1"]["leaf_chunk_tokens"] == 8000
    assert "notes" not in export["policy_settings"]["codex_gpt_long_context@1"]
    assert export["fixture_suite"] == summary["fixture_suite"]
    assert export["policy_comparison"] == summary["policy_comparison"]
    assert "database_path" not in serialized
    assert "hermes_home" not in serialized
    assert "/home/w0lf" not in serialized
    assert "messages" not in serialized
