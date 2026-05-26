"""Tests for deterministic LCM benchmark replay."""

import json
from pathlib import Path

import pytest

import hermes_lcm.engine as lcm_engine

from benchmarking.fixtures import make_synthetic_fixture
from benchmarking.replay import run_replay, run_replays
from benchmarking.types import Canary, LCMPolicy, ReplayFixture, SummaryFailureMode


def _small_policy(**overrides):
    values = {
        "name": "small_policy",
        "context_length": 400,
        "context_threshold": 0.20,
        "fresh_tail_count": 1,
        "leaf_chunk_tokens": 20,
        "condensation_fanin": 4,
        "incremental_max_depth": 1,
        "dynamic_leaf_chunk_enabled": False,
    }
    values.update(overrides)
    return LCMPolicy(**values)


def test_replay_below_threshold_does_not_compress(tmp_path):
    fixture = ReplayFixture(
        name="tiny_fixture",
        messages=[
            {"role": "system", "content": "You are a test agent."},
            {"role": "user", "content": "small hello"},
        ],
    )
    policy = _small_policy(context_length=10_000, context_threshold=0.90)

    metrics = run_replay(fixture, policy, output_dir=tmp_path)

    assert metrics.compaction_attempts == 0
    assert metrics.compression_count == 0
    assert metrics.prompt_tokens_before == metrics.prompt_tokens_after
    assert Path(metrics.database_path).is_relative_to(tmp_path)


def test_replay_defaults_partial_summary_profile_failure_mode_to_none(tmp_path):
    fixture = ReplayFixture(
        name="partial_summary_profile",
        messages=[
            {"role": "system", "content": "You are a test agent."},
            {"role": "user", "content": "small hello"},
        ],
        benchmark_profile={"summary_level": 2},
    )
    policy = _small_policy(context_length=10_000, context_threshold=0.90)

    metrics = run_replay(fixture, policy, output_dir=tmp_path)

    assert metrics.summary_level == 2
    assert metrics.summary_failure_mode is SummaryFailureMode.NONE


def test_replay_above_threshold_compresses_and_reports_canary_recall(tmp_path):
    fixture = make_synthetic_fixture(
        name="pressure",
        message_pairs=8,
        canary_count=2,
        filler_words=80,
    )
    policy = _small_policy()

    metrics = run_replay(fixture, policy, output_dir=tmp_path)

    assert metrics.compaction_attempts == 1
    assert metrics.compression_count >= 1
    assert metrics.prompt_tokens_before > metrics.prompt_tokens_after
    assert metrics.active_canaries_found >= 1
    assert metrics.retrieval_canaries_found == metrics.total_canaries == 2
    assert metrics.failures == []


def test_replay_restores_summarizer_patch(tmp_path):
    original = lcm_engine.summarize_with_escalation
    fixture = make_synthetic_fixture(
        name="restore",
        message_pairs=6,
        canary_count=1,
        filler_words=80,
    )

    run_replay(fixture, _small_policy(), output_dir=tmp_path)

    assert lcm_engine.summarize_with_escalation is original


def test_replay_uses_output_directory_for_state_and_not_home(tmp_path, monkeypatch):
    fake_home = tmp_path / "fake-home"
    output_dir = tmp_path / "benchmark-output"
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("HERMES_HOME", str(fake_home / ".hermes"))
    fixture = make_synthetic_fixture(
        name="sandbox",
        message_pairs=6,
        canary_count=1,
        filler_words=80,
    )

    metrics = run_replay(fixture, _small_policy(), output_dir=output_dir)

    assert Path(metrics.database_path).is_relative_to(output_dir)
    assert Path(metrics.hermes_home).is_relative_to(output_dir)
    assert not (fake_home / ".hermes" / "lcm.db").exists()


def test_replay_refuses_to_reuse_existing_non_empty_run_directory(tmp_path):
    fixture = ReplayFixture(
        name="stale_fixture",
        messages=[{"role": "user", "content": "small hello"}],
    )
    policy = _small_policy(context_length=10_000, context_threshold=0.90)
    run_dir = tmp_path / "stale_fixture__small_policy__v1"
    run_dir.mkdir()
    (run_dir / "stale.txt").write_text("old benchmark output")

    with pytest.raises(ValueError, match="Refusing to reuse non-empty benchmark run directory"):
        run_replay(fixture, policy, output_dir=tmp_path)


def test_replay_refuses_run_directory_path_that_is_file(tmp_path):
    fixture = ReplayFixture(
        name="file_fixture",
        messages=[{"role": "user", "content": "small hello"}],
    )
    policy = _small_policy(context_length=10_000, context_threshold=0.90)
    run_dir = tmp_path / "file_fixture__small_policy__v1"
    run_dir.write_text("not a directory")

    with pytest.raises(ValueError, match="Refusing to reuse benchmark run path"):
        run_replay(fixture, policy, output_dir=tmp_path)


def test_run_replays_isolates_same_name_policy_versions(tmp_path):
    fixture = ReplayFixture(
        name="versioned_fixture",
        messages=[
            {"role": "system", "content": "You are a test agent."},
            {"role": "user", "content": "small hello"},
        ],
    )
    policies = [
        _small_policy(name="candidate", policy_version="1", context_length=10_000, context_threshold=0.90),
        _small_policy(name="candidate", policy_version="2", context_length=10_000, context_threshold=0.90),
    ]

    metrics = run_replays([fixture], policies, output_dir=tmp_path)

    assert [row.policy_version for row in metrics] == ["1", "2"]
    assert len({row.database_path for row in metrics}) == 2
    assert (tmp_path / "versioned_fixture__candidate__v1" / "metrics.json").exists()
    assert (tmp_path / "versioned_fixture__candidate__v2" / "metrics.json").exists()


def test_replay_retrieval_expands_raw_hits(tmp_path):
    fixture = ReplayFixture(
        name="raw_hit",
        messages=[
            {"role": "system", "content": "You are a test agent."},
            {"role": "user", "content": "CANARY_RAW = VALUE_RAW " + ("filler " * 120)},
            {"role": "assistant", "content": "Acknowledged."},
        ],
        canaries=[Canary(id="CANARY_RAW", value="VALUE_RAW", expected_query="CANARY_RAW")],
    )

    metrics = run_replay(fixture, _small_policy(), output_dir=tmp_path)
    raw_metrics = json.loads((tmp_path / "raw_hit__small_policy__v1" / "metrics.json").read_text())

    assert metrics.retrieval_canaries_found == 1
    assert raw_metrics["retrieval_canaries_found"] == 1
    assert raw_metrics["database_path"] == metrics.database_path


def test_replay_reports_headroom_fresh_tail_and_chatter_risk_metrics(tmp_path):
    fixture = make_synthetic_fixture(
        name="pressure_metrics",
        message_pairs=8,
        canary_count=2,
        filler_words=80,
    )

    metrics = run_replay(fixture, _small_policy(), output_dir=tmp_path)
    raw_metrics = json.loads((tmp_path / "pressure_metrics__small_policy__v1" / "metrics.json").read_text())

    assert metrics.fresh_tail_message_count == 1
    assert metrics.fresh_tail_tokens > 0
    assert metrics.fresh_tail_pressure_ratio > 0
    assert metrics.estimated_next_turn_tokens > 0
    assert metrics.post_compaction_headroom_ratio == metrics.post_compaction_headroom_tokens / metrics.threshold_tokens
    assert metrics.active_canary_recall == metrics.active_canaries_found / metrics.total_canaries
    assert metrics.retrieval_canary_recall == 1.0
    assert metrics.repeated_compaction_risk is True
    assert raw_metrics["repeated_compaction_risk"] is True
