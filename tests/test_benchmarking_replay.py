"""Tests for deterministic LCM benchmark replay."""

import json
from pathlib import Path

import hermes_lcm.engine as lcm_engine

from benchmarking.fixtures import make_synthetic_fixture
from benchmarking.replay import run_replay
from benchmarking.types import Canary, LCMPolicy, ReplayFixture


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
    raw_metrics = json.loads((tmp_path / "raw_hit" / "metrics.json").read_text())

    assert metrics.retrieval_canaries_found == 1
    assert raw_metrics["retrieval_canaries_found"] == 1
    assert raw_metrics["database_path"] == metrics.database_path
