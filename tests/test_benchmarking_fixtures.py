"""Tests for benchmark fixture loading and deterministic generation."""

import json

import pytest

from benchmarking.fixtures import (
    fixture_from_dict,
    load_fixture,
    make_summary_failure_fixture,
    make_synthetic_fixture,
    parse_synthetic_fixture_spec,
)
from benchmarking.policies import load_policy
from benchmarking.replay import run_replay
from benchmarking.types import SummaryFailureMode


def test_load_fixture_parses_canaries(tmp_path):
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps({
        "name": "long_history_canaries",
        "tags": ["long_history", "exact_recall"],
        "messages": [
            {"role": "system", "content": "You are a test agent."},
            {"role": "user", "content": "CANARY_ALPHA = VALUE_ALPHA"},
        ],
        "canaries": [
            {"id": "CANARY_ALPHA", "value": "VALUE_ALPHA", "expected_query": "CANARY_ALPHA"},
        ],
    }))

    fixture = load_fixture(path)

    assert fixture.name == "long_history_canaries"
    assert fixture.tags == ["long_history", "exact_recall"]
    assert fixture.canaries[0].value == "VALUE_ALPHA"


def test_load_fixture_rejects_missing_required_keys(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"name": "bad"}))

    with pytest.raises(ValueError, match="messages"):
        load_fixture(path)


def test_fixture_preserves_benchmark_profile_metadata():
    fixture = fixture_from_dict({
        "name": "summary_timeout_probe",
        "tags": ["synthetic", "summary_failure"],
        "messages": [
            {"role": "system", "content": "You are a test agent."},
            {"role": "user", "content": "CANARY_TIMEOUT = VALUE_TIMEOUT"},
        ],
        "canaries": [
            {"id": "CANARY_TIMEOUT", "value": "VALUE_TIMEOUT", "expected_query": "CANARY_TIMEOUT"},
        ],
        "benchmark_profile": {
            "summary_level": 3,
            "summary_failure_mode": "llm_timeout_then_truncate",
        },
    })

    assert fixture.benchmark_profile == {
        "summary_level": 3,
        "summary_failure_mode": "llm_timeout_then_truncate",
    }
    assert fixture.to_dict()["benchmark_profile"] == fixture.benchmark_profile


def test_fixture_expands_scrubbed_benchmark_repeat_markers():
    fixture = fixture_from_dict({
        "name": "repeat_shape",
        "messages": [
            {"role": "user", "content": "repeat me", "benchmark_repeat": 3},
        ],
    })

    assert len(fixture.messages) == 3
    assert all("benchmark_repeat" not in message for message in fixture.messages)
    assert fixture.messages[0]["content"].endswith("[scrubbed benchmark repeat 001/003]")
    assert fixture.messages[2]["content"].endswith("[scrubbed benchmark repeat 003/003]")


@pytest.mark.parametrize("repeat", [3, "3", 3.0])
def test_fixture_accepts_integral_benchmark_repeat_markers(repeat):
    fixture = fixture_from_dict({
        "name": "repeat_shape",
        "messages": [
            {"role": "user", "content": "repeat me", "benchmark_repeat": repeat},
        ],
    })

    assert len(fixture.messages) == 3


@pytest.mark.parametrize("repeat", [1.9, "1.9", True])
def test_fixture_rejects_non_integer_benchmark_repeat_markers(repeat):
    with pytest.raises(ValueError, match="benchmark_repeat must be an integer"):
        fixture_from_dict({
            "name": "fractional_repeat",
            "messages": [
                {"role": "user", "content": "repeat me", "benchmark_repeat": repeat},
            ],
        })


def test_fixture_rejects_unbounded_benchmark_repeat_marker():
    with pytest.raises(ValueError, match="benchmark_repeat exceeds maximum"):
        fixture_from_dict({
            "name": "too_big",
            "messages": [
                {"role": "user", "content": "repeat me", "benchmark_repeat": 121},
            ],
        })


def test_make_summary_failure_fixture_marks_profile_and_tags():
    fixture = make_summary_failure_fixture(
        name="timeout_probe",
        summary_level=3,
        summary_failure_mode="llm_timeout_then_truncate",
        message_pairs=4,
        canary_count=1,
        filler_words=6,
    )

    assert fixture.benchmark_profile == {
        "summary_level": 3,
        "summary_failure_mode": "llm_timeout_then_truncate",
    }
    assert "summary_failure" in fixture.tags
    assert "timeout_probe_filler_5" in fixture.messages[1]["content"]


def test_make_summary_failure_fixture_accepts_enum_failure_mode():
    fixture = make_summary_failure_fixture(
        name="timeout_probe",
        summary_level=3,
        summary_failure_mode=SummaryFailureMode.LLM_TIMEOUT_THEN_TRUNCATE,
        message_pairs=4,
        canary_count=1,
        filler_words=6,
    )

    assert fixture.benchmark_profile == {
        "summary_level": 3,
        "summary_failure_mode": "llm_timeout_then_truncate",
    }


def test_make_synthetic_fixture_is_deterministic():
    first = make_synthetic_fixture(
        name="deterministic",
        message_pairs=4,
        canary_count=2,
        filler_words=6,
    )
    second = make_synthetic_fixture(
        name="deterministic",
        message_pairs=4,
        canary_count=2,
        filler_words=6,
    )

    assert first == second
    assert len(first.messages) == 9  # system plus four user/assistant pairs
    assert [canary.id for canary in first.canaries] == [
        "CANARY_DETERMINISTIC_0000",
        "CANARY_DETERMINISTIC_0001",
    ]
    assert "CANARY_DETERMINISTIC_0000 = VALUE_DETERMINISTIC_0000" in first.messages[1]["content"]


def test_parse_synthetic_fixture_spec_builds_named_fixture():
    fixture = parse_synthetic_fixture_spec("codex_pressure_probe:5:2:7")

    assert fixture.name == "codex_pressure_probe"
    assert len(fixture.messages) == 11
    assert len(fixture.canaries) == 2
    assert fixture.tags == ["synthetic", "deterministic"]
    assert "codex_pressure_probe_filler_6" in fixture.messages[1]["content"]


def test_parse_synthetic_fixture_spec_rejects_invalid_specs():
    with pytest.raises(ValueError, match="name:pairs:canaries:filler_words"):
        parse_synthetic_fixture_spec("bad")
    with pytest.raises(ValueError, match="message_pairs must be positive"):
        parse_synthetic_fixture_spec("bad:0:1:5")
    with pytest.raises(ValueError, match="canary_count cannot be negative"):
        parse_synthetic_fixture_spec("bad:1:-1:5")
    with pytest.raises(ValueError, match="canary_count cannot exceed message_pairs"):
        parse_synthetic_fixture_spec("bad:1:2:5")
    with pytest.raises(ValueError, match="message_pairs exceeds maximum"):
        parse_synthetic_fixture_spec("bad:251:1:5")
    with pytest.raises(ValueError, match="filler_words exceeds maximum"):
        parse_synthetic_fixture_spec("bad:1:1:2001")


def test_committed_summary_failure_fixtures_load_profiles():
    timeout_fixture = load_fixture("benchmarks/fixtures/summary_timeout_probe.json")
    refusal_fixture = load_fixture("benchmarks/fixtures/summary_refusal_probe.json")

    assert timeout_fixture.benchmark_profile == {
        "summary_level": 3,
        "summary_failure_mode": "llm_timeout_then_truncate",
    }
    assert refusal_fixture.benchmark_profile == {
        "summary_level": 3,
        "summary_failure_mode": "llm_refusal_then_truncate",
    }
    assert "summary_failure" in timeout_fixture.tags
    assert timeout_fixture.canaries[0].expected_query == "CANARY_TIMEOUT"


def test_committed_long_history_fixture_loads():
    fixture = load_fixture("benchmarks/fixtures/long_history_canaries.json")

    assert fixture.name == "long_history_canaries"
    assert fixture.canaries
    assert fixture.messages[0]["role"] == "system"


def test_committed_scrubbed_operator_shape_fixtures_load_without_raw_repeat_markers():
    coding = load_fixture("benchmarks/fixtures/scrubbed_operator_coding_tool_heavy.json")
    chatter = load_fixture("benchmarks/fixtures/scrubbed_operator_chatter_repeated_compaction.json")

    assert {"real_shape", "scrubbed_operator", "pressure_replay"}.issubset(coding.tags)
    assert {"real_shape", "scrubbed_operator", "pressure_replay"}.issubset(chatter.tags)
    assert coding.benchmark_profile["raw_transcript_included"] is False
    assert chatter.benchmark_profile["raw_transcript_included"] is False
    assert len(coding.messages) > 100
    assert len(chatter.messages) > 100
    assert all("benchmark_repeat" not in message for message in [*coding.messages, *chatter.messages])
    assert coding.canaries[0].expected_query == "CANARY_OPERATOR_BUILD"
    assert chatter.canaries[0].expected_query == "CANARY_OPERATOR_CHATTER"


def test_committed_chatter_fixture_with_pressure_policy_compacts(tmp_path):
    fixture = load_fixture("benchmarks/fixtures/repeated_compaction_chatter.json")
    policy = load_policy("benchmarks/policies/pressure_smoke.yaml")

    metrics = run_replay(fixture, policy, output_dir=tmp_path)

    assert fixture.tags == ["compaction_chatter", "synthetic"]
    assert policy.name == "pressure_smoke"
    assert metrics.compaction_attempts == 1
    assert metrics.compression_count >= 1
    assert metrics.repeated_compaction_risk is True
