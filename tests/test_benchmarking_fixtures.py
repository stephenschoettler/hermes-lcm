"""Tests for benchmark fixture loading and deterministic generation."""

import json

import pytest

from benchmarking.fixtures import load_fixture, make_synthetic_fixture


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


def test_committed_long_history_fixture_loads():
    fixture = load_fixture("benchmarks/fixtures/long_history_canaries.json")

    assert fixture.name == "long_history_canaries"
    assert fixture.canaries
    assert fixture.messages[0]["role"] == "system"
