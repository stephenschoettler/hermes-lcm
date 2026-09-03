"""Regression tests for cleanup-only externalization below the LCM threshold."""

import time
from unittest.mock import Mock

import pytest

import hermes_lcm.engine as lcm_engine
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


@pytest.fixture
def make_engine(tmp_path):
    engines = []

    def build(**overrides):
        index = len(engines)
        settings = {
            "database_path": str(tmp_path / f"cleanup-{index}.db"),
            "fresh_tail_count": 2,
            "leaf_chunk_tokens": 1,
            "dynamic_leaf_chunk_enabled": False,
            "threshold_full_sweep_enabled": False,
            "large_output_externalization_enabled": True,
            "large_output_externalization_threshold_chars": 1_000,
            "large_output_externalization_path": str(
                tmp_path / f"externalized-{index}"
            ),
        }
        settings.update(overrides)
        engine = LCMEngine(
            config=LCMConfig(**settings),
            hermes_home=str(tmp_path / f"hermes-{index}"),
        )
        engine.on_session_start(
            f"cleanup-session-{index}",
            conversation_id=f"cleanup-conversation-{index}",
            context_length=1_000_000,
        )
        engines.append(engine)
        return engine

    yield build

    for engine in engines:
        engine.shutdown()


RAW_PAYLOAD = "INJECTED_RAW_PAYLOAD:" + ("x" * 5_000)


def current_raw_payload_messages():
    return [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "old request " * 20},
        {"role": "assistant", "content": "old answer " * 20},
        {"role": "user", "content": RAW_PAYLOAD},
    ]


def tool_pair(call_id, payload):
    return [
        {
            "role": "assistant",
            "content": "running tool",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "content": payload},
    ]


def tool_cleanup_case(make_engine):
    engine = make_engine(
        large_output_externalization_threshold_chars=1_000_000,
        large_output_active_replay_stubbing_enabled=True,
        large_output_active_replay_stub_threshold_tokens=5,
    )
    engine.threshold_tokens = 500_000
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "old request " * 20},
        {"role": "assistant", "content": "old answer " * 20},
        *tool_pair("current-call", "current tool payload " * 100),
    ]
    return engine, messages


def test_current_oversized_user_stays_provider_visible_without_preflight_cleanup(
    make_engine,
    monkeypatch,
):
    engine = make_engine()
    engine.threshold_tokens = 500_000
    messages = current_raw_payload_messages()
    summary_spy = Mock(
        side_effect=AssertionError("current payload must not trigger summarization")
    )
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summary_spy)

    assert engine.should_compress_preflight(messages) is False
    stored = engine._store.get_session_messages(engine._session_id)
    assert stored[-1]["content"].startswith(
        "[Externalized payload: kind=raw_payload;"
    )
    provider_messages = engine._cached_active_replay_messages(messages)

    assert provider_messages is not None
    assert provider_messages[-1]["content"] == RAW_PAYLOAD
    assert engine._dag.get_session_node_count(engine._session_id) == 0
    assert engine.compression_count == 0
    summary_spy.assert_not_called()


def test_subthreshold_tool_result_cleanup_does_not_create_leaf(
    make_engine,
    monkeypatch,
):
    engine, messages = tool_cleanup_case(make_engine)
    summary_spy = Mock(
        side_effect=AssertionError("sub-threshold cleanup must not summarize")
    )
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summary_spy)

    assert engine.should_compress_preflight(messages) is True
    result = engine.compress(
        messages,
        current_tokens=engine.threshold_tokens // 4,
    )

    assert any(
        str(message.get("content", "")).startswith(
            "[Externalized tool output:"
        )
        for message in result
    )
    assert any(message.get("content") == "old request " * 20 for message in result)
    assert engine._dag.get_session_node_count(engine._session_id) == 0
    assert engine.compression_count == 0
    assert engine.last_compression_status == "sanitized"
    summary_spy.assert_not_called()


def test_direct_subthreshold_tool_cleanup_does_not_create_leaf(
    make_engine,
    monkeypatch,
):
    """compress() must enforce cleanup-only even without a preflight handoff."""
    engine, messages = tool_cleanup_case(make_engine)
    summary_spy = Mock(
        side_effect=AssertionError("direct sub-threshold cleanup must not summarize")
    )
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summary_spy)

    result = engine.compress(
        messages,
        current_tokens=engine.threshold_tokens // 4,
    )

    assert any(
        str(message.get("content", "")).startswith(
            "[Externalized tool output:"
        )
        for message in result
    )
    assert any(message.get("content") == "old request " * 20 for message in result)
    assert engine._dag.get_session_node_count(engine._session_id) == 0
    assert engine.compression_count == 0
    assert engine.last_compression_status == "sanitized"
    summary_spy.assert_not_called()


def test_above_threshold_tool_cleanup_still_compacts(make_engine, monkeypatch):
    engine, messages = tool_cleanup_case(make_engine)
    # Preflight only sees the sub-threshold message estimate.  The host-provided
    # count at compress time is authoritative and must still permit compaction.
    summary_spy = Mock(
        return_value=(
            "old work summary\nExpand for details about: old work",
            1,
        )
    )
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summary_spy)

    assert engine.should_compress_preflight(messages) is True
    result = engine.compress(messages, current_tokens=engine.threshold_tokens)

    assert result
    assert engine._dag.get_session_node_count(engine._session_id) == 1
    assert engine.compression_count == 1
    assert engine.last_compression_status == "compacted"
    assert summary_spy.call_count >= 1


def test_manual_force_tool_cleanup_still_compacts(make_engine, monkeypatch):
    engine, messages = tool_cleanup_case(make_engine)
    summary_spy = Mock(
        return_value=(
            "manually forced summary\nExpand for details about: old work",
            1,
        )
    )
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summary_spy)

    assert engine.should_compress_preflight(messages) is True
    result = engine.compress(
        messages,
        current_tokens=engine.threshold_tokens // 4,
        force=True,
    )

    assert result
    assert engine._dag.get_session_node_count(engine._session_id) == 1
    assert engine.compression_count == 1
    assert engine.last_compression_status == "compacted"
    assert summary_spy.call_count >= 1


def test_boundary_cooldown_cleanup_outranks_threshold(make_engine, monkeypatch):
    engine, messages = tool_cleanup_case(make_engine)
    engine.threshold_tokens = 100
    engine._last_boundary_skip_time = time.time()
    summary_spy = Mock(
        side_effect=AssertionError("boundary cooldown cleanup must not summarize")
    )
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summary_spy)

    assert engine.should_compress_preflight(messages) is True
    result = engine.compress(messages, current_tokens=engine.threshold_tokens)

    assert result
    assert engine._dag.get_session_node_count(engine._session_id) == 0
    assert engine.compression_count == 0
    assert engine.last_compression_status == "sanitized"
    summary_spy.assert_not_called()


def test_overflow_tool_cleanup_still_runs_recovery(make_engine, monkeypatch):
    engine, messages = tool_cleanup_case(make_engine)
    engine._config.max_assembly_tokens = 100
    summary_spy = Mock(
        return_value=(
            "overflow recovery summary\nExpand for details about: old work",
            1,
        )
    )
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summary_spy)

    assert engine._should_force_overflow_recovery(messages=messages) is True
    assert engine.should_compress_preflight(messages) is True
    result = engine.compress(messages, current_tokens=10_000)

    assert result
    assert engine._dag.get_session_node_count(engine._session_id) >= 1
    assert engine.last_compression_status in {"compacted", "overflow_recovery"}
    assert summary_spy.call_count >= 1
