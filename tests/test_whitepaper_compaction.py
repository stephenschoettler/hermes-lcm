"""Regression tests for the whitepaper's compaction invariants."""

from __future__ import annotations

import sqlite3
import time

import pytest

import hermes_lcm.escalation as escalation
from agent.context_engine import ContextEngine  # noqa: F401 - primes plugin host imports
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_messages_tokens, count_tokens


@pytest.fixture
def make_engine(tmp_path):
    engines: list[LCMEngine] = []

    def factory(**overrides) -> LCMEngine:
        config = LCMConfig(database_path=str(tmp_path / f"lcm-{len(engines)}.db"))
        for name, value in overrides.items():
            setattr(config, name, value)
        engine = LCMEngine(config=config)
        engine.on_session_start(
            f"whitepaper-session-{len(engines)}",
            context_length=220,
        )
        engines.append(engine)
        return engine

    yield factory

    for engine in engines:
        engine.shutdown()


def _cheap_summary(engine: LCMEngine) -> None:
    def summarize(chunk, **_kwargs):
        return chunk, count_messages_tokens(chunk), "Persisted compact summary.", 1, 0

    engine._summarize_leaf_chunk_with_rescue = summarize


def test_compression_sqlite_lock_preserves_active_context(monkeypatch, make_engine):
    engine = make_engine()
    messages = [{"role": "user", "content": "keep this exact turn"}]

    def locked(*_args, **_kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(engine, "_compress_impl", locked)
    monkeypatch.setattr(
        engine,
        "_ingest_messages",
        lambda value, **_kwargs: value,
    )

    assert engine.compress(messages, force=True) == messages
    assert engine._last_compression_status == "noop"
    assert "sqlite writer busy" in engine._last_compression_noop_reason


def test_per_turn_ingest_spools_when_another_connection_holds_writer(tmp_path):
    db_path = tmp_path / "durable-lock.db"
    engine = LCMEngine(
        config=LCMConfig(database_path=str(db_path)),
        hermes_home=str(tmp_path),
    )
    blocker = sqlite3.connect(db_path, isolation_level=None)
    try:
        engine.on_session_start("durable-session", platform="cli")
        blocker.execute("BEGIN IMMEDIATE")

        started_at = time.monotonic()
        engine.ingest([{"role": "user", "content": "must survive contention"}])
        elapsed = time.monotonic() - started_at

        assert engine._store.pending_ingest_count() == 1
        assert engine._consecutive_ingest_failures == 0
        assert elapsed < 2.0

        blocker.rollback()
        assert engine._store.drain_pending_ingest() == 1
        assert engine._store.pending_ingest_count() == 0
        assert engine._store.get_session_count("durable-session") == 1
    finally:
        blocker.rollback()
        blocker.close()
        engine.shutdown()


def test_native_compaction_boundary_archives_without_local_resummarization(make_engine):
    engine = make_engine(fresh_tail_count=1)
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "old context " * 80},
        {
            "role": "assistant",
            "content": "continued",
            "codex_compaction_items": [{
                "type": "compaction",
                "encrypted_content": "opaque-native-capsule",
                "_issuer_kind": "codex_backend",
            }],
        },
        {"role": "user", "content": "new turn"},
    ]
    engine.threshold_tokens = 10

    assert engine.should_compress_preflight(messages) is False
    assert engine.last_compression_noop_reason == "provider-native compaction boundary active"
    assert engine._store.get_session_count(engine._session_id) == len(messages)
    assert engine._dag.get_session_nodes(engine._session_id) == []


def test_default_context_length_is_a_hard_cap_inside_fresh_tail(make_engine):
    engine = make_engine(fresh_tail_count=32)
    _cheap_summary(engine)
    messages = [
        {"role": "user", "content": f"turn-{index} " + ("detail " * 45)}
        for index in range(4)
    ]
    assert len(messages) < engine._config.fresh_tail_count
    assert count_messages_tokens(messages) > engine.context_length
    assert engine._effective_assembly_token_cap() is None

    assert engine.should_compress_preflight(messages) is True
    compacted = engine.compress(messages, current_tokens=count_messages_tokens(messages))

    assert count_messages_tokens(compacted) < engine.context_length
    assert compacted[-1] == messages[-1]
    assert engine._dag.get_session_nodes(engine._session_id)


def test_hard_pressure_preserves_the_newest_tool_call_group(make_engine):
    engine = make_engine(fresh_tail_count=32)
    _cheap_summary(engine)
    messages = [
        {"role": "user", "content": "old context " * 55},
        {"role": "assistant", "content": "older answer " * 30},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-latest",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-latest", "content": "ok"},
    ]
    assert count_messages_tokens(messages) > engine.context_length

    compacted = engine.compress(messages, current_tokens=count_messages_tokens(messages))

    latest_assistant = next(
        message
        for message in compacted
        if message.get("tool_calls")
        and message["tool_calls"][0].get("id") == "call-latest"
    )
    assert latest_assistant["role"] == "assistant"
    assert any(message.get("tool_call_id") == "call-latest" for message in compacted)
    assert count_messages_tokens(compacted) < engine.context_length


def test_single_oversized_latest_message_is_losslessly_stored_but_fitted(make_engine):
    engine = make_engine(fresh_tail_count=32)
    message = {"role": "user", "content": "oversized-current-turn " * 500}
    messages = [message]
    assert count_messages_tokens(messages) > engine.context_length

    compacted = engine.compress(messages, current_tokens=count_messages_tokens(messages))

    assert count_messages_tokens(compacted) < engine.context_length
    assert compacted[0]["role"] == "user"
    assert "active-context truncation" in compacted[0]["content"]
    stored = engine._store.get_session_messages(engine._session_id)
    assert len(stored) == 1
    assert stored[0]["content"] == message["content"]


@pytest.mark.parametrize(
    "text",
    [
        "x",
        "short source " * 80,
    ],
)
def test_l3_is_strictly_smaller_even_below_configured_truncation_cap(monkeypatch, text):
    monkeypatch.setattr(escalation, "_invoke_summary_llm_chain", lambda *_a, **_k: None)
    source_tokens = count_tokens(text)
    assert 0 < source_tokens <= 512

    result, level = escalation.summarize_with_escalation(
        text,
        source_tokens=source_tokens,
        token_budget=256,
        l3_truncate_tokens=512,
    )

    assert level == 3
    assert count_tokens(result) < source_tokens


def test_provider_exception_skips_duplicate_l2_wait_and_uses_l3(monkeypatch):
    attempts = []

    def unavailable(*_args, **_kwargs):
        attempts.append(1)
        raise TimeoutError("auxiliary provider unavailable")

    monkeypatch.setattr(escalation, "_invoke_summary_llm", unavailable)
    text = "provider outage must converge without another network wait " * 200
    source_tokens = count_tokens(text)

    result, level = escalation.summarize_with_escalation(
        text,
        source_tokens=source_tokens,
        token_budget=256,
    )

    assert level == 3
    assert count_tokens(result) < source_tokens
    assert len(attempts) == 1


def test_leaf_publication_fails_closed_when_raw_lineage_is_missing(make_engine):
    engine = make_engine(fresh_tail_count=1, leaf_chunk_tokens=1)
    _cheap_summary(engine)
    messages = [
        {"role": "user", "content": "first persisted turn " * 20},
        {"role": "assistant", "content": "first persisted reply " * 20},
        {"role": "user", "content": "newest protected turn"},
    ]
    engine._get_store_id_map_for_messages = lambda _messages: {}

    with pytest.raises(RuntimeError, match="raw store lineage"):
        engine.compress(messages, current_tokens=count_messages_tokens(messages))

    assert engine._dag.get_session_nodes(engine._session_id) == []
    assert engine._last_compacted_store_id == 0
    assert messages[-1]["content"] == "newest protected turn"


def test_filtered_only_marker_advances_without_false_summary_lineage(make_engine):
    engine = make_engine(fresh_tail_count=1, leaf_chunk_tokens=1)
    messages = [
        {"role": "assistant", "content": "derived ignored reply"},
        {"role": "user", "content": "newest protected turn"},
    ]
    engine._is_generated_ignored_dependent_reply = lambda _message, text: (
        text == "derived ignored reply"
    )
    engine._remember_generated_ignored_dependent_reply = lambda *_a, **_k: None

    engine.compress(messages, current_tokens=count_messages_tokens(messages))

    assert engine._dag.get_session_nodes(engine._session_id) == []
    assert engine._store.get_session_messages(engine._session_id)[0]["content"] == (
        "derived ignored reply"
    )
    state = engine._lifecycle.get_by_conversation(engine._conversation_id)
    assert state.current_frontier_store_id > 0


def test_file_ids_propagate_from_raw_messages_through_condensation(make_engine, monkeypatch):
    engine = make_engine(fresh_tail_count=1, leaf_chunk_tokens=1)
    _cheap_summary(engine)
    file_id = "file_0123456789abcdef"
    messages = [
        {"role": "user", "content": '{"output_file_id":"%s"}' % file_id},
        {"role": "user", "content": "newest protected turn"},
    ]

    engine.compress(messages, current_tokens=count_messages_tokens(messages))
    leaf = engine._dag.get_session_nodes(engine._session_id)[0]
    assert leaf.file_ids == [file_id]

    monkeypatch.setattr(
        "hermes_lcm.engine.summarize_with_escalation",
        lambda **_kwargs: ("condensed summary", 1),
    )
    engine._condense_summary_nodes([leaf])
    condensed = engine._dag.get_session_nodes(engine._session_id, depth=1)[0]
    assert condensed.file_ids == [file_id]
