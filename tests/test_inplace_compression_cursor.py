"""Same-binding compression callbacks must preserve the returned-list cursor."""

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


@pytest.fixture
def engine(tmp_path):
    instance = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "memory.db")))
    instance.on_session_start("example-session", platform="cli", conversation_id="example-conversation", context_length=200_000)
    try:
        yield instance
    finally:
        instance.shutdown()


def boundary(engine, **kwargs):
    engine.on_session_start("example-session", boundary_reason="compression", old_session_id="example-session",
                            platform="cli", conversation_id="example-conversation", **kwargs)


def test_inplace_boundary_preserves_cursor_and_only_stores_new_suffix(engine):
    original = [{"role": "user", "content": "original request"}, {"role": "assistant", "content": "original answer"}]
    engine._ingest_messages(original)
    engine._last_compacted_store_id = 17
    engine.compression_count = 3
    engine.last_prompt_tokens = 123
    engine._last_active_replay_source_identities = [("stale",)]
    engine._last_active_replay_messages = [{"role": "user", "content": "stale"}]
    active = original + [{"role": "user", "content": "synthetic continuity"}]
    boundary(engine, in_place=True, active_message_count=len(active))
    assert engine._ingest_cursor == len(active)
    assert engine._ingest_cursor_needs_reconcile is False
    assert engine._last_compacted_store_id == 17
    assert engine.compression_count == 3
    assert engine.last_prompt_tokens == 123
    assert engine._last_active_replay_source_identities == []
    assert engine._last_active_replay_messages == []
    engine._ingest_messages(active + [{"role": "user", "content": "new request"}])
    contents = [row["content"] for row in engine._store.get_session_messages("example-session")]
    assert contents == ["original request", "original answer", "new request"]


@pytest.mark.parametrize("count", [None, True, -1, "3"])
def test_absent_or_invalid_host_count_keeps_compressors_cursor(engine, count):
    engine._ingest_cursor = 2
    boundary(engine, active_message_count=count)
    assert engine._ingest_cursor == 2
    assert engine._ingest_cursor_needs_reconcile is False


def test_explicit_non_inplace_callback_keeps_normal_reconciliation(engine):
    boundary(engine, in_place=False, active_message_count=3)
    assert engine._last_ingest_reconciliation["reason"] != "same-session in-place compression boundary"


def test_different_conversation_keeps_normal_rebinding(engine):
    engine.on_session_start("example-session", boundary_reason="compression", old_session_id="example-session",
                            platform="cli", conversation_id="other-conversation", in_place=True, active_message_count=3)
    assert engine._last_ingest_reconciliation["reason"] != "same-session in-place compression boundary"
