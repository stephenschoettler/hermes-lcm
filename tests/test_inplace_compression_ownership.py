"""Same-session compression callbacks retain their exact durable owner."""

import copy
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.lifecycle_state import LifecycleStateStore


@pytest.fixture
def engine(tmp_path):
    config = LCMConfig(
        database_path=str(tmp_path / "ownership.db"), fresh_tail_count=2,
        leaf_chunk_tokens=1, dynamic_leaf_chunk_enabled=False,
        threshold_full_sweep_enabled=False,
    )
    instance = LCMEngine(config, hermes_home=str(tmp_path / "profile"))
    instance.on_session_start("continuing-session", conversation_id="conversation", platform="cli")
    try:
        yield instance
    finally:
        instance.shutdown()


def boundary(engine):
    engine.on_session_start(
        "continuing-session", boundary_reason="compression", old_session_id="continuing-session",
        conversation_id="conversation", platform="cli", in_place=True,
    )


@pytest.mark.parametrize("host_outcome", ["old_host_committed", "committed", "refused", "archive_failed"])
def test_compression_boundary_preserves_owner_and_next_leaf(engine, monkeypatch, host_outcome):
    messages = [
        {"role": "user", "content": "The repeatable request. " * 100},
        {"role": "assistant", "content": "The original answer. " * 100},
        {"role": "user", "content": "The recent question."},
        {"role": "assistant", "content": "The recent answer."},
    ]
    original = copy.deepcopy(messages)
    engine.ingest(messages)
    rows = engine._store.get_session_messages(engine._session_id)
    engine._last_compacted_store_id = rows[0]["store_id"]
    engine._lifecycle.advance_frontier("conversation", engine._session_id, engine._last_compacted_store_id)
    cursor = engine._ingest_cursor
    engine.compression_count = 2
    if host_outcome == "old_host_committed":
        # Older Core sends a false end before the successful same-ID start.
        engine.on_session_end(engine._session_id, messages)
        assert engine._lifecycle.get_by_conversation("conversation").current_session_id is None
        boundary(engine)
    elif host_outcome == "committed":
        boundary(engine)
    # Corrected hosts do not emit a successful boundary after refusal/archive failure.
    state = engine._lifecycle.get_by_conversation("conversation")
    assert engine._ingest_cursor == cursor
    assert engine.compression_count == 2
    replay = messages + [{"role": "user", "content": messages[0]["content"]}]
    engine.ingest(replay)
    assert len(engine._store.get_session_messages(engine._session_id)) == len(rows) + 1
    assert engine._store.get_session_messages(engine._session_id)[:len(rows)] == rows
    monkeypatch.setattr(engine, "_summarize_leaf_chunk_with_rescue", lambda chunk, **kwargs: (chunk, 1000, "faithful next leaf", 1, 0))
    compressed = engine.compress(replay, current_tokens=100000)
    assert compressed != replay
    assert len(engine._dag.get_session_nodes(engine._session_id)) == 1
    after = engine._lifecycle.get_by_conversation("conversation")
    assert state.current_session_id == after.current_session_id == engine._session_id
    assert state.current_frontier_store_id == rows[0]["store_id"]
    assert after.current_frontier_store_id == engine._last_compacted_store_id
    assert messages == original


@pytest.mark.parametrize("ownership", ["missing", "replacement", "wrong_finalized"])
def test_stale_same_session_callback_cannot_claim_another_owner(engine, ownership):
    engine.ingest([{"role": "user", "content": "original request"}])
    engine._ingest_cursor = 1
    with sqlite3.connect(engine._store.db_path) as writer:
        if ownership == "missing":
            writer.execute("DELETE FROM lcm_lifecycle_state WHERE conversation_id='conversation'")
        elif ownership == "replacement":
            writer.execute("UPDATE lcm_lifecycle_state SET current_session_id='replacement-session' WHERE conversation_id='conversation'")
        else:
            writer.execute("UPDATE lcm_lifecycle_state SET current_session_id=NULL, last_finalized_session_id='other-session' WHERE conversation_id='conversation'")
    before = engine._lifecycle.get_by_conversation("conversation")
    with pytest.raises(RuntimeError, match="ownership"):
        boundary(engine)
    assert engine._lifecycle.get_by_conversation("conversation") == before
    assert engine._ingest_cursor == 1


def test_finalized_same_session_restores_exact_runtime_frontier_and_preserves_markers(engine):
    engine.ingest([{"role": "user", "content": "original request"}])
    engine._last_compacted_store_id = 7
    engine.on_session_end(engine._session_id, [])
    with sqlite3.connect(engine._store.db_path) as writer:
        writer.execute(
            "UPDATE lcm_lifecycle_state SET last_finalized_frontier_store_id=99, "
            "debt_kind='raw', debt_size_estimate=123 WHERE conversation_id='conversation'"
        )
    before = engine._lifecycle.get_by_conversation("conversation")
    boundary(engine)
    after = engine._lifecycle.get_by_conversation("conversation")
    assert after.current_session_id == engine._session_id
    assert after.current_frontier_store_id == 7
    for name in ["last_finalized_session_id", "last_finalized_frontier_store_id", "last_finalized_at", "debt_kind", "debt_size_estimate", "debt_updated_at"]:
        assert getattr(after, name) == getattr(before, name)


def test_resume_update_failure_rolls_back_ownership_and_keeps_cursor(engine):
    engine.ingest([{"role": "user", "content": "original request"}])
    engine.on_session_end(engine._session_id, [])
    before = engine._lifecycle.get_by_conversation("conversation")
    with sqlite3.connect(engine._store.db_path) as writer:
        writer.execute(
            "CREATE TRIGGER reject_resume BEFORE UPDATE ON lcm_lifecycle_state "
            "BEGIN SELECT RAISE(ABORT, 'injected resume failure'); END"
        )
    with pytest.raises(sqlite3.IntegrityError, match="injected resume failure"):
        boundary(engine)
    reader = LifecycleStateStore(engine._store.db_path)
    try:
        assert reader.get_by_conversation("conversation") == before
    finally:
        reader.close()
    assert engine._ingest_cursor == 1


def test_already_current_owner_is_unchanged_even_when_runtime_frontier_differs(engine):
    engine.ingest([{"role": "user", "content": "original request"}])
    engine._lifecycle.advance_frontier("conversation", engine._session_id, 11)
    engine._last_compacted_store_id = 7
    before = engine._lifecycle.get_by_conversation("conversation")
    boundary(engine)
    assert engine._lifecycle.get_by_conversation("conversation") == before
    assert engine._last_compacted_store_id == 7


def test_competing_connection_cannot_be_overwritten_after_resume_waits(engine):
    engine.on_session_end(engine._session_id, [])
    attempted = Event()
    writer = sqlite3.connect(engine._store.db_path)
    writer.execute("BEGIN IMMEDIATE")
    writer.execute("UPDATE lcm_lifecycle_state SET current_session_id='replacement-session' WHERE conversation_id='conversation'")

    def callback():
        attempted.set()
        boundary(engine)

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(callback)
            assert attempted.wait(timeout=5)
            writer.commit()
            with pytest.raises(RuntimeError, match="ownership"):
                future.result(timeout=10)
        assert engine._lifecycle.get_by_conversation("conversation").current_session_id == "replacement-session"
    finally:
        writer.close()
