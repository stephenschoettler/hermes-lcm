"""Compression publication contention: run with pytest on this module.

Uses isolated real SQLite stores; only summary generation/publication faults are
injected. No provider calls or live profile databases are used.
"""

import copy
import sqlite3

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_messages_tokens


@pytest.fixture
def publication_engine(tmp_path, monkeypatch):
    engine = LCMEngine(config=LCMConfig(
        database_path=str(tmp_path / "publication.db"),
        fresh_tail_count=2,
        leaf_chunk_tokens=1,
        threshold_full_sweep_enabled=True,
        incremental_max_depth=0,
    ), hermes_home=str(tmp_path / "home"))
    engine._session_id = "publication-session"
    engine.threshold_tokens = 1
    monkeypatch.setattr(engine, "_summarize_leaf_chunk_with_rescue", lambda chunk, **kw: (
        chunk, count_messages_tokens(chunk), "Published project decisions", 1, 0,
    ))
    try:
        yield engine
    finally:
        engine.shutdown()


def transcript():
    return [{"role": "system", "content": "Keep project decisions."}] + [
        {"role": "user", "content": f"Decision {i}: keep source history. " * 20}
        for i in range(4)
    ] + [
        {"role": "user", "content": "What remains?"},
        {"role": "assistant", "content": "Verify the decisions."},
    ]


def lock_error(kind):
    error = sqlite3.OperationalError(kind)
    if kind == "extended busy":
        error.sqlite_errorcode = sqlite3.SQLITE_BUSY | (2 << 8)
    return error


@pytest.mark.parametrize("kind", ["database is locked", "database table is locked", "extended busy"])
def test_first_publication_lock_preserves_transcript_and_host_abort(publication_engine, monkeypatch, kind):
    engine = publication_engine
    messages = transcript()
    original = copy.deepcopy(messages)
    engine._ingest_messages(messages)
    before = engine._store._conn.iterdump()
    before = list(before)
    frontier = engine._last_compacted_store_id

    def fail(node):
        raise lock_error(kind)

    with monkeypatch.context() as patch:
        patch.setattr(engine._dag, "add_node", fail)
        result = engine.compress(messages, current_tokens=count_messages_tokens(messages))

    assert result is messages
    assert messages == original
    assert engine._dag.get_session_nodes(engine._session_id) == []
    assert engine._last_compacted_store_id == frontier
    assert list(engine._store._conn.iterdump()) == before
    assert engine.compression_count == 0
    assert engine._last_compress_aborted is True
    assert engine._last_summary_error
    assert engine.last_compression_status == "error"
    assert engine.last_compression_noop_reason == ""
    assert engine.get_status()["threshold_full_sweep"]["status"] == "error"

    # A later successful attempt must not inherit host-visible abort state.
    result = engine.compress(messages, current_tokens=count_messages_tokens(messages))
    assert result != messages
    assert engine._last_compress_aborted is False
    assert engine._last_summary_error is None


def test_empty_noop_clears_stale_abort(publication_engine):
    engine = publication_engine
    engine._last_compress_aborted = True
    engine._last_summary_error = "previous lock"
    assert engine.compress([]) == []
    assert engine._last_compress_aborted is False
    assert engine._last_summary_error is None
    assert engine.last_compression_status == "noop"


@pytest.mark.parametrize("error", [sqlite3.OperationalError("no such table: summary_nodes"), sqlite3.IntegrityError("invalid node")])
def test_unrelated_publication_errors_propagate(publication_engine, monkeypatch, error):
    def fail(node):
        raise error

    monkeypatch.setattr(publication_engine._dag, "add_node", fail)
    with pytest.raises(type(error), match=str(error)):
        publication_engine.compress(transcript(), current_tokens=1000)
    assert publication_engine.last_compression_status == "error"


@pytest.mark.parametrize("sweep", [True, False])
def test_later_publication_lock_finalizes_only_committed_progress(publication_engine, monkeypatch, sweep):
    engine = publication_engine
    engine._config.threshold_full_sweep_enabled = sweep
    engine._config.dynamic_leaf_chunk_enabled = True
    engine._config.dynamic_leaf_chunk_max = 100
    messages = transcript()
    engine._ingest_messages(messages)
    original_ids = engine._get_store_id_map_for_messages(messages)
    publish = engine._dag.add_node
    calls = 0
    locked = False

    def fail_second(node):
        nonlocal calls, locked
        calls += 1
        if calls == 2:
            locked = True
            raise sqlite3.OperationalError("database is locked")
        return publish(node)

    # Post-lock bookkeeping must not attempt another write on contended stores.
    for name in ("_refresh_raw_backlog_debt", "_write_generated_ignored_placeholder_hash_counts", "_write_generated_ignored_placeholder_hash_ordinals"):
        original = getattr(engine, name)

        def guard(*args, _original=original, **kwargs):
            assert not locked, "bookkeeping write after publication lock"
            return _original(*args, **kwargs)

        monkeypatch.setattr(engine, name, guard)
    monkeypatch.setattr(engine._dag, "add_node", fail_second)
    result = engine.compress(messages, current_tokens=10000)
    nodes = engine._dag.get_session_nodes(engine._session_id)
    assert calls == 2
    assert len(nodes) == 1
    assert engine._last_compress_aborted is False
    assert engine.last_compression_status == "compacted"
    assert engine.compression_count == 1
    assert "Published project decisions" in str(result)
    consumed = set(nodes[0].source_ids)
    for message in messages[1:]:
        source_id = original_ids[id(message)]
        if source_id not in consumed:
            assert message in result
        else:
            assert message not in result
    if sweep:
        telemetry = engine.get_status()["threshold_full_sweep"]
        assert telemetry["status"] == "partial"
        assert telemetry["leaf_passes"] == 1
        assert telemetry["stop_reason"] == "leaf_publication_locked"

    # Continuing from the returned context must not summarize committed sources again.
    locked = False
    monkeypatch.setattr(engine._dag, "add_node", publish)
    engine.compress(result, current_tokens=10000)
    leaves = engine._dag.get_session_nodes(engine._session_id, depth=0)
    all_sources = [source for node in leaves for source in node.source_ids]
    assert len(all_sources) == len(set(all_sources))
    assert set(all_sources) == {original_ids[id(message)] for message in messages[1:-2]}


def test_real_sqlite_writer_contention_aborts_and_recovers(publication_engine, monkeypatch):
    engine = publication_engine
    engine._dag._conn.execute("PRAGMA busy_timeout=0")
    blocker = sqlite3.connect(engine._config.database_path)
    summarize = engine._summarize_leaf_chunk_with_rescue

    def summarize_then_lock(chunk, **kwargs):
        result = summarize(chunk, **kwargs)
        blocker.execute("BEGIN IMMEDIATE")
        return result

    messages = transcript()
    try:
        with monkeypatch.context() as patch:
            patch.setattr(engine, "_summarize_leaf_chunk_with_rescue", summarize_then_lock)
            assert engine.compress(messages, current_tokens=1000) is messages
        assert engine._last_compress_aborted is True
        assert not engine._dag._conn.in_transaction
    finally:
        blocker.rollback()
        blocker.close()
    assert engine.compress(messages, current_tokens=1000) != messages
    assert engine._last_compress_aborted is False


@pytest.mark.parametrize("sweep", [True, False])
@pytest.mark.parametrize("failure", ["lock", "operational", "integrity"])
def test_condensation_publication_lock_keeps_usable_leaves(publication_engine, monkeypatch, sweep, failure):
    import hermes_lcm.engine as engine_module

    engine = publication_engine
    engine._config.threshold_full_sweep_enabled = sweep
    engine._config.incremental_max_depth = 2
    engine._config.condensation_fanin = 2
    engine._config.cache_friendly_condensation_enabled = False
    engine._config.summary_prefix_target_tokens = 1
    engine._config.dynamic_leaf_chunk_enabled = False
    engine._dag.add_node(SummaryNode(
        session_id=engine._session_id, summary="Earlier project decisions",
        token_count=100, source_ids=[999],
    ))
    publish = engine._dag.add_node
    locked = False

    def fail_parent(node):
        nonlocal locked
        if node.depth > 0:
            locked = True
            if failure == "operational":
                raise sqlite3.OperationalError("no such table: summary_nodes")
            if failure == "integrity":
                raise sqlite3.IntegrityError("invalid parent")
            raise sqlite3.OperationalError("database is locked")
        return publish(node)

    monkeypatch.setattr(engine._dag, "add_node", fail_parent)
    monkeypatch.setattr(engine_module, "summarize_with_escalation", lambda **kw: ("Failed parent", 1))
    for name in ("_refresh_raw_backlog_debt", "_write_generated_ignored_placeholder_hash_counts", "_write_generated_ignored_placeholder_hash_ordinals"):
        original = getattr(engine, name)

        def guard(*args, _original=original, **kwargs):
            assert not locked, "bookkeeping write after condensation lock"
            return _original(*args, **kwargs)

        monkeypatch.setattr(engine, name, guard)
    if failure != "lock":
        error_type = sqlite3.OperationalError if failure == "operational" else sqlite3.IntegrityError
        with pytest.raises(error_type):
            engine.compress(transcript(), current_tokens=1000)
        return
    result = engine.compress(transcript(), current_tokens=1000)
    assert locked
    assert engine._last_compress_aborted is False
    assert engine.last_compression_status == "compacted"
    assert "Failed parent" not in str(result)
    assert "Earlier project decisions" in str(result)
    assert "Published project decisions" in str(result)
    assert all(node.depth == 0 for node in engine._dag.get_session_nodes(engine._session_id))
    if sweep:
        telemetry = engine.get_status()["threshold_full_sweep"]
        assert telemetry["status"] == "partial"
        assert telemetry["stop_reason"] == "condensation_publication_locked"


@pytest.mark.parametrize("stage", ["insert", "commit"])
def test_failed_add_node_rolls_back_and_connection_is_reusable(tmp_path, stage):
    dag = SummaryDAG(str(tmp_path / "dag.db"))
    connection = dag._conn

    class FailingConnection:
        def execute(self, *args):
            if stage == "insert":
                # Model a failing trigger after earlier transaction work.
                connection.execute("INSERT INTO summary_nodes(session_id, depth, summary, token_count, source_token_count, source_ids, source_type, created_at) VALUES('pending', 0, 'pending', 1, 1, '[]', 'messages', 1)")
                raise sqlite3.OperationalError("database table is locked")
            return connection.execute(*args)

        def commit(self):
            raise sqlite3.OperationalError("database is locked")

        def rollback(self):
            connection.rollback()

    node = SummaryNode(session_id="test", summary="decision", source_ids=[1])
    try:
        dag._conn = FailingConnection()
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            dag.add_node(node)
        assert not connection.in_transaction
        assert node.node_id == 0
        dag._conn = connection
        assert dag.get_session_nodes("pending") == []
        assert dag.get_session_nodes("test") == []
        assert dag.add_node(node) > 0
        assert len(dag.get_session_nodes("test")) == 1
    finally:
        dag._conn = connection
        dag.close()


def test_failed_add_node_rollback_failure_is_not_reported_as_lock_abort(tmp_path):
    dag = SummaryDAG(str(tmp_path / "dag-rollback-failure.db"))
    connection = dag._conn

    class RollbackFailingConnection:
        def execute(self, *args):
            raise sqlite3.OperationalError("database is locked")

        def rollback(self):
            raise sqlite3.OperationalError("disk I/O error during rollback")

    try:
        dag._conn = RollbackFailingConnection()
        with pytest.raises(RuntimeError, match="failed to roll back summary-node publication") as caught:
            dag.add_node(SummaryNode(session_id="test", summary="decision", source_ids=[1]))
        assert not isinstance(caught.value, sqlite3.Error)
    finally:
        dag._conn = connection
        dag.close()
