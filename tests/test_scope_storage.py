from __future__ import annotations

import sqlite3

from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.scope_storage import (
    ScopeBackfillIncompleteError,
    backfill_scopes,
    enumerate_scope_writers,
    setup_teams_scope,
    verify_scope_storage,
)
from hermes_lcm.rollup_store import RollupStore
from hermes_lcm.store import MessageStore


def test_non_teams_store_keeps_legacy_rows_and_read_shape(tmp_path):
    db_path = tmp_path / "non-teams.db"
    store = MessageStore(db_path)
    dag = SummaryDAG(db_path)
    try:
        store_id = store.append("legacy-session", {"role": "user", "content": "hello"})
        node_id = dag.add_node(
            SummaryNode(session_id="legacy-session", summary="hello", created_at=1.0)
        )
        message = store.get(store_id)
        node = dag.get_node(node_id)
        assert message["content"] == "hello"
        assert "access_scope" not in message
        assert node is not None and node.access_scope is None
        assert store.connection.execute(
            "SELECT access_scope FROM messages WHERE store_id=?", (store_id,)
        ).fetchone()[0] is None
        assert verify_scope_storage(store.connection, teams_enabled=False)["status"] == "not-enabled"
    finally:
        dag.close()
        store.close()


def test_teams_setup_backfill_is_idempotent_and_resumable(tmp_path):
    db_path = tmp_path / "backfill.db"
    store = MessageStore(db_path)
    dag = SummaryDAG(db_path)
    try:
        store.append_batch(
            "session-a",
            [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}],
        )
        dag.add_node(SummaryNode(session_id="session-a", summary="a", created_at=1.0))
        first = setup_teams_scope(
            store.connection, lambda session_id: f"owner:{session_id}", batch_size=1
        )
        assert first["total_updated"] == 3
        second = backfill_scopes(
            store.connection, lambda session_id: f"owner:{session_id}", batch_size=1
        )
        assert second["total_updated"] == 0
        result = verify_scope_storage(store.connection)
        assert result["status"] == "verified"
        assert result["tables"]["messages"]["stamped"] == 2
        assert result["tables"]["messages"]["unstamped"] == 0
        assert result["tables"]["summary_nodes"]["stamped"] == 1
    finally:
        dag.close()
        store.close()


def test_backfill_resumes_after_a_committed_batch(tmp_path):
    store = MessageStore(tmp_path / "resume.db")
    try:
        store.append_batch(
            "session-a",
            [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}],
        )
        calls = 0

        def interrupted_owner(session_id):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("simulated setup interruption")
            return f"owner:{session_id}"

        # The run still fails loudly, but only AFTER every table has been
        # attempted, so the report names what succeeded as well as what did
        # not. Isolation stops one table cancelling the tables after it; it
        # does not let a caller proceed as though the enable had worked.
        try:
            backfill_scopes(store.connection, interrupted_owner, batch_size=1)
        except ScopeBackfillIncompleteError as exc:
            assert "simulated setup interruption" in exc.report["failures"]["messages"]
            assert exc.report["complete"] is False
            # This fixture is a MessageStore, so `messages` is the only
            # scope-bearing table present -- the cross-table isolation property
            # is pinned in tests/test_teams_preflight.py, which builds two.
        else:
            raise AssertionError("the interruption must stop the first run")
        assert store.connection.execute(
            "SELECT COUNT(*) FROM messages WHERE access_scope IS NOT NULL"
        ).fetchone()[0] == 1

        resumed = backfill_scopes(
            store.connection, lambda session_id: f"owner:{session_id}", batch_size=1
        )
        assert resumed["total_updated"] == 1
        assert store.connection.execute(
            "SELECT COUNT(*) FROM messages WHERE access_scope IS NULL"
        ).fetchone()[0] == 0
    finally:
        store.close()


def test_empty_store_is_nothing_to_verify_not_verified():
    conn = sqlite3.connect(":memory:")
    result = verify_scope_storage(conn)
    assert result["status"] == "nothing-to-verify"
    assert result["status"] != "verified"
    assert result["observed_rows"] == 0


def test_unstamped_row_counts_fail_honestly(tmp_path):
    store = MessageStore(tmp_path / "unstamped.db")
    try:
        store.append("session-a", {"role": "user", "content": "a"})
        result = verify_scope_storage(store.connection)
        assert result["status"] == "fail"
        assert result["tables"]["messages"] == {
            "exists": True,
            "stamped": 0,
            "unstamped": 1,
            "total": 1,
        }
    finally:
        store.close()


def test_writer_guard_names_a_writer_that_omits_scope(tmp_path):
    source = tmp_path / "writers.py"
    source.write_text(
        """
def bad_writer(conn):
    conn.execute(\"INSERT INTO messages(session_id, role) VALUES (?, ?)\", ('s', 'user'))
""",
        encoding="utf-8",
    )
    writers = enumerate_scope_writers(tmp_path)
    assert len(writers) == 1
    assert not writers[0].populates_scope
    assert "bad_writer" in writers[0].name


def test_rollup_partition_scope_does_not_mask_unstamped_access_scope(tmp_path):
    rollups = RollupStore(tmp_path / "rollup-mutation.db")
    try:
        conn = rollups.connection
        conn.execute(
            """
            INSERT INTO lcm_rollups(
                period_kind, period_start, scope, access_scope, status
            ) VALUES ('day', '2026-08-05', 'partition-session', NULL, 'stale')
            """
        )
        conn.commit()
        result = verify_scope_storage(conn)
        assert result["status"] == "fail"
        assert result["tables"]["lcm_rollups"] == {
            "exists": True,
            "stamped": 0,
            "unstamped": 1,
            "total": 1,
        }
        assert "1 unstamped access-scope row(s)" in result["message"]
    finally:
        rollups.close()


def test_rollup_backfill_stamps_access_scope_without_rewriting_partition(tmp_path):
    rollups = RollupStore(tmp_path / "rollup-backfill.db")
    try:
        conn = rollups.connection
        conn.execute(
            """
            INSERT INTO lcm_rollups(
                period_kind, period_start, scope, access_scope, status
            ) VALUES ('day', '2026-08-05', 'session-a', NULL, 'stale')
            """
        )
        conn.commit()
        result = setup_teams_scope(
            conn, lambda session_id: f"owner:{session_id}"
        )
        assert result["updated"]["lcm_rollups"] == 1
        row = conn.execute(
            "SELECT scope, access_scope FROM lcm_rollups"
        ).fetchone()
        assert tuple(row) == ("session-a", "owner:session-a")
    finally:
        rollups.close()
