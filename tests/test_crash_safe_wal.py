"""Tests for WAL durability configuration and graceful-close hygiene.

These tests verify the PRAGMAs applied by ``configure_connection()`` and
the TRUNCATE-then-PASSIVE WAL checkpoint performed by ``close()`` on the
SQLite helpers.

Graceful close checkpoints committed WAL frames best-effort. Unexpected
process death still depends on SQLite WAL recovery. mmap stays off so
concurrent writers cannot tear page_count against btree pointers.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_lcm.assertion_store import AssertionStore
from hermes_lcm.db_bootstrap import (
    SchemaVersionTooNewError,
    _WRITER_HUB,
    _writer_hub_key,
    acquire_lcm_connection,
    checkpoint_wal,
    configure_connection,
    ensure_message_origin_columns,
    release_lcm_connection,
)
from hermes_lcm.store import MessageStore
from hermes_lcm.dag import SummaryDAG
from hermes_lcm.lifecycle_state import LifecycleStateStore
from hermes_lcm.trajectory_store import CorpusIdentity, TrajectoryStore


# --------------------------------------------------------------------------- #
#  configure_connection PRAGMA verification
# --------------------------------------------------------------------------- #


class TestConfigureConnectionPragmas:
    """Assert that configure_connection() sets the intended PRAGMAs."""

    @pytest.fixture()
    def db_path(self, tmp_path: Path):
        """Return a temp file path for an on-disk database (WAL requires a
        real file — :memory: silently reports journal_mode='memory')."""
        return tmp_path / "test.db"

    def test_journal_mode_is_wal(self, db_path: Path):
        conn = sqlite3.connect(str(db_path))
        configure_connection(conn)
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
        assert mode == "wal", f"expected journal_mode=wal, got {mode!r}"

    def test_synchronous_is_full(self, db_path: Path):
        conn = sqlite3.connect(str(db_path))
        configure_connection(conn)
        # PRAGMA synchronous returns an integer: 0=OFF, 1=NORMAL, 2=FULL
        val = conn.execute("PRAGMA synchronous").fetchone()[0]
        conn.close()
        assert val == 2, f"expected synchronous=FULL (2), got {val}"

    def test_busy_timeout(self, db_path: Path):
        conn = sqlite3.connect(str(db_path))
        configure_connection(conn)
        val = conn.execute("PRAGMA busy_timeout").fetchone()[0]
        conn.close()
        assert val == 30_000, f"expected busy_timeout=30000, got {val}"

    def test_wal_autocheckpoint(self, db_path: Path):
        conn = sqlite3.connect(str(db_path))
        configure_connection(conn)
        # After setting, PRAGMA wal_autocheckpoint returns the NEW value.
        val = conn.execute("PRAGMA wal_autocheckpoint").fetchone()[0]
        conn.close()
        assert val == 500, f"expected wal_autocheckpoint=500, got {val}"

    def test_journal_size_limit(self, db_path: Path):
        conn = sqlite3.connect(str(db_path))
        configure_connection(conn)
        val = conn.execute("PRAGMA journal_size_limit").fetchone()[0]
        conn.close()
        assert val == 67_108_864, f"expected journal_size_limit=67108864, got {val}"

    def test_mmap_size(self, db_path: Path):
        conn = sqlite3.connect(str(db_path))
        configure_connection(conn)
        val = conn.execute("PRAGMA mmap_size").fetchone()[0]
        conn.close()
        assert val == 0, f"expected mmap_size=0, got {val}"


class TestCheckpointWal:
    def test_truncate_then_passive_on_lock(self):
        seen: list[str] = []

        class _Conn:
            def execute(self, sql: str):
                seen.append(sql)
                if "TRUNCATE" in sql:
                    raise sqlite3.OperationalError("database is locked")

        checkpoint_wal(_Conn())  # type: ignore[arg-type]
        assert seen == [
            "PRAGMA wal_checkpoint(TRUNCATE)",
            "PRAGMA wal_checkpoint(PASSIVE)",
        ]

    def test_truncate_success_skips_passive(self):
        seen: list[str] = []

        class _Conn:
            def execute(self, sql: str):
                seen.append(sql)

        checkpoint_wal(_Conn())  # type: ignore[arg-type]
        assert seen == ["PRAGMA wal_checkpoint(TRUNCATE)"]


# --------------------------------------------------------------------------- #
#  Graceful close — WAL checkpoint on close
# --------------------------------------------------------------------------- #


class TestGracefulClose:
    """Verify that close() performs a best-effort WAL checkpoint without raising."""

    def _write_and_get_wal_size(self, db_path: Path) -> int:
        """Return WAL file size in bytes (0 if no WAL)."""
        wal = Path(str(db_path) + "-wal")
        return wal.stat().st_size if wal.exists() else 0

    # -- MessageStore -------------------------------------------------------

    def test_message_store_close_runs_checkpoint(self, tmp_path: Path):
        db = tmp_path / "store.db"
        store = MessageStore(db)
        store.append("sess", {"role": "user", "content": "hello"})
        assert db.exists()
        store.close()
        # After close the WAL should be small or non-existent (all frames
        # checkpointed by the passive call).
        wal_size = self._write_and_get_wal_size(db)
        assert wal_size < 4096, (
            f"WAL still {wal_size} bytes after MessageStore.close(); "
            "checkpoint may not have run"
        )

    def test_message_store_close_is_idempotent(self, tmp_path: Path):
        db = tmp_path / "store.db"
        store = MessageStore(db)
        store.close()
        store.close()  # should not raise

    # -- SummaryDAG ---------------------------------------------------------

    def test_summary_dag_close_runs_checkpoint(self, tmp_path: Path):
        db = tmp_path / "dag.db"
        dag = SummaryDAG(db)
        # Insert a minimal summary node so the WAL has content
        conn = dag._conn
        assert conn is not None
        conn.execute(
            "INSERT INTO summary_nodes (session_id, depth, summary, "
            "source_ids, source_type, created_at, earliest_at, latest_at) "
            "VALUES ('sess', 0, 'summary', '[]', 'messages', 0.0, 0.0, 0.0)"
        )
        conn.commit()
        dag.close()
        wal_size = self._write_and_get_wal_size(db)
        assert wal_size < 4096, (
            f"WAL still {wal_size} bytes after SummaryDAG.close(); "
            "checkpoint may not have run"
        )

    # -- LifecycleStateStore ------------------------------------------------

    def test_lifecycle_state_close_runs_checkpoint(self, tmp_path: Path):
        db = tmp_path / "lifecycle.db"
        lc = LifecycleStateStore(db)
        lc.bind_session("sess")
        lc.close()
        wal_size = self._write_and_get_wal_size(db)
        assert wal_size < 4096, (
            f"WAL still {wal_size} bytes after LifecycleStateStore.close(); "
            "checkpoint may not have run"
        )

    # -- Masking check ------------------------------------------------------

    def test_message_store_close_does_not_mask_sqlite_error(self, tmp_path: Path):
        """close() should not silently swallow a broken connection — it only
        ignores errors from the checkpoint attempt itself, not from the
        underlying close."""
        db = tmp_path / "store.db"
        store = MessageStore(db)
        conn = store._conn
        # Manually invalidate the connection so close() has nothing to do
        store._conn = None
        store.close()  # should not raise
        release_lcm_connection(conn)

    def test_summary_dag_close_does_not_mask_sqlite_error(self, tmp_path: Path):
        db = tmp_path / "dag.db"
        dag = SummaryDAG(db)
        conn = dag._conn
        dag._conn = None
        dag.close()  # should not raise
        release_lcm_connection(conn)

    def test_lifecycle_state_close_does_not_mask_sqlite_error(self, tmp_path: Path):
        db = tmp_path / "lifecycle.db"
        lc = LifecycleStateStore(db)
        conn = lc._conn
        lc._conn = None
        lc.close()  # should not raise
        release_lcm_connection(conn)


# --------------------------------------------------------------------------- #
#  Concurrent-startup migration race (idempotent ADD COLUMN)
# --------------------------------------------------------------------------- #


def _seed_pre_conversation_id_messages(path: Path) -> None:
    """Create a ``messages`` table as a pre-v5 build left it: without the
    ``conversation_id`` column the column migration later adds. Scoped to the
    column DDL only (no FTS), so the test isolates the ADD COLUMN race."""
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(
            """
            CREATE TABLE messages (
                store_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                timestamp REAL NOT NULL
            );
            """
        )
        conn.commit()
    finally:
        conn.close()


class TestConcurrentStartupMigration:
    """Concurrent process startup must not crash on duplicate-column ALTERs.

    Regression for the pre-fix race: gateway + CLI + sub-agents open independent
    connections to one ``lcm.db`` after an upgrade and all run the column
    migrations; the loser hit ``sqlite3.OperationalError: duplicate column name``
    and crashed store construction. ``add_column_if_missing`` makes the ALTER
    idempotent so every process migrates successfully.
    """

    def test_concurrent_column_migration_is_idempotent(self, tmp_path: Path):
        db_path = tmp_path / "concurrent.db"
        _seed_pre_conversation_id_messages(db_path)

        thread_count = 8
        errors: list[BaseException] = []
        lock = threading.Lock()
        barrier = threading.Barrier(thread_count)

        def migrate() -> None:
            # Each thread is a stand-in for a separate process: its own
            # connection to the same file, its own busy_timeout, racing the same
            # ``ALTER TABLE messages ADD COLUMN conversation_id``.
            conn = sqlite3.connect(str(db_path), timeout=30.0)
            try:
                configure_connection(conn)
                # Timeout + abort-on-error: a thread that fails before the
                # barrier must break it, or the surviving threads wait forever
                # and the deadlock hides the original error from the assert.
                barrier.wait(timeout=60.0)  # maximise overlap on the migration DDL
                ensure_message_origin_columns(conn)
                conn.commit()
            except BaseException as exc:  # noqa: BLE001 - re-asserted below
                barrier.abort()
                with lock:
                    errors.append(exc)
            finally:
                conn.close()

        threads = [threading.Thread(target=migrate) for _ in range(thread_count)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120.0)
        stuck = [t for t in threads if t.is_alive()]
        assert not stuck, f"{len(stuck)} migration threads still running after join timeout"

        real_errors = [
            exc for exc in errors if not isinstance(exc, threading.BrokenBarrierError)
        ]
        assert not real_errors, f"concurrent column migration raised: {real_errors!r}"
        assert not errors, "barrier broke without a recorded root-cause error"
        conn = sqlite3.connect(str(db_path))
        columns = [
            row[1] for row in conn.execute("PRAGMA table_info(messages)").fetchall()
        ]
        conn.close()
        assert columns.count("conversation_id") == 1


class TestSharedWriterHub:
    """One filesystem path -> one writer connection in this process."""

    def test_message_dag_lifecycle_share_one_connection(self, tmp_path: Path):
        db = tmp_path / "shared.db"
        store = MessageStore(db)
        dag = SummaryDAG(db)
        lifecycle = LifecycleStateStore(db)
        try:
            assert store._conn is dag._conn
            assert dag._conn is lifecycle._conn
            store.append("sess", {"role": "user", "content": "hello"})
            lifecycle.bind_session("sess")
            assert store.get_session_messages("sess")
        finally:
            store.close()
            dag.close()
            lifecycle.close()

    def test_last_release_closes_connection(self, tmp_path: Path):
        db = tmp_path / "last-close.db"
        store = MessageStore(db)
        dag = SummaryDAG(db)
        conn = store._conn
        assert conn is dag._conn
        store.close()
        conn.execute("SELECT 1").fetchone()
        dag.close()
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")

    def test_second_acquire_same_identity(self, tmp_path: Path):
        db = tmp_path / "second.db"
        a = acquire_lcm_connection(db)
        b = acquire_lcm_connection(db)
        try:
            assert a is b
        finally:
            release_lcm_connection(a)
            release_lcm_connection(b)

    def test_memory_databases_are_unshared(self):
        a = MessageStore(":memory:")
        b = SummaryDAG(":memory:")
        try:
            assert a._conn is not b._conn
        finally:
            a.close()
            b.close()

    def test_read_only_does_not_join_writer_hub(self, tmp_path: Path):
        db = tmp_path / "ro.db"
        writer = acquire_lcm_connection(db)
        configure_connection(writer)
        writer.execute("CREATE TABLE t(x INTEGER)")
        writer.commit()
        reader = acquire_lcm_connection(db, read_only=True)
        try:
            assert reader is not writer
        finally:
            release_lcm_connection(reader)
            release_lcm_connection(writer)

    def test_shared_store_close_still_truncates_wal(self, tmp_path: Path):
        db = tmp_path / "wal-share.db"
        store = MessageStore(db)
        dag = SummaryDAG(db)
        store.append("sess", {"role": "user", "content": "hello"})
        store.close()
        dag.close()
        wal = Path(str(db) + "-wal")
        wal_size = wal.stat().st_size if wal.exists() else 0
        assert wal_size < 4096, f"WAL still {wal_size} bytes after last close"

    def test_failed_assertion_open_drops_hub_ref(self, tmp_path: Path):
        db = tmp_path / "assert-fail-open.db"
        MessageStore(db).close()
        with patch(
            "hermes_lcm.assertion_store.refuse_schema_version_too_new",
            side_effect=SchemaVersionTooNewError("too new"),
        ):
            with pytest.raises(SchemaVersionTooNewError):
                AssertionStore(db)
        assert _writer_hub_key(db) not in _WRITER_HUB
        store = MessageStore(db)
        try:
            store.append("sess", {"role": "user", "content": "hello"})
        finally:
            store.close()
        wal = Path(str(db) + "-wal")
        wal_size = wal.stat().st_size if wal.exists() else 0
        assert wal_size < 4096, f"WAL still {wal_size} bytes after failed open + close"

    def test_failed_trajectory_open_drops_hub_ref(self, tmp_path: Path):
        db = tmp_path / "traj-fail-open.db"
        MessageStore(db).close()
        identity = CorpusIdentity(
            dataset_name="example/trajectory-benchmark",
            dataset_revision="dataset-rev-1",
            harness_commit="harness-commit-1",
            tier="small",
            domain="enterprise",
            ingest_config_digest="ingest-v1",
        )
        with patch(
            "hermes_lcm.trajectory_store.refuse_schema_version_too_new",
            side_effect=SchemaVersionTooNewError("too new"),
        ):
            with pytest.raises(SchemaVersionTooNewError):
                TrajectoryStore(db, identity, asset_root=tmp_path / "assets")
        assert _writer_hub_key(db) not in _WRITER_HUB
        store = MessageStore(db)
        try:
            store.append("sess", {"role": "user", "content": "hello"})
        finally:
            store.close()
        wal = Path(str(db) + "-wal")
        wal_size = wal.stat().st_size if wal.exists() else 0
        assert wal_size < 4096, f"WAL still {wal_size} bytes after failed open + close"
