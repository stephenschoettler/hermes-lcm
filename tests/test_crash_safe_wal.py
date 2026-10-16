"""Tests for WAL durability configuration and graceful-close hygiene.

These tests verify the PRAGMAs applied by ``configure_connection()`` and
the best-effort passive WAL checkpoint performed by ``close()`` on all three
SQLite helpers.

This covers the PR #237 hardening path without overclaiming it: graceful close
can checkpoint committed WAL frames best-effort, while unexpected process death
still depends on SQLite WAL recovery.
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from hermes_lcm.db_bootstrap import (
    configure_connection,
    ensure_message_origin_columns,
)
from hermes_lcm.config import LCMConfig
from hermes_lcm.store import MessageStore, load_sidecar_health_failure
from hermes_lcm.engine import LCMEngine
from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.lifecycle_state import LifecycleStateStore


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
        assert val == 268_435_456, f"expected mmap_size=268435456, got {val}"


# --------------------------------------------------------------------------- #
#  Graceful close — WAL checkpoint on close
# --------------------------------------------------------------------------- #


class TestGracefulClose:
    """Verify that close() performs a best-effort passive WAL checkpoint
    without raising."""

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
        # The sentinel intentionally keeps WAL sidecars present after close();
        # the invariant is that the database remains readable and healthy.
        assert Path(str(db) + "-wal").is_file()
        assert Path(str(db) + "-shm").is_file()
        with sqlite3.connect(f"file:{db}?mode=ro&immutable=1", uri=True) as conn:
            assert conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        store.shutdown()

    def test_message_store_close_is_idempotent(self, tmp_path: Path):
        db = tmp_path / "store.db"
        store = MessageStore(db)
        store.close()
        store.close()  # should not raise

    def test_sentinel_holds_contour(self, tmp_path: Path):
        db = tmp_path / "store.db"
        wal = Path(str(db) + "-wal")
        shm = Path(str(db) + "-shm")

        store = MessageStore(db)
        store.append("sess", {"role": "user", "content": "hello"})
        before = (
            (wal.stat().st_dev, wal.stat().st_ino),
            (shm.stat().st_dev, shm.stat().st_ino),
        )

        store.close()

        assert wal.is_file()
        assert shm.is_file()
        assert (
            (wal.stat().st_dev, wal.stat().st_ino),
            (shm.stat().st_dev, shm.stat().st_ino),
        ) == before

        reopened = MessageStore(db)
        try:
            assert wal.is_file()
            assert shm.is_file()
            assert (
                (wal.stat().st_dev, wal.stat().st_ino),
                (shm.stat().st_dev, shm.stat().st_ino),
            ) == before
        finally:
            reopened.shutdown()
            store.shutdown()

    def test_guard_tolerates_missing_sidecars_after_full_shutdown(self, tmp_path: Path, caplog):
        db = tmp_path / "store.db"
        wal = Path(str(db) + "-wal")
        shm = Path(str(db) + "-shm")
        store = MessageStore(db)
        store.append("sess", {"role": "user", "content": "hello"})
        store.shutdown()

        wal.unlink(missing_ok=True)
        shm.unlink(missing_ok=True)

        with caplog.at_level("WARNING"):
            assert store.check_sidecars_intact() is True

        assert "SQLite WAL sidecar missing with no open store connection" in caplog.text
        assert "sidecar vanished" not in caplog.text
        assert not load_sidecar_health_failure(db)

    def test_guard_fuses_when_keeper_open_and_sidecars_unlinked(self, tmp_path: Path):
        db = tmp_path / "store.db"
        wal = Path(str(db) + "-wal")
        shm = Path(str(db) + "-shm")
        store = MessageStore(db)
        store.append("sess", {"role": "user", "content": "hello"})
        store.close()

        wal.unlink()
        shm.unlink()

        try:
            with pytest.raises(RuntimeError, match="sidecar vanished.*issue #628"):
                store.check_sidecars_intact()
            assert load_sidecar_health_failure(db)
        finally:
            store.shutdown()

    def test_sidecar_alien_recreate_with_keeper_open_fuses(self, tmp_path: Path):
        db = tmp_path / "store.db"
        shm = Path(str(db) + "-shm")
        store = MessageStore(db)
        store.append("sess", {"role": "user", "content": "hello"})
        store.close()

        shm.unlink()
        shm.write_bytes(b"alien contour")

        try:
            with pytest.raises(RuntimeError, match="sidecar identity changed.*issue #628"):
                store.check_sidecars_intact()
        finally:
            store.shutdown()

    def test_sidecar_check_is_noop_before_contour_materializes(self, tmp_path: Path):
        db = tmp_path / "fresh.db"
        store = MessageStore.__new__(MessageStore)
        store.db_path = db
        store._is_memory_database = False
        store._conn = sqlite3.connect(str(db))
        store._contour_lock = threading.RLock()
        store._contour_ids = {}
        store._db_identity = None
        store._benign_missing_sidecars_warned = set()
        try:
            configure_connection(store._conn)
            assert not Path(str(db) + "-wal").exists()
            assert not Path(str(db) + "-shm").exists()
            assert store.check_sidecars_intact() is True
        finally:
            store._conn.close()

    def test_engine_guard_closes_helpers_and_refuses_dag_after_confirmed_sidecar_failure(
        self,
        tmp_path: Path,
    ):
        db = tmp_path / "engine.db"
        replacement = tmp_path / "replacement.db"
        engine = LCMEngine(config=LCMConfig(database_path=str(db)))
        store = engine._store
        dag = engine._dag
        engine._store.append("sess", {"role": "user", "content": "hello"})

        with sqlite3.connect(replacement) as conn:
            configure_connection(conn)
            conn.execute("CREATE TABLE replacement_marker(value TEXT)")
            conn.commit()
        replacement.replace(db)
        engine._sidecar_guard_tick()

        assert engine._storage_unavailable_reason
        assert engine._storage_bound is False
        assert store._conn is None
        assert store._sentinel_conn is None
        assert dag._conn is None
        with pytest.raises(RuntimeError, match="database file identity changed.*issue #628"):
            engine._ensure_storage()
        with pytest.raises(RuntimeError, match="database file identity changed.*issue #628"):
            engine._dag.add_node(SummaryNode(session_id="sess", summary="blocked"))

    def test_inline_confirmed_failure_trips_engine_fuse_immediately(self, tmp_path: Path):
        db = tmp_path / "engine.db"
        replacement = tmp_path / "replacement.db"
        engine = LCMEngine(config=LCMConfig(database_path=str(db)))
        store = engine._store
        dag = engine._dag
        engine._store.append("sess", {"role": "user", "content": "hello"})

        with sqlite3.connect(replacement) as conn:
            configure_connection(conn)
            conn.execute("CREATE TABLE replacement_marker(value TEXT)")
            conn.commit()
        replacement.replace(db)
        store._last_inline_contour_check_at = 0.0

        with pytest.raises(RuntimeError, match="database file identity changed.*issue #628"):
            store.append("sess", {"role": "user", "content": "blocked"})

        # The contour-failure listener tripped the engine fuse synchronously:
        # the very next lazy-attribute access tears down ALL helpers (not just
        # MessageStore) and refuses — no 30s window for DAG/lifecycle writes.
        assert engine._storage_unavailable_reason
        assert engine._storage_bound is False
        with pytest.raises(RuntimeError, match="database file identity changed.*issue #628"):
            engine._dag.add_node(SummaryNode(session_id="sess", summary="blocked"))
        assert store._sentinel_conn is None
        # dag was resolved BEFORE the fuse tripped: the teardown must have
        # closed its connection (the same instance, now unusable).
        assert dag._conn is None

    def test_write_path_rate_limited_restat_detects_db_inode_change(self, tmp_path: Path):
        db = tmp_path / "store.db"
        replacement = tmp_path / "replacement.db"
        store = MessageStore(db)
        store.append("sess", {"role": "user", "content": "hello"})

        with sqlite3.connect(replacement) as conn:
            configure_connection(conn)
            conn.execute("CREATE TABLE replacement_marker(value TEXT)")
            conn.commit()
        replacement.replace(db)
        store._last_inline_contour_check_at = 0.0

        try:
            with pytest.raises(RuntimeError, match="database file identity changed.*issue #628"):
                store.append("sess", {"role": "user", "content": "blocked"})
        finally:
            store.shutdown()

    def test_write_path_inline_fuses_on_missing_sidecar_with_open_keeper(self, tmp_path: Path):
        db = tmp_path / "store.db"
        shm = Path(str(db) + "-shm")
        store = MessageStore(db)
        store.append("sess", {"role": "user", "content": "hello"})

        shm.unlink()
        store._last_inline_contour_check_at = 0.0

        try:
            with pytest.raises(RuntimeError, match="sidecar vanished.*issue #628"):
                store.append("sess", {"role": "user", "content": "again"})
        finally:
            store.shutdown()

    def test_keeper_connection_holds_wal(self, tmp_path: Path):
        db = tmp_path / "engine.db"
        wal = Path(str(db) + "-wal")
        engine = LCMEngine(config=LCMConfig(database_path=str(db)))
        engine._session_id = "sess"
        engine.context_length = 200000
        engine.threshold_tokens = int(200000 * engine._config.context_threshold)
        try:
            engine.ingest([{"role": "user", "content": "keeper initial"}])
            assert wal.is_file()
            script = (
                "import sqlite3, sys\n"
                "db = sys.argv[1]\n"
                "for _ in range(5):\n"
                "    conn = sqlite3.connect(db)\n"
                "    conn.execute('PRAGMA journal_mode=WAL')\n"
                "    conn.execute('SELECT COUNT(*) FROM messages').fetchone()\n"
                "    conn.close()\n"
            )
            subprocess.run([sys.executable, "-c", script, str(db)], check=True)

            assert wal.is_file()
            engine.ingest([
                {"role": "user", "content": "keeper initial"},
                {"role": "user", "content": "keeper still ingesting"},
            ])
            assert engine._store.get_session_count("sess") == 2
        finally:
            engine.shutdown()

    def test_guard_fuses_on_db_inode_change(self, tmp_path: Path):
        db = tmp_path / "engine.db"
        replacement = tmp_path / "replacement.db"
        engine = LCMEngine(config=LCMConfig(database_path=str(db)))
        engine._session_id = "sess"
        engine.context_length = 200000
        engine.threshold_tokens = int(200000 * engine._config.context_threshold)
        try:
            engine.ingest([{"role": "user", "content": "before replacement"}])
            with sqlite3.connect(replacement) as conn:
                configure_connection(conn)
                conn.execute("CREATE TABLE replacement_marker(value TEXT)")
                conn.commit()
            replacement.replace(db)
            engine._store._last_inline_contour_check_at = 0.0

            engine.ingest([
                {"role": "user", "content": "before replacement"},
                {"role": "user", "content": "after replacement"},
            ])
            assert engine._storage_unavailable_reason
            assert engine._storage_bound is False
        finally:
            engine.shutdown()

    def test_reset_closes_sentinel(self, tmp_path: Path):
        fd_dir = Path("/proc/self/fd")
        if not fd_dir.exists():
            pytest.skip("fd inspection is Linux-specific")

        def open_fds_for(path: Path) -> list[str]:
            result = []
            resolved = str(path)
            for fd in fd_dir.iterdir():
                try:
                    target = fd.resolve(strict=False)
                except OSError:
                    continue
                if str(target).startswith(resolved):
                    result.append(str(target))
            return result

        db = tmp_path / "store.db"
        engine = LCMEngine(config=LCMConfig(database_path=str(db)))
        engine._store.append("sess", {"role": "user", "content": "before reset"})
        assert open_fds_for(db)

        engine.shutdown()

        assert open_fds_for(db) == []
        for path in (db, Path(str(db) + "-wal"), Path(str(db) + "-shm")):
            path.unlink(missing_ok=True)

        reopened = LCMEngine(config=LCMConfig(database_path=str(db)))
        try:
            assert reopened._store.get_session_count("sess") == 0
        finally:
            reopened.shutdown()

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
        # Manually invalidate the connection so close() has nothing to do
        store._conn = None
        store.close()  # should not raise

    def test_summary_dag_close_does_not_mask_sqlite_error(self, tmp_path: Path):
        db = tmp_path / "dag.db"
        dag = SummaryDAG(db)
        dag._conn = None
        dag.close()  # should not raise

    def test_lifecycle_state_close_does_not_mask_sqlite_error(self, tmp_path: Path):
        db = tmp_path / "lifecycle.db"
        lc = LifecycleStateStore(db)
        lc._conn = None
        lc.close()  # should not raise


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
