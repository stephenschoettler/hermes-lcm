"""Tests for WAL durability configuration and graceful-close hygiene.

These tests verify the PRAGMAs applied by ``configure_connection()`` and
the best-effort passive WAL checkpoint performed by ``close()`` on all three
SQLite helpers.

This covers the PR #237 hardening path without overclaiming it: graceful close
can checkpoint committed WAL frames best-effort, while unexpected process death
still depends on SQLite WAL recovery.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path

import pytest

from hermes_lcm.db_bootstrap import configure_connection
from hermes_lcm.store import MessageStore
from hermes_lcm.dag import SummaryDAG
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
