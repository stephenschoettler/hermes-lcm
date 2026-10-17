"""Bounded-wait guarantees for ``_temporary_sqlite_busy_timeout``.

These cover the defect where a gateway lifecycle hook could stall ~1s despite
asking for a 50ms budget: ``PRAGMA busy_timeout`` alone overshoots a short
window by more than an order of magnitude under real WAL writer contention.

The assertions are deliberately generous in absolute terms (the point is
"bounded, not 17x over") so they do not become scheduler-jitter flakes, while
still failing loudly if the PRAGMA-only behaviour ever comes back.
"""
import sqlite3
import time

import pytest

from hermes_lcm.sqlite_util import (
    _is_sqlite_locked_error,
    _temporary_sqlite_busy_timeout,
    _wait_for_write_lock,
)

BUDGET_MS = 50
# The unfixed implementation measured ~850ms for a 50ms budget. Anything under
# this proves the wait is actually bounded without pinning an exact timing.
MAX_ELAPSED_S = 0.30


def _wal_db(tmp_path):
    db = tmp_path / "bounded.db"
    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE t(x)")
    conn.commit()
    conn.close()
    return db


@pytest.fixture
def locked_db(tmp_path):
    """A WAL database whose write lock is held by another connection."""
    db = _wal_db(tmp_path)
    locker = sqlite3.connect(str(db), timeout=1.0, isolation_level=None)
    locker.execute("BEGIN IMMEDIATE")
    try:
        yield db
    finally:
        try:
            locker.execute("ROLLBACK")
        finally:
            locker.close()


def test_entry_is_bounded_when_the_write_lock_is_held(locked_db):
    conn = sqlite3.connect(str(locked_db))
    conn.execute("PRAGMA busy_timeout=750")  # deliberately larger than the budget
    try:
        started = time.monotonic()
        with _temporary_sqlite_busy_timeout([conn], BUDGET_MS):
            with pytest.raises(sqlite3.OperationalError) as excinfo:
                conn.execute("INSERT INTO t VALUES (1)")
                conn.commit()
        elapsed = time.monotonic() - started
    finally:
        conn.close()

    # The caller's own statement must be what fails, so its specific recovery
    # branch and diagnostics still run.
    assert _is_sqlite_locked_error(excinfo.value)
    assert elapsed < MAX_ELAPSED_S, f"bounded wait overshot: {elapsed:.3f}s"


def test_multiple_connections_share_one_budget(locked_db):
    """Two connections must not each spend the full budget serially."""
    first = sqlite3.connect(str(locked_db))
    second = sqlite3.connect(str(locked_db))
    for conn in (first, second):
        conn.execute("PRAGMA busy_timeout=750")
    try:
        started = time.monotonic()
        with _temporary_sqlite_busy_timeout([first, second], BUDGET_MS):
            pass
        elapsed = time.monotonic() - started
    finally:
        first.close()
        second.close()

    assert elapsed < MAX_ELAPSED_S, f"per-connection budgets stacked: {elapsed:.3f}s"


def test_original_busy_timeout_is_restored_after_a_bounded_wait(locked_db):
    conn = sqlite3.connect(str(locked_db))
    conn.execute("PRAGMA busy_timeout=750")
    try:
        with _temporary_sqlite_busy_timeout([conn], BUDGET_MS):
            pass
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 750
    finally:
        conn.close()


def test_original_busy_timeout_is_restored_when_the_block_raises(locked_db):
    conn = sqlite3.connect(str(locked_db))
    conn.execute("PRAGMA busy_timeout=750")
    try:
        with pytest.raises(RuntimeError):
            with _temporary_sqlite_busy_timeout([conn], BUDGET_MS):
                raise RuntimeError("caller exploded")
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 750
    finally:
        conn.close()


def test_uncontended_path_is_not_slowed(tmp_path):
    """The common case must not pay for the contended case."""
    db = _wal_db(tmp_path)
    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        started = time.monotonic()
        with _temporary_sqlite_busy_timeout([conn], BUDGET_MS):
            conn.execute("INSERT INTO t VALUES (1)")
            conn.commit()
        elapsed = time.monotonic() - started
        assert conn.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 1
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    finally:
        conn.close()

    assert elapsed < 0.1, f"uncontended write regressed: {elapsed:.3f}s"


def test_probe_leaves_no_open_transaction(tmp_path):
    db = _wal_db(tmp_path)
    conn = sqlite3.connect(str(db))
    try:
        with _temporary_sqlite_busy_timeout([conn], BUDGET_MS):
            assert not conn.in_transaction
        assert not conn.in_transaction
    finally:
        conn.close()


def test_caller_owned_transaction_is_not_probed(tmp_path):
    """Probing a connection mid-transaction would silently end it."""
    db = _wal_db(tmp_path)
    conn = sqlite3.connect(str(db), isolation_level=None)
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("INSERT INTO t VALUES (1)")
        assert _wait_for_write_lock(conn, time.monotonic() + 1.0) is True
        # The caller's transaction must still be open and still hold its write.
        assert conn.in_transaction
        conn.execute("ROLLBACK")
        assert conn.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 0
    finally:
        conn.close()


def test_none_connections_are_ignored(tmp_path):
    db = _wal_db(tmp_path)
    conn = sqlite3.connect(str(db))
    try:
        with _temporary_sqlite_busy_timeout([None, conn, None], BUDGET_MS):
            conn.execute("INSERT INTO t VALUES (1)")
            conn.commit()
        assert conn.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 1
    finally:
        conn.close()


def test_zero_budget_does_not_wait(locked_db):
    conn = sqlite3.connect(str(locked_db))
    conn.execute("PRAGMA busy_timeout=750")
    try:
        started = time.monotonic()
        with _temporary_sqlite_busy_timeout([conn], 0):
            with pytest.raises(sqlite3.OperationalError):
                conn.execute("INSERT INTO t VALUES (1)")
        elapsed = time.monotonic() - started
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 750
    finally:
        conn.close()

    assert elapsed < 0.1, f"zero budget still waited: {elapsed:.3f}s"
