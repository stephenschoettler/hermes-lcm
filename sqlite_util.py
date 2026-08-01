"""SQLite lock-contention helpers shared by the LCM engine.

Isolated from ``engine.py`` (WS5 seam): lock-contention detection, bounded
``busy_timeout`` changes, and transaction-preserving savepoints are pure SQLite
concerns with no engine state. Callers keep their own policy constants (for
example the session-end timeout budget).
"""

from __future__ import annotations

import logging
import sqlite3
import uuid
from contextlib import contextmanager
from typing import Callable, Iterator, List, TypeVar


_T = TypeVar("_T")
logger = logging.getLogger(__name__)


def _is_sqlite_locked_error(exc: BaseException) -> bool:
    """Return True when an exception chain represents SQLite lock contention."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = str(current).lower()
        if isinstance(current, sqlite3.Error) and "locked" in message:
            return True
        current = current.__cause__ or current.__context__
    return False


def _is_sqlite_busy_snapshot_error(exc: BaseException) -> bool:
    """Return True only for SQLite's stale read-snapshot write-upgrade error."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if (
            isinstance(current, sqlite3.Error)
            and getattr(current, "sqlite_errorcode", None)
            == sqlite3.SQLITE_BUSY_SNAPSHOT
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def _run_sqlite_write_with_snapshot_retry(
    conn: sqlite3.Connection,
    operation: Callable[[], _T],
    *,
    operation_name: str,
) -> _T:
    """Run one write transaction, retrying one stale-snapshot upgrade.

    ``SQLITE_BUSY_SNAPSHOT`` cannot be fixed by waiting: the connection must
    roll back its old read snapshot and begin the write from current state.
    Retrying the complete transaction once is safe for bounded deterministic
    LCM writes. Ordinary ``SQLITE_BUSY`` already receives SQLite's configured
    busy timeout and is not multiplied here.
    """
    for attempt in range(2):
        try:
            with conn:
                return operation()
        except sqlite3.Error as exc:
            # ``Connection.__exit__`` normally rolled back already. Keep this
            # explicit so every error path leaves a reusable clean connection.
            if conn.in_transaction:
                conn.rollback()
            if attempt == 0 and _is_sqlite_busy_snapshot_error(exc):
                continue
            if _is_sqlite_locked_error(exc):
                logger.warning(
                    "SQLite write lock recovery exhausted "
                    "(operation=%s, code=%s, name=%s)",
                    operation_name,
                    getattr(exc, "sqlite_errorcode", None),
                    getattr(exc, "sqlite_errorname", None),
                )
            raise
    raise AssertionError("unreachable SQLite write retry state")


def _sqlite_busy_timeout_ms(conn: sqlite3.Connection) -> int:
    row = conn.execute("PRAGMA busy_timeout").fetchone()
    return int(row[0]) if row and row[0] is not None else 0


@contextmanager
def _sqlite_savepoint(conn: sqlite3.Connection) -> Iterator[None]:
    """Isolate helper writes without taking ownership of a caller transaction."""
    # UUID hex contains only identifier-safe characters and keeps every nested
    # helper's SAVEPOINT name unique with a fixed upper bound on name length.
    name = f"lcm_{uuid.uuid4().hex}"
    conn.execute(f"SAVEPOINT {name}")
    try:
        yield
    except BaseException:
        try:
            conn.execute(f"ROLLBACK TO SAVEPOINT {name}")
        finally:
            conn.execute(f"RELEASE SAVEPOINT {name}")
        raise
    else:
        conn.execute(f"RELEASE SAVEPOINT {name}")


@contextmanager
def _temporary_sqlite_busy_timeout(
    connections: List[sqlite3.Connection | None],
    timeout_ms: int,
) -> Iterator[None]:
    """Temporarily bound SQLite lock waits for gateway-critical paths."""
    bounded_timeout = max(0, int(timeout_ms))
    originals: list[tuple[sqlite3.Connection, int]] = []
    for conn in connections:
        if conn is None:
            continue
        original = _sqlite_busy_timeout_ms(conn)
        conn.execute(f"PRAGMA busy_timeout={bounded_timeout}")
        originals.append((conn, original))
    try:
        yield
    finally:
        for conn, original in reversed(originals):
            conn.execute(f"PRAGMA busy_timeout={original}")
