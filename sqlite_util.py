"""SQLite lock-contention helpers shared by the LCM engine.

Isolated from ``engine.py`` (WS5 seam): lock-contention detection, bounded
``busy_timeout`` changes, and transaction-preserving savepoints are pure SQLite
concerns with no engine state. Callers keep their own policy constants (for
example the session-end timeout budget).
"""

from __future__ import annotations

import errno
import os
from pathlib import Path
import sqlite3
import time
import stat
import uuid
from contextlib import contextmanager
from typing import Iterator, List

# Poll granularity while waiting for a write lock inside a bounded window.
# Zero, deliberately. time.sleep() cannot be trusted at this scale on this
# platform: requesting 1ms measured ~42ms and 2ms measured ~106ms here, so ANY
# nonzero sleep can blow a 50ms budget in a single poll. sleep(0) yields the
# GIL without arming a timer.
#
# Tradeoff: this makes the wait a bounded busy-spin (repeated BEGIN IMMEDIATE)
# rather than a blocking sleep. Acceptable only because the window is small and
# explicitly budgeted by the caller — do NOT reuse this helper for long waits.
# A multi-second budget should sleep between probes and tolerate the overshoot.
_LOCK_POLL_INTERVAL_S = 0.0


_SQLITE_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")


def _sqlite_artifact_error(path: Path, reason: str) -> OSError:
    return OSError(errno.EPERM, f"refusing SQLite artifact {path.name!r}: {reason}", str(path))


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _validate_sqlite_artifact(path: Path, file_stat: os.stat_result) -> None:
    if not stat.S_ISREG(file_stat.st_mode):
        raise _sqlite_artifact_error(path, "not a regular file")
    if file_stat.st_nlink != 1:
        raise _sqlite_artifact_error(path, "link count is not one")


def _require_sqlite_artifact_absent(path: Path, *, directory_fd: int) -> None:
    """Accept a vanished sidecar only while its directory entry stays absent."""
    try:
        current = os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    _validate_sqlite_artifact(path, current)
    raise _sqlite_artifact_error(path, "directory entry changed while opening")


def _open_private_sqlite_directory(path: Path) -> int:
    directory = path.parent
    expected = os.stat(directory, follow_symlinks=False)
    if not stat.S_ISDIR(expected.st_mode):
        raise _sqlite_artifact_error(path, "parent is not a regular directory")
    if expected.st_mode & 0o022:
        raise _sqlite_artifact_error(path, "parent directory is writable by another user")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_BINARY", 0)
    directory_fd = os.open(directory, flags)
    opened = os.fstat(directory_fd)
    if not stat.S_ISDIR(opened.st_mode) or not _same_file_identity(expected, opened):
        os.close(directory_fd)
        raise _sqlite_artifact_error(path, "parent directory changed while opening")
    return directory_fd


def _chmod_sqlite_artifact_at(
    path: Path,
    *,
    directory_fd: int,
    create: bool,
    allow_sidecar_disappearance: bool = False,
) -> bool:
    expected: os.stat_result | None
    try:
        expected = os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        if not create:
            return False
        expected = None
    if expected is not None:
        _validate_sqlite_artifact(path, expected)

    flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
    if expected is None:
        flags |= os.O_CREAT | os.O_EXCL
    try:
        fd = os.open(path.name, flags, 0o600, dir_fd=directory_fd)
    except FileExistsError:
        if expected is not None:
            raise
        return _chmod_sqlite_artifact_at(
            path,
            directory_fd=directory_fd,
            create=False,
        )
    except FileNotFoundError:
        if not allow_sidecar_disappearance:
            raise
        _require_sqlite_artifact_absent(path, directory_fd=directory_fd)
        return False
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode):
            raise _sqlite_artifact_error(path, "not a regular file")
        if expected is not None and not _same_file_identity(expected, opened):
            raise _sqlite_artifact_error(path, "directory entry changed while opening")
        if opened.st_nlink == 0 and allow_sidecar_disappearance:
            _require_sqlite_artifact_absent(path, directory_fd=directory_fd)
            return False
        _validate_sqlite_artifact(path, opened)
        os.fchmod(fd, 0o600)
        restricted = os.fstat(fd)
        if restricted.st_nlink == 0 and allow_sidecar_disappearance:
            _require_sqlite_artifact_absent(path, directory_fd=directory_fd)
            return False
        if restricted.st_nlink != 1:
            raise _sqlite_artifact_error(path, "link count changed while restricting permissions")
    finally:
        os.close(fd)
    return True


def _restrict_existing_sqlite_artifacts(db_path: Path) -> None:
    """Restrict verified, single-link SQLite files without following links."""
    if os.name != "posix":  # pragma: no cover - Windows compatibility fallback
        for artifact in (
            db_path,
            *(db_path.with_name(db_path.name + suffix) for suffix in _SQLITE_SIDECAR_SUFFIXES),
        ):
            try:
                artifact.chmod(0o600)
            except FileNotFoundError:
                continue
        return

    directory_fd = _open_private_sqlite_directory(db_path)
    try:
        _chmod_sqlite_artifact_at(
            db_path,
            directory_fd=directory_fd,
            create=False,
        )
        for suffix in _SQLITE_SIDECAR_SUFFIXES:
            _chmod_sqlite_artifact_at(
                db_path.with_name(db_path.name + suffix),
                directory_fd=directory_fd,
                create=False,
                allow_sidecar_disappearance=True,
            )
    finally:
        os.close(directory_fd)


def _prepare_private_sqlite_file(path: Path) -> None:
    """Create or tighten one SQLite file and its existing sidecars safely."""
    if os.name != "posix":  # pragma: no cover - Windows compatibility fallback
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
        fd = os.open(path, flags, 0o600)
        try:
            if hasattr(os, "fchmod"):
                os.fchmod(fd, 0o600)
            else:
                path.chmod(0o600)
        finally:
            os.close(fd)
        _restrict_existing_sqlite_artifacts(path)
        return

    directory_fd = _open_private_sqlite_directory(path)
    try:
        _chmod_sqlite_artifact_at(path, directory_fd=directory_fd, create=True)
        for suffix in _SQLITE_SIDECAR_SUFFIXES:
            _chmod_sqlite_artifact_at(
                path.with_name(path.name + suffix),
                directory_fd=directory_fd,
                create=False,
                allow_sidecar_disappearance=True,
            )
    finally:
        os.close(directory_fd)


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


def _wait_for_write_lock(conn: sqlite3.Connection, deadline: float) -> bool:
    """Poll until ``conn`` can take the write lock, or ``deadline`` passes.

    Returns True when the lock was observed free, False when the deadline
    passed with it still held. The probe transaction is always released, so
    this never leaves a transaction open on ``conn``.
    """
    if conn.in_transaction:
        # The caller owns an open transaction; probing would corrupt its scope.
        return True
    while True:
        try:
            conn.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as exc:
            if not _is_sqlite_locked_error(exc):
                raise
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            # Never sleep past the deadline: an unclamped sleep is what makes a
            # short budget overshoot when the OS rounds small sleeps up.
            time.sleep(min(_LOCK_POLL_INTERVAL_S, remaining))
            continue
        try:
            conn.execute("ROLLBACK")
        except sqlite3.OperationalError:
            pass
        return True


@contextmanager
def _temporary_sqlite_busy_timeout(
    connections: List[sqlite3.Connection | None],
    timeout_ms: int,
) -> Iterator[None]:
    """Temporarily bound SQLite lock waits for gateway-critical paths.

    ``PRAGMA busy_timeout`` alone cannot honor a short budget. Measured here
    against a held WAL writer lock, a single statement with ``busy_timeout=50``
    blocked ~850ms — 17x its own budget — deterministically. SQLite's busy
    handler sleeps in a fixed escalating pattern (1, 2, 5, 10, 15, 20, 25ms...)
    and only compares total elapsed time *between* those sleeps, so when short
    sleeps cost far more than requested (they do on this platform) it sails
    past a small budget before it next looks at the clock. Whatever the precise
    cause, the empirical result is what matters: the PRAGMA does not bound a
    sub-100ms window, which defeats the point of bounding a gateway hook.

    A progress handler does not help either — measured, not assumed: it fires
    per VM instruction, and a statement parked in the busy handler runs none.

    So bound the wait in Python instead. Poll for the write lock against a real
    wall-clock deadline with ``busy_timeout=0`` (each probe fails instantly).
    Once the lock looks free, install ``timeout_ms`` and run the caller's block
    normally.

    If the deadline passes while the lock is still held, do NOT raise from the
    context manager: enter the block with ``busy_timeout=0`` so the caller's
    own first statement fails immediately with "database is locked". That keeps
    the failure attributable to the specific operation the caller was running,
    so per-step diagnostics and recovery branches stay intact, rather than
    collapsing every lock loss into one generic error at the wrong layer.

    Residual race: the probe releases the lock before yielding, so another
    writer can take it in between. ``timeout_ms`` stays installed as a backstop
    for that window, which can still overshoot — but the common case this
    guards (a long-lived writer holding the lock) is now genuinely bounded.
    A connection already inside a transaction is left alone, since probing
    would disturb the caller's transaction.
    """
    bounded_timeout = max(0, int(timeout_ms))
    originals: list[tuple[sqlite3.Connection, int]] = []
    live = [conn for conn in connections if conn is not None]
    deadline = time.monotonic() + (bounded_timeout / 1000.0)
    try:
        for conn in live:
            originals.append((conn, _sqlite_busy_timeout_ms(conn)))
            # Probe with no internal wait so the Python clock is authoritative.
            conn.execute("PRAGMA busy_timeout=0")
            acquired = _wait_for_write_lock(conn, deadline)
            # Still locked at the deadline: leave busy_timeout at 0 so the
            # caller's own statement fails fast instead of waiting again.
            conn.execute(f"PRAGMA busy_timeout={bounded_timeout if acquired else 0}")
    except BaseException:
        for conn, original in reversed(originals):
            conn.execute(f"PRAGMA busy_timeout={original}")
        raise
    try:
        yield
    finally:
        for conn, original in reversed(originals):
            conn.execute(f"PRAGMA busy_timeout={original}")
