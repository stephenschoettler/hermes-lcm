"""SQLite lock-contention helpers shared by the LCM engine.

Isolated from ``engine.py`` (WS5 seam): lock-contention detection, bounded
``busy_timeout`` changes, and transaction-preserving savepoints are pure SQLite
concerns with no engine state. Callers keep their own policy constants (for
example the session-end timeout budget).
"""

from __future__ import annotations

import errno
import functools
import os
from pathlib import Path
import sqlite3
import stat
import threading
import uuid
from contextlib import contextmanager
from typing import Iterator, List


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


class StoreClosed(sqlite3.Error):
    """The connection was closed; the call never touched SQLite.

    Subclasses ``sqlite3.Error`` so existing fail-open ``except sqlite3.Error`` callers keep working.
    """


class GuardedConnection:
    """A ``check_same_thread=False`` connection whose every use and whose ``close()`` share one lock.

    Closing a sqlite3 connection while another thread is inside ``execute``/``fetch`` on it is a
    use-after-free in ``_sqlite3`` that can crash the interpreter. Every long-lived LCM connection
    is wrapped at ``sqlite3.connect`` so the rule holds for store methods and for raw
    ``store.connection`` users alike: a call waits for an in-flight one, and once ``close()`` ran
    every call raises :class:`StoreClosed`. The lock is a leaf: nothing acquires another lock
    while holding it.
    """

    __slots__ = ("_raw", "_guard_lock", "_closed")

    def __init__(self, raw: sqlite3.Connection) -> None:
        object.__setattr__(self, "_raw", raw)
        object.__setattr__(self, "_guard_lock", threading.RLock())
        object.__setattr__(self, "_closed", False)

    def _live(self) -> sqlite3.Connection:
        if self._closed:
            raise StoreClosed("LCM sqlite connection is closed")
        return self._raw

    def _call(self, name: str, *args, **kwargs):
        with self._guard_lock:
            result = getattr(self._live(), name)(*args, **kwargs)
            return GuardedCursor(result, self) if isinstance(result, sqlite3.Cursor) else result

    def __getattr__(self, name: str):
        if name in GuardedConnection.__slots__:  # half-built instance: never recurse
            raise AttributeError(name)
        with self._guard_lock:
            value = getattr(self._live(), name)
        if callable(value):
            return functools.partial(self._call, name)
        return value

    def __setattr__(self, name: str, value) -> None:
        with self._guard_lock:
            setattr(self._live(), name, value)

    def __enter__(self):
        with self._guard_lock:
            self._live().__enter__()
        return self

    def __exit__(self, *exc):
        with self._guard_lock:
            return self._live().__exit__(*exc)

    def backup(self, target, *args, **kwargs):
        if isinstance(target, GuardedConnection):
            with target._guard_lock:
                return self._call("backup", target._live(), *args, **kwargs)
        return self._call("backup", target, *args, **kwargs)

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        with self._guard_lock:
            if self._closed:
                return
            object.__setattr__(self, "_closed", True)
            self._raw.close()


class GuardedCursor:
    """A cursor from a :class:`GuardedConnection`; fetching runs under the connection's lock."""

    __slots__ = ("_raw", "_conn")

    def __init__(self, raw: sqlite3.Cursor, conn: GuardedConnection) -> None:
        self._raw = raw
        self._conn = conn

    def _call(self, name: str, *args, **kwargs):
        with self._conn._guard_lock:
            self._conn._live()
            result = getattr(self._raw, name)(*args, **kwargs)
            return self if result is self._raw else result

    def __getattr__(self, name: str):
        with self._conn._guard_lock:
            self._conn._live()
            value = getattr(self._raw, name)
        if callable(value):
            return functools.partial(self._call, name)
        return value

    def __iter__(self):
        return self

    def __next__(self):
        with self._conn._guard_lock:
            self._conn._live()
            return next(self._raw)


def guard_connection(raw: sqlite3.Connection) -> GuardedConnection:
    """Wrap a long-lived shared connection; see :class:`GuardedConnection`."""
    return GuardedConnection(raw)
