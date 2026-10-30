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
import stat
import time
import uuid
from contextlib import contextmanager
from functools import wraps
from typing import Iterator, List


class InitializationBudget:
    """One explicit, monotonic budget shared by a storage bind, never ambient.

    Five seconds leaves room inside the host's default ten-second import/load
    window. Hosts exposing a load deadline can supply the actual remaining time.
    Cancellation is polled between statements and every 1000 SQLite VM steps.
    """

    def __init__(self, seconds=5.0, *, deadline=None, cancelled=None):
        self.deadline = min(time.monotonic() + seconds, deadline if deadline is not None else float('inf'))
        self.cancelled = cancelled or (lambda: False)

    def expired(self):
        return self.cancelled() or time.monotonic() >= self.deadline

    def check(self):
        if self.expired():
            raise TimeoutError("LCM storage initialization deadline/cancellation reached")


class _InitializingConnection(sqlite3.Connection):
    _initialization_budget = None
    _progress_handler = (None, 0)

    def set_progress_handler(self, callback, n, /):
        super().set_progress_handler(callback, n)
        self._progress_handler = (callback, n)

    def execute(self, sql, parameters=(), /):
        budget = self._initialization_budget
        if budget is not None:
            # Cleanup must still work after cancellation. busy_timeout=0 avoids
            # an uninterruptible SQLite busy-handler sleep; retry owns a fresh
            # connection and rechecks the entire initialization attempt.
            cleanup = sql.lstrip().upper().startswith(("ROLLBACK", "RELEASE"))
            timeout_setting = sql.strip().lower().startswith("pragma busy_timeout=")
            if not cleanup and not timeout_setting:
                budget.check()
            if timeout_setting:
                sql = "PRAGMA busy_timeout=0"
        return super().execute(sql, parameters)

    def commit(self):
        if self._initialization_budget is not None:
            self._initialization_budget.check()
        return super().commit()


def execute_statements(conn, script):
    """Execute DDL including trigger bodies without an implicit COMMIT."""
    statement = ""
    cursor = None
    for char in script:
        statement += char
        if char == ";" and sqlite3.complete_statement(statement):
            cursor = conn.execute(statement)
            statement = ""
    if statement.strip():
        cursor = conn.execute(statement)
    return cursor


def connect_initializing(owner, *args, **kwargs):
    kwargs['factory'] = _InitializingConnection
    kwargs['timeout'] = 0
    conn = sqlite3.connect(*args, **kwargs)
    # Publish before any SQL, so even an interrupt during setup gets closed.
    owner._conn = conn
    budget = owner._init_budget
    conn._initialization_budget = budget
    conn.execute("PRAGMA busy_timeout=0")
    conn.set_progress_handler(lambda: int(budget.expired()), 1000)
    return conn


def storage_initialization(init):
    """Failure-atomic constructor, with fresh-connection retries for BUSY only."""
    @wraps(init)
    def wrapped(self, *args, init_budget=None, **kwargs):
        budget = init_budget or InitializationBudget()
        self._init_budget = budget
        while True:
            budget.check()
            try:
                init(self, *args, **kwargs)
                budget.check()
                conn = self._conn
                conn._initialization_budget = None
                conn.set_progress_handler(None, 0)
                conn.execute("PRAGMA busy_timeout=30000")
                self._init_budget = None
                return
            except BaseException as exc:
                conn = getattr(self, '_conn', None)
                if conn is not None:
                    try:
                        conn._initialization_budget = None
                        conn.set_progress_handler(None, 0)
                        conn.close()
                    except BaseException:
                        pass  # cleanup must never replace the causal exception
                    self._conn = None
                from .db_bootstrap import _is_sqlite_lock_error
                if (not _is_sqlite_lock_error(exc) or budget.expired()
                        or getattr(self, '_INIT_LOCK_RETRY_BUDGET_S', None) == 0):
                    raise
                time.sleep(min(0.01, max(0, budget.deadline - time.monotonic())))
    return wrapped


@contextmanager
def owned_write(conn, *, immediate=False, begin=None):
    """Commit our transaction; isolate a caller's transaction with a savepoint."""
    if conn.in_transaction:
        with _sqlite_savepoint(conn):
            yield
        return
    try:
        if begin is not None:
            begin()
        else:
            conn.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
        yield
        conn.commit()
    except BaseException:
        try:
            conn.rollback()
        except BaseException:
            pass
        raise


@contextmanager
def bounded_sqlite_owners(owners, timeout_ms):
    """Serialize temporary settings with connection owners and share one clock."""
    deadline = time.monotonic() + timeout_ms / 1000
    acquired = []
    handlers = []
    try:
        for conn, lock in owners:
            if not lock.acquire(timeout=max(0, deadline - time.monotonic())):
                raise sqlite3.OperationalError("database is locked: connection owner busy")
            acquired.append(lock)
            if isinstance(conn, _InitializingConnection):
                callback, steps = conn._progress_handler
                handlers.append((conn, callback, steps))
                conn.set_progress_handler(
                    lambda cb=callback: int(time.monotonic() >= deadline or (cb is not None and cb())),
                    min(1000, steps) if steps else 1000,
                )
        with _temporary_sqlite_busy_timeout(
            [conn for conn, _ in owners], max(0, int((deadline - time.monotonic()) * 1000))
        ):
            yield
    finally:
        for conn, callback, steps in reversed(handlers):
            conn.set_progress_handler(callback, steps)
        for lock in reversed(acquired):
            lock.release()

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

# The ``os.supports_*`` sets are keyed by the original builtin; keep a handle on
# it so the capability probe stays correct even if ``os.chmod`` is wrapped.
_OS_CHMOD = os.chmod


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
    raise _sqlite_artifact_error(path, "directory entry changed while restricting permissions")


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


def _create_private_sqlite_file(path: Path, *, directory_fd: int) -> bool:
    """Create a brand-new, empty, mode-0600 SQLite file; return False if it raced into existence.

    This is the ONE place a descriptor is opened on a SQLite artifact, and it is
    safe because of ``O_EXCL``: the call only succeeds when the directory entry
    did not exist, so the inode is new and no process (this one included) can
    hold a SQLite lock on it yet. Closing our descriptor therefore cannot
    cancel anyone's locks (see ``_chmod_sqlite_artifact_at``).
    """
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    flags |= getattr(os, "O_BINARY", 0)
    try:
        fd = os.open(path.name, flags, 0o600, dir_fd=directory_fd)
    except FileExistsError:
        return False
    try:
        created = os.fstat(fd)
        _validate_sqlite_artifact(path, created)
        # The creation mode is subject to the umask; make the result exact.
        os.fchmod(fd, 0o600)
    finally:
        os.close(fd)
    return True


def _chmod_existing_artifact_without_descriptor(path: Path, *, directory_fd: int) -> None:
    """chmod 0600 an existing directory entry without ever opening it.

    Prefer the kernel-enforced no-follow form (``fchmodat(AT_SYMLINK_NOFOLLOW)``,
    available on macOS/BSD): a symlink swapped in after validation is never
    followed. Where the platform lacks it (Linux), fall back to a plain
    ``fchmodat``; the caller has just lstat-validated a single-link regular
    file and re-validates the same entry afterwards.
    """
    if _OS_CHMOD in os.supports_dir_fd and _OS_CHMOD in os.supports_follow_symlinks:
        try:
            os.chmod(path.name, 0o600, dir_fd=directory_fd, follow_symlinks=False)
            return
        except NotImplementedError:
            pass
        except OSError as exc:
            if exc.errno not in (errno.ENOTSUP, errno.EOPNOTSUPP):
                raise
    os.chmod(path.name, 0o600, dir_fd=directory_fd)


def _chmod_sqlite_artifact_at(
    path: Path,
    *,
    directory_fd: int,
    create: bool,
    allow_sidecar_disappearance: bool = False,
) -> bool:
    """Make one SQLite artifact private WITHOUT opening a descriptor on an existing file.

    POSIX advisory locks are owned by the process, not by a descriptor:
    ``close()`` of ANY descriptor for a file releases EVERY lock the process
    holds on it (https://www.sqlite.org/howtocorrupt.html, section 2.2),
    including the locks SQLite's own live connections hold on the database and
    on the ``-shm`` WAL index. This runs from ``LCMStore._init_db`` and
    maintenance inside processes that already hold connections to the same
    database, so an ``open()``/``close()`` here would let a sibling process take
    the exclusive lock on its own close, checkpoint, and unlink the
    ``-wal``/``-shm`` the first process is still using; later openers then
    create a second WAL generation and two generations write one file.

    So an existing artifact is only ever inspected with ``lstat`` (no
    descriptor) and tightened with path-relative ``chmod`` (no descriptor);
    this holds whether or not the mode already is 0600. The still-checked
    invariants are the original ones: the entry must be a single-link regular
    file (never a symlink or hardlink), and it is re-validated after the chmod.

    Residual window versus a descriptor-based ``fchmod``: validate and chmod
    are two syscalls on the same *name*, not one atomic operation on one inode.
    A principal able to swap the entry in between must be able to write the
    directory, which ``_open_private_sqlite_directory`` has already required to
    be closed to group/other. Where the OS supports no-follow chmod a swapped-in
    symlink is never followed; a swapped-in hardlink or replacement is detected
    by the re-validation below (after the fact, as ``fchmod`` already was for
    link-count changes).
    """
    try:
        expected = os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        if not create:
            return False
        if _create_private_sqlite_file(path, directory_fd=directory_fd):
            return True
        # Lost a creation race: the file exists now, so treat it as existing.
        return _chmod_sqlite_artifact_at(
            path,
            directory_fd=directory_fd,
            create=False,
            allow_sidecar_disappearance=allow_sidecar_disappearance,
        )

    _validate_sqlite_artifact(path, expected)
    if stat.S_IMODE(expected.st_mode) == 0o600:
        return True  # nothing to tighten; no syscall beyond the lstat

    try:
        _chmod_existing_artifact_without_descriptor(path, directory_fd=directory_fd)
        restricted = os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        if not allow_sidecar_disappearance:
            raise
        _require_sqlite_artifact_absent(path, directory_fd=directory_fd)
        return False
    if not _same_file_identity(expected, restricted):
        raise _sqlite_artifact_error(path, "directory entry changed while restricting permissions")
    _validate_sqlite_artifact(path, restricted)
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
        if isinstance(current, sqlite3.Error):
            code = getattr(current, "sqlite_errorcode", None)
            if isinstance(code, int) and (code & 0xff) in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
                return True
            if "locked" in message or message in {"database is busy", "database table is busy", "database schema is busy"}:
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
            conn.execute(f"RELEASE SAVEPOINT {name}")
        except BaseException:
            pass
        raise
    else:
        conn.execute(f"RELEASE SAVEPOINT {name}")


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
    Once the lock looks free, keep no-wait mode for the caller's block so
    another writer winning the reservation race cannot start a busy sleep.

    If the deadline passes while the lock is still held, do NOT raise from the
    context manager: enter the block with ``busy_timeout=0`` so the caller's
    own first statement fails immediately with "database is locked". That keeps
    the failure attributable to the specific operation the caller was running,
    so per-step diagnostics and recovery branches stay intact, rather than
    collapsing every lock loss into one generic error at the wrong layer.

    A connection already inside a transaction is not probed; its owner's
    transaction scope is preserved. Callers sharing a connection must hold its
    owning lock while temporary settings are installed.
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
            _wait_for_write_lock(conn, deadline)
            # Still locked at the deadline: leave busy_timeout at 0 so the
            # caller's own statement fails fast instead of waiting again.
            # The reservation probe is advisory: another writer can win after
            # ROLLBACK. Keep no-wait mode in the body as well, closing the old
            # race into an uninterruptible busy-handler sleep.
            conn.execute("PRAGMA busy_timeout=0")
    except BaseException:
        for conn, original in reversed(originals):
            conn.execute(f"PRAGMA busy_timeout={original}")
        raise
    try:
        yield
    finally:
        for conn, original in reversed(originals):
            conn.execute(f"PRAGMA busy_timeout={original}")


def _wait_for_write_lock(conn: sqlite3.Connection, deadline: float, *, retain: bool = False) -> bool:
    """Poll until ``conn`` can take the write lock, or ``deadline`` passes.

    Returns True when the lock was observed free, False when the deadline
    passed with it still held. By default the probe transaction is released.
    ``retain=True`` is for an owning transaction scope that commits/rolls back
    the reservation itself; it closes the probe-to-BEGIN contention race.
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
        if not retain:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.OperationalError:
                pass
        return True
