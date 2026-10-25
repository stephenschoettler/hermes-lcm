"""Keep ``lcm.db``'s live WAL generation from being unlinked under open connections.

The corruption loop this fixes (reproduced deterministically 2026-10-18)
-----------------------------------------------------------------------
SQLite protects a live WAL generation with two POSIX advisory locks: SHARED on
the main file's lock range and the DMS byte of ``-shm``. A sibling process may
checkpoint and unlink ``-wal``/``-shm`` at its close only after taking both
EXCLUSIVE, so an open connection normally refuses that unlink.

POSIX advisory locks are owned per *process and inode*, not per descriptor
(sqlite.org/howtocorrupt.html 2.2). Any ``open()``/``close()`` of the database
or its ``-shm`` anywhere inside the holder process silently cancels ALL of that
process's locks on the file. This plugin does exactly that, constantly: the
read-only probe connections in ``command.py``, ``retrieval_core.py``,
``maintenance.py`` and the doctor checks each open and close their own
descriptor on the same inode, in the same process that holds the long-lived
store connections. Every one of them disarms those stores.

Once disarmed, the next sibling process's close (a gateway restart is the
common one) takes EXCLUSIVE, checkpoints and unlinks ``-wal``/``-shm`` while
the surviving processes still hold them open. Those survivors keep committing
into a now-deleted WAL inode, while the next process to open the database mints
a fresh generation on the same main file. Two WAL generations then checkpoint
into one ``lcm.db``: pages get claimed twice (observed byte-level on 2026-10-04
as metadata root page 6's rightmost child pointing at page 2753, an un-freed
overflow page holding ``messages.content``), and the database reads back as
``disk I/O error`` / ``vtable constructor failed: nodes_fts``.

Why the earlier fixes did not hold: ``mmap_size=0`` (corruption 4) and the
``CrossProcessWriteGate`` flock (corruption 6) both target concurrent *writes*.
SQLite already serializes multi-process writers correctly through a *shared*
WAL; the failure here is the shared WAL being destroyed, so neither fix could
stop it.

The fix
-------
Re-apply the same two lock ranges as *open file description* locks
(``F_OFD_SETLK``) on the descriptors SQLite itself holds. OFD locks belong to
the description rather than the process, so a stray ``close()`` elsewhere
cannot cancel them, they conflict with a foreign EXCLUSIVE exactly like
SQLite's own, and they die with the connection's own descriptor, so there is
nothing to retire. This is the mechanism Hermes core already uses to protect
``state.db`` (``hermes_state_lockguard``); it is reused here rather than
reimplemented.

Scope: readers and writers are unaffected (the guard duplicates the read-lock
geometry every reader already holds). Only the close-time EXCLUSIVE that
unlinks the sidecars is refused, which is the entire intent. The plugin issues
only ``wal_checkpoint(PASSIVE)``, which does not need those locks.

Rollback: ``LCM_DISABLE_WAL_GENERATION_GUARD=1`` in the environment restores
the previous (unguarded) behaviour with no code change. No on-disk state is
created, so there is nothing to clean up.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import sys
import threading

logger = logging.getLogger(__name__)

_DISABLED = bool(os.environ.get("LCM_DISABLE_WAL_GENERATION_GUARD", ""))

def _load_core_lockguard():
    """Import Hermes core's ``hermes_state_lockguard``.

    In production the plugin runs inside the gateway/webui/agent processes, so
    the core checkout is already on ``sys.path`` and the plain import wins.
    Tooling and this plugin's own test suite import the module with a bare
    interpreter, where it is not. Falling back to a path derived from this
    file's location (``<hermes home>/plugins/hermes-lcm`` ->
    ``<hermes home>/hermes-agent``) matters because a failed import does not
    raise: it silently disables the guard and lets the corruption class back
    in. Returns None only when core genuinely is not installed beside us.
    """
    try:
        import hermes_state_lockguard as module
        return module
    except ImportError:
        pass

    core = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "hermes-agent",
    )
    if not os.path.isfile(os.path.join(core, "hermes_state_lockguard.py")):
        return None
    if core not in sys.path:
        sys.path.append(core)
    try:
        import hermes_state_lockguard as module
        return module
    except ImportError:  # pragma: no cover - core present but unimportable
        logger.warning(
            "LCM WAL generation guard: found %s but could not import it; "
            "sidecar protection is OFF", core,
        )
        return None


_lockguard = _load_core_lockguard()


def supported() -> bool:
    """True when the OFD-lock guard can actually be armed on this host."""
    return not _DISABLED and _lockguard is not None and _lockguard.supported()


def main_database_path(conn: sqlite3.Connection) -> str | None:
    """Filesystem path backing ``main`` on *conn*, or None when there is none.

    Taken from the connection itself so every call site is covered without
    threading a path through ``configure_connection``'s signature. Returns None
    for ``:memory:`` and temporary databases, which report an empty file.
    """
    try:
        for _seq, name, filename in conn.execute("PRAGMA database_list"):
            if name == "main":
                return filename or None
    except sqlite3.Error:
        return None
    return None


# Guard records keyed by ``id(connection)``. sqlite3.Connection is not
# weakref-able, so the owning store hands the connection back through
# ``disarm()`` in its ``close()``. A connection dropped without close leaves one
# small dict entry behind; the OFD locks themselves die with its descriptors, so
# the kernel state is always correct even when the bookkeeping is not.
_held_by_conn: "dict[int, dict]" = {}
_held_lock = threading.Lock()


def arm(conn: sqlite3.Connection) -> bool:
    """Pin the WAL generation behind *conn* so no foreign close can unlink it.

    Idempotent and cheap: re-arming an already-locked description is a no-op at
    the kernel level. Safe to call on every connection setup. Returns True when
    the guard is now held for this database.
    """
    if not supported() or _lockguard is None:
        return False
    db_path = main_database_path(conn)
    if db_path is None:
        return False

    # The -shm file is what carries the DMS byte, and SQLite only mints it once
    # the connection actually enters the WAL index. A connection configured but
    # not yet read from may have none, which would leave half the guard unarmed.
    if not os.path.exists(db_path + "-shm"):
        try:
            conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
        except sqlite3.Error:
            pass

    with _held_lock:
        # Re-arming an existing connection extends its own record rather than
        # taking a second handle, so the counted release in disarm() stays
        # balanced however many times configure_connection runs on one handle.
        record = _held_by_conn.get(id(conn))
        held = _lockguard.hold(db_path, record)
        if not held:
            logger.debug("LCM WAL generation guard could not arm on %s", db_path)
            return False
        _held_by_conn[id(conn)] = held
        return True


def disarm(conn: sqlite3.Connection) -> None:
    """Drop *conn*'s claim, immediately before the owning store closes it.

    This is what lets a genuine last close still end the WAL generation
    normally (checkpoint and reset the sidecars). Core's counted release keeps
    the ranges locked while any other connection in this process still holds
    them, so one store closing never exposes the others. Call it before
    ``conn.close()``, never after: once the descriptors are gone there is
    nothing left to hand back.
    """
    if _lockguard is None:
        return
    with _held_lock:
        held = _held_by_conn.pop(id(conn), None)
    if held:
        _lockguard.release(held)
