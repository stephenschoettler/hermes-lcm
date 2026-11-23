"""A live WAL generation must survive descriptor churn and a foreign close.

Regression for the corruption loop diagnosed 2026-10-18.

SQLite protects a live WAL generation with two POSIX advisory locks: SHARED on
the main file's lock range and the DMS byte of ``-shm``. A sibling process may
checkpoint and unlink ``-wal``/``-shm`` at its close only after taking both
EXCLUSIVE, so an open connection normally refuses that unlink.

POSIX advisory locks are owned per *process and inode*, not per descriptor
(sqlite.org/howtocorrupt.html 2.2). Any raw ``open()``/``close()`` of the
database or its ``-shm`` anywhere inside the holder process silently cancels
ALL of that process's locks on the file, leaving its connections open but
unprotected. The next sibling process's close then takes EXCLUSIVE and unlinks
the sidecars while the survivors still hold them. Those survivors keep
committing into a deleted WAL inode while the next opener mints a fresh
generation on the same main file, so two generations checkpoint into one
``lcm.db``. The damage reads back as ``2nd reference to page N`` under
``integrity_check`` and ``disk I/O error`` to the engine.

SQLite defends its OWN in-process handles against this (it defers descriptor
closes per inode), which is why a probe *connection* is safe and a raw
descriptor is not. ``wal_generation_guard`` closes the remaining hole by
re-taking both ranges as OFD locks, which no foreign close can cancel.
"""

import os
import sqlite3
import subprocess
import sys

import pytest

from hermes_lcm.db_bootstrap import configure_connection
from hermes_lcm import wal_generation_guard

pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="WAL sidecar locking is POSIX-specific"
)

requires_guard = pytest.mark.skipif(
    not wal_generation_guard.supported(),
    reason="OFD locks unavailable on this host/runtime",
)


# A sibling PROCESS that writes and closes. Its close is what checkpoints and
# unlinks the sidecars once the survivor has been disarmed.
_SIBLING = r"""
import sqlite3, sys
conn = sqlite3.connect(sys.argv[1])
conn.execute("PRAGMA busy_timeout=30000")
conn.execute("PRAGMA journal_mode=WAL")
conn.execute("INSERT INTO t(v) VALUES('sibling')")
conn.commit()
conn.close()
"""


def _sidecar_inodes(db_path):
    inodes = {}
    for suffix in ("-wal", "-shm"):
        try:
            inodes[suffix] = os.stat(str(db_path) + suffix).st_ino
        except OSError:
            inodes[suffix] = None
    return inodes


def _churn_raw_descriptors(db_path):
    """What any non-SQLite reader of these paths does: open, then close."""
    for path in (str(db_path), str(db_path) + "-shm"):
        if os.path.exists(path):
            os.close(os.open(path, os.O_RDONLY))


def _run_sibling(db_path):
    done = subprocess.run(
        [sys.executable, "-I", "-c", _SIBLING, str(db_path)],
        capture_output=True, text=True, timeout=120,
    )
    assert done.returncode == 0, done.stderr


def _open_live_store(db_path):
    """A configured connection holding a WAL generation, as a store would."""
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE IF NOT EXISTS t(v TEXT)")
    conn.commit()
    configure_connection(conn)
    conn.execute("INSERT INTO t(v) VALUES('survivor')")
    conn.commit()
    conn.execute("SELECT count(*) FROM t").fetchone()
    return conn


@pytest.fixture
def live_store(tmp_path):
    db_path = tmp_path / "lcm.db"
    conn = _open_live_store(db_path)
    try:
        yield db_path, conn
    finally:
        conn.close()


@requires_guard
def test_generation_survives_descriptor_churn_and_foreign_close(live_store):
    """The corruption path end to end: raw churn disarms, then a foreign close.

    Without the guard this is the exact sequence that strands every live
    process on a deleted WAL inode.
    """
    db_path, conn = live_store
    before = _sidecar_inodes(db_path)
    assert before["-wal"] is not None and before["-shm"] is not None, (
        "configure_connection must leave a WAL generation in place"
    )

    _churn_raw_descriptors(db_path)
    _run_sibling(db_path)

    after = _sidecar_inodes(db_path)
    assert after == before, (
        "the sibling's close ended the generation under a live connection: "
        f"{before} -> {after}. The survivor now writes into a deleted WAL "
        "inode while the next opener mints a fresh one on the same db file."
    )


def test_probe_connection_alone_does_not_end_the_generation(live_store):
    """SQLite's own in-process handling must keep covering probe connections.

    The plugin opens read-only probes constantly (``command.py``,
    ``retrieval_core.py``, the doctor checks). SQLite defers its own descriptor
    closes per inode so these do not cancel the holder's locks. This pins that
    assumption: if it ever regresses, probes become a second strand vector.
    """
    db_path, conn = live_store
    before = _sidecar_inodes(db_path)

    probe = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    probe.execute("SELECT count(*) FROM t").fetchone()
    probe.close()
    _run_sibling(db_path)

    assert _sidecar_inodes(db_path) == before


@requires_guard
def test_committed_rows_survive_churn_and_foreign_close(live_store):
    """The guard must not cost durability, visibility, or integrity."""
    db_path, conn = live_store
    _churn_raw_descriptors(db_path)
    _run_sibling(db_path)

    conn.execute("INSERT INTO t(v) VALUES('after')")
    conn.commit()

    values = {row[0] for row in conn.execute("SELECT v FROM t")}
    assert {"survivor", "sibling", "after"} <= values
    assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


@requires_guard
def test_arm_resolves_the_database_off_the_connection(live_store):
    """No call site passes a path, so ``arm`` must find it itself, repeatably."""
    db_path, conn = live_store
    assert wal_generation_guard.main_database_path(conn) == str(db_path)
    assert wal_generation_guard.arm(conn) is True  # idempotent re-arm


def test_memory_database_is_skipped():
    """A ``:memory:`` db has no sidecars to protect and must not be probed."""
    conn = sqlite3.connect(":memory:")
    try:
        assert wal_generation_guard.main_database_path(conn) is None
        assert wal_generation_guard.arm(conn) is False
    finally:
        conn.close()


def test_kill_switch_restores_unguarded_behaviour(tmp_path, monkeypatch):
    """``LCM_DISABLE_WAL_GENERATION_GUARD=1`` is the documented rollback.

    With the guard off the generation is destroyed again. Asserting that
    failure keeps this a real kill switch rather than a flag nobody checks,
    and pins the bug itself so the suite still describes it.
    """
    monkeypatch.setattr(wal_generation_guard, "_DISABLED", True)
    assert wal_generation_guard.supported() is False

    db_path = tmp_path / "lcm.db"
    conn = _open_live_store(db_path)
    try:
        before = _sidecar_inodes(db_path)
        _churn_raw_descriptors(db_path)
        _run_sibling(db_path)
        assert _sidecar_inodes(db_path) != before
    finally:
        conn.close()
