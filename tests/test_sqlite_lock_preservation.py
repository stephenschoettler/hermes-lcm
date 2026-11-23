"""Regression: permission tightening must not cancel live SQLite locks.

Opening a database file or its ``-wal``/``-shm``/``-journal`` sidecar with a
readable/writable descriptor and closing it cancels every POSIX advisory lock
this process holds on that file (SQLite "How To Corrupt An SQLite Database
File", section 2.2). Losing those locks lets a second process take the
WAL-index lock underneath a live connection, which is how a shared LCM database
can desynchronize and abort a process with SIGBUS.
"""

from __future__ import annotations

import os
from pathlib import Path
import sqlite3
import stat
import subprocess
import sys
import textwrap

import pytest

import hermes_lcm.sqlite_util as sqlite_util_module

pytestmark = pytest.mark.skipif(os.name != "posix", reason="requires POSIX advisory locks")

_CHILD_ATTEMPT = textwrap.dedent(
    """
    import sqlite3, sys
    conn = sqlite3.connect(sys.argv[1], timeout=0.05, isolation_level=None)
    try:
        conn.execute("BEGIN IMMEDIATE")
        print("ACQUIRED")
        conn.execute("ROLLBACK")
    except sqlite3.OperationalError as exc:
        print("BLOCKED" if "locked" in str(exc).lower() else "ERROR:" + str(exc))
    finally:
        conn.close()
    """
)


def _attempt_write_lock(db_path: Path) -> str:
    return subprocess.run(
        [sys.executable, "-c", _CHILD_ATTEMPT, str(db_path)],
        capture_output=True, text=True, check=True, timeout=30,
    ).stdout.strip()


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_restricting_artifacts_preserves_held_write_lock(tmp_path):
    db_path = tmp_path / "lcm.db"
    holder = sqlite3.connect(db_path, isolation_level=None)
    holder.execute("PRAGMA journal_mode=WAL")
    holder.execute("CREATE TABLE messages (content TEXT)")
    holder.execute("INSERT INTO messages VALUES ('retained')")
    holder.execute("BEGIN IMMEDIATE")

    shm_path = db_path.with_name(db_path.name + "-shm")
    wal_path = db_path.with_name(db_path.name + "-wal")
    for artifact in (db_path, shm_path, wal_path):
        assert artifact.exists(), artifact
        artifact.chmod(0o644)

    try:
        assert _attempt_write_lock(db_path) == "BLOCKED"
        sqlite_util_module._restrict_existing_sqlite_artifacts(db_path)
        # The holder's locks must survive the permission pass.
        assert _attempt_write_lock(db_path) == "BLOCKED"
        assert holder.execute("SELECT content FROM messages").fetchone()[0] == "retained"
        for artifact in (db_path, shm_path, wal_path):
            assert _mode(artifact) == 0o600, artifact
    finally:
        holder.execute("ROLLBACK")
        holder.close()


def test_restricting_artifacts_does_not_open_existing_artifacts_for_io(tmp_path, monkeypatch):
    """The guard is structural: no O_RDWR/O_RDONLY open of an existing artifact."""
    db_path = tmp_path / "lcm.db"
    holder = sqlite3.connect(db_path, isolation_level=None)
    holder.execute("PRAGMA journal_mode=WAL")
    holder.execute("CREATE TABLE messages (content TEXT)")
    holder.execute("INSERT INTO messages VALUES ('retained')")
    holder.commit()

    watched = {db_path.name, db_path.name + "-wal", db_path.name + "-shm"}
    opened_for_io = []
    real_open = os.open

    def recording_open(path, flags, *args, **kwargs):
        if isinstance(path, str) and path in watched and not flags & os.O_PATH:
            opened_for_io.append((path, flags))
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(sqlite_util_module.os, "open", recording_open)
    try:
        sqlite_util_module._restrict_existing_sqlite_artifacts(db_path)
        assert opened_for_io == []
        assert holder.execute("SELECT content FROM messages").fetchone()[0] == "retained"
    finally:
        holder.close()


def test_restricting_artifacts_still_refuses_symlinked_and_hardlinked_sidecars(tmp_path):
    for suffix in ("-wal", "-shm", "-journal"):
        db_path = tmp_path / f"linked{suffix}.db"
        holder = sqlite3.connect(db_path)
        holder.execute("CREATE TABLE messages (content TEXT)")
        holder.commit()
        holder.close()

        target = tmp_path / f"unrelated{suffix}.txt"
        target.write_text("shared", encoding="utf-8")
        target.chmod(0o644)
        sidecar = db_path.with_name(db_path.name + suffix)
        sidecar.unlink(missing_ok=True)
        sidecar.symlink_to(target)
        with pytest.raises(OSError, match="SQLite artifact"):
            sqlite_util_module._restrict_existing_sqlite_artifacts(db_path)
        assert _mode(target) == 0o644

        sidecar.unlink()
        os.link(target, sidecar)
        with pytest.raises(OSError, match="SQLite artifact"):
            sqlite_util_module._restrict_existing_sqlite_artifacts(db_path)
        assert _mode(target) == 0o644