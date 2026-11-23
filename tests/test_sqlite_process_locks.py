"""Process-level lock regressions for SQLite permission hardening."""

import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest
import hermes_lcm.sqlite_util as util

CHILD = """import sqlite3,sys
c=sqlite3.connect(sys.argv[1],timeout=0.05,isolation_level=None)
try:
 c.execute('BEGIN IMMEDIATE');print('ACQUIRED');c.execute('ROLLBACK')
except sqlite3.OperationalError as e:
 print('BLOCKED' if 'locked' in str(e).lower() else 'ERROR:'+str(e))
finally:c.close()
"""


@pytest.mark.skipif(os.name != "posix", reason="POSIX lock semantics")
@pytest.mark.parametrize("journal", ["WAL", "DELETE"])
@pytest.mark.parametrize(
    "operation", ["_restrict_existing_sqlite_artifacts", "_prepare_private_sqlite_file"]
)
def test_permissions_preserve_live_process_locks(tmp_path, journal, operation):
    db = tmp_path / "probe.db"
    c = sqlite3.connect(db, isolation_level=None)
    try:
        c.execute("PRAGMA journal_mode=" + journal)
        c.execute("CREATE TABLE t(value TEXT)")
        c.execute("INSERT INTO t VALUES('retained')")
        c.execute("BEGIN IMMEDIATE")

        def attempt():
            return subprocess.check_output(
                [sys.executable, "-c", CHILD, str(db)], text=True, timeout=15
            ).strip()

        assert attempt() == "BLOCKED"
        getattr(util, operation)(db)
        assert attempt() == "BLOCKED"
        assert c.execute("SELECT value FROM t").fetchone()[0] == "retained"
        assert db.stat().st_mode & 0o777 == 0o600
        c.execute("ROLLBACK")
    finally:
        c.close()


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin isolation boundary")
def test_parent_does_not_open_sqlite_artifacts(tmp_path, monkeypatch):
    db = tmp_path / "probe.db"
    c = sqlite3.connect(db)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("CREATE TABLE t(x)")
    c.commit()
    original = os.open

    def guarded(path, *args, **kwargs):
        assert Path(path).name not in ["probe.db", "probe.db-wal", "probe.db-shm"]
        return original(path, *args, **kwargs)

    monkeypatch.setattr(util.os, "open", guarded)
    try:
        util._restrict_existing_sqlite_artifacts(db)
    finally:
        c.close()


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin isolation boundary")
def test_permission_worker_failures_fail_closed(tmp_path, monkeypatch):
    db = tmp_path / "probe.db"
    db.touch()

    def fail(*args, **kwargs):
        raise subprocess.TimeoutExpired("permission worker", 15)

    monkeypatch.setattr(util.subprocess, "run", fail)
    with pytest.raises(OSError, match="permission worker"):
        util._restrict_existing_sqlite_artifacts(db)


def test_permission_worker_uses_inode_pinned_path(monkeypatch):
    # The isolated worker holds no SQLite locks, so it must not use the
    # name-based chmod path that cannot pin the inode between stat and chmod.
    monkeypatch.setattr(util, "_IN_PERMISSION_WORKER", True)
    assert util._sqlite_artifact_supports_descriptor_free_chmod() is False


@pytest.mark.parametrize("payload", [
    '{"ok": false, "errno": "x", "reason": 5}',
    '{"ok": false, "errno": -1}',
    '{"ok": false}',
])
def test_permission_worker_malformed_refusal_fails_closed(tmp_path, monkeypatch, payload):
    db = tmp_path / "probe.db"
    db.touch()
    monkeypatch.setattr(
        util.subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=payload, stderr=""),
    )
    with pytest.raises(OSError) as info:
        util._run_sqlite_permission_worker("restrict", db)
    assert info.value.errno == util.errno.EPERM
    assert isinstance(info.value.strerror, str) and info.value.strerror


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin isolation boundary")
def test_real_worker_refuses_hardlinked_sidecar_without_touching_target(tmp_path):
    db = tmp_path / "probe.db"
    sqlite3.connect(db).close()
    target = tmp_path / "unrelated.txt"
    target.write_text("shared")
    target.chmod(0o644)
    os.link(target, tmp_path / "probe.db-wal")
    with pytest.raises(OSError):
        util._restrict_existing_sqlite_artifacts(db)
    assert stat_mode(target) == 0o644


def stat_mode(path):
    return os.stat(path).st_mode & 0o777
