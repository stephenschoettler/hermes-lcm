"""Regression coverage for private SQLite storage artifacts."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import sqlite3
import stat
from types import SimpleNamespace

import pytest

from hermes_lcm.maintenance import backup_database, rotate_backup_database
from hermes_lcm.store import MessageStore


_SQLITE_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")
pytestmark = pytest.mark.skipif(os.name != "posix", reason="requires POSIX mode semantics")


@contextmanager
def _process_umask(mask: int):
    previous = os.umask(mask)
    try:
        yield
    finally:
        os.umask(previous)


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _sqlite_artifacts(path: Path) -> list[Path]:
    return [path, *(path.with_name(path.name + suffix) for suffix in _SQLITE_SIDECAR_SUFFIXES)]


def _assert_private_sqlite_artifacts(path: Path) -> None:
    artifacts = [artifact for artifact in _sqlite_artifacts(path) if artifact.exists()]
    assert artifacts
    assert {artifact.name: _mode(artifact) for artifact in artifacts} == {
        artifact.name: 0o600 for artifact in artifacts
    }


def test_message_store_creates_private_database_and_sidecars_under_umask_022(tmp_path):
    db_path = tmp_path / "database" / "lcm.db"

    with _process_umask(0o022):
        store = MessageStore(db_path)
        try:
            store.append("session", {"role": "user", "content": "private"})
            store.commit()

            assert _mode(db_path.parent) == 0o700
            assert db_path.with_name(db_path.name + "-wal").exists()
            assert db_path.with_name(db_path.name + "-shm").exists()
            _assert_private_sqlite_artifacts(db_path)
        finally:
            store.close()


def test_message_store_tightens_compatible_existing_database_artifacts(tmp_path):
    db_dir = tmp_path / "existing"
    db_dir.mkdir(mode=0o755)
    db_dir.chmod(0o755)
    db_path = db_dir / "lcm.db"
    existing = sqlite3.connect(db_path)
    try:
        existing.execute("PRAGMA journal_mode=WAL")
        existing.execute("CREATE TABLE legacy (value TEXT)")
        existing.execute("INSERT INTO legacy VALUES ('retained')")
        existing.commit()

        wal_path = db_path.with_name(db_path.name + "-wal")
        shm_path = db_path.with_name(db_path.name + "-shm")
        assert wal_path.exists()
        assert shm_path.exists()
        for artifact in (db_path, wal_path, shm_path):
            artifact.chmod(0o644)

        with _process_umask(0o022):
            store = MessageStore(db_path)
            try:
                assert store.connection.execute("SELECT value FROM legacy").fetchone()[0] == "retained"
                assert _mode(db_dir) == 0o700
                _assert_private_sqlite_artifacts(db_path)
            finally:
                store.close()
    finally:
        existing.close()


def test_maintenance_creates_private_backups_and_tightens_existing_slot(tmp_path):
    db_path = tmp_path / "database" / "lcm.db"
    store = MessageStore(db_path)
    store.append("session", {"role": "user", "content": "backup"})
    store.commit()

    backup_dir = tmp_path / "backups"
    backup_dir.mkdir(mode=0o755)
    backup_dir.chmod(0o755)
    rotate_path = backup_dir / "rotate-latest.sqlite3"
    rotate_path.write_bytes(b"stale")
    rotate_sidecars = [
        rotate_path.with_name(rotate_path.name + suffix)
        for suffix in _SQLITE_SIDECAR_SUFFIXES
    ]
    for artifact in [rotate_path, *rotate_sidecars]:
        if not artifact.exists():
            artifact.write_bytes(b"stale")
        artifact.chmod(0o644)

    engine = SimpleNamespace(
        _store=store,
        _dag=SimpleNamespace(_conn=store.connection),
        _lifecycle=None,
        backup_dir=lambda: backup_dir,
        rotate_backup_path=lambda: rotate_path,
    )

    try:
        with _process_umask(0o022):
            timestamped = backup_database(engine)
            rotated = rotate_backup_database(engine)

        assert timestamped["ok"] is True
        assert rotated["ok"] is True
        assert _mode(backup_dir) == 0o700
        _assert_private_sqlite_artifacts(timestamped["backup_path"])
        _assert_private_sqlite_artifacts(rotate_path)
        assert not rotate_path.with_name(rotate_path.name + ".tmp").exists()

        for backup_path in (timestamped["backup_path"], rotate_path):
            with sqlite3.connect(backup_path) as restored:
                assert restored.execute("PRAGMA quick_check").fetchone()[0] == "ok"
                assert restored.execute("SELECT content FROM messages").fetchone()[0] == "backup"
    finally:
        store.close()


def test_rotate_backup_failure_preserves_existing_atomic_slot(tmp_path, monkeypatch):
    db_path = tmp_path / "database" / "lcm.db"
    store = MessageStore(db_path)
    store.append("session", {"role": "user", "content": "backup"})
    store.commit()

    backup_dir = tmp_path / "backups"
    backup_dir.mkdir(mode=0o755)
    rotate_path = backup_dir / "rotate-latest.sqlite3"
    previous_backup = b"known-good-backup"
    rotate_path.write_bytes(previous_backup)
    rotate_path.chmod(0o644)
    engine = SimpleNamespace(
        _store=store,
        _dag=SimpleNamespace(_conn=store.connection),
        _lifecycle=None,
        rotate_backup_path=lambda: rotate_path,
    )

    def fail_backup(_destination):
        raise sqlite3.OperationalError("synthetic backup failure")

    monkeypatch.setattr(store, "backup", fail_backup)
    try:
        with _process_umask(0o022):
            result = rotate_backup_database(engine)

        assert result["ok"] is False
        assert result["error"] == "synthetic backup failure"
        assert rotate_path.read_bytes() == previous_backup
        assert _mode(backup_dir) == 0o700
        assert _mode(rotate_path) == 0o600
        assert not rotate_path.with_name(rotate_path.name + ".tmp").exists()
    finally:
        store.close()
