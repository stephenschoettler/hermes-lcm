"""Regression coverage for private SQLite storage artifacts."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import sqlite3
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest

import hermes_lcm.db_bootstrap as db_bootstrap_module
import hermes_lcm.maintenance as maintenance_module
import hermes_lcm.sqlite_util as sqlite_util_module
from hermes_lcm.maintenance import backup_database, rotate_backup_database
from hermes_lcm.store import MessageStore, build_message_fts_spec


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


def _seed_searchable_store(db_path: Path) -> None:
    store = MessageStore(db_path)
    try:
        store.append(
            "sidecar-race",
            {"role": "user", "content": "durable sidecar lifecycle token"},
        )
        store.commit()
    finally:
        store.close()


def _assert_searchable_store_integrity(store: MessageStore) -> None:
    conn = store.connection
    assert conn is not None
    assert conn.execute("SELECT content FROM messages").fetchone()[0] == (
        "durable sidecar lifecycle token"
    )
    assert store.search("lifecycle", limit=10)[0]["content"] == (
        "durable sidecar lifecycle token"
    )
    assert db_bootstrap_module.check_external_content_fts_integrity(
        conn, build_message_fts_spec()
    )["status"] == "pass"
    assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


@pytest.fixture(params=["path-descriptor", "regular-descriptor"])
def chmod_descriptor(request, monkeypatch):
    """Run a permission test through both artifact chmod strategies."""
    if request.param == "path-descriptor":
        if not sqlite_util_module._CHMOD_THROUGH_PATH_DESCRIPTOR:
            pytest.skip("requires O_PATH and /proc/self/fd")
    else:
        monkeypatch.setattr(sqlite_util_module, "_CHMOD_THROUGH_PATH_DESCRIPTOR", False)
    return request.param


def _hold_wal_connection(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE held (value TEXT)")
    conn.execute("INSERT INTO held VALUES ('held')")
    conn.commit()
    assert conn.execute("SELECT count(*) FROM held").fetchone()[0] == 1
    return conn


def _run_permission_helper(helper: str, db_path: Path) -> MessageStore | None:
    if helper == "prepare":
        sqlite_util_module._prepare_private_sqlite_file(db_path)
    elif helper == "restrict":
        sqlite_util_module._restrict_existing_sqlite_artifacts(db_path)
    else:
        return MessageStore(db_path)
    return None


def _assert_wal_survives_other_process_close(db_path: Path, held: sqlite3.Connection) -> None:
    """Another process closing its last connection must not delete the live WAL.

    If this process's POSIX locks were released, the other process believes it
    holds the last connection, checkpoints, and unlinks -wal/-shm while
    ``held`` still uses them; a fresh connection then pairs the orphaned index
    with a new WAL and fails with ``disk I/O error``.
    """
    sidecars = [db_path.with_name(db_path.name + suffix) for suffix in ("-wal", "-shm")]
    before = [os.stat(sidecar).st_ino for sidecar in sidecars]
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sqlite3, sys\n"
            "conn = sqlite3.connect(sys.argv[1])\n"
            "conn.execute(\"INSERT INTO held VALUES ('other process')\")\n"
            "conn.commit()\n"
            "conn.close()\n",
            str(db_path),
        ],
        check=True,
        timeout=60,
    )

    assert [sidecar.exists() for sidecar in sidecars] == [True, True]
    assert [os.stat(sidecar).st_ino for sidecar in sidecars] == before
    held.execute("INSERT INTO held VALUES ('held again')")
    held.commit()
    fresh = sqlite3.connect(db_path)
    try:
        assert fresh.execute("SELECT count(*) FROM held").fetchone()[0] == 3
        assert fresh.execute("PRAGMA quick_check").fetchone()[0] == "ok"
    finally:
        fresh.close()


@pytest.mark.parametrize("helper", ["prepare", "restrict", "message_store"])
def test_permission_helpers_keep_sqlite_locks_on_private_database(
    tmp_path,
    helper,
    chmod_descriptor,
):
    db_path = tmp_path / "lcm.db"
    with _process_umask(0o077):
        held = _hold_wal_connection(db_path)
    store = None
    try:
        store = _run_permission_helper(helper, db_path)

        _assert_private_sqlite_artifacts(db_path)
        _assert_wal_survives_other_process_close(db_path, held)
    finally:
        if store is not None:
            store.close()
        held.close()


@pytest.mark.skipif(
    not sqlite_util_module._CHMOD_THROUGH_PATH_DESCRIPTOR,
    reason="tightening without a lock-releasing close requires O_PATH and /proc/self/fd",
)
@pytest.mark.parametrize("helper", ["prepare", "restrict", "message_store"])
def test_permission_helpers_tighten_loose_database_without_releasing_sqlite_locks(
    tmp_path,
    helper,
):
    db_path = tmp_path / "lcm.db"
    with _process_umask(0o022):
        held = _hold_wal_connection(db_path)
    store = None
    try:
        for artifact in _sqlite_artifacts(db_path):
            if artifact.exists():
                artifact.chmod(0o644)

        store = _run_permission_helper(helper, db_path)

        _assert_private_sqlite_artifacts(db_path)
        _assert_wal_survives_other_process_close(db_path, held)
    finally:
        if store is not None:
            store.close()
        held.close()


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


def test_message_store_refuses_created_directory_swap_before_chmod(tmp_path, monkeypatch):
    shared_parent = tmp_path / "shared"
    shared_parent.mkdir(mode=0o777)
    shared_parent.chmod(0o777)
    db_dir = shared_parent / "database"
    db_path = db_dir / "lcm.db"
    displaced_dir = shared_parent / "database-displaced"
    unrelated_target = tmp_path / "unrelated-target"
    unrelated_target.mkdir(mode=0o755)
    unrelated_target.chmod(0o755)
    real_open = os.open
    real_path_chmod = Path.chmod
    swapped = False

    def swap_created_directory():
        nonlocal swapped
        if swapped or not db_dir.is_dir():
            return
        swapped = True
        db_dir.rename(displaced_dir)
        db_dir.symlink_to(unrelated_target, target_is_directory=True)

    def swapping_open(path, flags, mode=0o777, *, dir_fd=None):
        if dir_fd is not None and path == db_dir.name:
            swap_created_directory()
        if dir_fd is None:
            return real_open(path, flags, mode)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    def swapping_path_chmod(path, mode, *, follow_symlinks=True):
        if path == db_dir:
            swap_created_directory()
        return real_path_chmod(path, mode, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(os, "open", swapping_open)
    monkeypatch.setattr(Path, "chmod", swapping_path_chmod)

    with pytest.raises(OSError):
        MessageStore(db_path)

    assert swapped is True
    assert _mode(unrelated_target) == 0o755


def test_message_store_tightens_compatible_existing_database_artifacts(tmp_path, chmod_descriptor):
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
                assert _mode(db_dir) == 0o755
                _assert_private_sqlite_artifacts(db_path)
            finally:
                store.close()
    finally:
        existing.close()


@pytest.mark.parametrize("suffix", _SQLITE_SIDECAR_SUFFIXES)
def test_message_store_refuses_symlinked_sidecar_before_chmod(tmp_path, suffix, chmod_descriptor):
    db_path = tmp_path / "lcm.db"
    target = tmp_path / "unrelated.txt"
    target.write_text("shared", encoding="utf-8")
    target.chmod(0o644)
    db_path.with_name(db_path.name + suffix).symlink_to(target)

    with pytest.raises(OSError, match="SQLite artifact"):
        MessageStore(db_path)

    assert _mode(target) == 0o644


@pytest.mark.parametrize("suffix", _SQLITE_SIDECAR_SUFFIXES)
def test_message_store_refuses_hardlinked_sidecar_before_chmod(tmp_path, suffix, chmod_descriptor):
    db_path = tmp_path / "lcm.db"
    target = tmp_path / "unrelated.txt"
    target.write_text("shared", encoding="utf-8")
    target.chmod(0o644)
    os.link(target, db_path.with_name(db_path.name + suffix))

    with pytest.raises(OSError, match="SQLite artifact"):
        MessageStore(db_path)

    assert _mode(target) == 0o644


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_message_store_refuses_sidecar_link_swap_before_chmod(
    tmp_path,
    monkeypatch,
    link_kind,
    chmod_descriptor,
):
    db_path = tmp_path / "lcm.db"
    sidecar = db_path.with_name(db_path.name + "-wal")
    sidecar.write_text("replace me", encoding="utf-8")
    target = tmp_path / "unrelated.txt"
    target.write_text("shared", encoding="utf-8")
    target.chmod(0o644)
    real_open = os.open
    swapped = False

    def swapping_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if not swapped and dir_fd is not None and path == sidecar.name:
            swapped = True
            sidecar.unlink()
            if link_kind == "symlink":
                sidecar.symlink_to(target)
            else:
                os.link(target, sidecar)
        if dir_fd is None:
            return real_open(path, flags, mode)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(sqlite_util_module.os, "open", swapping_open)

    with pytest.raises(OSError):
        MessageStore(db_path)

    assert swapped is True
    assert _mode(target) == 0o644


def test_message_store_refuses_sidecar_replacement_after_open_before_fstat(
    tmp_path,
    monkeypatch,
    chmod_descriptor,
):
    db_path = tmp_path / "lcm.db"
    sidecar = db_path.with_name(db_path.name + "-journal")
    replacement = tmp_path / "replacement-journal"
    sidecar.write_bytes(b"replace me")
    replacement.write_bytes(b"unrelated")
    sidecar.chmod(0o644)
    replacement.chmod(0o644)
    real_open = os.open
    swapped = False

    def swapping_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if dir_fd is None:
            return real_open(path, flags, mode)
        fd = real_open(path, flags, mode, dir_fd=dir_fd)
        if not swapped and dir_fd is not None and path == sidecar.name:
            swapped = True
            sidecar.unlink()
            replacement.rename(sidecar)
        return fd

    monkeypatch.setattr(sqlite_util_module.os, "open", swapping_open)

    with pytest.raises(OSError, match="directory entry changed while opening"):
        MessageStore(db_path)

    assert swapped is True
    assert _mode(sidecar) == 0o644


def test_message_store_tolerates_sidecar_disappearing_between_stat_and_open(
    tmp_path,
    monkeypatch,
    chmod_descriptor,
):
    db_path = tmp_path / "lcm.db"
    _seed_searchable_store(db_path)
    journal = db_path.with_name(db_path.name + "-journal")
    journal.write_bytes(b"transient rollback journal")
    real_open = os.open
    disappeared = False

    def disappearing_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal disappeared
        if not disappeared and dir_fd is not None and path == journal.name:
            disappeared = True
            journal.unlink()
        if dir_fd is None:
            return real_open(path, flags, mode)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(sqlite_util_module.os, "open", disappearing_open)

    store = MessageStore(db_path)
    try:
        assert disappeared is True
        _assert_searchable_store_integrity(store)
    finally:
        store.close()


def test_message_store_tolerates_sidecar_unlinked_between_open_and_fstat(
    tmp_path,
    monkeypatch,
    chmod_descriptor,
):
    db_path = tmp_path / "lcm.db"
    _seed_searchable_store(db_path)
    journal = db_path.with_name(db_path.name + "-journal")
    journal.write_bytes(b"transient rollback journal")
    real_open = os.open
    unlinked = False

    def unlinking_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal unlinked
        if dir_fd is None:
            return real_open(path, flags, mode)
        fd = real_open(path, flags, mode, dir_fd=dir_fd)
        if not unlinked and path == journal.name:
            unlinked = True
            journal.unlink()
        return fd

    monkeypatch.setattr(sqlite_util_module.os, "open", unlinking_open)

    store = MessageStore(db_path)
    try:
        assert unlinked is True
        _assert_searchable_store_integrity(store)
    finally:
        store.close()


def test_message_store_preserves_sqlite_memory_sentinel_and_cwd_permissions(tmp_path, monkeypatch):
    tmp_path.chmod(0o755)
    monkeypatch.chdir(tmp_path)

    store = MessageStore(":memory:")
    try:
        store.append("session", {"role": "user", "content": "memory-only"})
        store.commit()
        assert store.connection.execute("SELECT content FROM messages").fetchone()[0] == "memory-only"
    finally:
        store.close()

    assert _mode(tmp_path) == 0o755
    assert not (tmp_path / ":memory:").exists()


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


@pytest.mark.parametrize("operation", ["timestamped", "rotate"])
def test_maintenance_backups_work_without_fchmod(tmp_path, monkeypatch, operation):
    db_path = tmp_path / "database" / "lcm.db"
    store = MessageStore(db_path)
    store.append("session", {"role": "user", "content": "backup"})
    store.commit()
    backup_dir = tmp_path / "backups"
    rotate_path = backup_dir / "rotate-latest.sqlite3"
    engine = SimpleNamespace(
        _store=store,
        _dag=SimpleNamespace(_conn=store.connection),
        _lifecycle=None,
        backup_dir=lambda: backup_dir,
        rotate_backup_path=lambda: rotate_path,
    )

    if hasattr(maintenance_module, "_FCHMOD"):
        monkeypatch.setattr(maintenance_module, "_FCHMOD", None)
    else:  # RED control for the rejected head, which read os.fchmod directly.
        class _OSWithoutFchmod:
            fchmod = None

            def __getattr__(self, name):
                return getattr(os, name)

        monkeypatch.setattr(maintenance_module, "os", _OSWithoutFchmod())
    try:
        result = (
            backup_database(engine)
            if operation == "timestamped"
            else rotate_backup_database(engine)
        )

        assert result["ok"] is True
        assert _mode(backup_dir) == 0o700
        _assert_private_sqlite_artifacts(result["backup_path"])
    finally:
        store.close()


@pytest.mark.parametrize("operation", ["timestamped", "rotate"])
def test_maintenance_rejects_symlinked_backup_directory_before_chmod(
    tmp_path,
    operation,
):
    db_path = tmp_path / "database" / "lcm.db"
    store = MessageStore(db_path)
    store.append("session", {"role": "user", "content": "backup"})
    store.commit()

    backup_parent = tmp_path / "backups"
    backup_parent.mkdir()
    backup_dir = backup_parent / "lcm"
    unrelated_target = tmp_path / "unrelated-target"
    unrelated_target.mkdir(mode=0o755)
    unrelated_target.chmod(0o755)
    backup_dir.symlink_to(unrelated_target, target_is_directory=True)
    rotate_path = backup_dir / "rotate-latest.sqlite3"
    engine = SimpleNamespace(
        _store=store,
        _dag=SimpleNamespace(_conn=store.connection),
        _lifecycle=None,
        backup_dir=lambda: backup_dir,
        rotate_backup_path=lambda: rotate_path,
    )

    try:
        result = (
            backup_database(engine)
            if operation == "timestamped"
            else rotate_backup_database(engine)
        )

        assert result["ok"] is False
        assert _mode(unrelated_target) == 0o755
        assert list(unrelated_target.iterdir()) == []
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


def test_timestamped_backup_flush_failure_leaves_no_empty_artifact(tmp_path, monkeypatch):
    db_path = tmp_path / "database" / "lcm.db"
    store = MessageStore(db_path)
    backup_dir = tmp_path / "backups"
    engine = SimpleNamespace(
        _store=store,
        _dag=SimpleNamespace(_conn=store.connection),
        _lifecycle=None,
        backup_dir=lambda: backup_dir,
    )

    def fail_flush(_engine):
        raise sqlite3.OperationalError("synthetic flush failure")

    monkeypatch.setattr(maintenance_module, "flush_engine_connections", fail_flush)
    try:
        result = backup_database(engine)

        assert result["ok"] is False
        assert result["error"] == "synthetic flush failure"
        assert list(backup_dir.glob("*.sqlite3")) == []
    finally:
        store.close()
