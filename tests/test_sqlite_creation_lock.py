"""Creating an artifact must not unlock another thread's SQLite connection."""
import os
import sqlite3
import subprocess
import sys
import threading

import pytest

from hermes_lcm import sqlite_util


pytestmark = pytest.mark.skipif(
    not sqlite_util._CHMOD_THROUGH_PATH_DESCRIPTOR,
    reason="Requires Linux O_PATH and procfs",
)


def _competing_database_lock(db):
    return subprocess.check_output(
        [sys.executable, "-c", """
import errno, fcntl, os, sys
fd = os.open(sys.argv[1], os.O_RDWR)
try:
    # SQLite holds a read lock on its SHARED byte range in both journal modes.
    fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB, 510, 0x40000002, os.SEEK_SET)
except OSError as exc:
    if exc.errno not in (errno.EACCES, errno.EAGAIN):
        raise
    print('blocked')
else:
    print('acquired')
finally:
    os.close(fd)
""", str(db)], text=True, timeout=10,
    ).strip()


@pytest.mark.parametrize("journal", ["DELETE", "WAL"])
def test_creation_preserves_another_threads_live_writer(tmp_path, monkeypatch, journal):
    db = tmp_path / "lcm.db"
    release_writer = threading.Event()
    writer_ready = threading.Event()
    errors = []

    def hold_writer():
        try:
            conn = sqlite3.connect(db)
            try:
                conn.execute(f"PRAGMA journal_mode={journal}")
                conn.execute("CREATE TABLE sample (value INTEGER)")
                conn.commit()
                conn.execute("BEGIN IMMEDIATE")
                writer_ready.set()
                if not release_writer.wait(20):
                    raise TimeoutError("test did not release SQLite writer")
                conn.rollback()
            finally:
                conn.close()
        except BaseException as exc:
            errors.append(exc)
            writer_ready.set()

    writer = threading.Thread(target=hold_writer)

    def connect_after_creation():
        writer.start()
        assert writer_ready.wait(10), "SQLite writer did not start"
        assert not errors
        assert _competing_database_lock(db) == "blocked"

    original_open = os.open
    original_mknod = os.mknod

    def open_then_connect(path, flags, *args, **kwargs):
        fd = original_open(path, flags, *args, **kwargs)
        if path == db.name and flags & os.O_CREAT:
            try:
                connect_after_creation()
            except BaseException:
                os.close(fd)
                raise
        return fd

    def mknod_then_connect(path, *args, **kwargs):
        original_mknod(path, *args, **kwargs)
        if path == db.name:
            connect_after_creation()

    # Force the same interleaving through both the original and fixed creator.
    # Real SQLite and a separate process decide whether the lock survived.
    monkeypatch.setattr(os, "open", open_then_connect)
    monkeypatch.setattr(os, "mknod", mknod_then_connect)
    try:
        sqlite_util._prepare_private_sqlite_file(db)
        assert writer_ready.is_set()
        assert _competing_database_lock(db) == "blocked"
    finally:
        release_writer.set()
        if writer.ident is not None:
            writer.join(10)
            assert not writer.is_alive()
    assert not errors


@pytest.mark.parametrize("entry", ["regular", "symlink", "hardlink", "directory"])
def test_creation_race_validates_winning_entry(tmp_path, monkeypatch, entry):
    db = tmp_path / "lcm.db"
    target = tmp_path / "other"
    target.write_bytes(b"do not change")
    target.chmod(0o644)
    original_mknod = os.mknod

    def create_competing_entry(path, *args, **kwargs):
        assert path == db.name
        if entry == "regular":
            db.write_bytes(b"")
            db.chmod(0o644)
        elif entry == "symlink":
            db.symlink_to(target)
        elif entry == "hardlink":
            os.link(target, db)
        else:
            db.mkdir()
        original_mknod(path, *args, **kwargs)  # raises FileExistsError

    monkeypatch.setattr(os, "mknod", create_competing_entry)
    if entry == "regular":
        sqlite_util._prepare_private_sqlite_file(db)
        assert db.stat().st_mode & 0o777 == 0o600
    else:
        with pytest.raises(OSError, match="not a regular file|link count is not one"):
            sqlite_util._prepare_private_sqlite_file(db)
    assert target.stat().st_mode & 0o777 == 0o644
    assert target.read_bytes() == b"do not change"


def test_creation_denial_does_not_fall_back_to_data_descriptor(tmp_path, monkeypatch):
    db = tmp_path / "lcm.db"

    def deny_creation(*args, **kwargs):
        raise PermissionError("mknod denied")

    monkeypatch.setattr(os, "mknod", deny_creation)
    with pytest.raises(PermissionError, match="mknod denied"):
        sqlite_util._prepare_private_sqlite_file(db)
    assert not db.exists()


def test_creation_disappearance_fails_closed(tmp_path, monkeypatch):
    db = tmp_path / "lcm.db"
    original_mknod = os.mknod

    def create_then_unlink(path, *args, **kwargs):
        original_mknod(path, *args, **kwargs)
        db.unlink()

    monkeypatch.setattr(os, "mknod", create_then_unlink)
    with pytest.raises(OSError, match="disappeared while creating"):
        sqlite_util._prepare_private_sqlite_file(db)
    assert not db.exists()
