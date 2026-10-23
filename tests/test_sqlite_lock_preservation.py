"""Regression coverage: hardening SQLite artifacts must never hold a descriptor.

POSIX advisory locks belong to the *process*, not the descriptor: closing ANY
descriptor that refers to a file silently releases every lock the process holds
on it (https://www.sqlite.org/howtocorrupt.html, section 2.2). The artifact
hardening helpers used to ``os.open``/``os.close`` the database and its
``-wal``/``-shm`` sidecars on every store construction. Inside a process that
already held SQLite connections (a long-lived gateway), that cancelled the
connections' locks; the next sibling process to close its own connection then
took the exclusive lock, checkpointed, and *unlinked* the ``-wal``/``-shm`` the
gateway was still using. New openers recreated them, so two generations wrote
one database file and the file's page versions diverged.

The invariant is therefore structural, not conditional on the file's current
mode: restricting an EXISTING SQLite artifact must never open a descriptor on
it, whether or not the mode already is 0600. These tests assert the observable
consequence (a sibling process opening and closing the database must not unlink
or replace a live holder's sidecars) and the mechanism (no existing artifact is
ever opened, and loose modes are still tightened by path-based ``chmod``).
"""

from __future__ import annotations

import os
from pathlib import Path
import sqlite3
import stat
import subprocess
import sys

import pytest

import hermes_lcm.sqlite_util as sqlite_util_module
from hermes_lcm.store import MessageStore

pytestmark = pytest.mark.skipif(os.name != "posix", reason="requires POSIX fcntl lock semantics")

_HELPERS = ["_restrict_existing_sqlite_artifacts", "_prepare_private_sqlite_file"]
# 0600 is the steady state; the others are the "mode really needs tightening"
# cases (legacy 0644 files, a group-readable restore, an over-tight leftover).
_MODES = [0o600, 0o644, 0o640, 0o400]

_SIBLING = (
    "import sqlite3, sys\n"
    "c = sqlite3.connect(sys.argv[1])\n"
    "c.execute('PRAGMA journal_mode=WAL')\n"
    "c.execute('SELECT count(*) FROM sqlite_master').fetchone()\n"
    "c.close()\n"
)


def _sibling_opens_and_closes(db_path: Path) -> None:
    """Model a short-lived CLI/health-probe process touching the same database."""
    subprocess.run([sys.executable, "-c", _SIBLING, str(db_path)], check=True, timeout=60)


def _identity(path: Path) -> tuple[int, int] | None:
    try:
        st = path.stat()
    except FileNotFoundError:
        return None
    return (st.st_dev, st.st_ino)


def _artifacts(db_path: Path) -> list[Path]:
    return [db_path, db_path.with_name(db_path.name + "-wal"), db_path.with_name(db_path.name + "-shm")]


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _wal_holder(db_path: Path, *, mode: int = 0o600) -> sqlite3.Connection:
    """A connection with live WAL state, like the gateway's long-lived store."""
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE IF NOT EXISTS t (k TEXT PRIMARY KEY, v TEXT)")
    conn.execute("INSERT OR REPLACE INTO t VALUES ('a', '1')")
    conn.commit()
    conn.execute("SELECT count(*) FROM t").fetchone()
    for artifact in _artifacts(db_path):
        # Keep the holder's own database writable even for the over-tight case.
        os.chmod(artifact, mode if artifact != db_path else mode | 0o200)
    return conn


def _assert_generation_survives_sibling(db_path: Path, holder: sqlite3.Connection) -> None:
    wal = db_path.with_name(db_path.name + "-wal")
    shm = db_path.with_name(db_path.name + "-shm")
    before = (_identity(wal), _identity(shm))
    assert None not in before, "holder must have live -wal/-shm before the probe"

    _sibling_opens_and_closes(db_path)

    after = (_identity(wal), _identity(shm))
    assert after == before, (
        "a sibling process opening/closing the database unlinked or replaced the "
        "holder's -wal/-shm: the holder's SQLite locks were dropped"
    )
    # The holder is still attached to the one live generation and the file is sound.
    holder.execute("CREATE TABLE IF NOT EXISTS lock_probe (k TEXT PRIMARY KEY)")
    holder.execute("INSERT OR REPLACE INTO lock_probe VALUES ('after-sibling')")
    holder.commit()
    assert holder.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


@pytest.fixture(params=["native", "fallback"])
def chmod_strategy(request, monkeypatch):
    """Exercise both descriptor-free chmod strategies on every platform.

    ``native`` uses ``chmod(dir_fd=..., follow_symlinks=False)`` where the OS
    provides it (macOS/BSD); ``fallback`` simulates a platform without it
    (Linux), which uses a plain ``chmod(dir_fd=...)`` after lstat validation.
    """
    if request.param == "native":
        if not (
            os.chmod in os.supports_dir_fd and os.chmod in os.supports_follow_symlinks
        ):
            pytest.skip("platform has no no-follow chmod; covered by the fallback leg")
    else:
        monkeypatch.setattr(
            os,
            "supports_follow_symlinks",
            set(os.supports_follow_symlinks) - {os.chmod},
        )
    return request.param


def test_control_sibling_close_leaves_live_wal_generation_alone(tmp_path):
    """Sanity: with no stray descriptor work, the sibling must not disturb the holder."""
    db_path = tmp_path / "lcm.db"
    holder = _wal_holder(db_path)
    try:
        _assert_generation_survives_sibling(db_path, holder)
    finally:
        holder.close()


@pytest.mark.parametrize("mode", _MODES, ids=lambda m: f"{m:04o}")
@pytest.mark.parametrize("helper", _HELPERS)
def test_artifact_hardening_does_not_strand_live_wal_generation(
    tmp_path, chmod_strategy, helper, mode
):
    db_path = tmp_path / "lcm.db"
    holder = _wal_holder(db_path, mode=mode)
    try:
        getattr(sqlite_util_module, helper)(db_path)
        assert {p.name: _mode(p) for p in _artifacts(db_path)} == {
            p.name: 0o600 for p in _artifacts(db_path)
        }
        _assert_generation_survives_sibling(db_path, holder)
    finally:
        holder.close()


@pytest.mark.parametrize("loose", [False, True], ids=["private", "loose"])
def test_second_store_in_same_process_does_not_strand_first_stores_wal(tmp_path, loose):
    """The production shape: gateway holds a store, a cron job builds another."""
    db_path = tmp_path / "lcm.db"
    first = MessageStore(db_path)
    try:
        first.append("s", {"role": "user", "content": "alpha"})
        first.commit()
        if loose:
            for artifact in _artifacts(db_path):
                os.chmod(artifact, 0o644)
        second = MessageStore(db_path)  # _init_db re-runs the artifact hardening
        try:
            assert all(_mode(p) == 0o600 for p in _artifacts(db_path))
            _assert_generation_survives_sibling(db_path, first.connection)
        finally:
            second.close()
    finally:
        first.close()


@pytest.mark.parametrize("mode", _MODES, ids=lambda m: f"{m:04o}")
@pytest.mark.parametrize("helper", _HELPERS)
def test_existing_artifacts_are_never_opened(tmp_path, monkeypatch, chmod_strategy, helper, mode):
    """Mechanism: no descriptor on an existing db/-wal/-shm means no lock can be dropped."""
    db_path = tmp_path / "lcm.db"
    holder = _wal_holder(db_path, mode=mode)
    try:
        names = {p.name for p in _artifacts(db_path)}
        opened: list[str] = []
        real_open = os.open

        def recording_open(path, flags, mode=0o777, *, dir_fd=None):
            opened.append(os.path.basename(os.fspath(path)))
            if dir_fd is None:
                return real_open(path, flags, mode)
            return real_open(path, flags, mode, dir_fd=dir_fd)

        monkeypatch.setattr(sqlite_util_module.os, "open", recording_open)
        getattr(sqlite_util_module, helper)(db_path)

        assert not names.intersection(opened), (
            f"{helper} opened existing SQLite artifacts (mode {mode:04o}): "
            f"{sorted(names.intersection(opened))}"
        )
        assert all(_mode(p) == 0o600 for p in _artifacts(db_path))
    finally:
        holder.close()


def test_loose_artifacts_are_tightened_by_path_chmod(tmp_path, monkeypatch, chmod_strategy):
    """Tightening goes through chmod relative to the verified directory, never a descriptor."""
    db_path = tmp_path / "lcm.db"
    holder = _wal_holder(db_path, mode=0o644)
    try:
        calls: list[tuple[str, dict]] = []
        real_chmod = os.chmod

        def recording_chmod(path, mode, **kwargs):
            calls.append((os.fspath(path), kwargs))
            return real_chmod(path, mode, **kwargs)

        monkeypatch.setattr(sqlite_util_module.os, "chmod", recording_chmod)
        sqlite_util_module._restrict_existing_sqlite_artifacts(db_path)

        assert sorted(name for name, _ in calls) == sorted(p.name for p in _artifacts(db_path))
        for _, kwargs in calls:
            assert isinstance(kwargs.get("dir_fd"), int)
            if chmod_strategy == "native":
                assert kwargs.get("follow_symlinks") is False
            else:
                assert kwargs.get("follow_symlinks", True) is True
        assert all(_mode(p) == 0o600 for p in _artifacts(db_path))
    finally:
        holder.close()


def test_already_private_artifacts_are_not_touched_at_all(tmp_path, monkeypatch):
    """Steady state: nothing to tighten means no open and no chmod."""
    db_path = tmp_path / "lcm.db"
    holder = _wal_holder(db_path)
    try:
        calls: list[str] = []
        real_chmod = os.chmod

        def recording_chmod(path, mode, **kwargs):
            calls.append(os.fspath(path))
            return real_chmod(path, mode, **kwargs)

        monkeypatch.setattr(sqlite_util_module.os, "chmod", recording_chmod)
        sqlite_util_module._restrict_existing_sqlite_artifacts(db_path)
        assert calls == []
    finally:
        holder.close()


@pytest.mark.parametrize("umask", [0o022, 0o077, 0o277])
def test_new_database_file_is_created_exclusively_and_private(tmp_path, monkeypatch, umask):
    """Creation is the one place a descriptor is acceptable: the file does not exist yet.

    ``O_EXCL`` guarantees the inode is brand new, so no process (this one
    included) can hold an SQLite lock on it before our descriptor exists.
    """
    db_path = tmp_path / "fresh" / "lcm.db"
    db_path.parent.mkdir(mode=0o700)
    flags_seen: list[tuple[str, int]] = []
    real_open = os.open

    def recording_open(path, flags, mode=0o777, *, dir_fd=None):
        flags_seen.append((os.path.basename(os.fspath(path)), flags))
        if dir_fd is None:
            return real_open(path, flags, mode)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(sqlite_util_module.os, "open", recording_open)
    previous = os.umask(umask)
    try:
        sqlite_util_module._prepare_private_sqlite_file(db_path)
    finally:
        os.umask(previous)

    creates = [flags for name, flags in flags_seen if name == db_path.name]
    assert len(creates) == 1
    assert creates[0] & os.O_CREAT and creates[0] & os.O_EXCL
    assert _mode(db_path) == 0o600
    assert db_path.stat().st_size == 0

    # A second call now sees an existing file and must not open it again.
    flags_seen.clear()
    sqlite_util_module._prepare_private_sqlite_file(db_path)
    assert [name for name, _ in flags_seen if name == db_path.name] == []


def test_restrict_does_not_create_missing_artifacts(tmp_path):
    db_path = tmp_path / "lcm.db"
    sqlite_util_module._restrict_existing_sqlite_artifacts(db_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("suffix", ["", "-wal", "-shm", "-journal"])
def test_symlinked_artifact_is_refused_without_touching_target(tmp_path, chmod_strategy, suffix):
    db_path = tmp_path / "lcm.db"
    target = tmp_path / "unrelated.txt"
    target.write_text("shared", encoding="utf-8")
    target.chmod(0o644)
    (tmp_path / ("lcm.db" + suffix)).symlink_to(target)

    with pytest.raises(OSError, match="SQLite artifact"):
        sqlite_util_module._restrict_existing_sqlite_artifacts(db_path)

    assert _mode(target) == 0o644


@pytest.mark.parametrize("suffix", ["", "-wal", "-shm", "-journal"])
def test_hardlinked_artifact_is_refused_without_touching_target(tmp_path, chmod_strategy, suffix):
    db_path = tmp_path / "lcm.db"
    target = tmp_path / "unrelated.txt"
    target.write_text("shared", encoding="utf-8")
    target.chmod(0o644)
    os.link(target, tmp_path / ("lcm.db" + suffix))

    with pytest.raises(OSError, match="SQLite artifact"):
        sqlite_util_module._restrict_existing_sqlite_artifacts(db_path)

    assert _mode(target) == 0o644


def test_native_nofollow_chmod_never_follows_a_swapped_in_symlink(tmp_path, monkeypatch):
    """Where the OS offers no-follow chmod, a symlink swapped in after lstat is inert."""
    if not (os.chmod in os.supports_dir_fd and os.chmod in os.supports_follow_symlinks):
        pytest.skip("platform has no no-follow chmod")
    db_path = tmp_path / "lcm.db"
    holder = _wal_holder(db_path, mode=0o644)
    wal = db_path.with_name(db_path.name + "-wal")
    target = tmp_path / "unrelated.txt"
    target.write_text("shared", encoding="utf-8")
    target.chmod(0o644)
    real_chmod = os.chmod
    swapped = False

    def swapping_chmod(path, mode, **kwargs):
        nonlocal swapped
        if not swapped and path == wal.name:
            swapped = True
            wal.rename(tmp_path / "displaced-wal")
            wal.symlink_to(target)
        return real_chmod(path, mode, **kwargs)

    monkeypatch.setattr(sqlite_util_module.os, "chmod", swapping_chmod)
    try:
        with pytest.raises(OSError):
            sqlite_util_module._restrict_existing_sqlite_artifacts(db_path)
    finally:
        wal.unlink()
        (tmp_path / "displaced-wal").rename(wal)
        holder.close()

    assert swapped is True
    assert _mode(target) == 0o644
