"""Backup and rotate maintenance operations for the LCM store.

These are the data-layer maintenance primitives behind ``/lcm backup`` and
``/lcm rotate``: they flush the engine's SQLite connections and snapshot the
store to a timestamped or rolling backup file. They are pure functions that
take the engine so the command layer (``command.py``) keeps only the text
formatting, and the store/dag/lifecycle connection handling lives in one place.
"""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import sqlite3
from typing import Any


_SQLITE_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")


def _restrict_existing_sqlite_artifacts(db_path: Path) -> None:
    for artifact in (
        db_path,
        *(db_path.with_name(db_path.name + suffix) for suffix in _SQLITE_SIDECAR_SUFFIXES),
    ):
        try:
            artifact.chmod(0o600)
        except FileNotFoundError:
            continue


def _prepare_private_backup_directory(path: Path) -> None:
    path.mkdir(parents=True, mode=0o700, exist_ok=True)
    path.chmod(0o700)


def _prepare_private_sqlite_file(path: Path) -> None:
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    fd = os.open(path, flags, 0o600)
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(fd, 0o600)
        else:  # pragma: no cover - Windows fallback
            path.chmod(0o600)
    finally:
        os.close(fd)
    _restrict_existing_sqlite_artifacts(path)


def flush_engine_connections(engine) -> None:
    """Commit pending writes on every SQLite connection the engine owns.

    Shared by ``backup_database`` (timestamped backup) and
    ``rotate_backup_database`` (rolling backup) so the connection-flush
    contract stays in one place.
    """
    engine._store.commit()
    engine._dag._conn.commit()
    lifecycle_conn = getattr(getattr(engine, "_lifecycle", None), "_conn", None)
    if lifecycle_conn is not None:
        lifecycle_conn.commit()
    assertion_store = getattr(engine, "_assertions", None)
    if assertion_store is not None:
        # AssertionStore owns a multi-statement publication transaction. Its
        # lock-taking API must serialize this flush with publish_source() so a
        # backup cannot commit a half-written receipt behind the publisher.
        assertion_store.commit()
    query_views = getattr(engine, "_query_views", None)
    if query_views is not None:
        query_views.commit()


def backup_database(engine) -> dict[str, Any]:
    db_path = Path(engine._store.db_path)
    if not db_path.exists():
        return {
            "ok": False,
            "db_path": db_path,
            "error": "database file does not exist",
        }

    backup_dir = engine.backup_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"{db_path.stem}-{timestamp}.sqlite3"

    try:
        _prepare_private_backup_directory(backup_dir)
        flush_engine_connections(engine)
        _prepare_private_sqlite_file(backup_path)

        dest = sqlite3.connect(str(backup_path))
        try:
            engine._store.backup(dest)
        finally:
            dest.close()
        _restrict_existing_sqlite_artifacts(backup_path)
    except (OSError, sqlite3.Error) as exc:
        return {
            "ok": False,
            "db_path": db_path,
            "error": str(exc),
        }

    backup_size = backup_path.stat().st_size if backup_path.exists() else 0
    return {
        "ok": True,
        "db_path": db_path,
        "backup_path": backup_path,
        "backup_size": backup_size,
    }


def rotate_backup_database(engine) -> dict[str, Any]:
    """Write a rolling rotate-latest SQLite snapshot of the LCM store.

    Atomic via tmp-then-rename so the slot is never half-written. Unlike
    ``backup_database`` which produces timestamped files, this overwrites a
    single rolling slot so disk usage stays bounded across repeated rotates.
    """
    db_path = Path(engine._store.db_path)
    if not db_path.exists():
        return {
            "ok": False,
            "db_path": db_path,
            "error": "database file does not exist",
        }

    backup_path = engine.rotate_backup_path()
    backup_dir = backup_path.parent
    tmp_path = backup_path.with_name(backup_path.name + ".tmp")

    try:
        _prepare_private_backup_directory(backup_dir)
        _restrict_existing_sqlite_artifacts(backup_path)
        flush_engine_connections(engine)

        if tmp_path.exists():
            tmp_path.unlink()
        _prepare_private_sqlite_file(tmp_path)
        dest = sqlite3.connect(str(tmp_path))
        try:
            engine._store.backup(dest)
        finally:
            dest.close()
        _restrict_existing_sqlite_artifacts(tmp_path)
        # Atomic replace so the rolling slot is never half-written.
        tmp_path.replace(backup_path)
        _restrict_existing_sqlite_artifacts(backup_path)
    except (OSError, sqlite3.Error) as exc:
        # Best-effort cleanup of the tmp file if something failed midway.
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        return {
            "ok": False,
            "db_path": db_path,
            "backup_path": backup_path,
            "error": str(exc),
        }

    backup_size = backup_path.stat().st_size if backup_path.exists() else 0
    return {
        "ok": True,
        "db_path": db_path,
        "backup_path": backup_path,
        "backup_size": backup_size,
    }
