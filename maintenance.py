"""Backup and rotate maintenance operations for the LCM store.

These are the data-layer maintenance primitives behind ``/lcm backup`` and
``/lcm rotate``: they flush the engine's SQLite connections and snapshot the
store to a timestamped or rolling backup file. They are pure functions that
take the engine so the command layer (``command.py``) keeps only the text
formatting, and the store/dag/lifecycle connection handling lives in one place.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sqlite3
from typing import Any

from . import access_policy as _access_policy
AuthorizationRequiredError = _access_policy.AuthorizationRequiredError
policy_for_engine = _access_policy.policy_for_engine
policy_access_context = _access_policy.policy_access_context


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
    # Authorize before checking the database path or creating a backup so a
    # denied principal gets no maintenance target or filesystem disclosure.
    policy = policy_for_engine(engine)
    access_context = policy_access_context(engine)
    expected_scope = {"kind": "backup", "operation": "backup_database"}
    expected_scope["required_scope"] = "owner_only"
    decision = policy.authorize_operation(access_context, "owner_only", expected_scope)
    policy.audit_decision(
        access_context, "owner_only", decision.denial_reason, decision.public()
    )
    if not decision.allowed:
        raise AuthorizationRequiredError("authorize_operation", decision.public().denial_reason)
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
        backup_dir.mkdir(parents=True, exist_ok=True)
        flush_engine_connections(engine)

        dest = sqlite3.connect(str(backup_path))
        try:
            engine._store.backup(dest)
        finally:
            dest.close()
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
    # Rotation is the same owner-scoped maintenance surface as a timestamped
    # backup; gate it before resolving paths or touching disk.
    policy = policy_for_engine(engine)
    access_context = policy_access_context(engine)
    expected_scope = {"kind": "backup", "operation": "rotate_backup_database"}
    expected_scope["required_scope"] = "owner_only"
    decision = policy.authorize_operation(access_context, "owner_only", expected_scope)
    policy.audit_decision(
        access_context, "owner_only", decision.denial_reason, decision.public()
    )
    if not decision.allowed:
        raise AuthorizationRequiredError("authorize_operation", decision.public().denial_reason)
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
        backup_dir.mkdir(parents=True, exist_ok=True)
        flush_engine_connections(engine)

        if tmp_path.exists():
            tmp_path.unlink()
        dest = sqlite3.connect(str(tmp_path))
        try:
            engine._store.backup(dest)
        finally:
            dest.close()
        # Atomic replace so the rolling slot is never half-written.
        tmp_path.replace(backup_path)
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
