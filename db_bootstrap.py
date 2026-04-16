"""Shared SQLite bootstrap helpers for hermes-lcm.

This module keeps startup DB initialization in one place so store/DAG use the
same schema-version marker, PRAGMA settings, and FTS repair behavior.
"""

from __future__ import annotations

import sqlite3
from typing import Iterable, Sequence

SCHEMA_VERSION = 4
SQLITE_BUSY_TIMEOUT_MS = 30_000


class ExternalContentFtsSpec:
    def __init__(
        self,
        *,
        table_name: str,
        content_table: str,
        content_rowid: str,
        indexed_column: str,
        trigger_sqls: Sequence[str],
    ) -> None:
        self.table_name = table_name
        self.content_table = content_table
        self.content_rowid = content_rowid
        self.indexed_column = indexed_column
        self.trigger_sqls = tuple(trigger_sqls)


def configure_connection(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")


def ensure_metadata_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """
    )


def get_schema_version(conn: sqlite3.Connection) -> int:
    ensure_metadata_table(conn)
    row = conn.execute(
        "SELECT value FROM metadata WHERE key = 'schema_version'"
    ).fetchone()
    if not row or row[0] is None:
        return 0
    try:
        return int(str(row[0]))
    except (TypeError, ValueError):
        return 0


def ensure_migration_state_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS lcm_migration_state (
            step_name TEXT PRIMARY KEY,
            completed_at REAL NOT NULL
        )
        """
    )


def ensure_lifecycle_state_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS lcm_lifecycle_state (
            conversation_id TEXT PRIMARY KEY,
            current_session_id TEXT,
            last_finalized_session_id TEXT,
            current_frontier_store_id INTEGER NOT NULL DEFAULT 0,
            last_finalized_frontier_store_id INTEGER NOT NULL DEFAULT 0,
            debt_kind TEXT,
            debt_size_estimate INTEGER NOT NULL DEFAULT 0,
            current_bound_at REAL,
            last_finalized_at REAL,
            debt_updated_at REAL,
            last_maintenance_attempt_at REAL,
            last_rollover_at REAL,
            last_reset_at REAL,
            updated_at REAL NOT NULL DEFAULT (strftime('%s','now'))
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_lcm_lifecycle_current_session ON lcm_lifecycle_state(current_session_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_lcm_lifecycle_last_finalized_session ON lcm_lifecycle_state(last_finalized_session_id)"
    )


def ensure_lifecycle_state_columns(conn: sqlite3.Connection) -> None:
    ensure_lifecycle_state_table(conn)
    columns = {
        row[1] for row in conn.execute("PRAGMA table_info(lcm_lifecycle_state)").fetchall()
    }
    if "debt_kind" not in columns:
        conn.execute("ALTER TABLE lcm_lifecycle_state ADD COLUMN debt_kind TEXT")
    if "debt_size_estimate" not in columns:
        conn.execute(
            "ALTER TABLE lcm_lifecycle_state ADD COLUMN debt_size_estimate INTEGER NOT NULL DEFAULT 0"
        )
    if "debt_updated_at" not in columns:
        conn.execute("ALTER TABLE lcm_lifecycle_state ADD COLUMN debt_updated_at REAL")
    if "last_maintenance_attempt_at" not in columns:
        conn.execute(
            "ALTER TABLE lcm_lifecycle_state ADD COLUMN last_maintenance_attempt_at REAL"
        )


def mark_migration_step_complete(conn: sqlite3.Connection, step_name: str) -> None:
    ensure_migration_state_table(conn)
    conn.execute(
        """
        INSERT INTO lcm_migration_state(step_name, completed_at)
        VALUES(?, strftime('%s','now'))
        ON CONFLICT(step_name) DO UPDATE SET completed_at = excluded.completed_at
        """,
        (step_name,),
    )


def set_schema_version(conn: sqlite3.Connection, version: int = SCHEMA_VERSION) -> None:
    ensure_metadata_table(conn)
    conn.execute(
        """
        INSERT INTO metadata(key, value)
        VALUES('schema_version', ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (str(version),),
    )


def get_existing_table_names(conn: sqlite3.Connection, names: Iterable[str]) -> set[str]:
    existing: set[str] = set()
    for name in names:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name = ?",
            (name,),
        ).fetchone()
        if row and row[0]:
            existing.add(row[0])
    return existing


def get_fts_shadow_table_names(table_name: str) -> list[str]:
    return [
        f"{table_name}_data",
        f"{table_name}_idx",
        f"{table_name}_docsize",
        f"{table_name}_config",
    ]


def quote_sql_identifier(identifier: str) -> str:
    if not identifier or not identifier.replace("_", "a").isalnum() or identifier[0].isdigit():
        raise ValueError(f"invalid SQL identifier: {identifier}")
    return f'"{identifier}"'


def _fts_needs_rebuild(conn: sqlite3.Connection, spec: ExternalContentFtsSpec) -> bool:
    shadow_tables = get_fts_shadow_table_names(spec.table_name)
    existing_tables = get_existing_table_names(conn, [spec.table_name, *shadow_tables])
    if spec.table_name not in existing_tables:
        return True
    if any(name not in existing_tables for name in shadow_tables):
        return True

    try:
        info = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name = ?",
            (spec.table_name,),
        ).fetchone()
        sql = (info[0] if info else "") or ""
        normalized = sql.lower()
        if "virtual table" not in normalized or "using fts5" not in normalized:
            return True

        columns = conn.execute(
            f"PRAGMA table_info({quote_sql_identifier(spec.table_name)})"
        ).fetchall()
        column_names = {row[1] for row in columns if len(row) > 1}
        if spec.indexed_column not in column_names:
            return True

        content_count = conn.execute(
            f"SELECT COUNT(*) FROM {quote_sql_identifier(spec.content_table)}"
        ).fetchone()[0]
        fts_count = conn.execute(
            f"SELECT COUNT(*) FROM {quote_sql_identifier(spec.table_name)}"
        ).fetchone()[0]
        if int(content_count or 0) != int(fts_count or 0):
            return True

        conn.execute(
            f"INSERT INTO {quote_sql_identifier(spec.table_name)}({quote_sql_identifier(spec.table_name)}, rank) VALUES('integrity-check', 1)"
        )
    except sqlite3.DatabaseError:
        return True

    return False


def _drop_fts_table(conn: sqlite3.Connection, table_name: str) -> None:
    conn.execute(f"DROP TABLE IF EXISTS {quote_sql_identifier(table_name)}")
    for shadow_name in get_fts_shadow_table_names(table_name):
        conn.execute(f"DROP TABLE IF EXISTS {quote_sql_identifier(shadow_name)}")


def ensure_external_content_fts(conn: sqlite3.Connection, spec: ExternalContentFtsSpec) -> None:
    if _fts_needs_rebuild(conn, spec):
        _drop_fts_table(conn, spec.table_name)
        conn.execute(
            f"""
            CREATE VIRTUAL TABLE {quote_sql_identifier(spec.table_name)} USING fts5(
                {quote_sql_identifier(spec.indexed_column)},
                content={quote_sql_identifier(spec.content_table)},
                content_rowid={quote_sql_identifier(spec.content_rowid)}
            )
            """
        )
        conn.execute(
            f"INSERT INTO {quote_sql_identifier(spec.table_name)}({quote_sql_identifier(spec.table_name)}) VALUES('rebuild')"
        )

    for trigger_sql in spec.trigger_sqls:
        conn.execute(trigger_sql)


def run_versioned_migrations(conn: sqlite3.Connection) -> None:
    ensure_metadata_table(conn)
    ensure_migration_state_table(conn)

    current_version = get_schema_version(conn)
    if current_version < 2:
        mark_migration_step_complete(conn, "v2_external_content_fts_triggers")
        current_version = 2

    if current_version < 3:
        ensure_lifecycle_state_table(conn)
        mark_migration_step_complete(conn, "v3_lifecycle_state")
        current_version = 3
    else:
        ensure_lifecycle_state_table(conn)

    ensure_lifecycle_state_columns(conn)
    if current_version < 4:
        mark_migration_step_complete(conn, "v4_lifecycle_debt_columns")
        current_version = 4

    set_schema_version(conn, current_version)
