"""Shared SQLite bootstrap helpers for hermes-lcm.

This module keeps startup DB initialization in one place so store/DAG use the
same schema-version marker, PRAGMA settings, and FTS repair behavior.
"""

from __future__ import annotations

import sqlite3
from typing import Iterable

SCHEMA_VERSION = 1
SQLITE_BUSY_TIMEOUT_MS = 5_000


class ExternalContentFtsSpec:
    def __init__(
        self,
        *,
        table_name: str,
        content_table: str,
        content_rowid: str,
        indexed_column: str,
        trigger_name: str,
        trigger_sql: str,
    ) -> None:
        self.table_name = table_name
        self.content_table = content_table
        self.content_rowid = content_rowid
        self.indexed_column = indexed_column
        self.trigger_name = trigger_name
        self.trigger_sql = trigger_sql


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

    conn.execute(spec.trigger_sql)
