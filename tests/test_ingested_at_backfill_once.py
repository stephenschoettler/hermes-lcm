"""The legacy ``ingested_at`` backfill must run once per database, not on every open.

``MessageStore._ensure_time_contract_columns`` copies ``timestamp`` into
``ingested_at`` for rows that predate the column. Without a done-marker the
statement is a full-table scan on every store open (``ingested_at IS NULL``
cannot use an index on a column that is NULL for zero rows), which on a large
``messages`` table stalls startup for minutes while finding nothing to do.
"""

from __future__ import annotations

import sqlite3

from hermes_lcm.db_bootstrap import is_migration_step_complete
from hermes_lcm.store import _INGESTED_AT_BACKFILL_STEP, MessageStore


class _SpyConn:
    """Forward everything to the real connection, recording executed SQL."""

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
        self.sql: list[str] = []

    def execute(self, sql, *args, **kwargs):
        self.sql.append(" ".join(str(sql).split()))
        return self._conn.execute(sql, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._conn, name)


def _backfill_statements(sql: list[str]) -> list[str]:
    return [s for s in sql if "UPDATE messages SET ingested_at" in s]


def test_first_open_backfills_and_marks_the_step(tmp_path):
    db_path = str(tmp_path / "lcm.db")
    store = MessageStore(db_path)
    assert store._conn is not None
    assert is_migration_step_complete(store._conn, _INGESTED_AT_BACKFILL_STEP)
    store.close()


def test_second_open_does_not_rerun_the_backfill(tmp_path):
    db_path = str(tmp_path / "lcm.db")
    MessageStore(db_path).close()

    store = MessageStore(db_path)
    assert store._conn is not None
    spy = _SpyConn(store._conn)
    store._conn = spy  # type: ignore[assignment]
    store._ensure_time_contract_columns()
    assert _backfill_statements(spy.sql) == [], spy.sql
    store.close()


def test_backfill_still_fills_legacy_rows_before_marking(tmp_path):
    """A pre-marker database with NULL ingested_at rows is backfilled exactly once."""
    db_path = str(tmp_path / "lcm.db")
    store = MessageStore(db_path)
    assert store._conn is not None
    store._conn.execute("DELETE FROM lcm_migration_state WHERE step_name = ?", (_INGESTED_AT_BACKFILL_STEP,))
    store._conn.execute(
        "INSERT INTO messages(session_id, role, content, timestamp, ingested_at) VALUES(?, ?, ?, ?, NULL)",
        ("s", "user", "legacy", 123.0),
    )
    store._conn.commit()
    store.close()

    store = MessageStore(db_path)
    assert store._conn is not None
    row = store._conn.execute("SELECT ingested_at FROM messages WHERE content = 'legacy'").fetchone()
    assert row is not None and row[0] == 123.0
    assert is_migration_step_complete(store._conn, _INGESTED_AT_BACKFILL_STEP)
    store.close()
