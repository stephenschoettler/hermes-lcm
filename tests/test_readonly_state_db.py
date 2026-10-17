from __future__ import annotations

import sqlite3

import pytest

from hermes_lcm.db_bootstrap import open_readonly_connection


def test_open_readonly_connection_is_read_only_and_waits_for_wal(tmp_path):
    db_path = tmp_path / "state.db"
    writer = sqlite3.connect(db_path)
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY)")
    writer.execute("INSERT INTO sessions VALUES ('session-1')")
    writer.commit()
    writer.close()

    conn = open_readonly_connection(db_path)
    try:
        assert conn.execute("SELECT id FROM sessions").fetchone()[0] == "session-1"
        assert conn.execute("PRAGMA query_only").fetchone()[0] == 1
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 120_000
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute("INSERT INTO sessions VALUES ('must-not-write')")
    finally:
        conn.close()
