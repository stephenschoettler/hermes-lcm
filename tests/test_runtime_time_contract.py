"""Normal-runtime source-time preservation for V4.2.

These fixtures intentionally do not use the benchmark-only
``_session_occurrence_dates`` compatibility sidecar.
"""

from __future__ import annotations

import sqlite3

from hermes_lcm import db_bootstrap
from hermes_lcm.dag import SummaryDAG
from hermes_lcm.store import MessageStore


def test_append_preserves_host_timestamp_separately_from_ingest_time(tmp_path):
    store = MessageStore(tmp_path / "time.db")
    try:
        source_time = 1_710_000_000.25
        store_id = store.append(
            "session-a",
            {"role": "user", "content": "source-timed", "timestamp": source_time},
        )
        row = store.get(store_id)
    finally:
        store.close()

    assert row["observed_at"] == source_time
    assert row["observed_at_source"] == "host_message_timestamp"
    assert row["ingested_at"] == row["timestamp"]
    assert row["ingested_at"] != row["observed_at"]


def test_append_batch_preserves_each_valid_host_timestamp(tmp_path):
    store = MessageStore(tmp_path / "batch-time.db")
    try:
        ids = store.append_batch(
            "session-a",
            [
                {"role": "user", "content": "one", "timestamp": 1_710_000_001},
                {"role": "assistant", "content": "two", "timestamp": "2024-03-09T16:00:02Z"},
            ],
        )
        rows = [store.get(store_id) for store_id in ids]
    finally:
        store.close()

    assert [row["observed_at"] for row in rows] == [1_710_000_001.0, 1_710_000_002.0]
    assert all(row["observed_at_source"] == "host_message_timestamp" for row in rows)
    assert all(row["ingested_at"] == row["timestamp"] for row in rows)


def test_invalid_or_naive_source_timestamp_stays_unknown(tmp_path):
    store = MessageStore(tmp_path / "invalid-time.db")
    try:
        store_ids = [
            store.append("session-a", {"role": "user", "content": "naive", "timestamp": "2024-03-09T16:00:02"}),
            store.append("session-a", {"role": "user", "content": "invalid", "timestamp": "not-a-time"}),
        ]
        rows = [store.get(store_id) for store_id in store_ids]
    finally:
        store.close()

    assert all(row["observed_at"] is None for row in rows)
    assert all(row["observed_at_source"] is None for row in rows)
    assert all(row["ingested_at"] == row["timestamp"] for row in rows)


def test_legacy_row_migration_backfills_ingest_only(tmp_path):
    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE messages (
            store_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT,
            tool_call_id TEXT,
            tool_calls TEXT,
            tool_name TEXT,
            timestamp REAL NOT NULL,
            token_estimate INTEGER DEFAULT 0,
            pinned INTEGER DEFAULT 0
        );
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT);
        INSERT INTO messages(session_id, role, content, timestamp)
        VALUES ('legacy', 'user', 'old row', 123.5);
        """
    )
    conn.commit()
    conn.close()

    store = MessageStore(db_path)
    try:
        row = store.get(1)
    finally:
        store.close()

    assert row["timestamp"] == 123.5
    assert row["ingested_at"] == 123.5
    assert row["observed_at"] is None
    assert row["observed_at_source"] is None


def test_fts_search_preserves_time_contract_fields(tmp_path):
    """FTS-path search() rows must carry the same time-sidecar values as a
    direct get() -- not the rank/snippet fields misread as ingested_at/
    observed_at once _MESSAGE_SELECT_COLUMNS grew past the FTS query's
    hardcoded column count."""
    store = MessageStore(tmp_path / "fts-time.db")
    try:
        source_time = 1_710_000_000.25
        store_id = store.append(
            "session-a",
            {"role": "user", "content": "the quick brown fox jumps over the lazy dog", "timestamp": source_time},
        )
        direct = store.get(store_id)
        [fts_row] = store.search("quick fox")
    finally:
        store.close()

    assert fts_row["store_id"] == store_id
    assert fts_row["ingested_at"] == direct["ingested_at"]
    assert isinstance(fts_row["ingested_at"], float)
    assert fts_row["observed_at"] == direct["observed_at"] == source_time
    assert fts_row["observed_at_source"] == direct["observed_at_source"] == "host_message_timestamp"


def test_schema_classifier_accepts_only_declared_time_sidecars(tmp_path):
    db_path = tmp_path / "classified.db"
    dag = SummaryDAG(db_path)
    store = MessageStore(db_path)
    try:
        store._conn.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES('schema_version', ?)",
            (str(db_bootstrap.SCHEMA_VERSION + 1),),
        )
        store._conn.commit()
        assert (
            db_bootstrap.classify_version_mismatch(store._conn)
            == db_bootstrap.VERSION_MISMATCH_INTERIM_STAMP
        )

        store._conn.execute("ALTER TABLE messages ADD COLUMN future_unknown TEXT")
        store._conn.commit()
        assert (
            db_bootstrap.classify_version_mismatch(store._conn)
            == db_bootstrap.VERSION_MISMATCH_GENUINELY_NEWER
        )
    finally:
        store.close()
        dag.close()


def test_steady_state_open_skips_ingest_backfill_write(tmp_path, monkeypatch):
    """Reopening a fully stamped store must not issue the full-table backfill UPDATE.

    On a multi-GB lcm.db that unconditional UPDATE full-scans ``messages`` inside
    a write transaction on every process start and, with concurrent starts,
    queues long enough to exceed the host's per-plugin load deadline.
    """
    db_path = tmp_path / "steady.db"
    store = MessageStore(db_path)
    store.append("s1", {"role": "user", "content": "stamped"})
    store.close()

    statements: list[str] = []
    real_connect = sqlite3.connect

    def traced_connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        conn.set_trace_callback(statements.append)
        return conn

    monkeypatch.setattr(sqlite3, "connect", traced_connect)
    reopened = MessageStore(db_path)
    reopened.close()

    assert not [s for s in statements if "SET ingested_at" in s]


def test_unstamped_row_left_after_column_exists_is_backfilled_on_next_open(tmp_path):
    """Self-healing: a NULL ingested_at left behind (e.g. crash between ALTER and
    backfill) is still found via the pending index and backfilled."""
    db_path = tmp_path / "heal.db"
    store = MessageStore(db_path)
    store.close()
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO messages(session_id, role, content, timestamp, ingested_at) "
        "VALUES ('s', 'user', 'orphan', 77.0, NULL)"
    )
    conn.commit()
    conn.close()

    reopened = MessageStore(db_path)
    try:
        row = reopened.get(1)
    finally:
        reopened.close()
    assert row["ingested_at"] == 77.0
