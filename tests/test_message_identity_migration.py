"""Migration tests for the message identity index + one-shot dedup.

Covers the post-pivot split between the two paths:

- ``run_message_identity_migration`` (auto, on every store open):
  NON-destructive. It only attempts the additive ``idx_msg_identity``
  unique index; a legacy DB with pre-existing duplicate clusters blocks
  the build, so the attempt soft-fails (no rows deleted, no marker) and
  is retried on the next open.
- ``run_message_identity_dedup`` (explicit operator action): destructive.
  Collapses duplicate identity clusters (keep earliest), records
  ``messages_identity_dedup_v1``, then builds the index; refuses with
  ``RuntimeError`` when a chunk/summary/frontier reference points at a
  to-be-deleted row.

These tests simulate a legacy DB (no index, duplicate clusters present)
and exercise both paths plus the time-contract self-gate.
"""

import sqlite3

import pytest

from hermes_lcm.db_bootstrap import (
    run_message_identity_dedup,
    run_message_identity_migration,
)
from hermes_lcm.store import MessageStore


def _make_legacy_db(db_path, clusters=3, copies_per_cluster=3,
                    with_time_contract=True):
    """Build a legacy messages table with duplicate identity clusters.

    Each cluster repeats the same (session_id, role, observed_at,
    tool_call_id, content) identity ``copies_per_cluster`` times with
    fresh (re-stamped) timestamps — the exact shape the live
    ~/.hermes/lcm.db acquired from host compaction replays.
    """
    if with_time_contract:
        cols = (
            ",\n            ingested_at REAL,\n"
            "            observed_at REAL,\n"
            "            observed_at_source TEXT"
        )
    else:
        # pre-time-contract shape: no ingested_at/observed_at columns
        cols = ""
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        f"""
        CREATE TABLE messages (
            store_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            source TEXT DEFAULT '',
            conversation_id TEXT DEFAULT '',
            role TEXT NOT NULL,
            content TEXT,
            tool_call_id TEXT,
            tool_calls TEXT,
            tool_name TEXT,
            timestamp REAL NOT NULL,
            token_estimate INTEGER DEFAULT 0,
            pinned INTEGER DEFAULT 0{cols}
        );
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT);
        INSERT INTO metadata(key, value) VALUES ('schema_version', '5');
        """
    )
    ids = []
    for c in range(clusters):
        for copy in range(copies_per_cluster):
            if with_time_contract:
                cur = conn.execute(
                    "INSERT INTO messages(session_id, role, content, timestamp, "
                    "ingested_at, observed_at, observed_at_source, tool_call_id) "
                    "VALUES('sess', 'assistant', ?, ?, ?, ?, 'host_message_timestamp', NULL)",
                    (f"dup text {c}", 1000.0 + copy, 1000.0 + copy, 5000.0 + c),
                )
            else:
                cur = conn.execute(
                    "INSERT INTO messages(session_id, role, content, timestamp, tool_call_id) "
                    "VALUES('sess', 'assistant', ?, ?, NULL)",
                    (f"dup text {c}", 1000.0 + copy),
                )
            ids.append(cur.lastrowid)
    conn.commit()
    conn.close()
    return ids  # ids[copies_per_cluster::copies_per_cluster] are the earliest per cluster


def _index_exists(conn):
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='index' AND name='idx_msg_identity'"
        ).fetchone()
        is not None
    )


def _identity_markers(conn):
    ensure = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='lcm_migration_state'"
    ).fetchone()
    if ensure is None:
        return set()
    return {
        r[0]
        for r in conn.execute(
            "SELECT step_name FROM lcm_migration_state WHERE step_name LIKE 'messages_identity%'"
        )
    }


def test_auto_path_is_non_destructive_on_legacy_dups(tmp_path):
    """Opening a legacy DB with duplicate clusters must NOT delete rows.

    The auto path runs on every store open (including, in the real
    deployment, against the live ~/.hermes/lcm.db), so it must never
    mutate data: with duplicate clusters blocking the unique index it
    soft-fails (no rows deleted, no marker) and retries on the next open.
    """
    db_path = tmp_path / "legacy.db"
    _make_legacy_db(db_path)

    store = MessageStore(db_path)  # auto path runs here
    conn = store._conn
    # Non-destructive: all 9 rows still present.
    assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 9
    # Index blocked by the duplicates; no marker recorded (retried next open).
    assert not _index_exists(conn)
    assert _identity_markers(conn) == set()
    # The store remains fully functional (the passive pass must not lock it out).
    new_id = store.append(
        "sess", {"role": "assistant", "content": "fresh unique msg", "timestamp": 9999.0}
    )
    assert new_id is not None
    assert store.get_session_count("sess") == 10
    store.close()

    # Idempotent re-open: same state, still no destructive surprise.
    store2 = MessageStore(db_path)
    assert store2._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 10
    assert not _index_exists(store2._conn)
    assert _identity_markers(store2._conn) == set()
    store2.close()


def test_auto_path_lands_index_on_clean_db(tmp_path):
    """A fresh (duplicate-free) DB gets the index on the first open."""
    db_path = tmp_path / "clean.db"
    store = MessageStore(db_path)
    conn = store._conn
    assert _index_exists(conn)
    assert _identity_markers(conn) == {"messages_identity_index_v1"}
    # Re-open: marker makes the step a no-op, index persists.
    store.close()
    store2 = MessageStore(db_path)
    assert _index_exists(store2._conn)
    assert store2.get_session_count("sess") == 0
    store2.close()


def test_explicit_dedup_collapses_legacy_clusters_and_enforces(tmp_path):
    """The operator's one-shot dedup keeps the earliest row per cluster,
    records both markers, lands the index, and makes re-sends no-ops."""
    db_path = tmp_path / "legacy.db"
    ids = _make_legacy_db(db_path)  # 3 clusters x 3 copies = 9 rows
    earliest = sorted(ids[::3])  # one earliest id per cluster

    conn = sqlite3.connect(str(db_path))
    deleted = run_message_identity_dedup(conn)
    conn.close()

    assert deleted == 6  # 3 clusters x 2 redundant copies
    # Survivors are exactly the earliest rows.
    conn = sqlite3.connect(str(db_path))
    got = sorted(r[0] for r in conn.execute("SELECT store_id FROM messages"))
    assert got == earliest
    # Markers + index.
    assert _identity_markers(conn) == {
        "messages_identity_dedup_v1",
        "messages_identity_index_v1",
    }
    assert _index_exists(conn)
    conn.close()

    # Enforcement end-to-end: re-adding an identical row is a no-op
    # returning the surviving (earliest) id.
    store = MessageStore(db_path)
    again = store.append(
        "sess", {"role": "assistant", "content": "dup text 0", "timestamp": 5000.0}
    )
    assert again == min(earliest)
    assert store.get_session_count("sess") == 3
    store.close()

    # Idempotent: second explicit run deletes nothing, no error.
    conn = sqlite3.connect(str(db_path))
    assert run_message_identity_dedup(conn) == 0
    conn.close()

    # Re-open via the store: state stable, index intact.
    store2 = MessageStore(db_path)
    assert store2.get_session_count("sess") == 3
    assert _index_exists(store2._conn)
    store2.close()


def test_dedup_refuses_when_chunk_refs_dangle(tmp_path):
    """If a chunk points at a to-be-deleted (non-earliest) copy, the
    explicit dedup must refuse — and leave the DB untouched."""
    db_path = tmp_path / "ref.db"
    ids = _make_legacy_db(db_path, clusters=1, copies_per_cluster=2)
    conn = sqlite3.connect(str(db_path))
    # Point a chunk at the NEWER (deletable) copy -> dedup must refuse.
    conn.execute(
        "CREATE TABLE lcm_chunk_meta (chunk_id TEXT, identity_hash TEXT, store_id INTEGER,"
        " chunk_index INTEGER, char_start INTEGER, char_end INTEGER,"
        " token_estimate INTEGER, embedded_at TEXT, archived INTEGER DEFAULT 0,"
        " PRIMARY KEY(chunk_id, identity_hash))"
    )
    conn.execute(
        "INSERT INTO lcm_chunk_meta(chunk_id, identity_hash, store_id) VALUES('c1','h',?)",
        (max(ids),),
    )
    conn.commit()
    conn.close()

    conn = sqlite3.connect(str(db_path))
    with pytest.raises(RuntimeError, match="refusing identity dedup"):
        run_message_identity_dedup(conn)
    # Refusal happens before any write: rows intact, no markers, no index.
    assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 2
    assert _identity_markers(conn) == set()
    assert not _index_exists(conn)
    conn.close()


def test_explicit_dedup_requires_time_contract_columns(tmp_path):
    """A pre-time-contract legacy DB (no observed_at/ingested_at) cannot
    be deduped: the identity key's time component is missing, so the
    explicit path fails loudly (the auto path instead no-ops silently)."""
    db_path = tmp_path / "pre_tc.db"
    _make_legacy_db(db_path, clusters=1, copies_per_cluster=2, with_time_contract=False)

    conn = sqlite3.connect(str(db_path))
    with pytest.raises(RuntimeError, match="observed_at/ingested_at not materialized"):
        run_message_identity_dedup(conn)
    # Auto path: silent no-op (self-gate), nothing recorded.
    assert run_message_identity_migration(conn) == 0
    assert _identity_markers(conn) == set()
    conn.close()
