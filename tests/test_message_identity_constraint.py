"""v6: message identity constraint — duplicate-proof storage.

Covers the storage-layer fix for issue #599 (no identity constraint) and the
rebind fresh-tail re-append (issue #631) residual:
  - the v6 migration dedups existing rows and creates the unique identity index
  - inserts of an already-stored identity return the existing store_id instead
    of raising or creating a duplicate row
  - the migration is idempotent and reference-safe (summary nodes, frontiers)
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from hermes_lcm.db_bootstrap import (
    SCHEMA_VERSION,
    ensure_message_identity_constraint,
    run_versioned_migrations,
)
from hermes_lcm.store import MessageStore


def _store(tmp_path) -> MessageStore:
    return MessageStore(tmp_path / "lcm-test.db")


def _append(store, *, session_id="s1", role="user", content="hello",
            timestamp=1000.0, tool_call_id=None) -> int:
    return store.append(
        session_id,
        {
            "role": role,
            "content": content,
            "timestamp": timestamp,
            "tool_call_id": tool_call_id,
        },
        source="test",
    )


def test_schema_version_bumped_to_v6():
    assert SCHEMA_VERSION == 6


def test_migration_creates_unique_identity_index(tmp_path):
    store = _store(tmp_path)
    run_versioned_migrations(store._conn)
    idx = store._conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='index' AND name='idx_msg_identity'"
    ).fetchone()
    assert idx is not None
    assert "UNIQUE" in idx[0]
    # version marker recorded
    row = store._conn.execute(
        "SELECT value FROM metadata WHERE key='schema_version'"
    ).fetchone()
    assert row and row[0] == "6"
    step = store._conn.execute(
        "SELECT 1 FROM lcm_migration_state WHERE step_name='v6_message_identity_constraint'"
    ).fetchone()
    assert step is not None


def test_duplicate_append_returns_existing_store_id(tmp_path):
    store = _store(tmp_path)
    first = _append(store, content="hello", timestamp=1000.0)
    # Same identity (session, role, observed_at, tool_call_id, content)
    second = _append(store, content="hello", timestamp=1000.0)
    assert second == first
    # No duplicate row
    n = store._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id='s1'"
    ).fetchone()[0]
    assert n == 1
    store.close()


def test_duplicate_append_with_null_tool_call_id(tmp_path):
    """NULL tool_call_id must be deduped (SQLite unique index treats NULLs
    as distinct; the COALESCEd index must not)."""
    store = _store(tmp_path)
    first = _append(store, content="msg", timestamp=2000.0, tool_call_id=None)
    second = _append(store, content="msg", timestamp=2000.0, tool_call_id=None)
    assert second == first
    n = store._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id='s1'"
    ).fetchone()[0]
    assert n == 1
    store.close()


def test_distinct_content_still_inserted(tmp_path):
    store = _store(tmp_path)
    a = _append(store, content="one", timestamp=3000.0)
    b = _append(store, content="two", timestamp=3000.0)
    c = _append(store, content="one", timestamp=3001.0)  # different observed_at
    assert len({a, b, c}) == 3
    n = store._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id='s1'"
    ).fetchone()[0]
    assert n == 3
    store.close()


def test_batch_duplicate_returns_existing_ids(tmp_path):
    store = _store(tmp_path)
    ids1 = store.append_batch(
        "s1",
        [
            {"role": "user", "content": "a", "timestamp": 4000.0},
            {"role": "assistant", "content": "b", "timestamp": 4001.0},
        ],
        source="test",
    )
    # Replaying the same batch must not create new rows and must return the
    # same store_ids.
    ids2 = store.append_batch(
        "s1",
        [
            {"role": "user", "content": "a", "timestamp": 4000.0},
            {"role": "assistant", "content": "b", "timestamp": 4001.0},
        ],
        source="test",
    )
    assert ids2 == ids1
    n = store._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id='s1'"
    ).fetchone()[0]
    assert n == 2
    store.close()


def _seed_duplicates(store):
    """Seed a duplicate row + a summary node + a frontier pointing at it,
    simulating the pre-v6 production state."""
    con = store._conn
    # summary_nodes is created lazily by the compression engine, not by the
    # migration — create it here to mirror a compacted legacy DB.
    con.execute(
        """CREATE TABLE IF NOT EXISTS summary_nodes(
            node_id INTEGER, session_id TEXT, depth INTEGER, summary TEXT,
            token_count INTEGER, source_token_count INTEGER,
            source_ids TEXT, source_type TEXT, created_at REAL,
            earliest_at REAL, latest_at REAL, expand_hint TEXT
        )"""
    )
    base = {
        "session_id": "dup-s", "source": "test", "conversation_id": "dup-s",
        "role": "user", "content": "dup content", "tool_call_id": None,
        "tool_calls": None, "tool_name": None, "token_estimate": 0,
        "pinned": 0, "observed_at": 5000.0,
        "observed_at_source": "host_message_timestamp",
    }
    con.execute(
        """INSERT INTO messages
           (session_id, source, conversation_id, role, content, tool_call_id, tool_calls,
            tool_name, timestamp, token_estimate, pinned, ingested_at,
            observed_at, observed_at_source)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (base["session_id"], base["source"], base["conversation_id"], base["role"],
         base["content"], base["tool_call_id"], base["tool_calls"], base["tool_name"],
         5000.0, 0, 0, 5000.0, 5000.0, base["observed_at_source"]),
    )
    keep_id = con.execute("SELECT last_insert_rowid()").fetchone()[0]
    con.execute(
        """INSERT INTO messages
           (session_id, source, conversation_id, role, content, tool_call_id, tool_calls,
            tool_name, timestamp, token_estimate, pinned, ingested_at,
            observed_at, observed_at_source)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (base["session_id"], base["source"], base["conversation_id"], base["role"],
         base["content"], base["tool_call_id"], base["tool_calls"], base["tool_name"],
         5001.0, 0, 0, 5001.0, 5000.0, base["observed_at_source"]),
    )
    dup_id = con.execute("SELECT last_insert_rowid()").fetchone()[0]
    # summary node referencing the redundant row
    con.execute(
        "INSERT INTO summary_nodes (node_id, session_id, depth, summary, source_ids, "
        "source_type, created_at) VALUES (1, 'dup-s', 0, 's', ?, 'messages', 0.0)",
        (json.dumps([keep_id, dup_id]),),
    )
    # frontier pointing at the redundant row
    con.execute(
        "INSERT INTO lcm_lifecycle_state (conversation_id, current_session_id, "
        "current_frontier_store_id, last_finalized_frontier_store_id, updated_at) "
        "VALUES ('dup-s', 'dup-s', ?, 0, 0.0)",
        (dup_id,),
    )
    con.commit()
    return keep_id, dup_id


def test_migration_dedups_and_remaps_references(tmp_path):
    store = _store(tmp_path)
    con = store._conn
    # Simulate a legacy v5 database: no identity index, no v6 step.
    con.execute("DROP INDEX IF EXISTS idx_msg_identity")
    con.execute("DELETE FROM lcm_migration_state WHERE step_name='v6_message_identity_constraint'")
    con.execute("INSERT INTO metadata(key, value) VALUES('schema_version','5') "
                "ON CONFLICT(key) DO UPDATE SET value='5'")
    con.commit()
    keep_id, dup_id = _seed_duplicates(store)
    # Run the v6 migration (simulates restart of a v5 DB on the new build)
    run_versioned_migrations(store._conn)
    # duplicate row gone, canonical kept
    assert con.execute("SELECT COUNT(*) FROM messages WHERE session_id='dup-s'").fetchone()[0] == 1
    assert con.execute("SELECT store_id FROM messages WHERE session_id='dup-s'").fetchone()[0] == keep_id
    # summary node remapped to canonical, de-duplicated
    node = json.loads(con.execute("SELECT source_ids FROM summary_nodes WHERE node_id=1").fetchone()[0])
    assert node == [keep_id]
    # frontier remapped to canonical
    fr = con.execute(
        "SELECT current_frontier_store_id FROM lcm_lifecycle_state WHERE conversation_id='dup-s'"
    ).fetchone()[0]
    assert fr == keep_id
    # unique index now present
    assert con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='index' AND name='idx_msg_identity'"
    ).fetchone() is not None
    store.close()


def test_migration_idempotent_on_clean_db(tmp_path):
    store = _store(tmp_path)
    _append(store, content="x", timestamp=6000.0)
    run_versioned_migrations(store._conn)
    total1 = store._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    # second run: nothing to dedup, index IF NOT EXISTS, no error
    run_versioned_migrations(store._conn)
    total2 = store._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    assert total1 == total2 == 1
    store.close()


def test_fresh_db_bootstraps_at_v6_with_index(tmp_path):
    """A brand-new database created by the v6 code must have the index."""
    store = _store(tmp_path)
    # MessageStore.__init__ already ran bootstrap+migrations on connect
    idx = store._conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='index' AND name='idx_msg_identity'"
    ).fetchone()
    assert idx is not None
    version = store._conn.execute(
        "SELECT value FROM metadata WHERE key='schema_version'"
    ).fetchone()[0]
    assert version == "6"
    store.close()


def _append_tool(store, *, session_id="s1", content="ok", timestamp=1000.0,
                 tool_call_id="tc-1") -> int:
    return store.append(
        session_id,
        {
            "role": "tool",
            "content": content,
            "timestamp": timestamp,
            "tool_call_id": tool_call_id,
        },
        source="test",
    )


def test_tool_call_id_unique_index_created_on_fresh_db(tmp_path):
    store = _store(tmp_path)
    idx = store._conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='index' AND name='idx_msg_tool_call_id'"
    ).fetchone()
    assert idx is not None
    assert "UNIQUE" in idx[0]
    # Partial: must carry a WHERE clause on tool_call_id
    assert "WHERE" in idx[0] and "tool_call_id" in idx[0]
    # step recorded
    step = store._conn.execute(
        "SELECT 1 FROM lcm_migration_state WHERE step_name='tool_call_id_unique_v1'"
    ).fetchone()
    assert step is not None
    # numeric schema_version unchanged (base-build compatible)
    version = store._conn.execute(
        "SELECT value FROM metadata WHERE key='schema_version'"
    ).fetchone()[0]
    assert version == "6"
    store.close()


def test_tool_reingest_different_observed_at_returns_existing_store_id(tmp_path):
    """Same tool_call_id re-ingested with a DIFFERENT timestamp must collapse
    to the existing row (the re-ingestion gap the v6 identity index misses)."""
    store = _store(tmp_path)
    first = _append_tool(store, content="ok", timestamp=1000.0, tool_call_id="tc-1")
    # Same tool_call_id, different observed_at (replay / re-bind)
    second = _append_tool(store, content="ok", timestamp=1039.0, tool_call_id="tc-1")
    assert second == first
    n = store._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id='s1'"
    ).fetchone()[0]
    assert n == 1
    store.close()


def test_parallel_tool_calls_same_content_distinct_tool_call_id_preserved(tmp_path):
    """Parallel calls that return identical content but have DIFFERENT
    tool_call_ids are distinct messages and must NOT be merged."""
    store = _store(tmp_path)
    a = _append_tool(store, content="ok", timestamp=1000.0, tool_call_id="tc-a")
    b = _append_tool(store, content="ok", timestamp=1000.0, tool_call_id="tc-b")
    assert a != b
    n = store._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id='s1'"
    ).fetchone()[0]
    assert n == 2
    store.close()


def test_distinct_content_different_tool_call_id_preserved(tmp_path):
    store = _store(tmp_path)
    a = _append_tool(store, content="one", timestamp=1000.0, tool_call_id="tc-a")
    b = _append_tool(store, content="two", timestamp=1000.0, tool_call_id="tb")
    assert a != b
    store.close()


def test_migration_dedups_existing_tool_call_id_reingests(tmp_path):
    """Simulate a legacy DB that has two rows sharing a tool_call_id (different
    observed_at), plus a summary node + frontier referencing the redundant row.
    The named migration must collapse to the canonical row and remap refs."""
    store = _store(tmp_path)
    con = store._conn
    # Simulate a legacy pre-migration DB: drop the new index and clear the
    # step marker so the duplicate rows can be seeded (mirrors the v6 test).
    con.execute("DROP INDEX IF EXISTS idx_msg_tool_call_id")
    con.execute("DELETE FROM lcm_migration_state WHERE step_name='tool_call_id_unique_v1'")
    con.commit()
    con.execute(
        """CREATE TABLE IF NOT EXISTS summary_nodes(
            node_id INTEGER, session_id TEXT, depth INTEGER, summary TEXT,
            token_count INTEGER, source_token_count INTEGER,
            source_ids TEXT, source_type TEXT, created_at REAL,
            earliest_at REAL, latest_at REAL, expand_hint TEXT
        )"""
    )
    base = {
        "session_id": "tci-s", "source": "test", "conversation_id": "tci-s",
        "role": "tool", "content": "tool result", "tool_call_id": "tc-x",
        "tool_calls": None, "tool_name": "x", "token_estimate": 0,
        "pinned": 0, "observed_at": 1000.0,
        "observed_at_source": "host_message_timestamp",
    }
    con.execute(
        """INSERT INTO messages
           (session_id, source, conversation_id, role, content, tool_call_id, tool_calls,
            tool_name, timestamp, token_estimate, pinned, ingested_at,
            observed_at, observed_at_source)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (base["session_id"], base["source"], base["conversation_id"], base["role"],
         base["content"], base["tool_call_id"], base["tool_calls"], base["tool_name"],
         1000.0, 0, 0, 1000.0, 1000.0, base["observed_at_source"]),
    )
    keep_id = con.execute("SELECT last_insert_rowid()").fetchone()[0]
    # re-ingest of the SAME tool_call_id, different observed_at
    con.execute(
        """INSERT INTO messages
           (session_id, source, conversation_id, role, content, tool_call_id, tool_calls,
            tool_name, timestamp, token_estimate, pinned, ingested_at,
            observed_at, observed_at_source)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (base["session_id"], base["source"], base["conversation_id"], base["role"],
         base["content"], base["tool_call_id"], base["tool_calls"], base["tool_name"],
         1039.0, 0, 0, 1039.0, 1039.0, base["observed_at_source"]),
    )
    dup_id = con.execute("SELECT last_insert_rowid()").fetchone()[0]
    con.execute(
        "INSERT INTO summary_nodes (node_id, session_id, depth, summary, source_ids, "
        "source_type, created_at) VALUES (1, 'tci-s', 0, 's', ?, 'messages', 0.0)",
        (json.dumps([keep_id, dup_id]),),
    )
    con.execute(
        "INSERT INTO lcm_lifecycle_state (conversation_id, current_session_id, "
        "current_frontier_store_id, last_finalized_frontier_store_id, updated_at) "
        "VALUES ('tci-s', 'tci-s', ?, 0, 0.0)",
        (dup_id,),
    )
    con.commit()

    # Run the named migration (idempotent) — as construction does.
    from hermes_lcm.db_bootstrap import ensure_tool_call_id_unique_constraint
    ensure_tool_call_id_unique_constraint(con)

    assert con.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id='tci-s'"
    ).fetchone()[0] == 1
    assert con.execute(
        "SELECT store_id FROM messages WHERE session_id='tci-s'"
    ).fetchone()[0] == keep_id
    node = json.loads(con.execute(
        "SELECT source_ids FROM summary_nodes WHERE node_id=1"
    ).fetchone()[0])
    assert node == [keep_id]
    fr = con.execute(
        "SELECT current_frontier_store_id FROM lcm_lifecycle_state WHERE conversation_id='tci-s'"
    ).fetchone()[0]
    assert fr == keep_id
    # unique index present
    assert con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='index' AND name='idx_msg_tool_call_id'"
    ).fetchone() is not None
    # idempotent: a second run is a no-op
    ensure_tool_call_id_unique_constraint(con)
    assert con.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id='tci-s'"
    ).fetchone()[0] == 1
    store.close()
