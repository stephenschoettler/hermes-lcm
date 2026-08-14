"""Storage invariants required by the LCM whitepaper.

These tests intentionally exercise the persistence primitives directly.  The
engine wires them into compaction and session shutdown separately.
"""

from __future__ import annotations

import json
import sqlite3
import time

import pytest

from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.lifecycle_state import LifecycleStateStore
from hermes_lcm.store import MessageStore


def _message(store: MessageStore, session_id: str, content: str = "source") -> int:
    return store.append(
        session_id,
        {"role": "user", "content": content},
        token_estimate=3,
        conversation_id=session_id,
    )


def test_provider_state_is_archived_but_not_exposed_or_indexed(tmp_path):
    store = MessageStore(tmp_path / "lcm-provider-state.db")
    capsule = [{
        "type": "compaction",
        "encrypted_content": "opaque-native-capsule",
        "_issuer_kind": "codex_backend",
    }]

    store_id = store.append(
        "session-a",
        {
            "role": "assistant",
            "content": "visible answer",
            "codex_compaction_items": capsule,
        },
    )

    stored = store.get(store_id)
    assert "provider_state" not in stored
    assert store.to_openai_msg(stored) == {
        "role": "assistant",
        "content": "visible answer",
    }
    assert store.get_provider_state(store_id) == {
        "codex_compaction_items": capsule,
    }
    assert store.search("opaque-native-capsule", session_id="session-a") == []


def test_relational_provenance_enforces_sources_depth_and_session(tmp_path):
    db_path = tmp_path / "lcm.db"
    store = MessageStore(db_path)
    source_id = _message(store, "session-a")
    dag = SummaryDAG(db_path)

    leaf = SummaryNode(
        session_id="session-a",
        depth=0,
        summary="leaf",
        source_ids=[source_id],
        source_type="messages",
    )
    leaf_id = dag.add_node(leaf)
    assert dag.connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert dag.connection.execute(
        "SELECT message_store_id FROM lcm_summary_message_sources "
        "WHERE summary_node_id = ?",
        (leaf_id,),
    ).fetchall() == [(source_id,)]

    parent_id = dag.add_node(SummaryNode(
        session_id="session-a",
        depth=1,
        summary="parent",
        source_ids=[leaf_id],
        source_type="nodes",
    ))
    assert dag.connection.execute(
        "SELECT source_node_id FROM lcm_summary_node_sources "
        "WHERE summary_node_id = ?",
        (parent_id,),
    ).fetchall() == [(leaf_id,)]

    with pytest.raises(ValueError, match="does not exist"):
        dag.add_node(SummaryNode(
            session_id="session-a",
            depth=0,
            summary="missing message",
            source_ids=[source_id + 999],
            source_type="messages",
        ))
    with pytest.raises(ValueError, match="same session"):
        dag.add_node(SummaryNode(
            session_id="session-b",
            depth=1,
            summary="cross-session parent",
            source_ids=[leaf_id],
            source_type="nodes",
        ))
    with pytest.raises(ValueError, match="lower depth"):
        dag.add_node(SummaryNode(
            session_id="session-a",
            depth=0,
            summary="cycle-shaped parent",
            source_ids=[leaf_id],
            source_type="nodes",
        ))

    dag.close()
    store.close()


def test_relational_provenance_migrates_valid_legacy_json_edges(tmp_path):
    db_path = tmp_path / "legacy.db"
    store = MessageStore(db_path)
    source_id = _message(store, "legacy")
    dag = SummaryDAG(db_path)
    dag.close()
    store.connection.execute(
        """INSERT INTO summary_nodes(
               session_id, depth, summary, token_count, source_token_count,
               source_ids, source_type, created_at, earliest_at, latest_at,
               expand_hint
           ) VALUES (?, 0, 'legacy leaf', 1, 3, ?, 'messages', ?, NULL, NULL, '')""",
        ("legacy", json.dumps([source_id]), time.time()),
    )
    node_id = store.connection.execute("SELECT last_insert_rowid()").fetchone()[0]
    store.connection.commit()
    store.close()

    dag = SummaryDAG(db_path)
    assert dag.connection.execute(
        "SELECT message_store_id FROM lcm_summary_message_sources "
        "WHERE summary_node_id = ?",
        (node_id,),
    ).fetchall() == [(source_id,)]
    dag.close()


def test_retention_preserves_transitive_node_dependencies(tmp_path):
    db_path = tmp_path / "retention.db"
    store = MessageStore(db_path)
    source_id = _message(store, "retain")
    other_source_id = _message(store, "retain", "unrelated")
    dag = SummaryDAG(db_path)

    leaf = dag.add_node(SummaryNode(
        session_id="retain", depth=0, summary="leaf",
        source_ids=[source_id], source_type="messages",
    ))
    middle = dag.add_node(SummaryNode(
        session_id="retain", depth=1, summary="middle",
        source_ids=[leaf], source_type="nodes",
    ))
    top = dag.add_node(SummaryNode(
        session_id="retain", depth=2, summary="top",
        source_ids=[middle], source_type="nodes",
    ))
    unrelated = dag.add_node(SummaryNode(
        session_id="retain", depth=0, summary="unrelated",
        source_ids=[other_source_id], source_type="messages",
    ))

    assert dag.delete_below_depth("retain", 2) == 1
    assert dag.get_node(unrelated) is None
    assert [dag.get_node(node_id).node_id for node_id in (leaf, middle, top)] == [
        leaf, middle, top,
    ]
    dag.close()
    store.close()


def test_publish_node_and_frontier_is_one_transaction(tmp_path):
    db_path = tmp_path / "atomic.db"
    store = MessageStore(db_path)
    source_id = _message(store, "atomic")
    lifecycle = LifecycleStateStore(db_path)
    lifecycle.bind_session("atomic", conversation_id="conversation")
    dag = SummaryDAG(db_path)

    node = SummaryNode(
        session_id="atomic", depth=0, summary="atomic leaf",
        source_ids=[source_id], source_type="messages",
    )
    node_id = dag.publish_node_and_advance_frontier(
        node,
        conversation_id="conversation",
        frontier_store_id=source_id,
    )
    assert node_id == node.node_id
    assert lifecycle.get_by_conversation("conversation").current_frontier_store_id == source_id

    dag.connection.execute(
        """CREATE TRIGGER reject_frontier_advance
           BEFORE UPDATE OF current_frontier_store_id ON lcm_lifecycle_state
           BEGIN SELECT RAISE(ABORT, 'frontier rejected'); END"""
    )
    before = dag.connection.execute("SELECT COUNT(*) FROM summary_nodes").fetchone()[0]
    with pytest.raises(sqlite3.IntegrityError, match="frontier rejected"):
        dag.publish_node_and_advance_frontier(
            SummaryNode(
                session_id="atomic", depth=0, summary="must roll back",
                source_ids=[source_id], source_type="messages",
            ),
            conversation_id="conversation",
            frontier_store_id=source_id + 1,
        )
    assert dag.connection.execute("SELECT COUNT(*) FROM summary_nodes").fetchone()[0] == before

    dag.close()
    lifecycle.close()
    store.close()


def test_durable_pending_ingest_replays_exactly_once_after_lock(tmp_path):
    db_path = tmp_path / "pending.db"
    store = MessageStore(db_path)
    blocker = sqlite3.connect(db_path, timeout=0.1)
    blocker.execute("BEGIN IMMEDIATE")
    try:
        result = store.append_batch_durable(
            "pending-session",
            [{"role": "assistant", "content": "final response"}],
            [4],
            source="cli",
            conversation_id="pending-session",
            busy_timeout_ms=1,
        )
        assert result.persisted is False
        assert result.pending_batch_id
        assert store.pending_ingest_count() == 1
    finally:
        blocker.rollback()
        blocker.close()

    assert store.drain_pending_ingest() == 1
    assert store.pending_ingest_count() == 0
    assert store.get_session_count("pending-session") == 1

    # A receipt makes replay idempotent across a crash after DB commit but
    # before removal of the durable sidecar file.
    store._write_pending_ingest_file(
        result.pending_batch_id,
        {
            "batch_id": result.pending_batch_id,
            "session_id": "pending-session",
            "messages": [{"role": "assistant", "content": "final response"}],
            "token_estimates": [4],
            "source": "cli",
            "conversation_id": "pending-session",
        },
    )
    assert store.drain_pending_ingest() == 1
    assert store.get_session_count("pending-session") == 1
    store.close()
