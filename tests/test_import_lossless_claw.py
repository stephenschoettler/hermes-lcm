"""Tests for the lossless-claw/OpenClaw LCM importer."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import pytest

from hermes_lcm.dag import SummaryDAG
from hermes_lcm.store import MessageStore


REPO_ROOT = Path(__file__).resolve().parent.parent
IMPORTER_PATH = REPO_ROOT / "scripts" / "import_lossless_claw.py"


def load_importer_module():
    spec = importlib.util.spec_from_file_location("import_lossless_claw", IMPORTER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def create_lossless_source(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE conversations (
            conversation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            session_key TEXT,
            active INTEGER NOT NULL DEFAULT 1,
            title TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE messages (
            message_id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            seq INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            token_count INTEGER NOT NULL DEFAULT 0,
            identity_hash TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE (conversation_id, seq)
        );
        CREATE TABLE message_parts (
            part_id TEXT PRIMARY KEY,
            message_id INTEGER NOT NULL,
            session_id TEXT NOT NULL,
            part_type TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            text_content TEXT,
            is_ignored INTEGER DEFAULT 0,
            is_synthetic INTEGER DEFAULT 0,
            tool_call_id TEXT,
            tool_name TEXT,
            tool_input TEXT,
            tool_output TEXT,
            tool_error TEXT,
            metadata TEXT,
            UNIQUE (message_id, ordinal)
        );
        """
    )
    conn.execute(
        """INSERT INTO conversations
           (conversation_id, session_id, session_key, title, created_at, updated_at)
           VALUES (1, 'runtime-session-1', 'telegram:direct:503782402:conversation:88',
                   'Sammy direct', '2026-04-20 12:00:00', '2026-04-20 12:00:00')"""
    )
    conn.execute(
        """INSERT INTO messages
           (message_id, conversation_id, seq, role, content, token_count, created_at)
           VALUES (10, 1, 1, 'user', 'hello from old OpenClaw', 7, '2026-04-20 12:00:01')"""
    )
    conn.execute(
        """INSERT INTO messages
           (message_id, conversation_id, seq, role, content, token_count, created_at)
           VALUES (11, 1, 2, 'assistant', 'reply from old OpenClaw', 8, '2026-04-20 12:00:02')"""
    )
    conn.commit()
    conn.close()


def add_shared_session_key_conversation(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """INSERT INTO conversations
           (conversation_id, session_id, session_key, title, created_at, updated_at)
           VALUES (2, 'runtime-session-2', 'telegram:direct:503782402:conversation:88',
                   'Second direct', '2026-04-20 12:01:00', '2026-04-20 12:01:00')"""
    )
    conn.execute(
        """INSERT INTO messages
           (message_id, conversation_id, seq, role, content, token_count, created_at)
           VALUES (12, 2, 1, 'user', 'hello from second conversation', 5, '2026-04-20 12:01:01')"""
    )
    conn.commit()
    conn.close()


def create_summary_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE summaries (
            summary_id TEXT PRIMARY KEY,
            conversation_id INTEGER NOT NULL,
            depth INTEGER NOT NULL,
            kind TEXT NOT NULL,
            content TEXT NOT NULL,
            token_count INTEGER NOT NULL DEFAULT 0,
            source_message_token_count INTEGER NOT NULL DEFAULT 0,
            descendant_token_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            earliest_at TEXT NOT NULL,
            latest_at TEXT NOT NULL,
            expand_hint TEXT DEFAULT ''
        );
        CREATE TABLE summary_messages (
            summary_id TEXT NOT NULL,
            message_id INTEGER NOT NULL,
            ordinal INTEGER NOT NULL
        );
        CREATE TABLE summary_parents (
            summary_id TEXT NOT NULL,
            parent_summary_id TEXT NOT NULL,
            ordinal INTEGER NOT NULL
        );
        """
    )


def add_lossless_summaries(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    create_summary_tables(conn)
    conn.execute(
        """INSERT INTO summaries
           (summary_id, conversation_id, depth, kind, content, token_count,
            source_message_token_count, descendant_token_count,
            created_at, earliest_at, latest_at, expand_hint)
           VALUES ('leaf-1', 1, 0, 'leaf', 'leaf pineapple memory', 5,
                   15, 0, '2026-04-20 12:00:10',
                   '2026-04-20 12:00:01', '2026-04-20 12:00:02',
                   'leaf hint')"""
    )
    conn.execute(
        """INSERT INTO summaries
           (summary_id, conversation_id, depth, kind, content, token_count,
            source_message_token_count, descendant_token_count,
            created_at, earliest_at, latest_at, expand_hint)
           VALUES ('condensed-1', 1, 1, 'condensed', 'condensed pineapple memory', 7,
                   0, 33, '2026-04-20 12:00:20',
                   '2026-04-20 12:00:01', '2026-04-20 12:00:20',
                   'condensed hint')"""
    )
    conn.executemany(
        """INSERT INTO summary_messages (summary_id, message_id, ordinal)
           VALUES (?, ?, ?)""",
        [("leaf-1", 10, 0), ("leaf-1", 10, 1), ("leaf-1", 11, 2)],
    )
    conn.execute(
        """INSERT INTO summary_parents (summary_id, parent_summary_id, ordinal)
           VALUES ('condensed-1', 'leaf-1', 0)"""
    )
    conn.commit()
    conn.close()


def add_unresolved_leaf_summary(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    create_summary_tables(conn)
    conn.execute(
        """INSERT INTO messages
           (message_id, conversation_id, seq, role, content, token_count, created_at)
           VALUES (12, 1, 3, 'assistant', '', 0, '2026-04-20 12:00:03')"""
    )
    conn.execute(
        """INSERT INTO summaries
           (summary_id, conversation_id, depth, kind, content, token_count,
            source_message_token_count, descendant_token_count,
            created_at, earliest_at, latest_at, expand_hint)
           VALUES ('empty-leaf', 1, 0, 'leaf', 'unresolvable empty stub summary', 3,
                   9, 0, '2026-04-20 12:00:11',
                   '2026-04-20 12:00:03', '2026-04-20 12:00:03',
                   '')"""
    )
    conn.execute(
        """INSERT INTO summary_messages (summary_id, message_id, ordinal)
           VALUES ('empty-leaf', 12, 0)"""
    )
    conn.commit()
    conn.close()


def add_partially_unresolved_leaf_summary(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    create_summary_tables(conn)
    conn.execute(
        """INSERT INTO summaries
           (summary_id, conversation_id, depth, kind, content, token_count,
            source_message_token_count, descendant_token_count,
            created_at, earliest_at, latest_at, expand_hint)
           VALUES ('partial-leaf', 1, 0, 'leaf', 'partial leaf should not import', 3,
                   15, 0, '2026-04-20 12:00:10',
                   '2026-04-20 12:00:01', '2026-04-20 12:00:02',
                   '')"""
    )
    conn.executemany(
        """INSERT INTO summary_messages (summary_id, message_id, ordinal)
           VALUES (?, ?, ?)""",
        [("partial-leaf", 10, 0), ("partial-leaf", 999, 1)],
    )
    conn.commit()
    conn.close()


def add_partially_unresolved_parent_summary(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    create_summary_tables(conn)
    conn.execute(
        """INSERT INTO summaries
           (summary_id, conversation_id, depth, kind, content, token_count,
            source_message_token_count, descendant_token_count,
            created_at, earliest_at, latest_at, expand_hint)
           VALUES ('leaf-1', 1, 0, 'leaf', 'leaf parent can import', 5,
                   15, 0, '2026-04-20 12:00:10',
                   '2026-04-20 12:00:01', '2026-04-20 12:00:02',
                   '')"""
    )
    conn.execute(
        """INSERT INTO summaries
           (summary_id, conversation_id, depth, kind, content, token_count,
            source_message_token_count, descendant_token_count,
            created_at, earliest_at, latest_at, expand_hint)
           VALUES ('partial-condensed', 1, 1, 'condensed', 'partial condensed should not import', 7,
                   0, 33, '2026-04-20 12:00:20',
                   '2026-04-20 12:00:01', '2026-04-20 12:00:20',
                   '')"""
    )
    conn.executemany(
        """INSERT INTO summary_messages (summary_id, message_id, ordinal)
           VALUES (?, ?, ?)""",
        [("leaf-1", 10, 0), ("leaf-1", 11, 1)],
    )
    conn.executemany(
        """INSERT INTO summary_parents (summary_id, parent_summary_id, ordinal)
           VALUES (?, ?, ?)""",
        [("partial-condensed", "leaf-1", 0), ("partial-condensed", "missing-parent", 1)],
    )
    conn.commit()
    conn.close()


def test_import_preserves_concrete_session_ids_when_session_key_is_shared(tmp_path: Path):
    importer = load_importer_module()
    source_db = tmp_path / "lossless.db"
    target_db = tmp_path / "target-lcm.db"
    create_lossless_source(source_db)
    add_shared_session_key_conversation(source_db)

    result = importer.import_lossless_claw(
        source_db=source_db,
        target_db=target_db,
        namespace="openclaw-lcm",
        agent="sammy",
        import_id="fixture-import",
        apply=True,
    )

    assert result.imported == 3
    db = sqlite3.connect(target_db)
    imported_sessions = db.execute(
        """SELECT DISTINCT session_id
           FROM messages
           WHERE source != 'existing-source'
           ORDER BY session_id"""
    ).fetchall()
    db.close()

    assert imported_sessions == [
        ("openclaw-lcm:agent:sammy:runtime-session-1",),
        ("openclaw-lcm:agent:sammy:runtime-session-2",),
    ]


def test_import_can_group_by_session_key_when_explicitly_requested(tmp_path: Path):
    importer = load_importer_module()
    source_db = tmp_path / "lossless.db"
    target_db = tmp_path / "target-lcm.db"
    create_lossless_source(source_db)
    add_shared_session_key_conversation(source_db)

    result = importer.import_lossless_claw(
        source_db=source_db,
        target_db=target_db,
        namespace="openclaw-lcm",
        agent="sammy",
        import_id="fixture-import",
        session_identity="session_key",
        apply=True,
    )

    assert result.imported == 3
    db = sqlite3.connect(target_db)
    imported_sessions = db.execute("SELECT DISTINCT session_id FROM messages").fetchall()
    db.close()

    assert imported_sessions == [
        ("openclaw-lcm:agent:sammy:telegram:direct:503782402:conversation:88",),
    ]


def test_dry_run_does_not_create_target_db(tmp_path: Path):
    importer = load_importer_module()
    source_db = tmp_path / "lossless.db"
    target_db = tmp_path / "target-lcm.db"
    create_lossless_source(source_db)

    result = importer.import_lossless_claw(
        source_db=source_db,
        target_db=target_db,
        namespace="openclaw-lcm",
        agent="sammy",
        import_id="fixture-import",
        apply=False,
    )

    assert result.scanned == 2
    assert result.eligible == 2
    assert result.would_import == 2
    assert result.imported == 0
    assert result.skipped_existing == 0
    assert result.backup_path is None
    assert not target_db.exists()


def test_dry_run_handles_uri_reserved_source_db_path(tmp_path: Path):
    importer = load_importer_module()
    source_db = tmp_path / "lossless#archive?.db"
    target_db = tmp_path / "target-lcm.db"
    create_lossless_source(source_db)

    result = importer.import_lossless_claw(
        source_db=source_db,
        target_db=target_db,
        namespace="openclaw-lcm",
        agent="sammy",
        import_id="fixture-import",
        apply=False,
    )

    assert result.scanned == 2
    assert result.eligible == 2
    assert result.would_import == 2
    assert not target_db.exists()


def test_apply_imports_lossless_summaries_as_summary_nodes(tmp_path: Path):
    importer = load_importer_module()
    source_db = tmp_path / "lossless.db"
    target_db = tmp_path / "target-lcm.db"
    create_lossless_source(source_db)
    add_lossless_summaries(source_db)

    result = importer.import_lossless_claw(
        source_db=source_db,
        target_db=target_db,
        namespace="openclaw-lcm",
        agent="sammy",
        import_id="fixture-import",
        include_summaries=True,
        apply=True,
    )

    assert result.imported == 2
    assert result.summaries_imported == 2
    conn = sqlite3.connect(target_db)
    conn.row_factory = sqlite3.Row
    message_map = {
        int(row["source_message_id"]): int(row["target_store_id"])
        for row in conn.execute(
            """SELECT source_message_id, target_store_id
               FROM lcm_imported_messages
               WHERE import_id = 'fixture-import'"""
        )
    }
    rows = conn.execute(
        """SELECT node_id, depth, summary, source_token_count, source_ids,
                  source_type, created_at, earliest_at, latest_at
           FROM summary_nodes
           ORDER BY depth, node_id"""
    ).fetchall()
    fts_count = conn.execute(
        "SELECT COUNT(*) FROM nodes_fts WHERE nodes_fts MATCH ?",
        ("pineapple",),
    ).fetchone()[0]
    conn.close()

    assert len(rows) == 2
    leaf = rows[0]
    condensed = rows[1]
    assert leaf["source_type"] == "messages"
    assert json.loads(leaf["source_ids"]) == [message_map[10], message_map[11]]
    assert condensed["source_type"] == "nodes"
    assert json.loads(condensed["source_ids"]) == [leaf["node_id"]]
    assert leaf["source_token_count"] == 15
    assert condensed["source_token_count"] == 33
    dag = SummaryDAG(target_db)
    condensed_node = dag.get_node(condensed["node_id"])
    assert condensed_node is not None
    assert [node.node_id for node in dag.get_source_nodes(condensed_node)] == [leaf["node_id"]]
    subtree = dag.describe_subtree(condensed["node_id"])
    dag.close()
    assert subtree["source_type"] == "nodes"
    assert subtree["children"] == [
        {
            "node_id": leaf["node_id"],
            "depth": 0,
            "token_count": 5,
            "source_token_count": 15,
            "expand_hint": "leaf hint",
        }
    ]
    assert leaf["created_at"] == pytest.approx(1776686410.0)
    assert leaf["earliest_at"] == pytest.approx(1776686401.0)
    assert leaf["latest_at"] == pytest.approx(1776686402.0)
    assert condensed["created_at"] == pytest.approx(1776686420.0)
    assert condensed["earliest_at"] == pytest.approx(1776686401.0)
    assert condensed["latest_at"] == pytest.approx(1776686420.0)
    assert fts_count == 2


def test_apply_summary_import_is_idempotent_for_same_import_id(tmp_path: Path):
    importer = load_importer_module()
    source_db = tmp_path / "lossless.db"
    target_db = tmp_path / "target-lcm.db"
    create_lossless_source(source_db)
    add_lossless_summaries(source_db)

    first = importer.import_lossless_claw(
        source_db=source_db,
        target_db=target_db,
        agent="sammy",
        import_id="fixture-import",
        include_summaries=True,
        apply=True,
    )
    second = importer.import_lossless_claw(
        source_db=source_db,
        target_db=target_db,
        agent="sammy",
        import_id="fixture-import",
        include_summaries=True,
        apply=True,
    )

    assert first.summaries_imported == 2
    assert second.summaries_imported == 0
    assert second.summaries_skipped_existing == 2
    conn = sqlite3.connect(target_db)
    assert conn.execute("SELECT COUNT(*) FROM summary_nodes").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM lcm_imported_summaries").fetchone()[0] == 2
    conn.close()


def test_summary_leaf_with_only_skipped_empty_messages_is_skipped(tmp_path: Path):
    importer = load_importer_module()
    source_db = tmp_path / "lossless.db"
    target_db = tmp_path / "target-lcm.db"
    create_lossless_source(source_db)
    add_unresolved_leaf_summary(source_db)

    result = importer.import_lossless_claw(
        source_db=source_db,
        target_db=target_db,
        agent="sammy",
        import_id="fixture-import",
        include_summaries=True,
        apply=True,
    )

    assert result.imported == 2
    assert result.skipped_empty == 1
    assert result.summaries_imported == 0
    assert result.summaries_skipped_unresolved == 1
    conn = sqlite3.connect(target_db)
    assert conn.execute("SELECT COUNT(*) FROM summary_nodes").fetchone()[0] == 0
    conn.close()


def test_summary_leaf_with_partially_unresolved_messages_is_skipped(tmp_path: Path):
    importer = load_importer_module()
    source_db = tmp_path / "lossless.db"
    target_db = tmp_path / "target-lcm.db"
    create_lossless_source(source_db)
    add_partially_unresolved_leaf_summary(source_db)

    result = importer.import_lossless_claw(
        source_db=source_db,
        target_db=target_db,
        agent="sammy",
        import_id="fixture-import",
        include_summaries=True,
        apply=True,
    )

    assert result.imported == 2
    assert result.summaries_imported == 0
    assert result.summaries_skipped_unresolved == 1
    conn = sqlite3.connect(target_db)
    assert conn.execute("SELECT COUNT(*) FROM summary_nodes").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM lcm_imported_summaries").fetchone()[0] == 0
    conn.close()


def test_summary_condensed_with_partially_unresolved_parents_is_skipped(tmp_path: Path):
    importer = load_importer_module()
    source_db = tmp_path / "lossless.db"
    target_db = tmp_path / "target-lcm.db"
    create_lossless_source(source_db)
    add_partially_unresolved_parent_summary(source_db)

    result = importer.import_lossless_claw(
        source_db=source_db,
        target_db=target_db,
        agent="sammy",
        import_id="fixture-import",
        include_summaries=True,
        apply=True,
    )

    assert result.imported == 2
    assert result.summaries_imported == 1
    assert result.summaries_skipped_unresolved == 1
    conn = sqlite3.connect(target_db)
    rows = conn.execute(
        "SELECT depth, summary, source_type FROM summary_nodes ORDER BY node_id"
    ).fetchall()
    assert rows == [(0, "leaf parent can import", "messages")]
    assert conn.execute("SELECT COUNT(*) FROM lcm_imported_summaries").fetchone()[0] == 1
    conn.close()


def test_dry_run_include_summaries_does_not_create_target_db(tmp_path: Path):
    importer = load_importer_module()
    source_db = tmp_path / "lossless.db"
    target_db = tmp_path / "target-lcm.db"
    create_lossless_source(source_db)
    add_lossless_summaries(source_db)

    result = importer.import_lossless_claw(
        source_db=source_db,
        target_db=target_db,
        agent="sammy",
        import_id="fixture-import",
        include_summaries=True,
        apply=False,
    )

    assert result.summaries_would_import == 2
    assert not target_db.exists()


def test_apply_import_routes_oversized_payloads_through_ingest_protection(tmp_path: Path, monkeypatch):
    importer = load_importer_module()
    source_db = tmp_path / "lossless.db"
    target_db = tmp_path / "target-lcm.db"
    externalized_dir = tmp_path / "externalized"
    create_lossless_source(source_db)

    large_content = "IMPORT_RAW_NEEDLE:" + ("q" * 5000)
    conn = sqlite3.connect(source_db)
    conn.execute("UPDATE messages SET content = ? WHERE message_id = 10", (large_content,))
    conn.commit()
    conn.close()

    monkeypatch.setenv("LCM_LARGE_OUTPUT_EXTERNALIZATION_ENABLED", "1")
    monkeypatch.setenv("LCM_LARGE_OUTPUT_EXTERNALIZATION_THRESHOLD_CHARS", "200")
    monkeypatch.setenv("LCM_LARGE_OUTPUT_EXTERNALIZATION_PATH", str(externalized_dir))

    result = importer.import_lossless_claw(
        source_db=source_db,
        target_db=target_db,
        namespace="openclaw-lcm",
        agent="sammy",
        import_id="fixture-import",
        apply=True,
    )

    assert result.imported == 2
    db = sqlite3.connect(target_db)
    content = db.execute(
        "SELECT content FROM messages WHERE role = 'user' ORDER BY store_id LIMIT 1"
    ).fetchone()[0]
    db.close()
    assert content.startswith("[Externalized payload: kind=raw_payload;")
    assert "IMPORT_RAW_NEEDLE" not in content

    payload_files = list(externalized_dir.glob("*.json"))
    assert len(payload_files) == 1
    payload = json.loads(payload_files[0].read_text())
    assert payload["kind"] == "raw_payload"
    assert payload["session_id"] == "openclaw-lcm:agent:sammy:runtime-session-1"
    assert payload["content"] == large_content


def test_apply_imports_messages_with_provenance_backup_and_search(tmp_path: Path):
    importer = load_importer_module()
    source_db = tmp_path / "lossless.db"
    target_db = tmp_path / "target-lcm.db"
    create_lossless_source(source_db)
    existing_store = MessageStore(target_db)  # existing DB should be backed up before import writes
    existing_store.append(
        "existing-session",
        {"role": "user", "content": "preexisting committed WAL row"},
        token_estimate=3,
        source="existing-source",
    )

    result = importer.import_lossless_claw(
        source_db=source_db,
        target_db=target_db,
        namespace="openclaw-lcm",
        agent="sammy",
        import_id="fixture-import",
        apply=True,
    )

    assert result.scanned == 2
    assert result.imported == 2
    assert result.would_import == 0
    assert result.backup_path is not None
    assert Path(result.backup_path).exists()
    backup_conn = sqlite3.connect(result.backup_path)
    backup_rows = backup_conn.execute("SELECT session_id, content FROM messages").fetchall()
    backup_conn.close()
    existing_store.close()
    assert backup_rows == [("existing-session", "preexisting committed WAL row")]

    conn = sqlite3.connect(target_db)
    rows = conn.execute(
        """SELECT session_id, source, role, content, timestamp, token_estimate
           FROM messages WHERE source != 'existing-source' ORDER BY store_id"""
    ).fetchall()
    conn.close()

    expected_session = "openclaw-lcm:agent:sammy:runtime-session-1"
    assert rows[0][0] == expected_session
    assert rows[0][1] == expected_session
    assert rows[0][2] == "user"
    assert rows[0][3] == "hello from old OpenClaw"
    assert rows[0][4] == pytest.approx(1776686401.0)
    assert rows[0][5] > 0

    searchable = MessageStore(target_db).search("old OpenClaw", session_id=None, limit=5)
    assert [row["content"] for row in searchable] == [
        "reply from old OpenClaw",
        "hello from old OpenClaw",
    ]


def test_apply_backs_up_uri_reserved_target_db_path(tmp_path: Path):
    importer = load_importer_module()
    source_db = tmp_path / "lossless.db"
    target_db = tmp_path / "target#archive?.db"
    create_lossless_source(source_db)
    existing_store = MessageStore(target_db)
    existing_store.append(
        "existing-session",
        {"role": "user", "content": "preexisting committed WAL row"},
        token_estimate=3,
        source="existing-source",
    )
    existing_store.close()

    result = importer.import_lossless_claw(
        source_db=source_db,
        target_db=target_db,
        namespace="openclaw-lcm",
        agent="sammy",
        import_id="fixture-import",
        apply=True,
    )

    assert result.imported == 2
    assert result.backup_path is not None
    backup_path = Path(result.backup_path)
    assert backup_path.exists()
    assert backup_path.name.startswith("target#archive?.db.backup-")

    backup_conn = sqlite3.connect(backup_path)
    backup_rows = backup_conn.execute("SELECT session_id, content FROM messages").fetchall()
    backup_conn.close()
    assert backup_rows == [("existing-session", "preexisting committed WAL row")]


def test_apply_is_idempotent_for_same_import_id(tmp_path: Path):
    importer = load_importer_module()
    source_db = tmp_path / "lossless.db"
    target_db = tmp_path / "target-lcm.db"
    create_lossless_source(source_db)

    first = importer.import_lossless_claw(
        source_db=source_db,
        target_db=target_db,
        agent="sammy",
        import_id="fixture-import",
        apply=True,
    )
    second = importer.import_lossless_claw(
        source_db=source_db,
        target_db=target_db,
        agent="sammy",
        import_id="fixture-import",
        apply=True,
    )

    assert first.imported == 2
    assert second.imported == 0
    assert second.skipped_existing == 2
    assert second.backup_path is None

    conn = sqlite3.connect(target_db)
    assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM lcm_imported_messages").fetchone()[0] == 2
    conn.close()


def test_invalid_source_schema_reports_required_columns(tmp_path: Path):
    importer = load_importer_module()
    source_db = tmp_path / "invalid-lossless.db"
    target_db = tmp_path / "target-lcm.db"
    conn = sqlite3.connect(source_db)
    conn.execute("CREATE TABLE conversations (conversation_id INTEGER PRIMARY KEY)")
    conn.execute("CREATE TABLE messages (message_id INTEGER PRIMARY KEY, conversation_id INTEGER)")
    conn.commit()
    conn.close()

    with pytest.raises(ValueError, match="missing required columns"):
        importer.import_lossless_claw(
            source_db=source_db,
            target_db=target_db,
            import_id="fixture-import",
            apply=False,
        )


def test_apply_maps_tool_metadata_from_message_parts(tmp_path: Path):
    importer = load_importer_module()
    source_db = tmp_path / "lossless.db"
    target_db = tmp_path / "target-lcm.db"
    create_lossless_source(source_db)
    conn = sqlite3.connect(source_db)
    conn.execute(
        """INSERT INTO messages
           (message_id, conversation_id, seq, role, content, token_count, created_at)
           VALUES (12, 1, 3, 'assistant', '', 3, '2026-04-20 12:00:03')"""
    )
    conn.execute(
        """INSERT INTO message_parts
           (part_id, message_id, session_id, part_type, ordinal, tool_call_id, tool_name, tool_input)
           VALUES ('part-tool-call', 12, 'runtime-session-1', 'tool', 0,
                   'call_123', 'lookup_memory', '{"query":"sammy"}')"""
    )
    conn.execute(
        """INSERT INTO messages
           (message_id, conversation_id, seq, role, content, token_count, created_at)
           VALUES (13, 1, 4, 'tool', '', 4, '2026-04-20 12:00:04')"""
    )
    conn.execute(
        """INSERT INTO message_parts
           (part_id, message_id, session_id, part_type, ordinal, tool_call_id, tool_name, tool_output)
           VALUES ('part-tool-result', 13, 'runtime-session-1', 'tool', 0,
                   'call_123', 'lookup_memory', 'remembered fact')"""
    )
    conn.execute(
        """INSERT INTO messages
           (message_id, conversation_id, seq, role, content, token_count, created_at)
           VALUES (14, 1, 5, 'tool', '', 5, '2026-04-20 12:00:05')"""
    )
    conn.execute(
        """INSERT INTO message_parts
           (part_id, message_id, session_id, part_type, ordinal, tool_call_id, tool_name, tool_output)
           VALUES ('part-tool-result-a', 14, 'runtime-session-1', 'tool', 0,
                   'call_456', 'lookup_memory', 'first chunk')"""
    )
    conn.execute(
        """INSERT INTO message_parts
           (part_id, message_id, session_id, part_type, ordinal, tool_call_id, tool_name, tool_output)
           VALUES ('part-tool-result-b', 14, 'runtime-session-1', 'tool', 1,
                   'call_456', 'lookup_memory', 'second chunk')"""
    )
    conn.commit()
    conn.close()

    result = importer.import_lossless_claw(
        source_db=source_db,
        target_db=target_db,
        agent="sammy",
        import_id="fixture-import",
        apply=True,
    )

    assert result.imported == 5

    conn = sqlite3.connect(target_db)
    assistant = conn.execute(
        "SELECT role, content, tool_calls FROM messages WHERE role = 'assistant' AND tool_calls IS NOT NULL"
    ).fetchone()
    tool = conn.execute(
        """SELECT role, content, tool_call_id, tool_name
           FROM messages WHERE role = 'tool' AND tool_call_id = 'call_123'"""
    ).fetchone()
    multipart_tool = conn.execute(
        """SELECT role, content, tool_call_id, tool_name
           FROM messages WHERE role = 'tool' AND tool_call_id = 'call_456'"""
    ).fetchone()
    conn.close()

    assert assistant[0] == "assistant"
    tool_calls = json.loads(assistant[2])
    assert tool_calls == [
        {
            "id": "call_123",
            "type": "function",
            "function": {"name": "lookup_memory", "arguments": '{"query":"sammy"}'},
        }
    ]
    assert tool == ("tool", "remembered fact", "call_123", "lookup_memory")
    assert multipart_tool == ("tool", "first chunk\nsecond chunk", "call_456", "lookup_memory")


def write_jsonl_session(path: Path, rows: list[dict[str, object] | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for row in rows:
        if isinstance(row, str):
            lines.append(row)
        else:
            lines.append(json.dumps(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def jsonl_header(session_id: str) -> dict[str, object]:
    return {"type": "session", "id": session_id, "timestamp": "2026-06-10T00:00:00Z"}


def jsonl_key(session_id: str, row_id: str) -> str:
    return json.dumps([session_id, row_id], separators=(",", ":"), ensure_ascii=False)


def jsonl_result_row_id(result_type: str, tool_call_id: str, content: object) -> str:
    if content is None:
        normalized_content = None
    elif isinstance(content, str):
        normalized_content = content
    else:
        normalized_content = json.dumps(content, ensure_ascii=False, sort_keys=True)
    content_hash = hashlib.sha256((normalized_content or "").encode("utf-8")).hexdigest()
    return "result:" + json.dumps(
        [result_type, tool_call_id, content_hash],
        separators=(",", ":"),
        ensure_ascii=False,
    )


def jsonl_tool_call_row_id(call_type: str, tool_call_id: str, tool_name: str, arguments: object) -> str:
    if isinstance(arguments, str):
        try:
            decoded_arguments = json.loads(arguments)
        except (TypeError, ValueError, json.JSONDecodeError):
            normalized_arguments = arguments
        else:
            if decoded_arguments is None:
                normalized_arguments = ""
            elif isinstance(decoded_arguments, str):
                normalized_arguments = decoded_arguments
            else:
                normalized_arguments = json.dumps(decoded_arguments, ensure_ascii=False, sort_keys=True)
    elif arguments is None:
        normalized_arguments = ""
    else:
        normalized_arguments = json.dumps(arguments, ensure_ascii=False, sort_keys=True)
    arguments_hash = hashlib.sha256(normalized_arguments.encode("utf-8")).hexdigest()
    return "tool_call:" + json.dumps(
        [call_type, tool_call_id, tool_name, arguments_hash],
        separators=(",", ":"),
        ensure_ascii=False,
    )


def jsonl_message(
    entry_id: str,
    role: str,
    content: object,
    *,
    parent_id: str | None = None,
    timestamp: str = "2026-06-10T00:00:01Z",
    **extra: object,
) -> dict[str, object]:
    row: dict[str, object] = {
        "type": "message",
        "id": entry_id,
        "parentId": parent_id,
        "timestamp": timestamp,
        "message": {"role": role, "content": content, **extra},
    }
    return row


def test_jsonl_import_dry_run_reports_without_creating_target(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "sessions" / "session-a.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            jsonl_header("session-a"),
            jsonl_message("m1", "user", "hello jsonl"),
            jsonl_message("m2", "assistant", "reply jsonl", parent_id="m1"),
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        namespace="openclaw-jsonl",
        agent="sammy",
        import_id="jsonl-import",
        apply=False,
    )

    assert result.scanned == 2
    assert result.eligible == 2
    assert result.would_import == 2
    assert result.imported == 0
    assert result.skipped_existing == 0
    assert result.invalid_rows == 0
    assert result.skipped_empty == 0
    assert result.warnings == []
    assert result.backup_path is None
    assert not target_db.exists()


def test_jsonl_import_apply_preserves_fields_and_search(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "session-a.jsonl"
    target_db = tmp_path / "target-lcm.db"
    tool_calls = [
        {
            "id": "call_123",
            "type": "function",
            "function": {"name": "lookup_memory", "arguments": '{"query":"sammy"}'},
        }
    ]
    write_jsonl_session(
        session_file,
        [
            jsonl_header("session-a"),
            jsonl_message("m1", "user", "hello searchable jsonl", timestamp="2026-06-10T00:00:01Z"),
            jsonl_message(
                "m2",
                "assistant",
                "",
                parent_id="m1",
                timestamp="2026-06-10T00:00:02Z",
                tool_calls=tool_calls,
            ),
            jsonl_message(
                "m3",
                "tool",
                "remembered jsonl fact",
                parent_id="m2",
                timestamp="2026-06-10T00:00:03Z",
                tool_call_id="call_123",
                tool_name="lookup_memory",
            ),
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        namespace="openclaw-jsonl",
        agent="sammy",
        import_id="jsonl-import",
        apply=True,
    )

    assert result.imported == 3
    assert result.to_dict()["warnings"] == []
    expected_session = "openclaw-jsonl:agent:sammy:session-a"
    conn = sqlite3.connect(target_db)
    rows = conn.execute(
        """SELECT session_id, source, role, content, tool_call_id, tool_calls, tool_name, timestamp
           FROM messages ORDER BY store_id"""
    ).fetchall()
    import_rows = conn.execute(
        """SELECT source_message_key, source_session
           FROM lcm_imported_messages
           WHERE import_id = 'jsonl-import'
           ORDER BY target_store_id"""
    ).fetchall()
    conn.close()

    assert [row[0] for row in rows] == [expected_session, expected_session, expected_session]
    assert [row[1] for row in rows] == [expected_session, expected_session, expected_session]
    assert rows[0][2:5] == ("user", "hello searchable jsonl", None)
    assert json.loads(rows[1][5]) == tool_calls
    assert rows[2][2:7] == ("tool", "remembered jsonl fact", "call_123", None, "lookup_memory")
    assert rows[0][7] == pytest.approx(1781049601.0)
    assert import_rows == [(jsonl_key("session-a", "m1"), "session-a"), (jsonl_key("session-a", "m2"), "session-a"), (jsonl_key("session-a", "m3"), "session-a")]
    assert [row["content"] for row in MessageStore(target_db).search("searchable", session_id=None)] == [
        "hello searchable jsonl"
    ]


def test_jsonl_import_is_idempotent_for_same_import_id(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "session-a.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [jsonl_header("session-a"), jsonl_message("m1", "user", "hello jsonl")],
    )

    first = importer.import_jsonl_sessions(
        files=[session_file], target_db=target_db, import_id="jsonl-import", apply=True
    )
    second = importer.import_jsonl_sessions(
        files=[session_file], target_db=target_db, import_id="jsonl-import", apply=True
    )

    assert first.imported == 1
    assert second.imported == 0
    assert second.skipped_existing == 1
    assert second.backup_path is None
    conn = sqlite3.connect(target_db)
    assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM lcm_imported_messages").fetchone()[0] == 1
    conn.close()


def test_jsonl_import_reports_invalid_and_empty_rows_without_aborting(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "mixed.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            jsonl_header("mixed"),
            "{not json}",
            {"type": "message", "id": "missing-message"},
            jsonl_message("empty", "assistant", ""),
            jsonl_message("valid", "user", "keep me"),
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file], target_db=target_db, import_id="jsonl-import", apply=True
    )

    assert result.scanned == 4
    assert result.eligible == 1
    assert result.imported == 1
    assert result.invalid_rows == 2
    assert result.skipped_empty == 1
    assert any("mixed.jsonl:2" in warning for warning in result.warnings)
    conn = sqlite3.connect(target_db)
    assert conn.execute("SELECT role, content FROM messages").fetchall() == [("user", "keep me")]
    conn.close()


def test_jsonl_import_reports_non_string_type_rows_without_aborting(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "non-string-type.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            jsonl_header("non-string-type"),
            jsonl_message("m1", "user", "root"),
            {"type": ["message"], "id": "bad", "parentId": "m1", "role": "assistant", "content": "bad"},
            jsonl_message("m2", "assistant", "current", parent_id="m1"),
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file], target_db=target_db, import_id="non-string-type", apply=True
    )

    assert result.scanned == 3
    assert result.eligible == 2
    assert result.imported == 2
    assert result.invalid_rows == 1
    assert result.skipped_empty == 0
    assert any("non-string-type.jsonl:3: unsupported row shape" in warning for warning in result.warnings)
    conn = sqlite3.connect(target_db)
    rows = conn.execute("SELECT role, content FROM messages ORDER BY store_id").fetchall()
    conn.close()
    assert rows == [("user", "root"), ("assistant", "current")]


def test_jsonl_import_wrapped_metadata_message_does_not_drive_leaf_pruning(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "wrapped-metadata-tail.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            jsonl_header("probe"),
            jsonl_message("m1", "user", "root"),
            jsonl_message("m2", "assistant", "current", parent_id="m1"),
            {"id": "meta", "parentId": "m1", "message": {"type": "reasoning", "content": "tail metadata"}},
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file], target_db=target_db, import_id="wrapped-metadata-tail", apply=True
    )

    assert result.scanned == 3
    assert result.eligible == 2
    assert result.imported == 2
    assert result.invalid_rows == 0
    assert result.skipped_empty == 1
    conn = sqlite3.connect(target_db)
    rows = conn.execute("SELECT role, content FROM messages ORDER BY store_id").fetchall()
    conn.close()
    assert rows == [("user", "root"), ("assistant", "current")]


def test_jsonl_import_untyped_envelope_metadata_without_content_does_not_drive_leaf_pruning(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "untyped-envelope-metadata-tail.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            {"type": "session", "id": "s"},
            {"type": "message", "id": "root", "message": {"role": "user", "content": "root"}},
            {
                "type": "message",
                "id": "leaf",
                "parentId": "root",
                "message": {"role": "assistant", "content": "current"},
            },
            {"id": "meta", "parentId": "root", "message": {"summary": "tail metadata"}},
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file], target_db=target_db, import_id="untyped-envelope-metadata-tail", apply=True
    )

    assert result.scanned == 3
    assert result.eligible == 2
    assert result.imported == 2
    assert result.invalid_rows == 0
    assert result.skipped_empty == 1
    conn = sqlite3.connect(target_db)
    rows = conn.execute("SELECT role, content FROM messages ORDER BY store_id").fetchall()
    conn.close()
    assert rows == [("user", "root"), ("assistant", "current")]


def test_jsonl_import_maps_openclaw_tool_call_and_tool_result_entries(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "session-a.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            jsonl_header("session-a"),
            {
                "type": "message",
                "id": "assistant-tool-call",
                "timestamp": "2026-06-10T00:00:02Z",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "I'll check."},
                        {
                            "type": "toolCall",
                            "toolCallId": "call_789",
                            "toolName": "lookup_memory",
                            "toolInput": {"query": "sammy"},
                        },
                    ],
                },
            },
            {
                "type": "message",
                "id": "tool-result",
                "timestamp": "2026-06-10T00:00:03Z",
                "message": {
                    "role": "toolResult",
                    "toolCallId": "call_789",
                    "toolName": "lookup_memory",
                    "content": "found it",
                },
            },
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file], target_db=target_db, import_id="tool-shapes", apply=True
    )

    assert result.imported == 2
    conn = sqlite3.connect(target_db)
    assistant = conn.execute("SELECT role, content, tool_calls FROM messages WHERE role = 'assistant'").fetchone()
    tool = conn.execute("SELECT role, content, tool_call_id, tool_name FROM messages WHERE role = 'tool'").fetchone()
    conn.close()

    assert assistant[0] == "assistant"
    tool_calls = json.loads(assistant[2])
    assert tool_calls == [
        {
            "id": "call_789",
            "type": "function",
            "function": {"name": "lookup_memory", "arguments": '{"query":"sammy"}'},
        }
    ]
    assert tool == ("tool", "found it", "call_789", "lookup_memory")


def test_jsonl_import_prefers_responses_function_call_call_id_over_item_id(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "responses-function-call.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            jsonl_header("responses-function-call"),
            {
                "type": "message",
                "id": "assistant-tool-call",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "function_call",
                            "id": "fc_lookup_memory",
                            "call_id": "call_lookup_memory",
                            "name": "lookup_memory",
                            "arguments": {"query": "sammy"},
                        },
                    ],
                },
            },
            {
                "type": "message",
                "id": "tool-result",
                "message": {
                    "role": "toolResult",
                    "call_id": "call_lookup_memory",
                    "name": "lookup_memory",
                    "content": "found it",
                },
            },
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file], target_db=target_db, import_id="responses-function-call", apply=True
    )

    assert result.imported == 2
    conn = sqlite3.connect(target_db)
    assistant = conn.execute("SELECT role, tool_calls FROM messages WHERE role = 'assistant'").fetchone()
    tool = conn.execute("SELECT role, content, tool_call_id, tool_name FROM messages WHERE role = 'tool'").fetchone()
    conn.close()

    assert assistant[0] == "assistant"
    assert json.loads(assistant[1]) == [
        {
            "id": "call_lookup_memory",
            "type": "function_call",
            "function": {"name": "lookup_memory", "arguments": '{"query":"sammy"}'},
        }
    ]
    assert tool == ("tool", "found it", "call_lookup_memory", "lookup_memory")


def test_jsonl_import_rejects_nested_responses_function_call_without_call_id(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "responses-function-call-missing-call-id.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            jsonl_header("responses-function-call-missing-call-id"),
            {
                "type": "message",
                "id": "assistant-tool-call",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "function_call",
                            "id": "fc_1",
                            "name": "lookup",
                            "arguments": "{}",
                        },
                    ],
                },
            },
            {"type": "function_call_output", "call_id": "call_1", "output": "result"},
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id="responses-function-call-missing-call-id",
        apply=True,
    )

    assert result.scanned == 2
    assert result.eligible == 0
    assert result.imported == 0
    assert result.invalid_rows == 1
    assert any("message content tool call item missing tool call id or name" in warning for warning in result.warnings)
    assert not target_db.exists()


@pytest.mark.parametrize("row_shape", ["top-level", "wrapped"])
def test_jsonl_import_rejects_responses_function_object_without_call_id(
    tmp_path: Path,
    row_shape: str,
):
    importer = load_importer_module()
    session_file = tmp_path / f"responses-function-object-missing-call-id-{row_shape}.jsonl"
    target_db = tmp_path / "target-lcm.db"
    function_call = {
        "type": "function_call",
        "id": "fc_1",
        "function": {"name": "lookup", "arguments": "{}"},
    }
    row = (
        {"type": "message", "id": "fc-envelope", "message": function_call}
        if row_shape == "wrapped"
        else function_call
    )
    write_jsonl_session(
        session_file,
        [
            jsonl_header(f"responses-function-object-missing-call-id-{row_shape}"),
            row,
            {"type": "function_call_output", "call_id": "call_1", "output": "result"},
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id=f"responses-function-object-missing-call-id-{row_shape}",
        apply=True,
    )

    assert result.scanned == 2
    assert result.eligible == 0
    assert result.imported == 0
    assert result.invalid_rows == 1
    assert any("function_call row missing tool call id or name" in warning for warning in result.warnings)
    assert not target_db.exists()


@pytest.mark.parametrize("tool_calls_key", ["tool_calls", "toolCalls"])
def test_jsonl_import_rejects_malformed_message_tool_calls_array(
    tmp_path: Path,
    tool_calls_key: str,
):
    importer = load_importer_module()
    session_file = tmp_path / f"malformed-{tool_calls_key}.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            jsonl_header("bad-tool-calls"),
            jsonl_message(
                "m1",
                "assistant",
                None,
                **{
                    tool_calls_key: [
                        {"type": "function", "function": {"name": "lookup", "arguments": "{}"}},
                    ],
                },
            ),
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file], target_db=target_db, import_id=f"malformed-{tool_calls_key}", apply=True
    )

    assert result.scanned == 1
    assert result.eligible == 0
    assert result.imported == 0
    assert result.invalid_rows == 1
    assert result.skipped_empty == 0
    assert any("message tool_calls item missing tool call id or name" in warning for warning in result.warnings)
    assert not target_db.exists()


def test_jsonl_import_reads_tool_calls_when_both_array_spellings_are_present(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "both-tool-call-array-spellings.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            jsonl_header("both-tool-call-array-spellings"),
            jsonl_message(
                "m1",
                "assistant",
                None,
                tool_calls=[],
                toolCalls=[
                    {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}},
                ],
            ),
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id="both-tool-call-array-spellings",
        apply=True,
    )

    assert result.scanned == 1
    assert result.eligible == 1
    assert result.imported == 1
    assert result.invalid_rows == 0
    assert result.skipped_empty == 0
    conn = sqlite3.connect(target_db)
    rows = conn.execute("SELECT role, content, tool_calls FROM messages ORDER BY store_id").fetchall()
    conn.close()

    assert rows[0][0:2] == ("assistant", None)
    assert json.loads(rows[0][2]) == [
        {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
    ]


def test_jsonl_import_deduplicates_same_tool_call_from_both_array_spellings(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "duplicate-tool-call-array-spellings.jsonl"
    target_db = tmp_path / "target-lcm.db"
    call = {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
    write_jsonl_session(
        session_file,
        [
            jsonl_header("duplicate-tool-call-array-spellings"),
            jsonl_message("m1", "assistant", None, tool_calls=[call], toolCalls=[call]),
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id="duplicate-tool-call-array-spellings",
        apply=True,
    )

    assert result.scanned == 1
    assert result.eligible == 1
    assert result.imported == 1
    assert result.invalid_rows == 0
    conn = sqlite3.connect(target_db)
    rows = conn.execute("SELECT role, content, tool_calls FROM messages ORDER BY store_id").fetchall()
    conn.close()

    assert rows[0][0:2] == ("assistant", None)
    assert json.loads(rows[0][2]) == [
        {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
    ]


def test_jsonl_import_rejects_malformed_tool_calls_when_both_array_spellings_are_present(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "both-tool-call-array-spellings-malformed.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            jsonl_header("both-tool-call-array-spellings-malformed"),
            jsonl_message(
                "m1",
                "assistant",
                None,
                tool_calls=[],
                toolCalls=[
                    {"id": "call_1", "type": ["function"], "function": {"name": "lookup", "arguments": "{}"}},
                ],
            ),
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id="both-tool-call-array-spellings-malformed",
        apply=True,
    )

    assert result.scanned == 1
    assert result.eligible == 0
    assert result.imported == 0
    assert result.invalid_rows == 1
    assert result.skipped_empty == 0
    assert any("message tool_calls item missing tool call id or name" in warning for warning in result.warnings)
    assert not target_db.exists()


def test_jsonl_import_rejects_non_string_message_tool_call_type_and_skips_output(
    tmp_path: Path,
):
    importer = load_importer_module()
    session_file = tmp_path / "non-string-tool-call-type.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            {"type": "session", "id": "s"},
            {
                "type": "message",
                "id": "bad",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_bad",
                            "type": ["function"],
                            "function": {"name": "lookup", "arguments": "{}"},
                        },
                    ],
                },
            },
            {"type": "function_call_output", "call_id": "call_bad", "output": "orphan"},
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id="non-string-tool-call-type",
        apply=True,
    )

    assert result.scanned == 2
    assert result.eligible == 0
    assert result.imported == 0
    assert result.invalid_rows == 1
    assert result.skipped_empty == 0
    assert any("message tool_calls item missing tool call id or name" in warning for warning in result.warnings)
    assert not target_db.exists()


def test_jsonl_import_rejects_non_list_message_tool_calls(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "non-list-tool-calls.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            jsonl_header("non-list-tool-calls"),
            {
                "type": "message",
                "id": "m1",
                "message": {
                    "role": "assistant",
                    "content": "text",
                    "tool_calls": {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    },
                },
            },
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id="non-list-tool-calls",
        apply=True,
    )

    assert result.scanned == 1
    assert result.eligible == 0
    assert result.imported == 0
    assert result.invalid_rows == 1
    assert result.skipped_empty == 0
    assert any("message tool_calls must be a list" in warning for warning in result.warnings)
    assert not target_db.exists()


def test_jsonl_import_maps_native_responses_function_call_and_output_items(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "responses-native-items.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            jsonl_header("responses-native-items"),
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "lookup",
                "arguments": "{}",
            },
            {"type": "function_call_output", "call_id": "call_1", "output": "result"},
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file], target_db=target_db, import_id="responses-native-items", apply=True
    )

    assert result.scanned == 2
    assert result.eligible == 2
    assert result.imported == 2
    assert result.invalid_rows == 0
    assert result.skipped_empty == 0
    conn = sqlite3.connect(target_db)
    rows = conn.execute(
        "SELECT role, content, tool_call_id, tool_calls, tool_name FROM messages ORDER BY store_id"
    ).fetchall()
    import_keys = conn.execute(
        "SELECT source_message_key FROM lcm_imported_messages ORDER BY target_store_id"
    ).fetchall()
    conn.close()

    assert rows[0][0:3] == ("assistant", None, None)
    assert json.loads(rows[0][3]) == [
        {"id": "call_1", "type": "function_call", "function": {"name": "lookup", "arguments": "{}"}}
    ]
    assert rows[0][4] is None
    assert rows[1] == ("tool", "result", "call_1", None, None)
    assert import_keys == [
        (jsonl_key("responses-native-items", "fc_1"),),
        (
            jsonl_key(
                "responses-native-items",
                jsonl_result_row_id("function_call_output", "call_1", "result"),
            ),
        ),
    ]


def test_jsonl_import_maps_bare_untyped_responses_function_call_and_output_items(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "responses-bare-items.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            jsonl_header("responses-bare-items"),
            {"call_id": "call_1", "name": "lookup", "arguments": {}},
            {"call_id": "call_1", "output": "result"},
        ],
    )

    first = importer.import_jsonl_sessions(
        files=[session_file], target_db=target_db, import_id="responses-bare-items", apply=True
    )
    second = importer.import_jsonl_sessions(
        files=[session_file], target_db=target_db, import_id="responses-bare-items", apply=True
    )

    assert first.scanned == 2
    assert first.eligible == 2
    assert first.imported == 2
    assert first.invalid_rows == 0
    assert second.imported == 0
    assert second.skipped_existing == 2
    conn = sqlite3.connect(target_db)
    rows = conn.execute(
        "SELECT role, content, tool_call_id, tool_calls, tool_name FROM messages ORDER BY store_id"
    ).fetchall()
    conn.close()

    assert rows[0][0:3] == ("assistant", None, None)
    assert json.loads(rows[0][3]) == [
        {"id": "call_1", "type": "function_call", "function": {"name": "lookup", "arguments": "{}"}}
    ]
    assert rows[1] == ("tool", "result", "call_1", None, None)


def test_jsonl_import_maps_bare_content_responses_function_output_item(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "responses-bare-content-output.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            jsonl_header("s"),
            {"type": "function_call", "id": "fc", "call_id": "call_1", "name": "lookup", "arguments": {}},
            {"call_id": "call_1", "content": "result"},
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id="responses-bare-content-output",
        apply=True,
    )

    assert result.scanned == 2
    assert result.eligible == 2
    assert result.imported == 2
    assert result.invalid_rows == 0
    conn = sqlite3.connect(target_db)
    rows = conn.execute(
        "SELECT role, content, tool_call_id, tool_calls, tool_name FROM messages ORDER BY store_id"
    ).fetchall()
    import_keys = conn.execute(
        "SELECT source_message_key FROM lcm_imported_messages ORDER BY target_store_id"
    ).fetchall()
    conn.close()

    assert rows[0][0:3] == ("assistant", None, None)
    assert json.loads(rows[0][3]) == [
        {"id": "call_1", "type": "function_call", "function": {"name": "lookup", "arguments": "{}"}}
    ]
    assert rows[1] == ("tool", "result", "call_1", None, None)
    assert import_keys == [
        (jsonl_key("s", "fc"),),
        (jsonl_key("s", jsonl_result_row_id("function_call_output", "call_1", "result")),),
    ]
    assert all("line:" not in key for (key,) in import_keys)


def test_jsonl_import_maps_typed_wrapped_responses_function_call_and_output_items(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "wrapped-responses.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            jsonl_header("wrapped-responses"),
            {
                "type": "message",
                "id": "fc-env",
                "message": {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "lookup",
                    "arguments": {"q": "x"},
                },
            },
            {
                "type": "custom_message",
                "id": "out-env",
                "message": {"type": "function_call_output", "id": "out_1", "call_id": "call_1", "output": "result"},
            },
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file], target_db=target_db, import_id="wrapped-responses", apply=True
    )

    assert result.scanned == 2
    assert result.eligible == 2
    assert result.imported == 2
    assert result.invalid_rows == 0
    conn = sqlite3.connect(target_db)
    rows = conn.execute(
        "SELECT role, content, tool_call_id, tool_calls, tool_name FROM messages ORDER BY store_id"
    ).fetchall()
    import_keys = conn.execute(
        "SELECT source_message_key FROM lcm_imported_messages ORDER BY target_store_id"
    ).fetchall()
    conn.close()

    assert rows[0][0:3] == ("assistant", None, None)
    assert json.loads(rows[0][3]) == [
        {"id": "call_1", "type": "function_call", "function": {"name": "lookup", "arguments": '{"q":"x"}'}}
    ]
    assert rows[0][4] is None
    assert rows[1] == ("tool", "result", "call_1", None, None)
    assert import_keys == [
        (jsonl_key("wrapped-responses", "fc_1"),),
        (jsonl_key("wrapped-responses", "out_1"),),
    ]


def test_jsonl_import_maps_untyped_wrapped_responses_function_call_and_output_items(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "untyped-wrapped-responses.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            jsonl_header("untyped-wrapped-responses"),
            {
                "id": "fc-env",
                "message": {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "lookup",
                    "arguments": {"q": "x"},
                },
            },
            {
                "id": "out-env",
                "message": {"type": "function_call_output", "call_id": "call_1", "output": "result"},
            },
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file], target_db=target_db, import_id="untyped-wrapped-responses", apply=True
    )

    assert result.scanned == 2
    assert result.eligible == 2
    assert result.imported == 2
    assert result.invalid_rows == 0
    conn = sqlite3.connect(target_db)
    rows = conn.execute(
        "SELECT role, content, tool_call_id, tool_calls, tool_name FROM messages ORDER BY store_id"
    ).fetchall()
    import_keys = conn.execute(
        "SELECT source_message_key FROM lcm_imported_messages ORDER BY target_store_id"
    ).fetchall()
    conn.close()

    assert rows[0][0:3] == ("assistant", None, None)
    assert json.loads(rows[0][3]) == [
        {"id": "call_1", "type": "function_call", "function": {"name": "lookup", "arguments": '{"q":"x"}'}}
    ]
    assert rows[1] == ("tool", "result", "call_1", None, None)
    assert import_keys == [
        (jsonl_key("untyped-wrapped-responses", "fc-env"),),
        (jsonl_key("untyped-wrapped-responses", "out-env"),),
    ]


def test_jsonl_import_preserves_untyped_top_level_message_with_nested_metadata(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "untyped-top-level-with-message-metadata.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            jsonl_header("untyped-top-level-with-message-metadata"),
            {
                "id": "m1",
                "role": "user",
                "content": "visible top-level content",
                "message": {"type": "reasoning", "summary": "auxiliary metadata"},
            },
        ],
    )

    first = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id="untyped-top-level-with-message-metadata",
        apply=True,
    )
    second = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id="untyped-top-level-with-message-metadata",
        apply=True,
    )

    assert first.scanned == 1
    assert first.eligible == 1
    assert first.imported == 1
    assert first.invalid_rows == 0
    assert second.imported == 0
    assert second.skipped_existing == 1
    conn = sqlite3.connect(target_db)
    rows = conn.execute("SELECT role, content FROM messages ORDER BY store_id").fetchall()
    import_keys = conn.execute(
        "SELECT source_message_key FROM lcm_imported_messages ORDER BY target_store_id"
    ).fetchall()
    conn.close()

    assert rows == [("user", "visible top-level content")]
    assert import_keys == [(jsonl_key("untyped-top-level-with-message-metadata", "m1"),)]


@pytest.mark.parametrize("wrapper_type", ["message", "custom_message"])
def test_jsonl_import_preserves_typed_top_level_message_with_nested_metadata(
    tmp_path: Path,
    wrapper_type: str,
):
    importer = load_importer_module()
    session_file = tmp_path / f"{wrapper_type}-top-level-with-message-metadata.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            {"type": "session", "id": "s"},
            {
                "type": wrapper_type,
                "id": "m1",
                "role": "user",
                "content": "visible top-level content",
                "message": {"type": "reasoning", "summary": "auxiliary metadata"},
            },
        ],
    )

    first = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id=f"{wrapper_type}-top-level-with-message-metadata",
        apply=True,
    )
    second = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id=f"{wrapper_type}-top-level-with-message-metadata",
        apply=True,
    )

    assert first.scanned == 1
    assert first.eligible == 1
    assert first.imported == 1
    assert first.invalid_rows == 0
    assert first.skipped_empty == 0
    assert second.imported == 0
    assert second.skipped_existing == 1
    conn = sqlite3.connect(target_db)
    rows = conn.execute("SELECT role, content FROM messages ORDER BY store_id").fetchall()
    import_keys = conn.execute(
        "SELECT source_message_key FROM lcm_imported_messages ORDER BY target_store_id"
    ).fetchall()
    conn.close()

    assert rows == [("user", "visible top-level content")]
    assert import_keys == [(jsonl_key("s", "m1"),)]


@pytest.mark.parametrize("wrapper_type", ["message", "custom_message"])
def test_jsonl_import_preserves_typed_top_level_message_when_nested_role_metadata_has_no_content(
    tmp_path: Path,
    wrapper_type: str,
):
    importer = load_importer_module()
    session_file = tmp_path / f"{wrapper_type}-top-level-with-role-metadata.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            {"type": "session", "id": "s"},
            {
                "type": wrapper_type,
                "id": "m1",
                "role": "user",
                "content": "visible top-level content",
                "message": {"role": "user", "metadata": {"aux": True}},
            },
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id=f"{wrapper_type}-top-level-with-role-metadata",
        apply=True,
    )

    assert result.scanned == 1
    assert result.eligible == 1
    assert result.imported == 1
    assert result.skipped_empty == 0
    assert result.invalid_rows == 0
    conn = sqlite3.connect(target_db)
    rows = conn.execute("SELECT role, content FROM messages ORDER BY store_id").fetchall()
    conn.close()

    assert rows == [("user", "visible top-level content")]


def test_jsonl_import_does_not_alias_nested_metadata_id_over_top_level_message(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "typed-top-level-metadata-id-collision.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            {"type": "session", "id": "s"},
            {
                "type": "message",
                "id": "m1",
                "role": "user",
                "content": "root",
                "message": {"id": "m2", "type": "reasoning", "summary": "aux"},
            },
            {
                "type": "message",
                "id": "env2",
                "message": {"id": "m2", "parentId": "m1", "role": "assistant", "content": "leaf"},
            },
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id="typed-top-level-metadata-id-collision",
        apply=True,
    )

    assert result.scanned == 2
    assert result.eligible == 2
    assert result.imported == 2
    assert result.invalid_rows == 0
    conn = sqlite3.connect(target_db)
    rows = conn.execute("SELECT role, content FROM messages ORDER BY store_id").fetchall()
    import_keys = conn.execute(
        "SELECT source_message_key FROM lcm_imported_messages ORDER BY target_store_id"
    ).fetchall()
    conn.close()

    assert rows == [("user", "root"), ("assistant", "leaf")]
    assert import_keys == [(jsonl_key("s", "m1"),), (jsonl_key("s", "m2"),)]


@pytest.mark.parametrize("tool_calls_key", ["tool_calls", "toolCalls"])
def test_jsonl_import_empty_tool_calls_metadata_does_not_prune_real_leaf(
    tmp_path: Path,
    tool_calls_key: str,
):
    importer = load_importer_module()
    session_file = tmp_path / f"empty-{tool_calls_key}-metadata-branch.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            {"type": "session", "id": "s"},
            {"type": "message", "id": "m1", "message": {"role": "user", "content": "root"}},
            {
                "type": "message",
                "id": "m2",
                "parentId": "m1",
                "message": {"role": "assistant", "content": "current"},
            },
            {
                "type": "message",
                "id": "meta",
                "parentId": "m1",
                "message": {"type": "reasoning", "content": "tail metadata", tool_calls_key: []},
            },
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id=f"empty-{tool_calls_key}-metadata-branch",
        apply=True,
    )

    assert result.scanned == 3
    assert result.eligible == 2
    assert result.imported == 2
    assert result.skipped_empty == 1
    assert result.invalid_rows == 0
    conn = sqlite3.connect(target_db)
    rows = conn.execute("SELECT role, content FROM messages ORDER BY store_id").fetchall()
    import_keys = conn.execute(
        "SELECT source_message_key FROM lcm_imported_messages ORDER BY target_store_id"
    ).fetchall()
    conn.close()

    assert rows == [("user", "root"), ("assistant", "current")]
    assert import_keys == [(jsonl_key("s", "m1"),), (jsonl_key("s", "m2"),)]


def test_jsonl_import_prunes_untyped_wrapped_native_call_branch_by_nested_parent(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "untyped-wrapped-native-call-branch.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            {"type": "session", "id": "s"},
            {"id": "env-root", "message": {"id": "root", "role": "user", "content": "root"}},
            {
                "id": "env-old",
                "message": {"id": "old", "parentId": "root", "role": "assistant", "content": "old branch"},
            },
            {
                "id": "env-call",
                "message": {
                    "id": "nested-call",
                    "type": "function_call",
                    "parentId": "root",
                    "call_id": "call_1",
                    "name": "lookup",
                    "arguments": {},
                },
            },
            {
                "id": "env-final",
                "message": {"id": "final", "parentId": "nested-call", "role": "assistant", "content": "final"},
            },
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id="untyped-wrapped-native-call-branch",
        apply=True,
    )

    assert result.scanned == 4
    assert result.eligible == 3
    assert result.imported == 3
    assert result.invalid_rows == 0
    conn = sqlite3.connect(target_db)
    rows = conn.execute(
        "SELECT role, content, tool_calls FROM messages ORDER BY store_id"
    ).fetchall()
    import_keys = conn.execute(
        "SELECT source_message_key FROM lcm_imported_messages ORDER BY target_store_id"
    ).fetchall()
    conn.close()

    assert rows[0] == ("user", "root", None)
    assert rows[1][0:2] == ("assistant", None)
    assert json.loads(rows[1][2]) == [
        {"id": "call_1", "type": "function_call", "function": {"name": "lookup", "arguments": "{}"}}
    ]
    assert rows[2] == ("assistant", "final", None)
    assert import_keys == [
        (jsonl_key("s", "env-root"),),
        (jsonl_key("s", "env-call"),),
        (jsonl_key("s", "env-final"),),
    ]


def test_jsonl_import_rejects_non_string_nested_message_type(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "nested-nonstring.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            jsonl_header("nested-nonstring"),
            {
                "type": "message",
                "id": "bad-result",
                "message": {"type": ["toolResult"], "toolCallId": "call_1", "content": "result"},
            },
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file], target_db=target_db, import_id="nested-nonstring", apply=True
    )

    assert result.scanned == 1
    assert result.eligible == 0
    assert result.imported == 0
    assert result.invalid_rows == 1
    assert any("nested-nonstring.jsonl:2" in warning for warning in result.warnings)
    assert any("non-string nested type" in warning for warning in result.warnings)
    assert not target_db.exists()


def test_jsonl_import_imports_responses_function_output_from_later_catchup(
    tmp_path: Path,
):
    importer = load_importer_module()
    first_file = tmp_path / "responses-catchup-1.jsonl"
    second_file = tmp_path / "responses-catchup-2.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        first_file,
        [
            jsonl_header("responses-catchup"),
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "lookup",
                "arguments": "{}",
            },
        ],
    )

    first = importer.import_jsonl_sessions(
        files=[first_file],
        target_db=target_db,
        import_id="responses-catchup",
        apply=True,
    )
    write_jsonl_session(
        second_file,
        [
            {"type": "function_call_output", "call_id": "call_1", "output": "result"},
        ],
    )
    second = importer.import_jsonl_sessions(
        files=[second_file],
        target_db=target_db,
        import_id="responses-catchup",
        apply=True,
    )

    assert first.imported == 1
    assert second.scanned == 1
    assert second.eligible == 1
    assert second.imported == 1
    assert second.invalid_rows == 0
    conn = sqlite3.connect(target_db)
    rows = conn.execute(
        "SELECT role, content, tool_call_id, tool_calls, tool_name FROM messages ORDER BY store_id"
    ).fetchall()
    import_keys = conn.execute(
        "SELECT source_message_key FROM lcm_imported_messages ORDER BY target_store_id"
    ).fetchall()
    conn.close()

    assert rows[0][0:3] == ("assistant", None, None)
    assert json.loads(rows[0][3]) == [
        {"id": "call_1", "type": "function_call", "function": {"name": "lookup", "arguments": "{}"}}
    ]
    assert rows[1] == ("tool", "result", "call_1", None, None)
    assert import_keys == [
        (jsonl_key("responses-catchup", "fc_1"),),
        (
            jsonl_key(
                "responses-catchup",
                jsonl_result_row_id("function_call_output", "call_1", "result"),
            ),
        ),
    ]


def test_jsonl_import_resolves_responses_output_before_call_file_in_same_run(
    tmp_path: Path,
):
    importer = load_importer_module()
    output_file = tmp_path / "a-output.jsonl"
    call_file = tmp_path / "b-call.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        output_file,
        [{"type": "function_call_output", "call_id": "call_1", "output": "result"}],
    )
    write_jsonl_session(
        call_file,
        [
            jsonl_header("catchup-order"),
            {
                "type": "function_call",
                "id": "fc1",
                "call_id": "call_1",
                "name": "lookup",
                "arguments": {},
            },
        ],
    )

    first = importer.import_jsonl_sessions(
        files=[output_file, call_file],
        target_db=target_db,
        import_id="responses-catchup-order",
        apply=True,
    )
    second = importer.import_jsonl_sessions(
        files=[output_file, call_file],
        target_db=target_db,
        import_id="responses-catchup-order",
        apply=True,
    )

    assert first.scanned == 2
    assert first.eligible == 2
    assert first.imported == 2
    assert first.invalid_rows == 0
    assert second.imported == 0
    assert second.skipped_existing == 2
    conn = sqlite3.connect(target_db)
    rows = conn.execute(
        "SELECT role, content, tool_call_id, tool_calls, tool_name FROM messages ORDER BY store_id"
    ).fetchall()
    import_rows = conn.execute(
        """SELECT source_session, source_message_key
           FROM lcm_imported_messages
           ORDER BY source_message_key"""
    ).fetchall()
    conn.close()

    assert rows[0][0:3] == ("assistant", None, None)
    assert json.loads(rows[0][3]) == [
        {"id": "call_1", "type": "function_call", "function": {"name": "lookup", "arguments": "{}"}}
    ]
    assert rows[1] == ("tool", "result", "call_1", None, None)
    assert import_rows == [
        ("catchup-order", jsonl_key("catchup-order", "fc1")),
        (
            "catchup-order",
            jsonl_key(
                "catchup-order",
                jsonl_result_row_id("function_call_output", "call_1", "result"),
            ),
        ),
    ]


def test_jsonl_import_idless_native_function_output_key_survives_line_shift(
    tmp_path: Path,
):
    importer = load_importer_module()
    session_file = tmp_path / "responses-native-line-shift.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            jsonl_header("responses-native-line-shift"),
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "lookup",
                "arguments": "{}",
            },
            {"type": "function_call_output", "call_id": "call_1", "output": "result"},
        ],
    )

    first = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id="responses-native-line-shift",
        apply=True,
    )
    write_jsonl_session(
        session_file,
        [
            jsonl_header("responses-native-line-shift"),
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "lookup",
                "arguments": "{}",
            },
            {"type": "reasoning", "id": "rs_1", "summary": "inserted metadata"},
            {"type": "function_call_output", "call_id": "call_1", "output": "result"},
        ],
    )
    second = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id="responses-native-line-shift",
        apply=True,
    )

    assert first.imported == 2
    assert second.scanned == 3
    assert second.eligible == 2
    assert second.imported == 0
    assert second.skipped_existing == 2
    conn = sqlite3.connect(target_db)
    import_keys = conn.execute(
        "SELECT source_message_key FROM lcm_imported_messages ORDER BY target_store_id"
    ).fetchall()
    message_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    conn.close()

    assert message_count == 2
    assert import_keys == [
        (jsonl_key("responses-native-line-shift", "fc_1"),),
        (
            jsonl_key(
                "responses-native-line-shift",
                jsonl_result_row_id("function_call_output", "call_1", "result"),
            ),
        ),
    ]
    assert all("line:" not in key for (key,) in import_keys)


def test_jsonl_import_idless_responses_function_call_key_survives_line_shift(
    tmp_path: Path,
):
    importer = load_importer_module()
    session_file = tmp_path / "responses-idless-call-line-shift.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            jsonl_header("responses-idless-call-line-shift"),
            {"type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": {}},
            {"type": "function_call_output", "call_id": "call_1", "output": "result"},
        ],
    )

    first = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id="responses-idless-call-line-shift",
        apply=True,
    )
    write_jsonl_session(
        session_file,
        [
            jsonl_header("responses-idless-call-line-shift"),
            {"type": "reasoning", "id": "rs_1", "summary": "inserted metadata"},
            {"type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": {}},
            {"type": "function_call_output", "call_id": "call_1", "output": "result"},
        ],
    )
    second = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id="responses-idless-call-line-shift",
        apply=True,
    )

    assert first.imported == 2
    assert second.scanned == 3
    assert second.eligible == 2
    assert second.imported == 0
    assert second.skipped_existing == 2
    conn = sqlite3.connect(target_db)
    rows = conn.execute(
        "SELECT role, content, tool_call_id, tool_calls, tool_name FROM messages ORDER BY store_id"
    ).fetchall()
    import_keys = conn.execute(
        "SELECT source_message_key FROM lcm_imported_messages ORDER BY target_store_id"
    ).fetchall()
    conn.close()

    assert rows[0][0:3] == ("assistant", None, None)
    assert json.loads(rows[0][3]) == [
        {"id": "call_1", "type": "function_call", "function": {"name": "lookup", "arguments": "{}"}}
    ]
    assert rows[1] == ("tool", "result", "call_1", None, None)
    assert import_keys == [
        (
            jsonl_key(
                "responses-idless-call-line-shift",
                jsonl_tool_call_row_id("function_call", "call_1", "lookup", {}),
            ),
        ),
        (
            jsonl_key(
                "responses-idless-call-line-shift",
                jsonl_result_row_id("function_call_output", "call_1", "result"),
            ),
        ),
    ]
    assert all("line:" not in key for (key,) in import_keys)


def test_jsonl_import_idless_responses_function_call_skips_legacy_line_key_catchup(
    tmp_path: Path,
):
    importer = load_importer_module()
    session_file = tmp_path / "responses-idless-call-legacy-catchup.jsonl"
    target_db = tmp_path / "target-lcm.db"
    import_id = "responses-idless-call-legacy-catchup"
    write_jsonl_session(
        session_file,
        [
            {"type": "session", "id": "s"},
            {"type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": {}},
            {"type": "function_call_output", "call_id": "call_1", "output": "result"},
        ],
    )

    first = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id=import_id,
        apply=True,
    )
    conn = sqlite3.connect(target_db)
    target_store_ids = [
        store_id
        for (store_id,) in conn.execute(
            "SELECT target_store_id FROM lcm_imported_messages ORDER BY target_store_id"
        ).fetchall()
    ]
    for target_store_id, row_id in zip(target_store_ids, ["line:2", "line:3"], strict=True):
        source_message_key = jsonl_key("s", row_id)
        conn.execute(
            """UPDATE lcm_imported_messages
               SET source_message_id = ?, source_message_key = ?
               WHERE import_id = ? AND target_store_id = ?""",
            (
                importer._stable_positive_int(source_message_key),
                source_message_key,
                import_id,
                target_store_id,
            ),
        )
    conn.commit()
    conn.close()

    second = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id=import_id,
        apply=True,
    )

    assert first.imported == 2
    assert second.scanned == 2
    assert second.eligible == 2
    assert second.imported == 0
    assert second.skipped_existing == 2
    assert second.invalid_rows == 0
    conn = sqlite3.connect(target_db)
    message_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    import_keys = conn.execute(
        "SELECT source_message_key FROM lcm_imported_messages ORDER BY target_store_id"
    ).fetchall()
    conn.close()
    assert message_count == 2
    assert import_keys == [(jsonl_key("s", "line:2"),), (jsonl_key("s", "line:3"),)]


def test_jsonl_import_idless_responses_function_call_skips_legacy_grouped_line_key_catchup(
    tmp_path: Path,
):
    importer = load_importer_module()
    session_file = tmp_path / "responses-idless-grouped-call-legacy-catchup.jsonl"
    target_db = tmp_path / "target-lcm.db"
    import_id = "responses-idless-grouped-call-legacy-catchup"
    write_jsonl_session(
        session_file,
        [
            {"type": "session", "id": "s"},
            {"type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": {}},
            {"type": "function_call", "call_id": "call_2", "name": "search", "arguments": {}},
        ],
    )

    first = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id=import_id,
        apply=True,
    )
    legacy_grouped_row_id = "function_calls:" + json.dumps(
        ["line:2", "line:3"],
        separators=(",", ":"),
        ensure_ascii=False,
    )
    legacy_grouped_key = jsonl_key("s", legacy_grouped_row_id)
    conn = sqlite3.connect(target_db)
    target_store_id = conn.execute(
        "SELECT target_store_id FROM lcm_imported_messages WHERE import_id = ?",
        (import_id,),
    ).fetchone()[0]
    conn.execute(
        """UPDATE lcm_imported_messages
           SET source_message_id = ?, source_message_key = ?
           WHERE import_id = ? AND target_store_id = ?""",
        (
            importer._stable_positive_int(legacy_grouped_key),
            legacy_grouped_key,
            import_id,
            target_store_id,
        ),
    )
    conn.commit()
    conn.close()

    second = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id=import_id,
        apply=True,
    )

    assert first.scanned == 2
    assert first.eligible == 1
    assert first.imported == 1
    assert second.scanned == 2
    assert second.eligible == 1
    assert second.imported == 0
    assert second.skipped_existing == 1
    assert second.invalid_rows == 0
    conn = sqlite3.connect(target_db)
    rows = conn.execute("SELECT role, content, tool_calls FROM messages ORDER BY store_id").fetchall()
    import_keys = conn.execute(
        "SELECT source_message_key FROM lcm_imported_messages ORDER BY target_store_id"
    ).fetchall()
    conn.close()

    assert len(rows) == 1
    assert rows[0][0:2] == ("assistant", None)
    assert json.loads(rows[0][2]) == [
        {"id": "call_1", "type": "function_call", "function": {"name": "lookup", "arguments": "{}"}},
        {"id": "call_2", "type": "function_call", "function": {"name": "search", "arguments": "{}"}},
    ]
    assert import_keys == [(legacy_grouped_key,)]


def test_jsonl_import_appended_responses_function_call_does_not_reimport_existing_sibling(
    tmp_path: Path,
):
    importer = load_importer_module()
    session_file = tmp_path / "responses-native-appended-sibling.jsonl"
    target_db = tmp_path / "target-lcm.db"
    import_id = "responses-native-appended-sibling"
    first_rows = [
        {"type": "session", "id": "s"},
        {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "lookup", "arguments": {}},
    ]
    write_jsonl_session(session_file, first_rows)

    first = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id=import_id,
        apply=True,
    )
    write_jsonl_session(
        session_file,
        [
            *first_rows,
            {"type": "function_call", "id": "fc_2", "call_id": "call_2", "name": "search", "arguments": {}},
        ],
    )
    second = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id=import_id,
        apply=True,
    )

    assert first.scanned == 1
    assert first.eligible == 1
    assert first.imported == 1
    assert second.scanned == 2
    assert second.eligible == 2
    assert second.imported == 1
    assert second.skipped_existing == 1
    assert second.invalid_rows == 0
    conn = sqlite3.connect(target_db)
    rows = conn.execute("SELECT role, content, tool_calls FROM messages ORDER BY store_id").fetchall()
    import_keys = conn.execute(
        "SELECT source_message_key FROM lcm_imported_messages ORDER BY target_store_id"
    ).fetchall()
    conn.close()

    assert rows[0][0:2] == ("assistant", None)
    assert json.loads(rows[0][2]) == [
        {"id": "call_1", "type": "function_call", "function": {"name": "lookup", "arguments": "{}"}}
    ]
    assert rows[1][0:2] == ("assistant", None)
    assert json.loads(rows[1][2]) == [
        {"id": "call_2", "type": "function_call", "function": {"name": "search", "arguments": "{}"}}
    ]
    assert import_keys == [(jsonl_key("s", "fc_1"),), (jsonl_key("s", "fc_2"),)]


def test_jsonl_import_malformed_row_does_not_split_existing_responses_group_on_catchup(
    tmp_path: Path,
):
    importer = load_importer_module()
    session_file = tmp_path / "responses-native-malformed-row-catchup.jsonl"
    target_db = tmp_path / "target-lcm.db"
    import_id = "responses-native-malformed-row-catchup"
    first_rows = [
        {"type": "session", "id": "s"},
        {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "lookup", "arguments": {}},
        {"type": "function_call", "id": "fc_2", "call_id": "call_2", "name": "search", "arguments": {}},
    ]
    write_jsonl_session(session_file, first_rows)

    first = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id=import_id,
        apply=True,
    )
    write_jsonl_session(
        session_file,
        [
            *first_rows[:2],
            "{bad json",
            first_rows[2],
        ],
    )
    second = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id=import_id,
        apply=True,
    )

    assert first.scanned == 2
    assert first.eligible == 1
    assert first.imported == 1
    assert first.invalid_rows == 0
    assert second.scanned == 3
    assert second.eligible == 1
    assert second.imported == 0
    assert second.skipped_existing == 1
    assert second.invalid_rows == 1
    assert any("responses-native-malformed-row-catchup.jsonl:3: invalid JSON" in warning for warning in second.warnings)
    conn = sqlite3.connect(target_db)
    rows = conn.execute("SELECT role, content, tool_calls FROM messages ORDER BY store_id").fetchall()
    import_keys = conn.execute(
        "SELECT source_message_key FROM lcm_imported_messages ORDER BY target_store_id"
    ).fetchall()
    conn.close()

    assert len(rows) == 1
    assert rows[0][0:2] == ("assistant", None)
    assert json.loads(rows[0][2]) == [
        {"id": "call_1", "type": "function_call", "function": {"name": "lookup", "arguments": "{}"}},
        {"id": "call_2", "type": "function_call", "function": {"name": "search", "arguments": "{}"}},
    ]
    assert import_keys == [(jsonl_key("s", 'function_calls:["fc_1","fc_2"]'),)]


def test_jsonl_import_malformed_json_object_does_not_split_existing_responses_group_on_catchup(
    tmp_path: Path,
):
    importer = load_importer_module()
    session_file = tmp_path / "responses-native-malformed-object-catchup.jsonl"
    target_db = tmp_path / "target-lcm.db"
    import_id = "responses-native-malformed-object-catchup"
    first_rows = [
        {"type": "session", "id": "s"},
        {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "lookup", "arguments": {}},
        {"type": "function_call", "id": "fc_2", "call_id": "call_2", "name": "search", "arguments": {}},
    ]
    write_jsonl_session(session_file, first_rows)

    first = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id=import_id,
        apply=True,
    )
    write_jsonl_session(
        session_file,
        [
            *first_rows[:2],
            {"type": ["reasoning"], "id": "bad"},
            first_rows[2],
        ],
    )
    second = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id=import_id,
        apply=True,
    )

    assert first.scanned == 2
    assert first.eligible == 1
    assert first.imported == 1
    assert first.invalid_rows == 0
    assert second.scanned == 3
    assert second.eligible == 1
    assert second.imported == 0
    assert second.skipped_existing == 1
    assert second.invalid_rows == 1
    assert any(
        "responses-native-malformed-object-catchup.jsonl:3: unsupported row shape" in warning
        for warning in second.warnings
    )
    conn = sqlite3.connect(target_db)
    rows = conn.execute("SELECT role, content, tool_calls FROM messages ORDER BY store_id").fetchall()
    import_keys = conn.execute(
        "SELECT source_message_key FROM lcm_imported_messages ORDER BY target_store_id"
    ).fetchall()
    conn.close()

    assert len(rows) == 1
    assert rows[0][0:2] == ("assistant", None)
    assert json.loads(rows[0][2]) == [
        {"id": "call_1", "type": "function_call", "function": {"name": "lookup", "arguments": "{}"}},
        {"id": "call_2", "type": "function_call", "function": {"name": "search", "arguments": "{}"}},
    ]
    assert import_keys == [(jsonl_key("s", 'function_calls:["fc_1","fc_2"]'),)]


def test_jsonl_import_nested_non_string_type_does_not_split_existing_responses_group_on_catchup(
    tmp_path: Path,
):
    importer = load_importer_module()
    session_file = tmp_path / "responses-native-nested-nonstring-type-catchup.jsonl"
    target_db = tmp_path / "target-lcm.db"
    import_id = "responses-native-nested-nonstring-type-catchup"
    first_rows = [
        {"type": "session", "id": "s"},
        {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "lookup", "arguments": {}},
        {"type": "function_call", "id": "fc_2", "call_id": "call_2", "name": "search", "arguments": {}},
    ]
    write_jsonl_session(session_file, first_rows)

    first = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id=import_id,
        apply=True,
    )
    write_jsonl_session(
        session_file,
        [
            *first_rows[:2],
            {"type": "message", "id": "bad", "message": {"type": ["reasoning"]}},
            first_rows[2],
        ],
    )
    second = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id=import_id,
        apply=True,
    )

    assert first.scanned == 2
    assert first.eligible == 1
    assert first.imported == 1
    assert first.invalid_rows == 0
    assert second.scanned == 3
    assert second.eligible == 1
    assert second.imported == 0
    assert second.skipped_existing == 1
    assert second.invalid_rows == 1
    assert any(
        "responses-native-nested-nonstring-type-catchup.jsonl:3: message row has non-string nested type" in warning
        for warning in second.warnings
    )
    conn = sqlite3.connect(target_db)
    rows = conn.execute("SELECT role, content, tool_calls FROM messages ORDER BY store_id").fetchall()
    import_keys = conn.execute(
        "SELECT source_message_key FROM lcm_imported_messages ORDER BY target_store_id"
    ).fetchall()
    conn.close()

    assert len(rows) == 1
    assert rows[0][0:2] == ("assistant", None)
    assert json.loads(rows[0][2]) == [
        {"id": "call_1", "type": "function_call", "function": {"name": "lookup", "arguments": "{}"}},
        {"id": "call_2", "type": "function_call", "function": {"name": "search", "arguments": "{}"}},
    ]
    assert import_keys == [(jsonl_key("s", 'function_calls:["fc_1","fc_2"]'),)]


def test_jsonl_import_unsupported_metadata_does_not_split_existing_responses_group_on_catchup(
    tmp_path: Path,
):
    importer = load_importer_module()
    session_file = tmp_path / "responses-native-unsupported-metadata-catchup.jsonl"
    target_db = tmp_path / "target-lcm.db"
    import_id = "responses-native-unsupported-metadata-catchup"
    first_rows = [
        {"type": "session", "id": "s"},
        {"type": "message", "id": "root", "message": {"role": "user", "content": "root"}},
        {
            "type": "function_call",
            "id": "fc_1",
            "parentId": "root",
            "call_id": "call_1",
            "name": "lookup",
            "arguments": {},
        },
        {
            "type": "function_call",
            "id": "fc_2",
            "parentId": "root",
            "call_id": "call_2",
            "name": "search",
            "arguments": {},
        },
    ]
    write_jsonl_session(session_file, first_rows)

    first = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id=import_id,
        apply=True,
    )
    write_jsonl_session(
        session_file,
        [
            *first_rows[:3],
            {"type": "reasoning", "id": "meta", "parentId": "other", "summary": "m"},
            first_rows[3],
        ],
    )
    second = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id=import_id,
        apply=True,
    )

    assert first.scanned == 3
    assert first.eligible == 2
    assert first.imported == 2
    assert first.invalid_rows == 0
    assert second.scanned == 4
    assert second.eligible == 2
    assert second.imported == 0
    assert second.skipped_existing == 2
    assert second.invalid_rows == 0
    conn = sqlite3.connect(target_db)
    rows = conn.execute("SELECT role, content, tool_calls FROM messages ORDER BY store_id").fetchall()
    import_keys = conn.execute(
        "SELECT source_message_key FROM lcm_imported_messages ORDER BY target_store_id"
    ).fetchall()
    conn.close()

    assert len(rows) == 2
    assert rows[0] == ("user", "root", None)
    assert rows[1][0:2] == ("assistant", None)
    assert json.loads(rows[1][2]) == [
        {"id": "call_1", "type": "function_call", "function": {"name": "lookup", "arguments": "{}"}},
        {"id": "call_2", "type": "function_call", "function": {"name": "search", "arguments": "{}"}},
    ]
    assert import_keys == [
        (jsonl_key("s", "root"),),
        (jsonl_key("s", 'function_calls:["fc_1","fc_2"]'),),
    ]


def test_jsonl_import_wrapped_metadata_does_not_split_existing_responses_group_on_catchup(
    tmp_path: Path,
):
    importer = load_importer_module()
    session_file = tmp_path / "responses-native-wrapped-metadata-catchup.jsonl"
    target_db = tmp_path / "target-lcm.db"
    import_id = "responses-native-wrapped-metadata-catchup"
    first_rows = [
        {"type": "session", "id": "s"},
        {"type": "message", "id": "root", "message": {"role": "user", "content": "root"}},
        {
            "type": "function_call",
            "id": "fc_1",
            "parentId": "root",
            "call_id": "call_1",
            "name": "lookup",
            "arguments": {},
        },
        {
            "type": "function_call",
            "id": "fc_2",
            "parentId": "root",
            "call_id": "call_2",
            "name": "search",
            "arguments": {},
        },
    ]
    write_jsonl_session(session_file, first_rows)

    first = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id=import_id,
        apply=True,
    )
    write_jsonl_session(
        session_file,
        [
            *first_rows[:3],
            {"type": "message", "id": "meta", "parentId": "other", "message": {"type": "reasoning", "summary": "m"}},
            first_rows[3],
        ],
    )
    second = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id=import_id,
        apply=True,
    )

    assert first.scanned == 3
    assert first.eligible == 2
    assert first.imported == 2
    assert first.invalid_rows == 0
    assert second.scanned == 4
    assert second.eligible == 2
    assert second.imported == 0
    assert second.skipped_existing == 2
    assert second.skipped_empty == 1
    assert second.invalid_rows == 0
    conn = sqlite3.connect(target_db)
    rows = conn.execute("SELECT role, content, tool_calls FROM messages ORDER BY store_id").fetchall()
    import_keys = conn.execute(
        "SELECT source_message_key FROM lcm_imported_messages ORDER BY target_store_id"
    ).fetchall()
    conn.close()

    assert len(rows) == 2
    assert rows[0] == ("user", "root", None)
    assert rows[1][0:2] == ("assistant", None)
    assert json.loads(rows[1][2]) == [
        {"id": "call_1", "type": "function_call", "function": {"name": "lookup", "arguments": "{}"}},
        {"id": "call_2", "type": "function_call", "function": {"name": "search", "arguments": "{}"}},
    ]
    assert import_keys == [
        (jsonl_key("s", "root"),),
        (jsonl_key("s", 'function_calls:["fc_1","fc_2"]'),),
    ]


def test_jsonl_import_groups_consecutive_native_responses_function_calls(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "responses-native-parallel.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            jsonl_header("responses-native-parallel"),
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "lookup",
                "arguments": {"query": "sammy"},
            },
            {
                "type": "function_call",
                "id": "fc_2",
                "call_id": "call_2",
                "name": "search",
                "arguments": {"query": "memory"},
            },
            {"type": "function_call_output", "call_id": "call_1", "output": "lookup result"},
            {"type": "function_call_output", "call_id": "call_2", "output": "search result"},
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file], target_db=target_db, import_id="responses-native-parallel", apply=True
    )

    assert result.scanned == 4
    assert result.eligible == 3
    assert result.imported == 3
    assert result.invalid_rows == 0
    conn = sqlite3.connect(target_db)
    rows = conn.execute(
        "SELECT role, content, tool_call_id, tool_calls, tool_name FROM messages ORDER BY store_id"
    ).fetchall()
    import_keys = conn.execute(
        "SELECT source_message_key FROM lcm_imported_messages ORDER BY target_store_id"
    ).fetchall()
    conn.close()

    assert rows[0][0:3] == ("assistant", None, None)
    assert json.loads(rows[0][3]) == [
        {
            "id": "call_1",
            "type": "function_call",
            "function": {"name": "lookup", "arguments": '{"query":"sammy"}'},
        },
        {
            "id": "call_2",
            "type": "function_call",
            "function": {"name": "search", "arguments": '{"query":"memory"}'},
        },
    ]
    assert rows[0][4] is None
    assert rows[1] == ("tool", "lookup result", "call_1", None, None)
    assert rows[2] == ("tool", "search result", "call_2", None, None)
    assert import_keys == [
        (jsonl_key("responses-native-parallel", 'function_calls:["fc_1","fc_2"]'),),
        (
            jsonl_key(
                "responses-native-parallel",
                jsonl_result_row_id("function_call_output", "call_1", "lookup result"),
            ),
        ),
        (
            jsonl_key(
                "responses-native-parallel",
                jsonl_result_row_id("function_call_output", "call_2", "search result"),
            ),
        ),
    ]


def test_jsonl_import_keeps_parent_linked_parallel_native_function_calls(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "responses-native-parent-parallel.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            jsonl_header("responses-native-parent-parallel"),
            jsonl_message("m1", "user", "root"),
            {
                "type": "function_call",
                "id": "fc_1",
                "parentId": "m1",
                "call_id": "call_1",
                "name": "lookup",
                "arguments": {"query": "sammy"},
            },
            {
                "type": "function_call",
                "id": "fc_2",
                "parentId": "m1",
                "call_id": "call_2",
                "name": "search",
                "arguments": {"query": "memory"},
            },
            {"type": "function_call_output", "call_id": "call_1", "output": "lookup result"},
            {"type": "function_call_output", "call_id": "call_2", "output": "search result"},
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file], target_db=target_db, import_id="responses-native-parent-parallel", apply=True
    )

    assert result.scanned == 5
    assert result.eligible == 4
    assert result.imported == 4
    assert result.invalid_rows == 0
    conn = sqlite3.connect(target_db)
    rows = conn.execute(
        "SELECT role, content, tool_call_id, tool_calls, tool_name FROM messages ORDER BY store_id"
    ).fetchall()
    import_keys = conn.execute(
        "SELECT source_message_key FROM lcm_imported_messages ORDER BY target_store_id"
    ).fetchall()
    conn.close()

    assert rows[0] == ("user", "root", None, None, None)
    assert rows[1][0:3] == ("assistant", None, None)
    assert json.loads(rows[1][3]) == [
        {
            "id": "call_1",
            "type": "function_call",
            "function": {"name": "lookup", "arguments": '{"query":"sammy"}'},
        },
        {
            "id": "call_2",
            "type": "function_call",
            "function": {"name": "search", "arguments": '{"query":"memory"}'},
        },
    ]
    assert rows[1][4] is None
    assert rows[2] == ("tool", "lookup result", "call_1", None, None)
    assert rows[3] == ("tool", "search result", "call_2", None, None)
    assert import_keys == [
        (jsonl_key("responses-native-parent-parallel", "m1"),),
        (jsonl_key("responses-native-parent-parallel", 'function_calls:["fc_1","fc_2"]'),),
        (
            jsonl_key(
                "responses-native-parent-parallel",
                jsonl_result_row_id("function_call_output", "call_1", "lookup result"),
            ),
        ),
        (
            jsonl_key(
                "responses-native-parent-parallel",
                jsonl_result_row_id("function_call_output", "call_2", "search result"),
            ),
        ),
    ]


def test_jsonl_import_keeps_responses_siblings_with_alias_equivalent_parents(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "responses-native-alias-parent-parallel.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            {"type": "session", "id": "s"},
            {"id": "env-root", "message": {"id": "root", "role": "user", "content": "root"}},
            {
                "type": "function_call",
                "id": "fc_1",
                "parentId": "root",
                "call_id": "call_1",
                "name": "lookup",
                "arguments": {},
            },
            {
                "type": "function_call",
                "id": "fc_2",
                "parentId": "env-root",
                "call_id": "call_2",
                "name": "search",
                "arguments": {},
            },
            {"type": "function_call_output", "call_id": "call_1", "output": "r1"},
            {"type": "function_call_output", "call_id": "call_2", "output": "r2"},
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id="responses-native-alias-parent-parallel",
        apply=True,
    )

    assert result.scanned == 5
    assert result.eligible == 4
    assert result.imported == 4
    assert result.invalid_rows == 0
    conn = sqlite3.connect(target_db)
    rows = conn.execute(
        "SELECT role, content, tool_call_id, tool_calls, tool_name FROM messages ORDER BY store_id"
    ).fetchall()
    import_keys = conn.execute(
        "SELECT source_message_key FROM lcm_imported_messages ORDER BY target_store_id"
    ).fetchall()
    conn.close()

    assert rows[0] == ("user", "root", None, None, None)
    assert rows[1][0:3] == ("assistant", None, None)
    assert json.loads(rows[1][3]) == [
        {"id": "call_1", "type": "function_call", "function": {"name": "lookup", "arguments": "{}"}},
        {"id": "call_2", "type": "function_call", "function": {"name": "search", "arguments": "{}"}},
    ]
    assert rows[1][4] is None
    assert rows[2] == ("tool", "r1", "call_1", None, None)
    assert rows[3] == ("tool", "r2", "call_2", None, None)
    assert import_keys == [
        (jsonl_key("s", "env-root"),),
        (jsonl_key("s", 'function_calls:["fc_1","fc_2"]'),),
        (jsonl_key("s", jsonl_result_row_id("function_call_output", "call_1", "r1")),),
        (jsonl_key("s", jsonl_result_row_id("function_call_output", "call_2", "r2")),),
    ]


@pytest.mark.parametrize("function_call_shape", ["top-level", "wrapped", "bare"])
def test_jsonl_import_splits_pending_responses_function_calls_on_parent_change(
    tmp_path: Path,
    function_call_shape: str,
):
    importer = load_importer_module()
    test_id = f"responses-parent-split-{function_call_shape}"
    session_file = tmp_path / f"{test_id}.jsonl"
    target_db = tmp_path / "target-lcm.db"

    def function_call_row(parent_id: str, call_id: str, name: str) -> dict[str, object]:
        message = {
            "type": "function_call",
            "call_id": call_id,
            "name": name,
            "arguments": {},
        }
        if function_call_shape == "wrapped":
            return {"type": "message", "parentId": parent_id, "message": message}
        if function_call_shape == "bare":
            return {
                "parentId": parent_id,
                "call_id": call_id,
                "name": name,
                "arguments": {},
            }
        return {"type": "function_call", "parentId": parent_id, **message}

    write_jsonl_session(
        session_file,
        [
            jsonl_header(test_id),
            jsonl_message("root", "user", "root"),
            jsonl_message("mid", "assistant", "mid", parent_id="root"),
            function_call_row("root", "call_1", "lookup"),
            function_call_row("mid", "call_2", "search"),
            {"type": "function_call_output", "call_id": "call_1", "output": "r1"},
            {"type": "function_call_output", "call_id": "call_2", "output": "r2"},
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file], target_db=target_db, import_id=test_id, apply=True
    )

    assert result.scanned == 6
    assert result.eligible == 6
    assert result.imported == 6
    assert result.invalid_rows == 0
    conn = sqlite3.connect(target_db)
    rows = conn.execute(
        "SELECT role, content, tool_call_id, tool_calls, tool_name FROM messages ORDER BY store_id"
    ).fetchall()
    conn.close()

    assert rows[0] == ("user", "root", None, None, None)
    assert rows[1] == ("assistant", "mid", None, None, None)
    assert rows[2][0:3] == ("assistant", None, None)
    assert json.loads(rows[2][3]) == [
        {"id": "call_1", "type": "function_call", "function": {"name": "lookup", "arguments": "{}"}}
    ]
    assert rows[3][0:3] == ("assistant", None, None)
    assert json.loads(rows[3][3]) == [
        {"id": "call_2", "type": "function_call", "function": {"name": "search", "arguments": "{}"}}
    ]
    assert rows[4] == ("tool", "r1", "call_1", None, None)
    assert rows[5] == ("tool", "r2", "call_2", None, None)


@pytest.mark.parametrize(
    ("wrapper_type", "metadata_message"),
    [
        ("message", {"type": "reasoning", "summary": "meta"}),
        ("custom_message", {"type": "reasoning", "summary": "meta"}),
        (None, {"type": "reasoning", "summary": "meta"}),
        ("message", {"summary": "meta"}),
        ("custom_message", {"summary": "meta"}),
        (None, {"summary": "meta"}),
    ],
    ids=[
        "message-reasoning",
        "custom-message-reasoning",
        "untyped-reasoning",
        "message-metadata",
        "custom-message-metadata",
        "untyped-metadata",
    ],
)
def test_jsonl_import_metadata_wrappers_do_not_split_responses_group_on_catchup(
    tmp_path: Path,
    wrapper_type: str | None,
    metadata_message: dict[str, object],
):
    importer = load_importer_module()
    session_file = tmp_path / "responses-metadata-catchup.jsonl"
    target_db = tmp_path / "target-lcm.db"
    import_id = f"responses-metadata-catchup-{wrapper_type or 'untyped'}-{len(metadata_message)}"
    initial_rows: list[dict[str, object]] = [
        {"type": "session", "id": "s"},
        {"type": "message", "id": "m1", "message": {"role": "user", "content": "root"}},
        {
            "type": "function_call",
            "id": "fc_1",
            "parentId": "m1",
            "call_id": "call_1",
            "name": "lookup",
            "arguments": {},
        },
        {
            "type": "function_call",
            "id": "fc_2",
            "parentId": "m1",
            "call_id": "call_2",
            "name": "search",
            "arguments": {},
        },
        {"type": "function_call_output", "call_id": "call_1", "output": "r1"},
        {"type": "function_call_output", "call_id": "call_2", "output": "r2"},
    ]
    metadata_row: dict[str, object] = {
        "id": "meta",
        "parentId": "m1",
        "message": {**metadata_message},
    }
    if wrapper_type is not None:
        metadata_row["type"] = wrapper_type

    write_jsonl_session(session_file, initial_rows)
    first = importer.import_jsonl_sessions(
        files=[session_file], target_db=target_db, import_id=import_id, apply=True
    )
    write_jsonl_session(session_file, initial_rows[:3] + [metadata_row] + initial_rows[3:])
    second = importer.import_jsonl_sessions(
        files=[session_file], target_db=target_db, import_id=import_id, apply=True
    )

    assert first.imported == 4
    assert first.invalid_rows == 0
    assert second.scanned == 6
    assert second.eligible == 4
    assert second.imported == 0
    assert second.skipped_existing == 4
    assert second.invalid_rows == 0
    conn = sqlite3.connect(target_db)
    rows = conn.execute(
        "SELECT role, content, tool_call_id, tool_calls, tool_name FROM messages ORDER BY store_id"
    ).fetchall()
    import_keys = conn.execute(
        "SELECT source_message_key FROM lcm_imported_messages ORDER BY target_store_id"
    ).fetchall()
    conn.close()

    assert rows[0] == ("user", "root", None, None, None)
    assert rows[1][0:3] == ("assistant", None, None)
    assert json.loads(rows[1][3]) == [
        {"id": "call_1", "type": "function_call", "function": {"name": "lookup", "arguments": "{}"}},
        {"id": "call_2", "type": "function_call", "function": {"name": "search", "arguments": "{}"}},
    ]
    assert rows[2] == ("tool", "r1", "call_1", None, None)
    assert rows[3] == ("tool", "r2", "call_2", None, None)
    assert import_keys == [
        (jsonl_key("s", "m1"),),
        (jsonl_key("s", 'function_calls:["fc_1","fc_2"]'),),
        (jsonl_key("s", jsonl_result_row_id("function_call_output", "call_1", "r1")),),
        (jsonl_key("s", jsonl_result_row_id("function_call_output", "call_2", "r2")),),
    ]


def test_jsonl_import_metadata_parented_to_previous_function_call_does_not_split_group_on_catchup(
    tmp_path: Path,
):
    importer = load_importer_module()
    session_file = tmp_path / "responses-metadata-previous-call-catchup.jsonl"
    target_db = tmp_path / "target-lcm.db"
    initial_rows: list[dict[str, object]] = [
        {"type": "session", "id": "s"},
        {"type": "message", "id": "root", "message": {"role": "user", "content": "root"}},
        {
            "type": "function_call",
            "id": "fc_1",
            "parentId": "root",
            "call_id": "call_1",
            "name": "lookup",
            "arguments": {},
        },
        {
            "type": "function_call",
            "id": "fc_2",
            "parentId": "root",
            "call_id": "call_2",
            "name": "search",
            "arguments": {},
        },
        {"type": "function_call_output", "call_id": "call_1", "output": "r1"},
        {"type": "function_call_output", "call_id": "call_2", "output": "r2"},
    ]

    write_jsonl_session(session_file, initial_rows)
    first = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id="responses-metadata-previous-call-catchup",
        apply=True,
    )
    write_jsonl_session(
        session_file,
        [
            *initial_rows[:3],
            {"type": "message", "id": "meta", "parentId": "fc_1", "message": {"type": "reasoning", "summary": "meta"}},
            *initial_rows[3:],
        ],
    )
    second = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id="responses-metadata-previous-call-catchup",
        apply=True,
    )

    assert first.imported == 4
    assert first.invalid_rows == 0
    assert second.scanned == 6
    assert second.eligible == 4
    assert second.imported == 0
    assert second.skipped_existing == 4
    assert second.invalid_rows == 0
    conn = sqlite3.connect(target_db)
    rows = conn.execute(
        "SELECT role, content, tool_call_id, tool_calls, tool_name FROM messages ORDER BY store_id"
    ).fetchall()
    import_keys = conn.execute(
        "SELECT source_message_key FROM lcm_imported_messages ORDER BY target_store_id"
    ).fetchall()
    conn.close()

    assert rows[0] == ("user", "root", None, None, None)
    assert rows[1][0:3] == ("assistant", None, None)
    assert json.loads(rows[1][3]) == [
        {"id": "call_1", "type": "function_call", "function": {"name": "lookup", "arguments": "{}"}},
        {"id": "call_2", "type": "function_call", "function": {"name": "search", "arguments": "{}"}},
    ]
    assert rows[2] == ("tool", "r1", "call_1", None, None)
    assert rows[3] == ("tool", "r2", "call_2", None, None)
    assert import_keys == [
        (jsonl_key("s", "root"),),
        (jsonl_key("s", 'function_calls:["fc_1","fc_2"]'),),
        (jsonl_key("s", jsonl_result_row_id("function_call_output", "call_1", "r1")),),
        (jsonl_key("s", jsonl_result_row_id("function_call_output", "call_2", "r2")),),
    ]


def test_jsonl_import_bare_metadata_parented_to_call_id_does_not_split_group_on_catchup(
    tmp_path: Path,
):
    importer = load_importer_module()
    session_file = tmp_path / "responses-bare-metadata-call-id-catchup.jsonl"
    target_db = tmp_path / "target-lcm.db"
    initial_rows: list[dict[str, object]] = [
        {"type": "session", "id": "s"},
        {"type": "message", "id": "root", "message": {"role": "user", "content": "root"}},
        {"parentId": "root", "call_id": "call_1", "name": "lookup", "arguments": {}},
        {"parentId": "root", "call_id": "call_2", "name": "search", "arguments": {}},
        {"type": "function_call_output", "call_id": "call_1", "output": "r1"},
        {"type": "function_call_output", "call_id": "call_2", "output": "r2"},
    ]

    write_jsonl_session(session_file, initial_rows)
    first = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id="responses-bare-metadata-call-id-catchup",
        apply=True,
    )
    write_jsonl_session(
        session_file,
        [
            *initial_rows[:3],
            {"id": "meta", "parentId": "call_1", "message": {"type": "reasoning", "summary": "m"}},
            *initial_rows[3:],
        ],
    )
    second = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id="responses-bare-metadata-call-id-catchup",
        apply=True,
    )

    assert first.imported == 4
    assert first.invalid_rows == 0
    assert second.scanned == 6
    assert second.eligible == 4
    assert second.imported == 0
    assert second.skipped_existing == 4
    assert second.invalid_rows == 0
    conn = sqlite3.connect(target_db)
    rows = conn.execute(
        "SELECT role, content, tool_call_id, tool_calls, tool_name FROM messages ORDER BY store_id"
    ).fetchall()
    import_keys = conn.execute(
        "SELECT source_message_key FROM lcm_imported_messages ORDER BY target_store_id"
    ).fetchall()
    conn.close()

    assert rows[0] == ("user", "root", None, None, None)
    assert rows[1][0:3] == ("assistant", None, None)
    assert json.loads(rows[1][3]) == [
        {"id": "call_1", "type": "function_call", "function": {"name": "lookup", "arguments": "{}"}},
        {"id": "call_2", "type": "function_call", "function": {"name": "search", "arguments": "{}"}},
    ]
    assert rows[2] == ("tool", "r1", "call_1", None, None)
    assert rows[3] == ("tool", "r2", "call_2", None, None)
    assert import_keys == [
        (jsonl_key("s", "root"),),
        (
            jsonl_key(
                "s",
                "function_calls:"
                + json.dumps(
                    [
                        jsonl_tool_call_row_id("function_call", "call_1", "lookup", {}),
                        jsonl_tool_call_row_id("function_call", "call_2", "search", {}),
                    ],
                    separators=(",", ":"),
                ),
            ),
        ),
        (jsonl_key("s", jsonl_result_row_id("function_call_output", "call_1", "r1")),),
        (jsonl_key("s", jsonl_result_row_id("function_call_output", "call_2", "r2")),),
    ]


def test_jsonl_import_keeps_parent_linked_native_function_call_siblings_across_reasoning_row(
    tmp_path: Path,
):
    importer = load_importer_module()
    session_file = tmp_path / "responses-native-parent-parallel-reasoning.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            jsonl_header("responses-native-parent-parallel-reasoning"),
            jsonl_message("m1", "user", "root"),
            {
                "type": "function_call",
                "id": "fc_1",
                "parentId": "m1",
                "call_id": "call_1",
                "name": "lookup",
                "arguments": {"query": "sammy"},
            },
            {
                "type": "function_call",
                "id": "fc_2",
                "parentId": "m1",
                "call_id": "call_2",
                "name": "search",
                "arguments": {"query": "memory"},
            },
            {"type": "function_call_output", "call_id": "call_1", "output": "lookup result"},
            {"type": "function_call_output", "call_id": "call_2", "output": "search result"},
        ],
    )

    first = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id="responses-native-parent-parallel-reasoning",
        apply=True,
    )
    write_jsonl_session(
        session_file,
        [
            jsonl_header("responses-native-parent-parallel-reasoning"),
            jsonl_message("m1", "user", "root"),
            {
                "type": "function_call",
                "id": "fc_1",
                "parentId": "m1",
                "call_id": "call_1",
                "name": "lookup",
                "arguments": {"query": "sammy"},
            },
            {"id": "rs_1", "parentId": "m1", "message": {"type": "reasoning", "summary": "interleaved"}},
            {
                "type": "function_call",
                "id": "fc_2",
                "parentId": "m1",
                "call_id": "call_2",
                "name": "search",
                "arguments": {"query": "memory"},
            },
            {"type": "function_call_output", "call_id": "call_1", "output": "lookup result"},
            {"type": "function_call_output", "call_id": "call_2", "output": "search result"},
        ],
    )
    second = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id="responses-native-parent-parallel-reasoning",
        apply=True,
    )

    assert first.scanned == 5
    assert first.eligible == 4
    assert first.imported == 4
    assert first.invalid_rows == 0
    assert second.scanned == 6
    assert second.eligible == 4
    assert second.imported == 0
    assert second.skipped_existing == 4
    assert second.invalid_rows == 0
    conn = sqlite3.connect(target_db)
    rows = conn.execute(
        "SELECT role, content, tool_call_id, tool_calls, tool_name FROM messages ORDER BY store_id"
    ).fetchall()
    import_keys = conn.execute(
        "SELECT source_message_key FROM lcm_imported_messages ORDER BY target_store_id"
    ).fetchall()
    conn.close()

    assert rows[0] == ("user", "root", None, None, None)
    assert rows[1][0:3] == ("assistant", None, None)
    assert json.loads(rows[1][3]) == [
        {
            "id": "call_1",
            "type": "function_call",
            "function": {"name": "lookup", "arguments": '{"query":"sammy"}'},
        },
        {
            "id": "call_2",
            "type": "function_call",
            "function": {"name": "search", "arguments": '{"query":"memory"}'},
        },
    ]
    assert rows[2] == ("tool", "lookup result", "call_1", None, None)
    assert rows[3] == ("tool", "search result", "call_2", None, None)
    assert import_keys == [
        (jsonl_key("responses-native-parent-parallel-reasoning", "m1"),),
        (
            jsonl_key(
                "responses-native-parent-parallel-reasoning",
                'function_calls:["fc_1","fc_2"]',
            ),
        ),
        (
            jsonl_key(
                "responses-native-parent-parallel-reasoning",
                jsonl_result_row_id("function_call_output", "call_1", "lookup result"),
            ),
        ),
        (
            jsonl_key(
                "responses-native-parent-parallel-reasoning",
                jsonl_result_row_id("function_call_output", "call_2", "search result"),
            ),
        ),
    ]


@pytest.mark.parametrize(
    ("row_type", "call_fields", "result_fields"),
    [
        (
            "toolCall",
            {"toolCallId": "call_1", "toolName": "lookup", "toolInput": {"query": "sammy"}},
            {"toolCallId": "call_1", "toolName": "lookup"},
        ),
        (
            "tool_call",
            {"tool_call_id": "call_1", "tool_name": "lookup", "tool_input": {"query": "sammy"}},
            {"tool_call_id": "call_1", "tool_name": "lookup"},
        ),
        (
            "toolUse",
            {"toolUseId": "call_1", "toolName": "lookup", "toolInput": {"query": "sammy"}},
            {"toolUseId": "call_1", "toolName": "lookup"},
        ),
        (
            "tool_use",
            {"tool_use_id": "call_1", "tool_name": "lookup", "tool_input": {"query": "sammy"}},
            {"tool_use_id": "call_1", "tool_name": "lookup"},
        ),
    ],
)
def test_jsonl_import_maps_top_level_openclaw_tool_call_rows(
    tmp_path: Path,
    row_type: str,
    call_fields: dict[str, object],
    result_fields: dict[str, object],
):
    importer = load_importer_module()
    session_file = tmp_path / f"top-level-{row_type}.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            jsonl_header("top-level-openclaw-call"),
            {"type": row_type, "id": "tc1", **call_fields},
            {"type": "toolResult", "id": "tr1", "content": "result", **result_fields},
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file], target_db=target_db, import_id=f"top-level-{row_type}", apply=True
    )

    assert result.scanned == 2
    assert result.eligible == 2
    assert result.imported == 2
    assert result.invalid_rows == 0
    assert result.skipped_empty == 0
    conn = sqlite3.connect(target_db)
    rows = conn.execute(
        "SELECT role, content, tool_call_id, tool_calls, tool_name FROM messages ORDER BY store_id"
    ).fetchall()
    conn.close()
    assert rows[0][0:3] == ("assistant", None, None)
    assert json.loads(rows[0][3]) == [
        {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": '{"query":"sammy"}'}}
    ]
    assert rows[0][4] is None
    assert rows[1] == ("tool", "result", "call_1", None, "lookup")


def test_jsonl_import_maps_top_level_tool_use_name_on_result_rows(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "top-level-tool-use-name.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            jsonl_header("top-level-tool-use-name"),
            {
                "type": "toolUse",
                "id": "tu1",
                "toolUseId": "call_1",
                "toolUseName": "lookup",
                "toolUseInput": {"query": "sammy"},
            },
            {"type": "toolResult", "id": "tr1", "toolUseId": "call_1", "toolUseName": "lookup", "content": "result"},
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file], target_db=target_db, import_id="top-level-tool-use-name", apply=True
    )

    assert result.scanned == 2
    assert result.eligible == 2
    assert result.imported == 2
    assert result.invalid_rows == 0
    conn = sqlite3.connect(target_db)
    rows = conn.execute(
        "SELECT role, content, tool_call_id, tool_calls, tool_name FROM messages ORDER BY store_id"
    ).fetchall()
    conn.close()

    assert rows[0][0:3] == ("assistant", None, None)
    assert json.loads(rows[0][3]) == [
        {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": '{"query":"sammy"}'}}
    ]
    assert rows[0][4] is None
    assert rows[1] == ("tool", "result", "call_1", None, "lookup")


def test_jsonl_import_accepts_openclaw_tool_use_with_empty_string_input(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "openclaw-tool-use-empty-string-input.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            {"type": "session", "id": "s"},
            {
                "type": "toolUse",
                "id": "tu1",
                "toolUseId": "call_1",
                "toolUseName": "lookup",
                "toolUseInput": "",
            },
            {"type": "toolResult", "toolUseId": "call_1", "toolUseName": "lookup", "content": "result"},
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id="openclaw-tool-use-empty-string-input",
        apply=True,
    )

    assert result.scanned == 2
    assert result.eligible == 2
    assert result.imported == 2
    assert result.invalid_rows == 0
    assert result.skipped_empty == 0
    conn = sqlite3.connect(target_db)
    rows = conn.execute(
        "SELECT role, content, tool_call_id, tool_calls, tool_name FROM messages ORDER BY store_id"
    ).fetchall()
    conn.close()

    assert rows[0][0:3] == ("assistant", None, None)
    assert json.loads(rows[0][3]) == [
        {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": ""}}
    ]
    assert rows[0][4] is None
    assert rows[1] == ("tool", "result", "call_1", None, "lookup")


def test_jsonl_import_rejects_openclaw_tool_use_without_input(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "openclaw-tool-use-missing-input.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            {"type": "session", "id": "s"},
            {"type": "toolUse", "id": "tu1", "toolUseId": "call_1", "toolUseName": "lookup"},
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id="openclaw-tool-use-missing-input",
        apply=True,
    )

    assert result.scanned == 1
    assert result.eligible == 0
    assert result.imported == 0
    assert result.invalid_rows == 1
    assert result.skipped_empty == 0
    assert any(
        "toolUse row missing tool call id or name, or input/arguments" in warning
        for warning in result.warnings
    )
    assert not target_db.exists()


def test_jsonl_import_maps_bare_untyped_openclaw_tool_use_and_result_rows(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "bare-openclaw-tool-use-result.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            jsonl_header("bare-openclaw-tool-use-result"),
            {"toolUseId": "call_1", "toolUseName": "lookup", "toolUseInput": {}},
            {"toolUseId": "call_1", "toolUseName": "lookup", "content": "result"},
        ],
    )

    first = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id="bare-openclaw-tool-use-result",
        apply=True,
    )
    second = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id="bare-openclaw-tool-use-result",
        apply=True,
    )

    assert first.scanned == 2
    assert first.eligible == 2
    assert first.imported == 2
    assert first.invalid_rows == 0
    assert second.imported == 0
    assert second.skipped_existing == 2
    conn = sqlite3.connect(target_db)
    rows = conn.execute(
        "SELECT role, content, tool_call_id, tool_calls, tool_name FROM messages ORDER BY store_id"
    ).fetchall()
    conn.close()

    assert rows[0][0:3] == ("assistant", None, None)
    assert json.loads(rows[0][3]) == [
        {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
    ]
    assert rows[1] == ("tool", "result", "call_1", None, "lookup")


def test_jsonl_import_maps_bare_openclaw_tool_use_with_generic_name_and_input_aliases(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "bare-openclaw-generic-alias-tool-use-result.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            jsonl_header("bare-openclaw-generic-alias-tool-use-result"),
            {"toolUseId": "call_1", "name": "lookup", "input": {"q": "x"}},
            {"toolUseId": "call_1", "content": "result"},
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id="bare-openclaw-generic-alias-tool-use-result",
        apply=True,
    )

    assert result.scanned == 2
    assert result.eligible == 2
    assert result.imported == 2
    assert result.invalid_rows == 0
    conn = sqlite3.connect(target_db)
    rows = conn.execute(
        "SELECT role, content, tool_call_id, tool_calls, tool_name FROM messages ORDER BY store_id"
    ).fetchall()
    conn.close()

    assert rows[0][0:3] == ("assistant", None, None)
    assert json.loads(rows[0][3]) == [
        {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": '{"q":"x"}'}}
    ]
    assert rows[1] == ("tool", "result", "call_1", None, None)


def test_jsonl_import_maps_bare_content_openclaw_tool_result_without_name(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "bare-openclaw-content-result.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            jsonl_header("s"),
            {"type": "toolUse", "id": "tu", "toolUseId": "call_1", "toolUseName": "lookup", "toolUseInput": {}},
            {"toolUseId": "call_1", "content": "result"},
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id="bare-openclaw-content-result",
        apply=True,
    )

    assert result.scanned == 2
    assert result.eligible == 2
    assert result.imported == 2
    assert result.invalid_rows == 0
    conn = sqlite3.connect(target_db)
    rows = conn.execute(
        "SELECT role, content, tool_call_id, tool_calls, tool_name FROM messages ORDER BY store_id"
    ).fetchall()
    import_keys = conn.execute(
        "SELECT source_message_key FROM lcm_imported_messages ORDER BY target_store_id"
    ).fetchall()
    conn.close()

    assert rows[0][0:3] == ("assistant", None, None)
    assert json.loads(rows[0][3]) == [
        {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
    ]
    assert rows[1] == ("tool", "result", "call_1", None, None)
    assert import_keys == [
        (jsonl_key("s", "tu"),),
        (jsonl_key("s", jsonl_result_row_id("tool_result", "call_1", "result")),),
    ]
    assert all("line:" not in key for (key,) in import_keys)


def test_jsonl_import_maps_wrapped_openclaw_tool_use_and_result_rows(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "wrapped-openclaw-tool-use-result.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            jsonl_header("wrapped-openclaw-tool-use-result"),
            {
                "type": "message",
                "id": "w-call",
                "message": {
                    "id": "nested-call-row",
                    "type": "toolUse",
                    "toolUseId": "call_1",
                    "toolUseName": "lookup",
                    "toolUseInput": {"q": "x"},
                },
            },
            {
                "type": "message",
                "id": "w-result",
                "message": {
                    "id": "nested-result-row",
                    "type": "toolResult",
                    "toolUseId": "call_1",
                    "toolUseName": "lookup",
                    "content": "result",
                },
            },
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id="wrapped-openclaw-tool-use-result",
        apply=True,
    )

    assert result.scanned == 2
    assert result.eligible == 2
    assert result.imported == 2
    assert result.invalid_rows == 0
    conn = sqlite3.connect(target_db)
    rows = conn.execute(
        "SELECT role, content, tool_call_id, tool_calls, tool_name FROM messages ORDER BY store_id"
    ).fetchall()
    import_keys = conn.execute(
        "SELECT source_message_key FROM lcm_imported_messages ORDER BY target_store_id"
    ).fetchall()
    conn.close()

    assert rows[0][0:3] == ("assistant", None, None)
    assert json.loads(rows[0][3]) == [
        {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": '{"q":"x"}'}}
    ]
    assert rows[0][4] is None
    assert rows[1] == ("tool", "result", "call_1", None, "lookup")
    assert import_keys == [
        (jsonl_key("wrapped-openclaw-tool-use-result", "nested-call-row"),),
        (jsonl_key("wrapped-openclaw-tool-use-result", "nested-result-row"),),
    ]


def test_jsonl_import_skips_wrapped_openclaw_tool_result_for_malformed_wrapped_tool_use(
    tmp_path: Path,
):
    importer = load_importer_module()
    session_file = tmp_path / "wrapped-openclaw-tool-result-orphan.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            jsonl_header("wrapped-openclaw-tool-result-orphan"),
            {
                "type": "message",
                "id": "bad-call",
                "message": {
                    "type": "toolUse",
                    "toolUseId": "call_bad",
                    "toolUseInput": {"q": "x"},
                },
            },
            {
                "type": "message",
                "id": "orphan-result",
                "message": {
                    "type": "toolResult",
                    "toolUseId": "call_bad",
                    "toolUseName": "lookup",
                    "content": "orphan",
                },
            },
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id="wrapped-openclaw-tool-result-orphan",
        apply=True,
    )

    assert result.scanned == 2
    assert result.eligible == 0
    assert result.imported == 0
    assert result.invalid_rows == 1
    assert any("toolUse row missing tool call id or name" in warning for warning in result.warnings)
    assert not target_db.exists()


def test_jsonl_import_rejects_wrapped_openclaw_tool_use_without_call_id_and_skips_result(
    tmp_path: Path,
):
    importer = load_importer_module()
    session_file = tmp_path / "wrapped-openclaw-tool-use-missing-call-id.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            {"type": "session", "id": "s"},
            {
                "type": "message",
                "id": "call-envelope",
                "message": {
                    "id": "message-row",
                    "type": "toolUse",
                    "toolUseName": "lookup",
                    "toolUseInput": {},
                },
            },
            {
                "type": "message",
                "id": "result-envelope",
                "message": {
                    "type": "toolResult",
                    "toolUseId": "call_1",
                    "toolUseName": "lookup",
                    "content": "result",
                },
            },
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id="wrapped-openclaw-tool-use-missing-call-id",
        apply=True,
    )

    assert result.scanned == 2
    assert result.eligible == 0
    assert result.imported == 0
    assert result.invalid_rows == 1
    assert any("toolUse row missing tool call id or name" in warning for warning in result.warnings)
    assert not target_db.exists()


def test_jsonl_import_maps_untyped_wrapped_openclaw_tool_use_and_result_rows(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "untyped-wrapped-openclaw-tool-use-result.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            {"type": "session", "id": "untyped-wrapped-tool"},
            {
                "id": "env-call",
                "message": {
                    "type": "toolUse",
                    "toolUseId": "call_1",
                    "toolUseName": "lookup",
                    "toolUseInput": {"q": "x"},
                },
            },
            {
                "id": "env-result",
                "message": {
                    "type": "toolResult",
                    "toolUseId": "call_1",
                    "toolUseName": "lookup",
                    "content": "result",
                },
            },
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id="untyped-wrapped-openclaw-tool-use-result",
        apply=True,
    )

    assert result.scanned == 2
    assert result.eligible == 2
    assert result.imported == 2
    assert result.invalid_rows == 0
    conn = sqlite3.connect(target_db)
    rows = conn.execute(
        "SELECT role, content, tool_call_id, tool_calls, tool_name FROM messages ORDER BY store_id"
    ).fetchall()
    import_keys = conn.execute(
        "SELECT source_message_key FROM lcm_imported_messages ORDER BY target_store_id"
    ).fetchall()
    conn.close()

    assert rows[0][0:3] == ("assistant", None, None)
    assert json.loads(rows[0][3]) == [
        {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": '{"q":"x"}'}}
    ]
    assert rows[0][4] is None
    assert rows[1] == ("tool", "result", "call_1", None, "lookup")
    assert import_keys == [
        (jsonl_key("untyped-wrapped-tool", "env-call"),),
        (jsonl_key("untyped-wrapped-tool", "env-result"),),
    ]


def test_jsonl_import_skips_untyped_wrapped_openclaw_tool_result_for_malformed_tool_use(
    tmp_path: Path,
):
    importer = load_importer_module()
    session_file = tmp_path / "untyped-wrapped-openclaw-tool-result-orphan.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            {"type": "session", "id": "untyped-wrapped-openclaw-tool-result-orphan"},
            {
                "id": "bad-call",
                "message": {
                    "type": "toolUse",
                    "toolUseId": "call_bad",
                    "toolUseInput": {"q": "x"},
                },
            },
            {
                "id": "orphan-result",
                "message": {
                    "type": "toolResult",
                    "toolUseId": "call_bad",
                    "toolUseName": "lookup",
                    "content": "orphan",
                },
            },
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id="untyped-wrapped-openclaw-tool-result-orphan",
        apply=True,
    )

    assert result.scanned == 2
    assert result.eligible == 0
    assert result.imported == 0
    assert result.invalid_rows == 1
    assert any("toolUse row missing tool call id or name" in warning for warning in result.warnings)
    assert not target_db.exists()


def test_jsonl_import_idless_openclaw_tool_result_key_survives_line_shift(
    tmp_path: Path,
):
    importer = load_importer_module()
    session_file = tmp_path / "top-level-tool-result-line-shift.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            jsonl_header("top-level-tool-result-line-shift"),
            {
                "type": "toolUse",
                "id": "tu1",
                "toolUseId": "call_1",
                "toolUseName": "lookup",
                "toolUseInput": {"query": "sammy"},
            },
            {"type": "toolResult", "toolUseId": "call_1", "toolUseName": "lookup", "content": "result"},
        ],
    )

    first = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id="top-level-tool-result-line-shift",
        apply=True,
    )
    write_jsonl_session(
        session_file,
        [
            jsonl_header("top-level-tool-result-line-shift"),
            {
                "type": "toolUse",
                "id": "tu1",
                "toolUseId": "call_1",
                "toolUseName": "lookup",
                "toolUseInput": {"query": "sammy"},
            },
            {"type": "reasoning", "id": "rs_1", "summary": "inserted metadata"},
            {"type": "toolResult", "toolUseId": "call_1", "toolUseName": "lookup", "content": "result"},
        ],
    )
    second = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id="top-level-tool-result-line-shift",
        apply=True,
    )

    assert first.imported == 2
    assert second.scanned == 3
    assert second.eligible == 2
    assert second.imported == 0
    assert second.skipped_existing == 2
    conn = sqlite3.connect(target_db)
    rows = conn.execute(
        "SELECT role, content, tool_call_id, tool_calls, tool_name FROM messages ORDER BY store_id"
    ).fetchall()
    import_keys = conn.execute(
        "SELECT source_message_key FROM lcm_imported_messages ORDER BY target_store_id"
    ).fetchall()
    conn.close()

    assert rows[0][0:3] == ("assistant", None, None)
    assert rows[1] == ("tool", "result", "call_1", None, "lookup")
    assert import_keys == [
        (jsonl_key("top-level-tool-result-line-shift", "tu1"),),
        (
            jsonl_key(
                "top-level-tool-result-line-shift",
                jsonl_result_row_id("tool_result", "call_1", "result"),
            ),
        ),
    ]
    assert all("line:" not in key for (key,) in import_keys)


def test_jsonl_import_idless_openclaw_tool_use_key_survives_line_shift(
    tmp_path: Path,
):
    importer = load_importer_module()
    session_file = tmp_path / "openclaw-idless-tool-use-line-shift.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            jsonl_header("openclaw-idless-tool-use-line-shift"),
            {"type": "toolUse", "toolUseId": "call_1", "toolUseName": "lookup", "toolUseInput": {}},
            {"type": "toolResult", "toolUseId": "call_1", "toolUseName": "lookup", "content": "result"},
        ],
    )

    first = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id="openclaw-idless-tool-use-line-shift",
        apply=True,
    )
    write_jsonl_session(
        session_file,
        [
            jsonl_header("openclaw-idless-tool-use-line-shift"),
            {"type": "reasoning", "id": "rs_1", "summary": "inserted metadata"},
            {"type": "toolUse", "toolUseId": "call_1", "toolUseName": "lookup", "toolUseInput": {}},
            {"type": "toolResult", "toolUseId": "call_1", "toolUseName": "lookup", "content": "result"},
        ],
    )
    second = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id="openclaw-idless-tool-use-line-shift",
        apply=True,
    )

    assert first.imported == 2
    assert second.scanned == 3
    assert second.eligible == 2
    assert second.imported == 0
    assert second.skipped_existing == 2
    conn = sqlite3.connect(target_db)
    rows = conn.execute(
        "SELECT role, content, tool_call_id, tool_calls, tool_name FROM messages ORDER BY store_id"
    ).fetchall()
    import_keys = conn.execute(
        "SELECT source_message_key FROM lcm_imported_messages ORDER BY target_store_id"
    ).fetchall()
    conn.close()

    assert rows[0][0:3] == ("assistant", None, None)
    assert json.loads(rows[0][3]) == [
        {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
    ]
    assert rows[1] == ("tool", "result", "call_1", None, "lookup")
    assert import_keys == [
        (
            jsonl_key(
                "openclaw-idless-tool-use-line-shift",
                jsonl_tool_call_row_id("toolUse", "call_1", "lookup", {}),
            ),
        ),
        (
            jsonl_key(
                "openclaw-idless-tool-use-line-shift",
                jsonl_result_row_id("tool_result", "call_1", "result"),
            ),
        ),
    ]
    assert all("line:" not in key for (key,) in import_keys)


@pytest.mark.parametrize(
    ("result_type", "result_fields"),
    [
        ("toolResult", {"toolUseId": "call_1", "toolUseName": "lookup"}),
        ("tool_result", {"tool_use_id": "call_1", "tool_use_name": "lookup"}),
    ],
)
def test_jsonl_import_imports_openclaw_tool_result_from_later_catchup(
    tmp_path: Path,
    result_type: str,
    result_fields: dict[str, object],
):
    importer = load_importer_module()
    test_id = result_type.replace("_", "-")
    first_file = tmp_path / f"openclaw-tool-catchup-{test_id}-1.jsonl"
    second_file = tmp_path / f"openclaw-tool-catchup-{test_id}-2.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        first_file,
        [
            jsonl_header(f"openclaw-tool-catchup-{test_id}"),
            {
                "type": "toolUse",
                "id": "tu1",
                "toolUseId": "call_1",
                "toolUseName": "lookup",
                "toolUseInput": {"query": "sammy"},
            },
        ],
    )

    first = importer.import_jsonl_sessions(
        files=[first_file],
        target_db=target_db,
        import_id=f"openclaw-tool-catchup-{test_id}",
        apply=True,
    )
    write_jsonl_session(
        second_file,
        [
            {"type": result_type, "content": "result", **result_fields},
        ],
    )
    second = importer.import_jsonl_sessions(
        files=[second_file],
        target_db=target_db,
        import_id=f"openclaw-tool-catchup-{test_id}",
        apply=True,
    )

    assert first.imported == 1
    assert second.scanned == 1
    assert second.eligible == 1
    assert second.imported == 1
    assert second.invalid_rows == 0
    conn = sqlite3.connect(target_db)
    rows = conn.execute(
        "SELECT role, content, tool_call_id, tool_calls, tool_name FROM messages ORDER BY store_id"
    ).fetchall()
    import_rows = conn.execute(
        """SELECT source_session, source_message_key
           FROM lcm_imported_messages
           ORDER BY target_store_id"""
    ).fetchall()
    conn.close()

    assert rows[0][0:3] == ("assistant", None, None)
    assert json.loads(rows[0][3]) == [
        {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": '{"query":"sammy"}'}}
    ]
    assert rows[1] == ("tool", "result", "call_1", None, "lookup")
    assert import_rows == [
        (
            f"openclaw-tool-catchup-{test_id}",
            jsonl_key(f"openclaw-tool-catchup-{test_id}", "tu1"),
        ),
        (
            f"openclaw-tool-catchup-{test_id}",
            jsonl_key(
                f"openclaw-tool-catchup-{test_id}",
                jsonl_result_row_id("tool_result", "call_1", "result"),
            ),
        ),
    ]


def test_jsonl_import_resolves_openclaw_tool_result_before_call_file_in_same_run(
    tmp_path: Path,
):
    importer = load_importer_module()
    output_file = tmp_path / "a-output.jsonl"
    call_file = tmp_path / "b-call.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        output_file,
        [
            {
                "type": "toolResult",
                "toolUseId": "call_1",
                "toolUseName": "lookup",
                "content": "result",
            },
        ],
    )
    write_jsonl_session(
        call_file,
        [
            jsonl_header("openclaw-catchup-order"),
            {
                "type": "toolUse",
                "id": "tu1",
                "toolUseId": "call_1",
                "toolUseName": "lookup",
                "toolUseInput": {},
            },
        ],
    )

    first = importer.import_jsonl_sessions(
        files=[output_file, call_file],
        target_db=target_db,
        import_id="openclaw-catchup-order",
        apply=True,
    )
    second = importer.import_jsonl_sessions(
        files=[output_file, call_file],
        target_db=target_db,
        import_id="openclaw-catchup-order",
        apply=True,
    )

    assert first.scanned == 2
    assert first.eligible == 2
    assert first.imported == 2
    assert first.invalid_rows == 0
    assert second.imported == 0
    assert second.skipped_existing == 2
    conn = sqlite3.connect(target_db)
    rows = conn.execute(
        "SELECT role, content, tool_call_id, tool_calls, tool_name FROM messages ORDER BY store_id"
    ).fetchall()
    import_rows = conn.execute(
        """SELECT source_session, source_message_key
           FROM lcm_imported_messages
           ORDER BY source_message_key"""
    ).fetchall()
    conn.close()

    assert rows[0][0:3] == ("assistant", None, None)
    assert json.loads(rows[0][3]) == [
        {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
    ]
    assert rows[1] == ("tool", "result", "call_1", None, "lookup")
    assert import_rows == [
        (
            "openclaw-catchup-order",
            jsonl_key(
                "openclaw-catchup-order",
                jsonl_result_row_id("tool_result", "call_1", "result"),
            ),
        ),
        ("openclaw-catchup-order", jsonl_key("openclaw-catchup-order", "tu1")),
    ]


def test_jsonl_import_skipped_changed_openclaw_tool_use_does_not_authorize_result(
    tmp_path: Path,
):
    importer = load_importer_module()
    session_file = tmp_path / "changed-openclaw-tool-use.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            {"type": "session", "id": "s"},
            {
                "type": "toolUse",
                "id": "tu",
                "toolUseId": "call_1",
                "toolUseName": "lookup",
                "toolUseInput": {},
            },
        ],
    )

    first = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id="changed-openclaw-tool-use",
        apply=True,
    )
    write_jsonl_session(
        session_file,
        [
            {"type": "session", "id": "s"},
            {
                "type": "toolUse",
                "id": "tu",
                "toolUseId": "call_bad",
                "toolUseName": "bad",
                "toolUseInput": {},
            },
            {
                "type": "toolResult",
                "toolUseId": "call_bad",
                "toolUseName": "bad",
                "content": "orphan",
            },
        ],
    )
    second = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id="changed-openclaw-tool-use",
        apply=True,
    )

    assert first.imported == 1
    assert second.scanned == 2
    assert second.eligible == 1
    assert second.imported == 0
    assert second.skipped_existing == 1
    assert second.invalid_rows == 0
    conn = sqlite3.connect(target_db)
    rows = conn.execute(
        "SELECT role, content, tool_call_id, tool_calls, tool_name FROM messages ORDER BY store_id"
    ).fetchall()
    keys = conn.execute("SELECT source_message_key FROM lcm_imported_messages ORDER BY target_store_id").fetchall()
    conn.close()

    assert rows[0][0:3] == ("assistant", None, None)
    assert json.loads(rows[0][3]) == [
        {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
    ]
    assert rows[0][4] is None
    assert keys == [(jsonl_key("s", "tu"),)]


def test_jsonl_import_skips_openclaw_tool_result_for_malformed_tool_use(
    tmp_path: Path,
):
    importer = load_importer_module()
    session_file = tmp_path / "openclaw-tool-result-orphan.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            jsonl_header("openclaw-tool-result-orphan"),
            {"type": "toolUse", "id": "bad", "toolUseId": "call_bad", "toolUseInput": {}},
            {"type": "toolResult", "toolUseId": "call_bad", "toolUseName": "lookup", "content": "orphan"},
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id="openclaw-tool-result-orphan",
        apply=True,
    )

    assert result.scanned == 2
    assert result.eligible == 0
    assert result.imported == 0
    assert result.invalid_rows == 1
    assert any("toolUse row missing tool call id or name" in warning for warning in result.warnings)
    assert not target_db.exists()


def test_jsonl_import_rejects_top_level_openclaw_tool_use_without_call_id_and_skips_result(
    tmp_path: Path,
):
    importer = load_importer_module()
    session_file = tmp_path / "openclaw-tool-use-missing-call-id.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            {"type": "session", "id": "s"},
            {"type": "toolUse", "id": "row1", "toolUseName": "lookup", "toolUseInput": {}},
            {"type": "toolResult", "toolUseId": "call_1", "toolUseName": "lookup", "content": "result"},
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id="openclaw-tool-use-missing-call-id",
        apply=True,
    )

    assert result.scanned == 2
    assert result.eligible == 0
    assert result.imported == 0
    assert result.invalid_rows == 1
    assert any("toolUse row missing tool call id or name" in warning for warning in result.warnings)
    assert not target_db.exists()


def test_jsonl_import_idless_wrapped_tool_result_key_survives_line_shift(
    tmp_path: Path,
):
    importer = load_importer_module()
    session_file = tmp_path / "wrapped-tool-result-line-shift.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            jsonl_header("wrapped-tool-result-line-shift"),
            {
                "type": "toolUse",
                "id": "tu1",
                "toolUseId": "call_1",
                "toolUseName": "lookup",
                "toolUseInput": {"query": "sammy"},
            },
            {
                "type": "message",
                "message": {"role": "toolResult", "toolCallId": "call_1", "content": "result"},
            },
        ],
    )

    first = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id="wrapped-tool-result-line-shift",
        apply=True,
    )
    write_jsonl_session(
        session_file,
        [
            jsonl_header("wrapped-tool-result-line-shift"),
            {
                "type": "toolUse",
                "id": "tu1",
                "toolUseId": "call_1",
                "toolUseName": "lookup",
                "toolUseInput": {"query": "sammy"},
            },
            {"type": "reasoning", "id": "rs_1", "summary": "inserted metadata"},
            {
                "type": "message",
                "message": {"role": "toolResult", "toolCallId": "call_1", "content": "result"},
            },
        ],
    )
    second = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id="wrapped-tool-result-line-shift",
        apply=True,
    )

    assert first.imported == 2
    assert second.scanned == 3
    assert second.eligible == 2
    assert second.imported == 0
    assert second.skipped_existing == 2
    conn = sqlite3.connect(target_db)
    rows = conn.execute(
        "SELECT role, content, tool_call_id, tool_calls, tool_name FROM messages ORDER BY store_id"
    ).fetchall()
    import_keys = conn.execute(
        "SELECT source_message_key FROM lcm_imported_messages ORDER BY target_store_id"
    ).fetchall()
    conn.close()

    assert rows[0][0:3] == ("assistant", None, None)
    assert rows[1] == ("tool", "result", "call_1", None, None)
    assert import_keys == [
        (jsonl_key("wrapped-tool-result-line-shift", "tu1"),),
        (
            jsonl_key(
                "wrapped-tool-result-line-shift",
                jsonl_result_row_id("tool_result", "call_1", "result"),
            ),
        ),
    ]
    assert all("line:" not in key for (key,) in import_keys)


def test_jsonl_import_keeps_parent_linked_top_level_tool_use_siblings(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "top-level-tool-use-parent-siblings.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            jsonl_header("top-level-tool-use-parent-siblings"),
            jsonl_message("m1", "user", "root"),
            {
                "type": "toolUse",
                "id": "tu1",
                "parentId": "m1",
                "toolUseId": "call_1",
                "toolName": "lookup",
                "toolInput": {"query": "sammy"},
            },
            {
                "type": "toolUse",
                "id": "tu2",
                "parentId": "m1",
                "toolUseId": "call_2",
                "toolName": "search",
                "toolInput": {"query": "memory"},
            },
            {"type": "toolResult", "toolUseId": "call_1", "toolName": "lookup", "content": "lookup result"},
            {"type": "toolResult", "toolUseId": "call_2", "toolName": "search", "content": "search result"},
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file], target_db=target_db, import_id="top-level-tool-use-parent-siblings", apply=True
    )

    assert result.scanned == 5
    assert result.eligible == 5
    assert result.imported == 5
    assert result.invalid_rows == 0
    conn = sqlite3.connect(target_db)
    rows = conn.execute(
        "SELECT role, content, tool_call_id, tool_calls, tool_name FROM messages ORDER BY store_id"
    ).fetchall()
    conn.close()

    assert rows[0] == ("user", "root", None, None, None)
    assert rows[1][0:3] == ("assistant", None, None)
    assert json.loads(rows[1][3]) == [
        {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": '{"query":"sammy"}'}}
    ]
    assert rows[2][0:3] == ("assistant", None, None)
    assert json.loads(rows[2][3]) == [
        {"id": "call_2", "type": "function", "function": {"name": "search", "arguments": '{"query":"memory"}'}}
    ]
    assert rows[3] == ("tool", "lookup result", "call_1", None, "lookup")
    assert rows[4] == ("tool", "search result", "call_2", None, "search")


def test_jsonl_import_follows_active_leaf_path_for_branched_exports(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "branched.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            jsonl_header("branched"),
            jsonl_message("root", "user", "root message"),
            {"type": "message", "id": "abandoned", "parentId": "root", "message": {"role": "assistant", "content": "old branch"}},
            {"type": "message", "id": "leaf", "parentId": "root", "message": {"role": "assistant", "content": "current branch"}},
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file], target_db=target_db, import_id="leaf-path", apply=True
    )

    assert result.scanned == 3
    assert result.eligible == 2
    assert result.imported == 2
    conn = sqlite3.connect(target_db)
    rows = conn.execute("SELECT content FROM messages ORDER BY store_id").fetchall()
    import_keys = conn.execute(
        "SELECT source_message_key FROM lcm_imported_messages ORDER BY target_store_id"
    ).fetchall()
    conn.close()
    assert rows == [("root message",), ("current branch",)]
    assert import_keys == [(jsonl_key("branched", "root"),), (jsonl_key("branched", "leaf"),)]


def test_jsonl_import_prunes_untyped_envelope_branch_with_nested_parent_ids(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "untyped-envelope-nested-parents.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            {"type": "session", "id": "s"},
            {"id": "env-root", "message": {"id": "root", "role": "user", "content": "root"}},
            {
                "id": "env-old",
                "message": {"id": "old", "parentId": "root", "role": "assistant", "content": "old"},
            },
            {
                "id": "env-leaf",
                "message": {"id": "leaf", "parentId": "root", "role": "assistant", "content": "current"},
            },
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file], target_db=target_db, import_id="untyped-envelope-nested-parents", apply=True
    )

    assert result.scanned == 3
    assert result.eligible == 2
    assert result.imported == 2
    assert result.invalid_rows == 0
    conn = sqlite3.connect(target_db)
    rows = conn.execute("SELECT content FROM messages ORDER BY store_id").fetchall()
    import_keys = conn.execute(
        "SELECT source_message_key FROM lcm_imported_messages ORDER BY target_store_id"
    ).fetchall()
    conn.close()
    assert rows == [("root",), ("current",)]
    assert import_keys == [(jsonl_key("s", "env-root"),), (jsonl_key("s", "env-leaf"),)]


def test_jsonl_import_prunes_untyped_envelope_branch_with_mixed_parent_aliases(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "untyped-envelope-mixed-parents.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            {"type": "session", "id": "s"},
            {"id": "env-root", "message": {"id": "root", "role": "user", "content": "root"}},
            {
                "id": "env-old",
                "message": {"id": "old", "parentId": "root", "role": "assistant", "content": "old"},
            },
            {
                "id": "env-leaf",
                "parentId": "env-root",
                "message": {"id": "leaf", "role": "assistant", "content": "current"},
            },
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file], target_db=target_db, import_id="untyped-envelope-mixed-parents", apply=True
    )

    assert result.scanned == 3
    assert result.eligible == 2
    assert result.imported == 2
    assert result.invalid_rows == 0
    conn = sqlite3.connect(target_db)
    rows = conn.execute("SELECT content FROM messages ORDER BY store_id").fetchall()
    import_keys = conn.execute(
        "SELECT source_message_key FROM lcm_imported_messages ORDER BY target_store_id"
    ).fetchall()
    conn.close()
    assert rows == [("root",), ("current",)]
    assert import_keys == [(jsonl_key("s", "env-root"),), (jsonl_key("s", "env-leaf"),)]


def test_jsonl_import_metadata_id_does_not_overwrite_importable_leaf_parent(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "metadata-id-collision.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            {"type": "session", "id": "s"},
            {"type": "message", "id": "root", "message": {"role": "user", "content": "root"}},
            {
                "type": "message",
                "id": "leaf",
                "parentId": "root",
                "message": {"role": "assistant", "content": "leaf"},
            },
            {"type": "reasoning", "id": "root", "summary": "metadata"},
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file], target_db=target_db, import_id="metadata-id-collision", apply=True
    )

    assert result.scanned == 3
    assert result.eligible == 2
    assert result.imported == 2
    assert result.invalid_rows == 0
    conn = sqlite3.connect(target_db)
    rows = conn.execute("SELECT content FROM messages ORDER BY store_id").fetchall()
    conn.close()
    assert rows == [("root",), ("leaf",)]


def test_jsonl_import_falls_back_to_no_pruning_for_dangling_parent_chain(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "dangling-parent.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            jsonl_header("dangling-parent"),
            jsonl_message("old", "user", "valid old row"),
            jsonl_message("child", "assistant", "child with missing parent", parent_id="missing"),
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file], target_db=target_db, import_id="dangling-parent", apply=True
    )

    assert result.scanned == 2
    assert result.eligible == 2
    assert result.imported == 2
    assert result.invalid_rows == 0
    conn = sqlite3.connect(target_db)
    rows = conn.execute("SELECT content FROM messages ORDER BY store_id").fetchall()
    conn.close()
    assert rows == [("valid old row",), ("child with missing parent",)]


def test_jsonl_import_falls_back_to_no_pruning_for_parent_cycle(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "parent-cycle.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            jsonl_header("parent-cycle"),
            jsonl_message("old", "user", "valid old row"),
            jsonl_message("m1", "assistant", "cycle first", parent_id="m2"),
            jsonl_message("m2", "user", "cycle second", parent_id="m1"),
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file], target_db=target_db, import_id="parent-cycle", apply=True
    )

    assert result.scanned == 3
    assert result.eligible == 3
    assert result.imported == 3
    assert result.invalid_rows == 0
    conn = sqlite3.connect(target_db)
    rows = conn.execute("SELECT content FROM messages ORDER BY store_id").fetchall()
    conn.close()
    assert rows == [("valid old row",), ("cycle first",), ("cycle second",)]


def test_jsonl_import_malformed_message_rows_do_not_drive_leaf_pruning(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "malformed-tail.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            jsonl_header("malformed-tail"),
            jsonl_message("m1", "user", "root"),
            jsonl_message("m2", "assistant", "current", parent_id="m1"),
            {"type": "message", "id": "bad", "parentId": "m2"},
            {"type": "message", "id": "m1"},
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file], target_db=target_db, import_id="malformed-tail", apply=True
    )

    assert result.scanned == 4
    assert result.eligible == 2
    assert result.imported == 2
    assert result.invalid_rows == 2
    assert any("malformed-tail.jsonl:4" in warning for warning in result.warnings)
    assert any("malformed-tail.jsonl:5" in warning for warning in result.warnings)
    conn = sqlite3.connect(target_db)
    rows = conn.execute("SELECT role, content FROM messages ORDER BY store_id").fetchall()
    conn.close()
    assert rows == [("user", "root"), ("assistant", "current")]


def test_jsonl_import_leaf_path_traverses_malformed_importable_middle_node(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "malformed-middle.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            jsonl_header("malformed-middle"),
            jsonl_message("m1", "user", "root"),
            {"type": "message", "id": "bad", "parentId": "m1"},
            jsonl_message("m2", "assistant", "current", parent_id="bad"),
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file], target_db=target_db, import_id="malformed-middle", apply=True
    )

    assert result.scanned == 3
    assert result.eligible == 2
    assert result.imported == 2
    assert result.invalid_rows == 1
    assert any("malformed-middle.jsonl:3" in warning for warning in result.warnings)
    conn = sqlite3.connect(target_db)
    rows = conn.execute("SELECT role, content FROM messages ORDER BY store_id").fetchall()
    conn.close()
    assert rows == [("user", "root"), ("assistant", "current")]


def test_jsonl_import_malformed_native_function_call_does_not_drive_leaf_pruning(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "malformed-native-function-call.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            jsonl_header("malformed-native-function-call"),
            jsonl_message("m1", "user", "root"),
            jsonl_message("m2", "assistant", "current", parent_id="m1"),
            {"type": "function_call", "id": "bad", "parentId": "m1", "call_id": "call_bad", "arguments": "{}"},
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file], target_db=target_db, import_id="malformed-native-function-call", apply=True
    )

    assert result.scanned == 3
    assert result.eligible == 2
    assert result.imported == 2
    assert result.invalid_rows == 1
    assert result.skipped_empty == 0
    assert any("malformed-native-function-call.jsonl:4" in warning for warning in result.warnings)
    assert any("function_call row missing tool call id or name" in warning for warning in result.warnings)
    conn = sqlite3.connect(target_db)
    rows = conn.execute("SELECT role, content FROM messages ORDER BY store_id").fetchall()
    conn.close()
    assert rows == [("user", "root"), ("assistant", "current")]


def test_jsonl_import_malformed_nested_tool_call_content_does_not_drive_leaf_pruning(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "malformed-nested-tool-call.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            jsonl_header("malformed-nested-tool-call"),
            jsonl_message("m1", "user", "root"),
            jsonl_message("m2", "assistant", "current", parent_id="m1"),
            {
                "type": "message",
                "id": "bad",
                "parentId": "m1",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "toolCall",
                            "toolCallId": "call_bad",
                            "toolInput": {},
                        }
                    ],
                },
            },
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file], target_db=target_db, import_id="malformed-nested-tool-call", apply=True
    )

    assert result.scanned == 3
    assert result.eligible == 2
    assert result.imported == 2
    assert result.invalid_rows == 1
    assert result.skipped_empty == 0
    assert any("malformed-nested-tool-call.jsonl:4" in warning for warning in result.warnings)
    assert any("message content tool call item missing tool call id or name" in warning for warning in result.warnings)
    conn = sqlite3.connect(target_db)
    rows = conn.execute("SELECT role, content FROM messages ORDER BY store_id").fetchall()
    conn.close()
    assert rows == [("user", "root"), ("assistant", "current")]


def test_jsonl_import_reports_non_string_nested_tool_call_type_with_leaf_pruning(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "malformed-nested-tool-call-type.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            jsonl_header("malformed-nested-tool-call-type"),
            jsonl_message("m1", "user", "root"),
            {
                "type": "message",
                "id": "bad",
                "parentId": "m1",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": ["toolCall"], "toolCallId": "call_1", "toolName": "lookup", "toolInput": {}},
                    ],
                },
            },
            jsonl_message("m2", "assistant", "current", parent_id="m1"),
            {
                "type": "message",
                "id": "tr1",
                "message": {
                    "role": "toolResult",
                    "toolCallId": "call_1",
                    "toolName": "lookup",
                    "content": "lookup result",
                },
            },
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file], target_db=target_db, import_id="malformed-nested-tool-call-type", apply=True
    )

    assert result.scanned == 4
    assert result.eligible == 2
    assert result.imported == 2
    assert result.invalid_rows == 1
    assert result.skipped_empty == 0
    assert any("malformed-nested-tool-call-type.jsonl:3" in warning for warning in result.warnings)
    assert any("non-string type" in warning for warning in result.warnings)
    conn = sqlite3.connect(target_db)
    rows = conn.execute("SELECT role, content FROM messages ORDER BY store_id").fetchall()
    conn.close()
    assert rows == [("user", "root"), ("assistant", "current")]


def test_jsonl_import_skips_typed_non_message_content_rows(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "typed-metadata.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            jsonl_header("typed-metadata"),
            jsonl_message("m1", "user", "root"),
            {"type": "custom", "id": "state-1", "parentId": "m1", "content": "extension state"},
            {"type": "model_change", "content": "gpt-5.5"},
            jsonl_message("m2", "assistant", "current", parent_id="state-1"),
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file], target_db=target_db, import_id="typed-metadata", apply=True
    )

    assert result.scanned == 4
    assert result.eligible == 2
    assert result.imported == 2
    assert result.invalid_rows == 0
    conn = sqlite3.connect(target_db)
    rows = conn.execute("SELECT role, content FROM messages ORDER BY store_id").fetchall()
    conn.close()
    assert rows == [("user", "root"), ("assistant", "current")]


def test_jsonl_import_preserves_custom_message_rows(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "custom-message.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            jsonl_header("custom-message"),
            {"type": "custom_message", "id": "c1", "customType": "runtime-note", "content": "extension context", "display": False},
            jsonl_message("m1", "assistant", "reply", parent_id="c1"),
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file], target_db=target_db, import_id="custom-message", apply=True
    )

    assert result.scanned == 2
    assert result.eligible == 2
    assert result.imported == 2
    conn = sqlite3.connect(target_db)
    rows = conn.execute("SELECT role, content FROM messages ORDER BY store_id").fetchall()
    conn.close()
    assert rows == [("custom", "extension context"), ("assistant", "reply")]


def test_jsonl_import_keeps_idless_bare_rows_when_pruning_leaf_paths(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "mixed-bare.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            jsonl_header("mixed-bare"),
            jsonl_message("m1", "user", "root"),
            {"role": "system", "content": "legacy/idless context"},
            jsonl_message("m2", "assistant", "current", parent_id="m1"),
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file], target_db=target_db, import_id="mixed-bare", apply=True
    )

    assert result.scanned == 3
    assert result.eligible == 3
    assert result.imported == 3
    conn = sqlite3.connect(target_db)
    rows = conn.execute("SELECT role, content FROM messages ORDER BY store_id").fetchall()
    conn.close()
    assert rows == [("user", "root"), ("system", "legacy/idless context"), ("assistant", "current")]


def test_jsonl_import_preserves_bare_generic_tool_role_without_call_id(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "bare-legacy-tool-role.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            jsonl_header("s"),
            {"role": "tool", "content": "standalone legacy payload"},
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file], target_db=target_db, import_id="bare-legacy-tool-role", apply=True
    )

    assert result.scanned == 1
    assert result.eligible == 1
    assert result.imported == 1
    assert result.invalid_rows == 0
    conn = sqlite3.connect(target_db)
    rows = conn.execute(
        "SELECT role, content, tool_call_id, tool_calls, tool_name FROM messages ORDER BY store_id"
    ).fetchall()
    keys = conn.execute("SELECT source_message_key FROM lcm_imported_messages ORDER BY target_store_id").fetchall()
    conn.close()

    assert rows == [("tool", "standalone legacy payload", None, None, None)]
    assert keys == [(jsonl_key("s", "line:2"),)]


def test_jsonl_import_preserves_wrapped_generic_tool_role_without_call_id(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "wrapped-legacy-tool-role.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            jsonl_header("s"),
            {
                "type": "message",
                "id": "m1",
                "message": {"role": "tool", "content": "standalone legacy payload"},
            },
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file], target_db=target_db, import_id="wrapped-legacy-tool-role", apply=True
    )

    assert result.scanned == 1
    assert result.eligible == 1
    assert result.imported == 1
    assert result.invalid_rows == 0
    conn = sqlite3.connect(target_db)
    rows = conn.execute(
        "SELECT role, content, tool_call_id, tool_calls, tool_name FROM messages ORDER BY store_id"
    ).fetchall()
    keys = conn.execute("SELECT source_message_key FROM lcm_imported_messages ORDER BY target_store_id").fetchall()
    conn.close()

    assert rows == [("tool", "standalone legacy payload", None, None, None)]
    assert keys == [(jsonl_key("s", "m1"),)]


def test_jsonl_import_leaf_path_traverses_non_message_metadata_nodes(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "metadata-chain.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            jsonl_header("metadata-chain"),
            jsonl_message("m1", "user", "root"),
            {"type": "model_change", "id": "meta-1", "parentId": "m1", "model": "gpt-5.5"},
            jsonl_message("m2", "assistant", "after metadata", parent_id="meta-1"),
            jsonl_message("m3", "user", "latest", parent_id="m2"),
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file], target_db=target_db, import_id="metadata-chain", apply=True
    )

    assert result.scanned == 4
    assert result.eligible == 3
    assert result.imported == 3
    assert result.invalid_rows == 0
    conn = sqlite3.connect(target_db)
    rows = conn.execute("SELECT source, role, content FROM messages ORDER BY store_id").fetchall()
    conn.close()
    assert rows == [
        ("openclaw-jsonl:agent:unknown:metadata-chain", "user", "root"),
        ("openclaw-jsonl:agent:unknown:metadata-chain", "assistant", "after metadata"),
        ("openclaw-jsonl:agent:unknown:metadata-chain", "user", "latest"),
    ]


def test_jsonl_import_keeps_active_top_level_tool_result_with_leaf_path(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "tool-branch.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            jsonl_header("tool-branch"),
            {"type": "message", "id": "m1", "message": {"role": "user", "content": "u"}},
            {
                "type": "message",
                "id": "m2",
                "parentId": "m1",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "toolCall", "toolCallId": "call_1", "toolName": "lookup", "toolInput": {}},
                    ],
                },
            },
            {"type": "toolResult", "id": "tr1", "toolCallId": "call_1", "toolName": "lookup", "content": "result"},
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file], target_db=target_db, import_id="top-level-tool-result", apply=True
    )

    assert result.scanned == 3
    assert result.eligible == 3
    assert result.imported == 3
    conn = sqlite3.connect(target_db)
    rows = conn.execute(
        "SELECT role, content, tool_call_id, tool_calls, tool_name FROM messages ORDER BY store_id"
    ).fetchall()
    conn.close()
    assert rows[0] == ("user", "u", None, None, None)
    assert rows[1][0] == "assistant"
    assert json.loads(rows[1][3]) == [
        {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
    ]
    assert rows[2] == ("tool", "result", "call_1", None, "lookup")


def test_jsonl_import_keeps_responses_top_level_tool_result_with_leaf_path(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "responses-tool-branch.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            jsonl_header("responses-tool-branch"),
            {"type": "message", "id": "m1", "message": {"role": "user", "content": "u"}},
            {
                "type": "message",
                "id": "m2",
                "parentId": "m1",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "function_call",
                            "id": "fc_1",
                            "call_id": "call_1",
                            "name": "lookup",
                            "arguments": {},
                        },
                    ],
                },
            },
            {"type": "toolResult", "id": "tr1", "call_id": "call_1", "name": "lookup", "content": "result"},
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file], target_db=target_db, import_id="responses-top-level-tool-result", apply=True
    )

    assert result.scanned == 3
    assert result.eligible == 3
    assert result.imported == 3
    conn = sqlite3.connect(target_db)
    rows = conn.execute(
        "SELECT role, content, tool_call_id, tool_calls, tool_name FROM messages ORDER BY store_id"
    ).fetchall()
    conn.close()
    assert rows[0] == ("user", "u", None, None, None)
    assert rows[1][0] == "assistant"
    assert json.loads(rows[1][3]) == [
        {"id": "call_1", "type": "function_call", "function": {"name": "lookup", "arguments": "{}"}}
    ]
    assert rows[2] == ("tool", "result", "call_1", None, "lookup")


def test_jsonl_import_keeps_native_responses_function_call_output_with_leaf_path(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "responses-native-output-branch.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            jsonl_header("responses-native-output-branch"),
            {"type": "message", "id": "m1", "message": {"role": "user", "content": "u"}},
            {
                "type": "message",
                "id": "old",
                "parentId": "m1",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "function_call",
                            "id": "fc_old",
                            "call_id": "call_old",
                            "name": "lookup",
                            "arguments": "{}",
                        },
                    ],
                },
            },
            {
                "type": "message",
                "id": "m2",
                "parentId": "m1",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "function_call",
                            "id": "fc_1",
                            "call_id": "call_1",
                            "name": "lookup",
                            "arguments": "{}",
                        },
                    ],
                },
            },
            {"type": "function_call_output", "call_id": "call_old", "output": "old result"},
            {"type": "function_call_output", "call_id": "call_1", "output": "result"},
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file], target_db=target_db, import_id="responses-native-output-branch", apply=True
    )

    assert result.scanned == 5
    assert result.eligible == 3
    assert result.imported == 3
    conn = sqlite3.connect(target_db)
    rows = conn.execute(
        "SELECT role, content, tool_call_id, tool_calls, tool_name FROM messages ORDER BY store_id"
    ).fetchall()
    conn.close()
    assert rows[0] == ("user", "u", None, None, None)
    assert rows[1][0] == "assistant"
    assert rows[1][2] is None
    assert json.loads(rows[1][3]) == [
        {"id": "call_1", "type": "function_call", "function": {"name": "lookup", "arguments": "{}"}}
    ]
    assert rows[2] == ("tool", "result", "call_1", None, None)


def test_jsonl_import_prunes_idless_native_function_call_on_abandoned_branch(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "responses-idless-native-abandoned-branch.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            jsonl_header("responses-idless-native-abandoned-branch"),
            jsonl_message("root", "user", "root"),
            jsonl_message("old", "assistant", "old branch", parent_id="root"),
            {"type": "function_call", "parentId": "old", "call_id": "call_old", "name": "lookup", "arguments": {}},
            jsonl_message("leaf", "assistant", "current branch", parent_id="root"),
            {"type": "function_call_output", "call_id": "call_old", "output": "old result"},
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id="responses-idless-native-abandoned-branch",
        apply=True,
    )

    assert result.scanned == 5
    assert result.eligible == 2
    assert result.imported == 2
    assert result.invalid_rows == 0
    conn = sqlite3.connect(target_db)
    rows = conn.execute(
        "SELECT role, content, tool_call_id, tool_calls, tool_name FROM messages ORDER BY store_id"
    ).fetchall()
    import_keys = conn.execute(
        "SELECT source_message_key FROM lcm_imported_messages ORDER BY target_store_id"
    ).fetchall()
    conn.close()
    assert rows == [
        ("user", "root", None, None, None),
        ("assistant", "current branch", None, None, None),
    ]
    assert import_keys == [
        (jsonl_key("responses-idless-native-abandoned-branch", "root"),),
        (jsonl_key("responses-idless-native-abandoned-branch", "leaf"),),
    ]


def test_jsonl_import_keeps_active_nested_tool_result_with_leaf_path(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "nested-tool-branch.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            jsonl_header("nested-tool-branch"),
            {"type": "message", "id": "m1", "message": {"role": "user", "content": "u"}},
            {
                "type": "message",
                "id": "m2",
                "parentId": "m1",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "toolCall", "toolCallId": "call_1", "toolName": "lookup", "toolInput": {}},
                    ],
                },
            },
            {
                "type": "message",
                "id": "tr1",
                "message": {
                    "role": "toolResult",
                    "toolCallId": "call_1",
                    "toolName": "lookup",
                    "content": "nested result",
                },
            },
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file], target_db=target_db, import_id="nested-tool-result", apply=True
    )

    assert result.scanned == 3
    assert result.eligible == 3
    assert result.imported == 3
    conn = sqlite3.connect(target_db)
    rows = conn.execute(
        "SELECT role, content, tool_call_id, tool_calls, tool_name FROM messages ORDER BY store_id"
    ).fetchall()
    conn.close()
    assert rows[0] == ("user", "u", None, None, None)
    assert rows[1][0] == "assistant"
    assert json.loads(rows[1][3]) == [
        {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
    ]
    assert rows[2] == ("tool", "nested result", "call_1", None, "lookup")


def test_jsonl_import_follows_leaf_path_through_tool_result_parent(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "tool-parent-chain.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            jsonl_header("tool-parent-chain"),
            {"type": "message", "id": "m1", "message": {"role": "user", "content": "u"}},
            {
                "type": "message",
                "id": "m2",
                "parentId": "m1",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "toolCall", "toolCallId": "call_1", "toolName": "lookup", "toolInput": {}},
                    ],
                },
            },
            {
                "type": "message",
                "id": "tr1",
                "parentId": "m2",
                "message": {
                    "role": "toolResult",
                    "toolCallId": "call_1",
                    "toolName": "lookup",
                    "content": "result",
                },
            },
            {"type": "message", "id": "m3", "parentId": "tr1", "message": {"role": "assistant", "content": "final"}},
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file], target_db=target_db, import_id="tool-parent-chain", apply=True
    )

    assert result.scanned == 4
    assert result.eligible == 4
    assert result.imported == 4
    conn = sqlite3.connect(target_db)
    rows = conn.execute("SELECT role, content, tool_call_id FROM messages ORDER BY store_id").fetchall()
    conn.close()
    assert rows[0] == ("user", "u", None)
    assert rows[1][0] == "assistant"
    assert rows[2] == ("tool", "result", "call_1")
    assert rows[3] == ("assistant", "final", None)


def test_jsonl_import_does_not_prune_when_only_tool_result_has_parent_edge(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "tool-only-parent.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            jsonl_header("tool-only-parent"),
            {"type": "message", "id": "m1", "message": {"role": "user", "content": "u"}},
            {
                "type": "message",
                "id": "m2",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "toolCall", "toolCallId": "call_1", "toolName": "lookup", "toolInput": {}},
                    ],
                },
            },
            {
                "type": "message",
                "id": "tr1",
                "parentId": "m2",
                "message": {
                    "role": "toolResult",
                    "toolCallId": "call_1",
                    "toolName": "lookup",
                    "content": "result",
                },
            },
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file], target_db=target_db, import_id="tool-only-parent", apply=True
    )

    assert result.scanned == 3
    assert result.eligible == 3
    assert result.imported == 3
    conn = sqlite3.connect(target_db)
    rows = conn.execute("SELECT role, content, tool_call_id FROM messages ORDER BY store_id").fetchall()
    conn.close()
    assert rows[0] == ("user", "u", None)
    assert rows[1][0] == "assistant"
    assert rows[2] == ("tool", "result", "call_1")


def test_jsonl_import_applies_leaf_path_per_session_section(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "multi-session.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            jsonl_header("session-a"),
            {"type": "message", "id": "a-root", "message": {"role": "user", "content": "a root"}},
            {"type": "message", "id": "a-old", "parentId": "a-root", "message": {"role": "assistant", "content": "a old"}},
            {"type": "message", "id": "a-leaf", "parentId": "a-root", "message": {"role": "assistant", "content": "a current"}},
            jsonl_header("session-b"),
            {"type": "message", "id": "b-root", "message": {"role": "user", "content": "b root"}},
            {"type": "message", "id": "b-leaf", "parentId": "b-root", "message": {"role": "assistant", "content": "b current"}},
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file], target_db=target_db, import_id="multi-session-leaves", apply=True
    )

    assert result.scanned == 5
    assert result.eligible == 4
    assert result.imported == 4
    assert result.conversations == 2
    conn = sqlite3.connect(target_db)
    rows = conn.execute("SELECT source, content FROM messages ORDER BY store_id").fetchall()
    conn.close()
    assert rows == [
        ("openclaw-jsonl:agent:unknown:session-a", "a root"),
        ("openclaw-jsonl:agent:unknown:session-a", "a current"),
        ("openclaw-jsonl:agent:unknown:session-b", "b root"),
        ("openclaw-jsonl:agent:unknown:session-b", "b current"),
    ]


def test_jsonl_import_leaf_path_uses_nested_message_ids(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "nested-id-branch.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            jsonl_header("nested-id-branch"),
            {"type": "message", "message": {"id": "root", "role": "user", "content": "root"}},
            {"type": "message", "message": {"id": "old", "parentId": "root", "role": "assistant", "content": "old branch"}},
            {"type": "message", "message": {"id": "leaf", "parentId": "root", "role": "assistant", "content": "current branch"}},
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file], target_db=target_db, import_id="nested-id-leaf", apply=True
    )

    assert result.scanned == 3
    assert result.eligible == 2
    assert result.imported == 2
    conn = sqlite3.connect(target_db)
    rows = conn.execute("SELECT content FROM messages ORDER BY store_id").fetchall()
    keys = conn.execute("SELECT source_message_key FROM lcm_imported_messages ORDER BY target_store_id").fetchall()
    conn.close()
    assert rows == [("root",), ("current branch",)]
    assert keys == [(jsonl_key("nested-id-branch", "root"),), (jsonl_key("nested-id-branch", "leaf"),)]


def test_jsonl_import_leaf_path_aliases_typed_wrapper_ids_and_preserves_nested_keys(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "mixed-id-namespace.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            jsonl_header("mixed-id-namespace"),
            {"type": "message", "id": "env-1", "message": {"id": "msg-root", "role": "user", "content": "root"}},
            {
                "type": "message",
                "id": "env-2",
                "message": {"id": "msg-old", "parentId": "msg-root", "role": "assistant", "content": "old"},
            },
            {
                "type": "message",
                "id": "env-3",
                "message": {"id": "msg-leaf", "parentId": "msg-root", "role": "assistant", "content": "current"},
            },
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file], target_db=target_db, import_id="mixed-id-namespace", apply=True
    )

    assert result.scanned == 3
    assert result.eligible == 2
    assert result.imported == 2
    conn = sqlite3.connect(target_db)
    rows = conn.execute("SELECT content FROM messages ORDER BY store_id").fetchall()
    keys = conn.execute("SELECT source_message_key FROM lcm_imported_messages ORDER BY target_store_id").fetchall()
    conn.close()
    assert rows == [("root",), ("current",)]
    assert keys == [
        (jsonl_key("mixed-id-namespace", "msg-root"),),
        (jsonl_key("mixed-id-namespace", "msg-leaf"),),
    ]


def test_jsonl_import_prunes_typed_wrapper_branch_with_mixed_parent_aliases(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "typed-wrapper-mixed-parents.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            {"type": "session", "id": "s"},
            {"type": "message", "id": "env-root", "message": {"id": "root", "role": "user", "content": "root"}},
            {
                "type": "message",
                "id": "env-old",
                "message": {"id": "old", "parentId": "root", "role": "assistant", "content": "old branch"},
            },
            {
                "type": "message",
                "id": "env-leaf",
                "parentId": "env-root",
                "message": {"id": "leaf", "role": "assistant", "content": "current"},
            },
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file], target_db=target_db, import_id="typed-wrapper-mixed-parents", apply=True
    )

    assert result.scanned == 3
    assert result.eligible == 2
    assert result.imported == 2
    assert result.invalid_rows == 0
    conn = sqlite3.connect(target_db)
    rows = conn.execute("SELECT content FROM messages ORDER BY store_id").fetchall()
    keys = conn.execute("SELECT source_message_key FROM lcm_imported_messages ORDER BY target_store_id").fetchall()
    conn.close()
    assert rows == [("root",), ("current",)]
    assert keys == [(jsonl_key("s", "root"),), (jsonl_key("s", "leaf"),)]


def test_jsonl_import_typed_wrapper_catchup_respects_origin_main_nested_source_keys(
    tmp_path: Path,
):
    importer = load_importer_module()
    session_file = tmp_path / "typed-wrapper-origin-main-catchup.jsonl"
    target_db = tmp_path / "target-lcm.db"
    import_id = "typed-wrapper-origin-main-catchup"
    initial_rows = [
        {"type": "session", "id": "s"},
        {"type": "message", "id": "env-root", "message": {"id": "root", "role": "user", "content": "root"}},
        {
            "type": "message",
            "id": "env-old",
            "message": {"id": "old", "parentId": "root", "role": "assistant", "content": "old"},
        },
    ]
    write_jsonl_session(session_file, initial_rows)

    first = importer.import_jsonl_sessions(
        files=[session_file], target_db=target_db, import_id=import_id, apply=True
    )
    conn = sqlite3.connect(target_db)
    for target_store_id, row_id in [(1, "root"), (2, "old")]:
        source_message_key = jsonl_key("s", row_id)
        conn.execute(
            """UPDATE lcm_imported_messages
               SET source_message_id = ?, source_message_key = ?
               WHERE import_id = ? AND target_store_id = ?""",
            (
                importer._stable_positive_int(source_message_key),
                source_message_key,
                import_id,
                target_store_id,
            ),
        )
    conn.commit()
    conn.close()

    write_jsonl_session(
        session_file,
        [
            *initial_rows,
            {
                "type": "message",
                "id": "env-new",
                "parentId": "env-root",
                "message": {"id": "new", "role": "assistant", "content": "new"},
            },
        ],
    )
    second = importer.import_jsonl_sessions(
        files=[session_file], target_db=target_db, import_id=import_id, apply=True
    )

    assert first.imported == 2
    assert second.scanned == 3
    assert second.eligible == 2
    assert second.imported == 1
    assert second.skipped_existing == 1
    assert second.invalid_rows == 0
    conn = sqlite3.connect(target_db)
    rows = conn.execute("SELECT content FROM messages ORDER BY store_id").fetchall()
    keys = conn.execute("SELECT source_message_key FROM lcm_imported_messages ORDER BY target_store_id").fetchall()
    conn.close()
    assert rows == [("root",), ("old",), ("new",)]
    assert keys == [
        (jsonl_key("s", "root"),),
        (jsonl_key("s", "old"),),
        (jsonl_key("s", "new"),),
    ]


def test_jsonl_import_untyped_envelope_keys_survive_envelope_parent_catchup(
    tmp_path: Path,
):
    importer = load_importer_module()
    session_file = tmp_path / "untyped-envelope-catchup.jsonl"
    target_db = tmp_path / "target-lcm.db"
    initial_rows = [
        {"type": "session", "id": "s"},
        {"id": "env-root", "message": {"id": "root", "role": "user", "content": "root"}},
        {
            "id": "env-leaf",
            "message": {"id": "leaf", "parentId": "root", "role": "assistant", "content": "leaf"},
        },
    ]
    write_jsonl_session(session_file, initial_rows)

    first = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id="untyped-envelope-catchup",
        apply=True,
    )
    write_jsonl_session(
        session_file,
        [
            *initial_rows,
            {
                "id": "env-new",
                "parentId": "env-root",
                "message": {"id": "new", "role": "assistant", "content": "new"},
            },
        ],
    )
    second = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id="untyped-envelope-catchup",
        apply=True,
    )

    assert first.imported == 2
    assert second.scanned == 3
    assert second.eligible == 2
    assert second.imported == 1
    assert second.skipped_existing == 1
    assert second.invalid_rows == 0
    conn = sqlite3.connect(target_db)
    rows = conn.execute("SELECT content FROM messages ORDER BY store_id").fetchall()
    keys = conn.execute("SELECT source_message_key FROM lcm_imported_messages ORDER BY target_store_id").fetchall()
    conn.close()
    assert rows == [("root",), ("leaf",), ("new",)]
    assert keys == [
        (jsonl_key("s", "env-root"),),
        (jsonl_key("s", "env-leaf"),),
        (jsonl_key("s", "env-new"),),
    ]


def test_jsonl_import_untyped_envelope_keys_survive_nested_parent_catchup(
    tmp_path: Path,
):
    importer = load_importer_module()
    session_file = tmp_path / "untyped-envelope-nested-catchup.jsonl"
    target_db = tmp_path / "target-lcm.db"
    initial_rows = [
        {"type": "session", "id": "s"},
        {"id": "env-root", "message": {"id": "root", "role": "user", "content": "root"}},
        {
            "id": "env-leaf",
            "parentId": "env-root",
            "message": {"id": "leaf", "role": "assistant", "content": "leaf"},
        },
    ]
    write_jsonl_session(session_file, initial_rows)

    first = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id="untyped-envelope-nested-catchup",
        apply=True,
    )
    write_jsonl_session(
        session_file,
        [
            *initial_rows,
            {
                "id": "env-new",
                "message": {"id": "new", "parentId": "root", "role": "assistant", "content": "new"},
            },
        ],
    )
    second = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        import_id="untyped-envelope-nested-catchup",
        apply=True,
    )

    assert first.imported == 2
    assert second.scanned == 3
    assert second.eligible == 2
    assert second.imported == 1
    assert second.skipped_existing == 1
    assert second.invalid_rows == 0
    conn = sqlite3.connect(target_db)
    rows = conn.execute("SELECT content FROM messages ORDER BY store_id").fetchall()
    keys = conn.execute("SELECT source_message_key FROM lcm_imported_messages ORDER BY target_store_id").fetchall()
    conn.close()
    assert rows == [("root",), ("leaf",), ("new",)]
    assert keys == [
        (jsonl_key("s", "env-root"),),
        (jsonl_key("s", "env-leaf"),),
        (jsonl_key("s", "env-new"),),
    ]


def test_jsonl_import_uses_nested_message_id_when_top_level_id_is_missing(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "session-a.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            jsonl_header("session-a"),
            {"type": "message", "timestamp": "2026-06-10T00:00:01Z", "message": {"id": "nested-m1", "role": "user", "content": "nested id"}},
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file], target_db=target_db, import_id="nested-id", apply=True
    )

    assert result.imported == 1
    conn = sqlite3.connect(target_db)
    assert conn.execute("SELECT source_message_key FROM lcm_imported_messages").fetchall() == [
        (jsonl_key("session-a", "nested-m1"),)
    ]
    conn.close()


def test_jsonl_import_source_message_keys_are_unambiguous_with_colons(tmp_path: Path):
    importer = load_importer_module()
    first_file = tmp_path / "first.jsonl"
    second_file = tmp_path / "second.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(first_file, [jsonl_header("a:b"), jsonl_message("c", "user", "first")])
    write_jsonl_session(second_file, [jsonl_header("a"), jsonl_message("b:c", "user", "second")])

    result = importer.import_jsonl_sessions(
        files=[first_file, second_file], target_db=target_db, import_id="colon-keys", apply=True
    )

    assert result.imported == 2
    assert result.invalid_rows == 0
    conn = sqlite3.connect(target_db)
    rows = conn.execute("SELECT source_message_key, source_session FROM lcm_imported_messages ORDER BY source_session").fetchall()
    conn.close()
    assert rows == [(jsonl_key("a", "b:c"), "a"), (jsonl_key("a:b", "c"), "a:b")]


def test_jsonl_cli_empty_source_dir_non_json_output_labels_jsonl(tmp_path: Path, capsys):
    importer = load_importer_module()
    empty_dir = tmp_path / "empty-sessions"
    empty_dir.mkdir()
    target_db = tmp_path / "target-lcm.db"

    exit_code = importer.main(["--source-jsonl-dir", str(empty_dir), "--target-db", str(target_db)])

    assert exit_code == 0
    assert capsys.readouterr().out.splitlines()[0] == "jsonl import dry-run"


def test_jsonl_import_missing_file_is_fatal_before_apply_writes(tmp_path: Path):
    importer = load_importer_module()
    missing_file = tmp_path / "missing.jsonl"
    target_db = tmp_path / "target-lcm.db"

    with pytest.raises(FileNotFoundError, match="source JSONL file not found"):
        importer.import_jsonl_sessions(files=[missing_file], target_db=target_db, import_id="missing", apply=True)

    assert not target_db.exists()


def test_jsonl_import_file_stem_fallback_does_not_collide_across_directories(tmp_path: Path):
    importer = load_importer_module()
    first_file = tmp_path / "a" / "session.jsonl"
    second_file = tmp_path / "b" / "session.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(first_file, [jsonl_message("m1", "user", "first without header")])
    write_jsonl_session(second_file, [jsonl_message("m1", "user", "second without header")])

    result = importer.import_jsonl_sessions(
        files=[first_file, second_file], target_db=target_db, import_id="fallback-collision", apply=True
    )

    assert result.imported == 2
    assert result.invalid_rows == 0
    conn = sqlite3.connect(target_db)
    rows = conn.execute("SELECT session_id, content FROM messages ORDER BY content").fetchall()
    keys = conn.execute("SELECT source_message_key FROM lcm_imported_messages ORDER BY source_message_key").fetchall()
    conn.close()
    assert [row[1] for row in rows] == ["first without header", "second without header"]
    assert len({row[0] for row in rows}) == 2
    assert len({key[0] for key in keys}) == 2


def test_jsonl_import_accepts_top_level_typed_message_rows(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "typed.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            jsonl_header("typed-session"),
            {
                "type": "message",
                "id": "top-level-1",
                "role": "user",
                "content": "top level typed content",
                "timestamp": "2026-06-10T00:00:01Z",
            },
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file], target_db=target_db, import_id="typed-jsonl", apply=True
    )

    assert result.imported == 1
    assert result.invalid_rows == 0
    conn = sqlite3.connect(target_db)
    assert conn.execute("SELECT session_id, content FROM messages").fetchall() == [
        ("openclaw-jsonl:agent:unknown:typed-session", "top level typed content")
    ]
    conn.close()


def test_jsonl_import_accepts_untyped_envelope_message_rows(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "untyped-envelope.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            jsonl_header("untyped-envelope"),
            {"id": "env-1", "message": {"id": "nested-1", "role": "user", "content": "hi"}},
            {
                "id": "env-2",
                "parentId": "env-1",
                "message": {"id": "nested-2", "role": "assistant", "content": "hello"},
            },
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file], target_db=target_db, import_id="untyped-envelope", apply=True
    )

    assert result.imported == 2
    assert result.invalid_rows == 0
    conn = sqlite3.connect(target_db)
    rows = conn.execute("SELECT role, content FROM messages ORDER BY store_id").fetchall()
    keys = conn.execute("SELECT source_message_key FROM lcm_imported_messages ORDER BY target_store_id").fetchall()
    conn.close()
    assert rows == [("user", "hi"), ("assistant", "hello")]
    assert keys == [
        (jsonl_key("untyped-envelope", "env-1"),),
        (jsonl_key("untyped-envelope", "env-2"),),
    ]


def test_jsonl_cli_directory_default_import_id_is_stable_for_catchup(tmp_path: Path, capsys):
    importer = load_importer_module()
    sessions_dir = tmp_path / "sessions"
    target_db = tmp_path / "target-lcm.db"
    first_file = sessions_dir / "session-a.jsonl"
    write_jsonl_session(first_file, [jsonl_header("session-a"), jsonl_message("m1", "user", "first")])

    first_code = importer.main(
        ["--source-jsonl-dir", str(sessions_dir), "--target-db", str(target_db), "--apply", "--json"]
    )
    first_report = json.loads(capsys.readouterr().out)
    second_file = sessions_dir / "session-b.jsonl"
    write_jsonl_session(second_file, [jsonl_header("session-b"), jsonl_message("m1", "user", "second")])
    second_code = importer.main(
        ["--source-jsonl-dir", str(sessions_dir), "--target-db", str(target_db), "--apply", "--json"]
    )
    second_report = json.loads(capsys.readouterr().out)

    assert first_code == 0
    assert second_code == 0
    assert second_report["import_id"] == first_report["import_id"]
    assert second_report["imported"] == 1
    assert second_report["skipped_existing"] == 1
    conn = sqlite3.connect(target_db)
    assert conn.execute("SELECT content FROM messages ORDER BY store_id").fetchall() == [("first",), ("second",)]
    conn.close()


def test_jsonl_import_preserves_generic_row_session_identity(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "export.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [
            {
                "id": "r1",
                "session_id": "generic-session-42",
                "source": "discord:test-thread",
                "role": "user",
                "content": "generic export row",
                "timestamp": "2026-06-10T00:00:01Z",
            }
        ],
    )

    result = importer.import_jsonl_sessions(
        files=[session_file],
        target_db=target_db,
        namespace="hermes-jsonl",
        agent="nabu",
        import_id="generic-jsonl",
        apply=True,
    )

    assert result.imported == 1
    conn = sqlite3.connect(target_db)
    assert conn.execute("SELECT session_id, source FROM messages").fetchall() == [
        ("hermes-jsonl:agent:nabu:generic-session-42", "hermes-jsonl:agent:nabu:generic-session-42")
    ]
    assert conn.execute("SELECT source_message_key, source_session FROM lcm_imported_messages").fetchall() == [
        (jsonl_key("generic-session-42", "r1"), "generic-session-42")
    ]
    conn.close()


def test_jsonl_cli_json_report_contains_reconciliation_fields(tmp_path: Path, capsys):
    importer = load_importer_module()
    session_file = tmp_path / "session-a.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [jsonl_header("session-a"), jsonl_message("m1", "user", "hello jsonl")],
    )

    exit_code = importer.main(
        [
            "--source-jsonl",
            str(session_file),
            "--target-db",
            str(target_db),
            "--import-id",
            "jsonl-import",
            "--json",
        ]
    )

    assert exit_code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["scanned"] == 1
    assert report["eligible"] == 1
    assert report["would_import"] == 1
    assert report["imported"] == 0
    assert report["skipped_existing"] == 0
    assert report["skipped_empty"] == 0
    assert report["invalid_rows"] == 0
    assert report["warnings"] == []
    assert not target_db.exists()


def test_jsonl_cli_allows_empty_source_dir_as_zero_row_dry_run(tmp_path: Path, capsys):
    importer = load_importer_module()
    empty_dir = tmp_path / "empty-sessions"
    empty_dir.mkdir()
    target_db = tmp_path / "target-lcm.db"

    exit_code = importer.main(
        [
            "--source-jsonl-dir",
            str(empty_dir),
            "--target-db",
            str(target_db),
            "--json",
        ]
    )

    assert exit_code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["scanned"] == 0
    assert report["eligible"] == 0
    assert report["would_import"] == 0
    assert report["imported"] == 0
    assert report["warnings"] == []
    assert not target_db.exists()


def test_jsonl_import_backs_up_existing_target_before_writes(tmp_path: Path):
    importer = load_importer_module()
    session_file = tmp_path / "session-a.jsonl"
    target_db = tmp_path / "target-lcm.db"
    write_jsonl_session(
        session_file,
        [jsonl_header("session-a"), jsonl_message("m1", "user", "hello jsonl")],
    )
    existing_store = MessageStore(target_db)
    existing_store.append(
        "existing-session",
        {"role": "user", "content": "preexisting committed WAL row"},
        token_estimate=3,
        source="existing-source",
    )
    existing_store.close()

    result = importer.import_jsonl_sessions(
        files=[session_file], target_db=target_db, import_id="jsonl-import", apply=True
    )

    assert result.imported == 1
    assert result.backup_path is not None
    backup_conn = sqlite3.connect(result.backup_path)
    assert backup_conn.execute("SELECT session_id, content FROM messages").fetchall() == [
        ("existing-session", "preexisting committed WAL row")
    ]
    backup_conn.close()
