"""Tests for the lossless-claw/OpenClaw LCM importer."""

from __future__ import annotations

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
