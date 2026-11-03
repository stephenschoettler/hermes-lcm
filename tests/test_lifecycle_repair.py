"""Tests for lifecycle_repair: table-scoped detection and in-place rebuild."""

from __future__ import annotations

import os
import sqlite3

import pytest

from hermes_lcm.lifecycle_repair import (  # noqa: F401  (path set up by conftest)
    LIFECYCLE_DDL,
    classify,
    heal_if_corrupt,
    is_corruption_error,
    rebuild,
    table_check,
)

PAGE_SIZE = 4096


def _build_db(path: str, *, lifecycle_rows: int = 20, message_rows: int = 40) -> None:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.execute("CREATE TABLE messages(id INTEGER PRIMARY KEY, content TEXT)")
    conn.execute(
        """
        CREATE TABLE lcm_lifecycle_state (
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
        "CREATE INDEX idx_lcm_lifecycle_current_session "
        "ON lcm_lifecycle_state(current_session_id)"
    )
    conn.execute(
        "CREATE INDEX idx_lcm_lifecycle_last_finalized_session "
        "ON lcm_lifecycle_state(last_finalized_session_id)"
    )
    conn.executemany(
        "INSERT INTO messages(content) VALUES(?)",
        [("y" * 200,) for _ in range(message_rows)],
    )
    conn.executemany(
        "INSERT INTO lcm_lifecycle_state(conversation_id, current_session_id, updated_at) "
        "VALUES(?,?,?)",
        [(f"conv-{i}", f"sess-{i}", 1.0) for i in range(lifecycle_rows)],
    )
    conn.commit()
    conn.close()


def _corrupt_lifecycle_root(path: str) -> None:
    """Overwrite the lifecycle b-tree root type byte, as a torn write would."""
    conn = sqlite3.connect(path)
    root = conn.execute(
        "SELECT rootpage FROM sqlite_master WHERE name='lcm_lifecycle_state'"
    ).fetchone()[0]
    conn.close()
    with open(path, "r+b") as handle:
        handle.seek((root - 1) * PAGE_SIZE)
        handle.write(b"\xff\xff")


def test_healthy_table_is_left_alone(tmp_path):
    db = str(tmp_path / "healthy.db")
    _build_db(db)
    conn = sqlite3.connect(db)
    try:
        assert table_check(conn) == ["ok"]
        outcome = heal_if_corrupt(conn)
        assert outcome["healed"] is False
        assert outcome["reason"] == "healthy"
        assert conn.execute("SELECT COUNT(*) FROM lcm_lifecycle_state").fetchone()[0] == 20
    finally:
        conn.close()


def test_lifecycle_only_corruption_is_rebuilt(tmp_path, monkeypatch):
    """Soft damage (logically inconsistent b-tree) is rebuilt in place."""
    db = str(tmp_path / "soft.db")
    _build_db(db)
    conn = sqlite3.connect(db)
    try:
        # Emulate the #643 signature: the table-scoped check reports damage and
        # the full check blames only lifecycle objects, while the file itself is
        # still droppable. Physical page damage is covered by the decline test.
        monkeypatch.setattr(
            "hermes_lcm.lifecycle_repair.table_check",
            lambda c: ["*** in database main ***\n"
                       "Tree 9 page 9 cell 3: Rowid 12 out of order"],
        )
        responses = iter([
            ["*** in database main ***\n"
             "Tree 9 page 9 cell 3: Rowid 12 out of order",
             "wrong # of entries in index idx_lcm_lifecycle_current_session"],
            ["ok"],  # post-rebuild verification
        ])
        monkeypatch.setattr(
            "hermes_lcm.lifecycle_repair._full_integrity", lambda c: next(responses)
        )
        outcome = heal_if_corrupt(conn)
    finally:
        conn.close()

    assert outcome["healed"] is True, outcome
    assert outcome["rows_before"] == 20
    assert outcome["rows_after"] == 0

    conn = sqlite3.connect(db)
    try:
        # content is untouched, the rebuild really did rebuild the table
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 40
        indexes = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' "
                "AND tbl_name='lcm_lifecycle_state'"
            ).fetchall()
        }
        assert "idx_lcm_lifecycle_current_session" in indexes
        assert "idx_lcm_lifecycle_last_finalized_session" in indexes
    finally:
        conn.close()


def test_physically_damaged_table_declines_without_raising(tmp_path):
    """Unreadable pages cannot be dropped in-process: decline, do not crash."""
    db = str(tmp_path / "hard.db")
    _build_db(db)
    _corrupt_lifecycle_root(db)

    conn = sqlite3.connect(db)
    try:
        assert table_check(conn) != ["ok"]
        outcome = heal_if_corrupt(conn)
    finally:
        conn.close()

    assert outcome["healed"] is False
    assert "malformed" in outcome["reason"] or "damage outside" in outcome["reason"]


def test_rebuild_preserves_messages_table(tmp_path):
    db = str(tmp_path / "rebuild.db")
    _build_db(db, lifecycle_rows=5, message_rows=7)
    conn = sqlite3.connect(db)
    try:
        stats = rebuild(conn)
        assert stats["rows_before"] == 5
        assert stats["rows_after"] == 0
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 7
    finally:
        conn.close()


def test_classify_resolves_tree_roots(tmp_path):
    db = str(tmp_path / "classify.db")
    _build_db(db)
    conn = sqlite3.connect(db)
    try:
        lifecycle_root = conn.execute(
            "SELECT rootpage FROM sqlite_master WHERE name='lcm_lifecycle_state'"
        ).fetchone()[0]
        messages_root = conn.execute(
            "SELECT rootpage FROM sqlite_master WHERE name='messages'"
        ).fetchone()[0]

        only, foreign = classify(
            [
                f"*** in database main ***\nTree {lifecycle_root} page 3 cell 1: Rowid 9 out of order",
                "wrong # of entries in index sqlite_autoindex_lcm_lifecycle_state_1",
            ],
            conn,
        )
        assert only is True
        assert foreign == []

        only, foreign = classify(
            [f"*** in database main ***\nTree {messages_root} page 4: invalid page number 0"],
            conn,
        )
        assert only is False
        assert len(foreign) == 1

        only, foreign = classify(["row 5 missing from index sqlite_autoindex_messages_1"], conn)
        assert only is False
    finally:
        conn.close()


@pytest.mark.parametrize(
    "message,expected",
    [
        ("database disk image is malformed", True),
        ("file is not a database", True),
        ("btreeInitPage() returns error code 11", True),
        ("UNIQUE constraint failed: lcm_lifecycle_state.conversation_id", False),
        ("database is locked", False),
    ],
)
def test_corruption_error_detection(message, expected):
    assert is_corruption_error(sqlite3.DatabaseError(message)) is expected


def test_ddl_matches_bootstrap_schema(tmp_path):
    """The rebuild must not drift from the schema db_bootstrap creates."""
    db = str(tmp_path / "ddl.db")
    conn = sqlite3.connect(db)
    try:
        conn.executescript(LIFECYCLE_DDL)
        columns = [
            row[1] for row in conn.execute("PRAGMA table_info(lcm_lifecycle_state)").fetchall()
        ]
        assert columns[:3] == ["conversation_id", "current_session_id", "last_finalized_session_id"]
        assert "updated_at" in columns
        assert table_check(conn) == ["ok"]
    finally:
        conn.close()
    assert os.path.getsize(db) > 0
