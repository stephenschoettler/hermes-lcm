"""Detection and repair of content-drifted external-content FTS triggers.

``_fts_missing_triggers`` only checks that each expected trigger exists *by
name*. A trigger left behind by an older schema — one whose body still
references a since-renamed column — therefore passes the health check while
aborting every write to the content table at runtime, and a repair pass cannot
fix it because the spec's ``CREATE TRIGGER IF NOT EXISTS`` no-ops against the
existing name.

These tests pin both halves: the drift must be *detected*, and a repair must
actually *replace* the drifted body.
"""

import sqlite3

import pytest

from hermes_lcm import db_bootstrap
from hermes_lcm.db_bootstrap import (
    ensure_external_content_fts,
    external_content_fts_needs_repair,
    repair_external_content_fts,
)
from hermes_lcm.store import build_message_fts_spec


def _make_db(tmp_path, name="drift.db"):
    """A messages table with the real message-FTS spec applied."""
    conn = sqlite3.connect(str(tmp_path / name))
    conn.executescript(
        """
        CREATE TABLE messages (
            store_id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT
        );
        INSERT INTO messages(content) VALUES ('hello world');
        """
    )
    spec = build_message_fts_spec()
    ensure_external_content_fts(conn, spec)
    conn.commit()
    return conn, spec


def _drift_insert_trigger(conn):
    """Replace msg_fts_insert with an older-schema body (renamed column)."""
    conn.execute("DROP TRIGGER msg_fts_insert")
    conn.execute(
        """
        CREATE TRIGGER msg_fts_insert AFTER INSERT ON messages BEGIN
            INSERT INTO messages_fts(rowid, content)
                VALUES (new.store_id, new.body_text);
        END;
        """
    )
    conn.commit()


def _trigger_body(conn, name):
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='trigger' AND name = ?", (name,)
    ).fetchone()
    return row[0] if row else None


def test_healthy_triggers_are_not_reported_as_stale(tmp_path):
    """Control: the freshly-created triggers must compare equal to the spec."""
    conn, spec = _make_db(tmp_path)
    try:
        assert db_bootstrap._fts_stale_triggers(conn, spec) is False
        assert external_content_fts_needs_repair(conn, spec) is False
    finally:
        conn.close()


def test_drifted_trigger_body_is_detected_as_needing_repair(tmp_path):
    conn, spec = _make_db(tmp_path)
    try:
        _drift_insert_trigger(conn)

        # Present by name, so the name-only check is satisfied...
        assert db_bootstrap._fts_missing_triggers(conn, spec) is False
        # ...but the body has drifted and every write now aborts.
        assert external_content_fts_needs_repair(conn, spec) is True
    finally:
        conn.close()


def test_drifted_trigger_aborts_writes_until_repaired(tmp_path):
    conn, spec = _make_db(tmp_path)
    try:
        _drift_insert_trigger(conn)

        with pytest.raises(sqlite3.OperationalError, match="body_text"):
            conn.execute("INSERT INTO messages(content) VALUES ('blocked')")
            conn.commit()
        conn.rollback()

        repair_external_content_fts(conn, spec, throttle=False)

        conn.execute("INSERT INTO messages(content) VALUES ('recovered')")
        conn.commit()
    finally:
        conn.close()


def test_repair_replaces_the_drifted_trigger_body(tmp_path):
    conn, spec = _make_db(tmp_path)
    try:
        _drift_insert_trigger(conn)
        assert "body_text" in _trigger_body(conn, "msg_fts_insert")

        repair_external_content_fts(conn, spec, throttle=False)

        body = _trigger_body(conn, "msg_fts_insert")
        assert "body_text" not in body
        assert "new.content" in body
    finally:
        conn.close()


def test_repair_reports_triggers_recreated_for_stale_triggers(tmp_path):
    conn, spec = _make_db(tmp_path)
    try:
        _drift_insert_trigger(conn)

        result = repair_external_content_fts(conn, spec, throttle=False)

        assert result["triggers_recreated"] is True
    finally:
        conn.close()


def test_missing_trigger_still_reported_and_recreated(tmp_path):
    """The pre-existing missing-by-name path must keep working unchanged."""
    conn, spec = _make_db(tmp_path)
    try:
        conn.execute("DROP TRIGGER msg_fts_insert")
        conn.commit()

        assert db_bootstrap._fts_missing_triggers(conn, spec) is True
        assert external_content_fts_needs_repair(conn, spec) is True

        result = repair_external_content_fts(conn, spec, throttle=False)

        assert result["triggers_recreated"] is True
        assert _trigger_body(conn, "msg_fts_insert") is not None
    finally:
        conn.close()


def test_repair_is_idempotent_and_leaves_no_stale_state(tmp_path):
    conn, spec = _make_db(tmp_path)
    try:
        _drift_insert_trigger(conn)
        repair_external_content_fts(conn, spec, throttle=False)

        second = repair_external_content_fts(conn, spec, throttle=False)

        assert second["triggers_recreated"] is False
        assert external_content_fts_needs_repair(conn, spec) is False
    finally:
        conn.close()


@pytest.mark.parametrize(
    "stored, spec_sql",
    [
        # SQLite strips IF NOT EXISTS when it stores the trigger source.
        (
            "CREATE TRIGGER t AFTER INSERT ON m BEGIN SELECT 1; END",
            "CREATE TRIGGER IF NOT EXISTS t AFTER INSERT ON m BEGIN SELECT 1; END;",
        ),
        # Indentation/newlines in the spec literal are not drift.
        (
            "CREATE TRIGGER t AFTER INSERT ON m BEGIN SELECT 1; END",
            "\n    CREATE TRIGGER t\n        AFTER INSERT ON m BEGIN\n        SELECT 1;\n    END;\n",
        ),
        # Keyword case is not drift.
        (
            "create trigger t after insert on m begin select 1; end",
            "CREATE TRIGGER t AFTER INSERT ON m BEGIN SELECT 1; END",
        ),
    ],
)
def test_normalization_ignores_cosmetic_differences(stored, spec_sql):
    """Cosmetic storage differences must not be mistaken for real drift."""
    assert db_bootstrap._normalize_trigger_sql(stored) == db_bootstrap._normalize_trigger_sql(spec_sql)


def test_normalization_still_detects_real_body_differences():
    a = "CREATE TRIGGER t AFTER INSERT ON m BEGIN INSERT INTO f VALUES (new.content); END"
    b = "CREATE TRIGGER t AFTER INSERT ON m BEGIN INSERT INTO f VALUES (new.body_text); END"

    assert db_bootstrap._normalize_trigger_sql(a) != db_bootstrap._normalize_trigger_sql(b)
