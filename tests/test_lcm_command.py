"""Tests for /lcm command surface and diagnostics."""

import json
from pathlib import Path
import importlib.util
import sqlite3
import sys

import pytest

from hermes_lcm import tools as lcm_tools
import hermes_lcm.command as command_mod
from hermes_lcm.command import _fmt_size, handle_lcm_command
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine


@pytest.fixture
def engine(tmp_path):
    config = LCMConfig()
    config.database_path = str(tmp_path / "lcm_test.db")
    hermes_home = tmp_path / "hermes_home"
    e = LCMEngine(config=config, hermes_home=str(hermes_home))
    e._session_id = "test-session"
    e._session_platform = "telegram"
    e.context_length = 200000
    e.threshold_tokens = int(200000 * config.context_threshold)
    return e


def _replace_with_header_only_sqlite_db(e: LCMEngine) -> Path:
    """Replace the active DB file with a valid SQLite header and no tables."""
    db_path = Path(e._store.db_path)
    e.shutdown()
    for path in (db_path, Path(str(db_path) + "-wal"), Path(str(db_path) + "-shm")):
        path.unlink(missing_ok=True)
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("PRAGMA user_version = 2")
        conn.commit()
    e._store._conn = sqlite3.connect(str(db_path), timeout=5.0, check_same_thread=False)
    e._dag._conn = sqlite3.connect(str(db_path), timeout=5.0, check_same_thread=False)
    e._lifecycle._conn = sqlite3.connect(
        str(db_path),
        timeout=30.0,
        check_same_thread=False,
        isolation_level=None,
    )
    e._lifecycle._conn.row_factory = sqlite3.Row
    return db_path


def test_lcm_engine_declares_automatic_compaction_silent(engine):
    assert engine.emit_automatic_compaction_status is False
    assert engine.quiet_mode is True


def test_lcm_status_default_reports_current_session(engine):
    result = handle_lcm_command("", engine)

    assert "LCM status" in result
    assert "engine: lcm" in result
    assert "session_id: test-session" in result
    assert "cache_metrics_available: no" in result
    assert "last_cache_read_tokens: 0" in result
    assert "last_cache_write_tokens: 0" in result
    assert "last_compression_status: idle" in result
    assert "last_compression_noop_reason: (none)" in result
    assert "store_messages: 0" in result
    assert "dag_nodes: 0" in result


def test_lcm_status_reports_last_compression_noop_reason(engine):
    engine._last_compression_status = "noop"
    engine._last_compression_noop_reason = "no eligible raw backlog outside fresh tail"

    result = handle_lcm_command("status", engine)

    assert "last_compression_status: noop" in result
    assert (
        "last_compression_noop_reason: no eligible raw backlog outside fresh tail"
        in result
    )


def test_lcm_status_reports_cache_usage_metrics_when_host_provides_them(engine):
    engine.update_from_response({
        "prompt_tokens": 1050,
        "completion_tokens": 120,
        "total_tokens": 1170,
        "input_tokens": 600,
        "output_tokens": 120,
        "cache_read_tokens": 400,
        "cache_write_tokens": 50,
        "reasoning_tokens": 30,
    })

    result = handle_lcm_command("status", engine)

    assert "cache_metrics_available: yes" in result
    assert "last_input_tokens: 600" in result
    assert "last_output_tokens: 120" in result
    assert "last_cache_read_tokens: 400" in result
    assert "last_cache_write_tokens: 50" in result
    assert "last_reasoning_tokens: 30" in result
    assert "cache_read_ratio: 38.1%" in result


def test_update_from_response_treats_zero_cache_keys_as_available(engine):
    engine.update_from_response({
        "prompt_tokens": 600,
        "completion_tokens": 120,
        "total_tokens": 720,
        "input_tokens": 600,
        "output_tokens": 120,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "reasoning_tokens": 0,
    })

    status = engine.get_status()

    assert status["cache_metrics_available"] is True
    assert status["last_cache_read_tokens"] == 0
    assert status["last_cache_write_tokens"] == 0
    assert status["cache_read_ratio"] == 0.0


def test_lcm_status_does_not_leak_prior_session_compaction_count_after_rebind(engine):
    engine.compression_count = 4
    engine.last_prompt_tokens = 8000
    engine.on_session_start("fresh-session", platform="telegram", context_length=200000)

    result = handle_lcm_command("status", engine)

    assert "session_id: fresh-session" in result
    assert "compression_count: 0" in result
    assert "store_messages: 0" in result
    assert "dag_nodes: 0" in result


def test_lcm_status_explains_unbound_runtime_before_first_session(tmp_path):
    config = LCMConfig(database_path=str(tmp_path / "lcm_unbound.db"))
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes_home"))
    engine._store.append("telegram:chat-1", {"role": "user", "content": "hello"}, token_estimate=7)

    result = handle_lcm_command("status", engine)

    assert "LCM status" in result
    assert "session_id: (unbound)" in result
    assert "session_platform: (unbound)" in result
    assert "threshold_tokens: (uninitialized)" in result
    assert "\nmessage_sessions_total:" not in result
    assert "\nmessages_total:" not in result
    assert "\nsummary_nodes_total:" not in result
    assert "\nsummary_node_sessions_total:" not in result
    assert "note: no active Hermes session has initialized LCM in this process yet" in result


def test_lcm_status_reports_runtime_identity(engine):
    result = handle_lcm_command("status", engine)
    repo_root = Path(__file__).resolve().parent.parent

    assert "plugin_name: hermes-lcm" in result
    assert "plugin_version: 0.15.0" in result
    assert f"plugin_path: {repo_root}" in result
    assert "module_path:" in result
    assert "database_path_source: config.database_path" in result
    assert f"hermes_home: {engine._hermes_home}" in result
    assert "conversation_id:" in result


def test_lcm_status_reports_source_lineage_breakdown(engine):
    engine._store.append("test-session", {"role": "user", "content": "cli message"}, source="cli")
    engine._store.append("test-session", {"role": "user", "content": "unknown message"})
    engine._store._conn.execute(
        """INSERT INTO messages
           (session_id, source, role, content, tool_call_id, tool_calls, tool_name, timestamp, token_estimate, pinned)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("test-session", "", "user", "legacy blank source", None, None, None, 1.0, 5, 0),
    )
    engine._store._conn.commit()

    result = handle_lcm_command("status", engine)

    assert "plugin_git_commit:" in result
    assert "plugin_git_branch:" in result
    assert "plugin_git_dirty:" in result
    assert "source_messages_total: 3" in result
    assert "source_attributed_messages: 1" in result
    assert "source_unknown_messages: 1" in result
    assert "source_legacy_blank_messages: 1" in result
    assert "source_effective_unknown_messages: 2" in result


def test_lcm_doctor_reports_health_checks(engine):
    result = handle_lcm_command("doctor", engine)
    repo_root = Path(__file__).resolve().parent.parent

    assert "LCM doctor" in result
    assert "sqlite_integrity: ok" in result
    assert "messages_fts: ok" in result
    assert "nodes_fts: ok" in result
    assert "plugin_name: hermes-lcm" in result
    assert "plugin_version: 0.15.0" in result
    assert f"plugin_path: {repo_root}" in result
    assert "plugin_git_commit:" in result


def test_lcm_doctor_reports_heartbeat_noise_rows_without_mutating_or_leaking_content(engine):
    engine._store.append("heartbeat-session", {"role": "assistant", "content": "Still working..."}, token_estimate=2)
    engine._store.append("heartbeat-session", {"role": "user", "content": "Still working..."}, token_estimate=2)
    before = engine._store.get_session_count("heartbeat-session")

    result = handle_lcm_command("doctor", engine)
    after = engine._store.get_session_count("heartbeat-session")

    assert after == before
    assert "heartbeat_noise_rows:" in result
    assert "heartbeat_progress" in result
    assert "Still working" not in result


def test_lcm_doctor_tool_reports_heartbeat_noise_as_read_only_payload_detail(engine):
    engine._store.append("heartbeat-session", {"role": "assistant", "content": "Still working..."}, token_estimate=2)
    engine._store.append("heartbeat-session", {"role": "user", "content": "Still working..."}, token_estimate=2)

    doctor = json.loads(lcm_tools.lcm_doctor({}, engine=engine))
    payload = next(check for check in doctor["checks"] if check["check"] == "payload_storage")
    rows = payload["detail"]["heartbeat_noise_rows"]

    assert payload["status"] == "warn"
    assert rows == [
        {
            "store_id": 1,
            "session_id": "heartbeat-session",
            "source": "unknown",
            "role": "assistant",
            "field": "content",
            "length": 16,
            "content_len": 16,
            "suspicious_category": "heartbeat_progress",
        }
    ]
    assert "Still working" not in json.dumps(payload)


def test_lcm_doctor_text_reports_missing_externalized_payload_refs(engine):
    storage_dir = Path(engine._hermes_home) / "lcm-large-outputs"
    storage_dir.mkdir(parents=True)
    (storage_dir / "referenced.json").write_text(json.dumps({"content": "stored", "content_chars": 6}))
    (storage_dir / "unreferenced.json").write_text(json.dumps({"content": "orphaned", "content_chars": 8}))
    engine._store.append(
        "externalized-integrity-session",
        {
            "role": "assistant",
            "content": "\n".join(
                [
                    "[Externalized LCM ingest payload: kind=ingest_payload; field=content; chars=6; bytes=6; ref=referenced.json]",
                    "[GC'd externalized payload: kind=raw_payload; role=assistant; chars=10; ref=missing.json]",
                ]
            ),
        },
        token_estimate=2,
    )

    result = handle_lcm_command("doctor", engine)

    assert "externalized_payload_refs_total: 2" in result
    assert "externalized_payload_refs_existing: 1" in result
    assert "externalized_payload_refs_missing: 1" in result
    assert "externalized_payload_files_unreferenced: 1" in result
    assert "missing_externalized_payload_refs:" in result
    assert "missing.json" in result
    assert "inspect missing externalized payload refs and restore from backups if needed" in result
    assert "stored" not in result
    assert "orphaned" not in result


def test_lcm_doctor_finds_heartbeat_noise_after_many_short_nonmatches(engine):
    for idx in range(120):
        engine._store.append(
            "heartbeat-session",
            {"role": "assistant", "content": f"normal short status {idx}"},
            token_estimate=2,
        )
    engine._store.append("heartbeat-session", {"role": "assistant", "content": "Still working..."}, token_estimate=2)

    doctor = json.loads(lcm_tools.lcm_doctor({}, engine=engine))
    payload = next(check for check in doctor["checks"] if check["check"] == "payload_storage")
    rows = payload["detail"]["heartbeat_noise_rows"]

    assert payload["status"] == "warn"
    assert len(rows) == 1
    assert rows[0]["suspicious_category"] == "heartbeat_progress"
    assert rows[0]["store_id"] == 121
    assert "Still working" not in json.dumps(payload)


def test_lcm_doctor_flags_header_only_database_schema(engine):
    db_path = _replace_with_header_only_sqlite_db(engine)

    result = handle_lcm_command("doctor", engine)

    assert "LCM doctor" in result
    assert "status: issues-found" in result
    assert f"database_path: {db_path}" in result
    assert "schema_core_tables: missing" in result
    assert "schema_missing_tables:" in result
    assert "messages" in result
    assert "summary_nodes" in result
    assert "lcm_lifecycle_state" in result
    assert "verify HERMES_HOME/LCM_DATABASE_PATH point at the database inspected by Hermes" in result


def test_lcm_doctor_tool_flags_header_only_database_schema(engine):
    db_path = _replace_with_header_only_sqlite_db(engine)

    doctor = json.loads(lcm_tools.lcm_doctor({}, engine=engine))
    schema_check = next(check for check in doctor["checks"] if check["check"] == "schema_core_tables")

    assert doctor["overall"] == "unhealthy"
    assert schema_check["status"] == "fail"
    assert schema_check["detail"]["database_path"] == str(db_path)
    assert schema_check["detail"]["existing_tables"] == []
    assert "messages" in schema_check["detail"]["missing_tables"]
    assert "summary_nodes" in schema_check["detail"]["missing_tables"]
    assert "lcm_lifecycle_state" in schema_check["detail"]["missing_tables"]


def test_lcm_doctor_handles_closed_store_connection(engine):
    db_path = Path(engine._store.db_path)
    engine.shutdown()

    result = handle_lcm_command("doctor", engine)

    assert "LCM doctor" in result
    assert "status: issues-found" in result
    assert f"database_path: {db_path}" in result
    assert "schema_core_tables: error:" in result
    assert "LCM store connection is not initialized" in result


def test_lcm_doctor_prioritizes_schema_inspection_error(engine, monkeypatch):
    def _schema_error(_conn, *, database_path="", required_tables=()):
        return {
            "database_path": database_path,
            "required_tables": ["messages"],
            "existing_tables": [],
            "missing_tables": ["messages"],
            "error": "database disk image is malformed",
        }

    monkeypatch.setattr(command_mod, "inspect_lcm_schema_health", _schema_error)

    result = handle_lcm_command("doctor", engine)

    assert "schema_core_tables: error: database disk image is malformed" in result
    assert "schema_core_tables: missing" not in result
    assert "verify SQLite can read sqlite_master for the database inspected by Hermes" in result


def test_lcm_doctor_distinguishes_observations_from_recommended_actions(tmp_path):
    config = LCMConfig(
        database_path=str(tmp_path / "lcm_doctor_actions.db"),
        ignore_session_patterns=["cron*"],
    )
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes_home"))
    engine._session_id = "live-session"
    engine._session_platform = "telegram"
    engine._conversation_id = "live-session"
    engine._lifecycle.bind_session("live-session")
    engine._lifecycle.record_debt("live-session", kind="raw_backlog", size_estimate=321)
    engine._store.append("cron_20260414", {"role": "user", "content": "scheduled report"}, token_estimate=12)

    result = handle_lcm_command("doctor", engine)

    assert "observations:" in result
    assert "recommended_actions:" in result
    assert "maintenance_debt" in result
    assert "cleanup_candidates" in result
    assert "/lcm doctor clean" in result
    assert "/lcm backup" in result


def test_lcm_doctor_reports_legacy_blank_source_as_observation_without_warning(engine):
    engine._store.append("sess-known", {"role": "user", "content": "cli message"}, source="cli")
    engine._store.append("sess-unknown", {"role": "user", "content": "unknown message"})
    engine._store._conn.execute(
        """INSERT INTO messages
           (session_id, source, role, content, tool_call_id, tool_calls, tool_name, timestamp, token_estimate, pinned)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("legacy-session", "", "user", "legacy blank source", None, None, None, 1.0, 5, 0),
    )
    engine._store._conn.commit()

    result = handle_lcm_command("doctor", engine)

    assert "status: ok" in result
    assert "source_lineage:" in result
    assert "legacy_blank=1" in result
    assert "effective_unknown=2" in result
    assert "recommended_actions:\n- none" in result
    assert "review legacy blank-source rows before any destructive cleanup" not in result


def test_lcm_doctor_source_reports_dry_run_without_mutating(engine):
    engine._store.append("sess-known", {"role": "user", "content": "cli message"}, source="cli")
    for source in (None, "", "   ", "\t\n"):
        engine._store._conn.execute(
            """INSERT INTO messages
               (session_id, source, role, content, tool_call_id, tool_calls, tool_name, timestamp, token_estimate, pinned)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("legacy-session", source, "user", "legacy blank source", None, None, None, 1.0, 5, 0),
        )
    engine._store._conn.commit()

    result = handle_lcm_command("doctor source", engine)
    stats_after = engine._store.get_source_stats()

    assert "LCM doctor source" in result
    assert "status: normalization-needed" in result
    assert "legacy_blank_messages: 4" in result
    assert "would_update_messages: 4" in result
    assert "affected_sessions: 1" in result
    assert "target_source: unknown" in result
    assert "note: read-only scan only — no source rows were updated" in result
    assert "note: use `/lcm doctor source apply` to create a backup and normalize legacy blank-source rows" in result
    assert stats_after["legacy_blank_source_messages"] == 4


def test_lcm_doctor_source_apply_is_backup_first_and_idempotent(tmp_path):
    config = LCMConfig(database_path=str(tmp_path / "lcm_source_apply.db"))
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes_home"))
    engine._session_id = "live-session"
    engine._session_platform = "telegram"
    engine._conversation_id = "live-session"
    engine._store.append("sess-known", {"role": "user", "content": "cli message"}, source="cli")
    for source in (None, "", "   ", "\t\n"):
        engine._store._conn.execute(
            """INSERT INTO messages
               (session_id, source, role, content, tool_call_id, tool_calls, tool_name, timestamp, token_estimate, pinned)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("legacy-session", source, "user", "legacy blank source", None, None, None, 1.0, 5, 0),
        )
    engine._store._conn.commit()

    first = handle_lcm_command("doctor source apply", engine)
    second = handle_lcm_command("doctor source apply", engine)
    stats_after = engine._store.get_source_stats()

    assert "LCM doctor source apply" in first
    assert "status: ok" in first
    assert "updated_messages: 4" in first
    assert "legacy_blank_before: 4" in first
    assert "legacy_blank_after: 0" in first
    assert "note: backup created before source normalization apply" in first
    backup_line = next(line for line in first.splitlines() if line.startswith("backup_path: "))
    assert Path(backup_line.split(": ", 1)[1]).exists()
    assert "updated_messages: 0" in second
    assert stats_after["messages_total"] == 5
    assert stats_after["attributed_messages"] == 1
    assert stats_after["normalized_unknown_messages"] == 4
    assert stats_after["legacy_blank_source_messages"] == 0


def test_lcm_doctor_reports_lifecycle_fragmentation_as_read_only_observation(engine):
    state_db = Path(engine._hermes_home) / "state.db"
    state_db.parent.mkdir(parents=True, exist_ok=True)
    state_conn = sqlite3.connect(state_db)
    state_conn.executescript(
        """
        CREATE TABLE sessions (id TEXT PRIMARY KEY);
        INSERT INTO sessions(id) VALUES ('current-with-message');
        INSERT INTO sessions(id) VALUES ('state-only');
        """
    )
    state_conn.commit()
    state_conn.close()
    engine._store.append("current-with-message", {"role": "user", "content": "covered"}, source="cli")
    engine._dag.add_node(SummaryNode(
        session_id="node-missing-in-state",
        depth=0,
        summary="summary-only coverage",
        token_count=5,
        source_token_count=5,
        source_ids=[],
        source_type="messages",
        created_at=1.0,
    ))
    engine._lifecycle._conn.execute(
        """INSERT INTO lcm_lifecycle_state
           (conversation_id, current_session_id, last_finalized_session_id, current_frontier_store_id, last_finalized_frontier_store_id, updated_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        ("conv-current", "current-with-message", "node-missing-in-state", 0, 0, 1.0),
    )
    engine._lifecycle._conn.execute(
        """INSERT INTO lcm_lifecycle_state
           (conversation_id, current_session_id, last_finalized_session_id, current_frontier_store_id, last_finalized_frontier_store_id, updated_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        ("conv-stale", "missing-current", "missing-final", 0, 0, 1.0),
    )
    engine._lifecycle._conn.commit()

    result = handle_lcm_command("doctor", engine)

    assert "status: action-recommended" in result
    assert "lifecycle_fragmentation:" in result
    assert "lifecycle_rows=2" in result
    assert "current_missing_in_lcm_any=1" in result
    assert "current_missing_in_state=1" in result
    assert "node_sessions_missing_in_state=1" in result
    assert "state_sessions_missing_in_lcm_any=1" in result
    assert "lifecycle_fragmentation_classification: warn; 4 categories need review" in result
    assert "lifecycle_category stale_lifecycle_current: count=1 sample=missing-current" in result
    assert "lifecycle_category stale_lifecycle_finalized: count=1 sample=missing-final" in result
    assert "lifecycle_category lcm_node_sessions_missing_in_state: count=1 sample=node-missing-in-state" in result
    assert "lifecycle_category state_only_sessions: count=1 sample=state-only" in result
    assert "inspect lifecycle fragmentation before any cleanup/repair behavior mutates state" in result
    assert "read-only" in result
    assert engine._lifecycle.row_count() == 2


def test_lcm_doctor_warns_on_lcm_sessions_without_lifecycle_references(engine):
    engine.on_session_start("current-session", platform="cli", context_length=200000)
    engine._store.append("current-session", {"role": "user", "content": "covered"}, source="cli")
    engine._store.append("message-only-session", {"role": "user", "content": "missing lifecycle"}, source="cli")
    engine._dag.add_node(SummaryNode(
        session_id="node-only-session",
        depth=0,
        summary="missing lifecycle reference",
        token_count=5,
        source_token_count=5,
        source_ids=[],
        source_type="messages",
        created_at=1.0,
    ))

    result = handle_lcm_command("doctor", engine)

    assert "status: action-recommended" in result
    assert "message_sessions_without_lifecycle_current=1" in result
    assert "message_sessions_without_lifecycle_reference=1" in result
    assert "node_sessions_without_lifecycle_reference=1" in result
    assert "inspect lifecycle fragmentation before any cleanup/repair behavior mutates state" in result


def test_lcm_doctor_does_not_warn_on_last_finalized_message_session(engine):
    engine.on_session_start(
        "current-session",
        platform="cli",
        context_length=200000,
        conversation_id="conversation",
    )
    engine._store.append("previous-session", {"role": "user", "content": "previous"}, source="cli")
    engine._store.append("current-session", {"role": "user", "content": "current"}, source="cli")
    engine._lifecycle.record_rollover(
        "conversation",
        old_session_id="previous-session",
        new_session_id="current-session",
    )

    result = handle_lcm_command("doctor", engine)

    assert "status: ok" in result
    assert "message_sessions_without_lifecycle_current=1" in result
    assert "message_sessions_without_lifecycle_reference=0" in result
    assert "inspect lifecycle fragmentation before any cleanup/repair behavior mutates state" not in result


def test_lcm_help_on_unknown_subcommand(engine):
    result = handle_lcm_command("wat", engine)

    assert "Unknown subcommand: wat" in result
    assert "/lcm status" in result
    assert "/lcm doctor" in result


def test_lcm_doctor_clean_rejects_unknown_extra_args(engine):
    result = handle_lcm_command("doctor clean foo", engine)

    assert "currently supports `clean`, `clean apply`, `repair`, `repair apply`, `source`, `source apply`, and `retention`" in result
    assert "/lcm doctor clean apply" in result
    assert "/lcm doctor repair" in result
    assert "/lcm doctor repair apply" in result
    assert "/lcm doctor source" in result
    assert "/lcm doctor source apply" in result
    assert "/lcm doctor retention" in result


def test_lcm_doctor_repair_reports_fts_drift_without_mutating(tmp_path):
    config = LCMConfig(database_path=str(tmp_path / "lcm_repair_drift.db"))
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes_home"))
    engine._session_id = "live-session"
    engine._session_platform = "telegram"
    engine._conversation_id = "live-session"
    engine._store.append("live-session", {"role": "user", "content": "repairable message search"}, token_estimate=4)
    engine._dag.add_node(SummaryNode(
        session_id="live-session",
        depth=0,
        summary="repairable summary search",
        token_count=5,
        source_token_count=4,
        source_ids=[1],
        source_type="messages",
        created_at=1.0,
    ))
    engine._store._conn.execute("DROP TRIGGER msg_fts_insert")
    engine._store._conn.execute("DROP TRIGGER nodes_fts_insert")
    engine._store._conn.commit()

    result = handle_lcm_command("doctor repair", engine)

    assert "LCM doctor repair" in result
    assert "status: repair-needed" in result
    assert "messages_fts: repair-needed" in result
    assert "nodes_fts: repair-needed" in result
    assert "note: read-only scan only — no FTS tables were repaired" in result
    remaining_triggers = {
        row[0]
        for row in engine._store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger' AND name IN ('msg_fts_insert', 'nodes_fts_insert')"
        ).fetchall()
    }
    assert remaining_triggers == set()


def test_lcm_doctor_repair_dry_run_works_with_read_only_database(tmp_path):
    db_path = tmp_path / "lcm_repair_readonly.db"
    config = LCMConfig(database_path=str(db_path))
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes_home"))
    engine._session_id = "live-session"
    engine._session_platform = "telegram"
    engine._conversation_id = "live-session"
    engine._store.append("live-session", {"role": "user", "content": "readonly message search"}, token_estimate=4)
    engine._dag.add_node(SummaryNode(
        session_id="live-session",
        depth=0,
        summary="readonly summary search",
        token_count=5,
        source_token_count=4,
        source_ids=[1],
        source_type="messages",
        created_at=1.0,
    ))
    engine._store.close()
    engine._dag.close()
    engine._lifecycle.close()

    ro_conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

    class FakeStore:
        _conn = ro_conn

    class FakeEngine:
        _store = FakeStore()

    try:
        result = handle_lcm_command("doctor repair", FakeEngine())
    finally:
        ro_conn.close()

    assert "LCM doctor repair" in result
    assert "status: ok" in result
    assert "messages_fts: ok" in result
    assert "nodes_fts: ok" in result
    assert "repair-needed" not in result
    assert "note: read-only scan only — no FTS tables were repaired" in result


def test_lcm_doctor_repair_apply_is_backup_first_and_rebuilds_fts(tmp_path):
    config = LCMConfig(database_path=str(tmp_path / "lcm_repair_apply.db"))
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes_home"))
    engine._session_id = "live-session"
    engine._session_platform = "telegram"
    engine._conversation_id = "live-session"
    engine._store.append("live-session", {"role": "user", "content": "repair apply message search"}, token_estimate=4)
    engine._dag.add_node(SummaryNode(
        session_id="live-session",
        depth=0,
        summary="repair apply summary search",
        token_count=5,
        source_token_count=4,
        source_ids=[1],
        source_type="messages",
        created_at=1.0,
    ))
    engine._store._conn.execute("DELETE FROM messages_fts")
    engine._store._conn.execute("DELETE FROM nodes_fts")
    engine._store._conn.commit()

    result = handle_lcm_command("doctor repair apply", engine)

    assert "LCM doctor repair apply" in result
    assert "status: ok" in result
    backup_line = next(line for line in result.splitlines() if line.startswith("backup_path: "))
    backup_path = Path(backup_line.split(": ", 1)[1])
    assert backup_path.exists()
    assert "messages_fts_rebuilt: yes" in result
    assert "nodes_fts_rebuilt: yes" in result
    assert engine._store._conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0] == 1
    assert engine._store._conn.execute("SELECT COUNT(*) FROM nodes_fts").fetchone()[0] == 1
    assert len(engine._store.search("message", session_id="live-session")) == 1
    assert len(engine._dag.search("summary", session_id="live-session")) == 1


def test_lcm_doctor_retention_reports_old_heavy_sessions(tmp_path):
    config = LCMConfig(database_path=str(tmp_path / "lcm_retention.db"))
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes_home"))
    engine._session_id = "live-session"
    engine._session_platform = "telegram"
    engine._conversation_id = "live-session"
    engine._lifecycle.bind_session("live-session")

    live_store_id = engine._store.append("live-session", {"role": "user", "content": "fresh chat"}, token_estimate=8)
    old_store_id = engine._store.append("old-heavy", {"role": "user", "content": "archived chunk"}, token_estimate=240)
    engine._store._conn.execute("UPDATE messages SET timestamp = ? WHERE store_id = ?", (1.0, old_store_id))
    engine._store._conn.execute("UPDATE messages SET timestamp = ? WHERE store_id = ?", (2000000000.0, live_store_id))
    engine._store._conn.commit()
    engine._dag.add_node(SummaryNode(
        session_id="old-heavy",
        depth=0,
        summary="old heavy summary",
        token_count=32,
        source_token_count=240,
        source_ids=[old_store_id],
        source_type="messages",
        created_at=1.0,
        earliest_at=1.0,
        latest_at=1.0,
    ))

    result = handle_lcm_command("doctor retention", engine)

    assert "LCM doctor retention" in result
    assert "status: analysis-ready" in result
    assert "sessions_analyzed: 1" in result
    assert "stale_sessions_30d: 0" in result
    assert "stale_sessions_90d: 0" in result
    assert "retained_tokens_30d: 0" in result
    assert "retained_tokens_90d: 0" in result
    assert "retention_candidates:" in result
    assert "live-session | protected=yes" in result
    assert "old-heavy" not in result
    assert "note: retention analysis is scoped to the active session only" in result
    assert "note: read-only analysis only — no rows were deleted" in result


def test_lcm_doctor_retention_counts_summary_only_sessions(tmp_path):
    config = LCMConfig(database_path=str(tmp_path / "lcm_retention_summary_only.db"))
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes_home"))
    engine._session_id = "live-session"
    engine._session_platform = "telegram"
    engine._conversation_id = "live-session"
    engine._lifecycle.bind_session("live-session")

    engine._dag.add_node(SummaryNode(
        session_id="summary-only",
        depth=0,
        summary="summary only node",
        token_count=37,
        source_token_count=200,
        source_ids=[101],
        source_type="messages",
        created_at=1.0,
        earliest_at=1.0,
        latest_at=1.0,
    ))

    result = handle_lcm_command("doctor retention", engine)

    assert "sessions_analyzed: 0" in result
    assert "stale_sessions_30d: 0" in result
    assert "retained_tokens_30d: 0" in result
    assert "summary-only" not in result
    assert "result: no stored sessions found for retention analysis" in result


def test_lcm_doctor_retention_keeps_stale_sessions_visible_when_list_is_truncated(tmp_path):
    config = LCMConfig(database_path=str(tmp_path / "lcm_retention_many.db"))
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes_home"))
    engine._session_id = "live-session"
    engine._session_platform = "telegram"
    engine._conversation_id = "live-session"
    engine._lifecycle.bind_session("live-session")

    for idx in range(21):
        store_id = engine._store.append(f"fresh-heavy-{idx:02d}", {"role": "user", "content": "fresh heavy"}, token_estimate=500 + idx)
        engine._store._conn.execute("UPDATE messages SET timestamp = ? WHERE store_id = ?", (2000000000.0, store_id))

    stale_id = engine._store.append("stale-small", {"role": "user", "content": "old tiny"}, token_estimate=5)
    engine._store._conn.execute("UPDATE messages SET timestamp = ? WHERE store_id = ?", (1.0, stale_id))
    engine._store._conn.commit()

    result = handle_lcm_command("doctor retention", engine)

    assert "stale_sessions_30d: 0" in result
    assert "sessions_analyzed: 0" in result
    assert "stale-small" not in result
    assert "result: no stored sessions found for retention analysis" in result


def test_lcm_doctor_clean_reports_pattern_matched_junk_candidates(tmp_path):
    config = LCMConfig(
        database_path=str(tmp_path / "lcm_clean.db"),
        ignore_session_patterns=["cron*"],
        ignore_session_patterns_source="env",
    )
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes_home"))
    engine._store.append("cron_20260414", {"role": "user", "content": "scheduled report"}, token_estimate=12)
    engine._store.append("normal_session", {"role": "user", "content": "real conversation"}, token_estimate=20)

    result = handle_lcm_command("doctor clean", engine)

    assert "LCM doctor clean" in result
    assert "status: candidates-found" in result
    assert "ignored_pattern_matches: 1" in result
    assert "cron_20260414" in result
    assert "normal_session" not in result


def test_lcm_doctor_clean_prefers_ignore_over_stateless_when_both_match(tmp_path):
    config = LCMConfig(
        database_path=str(tmp_path / "lcm_overlap.db"),
        ignore_session_patterns=["cron*"],
        stateless_session_patterns=["cron*"],
    )
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes_home"))
    engine._store.append("cron_20260414", {"role": "user", "content": "scheduled report"}, token_estimate=12)

    result = handle_lcm_command("doctor clean", engine)

    assert "ignored_pattern_matches: 1" in result
    assert "stateless_pattern_matches: 0" in result
    assert "class=ignored-pattern" in result


def test_lcm_doctor_clean_returns_error_on_schema_problem(engine):
    engine._store._conn = _FakeConn()

    result = handle_lcm_command("doctor clean", engine)

    assert "LCM doctor clean" in result
    assert "status: error" in result
    assert "malformed schema" in result


def test_lcm_backup_creates_sqlite_snapshot(engine):
    engine._store.append(engine._session_id, {"role": "user", "content": "hello backup"}, token_estimate=11)

    result = handle_lcm_command("backup", engine)

    assert "LCM backup" in result
    backup_line = next(line for line in result.splitlines() if line.startswith("backup_path: "))
    backup_path = Path(backup_line.split(": ", 1)[1])
    assert backup_path.exists()
    assert backup_path.stat().st_size > 0


def test_lcm_backup_returns_error_when_sqlite_backup_fails(engine, monkeypatch):
    def boom(_path):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(command_mod.sqlite3, "connect", boom)

    result = handle_lcm_command("backup", engine)

    assert "LCM backup" in result
    assert "status: error" in result
    assert "disk I/O error" in result


def test_lcm_doctor_clean_apply_is_backup_first_and_deletes_safe_candidates(tmp_path):
    config = LCMConfig(
        database_path=str(tmp_path / "lcm_clean_apply.db"),
        ignore_session_patterns=["cron*"],
        doctor_clean_apply_enabled=True,
    )
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes_home"))
    engine._session_id = "live-session"
    engine._session_platform = "telegram"
    engine._conversation_id = "live-session"
    engine._lifecycle.bind_session("live-session")

    engine._store.append("cron_20260414", {"role": "user", "content": "scheduled report"}, token_estimate=12)
    engine._store.append("normal_session", {"role": "user", "content": "real conversation"}, token_estimate=20)
    engine._dag.add_node(SummaryNode(
        session_id="cron_20260414",
        depth=0,
        summary="scheduled report summary",
        token_count=5,
        source_token_count=12,
        source_ids=[1],
        source_type="messages",
        created_at=1.0,
    ))
    engine._lifecycle.bind_session("cron_20260414")
    engine._lifecycle.finalize_session("cron_20260414", "cron_20260414", frontier_store_id=1)

    result = handle_lcm_command("doctor clean apply", engine)

    assert "LCM doctor clean apply" in result
    assert "status: ok" in result
    backup_line = next(line for line in result.splitlines() if line.startswith("backup_path: "))
    backup_path = Path(backup_line.split(": ", 1)[1])
    assert backup_path.exists()
    assert engine._store.get_range("cron_20260414") == []
    assert engine._dag.get_session_nodes("cron_20260414") == []
    assert engine._lifecycle.get_by_conversation("cron_20260414") is None
    assert len(engine._store.get_range("normal_session")) == 1


def test_lcm_doctor_clean_apply_aborts_if_backup_fails(tmp_path, monkeypatch):
    config = LCMConfig(
        database_path=str(tmp_path / "lcm_clean_apply_fail.db"),
        ignore_session_patterns=["cron*"],
        doctor_clean_apply_enabled=True,
    )
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes_home"))
    engine._session_id = "live-session"
    engine._session_platform = "telegram"
    engine._conversation_id = "live-session"
    engine._lifecycle.bind_session("live-session")
    engine._store.append("cron_20260414", {"role": "user", "content": "scheduled report"}, token_estimate=12)

    def boom(_path):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(command_mod.sqlite3, "connect", boom)

    result = handle_lcm_command("doctor clean apply", engine)

    assert "LCM doctor clean apply" in result
    assert "status: error" in result
    assert "backup failed" in result.lower()
    assert len(engine._store.get_range("cron_20260414")) == 1


def test_lcm_doctor_clean_apply_rolls_back_if_delete_fails_after_backup(tmp_path):
    config = LCMConfig(
        database_path=str(tmp_path / "lcm_clean_apply_rollback.db"),
        ignore_session_patterns=["cron*"],
        doctor_clean_apply_enabled=True,
    )
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes_home"))
    engine._session_id = "live-session"
    engine._session_platform = "telegram"
    engine._conversation_id = "live-session"
    engine._lifecycle.bind_session("live-session")

    store_id = engine._store.append("cron_20260414", {"role": "user", "content": "scheduled report"}, token_estimate=12)
    engine._dag.add_node(SummaryNode(
        session_id="cron_20260414",
        depth=0,
        summary="scheduled report summary",
        token_count=5,
        source_token_count=12,
        source_ids=[store_id],
        source_type="messages",
        created_at=1.0,
    ))
    engine._lifecycle.bind_session("cron_20260414")
    engine._lifecycle.finalize_session("cron_20260414", "cron_20260414", frontier_store_id=store_id)
    engine._store._conn.execute(
        """
        CREATE TRIGGER fail_cron_node_delete
        BEFORE DELETE ON summary_nodes
        WHEN old.session_id = 'cron_20260414'
        BEGIN
            SELECT RAISE(ABORT, 'node delete failed');
        END
        """
    )
    engine._store._conn.commit()

    result = handle_lcm_command("doctor clean apply", engine)

    assert "LCM doctor clean apply" in result
    assert "status: error" in result
    assert "node delete failed" in result
    assert "cleanup apply rolled back" in result
    backup_line = next(line for line in result.splitlines() if line.startswith("backup_path: "))
    assert Path(backup_line.split(": ", 1)[1]).exists()
    assert len(engine._store.get_range("cron_20260414")) == 1
    assert len(engine._dag.get_session_nodes("cron_20260414")) == 1
    assert engine._lifecycle.get_by_conversation("cron_20260414") is not None


def test_lcm_doctor_clean_apply_denied_by_default(tmp_path):
    config = LCMConfig(
        database_path=str(tmp_path / "lcm_clean_apply_denied.db"),
        ignore_session_patterns=["cron*"],
    )
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes_home"))
    engine._store.append("cron_20260414", {"role": "user", "content": "scheduled report"}, token_estimate=12)

    result = handle_lcm_command("doctor clean apply", engine)

    assert "LCM doctor clean apply" in result
    assert "status: denied" in result
    assert "disabled by default" in result
    assert len(engine._store.get_range("cron_20260414")) == 1


class _FakeCursor:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


class _FakeConn:
    def execute(self, query):
        if "PRAGMA integrity_check" in query:
            return _FakeCursor(("ok",))
        raise sqlite3.OperationalError("malformed schema")


def test_lcm_doctor_reports_issues_instead_of_raising_on_schema_errors(engine):
    engine._store._conn = _FakeConn()
    engine._dag._conn = _FakeConn()

    result = handle_lcm_command("doctor", engine)

    assert "LCM doctor" in result
    assert "status: issues-found" in result
    assert "malformed schema" in result
    assert "issues:" in result


def test_fmt_size_reports_megabytes_correctly():
    assert _fmt_size(15_360_000) == "14.6 MB"


def test_register_skips_slash_command_when_host_context_has_no_register_command(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))

    spec = importlib.util.spec_from_file_location(
        "hermes_lcm_init_runtime",
        str(Path(__file__).resolve().parent.parent / "__init__.py"),
        submodule_search_locations=[str(Path(__file__).resolve().parent.parent)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)

    class _Ctx:
        def __init__(self):
            self.engine = None

        def register_context_engine(self, engine):
            self.engine = engine

        def register_tool(self, name, toolset, schema, handler, description="", emoji=""):
            pass

    ctx = _Ctx()
    module.register(ctx)

    assert ctx.engine is not None


def test_register_skips_lcm_slash_command_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))
    monkeypatch.delenv("LCM_ENABLE_SLASH_COMMAND", raising=False)

    spec = importlib.util.spec_from_file_location(
        "hermes_lcm_init_runtime_disabled",
        str(Path(__file__).resolve().parent.parent / "__init__.py"),
        submodule_search_locations=[str(Path(__file__).resolve().parent.parent)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)

    class _Ctx:
        def __init__(self):
            self.engine = None
            self.commands = {}

        def register_context_engine(self, engine):
            self.engine = engine

        def register_tool(self, name, toolset, schema, handler, description="", emoji=""):
            pass

        def register_command(self, name, handler, description=""):
            self.commands[name] = (handler, description)

    ctx = _Ctx()
    module.register(ctx)

    assert ctx.engine is not None
    assert "lcm" not in ctx.commands


def test_register_allows_lcm_slash_command_when_explicitly_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))
    monkeypatch.setenv("LCM_ENABLE_SLASH_COMMAND", "1")

    spec = importlib.util.spec_from_file_location(
        "hermes_lcm_init_runtime_enabled",
        str(Path(__file__).resolve().parent.parent / "__init__.py"),
        submodule_search_locations=[str(Path(__file__).resolve().parent.parent)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)

    class _Ctx:
        def __init__(self):
            self.engine = None
            self.commands = {}

        def register_context_engine(self, engine):
            self.engine = engine

        def register_tool(self, name, toolset, schema, handler, description="", emoji=""):
            pass

        def register_command(self, name, handler, description=""):
            self.commands[name] = (handler, description)

    ctx = _Ctx()
    module.register(ctx)

    assert ctx.engine is not None
    assert "lcm" in ctx.commands
