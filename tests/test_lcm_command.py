"""Tests for /lcm command surface and diagnostics."""

from pathlib import Path
import importlib.util
import sqlite3
import sys

import pytest

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


def test_lcm_status_default_reports_current_session(engine):
    result = handle_lcm_command("", engine)

    assert "LCM status" in result
    assert "engine: lcm" in result
    assert "session_id: test-session" in result
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

    assert "source_messages_total: 3" in result
    assert "source_attributed_messages: 1" in result
    assert "source_unknown_messages: 1" in result
    assert "source_legacy_blank_messages: 1" in result
    assert "source_effective_unknown_messages: 2" in result


def test_lcm_doctor_reports_health_checks(engine):
    result = handle_lcm_command("doctor", engine)

    assert "LCM doctor" in result
    assert "sqlite_integrity: ok" in result
    assert "messages_fts: ok" in result
    assert "nodes_fts: ok" in result


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


def test_lcm_doctor_reports_legacy_blank_source_observation_and_action(engine):
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

    assert "source_lineage:" in result
    assert "legacy_blank=1" in result
    assert "effective_unknown=2" in result
    assert "review legacy blank-source rows before any destructive cleanup" in result


def test_lcm_help_on_unknown_subcommand(engine):
    result = handle_lcm_command("wat", engine)

    assert "Unknown subcommand: wat" in result
    assert "/lcm status" in result
    assert "/lcm doctor" in result


def test_lcm_doctor_clean_rejects_unknown_extra_args(engine):
    result = handle_lcm_command("doctor clean foo", engine)

    assert "currently supports `clean`, `clean apply`, and `retention`" in result
    assert "/lcm doctor clean apply" in result
    assert "/lcm doctor retention" in result


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

        def register_command(self, name, handler, description=""):
            self.commands[name] = (handler, description)

    ctx = _Ctx()
    module.register(ctx)

    assert ctx.engine is not None
    assert "lcm" in ctx.commands
