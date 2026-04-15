"""Tests for /lcm command surface and diagnostics."""

from pathlib import Path
import importlib.util
import sqlite3
import sys

import pytest

import hermes_lcm.command as command_mod
from hermes_lcm.command import _fmt_size, handle_lcm_command
from hermes_lcm.config import LCMConfig
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
    assert "message_sessions_total: 1" in result
    assert "messages_total: 1" in result
    assert "note: no active Hermes session has initialized LCM in this process yet" in result


def test_lcm_doctor_reports_health_checks(engine):
    result = handle_lcm_command("doctor", engine)

    assert "LCM doctor" in result
    assert "sqlite_integrity: ok" in result
    assert "messages_fts: ok" in result
    assert "nodes_fts: ok" in result


def test_lcm_help_on_unknown_subcommand(engine):
    result = handle_lcm_command("wat", engine)

    assert "Unknown subcommand: wat" in result
    assert "/lcm status" in result
    assert "/lcm doctor" in result


def test_lcm_doctor_clean_rejects_unknown_extra_args(engine):
    result = handle_lcm_command("doctor clean foo", engine)

    assert "currently supports only `clean`" in result
    assert "clean apply" not in result


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
