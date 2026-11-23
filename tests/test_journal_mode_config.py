"""Tests for the LCM plugin's database.journal_mode escape hatch.

The plugin used to run ``PRAGMA journal_mode=WAL`` unconditionally, so DELETE
was unavailable as an escape hatch for a WAL-coherency problem — the situation
behind the 2026-09-19/20/21 lcm.db corruptions. These tests pin both halves:
DELETE is honoured when requested, and a live WAL database is NEVER downgraded
(mirroring hermes_state_wal.apply_wal_with_fallback).

They also pin the FAILURE half: when the config cannot be read, the resolver
must report that rather than silently substituting ``wal``. That silent
substitution is what let ``database.journal_mode: delete`` be requested and
ignored with no signal anywhere.
"""

import logging
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import db_bootstrap  # noqa: E402
from db_bootstrap import configure_connection  # noqa: E402


def _fresh_db(tmp_path, name="t.db"):
    return tmp_path / name


def _mode(path):
    conn = sqlite3.connect(str(path))
    try:
        return conn.execute("PRAGMA journal_mode").fetchone()[0].lower()
    finally:
        conn.close()


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("LCM_JOURNAL_MODE", raising=False)
    monkeypatch.setattr(db_bootstrap, "_journal_mode_config_warned", False)
    yield


def _config_for(monkeypatch, mode):
    """Force the config-resolution path to report ``mode`` (no real config)."""
    monkeypatch.setattr(
        db_bootstrap,
        "_resolve_journal_mode_from_config",
        lambda log_failure=True: mode,
    )


@pytest.fixture
def _core_config(monkeypatch):
    """Replace core's ``load_config_readonly`` without needing hermes_cli on path."""
    import types

    module = types.ModuleType("hermes_cli.config")
    monkeypatch.setitem(sys.modules, "hermes_cli.config", module)
    return module


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("delete", "delete"),
        ("DELETE", "delete"),
        ("  delete  ", "delete"),
        ("wal", "wal"),
        ("truncate-then-explode", ""),
        ("", ""),
        (None, ""),
        (42, ""),
    ],
)
def test_config_value_validation(raw, expected, _core_config):
    """Only 'wal'/'delete' (case/space tolerant) are accepted; anything else ''."""
    _core_config.load_config_readonly = lambda: {"database": {"journal_mode": raw}}
    assert db_bootstrap._resolve_journal_mode_from_config() == expected


def test_unreadable_config_returns_empty_not_wal(_core_config):
    """THE BUG: an unreadable config must NOT be reported as 'wal'.

    Returning 'wal' made a failed read indistinguishable from an operator who
    asked for WAL, so the caller could not tell that a setting had been
    discarded.
    """

    def _boom():
        raise ModuleNotFoundError("No module named 'hermes_cli'")

    _core_config.load_config_readonly = _boom
    assert db_bootstrap._resolve_journal_mode_from_config(log_failure=False) == ""


def test_unreadable_config_logs_the_reason(caplog, _core_config):
    """A discarded operator setting must be traceable in the logs."""

    def _boom():
        raise TypeError("unsupported operand type(s) for |: 'type' and 'NoneType'")

    _core_config.load_config_readonly = _boom
    with caplog.at_level(logging.WARNING, logger=db_bootstrap.__name__):
        db_bootstrap._resolve_journal_mode_from_config()

    assert caplog.records, "an unreadable config produced no log record"
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "could not read database.journal_mode" in text
    assert "NOT" in text, "the warning must state that the setting is not applied"


def test_unreadable_config_falls_back_to_wal_without_raising(monkeypatch):
    """The fallback still happens — a broken config must not break the DB open.

    It just has to be traceable, which the previous bare ``return "wal"`` was not.
    """
    _config_for(monkeypatch, "")
    assert db_bootstrap._configured_journal_mode() == "wal"


def test_env_override_beats_an_unreadable_config(monkeypatch):
    """An explicit LCM_JOURNAL_MODE is honoured even with no readable config."""
    monkeypatch.setattr(
        db_bootstrap,
        "_resolve_journal_mode_from_config",
        lambda log_failure=True: pytest.fail("config must not be consulted"),
    )
    monkeypatch.setenv("LCM_JOURNAL_MODE", "delete")
    assert db_bootstrap._configured_journal_mode() == "delete"


def test_default_is_wal(tmp_path):
    db = _fresh_db(tmp_path)
    conn = sqlite3.connect(str(db))
    try:
        configure_connection(conn)
    finally:
        conn.close()
    assert _mode(db) == "wal"


def test_env_override_honours_delete(tmp_path, monkeypatch):
    monkeypatch.setenv("LCM_JOURNAL_MODE", "delete")
    db = _fresh_db(tmp_path)
    conn = sqlite3.connect(str(db))
    try:
        configure_connection(conn)
    finally:
        conn.close()
    assert _mode(db) == "delete"


def test_delete_is_never_live_downgraded(tmp_path, monkeypatch):
    """A live WAL database must keep WAL even when DELETE is requested."""
    db = _fresh_db(tmp_path)
    # Build it in WAL, with a second process-like handle still expected to exist.
    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE t (v TEXT)")
    conn.execute("INSERT INTO t VALUES ('kept')")
    conn.commit()
    assert _mode(db) == "wal"

    monkeypatch.setenv("LCM_JOURNAL_MODE", "delete")
    try:
        configure_connection(conn)  # must NOT flip the live database
        assert _mode(db) == "wal"
        assert conn.execute("SELECT v FROM t").fetchone()[0] == "kept"
    finally:
        conn.close()


def test_unrecognised_value_falls_back_to_wal(tmp_path, monkeypatch):
    monkeypatch.setenv("LCM_JOURNAL_MODE", "truncate-then-explode")
    db = _fresh_db(tmp_path)
    conn = sqlite3.connect(str(db))
    try:
        configure_connection(conn)
    finally:
        conn.close()
    assert _mode(db) == "wal"


def test_configured_delete_is_applied_to_a_fresh_db(tmp_path, monkeypatch):
    """database.journal_mode: delete on a NEW database actually takes effect."""
    _config_for(monkeypatch, "delete")
    db = _fresh_db(tmp_path)
    conn = sqlite3.connect(str(db))
    try:
        configure_connection(conn)
    finally:
        conn.close()
    assert _mode(db) == "delete"


def test_config_is_consulted_when_no_env_override(monkeypatch):
    _config_for(monkeypatch, "delete")
    assert db_bootstrap._configured_journal_mode() == "delete"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
