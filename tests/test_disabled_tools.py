"""Tests for the LCM_DISABLED_TOOLS env filter (local patch, Vocullum).

Disabled tools are excluded from injected schemas AND refused in
handle_tool_call, so they cost zero tokens per turn while remaining fully
reversible (unset the env var to restore all 15 tools).
"""
from __future__ import annotations

import json

import pytest

import hermes_lcm.embedding_provider  # noqa: F401  (module import side effects)
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

DISABLED = "lcm_compute,lcm_compile_evidence,lcm_evidence_pack,lcm_retrieve,lcm_query_state"


@pytest.fixture
def engine(tmp_path):
    config = LCMConfig(database_path=str(tmp_path / "disabled_tools.db"))
    e = LCMEngine(config=config)
    e._session_id = "test-session"
    try:
        yield e
    finally:
        e.shutdown()


def test_get_tool_schemas_full_without_env(engine, monkeypatch):
    monkeypatch.delenv("LCM_DISABLED_TOOLS", raising=False)
    names = {schema.get("name") for schema in engine.get_tool_schemas()}
    assert len(names) == 15
    assert "lcm_compute" in names
    assert "lcm_grep" in names


def test_get_tool_schemas_excludes_disabled(engine, monkeypatch):
    monkeypatch.setenv("LCM_DISABLED_TOOLS", DISABLED)
    names = {schema.get("name") for schema in engine.get_tool_schemas()}
    assert len(names) == 10
    assert "lcm_compute" not in names
    assert "lcm_query_state" not in names
    assert "lcm_grep" in names
    assert "lcm_recall" in names
    assert "lcm_status" in names


def test_handle_tool_call_refuses_disabled_before_ingest(engine, monkeypatch):
    monkeypatch.setenv("LCM_DISABLED_TOOLS", DISABLED)
    result = engine.handle_tool_call(
        "lcm_compute", {"expression": "1+1"}, messages=[{"role": "user", "content": "x"}]
    )
    payload = json.loads(result)
    assert payload["error"].startswith("LCM tool lcm_compute is disabled")


def test_handle_tool_call_allows_enabled_tools(engine, monkeypatch):
    monkeypatch.setenv("LCM_DISABLED_TOOLS", DISABLED)
    result = engine.handle_tool_call(
        "lcm_status", {}, messages=[{"role": "user", "content": "x"}]
    )
    # Enabled tools are dispatched, not refused with the disabled error.
    assert "disabled via LCM_DISABLED_TOOLS" not in result


def test_disabled_tool_names_parsing(monkeypatch):
    monkeypatch.setenv("LCM_DISABLED_TOOLS", " lcm_compute ,, lcm_retrieve ")
    assert LCMEngine._disabled_tool_names() == {"lcm_compute", "lcm_retrieve"}

    monkeypatch.setenv("LCM_DISABLED_TOOLS", "")
    assert LCMEngine._disabled_tool_names() == set()

    monkeypatch.delenv("LCM_DISABLED_TOOLS", raising=False)
    assert LCMEngine._disabled_tool_names() == set()
