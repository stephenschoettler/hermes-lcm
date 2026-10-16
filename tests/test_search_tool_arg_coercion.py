"""Hostile-argument coercion guards for the search-tool ``query`` argument.

A model is free to emit ``{"query": null}`` (or a number) for any ``lcm_*``
search tool. Every such tool must answer with the documented
``{"error": "No query provided"}`` payload rather than letting an
``AttributeError``/``TypeError`` escape the tool boundary, and must not treat a
non-string sentinel as a literal search term.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import hermes_lcm.tools as lcm_tools
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG
from hermes_lcm.store import MessageStore


@pytest.fixture
def engine(tmp_path):
    config = LCMConfig(database_path=str(tmp_path / "coercion.db"))
    store = MessageStore(config.database_path, ingest_protection_config=config)
    dag = SummaryDAG(config.database_path)
    engine = SimpleNamespace(
        _config=config,
        _store=store,
        _dag=dag,
        current_session_id="session-a",
    )
    try:
        yield engine
    finally:
        dag.close()
        store.close()


@pytest.mark.parametrize("mode", ["full_text", "semantic", "hybrid"])
def test_lcm_grep_null_query_returns_error_payload(engine, mode):
    payload = json.loads(lcm_tools.lcm_grep({"query": None, "mode": mode}, engine=engine))

    assert payload["error"] == "No query provided"


@pytest.mark.parametrize("mode", ["full_text", "semantic", "hybrid"])
def test_lcm_grep_missing_query_returns_error_payload(engine, mode):
    payload = json.loads(lcm_tools.lcm_grep({"mode": mode}, engine=engine))

    assert payload["error"] == "No query provided"


@pytest.mark.parametrize("mode", ["full_text", "semantic", "hybrid"])
def test_lcm_grep_whitespace_query_returns_error_payload(engine, mode):
    payload = json.loads(lcm_tools.lcm_grep({"query": "   ", "mode": mode}, engine=engine))

    assert payload["error"] == "No query provided"


def test_lcm_recall_null_query_returns_error_payload(engine):
    payload = json.loads(lcm_tools.lcm_recall({"query": None}, engine=engine))

    assert payload["error"] == "No query provided"


def test_lcm_recall_null_query_does_not_search_for_the_string_none(engine):
    """``str(None)`` is ``"None"`` — a truthy literal that must never be searched."""
    payload = json.loads(lcm_tools.lcm_recall({"query": None}, engine=engine))

    assert payload.get("query") != "None"


def test_lcm_expand_query_null_prompt_returns_error_payload(engine):
    payload = json.loads(lcm_tools.lcm_expand_query({"prompt": None}, engine=engine))

    assert payload["error"] == "prompt is required"


def test_lcm_grep_non_string_query_is_coerced_not_raised(engine):
    """A numeric query is a legitimate search term once coerced, not a crash."""
    payload = json.loads(lcm_tools.lcm_grep({"query": 42}, engine=engine))

    assert "error" not in payload or payload["error"] != "No query provided"
    assert payload.get("query") == "42"
