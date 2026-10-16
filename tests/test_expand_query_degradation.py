"""``lcm_expand_query`` degradation guards for unusable aux-LLM responses.

The expansion synthesis call is made through ``agent.auxiliary_client.call_llm``,
which can hand back an error-shaped or partial object under load. Reading
``response.choices[0].message.content`` unguarded turns that into an opaque
``TypeError``/``AttributeError`` that kills the whole tool call; the tool should
instead degrade to the same payload it already returns on a synthesis timeout,
so the caller keeps its node and raw matches.
"""

from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

import hermes_lcm.tools as lcm_tools
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.store import MessageStore


@pytest.fixture
def engine(tmp_path):
    config = LCMConfig(database_path=str(tmp_path / "expand.db"))
    store = MessageStore(config.database_path, ingest_protection_config=config)
    dag = SummaryDAG(config.database_path)
    engine = SimpleNamespace(
        _config=config,
        _store=store,
        _dag=dag,
        current_session_id="session-a",
    )
    store.append_batch(
        "session-a", [{"role": "user", "content": "the widget recovery code is ZX-9"}]
    )
    dag.add_node(
        SummaryNode(
            session_id="session-a",
            depth=0,
            summary="discussed the widget recovery code",
            token_count=8,
            source_token_count=16,
            source_ids=[1],
            source_type="messages",
            created_at=1.0,
            earliest_at=1.0,
            latest_at=1.0,
            expand_hint="widget recovery code",
        )
    )
    try:
        yield engine
    finally:
        dag.close()
        store.close()


def _stub_call_llm(monkeypatch, response):
    """Point ``agent.auxiliary_client.call_llm`` at a canned response object."""
    agent_module = sys.modules.get("agent")
    if agent_module is None:
        agent_module = ModuleType("agent")
        agent_module.__path__ = []
        monkeypatch.setitem(sys.modules, "agent", agent_module)

    auxiliary = ModuleType("agent.auxiliary_client")
    auxiliary.call_llm = lambda **kwargs: response
    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", auxiliary)
    monkeypatch.setattr(agent_module, "auxiliary_client", auxiliary, raising=False)


class _NonSubscriptableChoices:
    """Error-shaped response: ``choices`` is present but cannot be indexed.

    Mirrors an error envelope where the provider populated ``choices`` with a
    scalar/sentinel rather than a list of completions.
    """

    choices = object()


def _expand(engine):
    return json.loads(
        lcm_tools.lcm_expand_query(
            {"prompt": "what is the widget code?", "query": "widget"}, engine=engine
        )
    )


def test_none_choices_degrades_instead_of_raising(engine, monkeypatch):
    _stub_call_llm(monkeypatch, SimpleNamespace(choices=None))

    payload = _expand(engine)

    assert payload["degraded"] is True
    assert "synthesis unavailable" in payload["error"]


def test_missing_choices_attribute_degrades(engine, monkeypatch):
    _stub_call_llm(monkeypatch, SimpleNamespace())

    payload = _expand(engine)

    assert payload["degraded"] is True
    assert "synthesis unavailable" in payload["error"]


def test_empty_choices_list_degrades(engine, monkeypatch):
    _stub_call_llm(monkeypatch, SimpleNamespace(choices=[]))

    payload = _expand(engine)

    assert payload["degraded"] is True
    assert "synthesis unavailable" in payload["error"]


def test_non_subscriptable_choices_degrades(engine, monkeypatch):
    _stub_call_llm(monkeypatch, _NonSubscriptableChoices())

    payload = _expand(engine)

    assert payload["degraded"] is True
    assert "synthesis unavailable" in payload["error"]


def test_choice_without_message_degrades(engine, monkeypatch):
    _stub_call_llm(monkeypatch, SimpleNamespace(choices=[SimpleNamespace()]))

    payload = _expand(engine)

    assert payload["degraded"] is True


def test_degraded_payload_still_carries_matches(engine, monkeypatch):
    """Degrading must not throw away the retrieval work already done."""
    _stub_call_llm(monkeypatch, SimpleNamespace(choices=None))

    payload = _expand(engine)

    assert "matches" in payload
    assert "raw_matches" in payload
    assert "node_ids" in payload


def test_well_formed_response_still_answers(engine, monkeypatch):
    """Control: the guard must not perturb the happy path."""
    _stub_call_llm(
        monkeypatch,
        SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="The code is ZX-9."))]
        ),
    )

    payload = _expand(engine)

    assert payload.get("degraded") is not True
    assert payload["answer"] == "The code is ZX-9."
