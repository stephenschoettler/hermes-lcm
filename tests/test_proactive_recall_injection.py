"""Fixture tests for SPEC F — proactive memory injection at assembly.

Exercises the real lcm_recall pipeline (seeded cross-session summary + vectors)
through the engine's ``_build_proactive_recall_message`` and its placement in
``_assemble_context``: injection appears when a relevant cross-session memory
exists; respects the budget/floor/dedupe; is inert (byte-identical assembly)
when disabled/unwarmed/no-user-turn; and injects nothing on timeout/failure.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from types import ModuleType

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.extraction import strip_injected_context_blocks
from hermes_lcm import tools as lcm_tools
from hermes_lcm.vector_store import VectorStore


def _import_lcm_engine():
    try:
        from hermes_lcm.engine import LCMEngine
        return LCMEngine
    except ModuleNotFoundError as exc:
        if exc.name not in {"agent", "agent.context_engine"}:
            raise
        agent_module = sys.modules.get("agent")
        if agent_module is None:
            agent_module = ModuleType("agent")
            agent_module.__path__ = []
            sys.modules["agent"] = agent_module
        context_engine_module = ModuleType("agent.context_engine")

        class ContextEngine:
            def on_session_reset(self):
                return None

        setattr(context_engine_module, "ContextEngine", ContextEngine)
        sys.modules["agent.context_engine"] = context_engine_module
        setattr(agent_module, "context_engine", context_engine_module)
        sys.modules.pop("hermes_lcm.engine", None)
        from hermes_lcm.engine import LCMEngine
        return LCMEngine


LCMEngine = _import_lcm_engine()

CURRENT = "sess-current"
QUERY = "kanban dashboard sprint plan"


class MockProvider:
    provider_id = "mock"
    model_id = "mock-model"
    dim = 2

    def __init__(self, vector=(1.0, 0.0)):
        self.vector = list(vector)

    def embed_query(self, text: str) -> list[float]:
        return list(self.vector)


def _make_engine(tmp_path: Path, **overrides) -> "LCMEngine":
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        fresh_tail_count=overrides.pop("fresh_tail_count", 2),
        embeddings_enabled=overrides.pop("embeddings_enabled", True),
        embedding_provider="mock",
        embedding_model="mock-model",
        embedding_query_timeout_s=2.0,
        proactive_recall_enabled=overrides.pop("proactive_recall_enabled", True),
        **overrides,
    )
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    engine.on_session_start(CURRENT, platform="discord", conversation_id="c:1")
    return engine


def _add_summary(engine, summary, *, session_id, created_at, latest_at=None):
    return engine._dag.add_node(
        SummaryNode(
            session_id=session_id,
            depth=0,
            summary=summary,
            token_count=20,
            source_token_count=40,
            source_ids=[],
            source_type="messages",
            created_at=created_at,
            earliest_at=created_at,
            latest_at=latest_at if latest_at is not None else created_at,
            expand_hint=f"Expand {summary[:20]}",
        )
    )


def _seed_summary_vectors(engine, rows, *, provider="mock"):
    store = VectorStore(engine._store.db_path, config=engine._config)
    try:
        store.register_profile("mock-model", provider, 2)
        identity = store.capture_identity("mock-model", provider=provider)
        for node_id, vector in rows:
            store.record_embedding(str(node_id), "summary", "mock-model", vector, identity=identity)
    finally:
        store.close()


@pytest.fixture
def provider(monkeypatch):
    p = MockProvider()
    monkeypatch.setattr(lcm_tools, "resolve_provider", lambda _config: p)
    return p


def _seed_cross_session_hit(engine, *, session_id="session-a", summary="kanban board dashboard sprint plan"):
    node = _add_summary(engine, summary, session_id=session_id, created_at=time.time())
    _seed_summary_vectors(engine, [(node, [1.0, 0.0])])
    return node


def _tail(text=QUERY):
    return [{"role": "user", "content": text}]


# ── Injection appears for a relevant cross-session memory ──


def test_injection_appears_for_cross_session_relevant_memory(tmp_path, provider):
    engine = _make_engine(tmp_path)
    _seed_cross_session_hit(engine)

    msg = engine._build_proactive_recall_message(_tail(), "user", set())

    assert msg is not None
    assert msg["role"] == "user"
    assert "<relevant-memories>" in msg["content"]
    assert "kanban board dashboard sprint" in msg["content"]
    assert "expand:" in msg["content"]
    # Provenance-honest: labeled as retrieved, not asserted fact.
    assert "not asserted as fact" in msg["content"]
    assert engine._proactive_recall_injected_count == 1


def test_injected_block_is_stripped_before_ingest(tmp_path, provider):
    engine = _make_engine(tmp_path)
    _seed_cross_session_hit(engine)

    msg = engine._build_proactive_recall_message(_tail(), "user", set())

    # The <relevant-memories> wrapper is a known injected-context tag, so the
    # block is removed before compaction/ingest and never re-enters the store.
    assert strip_injected_context_blocks(msg["content"]).strip() == ""


# ── Inert / byte-identical postures ──


def test_disabled_is_inert(tmp_path, provider):
    engine = _make_engine(tmp_path, proactive_recall_enabled=False)
    _seed_cross_session_hit(engine)

    assert engine._build_proactive_recall_message(_tail(), "user", set()) is None
    assert engine._proactive_recall_injected_count == 0
    assert engine._proactive_recall_skipped_count == 0


def test_unwarmed_embeddings_is_inert(tmp_path, provider):
    engine = _make_engine(tmp_path, embeddings_enabled=False)
    _seed_cross_session_hit(engine)

    assert engine._build_proactive_recall_message(_tail(), "user", set()) is None
    assert engine._proactive_recall_injected_count == 0


def test_no_user_turn_is_inert(tmp_path, provider):
    engine = _make_engine(tmp_path)
    _seed_cross_session_hit(engine)

    tail = [{"role": "assistant", "content": "tool-only continuation"}]
    assert engine._build_proactive_recall_message(tail, "user", set()) is None


# ── Dedupe: current session + summary-prefix node ids ──


def test_dedupe_drops_current_session_hits(tmp_path, provider):
    engine = _make_engine(tmp_path)
    _seed_cross_session_hit(engine, session_id=CURRENT, summary="current kanban dashboard sprint")

    # Only a current-session hit exists -> everything is already in context.
    assert engine._build_proactive_recall_message(_tail(), "user", set()) is None
    assert engine._proactive_recall_skipped_count == 1


def test_dedupe_drops_summary_prefix_node_ids(tmp_path, provider):
    engine = _make_engine(tmp_path)
    node = _seed_cross_session_hit(engine)

    # Passing the node id as already-in-prefix suppresses it.
    assert engine._build_proactive_recall_message(_tail(), "user", {node}) is None


# ── Relevance floor ──


def test_min_score_floor_drops_weak_hits(tmp_path, provider):
    engine = _make_engine(tmp_path, proactive_recall_min_score=10.0)
    _seed_cross_session_hit(engine)

    assert engine._build_proactive_recall_message(_tail(), "user", set()) is None
    assert engine._proactive_recall_skipped_count == 1


def test_rerank_keeps_proactive_floor_on_reported_rrf_score(tmp_path, monkeypatch, provider):
    """Reranking changes order only; proactive recall still filters the score
    reported by lcm_recall, which remains on the RRF/composite scale."""
    engine = _make_engine(
        tmp_path,
        rerank_enabled=True,
        proactive_recall_min_score=0.01,
    )
    monkeypatch.setattr(
        lcm_tools,
        "lcm_recall",
        lambda *a, **k: (
            '{"hits":[{"snippet":"useful older context",'
            '"score":0.016,"from_current_session":false}]}'
        ),
    )

    msg = engine._build_proactive_recall_message(_tail(), "user", set())

    assert msg is not None
    assert "useful older context" in msg["content"]


# ── Budget cap ──


def test_budget_zero_is_inert(tmp_path, provider):
    engine = _make_engine(tmp_path, proactive_recall_budget_tokens=0)
    _seed_cross_session_hit(engine)

    assert engine._build_proactive_recall_message(_tail(), "user", set()) is None


def test_block_respects_token_budget(tmp_path, provider):
    from hermes_lcm.tokens import count_message_tokens

    engine = _make_engine(tmp_path, proactive_recall_budget_tokens=120)
    # Three distinct cross-session hits all match the query vector.
    for i, sid in enumerate(("session-a", "session-b", "session-c")):
        node = _add_summary(
            engine,
            f"kanban dashboard sprint plan detail number {i} " + "lorem ipsum " * 12,
            session_id=sid,
            created_at=time.time(),
        )
        _seed_summary_vectors(engine, [(node, [1.0, 0.0])])

    msg = engine._build_proactive_recall_message(_tail(), "user", set())

    assert msg is not None
    assert count_message_tokens(msg) <= 120
    assert engine._proactive_recall_injected_count == 1


# ── Failure / timeout inject nothing (never block assembly) ──


def test_timeout_injects_nothing(tmp_path, monkeypatch, provider):
    engine = _make_engine(tmp_path)
    _seed_cross_session_hit(engine)
    monkeypatch.setattr(
        lcm_tools, "lcm_recall",
        lambda *a, **k: '{"timeout": true, "hits": []}',
    )

    assert engine._build_proactive_recall_message(_tail(), "user", set()) is None
    assert engine._proactive_recall_timeout_count == 1


def test_recall_exception_injects_nothing(tmp_path, monkeypatch, provider):
    engine = _make_engine(tmp_path)
    _seed_cross_session_hit(engine)

    def _boom(*a, **k):
        raise RuntimeError("recall exploded")

    monkeypatch.setattr(lcm_tools, "lcm_recall", _boom)

    assert engine._build_proactive_recall_message(_tail(), "user", set()) is None
    assert engine._proactive_recall_skipped_count == 1


# ── Placement inside _assemble_context: stable position, never in fresh tail ──


def test_assemble_context_places_block_between_summary_and_tail(tmp_path, provider):
    engine = _make_engine(tmp_path)
    _seed_cross_session_hit(engine)
    # Give the current session a summary node so a summary prefix exists.
    _add_summary(engine, "current session working notes", session_id=CURRENT, created_at=time.time())

    system_msg = {"role": "system", "content": "system prompt"}
    tail = [
        {"role": "user", "content": QUERY},
        {"role": "assistant", "content": "an earlier answer"},
    ]
    result = engine._assemble_context(system_msg, tail)

    contents = [str(m.get("content", "")) for m in result]
    mem_idx = next(i for i, c in enumerate(contents) if "<relevant-memories>" in c)
    # The block is not the leading anchor and not the last (fresh-tail) message.
    assert mem_idx > 0
    assert mem_idx < len(result) - 1
    # Never inside the fresh tail: the original tail messages carry no block.
    assert not any("<relevant-memories>" in str(m.get("content", "")) for m in tail)


def test_assemble_context_default_off_has_no_block(tmp_path, provider):
    engine = _make_engine(tmp_path, proactive_recall_enabled=False)
    _seed_cross_session_hit(engine)
    _add_summary(engine, "current session working notes", session_id=CURRENT, created_at=time.time())

    result = engine._assemble_context(
        {"role": "system", "content": "system prompt"},
        [{"role": "user", "content": QUERY}],
    )
    assert not any("<relevant-memories>" in str(m.get("content", "")) for m in result)
