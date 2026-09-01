"""Tests for lcm_recall — the cross-conversation forever-memory recall tool.

Seeds summaries + chunks + raw messages across three synthetic sessions and
asserts the fused pipeline recalls cross-session WITHOUT a session filter, that
scope_bias and recency are soft ranking boosts (never filters), that chunk hits
dedupe against FTS by store_id, that rerank failures skip silently, and that the
degrade matrix (embeddings-off) still returns the FTS arm.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import time
from types import SimpleNamespace

import pytest

import hermes_lcm.tools as lcm_tools
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.store import MessageStore
from hermes_lcm.vector_store import EmbeddingIdentity, VectorStore

CURRENT = "session-cur"


class MockProvider:
    provider_id = "mock"
    model_id = "mock-model"
    dim = 2

    def __init__(self, vector=(1.0, 0.0)):
        self.vector = list(vector)
        self.queries: list[str] = []
        self.last_usage_tokens = 7

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return list(self.vector)


@pytest.fixture
def recall_engine(tmp_path):
    config = LCMConfig(
        database_path=str(tmp_path / "recall.db"),
        embeddings_enabled=True,
        embedding_provider="mock",
        embedding_model="mock-model",
        embedding_query_timeout_s=2.0,
    )
    store = MessageStore(config.database_path, ingest_protection_config=config)
    dag = SummaryDAG(config.database_path)
    engine = SimpleNamespace(
        _config=config,
        _store=store,
        _dag=dag,
        _hermes_home=str(tmp_path),
        current_session_id=CURRENT,
    )
    try:
        yield engine
    finally:
        dag.close()
        store.close()


def _add_summary(
    engine,
    summary,
    *,
    session_id,
    created_at,
    latest_at=None,
    source_ids=None,
):
    return engine._dag.add_node(
        SummaryNode(
            session_id=session_id,
            depth=0,
            summary=summary,
            token_count=20,
            source_token_count=40,
            source_ids=list(source_ids or []),
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


def _chunk_identity():
    return EmbeddingIdentity.canonical("mock", "mock-model", "", 2, "float32", "little", "chunk")


def _seed_chunk_vectors(engine, rows):
    """rows: (store_id, chunk_index, char_start, char_end, vector)."""
    store = VectorStore(engine._store.db_path, config=engine._config)
    try:
        store.register_profile("mock-model", "mock", 2, task="chunk")
        identity = _chunk_identity()
        for store_id, chunk_index, char_start, char_end, vector in rows:
            store.record_chunk_embedding(
                f"{store_id}:{chunk_index}",
                "mock-model",
                vector,
                store_id=store_id,
                chunk_index=chunk_index,
                char_start=char_start,
                char_end=char_end,
                token_estimate=5,
                identity=identity,
            )
    finally:
        store.close()


def _recall(engine, monkeypatch, provider=None, **args):
    monkeypatch.setattr(lcm_tools, "resolve_provider", lambda _config: provider or MockProvider())
    payload = json.loads(lcm_tools.lcm_recall({"query": "kanban dashboard sprint", **args}, engine=engine))
    return payload


def _summary_hit(engine, node_id):
    node = engine._dag.get_node(node_id)
    assert node is not None
    return {
        "kind": "summary",
        "node_id": node.node_id,
        "session_id": node.session_id,
        "timestamp": node.latest_at or node.created_at or 0,
        "snippet": node.summary[:300],
        "from_current_session": node.session_id == engine.current_session_id,
        "expand_hint": f"lcm_load_session(session_id='{node.session_id}')",
    }


def _non_strict(engine):
    """Pin a case to the pre-F35 delivery by disabling reference-strict.

    These cases exercise answer_ready MECHANICS (session diversity, the
    hydration budget, the response cap) using summary hits, which the default
    reference-strict delivery no longer hands out as evidence. Running them on
    the disabled path keeps them exercising the mechanic they were written for
    and doubles as the byte-identity guarantee for the opt-out.
    """
    engine._config.recall_reference_strict = False
    return engine


def _patch_summary_arm(monkeypatch, hits):
    monkeypatch.setattr(
        lcm_tools,
        "_lcm_recall_summary_arm",
        lambda *_args, **_kwargs: (list(hits), "full", len(hits), len(hits), []),
    )


def test_voyage_chunk_recall_uses_context_model(recall_engine, monkeypatch):
    summary = MockProvider()
    summary.provider_id = "voyage"
    summary.model_id = "voyage-3"
    chunk = MockProvider(vector=(0.0, 1.0))
    chunk.provider_id = "voyage"
    chunk.model_id = "voyage-context-4"
    recall_engine._config.embedding_provider = "voyage"
    recall_engine._config.embedding_model = "voyage-3"

    def resolve(config):
        return chunk if config.embedding_model == "voyage-context-4" else summary

    captured = {}

    def chunk_arm(_engine, *, query_vector, provider, **_kwargs):
        captured["model"] = provider.model_id
        captured["query_vector"] = query_vector
        return [], "none", None, None

    monkeypatch.setattr(lcm_tools, "resolve_provider", resolve)
    monkeypatch.setattr(lcm_tools, "_lcm_recall_fts_arm", lambda *_a, **_k: ([], None))
    monkeypatch.setattr(lcm_tools, "_lcm_recall_chunk_arm", chunk_arm)

    json.loads(
        lcm_tools.lcm_recall(
            {"query": "context query", "include": "verbatim"},
            engine=recall_engine,
        )
    )

    assert summary.queries == []
    assert chunk.queries == ["context query"]
    assert captured == {
        "model": "voyage-context-4",
        "query_vector": [0.0, 1.0],
    }


def test_recall_returns_cross_session_summaries_without_a_filter(recall_engine, monkeypatch):
    other_a = _add_summary(recall_engine, "kanban board dashboard sprint plan", session_id="session-a", created_at=10.0)
    other_b = _add_summary(recall_engine, "fleet archive sprint board", session_id="session-b", created_at=11.0)
    here = _add_summary(recall_engine, "unrelated current note", session_id=CURRENT, created_at=12.0)
    _seed_summary_vectors(
        recall_engine,
        [(other_a, [1.0, 0.0]), (other_b, [0.9, 0.436]), (here, [0.0, 1.0])],
    )

    payload = _recall(recall_engine, monkeypatch, include="summaries", scope_bias=0.0, limit=5)

    node_ids = [hit["node_id"] for hit in payload["hits"]]
    assert node_ids[0] == other_a and other_b in node_ids
    # The strongest hits come from OTHER conversations — no session filter applied.
    sessions = {hit["session_id"] for hit in payload["hits"][:2]}
    assert sessions == {"session-a", "session-b"}
    assert all(hit["kind"] == "summary" for hit in payload["hits"])
    assert payload["provenance"]["arms_run"] == ["summary"]


def test_summary_hit_carries_direct_source_store_id(recall_engine, monkeypatch):
    store_id = recall_engine._store.append(
        "session-a",
        {"role": "user", "content": "kanban dashboard sprint source row"},
    )
    node_id = _add_summary(
        recall_engine,
        "kanban dashboard sprint summary",
        session_id="session-a",
        created_at=10.0,
        source_ids=[store_id],
    )
    _seed_summary_vectors(recall_engine, [(node_id, [1.0, 0.0])])

    payload = _recall(
        recall_engine,
        monkeypatch,
        include="summaries",
        scope_bias=0.0,
        limit=5,
    )

    hit = next(hit for hit in payload["hits"] if hit["node_id"] == node_id)
    assert hit["kind"] == "summary"
    assert hit["store_id"] == store_id


def test_scope_bias_boosts_current_conversation_without_filtering(recall_engine, monkeypatch):
    cross = _add_summary(recall_engine, "cross conversation kanban", session_id="session-a", created_at=5.0)
    here = _add_summary(recall_engine, "current conversation kanban", session_id=CURRENT, created_at=5.0)
    # cross scores higher (rank 1); current is rank 2.
    _seed_summary_vectors(recall_engine, [(cross, [1.0, 0.0]), (here, [0.95, 0.312])])

    neutral = _recall(recall_engine, monkeypatch, include="summaries", scope_bias=0.0, limit=5)
    biased = _recall(recall_engine, monkeypatch, include="summaries", scope_bias=1.0, limit=5)

    assert [h["node_id"] for h in neutral["hits"][:2]] == [cross, here]
    # A full scope bias lifts the current-conversation hit above the cross one,
    # yet the cross hit is still returned (boost, not filter).
    assert biased["hits"][0]["node_id"] == here
    assert cross in {h["node_id"] for h in biased["hits"]}


def test_recency_boost_moves_ranking(recall_engine, monkeypatch):
    old_strong = _add_summary(recall_engine, "old kanban board", session_id="session-a", created_at=1.0, latest_at=1.0)
    new_weak = _add_summary(recall_engine, "new kanban board", session_id="session-b", created_at=1.0, latest_at=time.time())
    # old_strong scores higher on cosine (rank 1) but is ancient; new_weak is rank 2 but fresh.
    _seed_summary_vectors(recall_engine, [(old_strong, [1.0, 0.0]), (new_weak, [0.95, 0.312])])

    payload = _recall(recall_engine, monkeypatch, include="summaries", scope_bias=0.0, limit=5)

    assert payload["hits"][0]["node_id"] == new_weak
    assert old_strong in {h["node_id"] for h in payload["hits"]}


def test_chunk_hit_dedupes_against_fts_by_store_id(recall_engine, monkeypatch):
    store_id = recall_engine._store.append(
        CURRENT, {"role": "user", "content": "kanban dashboard sprint verbatim detail"}
    )
    _seed_chunk_vectors(recall_engine, [(store_id, 0, 0, 39, [1.0, 0.0])])

    payload = _recall(recall_engine, monkeypatch, include="verbatim", limit=10)

    excerpt_hits = [h for h in payload["hits"] if h.get("store_id") == store_id]
    assert len(excerpt_hits) == 1
    hit = excerpt_hits[0]
    # The same message surfaced via both arms fuses into one entry carrying the
    # chunk span, and the expand handle points at the exact offset.
    assert set(hit["arms"]) == {"fts", "chunk"}
    assert hit["chunk_span"]["char_start"] == 0
    assert "content_offset=0" in hit["expand_hint"]


def test_summary_and_chunk_for_same_session_coexist_no_swamp(recall_engine, monkeypatch):
    """C6 pathology assessment: the harness turn-level collapse (a summary marker
    swamping precise chunk keys under a fixed top-k coverage budget) does NOT exist
    in lcm_recall's rrf_fuse.

    A summary hit keys as ("node", node_id) and a chunk/message hit as
    ("message", store_id), so a summary and the chunks of its own session are
    DISTINCT fused entries that coexist in the heterogeneous result — one never
    suppresses the other, and lcm_recall has no per-turn coverage budget to dilute.
    Both granularities surface for the same session, both scoring perfectly.
    """
    node = _add_summary(recall_engine, "kanban dashboard sprint overview", session_id="session-a", created_at=5.0)
    _seed_summary_vectors(recall_engine, [(node, [1.0, 0.0])])
    store_id = recall_engine._store.append(
        "session-a", {"role": "user", "content": "kanban dashboard sprint precise verbatim detail"}
    )
    _seed_chunk_vectors(recall_engine, [(store_id, 0, 0, 45, [1.0, 0.0])])

    payload = _recall(recall_engine, monkeypatch, include="all", scope_bias=0.0, limit=10)

    summary_hits = [h for h in payload["hits"] if h["kind"] == "summary"]
    excerpt_hits = [h for h in payload["hits"] if h["kind"] == "message_excerpt"]
    # Both granularities survive fusion as separate entries (no swamp/suppression).
    assert any(h["node_id"] == node for h in summary_hits)
    assert any(h["store_id"] == store_id for h in excerpt_hits)
    # The precise chunk carries the chunk arm; the summary carries the summary arm.
    precise = next(h for h in excerpt_hits if h["store_id"] == store_id)
    assert "chunk" in precise["arms"]


def test_include_verbatim_excludes_summaries(recall_engine, monkeypatch):
    node = _add_summary(recall_engine, "kanban summary only", session_id="session-a", created_at=5.0)
    _seed_summary_vectors(recall_engine, [(node, [1.0, 0.0])])
    recall_engine._store.append(CURRENT, {"role": "user", "content": "kanban dashboard sprint raw"})

    payload = _recall(recall_engine, monkeypatch, include="verbatim", limit=10)

    assert "summary" not in payload["provenance"]["arms_run"]
    assert all(h["kind"] == "message_excerpt" for h in payload["hits"])


def test_embeddings_off_degrades_to_fts_arm(recall_engine, monkeypatch):
    recall_engine._config.embeddings_enabled = False
    recall_engine._store.append(CURRENT, {"role": "user", "content": "kanban dashboard sprint fallback"})

    payload = _recall(recall_engine, monkeypatch, include="all", limit=10)

    assert payload["degraded"] is True
    assert "disabled" in payload["degraded_reason"]
    assert payload["provenance"]["coverage"].get("summary") == "disabled"
    assert payload["hits"]
    assert all(h["kind"] == "message_excerpt" for h in payload["hits"])


def test_summaries_include_degrades_to_fts_when_embeddings_off(recall_engine, monkeypatch):
    """F4-degrade-to-fts: include='summaries' with embeddings disabled must still
    run the FTS arm (its only vector arm is dead) rather than returning nothing."""
    recall_engine._config.embeddings_enabled = False
    recall_engine._store.append(CURRENT, {"role": "user", "content": "kanban dashboard sprint summaries fallback"})

    payload = _recall(recall_engine, monkeypatch, include="summaries", limit=10)

    assert "fts" in payload["provenance"]["arms_run"]
    assert payload["hits"]
    assert all(h["kind"] == "message_excerpt" for h in payload["hits"])


def test_empty_vector_corpora_reports_coverage_none(recall_engine, monkeypatch):
    recall_engine._store.append(CURRENT, {"role": "user", "content": "kanban dashboard sprint only fts"})

    payload = _recall(recall_engine, monkeypatch, include="all", limit=10)

    # Vector corpora are empty (no summaries/chunks seeded) -> coverage none, but
    # the FTS arm still returns a hit, so the tool never bare-errors.
    assert payload["provenance"]["coverage"].get("summary") == "none"
    assert payload["degraded"] is True
    assert payload["hits"]


def test_rerank_disabled_by_default(recall_engine, monkeypatch):
    node = _add_summary(recall_engine, "kanban rerank off", session_id="session-a", created_at=5.0)
    _seed_summary_vectors(recall_engine, [(node, [1.0, 0.0])])

    payload = _recall(recall_engine, monkeypatch, include="summaries", limit=5)
    assert payload["provenance"]["rerank"] == "disabled"


def test_rerank_skips_silently_on_non_voyage_provider(recall_engine, monkeypatch):
    recall_engine._config.rerank_enabled = True
    node = _add_summary(recall_engine, "kanban rerank skip", session_id="session-a", created_at=5.0)
    _seed_summary_vectors(recall_engine, [(node, [1.0, 0.0])])

    payload = _recall(recall_engine, monkeypatch, include="summaries", limit=5)
    assert payload["provenance"]["rerank"].startswith("skipped")
    assert payload["hits"]  # order preserved from RRF, not dropped


def test_rerank_applies_and_reorders_with_voyage_provider(recall_engine, monkeypatch):
    recall_engine._config.rerank_enabled = True
    a = _add_summary(recall_engine, "kanban alpha", session_id="session-a", created_at=5.0)
    b = _add_summary(recall_engine, "kanban beta", session_id="session-b", created_at=5.0)
    # a is RRF rank 1, b rank 2 (seeded under the voyage identity the rerank
    # provider resolves KNN against).
    _seed_summary_vectors(recall_engine, [(a, [1.0, 0.0]), (b, [0.95, 0.312])], provider="voyage")

    class RerankProvider(MockProvider):
        provider_id = "voyage"

        def rerank(self, query, documents, *, top_k=None, timeout, model="rerank-2.5-lite"):
            # Flip relevance: the LAST document scores highest (index i -> score i),
            # returned in descending-relevance order as the real API does.
            return sorted(
                ((i, float(i)) for i in range(len(documents))), key=lambda item: -item[1]
            )

    payload = _recall(
        recall_engine, monkeypatch, provider=RerankProvider(), include="summaries", scope_bias=0.0, limit=5
    )
    assert payload["provenance"]["rerank"] == "applied"
    assert payload["hits"][0]["node_id"] == b


def test_rerank_failure_falls_back_to_rrf_order(recall_engine, monkeypatch):
    recall_engine._config.rerank_enabled = True
    a = _add_summary(recall_engine, "kanban gamma", session_id="session-a", created_at=5.0)
    b = _add_summary(recall_engine, "kanban delta", session_id="session-b", created_at=5.0)
    _seed_summary_vectors(recall_engine, [(a, [1.0, 0.0]), (b, [0.95, 0.312])], provider="voyage")

    class BrokenRerank(MockProvider):
        provider_id = "voyage"

        def rerank(self, *args, **kwargs):
            raise RuntimeError("rerank endpoint down")

    payload = _recall(
        recall_engine, monkeypatch, provider=BrokenRerank(), include="summaries", scope_bias=0.0, limit=5
    )
    assert payload["provenance"]["rerank"].startswith("skipped")
    assert payload["hits"][0]["node_id"] == a  # RRF order intact


def test_limit_is_capped_and_reported(recall_engine, monkeypatch):
    recall_engine._store.append(CURRENT, {"role": "user", "content": "kanban dashboard sprint cap"})
    payload = _recall(recall_engine, monkeypatch, include="verbatim", limit=1000)
    assert payload["limit"] == 25
    assert payload["limit_clamped_from"] == 1000


def test_missing_query_is_rejected(recall_engine):
    payload = json.loads(lcm_tools.lcm_recall({"query": "   "}, engine=recall_engine))
    assert "error" in payload


def test_answer_ready_is_opt_in_and_default_response_is_byte_compatible(
    recall_engine, monkeypatch
):
    summary = "kanban dashboard sprint " + "compact-default " * 240
    node = _add_summary(
        recall_engine,
        summary,
        session_id="session-a",
        created_at=10.0,
    )
    _seed_summary_vectors(recall_engine, [(node, [1.0, 0.0])])
    monkeypatch.setattr(lcm_tools, "resolve_provider", lambda _config: MockProvider())
    monkeypatch.setattr(lcm_tools.time, "time", lambda: 10.0)

    base_args = {
        "query": "kanban dashboard sprint",
        "include": "summaries",
        "limit": 1,
    }
    implicit_raw = lcm_tools.lcm_recall(base_args, engine=recall_engine)
    explicit_raw = lcm_tools.lcm_recall(
        {**base_args, "detail": "snippets"},
        engine=recall_engine,
    )
    payload = json.loads(implicit_raw)

    assert implicit_raw == explicit_raw
    assert "detail" not in payload
    assert len(payload["hits"][0]["snippet"]) == 300
    assert "content" not in payload["hits"][0]
    assert "answer_ready" not in payload["provenance"]


def test_answer_ready_delta_is_opt_in_and_returns_only_novel_exact_refs(
    recall_engine, monkeypatch
):
    first = recall_engine._store.append(
        "session-a", {"role": "user", "content": "kanban dashboard sprint alpha"}
    )
    second = recall_engine._store.append(
        "session-b", {"role": "user", "content": "kanban dashboard sprint beta"}
    )
    monkeypatch.setattr(lcm_tools, "resolve_provider", lambda _config: MockProvider())
    monkeypatch.setattr(lcm_tools, "_lcm_recall_summary_arm", lambda *_a, **_k: ([], "none", 0, 0, []))
    monkeypatch.setattr(lcm_tools, "_lcm_recall_chunk_arm", lambda *_a, **_k: ([], "none", 0, 0))

    primary = _recall(
        recall_engine,
        monkeypatch,
        include="verbatim",
        detail="answer_ready",
        limit=2,
        seen_refs=[],
    )
    refs = [hit["exact_ref"] for hit in primary["hits"]]
    assert {hit["store_id"] for hit in primary["hits"]} == {first, second}
    assert len(refs) == len(set(refs)) == 2

    delta = _recall(
        recall_engine,
        monkeypatch,
        include="verbatim",
        detail="answer_ready",
        limit=2,
        seen_refs=[refs[0]],
    )
    assert [hit["exact_ref"] for hit in delta["hits"]] == [refs[1]]
    assert delta["delta"]["novel_refs"] == [refs[1]]
    assert delta["delta"]["progress"] is True

    exhausted = _recall(
        recall_engine,
        monkeypatch,
        include="verbatim",
        detail="answer_ready",
        limit=2,
        seen_refs=refs,
    )
    assert exhausted["hits"] == []
    assert exhausted["delta"]["termination_reason"] == "no_novel_exact_ref"


def test_answer_ready_delta_refs_match_hits_after_response_cap_eviction(
    recall_engine, monkeypatch
):
    contents = [
        f"kanban dashboard sprint evidence-{index} " + (str(index) * 2_300)
        for index in range(3)
    ]
    store_ids = [
        recall_engine._store.append(
            f"session-{index}", {"role": "user", "content": content}
        )
        for index, content in enumerate(contents)
    ]
    all_refs = {
        f"lcm:{store_id}:0-{len(content)}"
        for store_id, content in zip(store_ids, contents)
    }
    monkeypatch.setattr(lcm_tools, "_LCM_RECALL_RESPONSE_CHAR_CAP", 6_000)
    monkeypatch.setattr(
        lcm_tools,
        "_lcm_recall_summary_arm",
        lambda *_a, **_k: ([], "none", 0, 0, []),
    )
    monkeypatch.setattr(
        lcm_tools,
        "_lcm_recall_chunk_arm",
        lambda *_a, **_k: ([], "none", 0, 0),
    )

    payload = _recall(
        recall_engine,
        monkeypatch,
        include="verbatim",
        detail="answer_ready",
        limit=3,
        seen_refs=[],
    )

    delivered_refs = [hit["exact_ref"] for hit in payload["hits"]]
    assert payload["provenance"]["answer_ready"]["response_truncated"] is True
    assert payload["delta"]["novel_refs"] == delivered_refs
    assert payload["delta"]["novel_ref_count"] == len(delivered_refs)
    assert all_refs - set(delivered_refs)
    assert not (all_refs - set(delivered_refs)) & set(payload["delta"]["novel_refs"])


def test_answer_ready_response_cap_evicts_summary_leads_without_delta_refs(
    recall_engine, monkeypatch
):
    response_cap = 6_000
    summary_leads = [
        {
            "node_id": index,
            "session_id": f"session-{index}-" + ("s" * 2_000),
            "expand_hint": "x" * 2_000,
        }
        for index in range(3)
    ]
    monkeypatch.setattr(lcm_tools, "_LCM_RECALL_RESPONSE_CHAR_CAP", response_cap)
    monkeypatch.setattr(
        lcm_tools,
        "_lcm_recall_summary_arm",
        lambda *_a, **_k: (
            [],
            "full",
            len(summary_leads),
            len(summary_leads),
            list(summary_leads),
        ),
    )
    monkeypatch.setattr(
        lcm_tools,
        "_lcm_recall_chunk_arm",
        lambda *_a, **_k: ([], "none", 0, 0),
    )

    payload = _recall(
        recall_engine,
        monkeypatch,
        include="summaries",
        detail="answer_ready",
        limit=3,
        seen_refs=[],
    )

    expansion = payload["provenance"]["answer_ready"]
    assert payload["hits"] == []
    assert len(expansion["summary_leads"]) < len(summary_leads)
    assert len(json.dumps(payload, ensure_ascii=False)) <= response_cap
    assert expansion["response_truncated"] is True
    assert payload["delta"]["novel_refs"] == []
    assert payload["delta"]["novel_ref_count"] == 0


def test_answer_ready_baseline_bytes_ignore_disabled_occurrence_extension(
    recall_engine, monkeypatch
):
    recall_engine._store.append(
        "session-a", {"role": "user", "content": "kanban dashboard sprint alpha"}
    )
    monkeypatch.setattr(lcm_tools, "resolve_provider", lambda _config: MockProvider())
    monkeypatch.setattr(lcm_tools.time, "time", lambda: 10.0)
    args = {
        "query": "kanban dashboard sprint",
        "include": "verbatim",
        "detail": "answer_ready",
        "limit": 1,
    }
    baseline = lcm_tools.lcm_recall(args, engine=recall_engine)
    explicitly_disabled = lcm_tools.lcm_recall(
        {**args, "include_occurrence_time": False}, engine=recall_engine
    )
    assert baseline == explicitly_disabled
    assert "occurrence_time" not in baseline
    assert "exact-ref-delta-v1" not in baseline


def test_occurrence_time_is_opt_in_and_uses_source_session_date(
    recall_engine, monkeypatch
):
    recall_engine._store.append(
        "session-a",
        {
            "role": "user",
            "content": "I finished the kanban dashboard sprint 5 days ago.",
        },
    )
    recall_engine._session_occurrence_dates = {"session-a": "2023-03-20"}
    payload = _recall(
        recall_engine,
        monkeypatch,
        include="verbatim",
        detail="answer_ready",
        limit=1,
        seen_refs=[],
        include_occurrence_time=True,
    )
    occurrence = payload["hits"][0]["occurrence_time"]
    assert occurrence["event_date"] == "2023-03-15"
    assert occurrence["event_time_source"] == "relative_to_session"
    assert occurrence["observed_at"] != occurrence["event_at"]


def test_occurrence_time_uses_host_observation_without_benchmark_sidecar(
    recall_engine, monkeypatch
):
    observed_at = datetime(2024, 3, 20, tzinfo=timezone.utc).timestamp()
    recall_engine._store.append(
        "session-a",
        {
            "role": "user",
            "content": "I finished the kanban dashboard sprint 5 days ago.",
            "timestamp": observed_at,
        },
    )
    payload = _recall(
        recall_engine,
        monkeypatch,
        include="verbatim",
        detail="answer_ready",
        limit=1,
        seen_refs=[],
        include_occurrence_time=True,
    )
    hit = payload["hits"][0]
    assert hit["occurrence_time"]["event_date"] == "2024-03-15"
    assert hit["observation_time"]["observed_at"] == observed_at
    assert hit["observation_time"]["source"] == "host_message_timestamp"


def test_occurrence_time_legacy_row_uses_ingest_fallback_without_relative_event(
    recall_engine, monkeypatch
):
    recall_engine._store.append(
        "session-a",
        {
            "role": "user",
            "content": "I finished the kanban dashboard sprint 5 days ago.",
        },
    )
    payload = _recall(
        recall_engine,
        monkeypatch,
        include="verbatim",
        detail="answer_ready",
        limit=1,
        seen_refs=[],
        include_occurrence_time=True,
    )
    hit = payload["hits"][0]
    assert hit["occurrence_time"]["event_time_source"] == "unknown"
    assert hit["occurrence_time"]["event_date"] is None
    assert hit["observation_time"]["observed_at"] is None
    assert hit["observation_time"]["source"] == "ingest_fallback"


def test_invalid_recall_detail_is_rejected(recall_engine):
    payload = json.loads(
        lcm_tools.lcm_recall(
            {"query": "kanban", "detail": "full-transcript"},
            engine=recall_engine,
        )
    )
    assert payload["error"] == "detail must be one of: snippets, answer_ready"


def test_recall_schema_exposes_answer_ready_as_opt_in():
    from hermes_lcm.schemas import LCM_RECALL

    detail = LCM_RECALL["parameters"]["properties"]["detail"]
    assert detail["enum"] == ["snippets", "answer_ready"]
    assert detail["default"] == "snippets"


def test_recall_reports_query_embedding_provider_and_usage(recall_engine, monkeypatch):
    provider = MockProvider()
    payload = _recall(
        recall_engine,
        monkeypatch,
        provider=provider,
        include="summaries",
    )

    assert payload["metrics"] == {
        "embedding_query_calls": 1,
        "embedding_query_tokens": 7,
        "embedding_query_tokens_complete": True,
        "embedding_queries": [
            {"provider": "mock", "model": "mock-model", "usage_tokens": 7}
        ],
    }


def test_answer_ready_applies_stable_post_rank_session_diversity(
    recall_engine, monkeypatch
):
    _non_strict(recall_engine)
    node_ids = []
    for index in range(7):
        node_ids.append(
            _add_summary(
                recall_engine,
                f"kanban same-session evidence {index}",
                session_id="session-a",
                created_at=10.0,
            )
        )
    for index in range(3):
        node_ids.append(
            _add_summary(
                recall_engine,
                f"kanban diverse-session evidence {index}",
                session_id=f"session-{index + 1}",
                created_at=10.0,
            )
        )
    _patch_summary_arm(
        monkeypatch,
        [_summary_hit(recall_engine, node_id) for node_id in node_ids],
    )

    payload = _recall(
        recall_engine,
        monkeypatch,
        include="summaries",
        detail="answer_ready",
        scope_bias=0.0,
        limit=8,
    )

    assert [hit["node_id"] for hit in payload["hits"]] == node_ids[:5] + node_ids[7:10]
    assert [hit["session_id"] for hit in payload["hits"]].count("session-a") == 5
    policy = payload["provenance"]["answer_ready"]
    assert policy["per_session_limit"] == 5
    assert policy["diversity_dropped_count"] == 2


def test_answer_ready_keeps_missing_session_refs_independently_eligible():
    entries = [
        {
            "hit": {"kind": "message_excerpt", "store_id": index, "session_id": None}
        }
        for index in range(7)
    ]

    selected, dropped = lcm_tools._lcm_recall_diverse_entries(
        entries,
        limit=7,
        per_session_limit=5,
    )

    assert [entry["hit"]["store_id"] for entry in selected] == list(range(7))
    assert dropped == 0


def test_answer_ready_centers_message_content_on_exact_chunk_span(
    recall_engine, monkeypatch
):
    match = "kanban dashboard sprint"
    content = "a" * 2_500 + match + "z" * 2_500
    store_id = recall_engine._store.append(
        "session-a",
        {"role": "user", "content": content},
        source="chat",
    )
    match_start = content.index(match)
    match_end = match_start + len(match)
    _seed_chunk_vectors(
        recall_engine,
        [(store_id, 0, match_start, match_end, [1.0, 0.0])],
    )

    payload = _recall(
        recall_engine,
        monkeypatch,
        include="verbatim",
        detail="answer_ready",
        scope_bias=0.0,
        limit=1,
    )

    hit = payload["hits"][0]
    expected_offset = (match_start + match_end) // 2 - 1_200
    assert hit["store_id"] == store_id
    assert hit["content_offset"] == expected_offset
    assert len(hit["content"]) == 2_400
    assert match in hit["content"]
    assert hit["content_chars"] == len(content)
    assert hit["content_truncated"] is True
    assert hit["content_source"] == "message"
    assert hit["role"] == "user"
    assert hit["source"] == "chat"
    assert hit["evidence_span"] == {
        "char_start": match_start,
        "char_end": match_end,
    }


def test_answer_ready_expands_summary_ref_with_2400_char_bound(
    recall_engine, monkeypatch
):
    _non_strict(recall_engine)
    summary = "kanban dashboard sprint " + "summary-evidence " * 240
    node = _add_summary(
        recall_engine,
        summary,
        session_id="session-a",
        created_at=10.0,
    )
    _seed_summary_vectors(recall_engine, [(node, [1.0, 0.0])])

    payload = _recall(
        recall_engine,
        monkeypatch,
        include="summaries",
        detail="answer_ready",
        limit=1,
    )

    hit = payload["hits"][0]
    assert hit["node_id"] == node
    assert hit["snippet"] == summary[:300]
    assert hit["content"] == summary[:2_400]
    assert hit["content_returned_chars"] == 2_400
    assert hit["content_truncated"] is True
    assert hit["content_source"] == "summary"
    assert hit["source"] == "summary"


def test_answer_ready_expands_only_first_eight_and_reports_policy(
    recall_engine, monkeypatch
):
    _non_strict(recall_engine)
    node_ids = [
        _add_summary(
            recall_engine,
            "kanban dashboard sprint " + (f"evidence-{index} " * 300),
            session_id=f"session-{index}",
            created_at=10.0,
        )
        for index in range(9)
    ]
    _patch_summary_arm(
        monkeypatch,
        [_summary_hit(recall_engine, node_id) for node_id in node_ids],
    )
    monkeypatch.setattr(lcm_tools, "resolve_provider", lambda _config: MockProvider())

    raw = lcm_tools.lcm_recall(
        {
            "query": "kanban dashboard sprint",
            "include": "summaries",
            "detail": "answer_ready",
            "scope_bias": 0.0,
            "limit": 9,
        },
        engine=recall_engine,
    )
    payload = json.loads(raw)

    assert len(raw) <= 64_000
    assert len(payload["hits"]) == 9
    assert all(len(hit["content"]) <= 2_400 for hit in payload["hits"][:8])
    assert "content" not in payload["hits"][8]
    policy = payload["provenance"]["answer_ready"]
    assert policy["expanded_hit_count"] == 8
    assert policy["expanded_hit_limit"] == 8
    assert policy["per_hit_char_cap"] == 2_400
    assert policy["snippet_char_cap"] == 300
    assert policy["response_char_cap"] == 64_000
    assert policy["response_truncated"] is False
    assert "whole hits only" in policy["response_policy"]
    assert "no additional retrieval search" in policy["hydration_policy"]


def test_answer_ready_enforces_complete_response_cap_and_marks_query_truncation(
    recall_engine, monkeypatch
):
    _non_strict(recall_engine)
    node = _add_summary(
        recall_engine,
        "bounded summary evidence",
        session_id="session-a",
        created_at=10.0,
    )
    _patch_summary_arm(monkeypatch, [_summary_hit(recall_engine, node)])
    monkeypatch.setattr(lcm_tools, "resolve_provider", lambda _config: MockProvider())

    raw = lcm_tools.lcm_recall(
        {
            "query": "q" * 70_000,
            "include": "summaries",
            "detail": "answer_ready",
            "limit": 1,
        },
        engine=recall_engine,
    )
    payload = json.loads(raw)

    assert len(raw) <= 64_000
    assert len(payload["query"]) == 4_096
    assert len(payload["hits"]) == 1
    assert payload["provenance"]["answer_ready"]["query_truncated"] is True


def test_answer_ready_hydration_uses_exact_reads_without_an_extra_search(
    recall_engine, monkeypatch
):
    match = "kanban dashboard sprint"
    content = "prefix " * 400 + match + " suffix" * 400
    store_id = recall_engine._store.append(
        "session-a",
        {"role": "user", "content": content},
    )
    start = content.index(match)
    _seed_chunk_vectors(
        recall_engine,
        [(store_id, 0, start, start + len(match), [1.0, 0.0])],
    )
    calls = {"search": 0, "get_batch": 0}
    real_search = MessageStore.search
    real_get_batch = MessageStore.get_batch

    def counted_search(self, *args, **kwargs):
        calls["search"] += 1
        return real_search(self, *args, **kwargs)

    def counted_get_batch(self, *args, **kwargs):
        calls["get_batch"] += 1
        return real_get_batch(self, *args, **kwargs)

    monkeypatch.setattr(MessageStore, "search", counted_search)
    monkeypatch.setattr(MessageStore, "get_batch", counted_get_batch)

    payload = _recall(
        recall_engine,
        monkeypatch,
        include="verbatim",
        detail="answer_ready",
        scope_bias=0.0,
        limit=1,
    )

    assert payload["hits"][0]["store_id"] == store_id
    assert calls == {"search": 1, "get_batch": 1}


def test_recall_scans_full_corpus_not_grep_recency_window(recall_engine, monkeypatch):
    """Recall must NOT inherit grep's 2000-recent bound, or 'all time' truncates."""
    recall_engine._config.recall_scan_rows = 25_000
    recall_engine._config.embedding_bounded_scan_rows = 2_000
    observed: list[int] = []

    from hermes_lcm.vector_store import KNNResult

    class BoundCapturingStore:
        def __init__(self, *_args, bounded_scan_rows=None, **_kwargs):
            observed.append(bounded_scan_rows)

        def knn(self, *_args, **_kwargs):
            return KNNResult(coverage="none")

        def knn_chunks(self, *_args, **_kwargs):
            return KNNResult(coverage="none")

        def close(self):
            pass

    monkeypatch.setattr(lcm_tools, "VectorStore", BoundCapturingStore)
    monkeypatch.setattr(lcm_tools, "resolve_provider", lambda _config: MockProvider())

    json.loads(lcm_tools.lcm_recall({"query": "anything", "include": "all"}, engine=recall_engine))

    # Both vector arms request the large recall bound, never the small grep one.
    assert observed and all(bound == 25_000 for bound in observed)


def test_chunk_hydrate_is_batched_not_n_plus_1(recall_engine, monkeypatch):
    """F4-chunk-hydrate-n-plus-1: hydrate_chunk_hits issues ONE batched JOIN over
    all ranked chunk ids, not a SELECT per hit, and preserves rank order."""
    import sqlite3 as _sqlite
    import hermes_lcm.retrieval_core as rc
    from hermes_lcm.retrieval_core import hydrate_chunk_hits

    contents = {}
    for i in range(5):
        sid = recall_engine._store.append(CURRENT, {"role": "user", "content": f"chunk excerpt number {i} body"})
        contents[sid] = i
    ranked = [(f"{sid}:0", 1.0 - 0.01 * n, "chunk") for n, sid in enumerate(contents)]
    # Seed the chunk meta rows the JOIN reads.
    _seed_chunk_vectors(recall_engine, [(sid, 0, 0, 15, [1.0, 0.0]) for sid in contents])

    select_count = {"n": 0}
    real_connect = _sqlite.connect

    class CountingConnection(_sqlite.Connection):
        def execute(self, sql, *args, **kw):
            if "lcm_chunk_meta" in sql:
                select_count["n"] += 1
            return super().execute(sql, *args, **kw)

    def counting_connect(*a, **k):
        k["factory"] = CountingConnection
        return real_connect(*a, **k)

    monkeypatch.setattr(rc.sqlite3, "connect", counting_connect)
    deadline = __import__("time").monotonic() + 30.0
    hits = hydrate_chunk_hits(recall_engine, ranked_rows=ranked, knn_limit=50, deadline=deadline, snippet_chars=200)

    assert len(hits) == 5
    assert select_count["n"] == 1  # single batched JOIN, not 5
    # Rank order preserved (highest score first).
    assert [h["store_id"] for h, _ in hits] == list(contents)


def test_recall_query_timeout_has_its_own_budget(monkeypatch, tmp_path):
    """sprint-opt-2: lcm_recall uses recall_query_timeout_s (default 8.0), env
    LCM_RECALL_QUERY_TIMEOUT_S, distinct from lcm_grep's 3.0s query deadline."""
    assert LCMConfig(database_path=str(tmp_path / "d.db")).recall_query_timeout_s == 8.0
    monkeypatch.setenv("LCM_RECALL_QUERY_TIMEOUT_S", "12.5")
    monkeypatch.setenv("LCM_EMBEDDING_QUERY_TIMEOUT_S", "3.0")
    cfg = LCMConfig.from_env()
    assert cfg.recall_query_timeout_s == 12.5
    assert cfg.embedding_query_timeout_s == 3.0  # grep's deadline untouched


def test_summary_source_expansion_refuses_an_expired_deadline(recall_engine):
    node = SimpleNamespace(
        node_id=1,
        session_id="session-a",
    )

    with pytest.raises(TimeoutError, match="summary source expansion"):
        lcm_tools._lcm_recall_summary_source_hits(
            recall_engine,
            [(node, 1.0)],
            current=CURRENT,
            candidate_limit=5,
            lead_limit=1,
            deadline=-1.0,
        )


def test_recall_arm_weights_default_and_env_lenient(monkeypatch, tmp_path):
    """B2: recall_arm_weights default to fts=0.5,summary=1,chunk=1 and the env
    override parses leniently -- unknown arms, malformed pairs, and non-numeric
    weights are dropped while unspecified arms keep their default."""
    assert LCMConfig(database_path=str(tmp_path / "d.db")).recall_arm_weights == {
        "fts": 0.5,
        "summary": 1.0,
        "chunk": 1.0,
    }
    monkeypatch.setenv("LCM_RECALL_ARM_WEIGHTS", "fts=0.7, chunk=0.9 ,bogus=1,summary=x,,junk")
    cfg = LCMConfig.from_env()
    assert cfg.recall_arm_weights == {"fts": 0.7, "summary": 1.0, "chunk": 0.9}


def test_recall_echoes_arm_weights_in_provenance(recall_engine, monkeypatch):
    """B2: the weights actually applied to the arms that ran are echoed back
    under provenance.arm_weights."""
    recall_engine._config.recall_arm_weights = {"fts": 0.5, "summary": 1.0, "chunk": 1.0}
    node = _add_summary(recall_engine, "kanban board dashboard sprint plan", session_id="session-a", created_at=10.0)
    _seed_summary_vectors(recall_engine, [(node, [1.0, 0.0])])

    payload = _recall(recall_engine, monkeypatch, include="summaries", limit=5)

    assert payload["provenance"]["arms_run"] == ["summary"]
    assert payload["provenance"]["arm_weights"] == {"summary": 1.0}


def test_recall_uses_recall_timeout_budget(recall_engine, monkeypatch):
    """lcm_recall builds its deadline from recall_query_timeout_s, not the grep one."""
    recall_engine._config.recall_query_timeout_s = 8.0
    recall_engine._config.embedding_query_timeout_s = 0.001  # would insta-timeout if used
    recall_engine._store.append(CURRENT, {"role": "user", "content": "kanban dashboard sprint budget"})

    payload = _recall(recall_engine, monkeypatch, include="verbatim", limit=5)
    assert payload.get("timeout") is not True
    assert payload["hits"]


def test_bounded_chunk_coverage_surfaces_as_degraded(recall_engine, monkeypatch):
    """SCAN-1: a chunk arm capped by recall_scan_max_rows reports a
    degraded_reasons entry naming the arm + scanned/total, instead of silently
    truncating."""
    recall_engine._config.recall_scan_max_rows = 1
    ids = []
    for i in range(3):
        sid = recall_engine._store.append(
            CURRENT, {"role": "user", "content": f"kanban dashboard sprint chunk {i}"}
        )
        ids.append(sid)
    _seed_chunk_vectors(
        recall_engine,
        [(sid, 0, 0, 20, [1.0, 0.0]) for sid in ids],
    )

    payload = _recall(recall_engine, monkeypatch, include="verbatim", limit=10)

    assert payload["provenance"]["coverage"].get("chunk") == "bounded"
    assert payload["degraded"] is True
    assert "chunk arm coverage bounded" in payload["degraded_reason"]
    assert "of 3 vectors" in payload["degraded_reason"]


def test_recall_scan_batches_the_whole_corpus_and_reaches_the_oldest(
    recall_engine, monkeypatch
):
    """F31 §2: recall_scan_rows is a BATCH SIZE, not a recency window.

    The gold vector is the OLDEST message in a corpus more than 2x the batch
    size — exactly the shape that collapsed all-gold recall to 0.000 when the
    scan only scored the most-recent rows.
    """
    recall_engine._config.recall_scan_rows = 2  # batch size; corpus is 7 vectors
    seeded = []
    # The gold message is appended FIRST, so every later filler is more recent:
    # under the old recency window it aged straight out of the scan.
    gold = recall_engine._store.append(
        CURRENT, {"role": "user", "content": "kanban dashboard sprint gold"}
    )
    seeded.append((gold, [1.0, 0.0]))
    for i in range(6):
        sid = recall_engine._store.append(
            CURRENT, {"role": "user", "content": f"unrelated filler note {i}"}
        )
        seeded.append((sid, [0.0, 1.0]))
    _seed_chunk_vectors(
        recall_engine,
        [(sid, 0, 0, 20, vector) for sid, vector in seeded],
    )

    payload = _recall(recall_engine, monkeypatch, include="verbatim", limit=1)

    assert payload["provenance"]["coverage"].get("chunk") == "full"
    assert [hit["store_id"] for hit in payload["hits"]] == [gold]


def test_recall_scan_reports_no_degraded_reason_by_default(recall_engine, monkeypatch):
    """Default config has no cap and no latency budget, so nothing truncates."""
    assert recall_engine._config.recall_scan_max_rows == 0
    assert recall_engine._config.recall_scan_budget_s == 0.0
    recall_engine._config.recall_scan_rows = 1  # one vector per batch
    ids = [
        recall_engine._store.append(
            CURRENT, {"role": "user", "content": f"kanban dashboard sprint chunk {i}"}
        )
        for i in range(5)
    ]
    _seed_chunk_vectors(
        recall_engine,
        [(sid, 0, 0, 20, [1.0, 0.0]) for sid in ids],
    )

    payload = _recall(recall_engine, monkeypatch, include="verbatim", limit=10)

    assert payload["provenance"]["coverage"].get("chunk") == "full"
    assert "degraded_reason" not in payload
    assert {hit["store_id"] for hit in payload["hits"]} == set(ids)


def test_two_stage_full_approx_coverage_surfaces_as_approximate(recall_engine, monkeypatch):
    """FIX 2: a two-stage (binary prescreen) summary arm reaches the whole corpus
    but ranks approximately, so it reports coverage='full_approx' and discloses
    the approximate prescreen in degraded_reason (like 'bounded' is disclosed),
    rather than passing as an exact 'full'."""
    recall_engine._config.embedding_binary_prescreen = True
    node = _add_summary(
        recall_engine, "kanban board dashboard sprint plan",
        session_id="session-a", created_at=10.0,
    )
    _seed_summary_vectors(recall_engine, [(node, [1.0, 0.0])])

    payload = _recall(recall_engine, monkeypatch, include="summaries", limit=5)

    assert payload["provenance"]["coverage"].get("summary") == "full_approx"
    assert payload["degraded"] is True
    assert "summary arm coverage full_approx" in payload["degraded_reason"]
    assert "approximate" in payload["degraded_reason"]
    # The corpus was still reached: the hit is returned, not dropped.
    assert node in {hit["node_id"] for hit in payload["hits"]}


def test_pooled_vector_store_survives_across_recall_calls(recall_engine, monkeypatch):
    """F2-matrix-cache-never-persists: back-to-back recalls reuse ONE pooled
    VectorStore whose matrix cache survives, instead of building+closing a fresh
    store (and clearing the cache) every call."""
    import hermes_lcm.retrieval_core as rc

    rc._reset_vector_store_pool()
    try:
        node = _add_summary(recall_engine, "kanban pooled cache", session_id="session-a", created_at=5.0)
        _seed_summary_vectors(recall_engine, [(node, [1.0, 0.0])])

        _recall(recall_engine, monkeypatch, include="summaries", limit=5)
        key = (str(recall_engine._store.db_path), 25_000)
        assert key in rc._vector_store_pool
        pooled = rc._vector_store_pool[key]["store"]
        # The pooled store's matrix cache is populated (survived the call).
        assert pooled._matrix_cache

        _recall(recall_engine, monkeypatch, include="summaries", limit=5)
        # Same instance reused, not rebuilt.
        assert rc._vector_store_pool[key]["store"] is pooled
    finally:
        rc._reset_vector_store_pool()


def test_plugin_unload_closes_pooled_vector_store(recall_engine, monkeypatch):
    import hermes_lcm.retrieval_core as rc

    rc._reset_vector_store_pool()
    try:
        node = _add_summary(
            recall_engine,
            "pooled unload cleanup",
            session_id="session-a",
            created_at=5.0,
        )
        _seed_summary_vectors(recall_engine, [(node, [1.0, 0.0])])
        _recall(recall_engine, monkeypatch, include="summaries", limit=5)
        key = (str(recall_engine._store.db_path), 25_000)
        pooled = rc._vector_store_pool[key]["store"]

        from hermes_lcm.config import LCMConfig
        from hermes_lcm.engine import LCMEngine

        lifecycle_engine = LCMEngine(
            config=LCMConfig(database_path=str(recall_engine._store.db_path))
        )
        lifecycle_engine.shutdown_all_instances()

        assert not rc._vector_store_pool
        assert pooled._conn is None
        from hermes_lcm.vector_store import VectorStore

        with pytest.raises(RuntimeError, match="closed during plugin unload"):
            rc._acquire_vector_store(
                recall_engine,
                vector_store_cls=VectorStore,
                scan_rows=25_000,
            )
    finally:
        rc._reset_vector_store_pool()


def test_plugin_unload_waits_for_active_pooled_query(recall_engine):
    import threading

    import hermes_lcm.retrieval_core as rc

    started = threading.Event()
    release = threading.Event()
    closed = threading.Event()
    errors = []

    class BlockingStore:
        _supports_pooling = True

        def __init__(self, *_args, **_kwargs):
            self._conn = None
            self.was_closed = False

        def close(self):
            self.was_closed = True

    def query(store):
        started.set()
        if not release.wait(2):
            raise TimeoutError("test query was not released")
        return store

    def run_query():
        try:
            rc._run_pooled_knn(
                recall_engine,
                vector_store_cls=BlockingStore,
                scan_rows=25_000,
                deadline=time.monotonic() + 5,
                query=query,
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    def close_pool():
        rc._close_vector_store_pool()
        closed.set()

    rc._reset_vector_store_pool()
    worker = threading.Thread(target=run_query)
    closer = threading.Thread(target=close_pool)
    try:
        worker.start()
        assert started.wait(2)
        closer.start()
        assert not closed.wait(0.1)
        release.set()
        worker.join(2)
        closer.join(2)
        assert not worker.is_alive()
        assert not closer.is_alive()
        assert not errors
        assert closed.is_set()
    finally:
        release.set()
        worker.join(2)
        closer.join(2)
        rc._reset_vector_store_pool()


def test_matrix_cache_is_bounded_lru_not_cleared_on_miss():
    """sprint-opt-6: distinct candidate sets coexist in a bounded LRU rather than
    each miss clearing the whole cache."""
    import numpy as np

    import tempfile
    from hermes_lcm.config import LCMConfig as _Cfg

    with tempfile.TemporaryDirectory() as d:
        vs = VectorStore(f"{d}/m.db", config=_Cfg(database_path=f"{d}/m.db", embeddings_enabled=True))
        try:
            vs.register_profile("mock-model", "mock", 2)
            identity = vs.capture_identity("mock-model", provider="mock")
            # Load several distinct candidate sets; all must remain cached (bounded).
            for i in range(3):
                vs._numpy_rows(np, identity.identity_hash, 2, [str(i)])
            assert len(vs._matrix_cache) == 3  # no clear-on-miss; all coexist
            # A fourth distinct set past the cap evicts the oldest, never all.
            for i in range(3, vs._MATRIX_CACHE_MAX_ENTRIES + 2):
                vs._numpy_rows(np, identity.identity_hash, 2, [str(i)])
            assert len(vs._matrix_cache) == vs._MATRIX_CACHE_MAX_ENTRIES
        finally:
            vs.close()


def test_json_doctor_surfaces_background_integrity_flag(recall_engine):
    """F1-json-doctor-background-flag-untested: the JSON lcm_doctor MCP tool (not
    just the text path) surfaces a pre-recorded background FTS-corruption flag."""
    from hermes_lcm.db_bootstrap import _record_integrity_failed
    from hermes_lcm.store import build_message_fts_spec

    # lcm_doctor reaches beyond the recall fixture's attribute set; supply the
    # few unguarded ones it touches (context-pressure short-circuits at 0).
    recall_engine.context_length = 0
    recall_engine.last_prompt_tokens = 0
    recall_engine.get_runtime_identity = lambda: {}

    conn = recall_engine._store.connection
    spec = build_message_fts_spec()
    _record_integrity_failed(conn, spec, detail="messages_fts malformed (background scan)")
    conn.commit()

    payload = json.loads(lcm_tools.lcm_doctor({}, engine=recall_engine))
    checks = {c["check"]: c for c in payload["checks"]}

    flag_check = checks.get("messages_fts_integrity_background_flag")
    assert flag_check is not None
    assert flag_check["status"] == "fail"
    assert "background integrity scan flagged" in flag_check["detail"]["guidance"]


def test_rrf_fuse_collapses_repeated_identity_within_arm():
    """RRF-1: a message chunked into several pieces must contribute ONE term per
    arm at its best rank, not one per chunk occurrence."""
    from hermes_lcm.retrieval_core import rrf_fuse

    # Arm 0 (chunk): message A appears 3x (ranks 2,3,4); message B once (rank 1).
    chunk_arm = [
        {"store_id": "B"},
        {"store_id": "A"},
        {"store_id": "A"},
        {"store_id": "A"},
    ]
    fused = rrf_fuse([chunk_arm], k=60)
    by_id = {entry["hit"]["store_id"]: entry for entry in fused}
    # A is collapsed to its best (first) rank 2 and counted once; B's genuine
    # rank-1 hit therefore out-scores it instead of losing to a 3x double-count.
    assert by_id["A"]["ranks"] == {0: 2}
    assert by_id["B"]["ranks"] == {0: 1}
    assert by_id["B"]["rrf_score"] > by_id["A"]["rrf_score"]
    assert fused[0]["hit"]["store_id"] == "B"


def test_rrf_fuse_default_weights_are_byte_identical_to_unweighted():
    """B2: passing explicit 1.0 weights must reproduce unweighted RRF bit-for-bit
    so lcm_grep's hybrid (which keeps 1.0 weights) is unchanged."""
    from hermes_lcm.retrieval_core import rrf_fuse

    fts_arm = [{"store_id": "A"}, {"store_id": "B"}]
    vec_arm = [{"store_id": "B"}, {"store_id": "C"}]
    arms = [fts_arm, vec_arm]

    unweighted = rrf_fuse(arms, k=60)
    weighted_ones = rrf_fuse(arms, k=60, weights=[1.0, 1.0])
    # A missing/short weights list falls back to 1.0 for every unspecified arm.
    weighted_short = rrf_fuse(arms, k=60, weights=[])

    def _scores(fused):
        return [(e["hit"].get("store_id"), e["rrf_score"], tuple(sorted(e["ranks"].items()))) for e in fused]

    assert _scores(weighted_ones) == _scores(unweighted)
    assert _scores(weighted_short) == _scores(unweighted)


def test_rrf_weights_rank_vector_best_first_on_weak_fts_corpus():
    """B2: on a strong-vector/weak-FTS shape, naive equal-weight RRF ranks a
    noise identity (that the weak FTS arm loves) above the vector-best one; the
    (0.5, 1, 1) arm weights restore the vector-best identity to the top --
    mirroring the −21 R@5 LongMemEval regression. k is shrunk so short arms
    spread rank terms far enough to exercise the flip cleanly."""
    from hermes_lcm.retrieval_core import rrf_fuse

    # Arm order is fts(0), summary(1), chunk(2) -- as lcm_recall builds it.
    # Noise N: FTS rank 1 (weak arm loves it) but only rank 5 in each vector arm.
    # Vector-best V: rank 1 in both vector arms, absent from FTS.
    fts_arm = [{"store_id": "N"}, {"store_id": "x1"}, {"store_id": "x2"}, {"store_id": "x3"}, {"store_id": "x4"}]
    summary_arm = [{"store_id": "V"}, {"store_id": "y1"}, {"store_id": "y2"}, {"store_id": "y3"}, {"store_id": "N"}]
    chunk_arm = [{"store_id": "V"}, {"store_id": "z1"}, {"store_id": "z2"}, {"store_id": "z3"}, {"store_id": "N"}]
    arms = [fts_arm, summary_arm, chunk_arm]

    naive = rrf_fuse(arms, k=10)
    assert naive[0]["hit"]["store_id"] == "N"  # equal weights get it wrong

    weighted = rrf_fuse(arms, k=10, weights=[0.5, 1.0, 1.0])
    assert weighted[0]["hit"]["store_id"] == "V"  # down-weighting FTS fixes it


def test_parse_arm_weights_rejects_negative_keeps_default(monkeypatch, caplog):
    """FIX-1: a negative env weight is invalid (it would invert RRF
    rank-monotonicity) -- the arm keeps its default and a warning is logged."""
    import logging as _logging

    monkeypatch.setenv("LCM_RECALL_ARM_WEIGHTS", "fts=-0.5,summary=1.0,chunk=0")
    with caplog.at_level(_logging.WARNING, logger="hermes_lcm.config"):
        cfg = LCMConfig.from_env()
    # fts falls back to its 0.5 default (negative dropped); chunk=0 is legal.
    assert cfg.recall_arm_weights == {"fts": 0.5, "summary": 1.0, "chunk": 0.0}
    assert any("negative weight" in rec.getMessage() for rec in caplog.records)


def test_rrf_fuse_clamps_negative_weight_no_inversion():
    """FIX-1: a negative arm weight in rrf_fuse is clamped to 0.0 (the arm drops
    out) rather than making a rank-1 hit score negative and inverting order."""
    from hermes_lcm.retrieval_core import rrf_fuse

    arm0 = [{"store_id": "A"}, {"store_id": "B"}]
    arm1 = [{"store_id": "C"}]
    # arm0 negative -> contributes 0; arm1 (weight 1.0) alone decides ordering.
    fused = rrf_fuse([arm0, arm1], k=60, weights=[-3.0, 1.0])
    by_id = {e["hit"]["store_id"]: e for e in fused}
    assert by_id["A"]["rrf_score"] == 0.0  # negative arm contributes nothing
    assert by_id["B"]["rrf_score"] == 0.0
    assert by_id["C"]["rrf_score"] > 0.0
    assert fused[0]["hit"]["store_id"] == "C"  # no negative-score inversion


def test_chunk_dedupe_keeps_best_ranked_span(recall_engine, monkeypatch):
    """F1-chunk-dedupe-wrong-span: when one message has several chunks, the merged
    hit keeps the BEST-ranked chunk's span, not the worst (last) one."""
    content = "kanban dashboard sprint verbatim detail tail segment here"
    store_id = recall_engine._store.append(CURRENT, {"role": "user", "content": content})
    # Chunk 0 (char 0-24) is the strong cosine-1.0 match; chunk 1 (char 33-57) is
    # a weak near-orthogonal match that must NOT overwrite the strong span.
    _seed_chunk_vectors(
        recall_engine,
        [
            (store_id, 0, 0, 24, [1.0, 0.0]),
            (store_id, 1, 33, 57, [0.05, 0.998]),
        ],
    )

    payload = _recall(recall_engine, monkeypatch, include="verbatim", limit=10)

    excerpt_hits = [h for h in payload["hits"] if h.get("store_id") == store_id]
    assert len(excerpt_hits) == 1
    hit = excerpt_hits[0]
    assert hit["chunk_span"]["char_start"] == 0 and hit["chunk_span"]["char_end"] == 24
    assert "content_offset=0" in hit["expand_hint"]


def test_chunk_fts_merge_snippet_and_offset_are_consistent(recall_engine, monkeypatch):
    """DEDUPE-1: the merged hit's snippet and content_offset describe the SAME
    span (both from the better-ranked chunk arm), never an FTS snippet glued to a
    chunk offset."""
    content = "prologue text then kanban dashboard sprint match zone trailing"
    match_start = content.index("kanban")
    match_end = match_start + len("kanban dashboard sprint match")
    store_id = recall_engine._store.append(CURRENT, {"role": "user", "content": content})
    _seed_chunk_vectors(recall_engine, [(store_id, 0, match_start, match_end, [1.0, 0.0])])

    payload = _recall(recall_engine, monkeypatch, include="verbatim", limit=10)

    hit = next(h for h in payload["hits"] if h.get("store_id") == store_id)
    assert set(hit["arms"]) == {"fts", "chunk"}
    # Snippet and expand offset both come from the chunk arm -> consistent.
    assert hit["snippet"] == content[match_start:match_end]
    assert f"content_offset={match_start}" in hit["expand_hint"]
    assert hit["chunk_span"]["char_start"] == match_start


def test_rerank_does_not_splice_voyage_score_onto_rrf_scale(recall_engine, monkeypatch):
    """RERANK-1: rerank only permutes the window; the reported score stays on the
    RRF scale rather than being replaced by the ~0-1 voyage relevance score."""
    recall_engine._config.rerank_enabled = True
    a = _add_summary(recall_engine, "kanban alpha", session_id="session-a", created_at=5.0)
    b = _add_summary(recall_engine, "kanban beta", session_id="session-b", created_at=5.0)
    _seed_summary_vectors(recall_engine, [(a, [1.0, 0.0]), (b, [0.95, 0.312])], provider="voyage")

    class RerankProvider(MockProvider):
        provider_id = "voyage"

        def rerank(self, query, documents, *, top_k=None, timeout, model="rerank-2.5-lite"):
            # Voyage-shaped scores in the 0..1 range, descending.
            return sorted(
                ((i, 0.9 - 0.1 * i) for i in range(len(documents))), key=lambda item: -item[1]
            )

    payload = _recall(
        recall_engine, monkeypatch, provider=RerankProvider(), include="summaries", scope_bias=0.0, limit=5
    )
    assert payload["provenance"]["rerank"] == "applied"
    # Had the 0.9 voyage score been spliced onto the RRF scale it would dwarf the
    # ~0.016 RRF score; the reported score must stay RRF-scaled.
    assert all(hit["score"] < 0.1 for hit in payload["hits"])


# -- Reference-strict delivery (FINDING-F35) ----------------------------------
#
# The #168 sanitization fix woke the summary arm's internal FTS queries, so
# uncitable kind:"summary" hits started reaching consumers that validate every
# evidence reference and fail CLOSED on an unreferenced card. These cases pin
# the invariant: no hit lacking a validated (store_id, char_start, char_end)
# source span is delivered, an omitted hit is backfilled by the next-ranked
# citable one, and the omission count is surfaced rather than silent.


def _stub_row_text(store_id):
    return f"row-{store_id} verbatim body text"


class _StubStore:
    """Minimal store for the selection unit tests; counts batched reads."""

    def __init__(self, store_ids):
        self._rows = {
            store_id: {"content": _stub_row_text(store_id)} for store_id in store_ids
        }
        self.batch_calls = []

    def get_batch(self, store_ids):
        self.batch_calls.append(list(store_ids))
        return {sid: self._rows[sid] for sid in store_ids if sid in self._rows}

    def drop(self, store_id):
        """Delete a row, as supported cleanup does mid-request."""
        self._rows.pop(store_id, None)


def _stub_engine(store_ids):
    return SimpleNamespace(_store=_StubStore(store_ids))


def _message_entry(
    store_id, *, session_id="session-a", chunk_span=None, citable=True, snippet=None
):
    """A ranked candidate.

    ``citable=False`` omits the offset a hit needs to be cited without hydration
    (the shape a pure-FTS hit has). ``snippet`` overrides the excerpt so a
    candidate can be citable-SHAPED yet fail verification against its row -- the
    case that forces the walk to keep reading.
    """
    hit = {
        "kind": "message_excerpt",
        "store_id": store_id,
        "session_id": session_id,
    }
    if citable:
        hit["content_offset"] = 0
        hit["snippet"] = _stub_row_text(store_id) if snippet is None else snippet
    if chunk_span is not None:
        hit["chunk_span"] = chunk_span
    return {"hit": hit}


def _summary_entry(node_id, *, session_id="session-s", store_id=None):
    hit = {"kind": "summary", "node_id": node_id, "session_id": session_id}
    if store_id is not None:
        hit["store_id"] = store_id
    return {"hit": hit}


def test_reference_strict_backfills_past_every_uncitable_candidate_shape():
    """The three uncitable shapes lose their slot to the next citable hit.

    Candidates 1 and 4 are uncitable summaries (the second carries the #164a
    ``store_id``, which names lineage rather than a citation); candidate 6 is a
    message that reached a post-hydration slot with no chunk span of its own.
    All three are skipped and the selection continues down the ranking, so the
    delivered count still reaches ``limit``.
    """
    span = {"char_start": 0, "char_end": 40}
    ordered = [
        _summary_entry(901),
        _message_entry(1, chunk_span=span),
        _message_entry(2, chunk_span=span),
        _summary_entry(902, store_id=7),
        _message_entry(3, chunk_span=span),
        _message_entry(4, citable=False),  # no offset, and slot 3 is past expanded_limit=2
        _message_entry(5, chunk_span=span),
        _message_entry(6, chunk_span=span),
    ]

    selected, dropped, unreferenced = lcm_tools._lcm_recall_citable_entries(
        ordered,
        limit=5,
        per_session_limit=5,
        expanded_limit=2,
        engine=_stub_engine(range(1, 10)),
    )

    assert [entry["hit"]["store_id"] for entry in selected] == [1, 2, 3, 5, 6]
    assert len(selected) == 5
    assert unreferenced == 3
    assert dropped == 0


def test_reference_strict_admits_an_unspanned_message_inside_the_hydration_budget():
    """Slot position decides: a chunk-span-less message is citable while the
    hydration budget still covers it (it will carry content_offset), and only
    becomes uncitable once it falls past that budget."""
    ordered = [_message_entry(index, citable=False) for index in range(1, 5)]

    selected, _dropped, unreferenced = lcm_tools._lcm_recall_citable_entries(
        ordered,
        limit=4,
        per_session_limit=5,
        expanded_limit=2,
        engine=_stub_engine(range(1, 5)),
    )

    assert [entry["hit"]["store_id"] for entry in selected] == [1, 2]
    assert unreferenced == 2


def test_reference_strict_skips_uncitable_before_it_consumes_session_quota():
    """An undelivered hit must not spend the per-session density budget it was
    never going to occupy."""
    ordered = [_summary_entry(900 + index, session_id="session-a") for index in range(3)]
    ordered += [_message_entry(index, session_id="session-a") for index in range(1, 6)]

    selected, dropped, unreferenced = lcm_tools._lcm_recall_citable_entries(
        ordered,
        limit=5,
        per_session_limit=5,
        expanded_limit=8,
        engine=_stub_engine(range(1, 6)),
    )

    assert [entry["hit"]["store_id"] for entry in selected] == [1, 2, 3, 4, 5]
    assert unreferenced == 3
    assert dropped == 0


def _only_vector_arms(monkeypatch):
    """Silence the FTS arm so summary-arm and chunk-arm ranks interleave.

    Seeded messages match the query lexically too, and that extra arm lifts them
    clear above the summary hits — which would leave the uncitable candidates
    below the cut and never exercise the backfill.
    """
    monkeypatch.setattr(lcm_tools, "_lcm_recall_fts_arm", lambda *_a, **_k: ([], None))


def _seed_citable_messages(engine, count, *, sessions=("session-a", "session-b")):
    """Seed messages that the chunk arm can retrieve with a verbatim span."""
    match = "kanban dashboard sprint"
    store_ids = []
    for index in range(count):
        content = f"{match} evidence body {index} " + "filler " * 20
        store_id = engine._store.append(
            sessions[index % len(sessions)],
            {"role": "user", "content": content},
            source="chat",
        )
        store_ids.append(store_id)
        _seed_chunk_vectors(engine, [(store_id, 0, 0, len(content), [1.0, 0.0])])
    return store_ids


def test_reference_strict_delivers_only_citable_hits_and_reports_the_omissions(
    recall_engine, monkeypatch
):
    """End-to-end: a mixed hit set delivers messages only, backfilled to LIMIT,
    with the uncitable summaries counted in provenance meta."""
    _only_vector_arms(monkeypatch)
    store_ids = _seed_citable_messages(recall_engine, 10)
    node_ids = [
        _add_summary(
            recall_engine,
            f"kanban dashboard sprint rollup {index}",
            session_id="session-s",
            created_at=10.0,
            latest_at=time.time(),
        )
        for index in range(3)
    ]
    _patch_summary_arm(
        monkeypatch,
        [_summary_hit(recall_engine, node_id) for node_id in node_ids],
    )

    payload = _recall(
        recall_engine,
        monkeypatch,
        detail="answer_ready",
        scope_bias=0.0,
        limit=6,
    )

    hits = payload["hits"]
    assert len(hits) == 6, "omitted summaries must be backfilled, not lost"
    assert {hit["kind"] for hit in hits} == {"message_excerpt"}
    assert all(hit["store_id"] in store_ids for hit in hits)
    # Every delivered hit PUBLISHES the span its text occupies, so a consumer
    # never has to guess an offset.
    assert all(
        isinstance(hit.get("content_offset"), int)
        and hit.get("content_returned_chars")
        for hit in hits
    )
    policy = payload["provenance"]["answer_ready"]
    assert policy["reference_strict"] is True
    assert policy["unreferenced_dropped_count"] == len(node_ids)
    assert policy["unreferenced_omitted_count"] == 0


def test_reference_strict_drops_the_164a_message_sourced_summary_too(
    recall_engine, monkeypatch
):
    """#164a populated store_id from ``source_ids[0]``. A leaf node's source_ids
    lists EVERY message it summarizes and its text is generated prose, so that
    store_id is lineage, not a citation — the hit stays undelivered."""
    _only_vector_arms(monkeypatch)
    store_ids = _seed_citable_messages(recall_engine, 4)
    node = _add_summary(
        recall_engine,
        "kanban dashboard sprint rollup",
        session_id="session-s",
        created_at=10.0,
        latest_at=time.time(),
        source_ids=store_ids,
    )
    summary_hit = _summary_hit(recall_engine, node)
    summary_hit["store_id"] = store_ids[0]
    _patch_summary_arm(monkeypatch, [summary_hit])

    payload = _recall(
        recall_engine,
        monkeypatch,
        detail="answer_ready",
        scope_bias=0.0,
        limit=4,
    )

    assert all(hit["kind"] == "message_excerpt" for hit in payload["hits"])
    assert payload["provenance"]["answer_ready"]["unreferenced_dropped_count"] == 1


def test_reference_strict_include_summaries_returns_nothing_rather_than_uncitable(
    recall_engine, monkeypatch
):
    """include='summaries' with the citation-bearing detail asks for citable
    delivery of non-citable material. The honest answer is an empty result plus
    the omission count — not a carve-out that reinstates the defect."""
    node_ids = [
        _add_summary(
            recall_engine,
            f"kanban dashboard sprint rollup {index}",
            session_id=f"session-{index}",
            created_at=10.0,
        )
        for index in range(3)
    ]
    _patch_summary_arm(
        monkeypatch,
        [_summary_hit(recall_engine, node_id) for node_id in node_ids],
    )

    payload = _recall(
        recall_engine,
        monkeypatch,
        include="summaries",
        detail="answer_ready",
        scope_bias=0.0,
        limit=5,
    )

    assert payload["hits"] == []
    assert payload["total_results"] == 0
    assert payload["provenance"]["answer_ready"]["unreferenced_dropped_count"] == 3


def test_reference_strict_leaves_the_snippets_default_untouched(
    recall_engine, monkeypatch
):
    """The default detail makes no source-span claim and carries no hydration,
    so strictness must not cost it evidence."""
    node = _add_summary(
        recall_engine,
        "kanban dashboard sprint rollup",
        session_id="session-a",
        created_at=10.0,
    )
    _seed_summary_vectors(recall_engine, [(node, [1.0, 0.0])])

    payload = _recall(recall_engine, monkeypatch, include="summaries", limit=5)

    assert [hit["node_id"] for hit in payload["hits"]] == [node]
    assert "answer_ready" not in payload["provenance"]


def test_reference_strict_disabled_never_enters_the_new_delivery_path(
    recall_engine, monkeypatch
):
    """Flag-off inertness: with the opt-out set, none of the reference-strict
    code runs and the response carries none of its provenance keys, so delivery
    is byte-identical to the pre-F35 path by construction."""
    _non_strict(recall_engine)
    _seed_citable_messages(recall_engine, 4)
    node_ids = [
        _add_summary(
            recall_engine,
            f"kanban dashboard sprint rollup {index}",
            session_id="session-s",
            created_at=10.0,
            latest_at=time.time(),
        )
        for index in range(2)
    ]
    _patch_summary_arm(
        monkeypatch,
        [_summary_hit(recall_engine, node_id) for node_id in node_ids],
    )

    def _forbidden(*_args, **_kwargs):
        raise AssertionError("reference-strict code ran on the disabled path")

    monkeypatch.setattr(lcm_tools, "_LcmRecallStrictSelector", _forbidden)
    monkeypatch.setattr(lcm_tools, "_lcm_recall_verified_span", _forbidden)

    payload = _recall(
        recall_engine,
        monkeypatch,
        detail="answer_ready",
        scope_bias=0.0,
        limit=6,
    )

    # The legacy path still delivers the (uncitable) summaries it always did.
    assert any(hit["kind"] == "summary" for hit in payload["hits"])
    policy = payload["provenance"]["answer_ready"]
    assert "reference_strict" not in policy
    assert "unreferenced_dropped_count" not in policy
    assert "unreferenced_omitted_count" not in policy
    assert "reference_policy" not in policy


# -- Cross-model review of PR #174 (four mandatory P2 findings) ---------------


def test_summary_only_session_still_yields_citable_evidence_in_strict_mode(
    recall_engine, monkeypatch
):
    """FINDING 1: strict mode must not amputate the summary arm.

    RRF keys a summary by node and a message by store_id, so dropping summary
    entries after fusion leaves message scores and order exactly as if the arm
    had never run. Here the gold session is reachable ONLY by summary
    similarity -- its messages share no query term and have no chunk vectors --
    so if the arm's influence were lost the session would contribute nothing.
    """
    store_ids = [
        recall_engine._store.append(
            "session-gold",
            {"role": "user", "content": f"the quarterly budget reconciliation note {index}"},
            source="chat",
        )
        for index in range(3)
    ]
    node = _add_summary(
        recall_engine,
        "kanban dashboard sprint rollup",
        session_id="session-gold",
        created_at=10.0,
        source_ids=store_ids,
    )
    _seed_summary_vectors(recall_engine, [(node, [1.0, 0.0])])

    payload = _recall(
        recall_engine,
        monkeypatch,
        detail="answer_ready",
        scope_bias=0.0,
        limit=5,
    )

    hits = payload["hits"]
    assert hits, "the summary-only session must still reach the caller"
    assert {hit["kind"] for hit in hits} == {"message_excerpt"}
    assert {hit["store_id"] for hit in hits} <= set(store_ids)
    # Real rows, cited truthfully -- not the node's generated prose.
    assert all(hit["session_id"] == "session-gold" for hit in hits)
    assert all(hit["content_offset"] == 0 for hit in hits)
    assert "kanban dashboard sprint rollup" not in json.dumps(hits)


def test_summary_nodes_come_back_as_non_evidence_leads(recall_engine, monkeypatch):
    """FINDING 1: adaptive retrieval's summary-lead path keeps working.

    ``_extract_search_leads`` walks the whole tool payload for locator handles,
    so surfacing the node ids in provenance restores the drill-down path that
    dropping summary hits would otherwise have closed -- without putting
    generated prose back into evidence.
    """
    node = _add_summary(
        recall_engine,
        "kanban dashboard sprint rollup",
        session_id="session-gold",
        created_at=10.0,
        source_ids=[
            recall_engine._store.append(
                "session-gold", {"role": "user", "content": "budget note"}, source="chat"
            )
        ],
    )
    _seed_summary_vectors(recall_engine, [(node, [1.0, 0.0])])

    payload = _recall(
        recall_engine, monkeypatch, detail="answer_ready", scope_bias=0.0, limit=5
    )

    leads = payload["provenance"]["answer_ready"]["summary_leads"]
    assert [lead["node_id"] for lead in leads] == [node]
    assert leads[0]["session_id"] == "session-gold"
    # A locator, never the summary text.
    assert "summary" not in leads[0]
    assert "snippet" not in leads[0]


def test_summary_leads_preserve_current_context_and_obey_response_limit(
    recall_engine, monkeypatch
):
    node_ids = []
    for index, session_id in enumerate((CURRENT, "session-a", "session-b")):
        source_id = recall_engine._store.append(
            session_id,
            {"role": "user", "content": f"budget note {index}"},
            source="chat",
        )
        node_ids.append(_add_summary(
            recall_engine,
            f"kanban dashboard sprint rollup {index}",
            session_id=session_id,
            created_at=float(index + 1),
            source_ids=[source_id],
        ))
    _seed_summary_vectors(
        recall_engine,
        [
            (node_ids[0], [1.0, 0.0]),
            (node_ids[1], [0.0, 1.0]),
            (node_ids[2], [0.0, 1.0]),
        ],
    )

    payload = _recall(
        recall_engine,
        monkeypatch,
        include="summaries",
        detail="answer_ready",
        scope_bias=0.0,
        limit=1,
    )

    leads = payload["provenance"]["answer_ready"]["summary_leads"]
    assert len(leads) == 1
    assert leads[0]["node_id"] == node_ids[0]
    assert leads[0]["from_current_session"] is True
    assert leads[0]["expand_hint"] == f"lcm_expand(node_id={node_ids[0]})"


def test_admitted_hit_publishes_the_true_chunk_offset_not_zero(
    recall_engine, monkeypatch
):
    """FINDING 2: a post-hydration hit must publish its real offset.

    ``__init__.py``'s ``_answer_ready_baseline`` substitutes ``content_offset``
    0 when the field is absent, which fabricates a reference for every excerpt
    that does not start at the beginning of its row.
    """
    match = "kanban dashboard sprint"
    content = "a" * 2_217 + match + "z" * 500
    store_id = recall_engine._store.append(
        "session-a", {"role": "user", "content": content}, source="chat"
    )
    _seed_chunk_vectors(
        recall_engine, [(store_id, 0, 2_217, len(content), [1.0, 0.0])]
    )
    # Push it past the hydration budget so it is admitted on its chunk span.
    filler = _seed_citable_messages(recall_engine, 9, sessions=("session-b", "session-c"))
    _only_vector_arms(monkeypatch)

    payload = _recall(
        recall_engine,
        monkeypatch,
        include="verbatim",
        detail="answer_ready",
        scope_bias=0.0,
        limit=25,
    )

    hit = next(h for h in payload["hits"] if h["store_id"] == store_id)
    assert "content" not in hit, "this case must exercise the un-hydrated path"
    assert hit["content_offset"] == 2_217
    assert hit["content_returned_chars"] == len(hit["snippet"])
    # The published span is where the text really is.
    start = hit["content_offset"]
    assert content[start:start + hit["content_returned_chars"]] == hit["snippet"]
    assert filler


def test_every_admitted_hit_publishes_a_public_offset(recall_engine, monkeypatch):
    """FINDING 2: no admitted hit may leave the offset for a consumer to guess."""
    _only_vector_arms(monkeypatch)
    _seed_citable_messages(recall_engine, 12)

    payload = _recall(
        recall_engine,
        monkeypatch,
        include="verbatim",
        detail="answer_ready",
        scope_bias=0.0,
        limit=12,
    )

    assert payload["hits"]
    for hit in payload["hits"]:
        assert isinstance(hit["content_offset"], int)
        assert hit["content_returned_chars"] > 0


def test_hydration_miss_refills_from_the_ranked_tail(recall_engine, monkeypatch):
    """FINDING 3: a row deleted between the reads must not underfill.

    Selection stops at ``limit``; if a selected row then vanishes, omitting it
    without drawing a replacement leaves the response short while citable
    candidates sit unexamined just below the cut.
    """
    _only_vector_arms(monkeypatch)
    _seed_citable_messages(recall_engine, 12)

    # Doom rows the run actually DELIVERS, so the refill is genuinely exercised
    # rather than the deletion landing on candidates below the cut.
    baseline = _recall(
        recall_engine,
        monkeypatch,
        include="verbatim",
        detail="answer_ready",
        scope_bias=0.0,
        limit=8,
    )
    assert len(baseline["hits"]) == 8
    doomed = {hit["store_id"] for hit in baseline["hits"][:2]}
    real_get_batch = recall_engine._store.get_batch

    def deleting_get_batch(ids):
        # Simulate delete_session_messages landing between the ranking read and
        # the hydration read: the rows are simply not there any more.
        return {
            key: value
            for key, value in real_get_batch(ids).items()
            if key not in doomed
        }

    monkeypatch.setattr(recall_engine._store, "get_batch", deleting_get_batch)

    payload = _recall(
        recall_engine,
        monkeypatch,
        include="verbatim",
        detail="answer_ready",
        scope_bias=0.0,
        limit=8,
    )

    assert len(payload["hits"]) == 8, "the count must survive a hydration miss"
    assert not (doomed & {hit["store_id"] for hit in payload["hits"]})
    assert payload["total_results"] == 8


def test_stale_chunk_span_never_becomes_a_strict_reference(
    recall_engine, monkeypatch
):
    """FINDING 4: a chunk index is not proof the row still says that.

    Cleanup can delete or rewrite the message after chunk hydration and before
    response shaping; without a check against the current row the stale hit
    would be delivered as reference-strict with a dangling span.
    """
    _only_vector_arms(monkeypatch)
    content = "kanban dashboard sprint " + "evidence " * 40
    store_id = recall_engine._store.append(
        "session-a", {"role": "user", "content": content}, source="chat"
    )
    _seed_chunk_vectors(recall_engine, [(store_id, 0, 0, len(content), [1.0, 0.0])])
    other = _seed_citable_messages(recall_engine, 9, sessions=("session-b", "session-c"))
    real_get_batch = recall_engine._store.get_batch

    def rewriting_get_batch(ids):
        rows = dict(real_get_batch(ids))
        if store_id in rows:
            # The row now holds different bytes than the chunk index recorded.
            rows[store_id] = {**rows[store_id], "content": "unrelated replacement"}
        return rows

    monkeypatch.setattr(recall_engine._store, "get_batch", rewriting_get_batch)

    payload = _recall(
        recall_engine,
        monkeypatch,
        include="verbatim",
        detail="answer_ready",
        scope_bias=0.0,
        limit=25,
    )

    delivered = {hit["store_id"] for hit in payload["hits"]}
    assert store_id not in delivered, "a stale span must not be delivered as strict"
    assert delivered <= set(other)
    for hit in payload["hits"]:
        assert isinstance(hit["content_offset"], int)


# -- Delta-2 review: verification folded INTO the selection walk --------------
#
# Findings 1, 2 and 4 all came from verification living AFTER admission as a
# separate refill pass: delta shaping could bypass the pass, a failed candidate
# had already spent its session quota, and each replacement paid its own read.
# Verifying before admitting removes all three by construction; finding 3 is the
# independent one -- fusion must keep every citable representation of a row.


def test_delta_mode_draws_replacements_from_the_ranked_tail(
    recall_engine, monkeypatch
):
    """FINDING 1: delta shaping must not silently return short.

    It discards entries whose refs the caller has already seen, which is the
    same underfill the resumable walk exists to prevent -- valid tail candidates
    were left unexamined while the response came back one hit light.
    """
    _only_vector_arms(monkeypatch)
    store_ids = _seed_citable_messages(
        recall_engine, 30, sessions=tuple(f"session-{i}" for i in range(6))
    )

    # Delta selects a 25-candidate wave up front, so the caller must have seen
    # enough of that wave that satisfying `limit` REQUIRES the ranked tail.
    first = _recall(
        recall_engine,
        monkeypatch,
        detail="answer_ready",
        include="verbatim",
        scope_bias=0.0,
        limit=25,
    )
    assert len(first["hits"]) == 25
    seen = [
        f"lcm:{hit['store_id']}:{hit['content_offset']}-"
        f"{hit['content_offset'] + hit['content_returned_chars']}"
        for hit in first["hits"][:20]
    ]

    delta = _recall(
        recall_engine,
        monkeypatch,
        detail="answer_ready",
        include="verbatim",
        scope_bias=0.0,
        limit=8,
        seen_refs=seen,
    )

    assert len(delta["hits"]) == 8, "seen refs must be replaced, not just removed"
    assert not ({hit["exact_ref"] for hit in delta["hits"]} & set(seen))
    assert {hit["store_id"] for hit in delta["hits"]} <= set(store_ids)


def test_failed_candidate_does_not_spend_session_quota():
    """FINDING 2: quota is for DELIVERED hits.

    Charging it at admission let five failing same-session candidates exhaust
    the per-session budget and block a valid sixth, returning nothing at all.
    """
    per_session = 5
    # Citable-SHAPED but unverifiable, so each one reaches the quota check --
    # a shape-rejected candidate never gets that far and proves nothing here.
    ordered = [
        _message_entry(index, session_id="session-a", snippet="not in the row")
        for index in range(1, 6)
    ]
    ordered.append(_message_entry(99, session_id="session-a"))

    selected, dropped, unreferenced = lcm_tools._lcm_recall_citable_entries(
        ordered,
        limit=5,
        per_session_limit=per_session,
        expanded_limit=0,
        engine=_stub_engine([*range(1, 6), 99]),
    )

    assert [entry["hit"]["store_id"] for entry in selected] == [99]
    assert unreferenced == 5
    assert dropped == 0, "a hit that was never delivered cannot be a diversity drop"


def test_fusion_keeps_a_rows_citable_representation_when_fts_is_the_base(
    recall_engine, monkeypatch
):
    """FINDING 3: fusion picks ONE base per row; it must not lose the others.

    Nine rows surface through BOTH the FTS arm and the summary-source arm, and
    none through the chunk arm -- so the existing chunk reconciliation cannot
    help. FTS wins the fused base by arm order, but an FTS snippet is a match
    window taken from deep inside the row while the hit claims offset 0, so past
    the hydration budget the ninth row was dropped even though the
    summary-source representation of that same row (its verbatim prefix) is
    perfectly citable.
    """
    match = "kanban dashboard sprint"
    contents = {}
    for index in range(9):
        session = f"session-{index}"
        store_id = recall_engine._store.append(
            session,
            {"role": "user", "content": "lead " * 60 + match + f" tail {index}"},
            source="chat",
        )
        contents[store_id] = recall_engine._store.get(store_id)["content"]
        node = _add_summary(
            recall_engine,
            f"{match} rollup {index}",
            session_id=session,
            created_at=10.0,
            source_ids=[store_id],
        )
        _seed_summary_vectors(recall_engine, [(node, [1.0, 0.0])])

    payload = _recall(
        recall_engine,
        monkeypatch,
        detail="answer_ready",
        scope_bias=0.0,
        limit=9,
    )

    assert len(payload["hits"]) == 9, "no row may be lost to its FTS representation"
    for hit in payload["hits"]:
        content = contents[hit["store_id"]]
        start = hit["content_offset"]
        text = hit.get("content") or hit["snippet"]
        assert content[start:start + len(text)] == text


def test_selection_reads_rows_in_batches_not_one_per_candidate():
    """FINDING 4: the single-batch contract must survive replacement.

    Verifying per replacement made the reader a query-per-candidate path (55
    singleton reads on an 80-candidate repro).
    """
    # Citable-SHAPED but unverifiable: each one must be read before it can be
    # rejected, which is exactly the path that used to read one row at a time.
    ordered = [
        _message_entry(index, session_id=f"session-{index}", snippet="not in the row")
        for index in range(1, 80)
    ]
    ordered.append(_message_entry(99, session_id="session-99"))
    engine = _stub_engine([*range(1, 80), 99])

    selected, _dropped, unreferenced = lcm_tools._lcm_recall_citable_entries(
        ordered,
        limit=1,
        per_session_limit=5,
        expanded_limit=0,
        engine=engine,
    )

    assert [entry["hit"]["store_id"] for entry in selected] == [99]
    assert unreferenced == 79
    calls = engine._store.batch_calls
    assert len(calls) <= 4, f"expected batched reads, got {len(calls)} for 80 candidates"
    assert max(len(call) for call in calls) > 1, "reads must actually be batched"


def test_hydration_reuses_the_rows_selection_already_read(
    recall_engine, monkeypatch
):
    """The verified snapshot is handed to hydration, so an admitted candidate
    cannot go unhydrated because the row vanished between two separate reads."""
    _only_vector_arms(monkeypatch)
    _seed_citable_messages(recall_engine, 10)
    calls = []
    real_get_batch = recall_engine._store.get_batch

    def counting_get_batch(ids):
        calls.append(list(ids))
        return real_get_batch(ids)

    monkeypatch.setattr(recall_engine._store, "get_batch", counting_get_batch)

    payload = _recall(
        recall_engine,
        monkeypatch,
        detail="answer_ready",
        include="verbatim",
        scope_bias=0.0,
        limit=8,
    )

    assert len(payload["hits"]) == 8
    # Selection's wave read is the only row read; hydration reuses it.
    assert len(calls) == 1, f"expected one batched read, got {len(calls)}"


# -- Delta-3 review ------------------------------------------------------------


def test_seen_delta_results_do_not_spend_session_quota(recall_engine, monkeypatch):
    """FINDING 1: an entry the caller already has was never delivered either.

    Quota is charged when the walk admits a candidate, but delta shaping drops
    already-seen references afterwards. With 20 citable rows in ONE session and
    the first five seen, the session budget was spent on those five and every
    replacement then failed the density check -- 0 results and 15 diversity
    drops with 15 novel rows still available.
    """
    _only_vector_arms(monkeypatch)
    store_ids = _seed_citable_messages(recall_engine, 20, sessions=("session-a",))

    first = _recall(
        recall_engine,
        monkeypatch,
        detail="answer_ready",
        include="verbatim",
        scope_bias=0.0,
        limit=5,
    )
    assert len(first["hits"]) == 5
    seen = [
        f"lcm:{hit['store_id']}:{hit['content_offset']}-"
        f"{hit['content_offset'] + hit['content_returned_chars']}"
        for hit in first["hits"]
    ]

    delta = _recall(
        recall_engine,
        monkeypatch,
        detail="answer_ready",
        include="verbatim",
        scope_bias=0.0,
        limit=5,
        seen_refs=seen,
    )

    assert len(delta["hits"]) == 5, "seen entries must refund the slot they never used"
    assert not ({hit["exact_ref"] for hit in delta["hits"]} & set(seen))
    assert {hit["store_id"] for hit in delta["hits"]} <= set(store_ids)


def test_a_missing_row_does_not_cascade_reads_or_retain_the_corpus():
    """FINDING 2: 'not prefetched' and 'prefetched but absent' are different.

    Without that distinction the walk kept calling for more waves looking for a
    row that will never arrive, loading every remaining wave and retaining all
    of it: 100 candidates with row 1 missing cost four reads [32,32,32,4] and
    held 99 rows just to select row 2.
    """
    ordered = [
        _message_entry(index, session_id=f"session-{index}")
        for index in range(1, 101)
    ]
    engine = _stub_engine(range(2, 101))  # row 1 is gone

    selector = lcm_tools._LcmRecallStrictSelector(
        ordered,
        engine=engine,
        per_session_limit=5,
        expanded_limit=0,
        wave_size=32,
    )
    selected = selector.take(1)

    assert [entry["hit"]["store_id"] for entry in selected] == [2]
    calls = engine._store.batch_calls
    assert len(calls) == 1, f"a missing row must not cascade waves: {[len(c) for c in calls]}"
    assert len(selector.rows) <= 32, f"retained {len(selector.rows)} rows, expected one wave"


def test_rejected_candidate_rows_are_not_retained():
    """FINDING 2 (retention half): rows outside the active wave are released.

    Reading in waves bounds the reads, but holding on to every row the walk
    rejected would still grow with the corpus. Only rows a delivered hit
    depends on -- hydration reads from this same snapshot -- need to survive.
    """
    ordered = [
        _message_entry(index, session_id=f"session-{index}", snippet="not in the row")
        for index in range(1, 100)
    ]
    ordered.append(_message_entry(100, session_id="session-100"))
    engine = _stub_engine(range(1, 101))

    selector = lcm_tools._LcmRecallStrictSelector(
        ordered,
        engine=engine,
        per_session_limit=5,
        expanded_limit=0,
        wave_size=32,
    )
    selected = selector.take(1)

    assert [entry["hit"]["store_id"] for entry in selected] == [100]
    assert selector.unreferenced_dropped == 99
    assert len(selector.rows) <= 2, (
        f"retained {len(selector.rows)} rows after rejecting 99 candidates"
    )


# -- Delta-4 review: the selection ledger --------------------------------------


def test_refund_reaches_the_ranking_when_limit_exceeds_the_session_cap(
    recall_engine, monkeypatch
):
    """FINDING 1: selection must not consume the ranking before refunds land.

    Asking for more than one session can supply made the first wave walk to
    EXHAUSTION -- admitting 5, density-dropping the other 15 -- so when delta
    shaping released the five seen entries there was nothing left to resume
    into. A density block is refundable, unlike a shape or verification
    failure, so those candidates are held in rank order instead of consumed.
    """
    _only_vector_arms(monkeypatch)
    store_ids = _seed_citable_messages(recall_engine, 20, sessions=("session-a",))

    first = _recall(
        recall_engine,
        monkeypatch,
        detail="answer_ready",
        include="verbatim",
        scope_bias=0.0,
        limit=10,
    )
    # One session, cap 5 -- the response can never exceed the cap.
    assert len(first["hits"]) == 5
    seen = [
        f"lcm:{hit['store_id']}:{hit['content_offset']}-"
        f"{hit['content_offset'] + hit['content_returned_chars']}"
        for hit in first["hits"]
    ]

    delta = _recall(
        recall_engine,
        monkeypatch,
        detail="answer_ready",
        include="verbatim",
        scope_bias=0.0,
        limit=10,
        seen_refs=seen,
    )

    assert len(delta["hits"]) == 5, "the refund must reach the deferred candidates"
    assert not ({hit["exact_ref"] for hit in delta["hits"]} & set(seen))
    assert {hit["store_id"] for hit in delta["hits"]} <= set(store_ids)


def test_double_release_is_a_counted_no_op_not_a_second_refund():
    """FINDING 2: refunds are per-entry, not aggregate decrements.

    Releasing one entry twice used to give the session two slots back, letting
    three hits through a cap of two.
    """
    ordered = [
        _message_entry(index, session_id="session-a") for index in range(1, 6)
    ]
    engine = _stub_engine(range(1, 6))
    selector = lcm_tools._LcmRecallStrictSelector(
        ordered, engine=engine, per_session_limit=2, expanded_limit=0
    )

    taken = selector.take(2)
    assert len(taken) == 2

    assert selector.release(taken[0]) is True
    assert selector.release(taken[0]) is False, "second release must be a no-op"
    assert selector.release(taken[0]) is False
    assert selector.ledger.double_releases == 2
    assert selector.ledger.live_count == 1
    assert selector.ledger.session_count("session-a") == 1

    # Exactly ONE slot came back, so the cap of two still holds.
    selector.take(2)
    assert selector.ledger.session_count("session-a") == 2


def test_released_entry_stops_pinning_its_row():
    """FINDING 2: retention is derived from the ledger, not a parallel set.

    Released stores stayed in the admitted-store set, so eviction never fired
    and 100 admitted-then-released entries retained all 100 rows.
    """
    ordered = [
        _message_entry(index, session_id=f"session-{index}")
        for index in range(1, 101)
    ]
    engine = _stub_engine(range(1, 101))
    selector = lcm_tools._LcmRecallStrictSelector(
        ordered, engine=engine, per_session_limit=5, expanded_limit=0, wave_size=32
    )

    for _ in range(100):
        taken = selector.take(selector.ledger.live_count + 1)
        if not taken:
            break
        selector.release(taken[0])

    assert selector.ledger.live_count == 0
    assert len(selector.rows) <= 32, (
        f"retained {len(selector.rows)} rows with nothing live"
    )


def _has_subsequence(history, pattern):
    """True when `pattern` appears in order (not necessarily adjacently)."""
    index = 0
    for event in history:
        if index < len(pattern) and pattern[index](event):
            index += 1
    return index == len(pattern)


def _is(name):
    return lambda event: event == name


def _startswith(prefix):
    return lambda event: event.startswith(prefix)


def test_ledger_invariants_hold_under_a_randomized_transition_sequence():
    """Property check: no interleaving of the lifecycle can break the books.

    Each review round found a different piece of walk state leaking at a
    different seam, so the accounting is exercised as a state machine rather
    than only at the seams already known to have failed.

    The reach assertions are on per-candidate HISTORIES, not on aggregate
    counters: a seed can easily satisfy "some row was deleted" and "some
    post-budget rejection happened" without those ever meeting in one
    candidate's life, which is precisely where the bugs have been living.
    """
    import random

    rng = random.Random(20260729)
    per_session_limit = 3
    wave_size = 16
    # expanded_limit > 0 so the SAME candidate can be judged in-budget or
    # post-budget as the live count crosses the threshold -- the transition that
    # makes a rejection's rule matter. A third of the candidates carry an excerpt
    # that does not match their row: admissible in-budget (where the text is cut
    # from the row) and not post-budget (where it must be found at its offset).
    expanded_limit = 4
    sessions = [f"session-{index % 7}" for index in range(1, 121)]
    ordered = [
        _message_entry(
            index,
            session_id=sessions[index - 1],
            snippet="not in the row" if index % 3 == 0 else None,
        )
        for index in range(1, 121)
    ]
    engine = _stub_engine(range(1, 121))
    selector = lcm_tools._LcmRecallStrictSelector(
        ordered,
        engine=engine,
        per_session_limit=per_session_limit,
        expanded_limit=expanded_limit,
        wave_size=wave_size,
    )
    ledger = selector.ledger
    admitted: list[dict] = []
    delivered: list[dict] = []
    released: list[dict] = []
    deleted: set[int] = set()
    by_store = {entry["hit"]["store_id"]: entry for entry in ordered}
    history: dict[int, list[str]] = {id(entry): ["untouched"] for entry in ordered}

    def status_of(entry):
        state = ledger.state(entry)
        if state is not None:
            return state
        if id(entry) in selector._blocked:
            return "blocked"
        rejected = selector._rejected.get(id(entry))
        if rejected is True:
            return "rejected-in"
        if rejected is False:
            return "rejected-post"
        return "untouched"

    def record(entry, event):
        log = history[id(entry)]
        if log[-1] != event:
            log.append(event)

    def observe():
        """Append each candidate's status change, and note released rows."""
        for entry in ordered:
            status = status_of(entry)
            record(entry, status)
            if status in ("blocked", "rejected-post"):
                if entry["hit"]["store_id"] not in selector.rows:
                    record(entry, "evicted")

    def check(step):
        for key in {selector._session_key(e["hit"]) for e in ordered}:
            assert ledger.session_count(key) <= per_session_limit, f"step {step}: {key}"
        live = [e for e in admitted if ledger.state(e) in ("admitted", "delivered")]
        assert ledger.live_count == len(live), f"step {step}: live drift"
        assert sum(ledger._session_counts.values()) == ledger.live_count, (
            f"step {step}: session totals drift"
        )
        for entry in released:
            assert ledger.state(entry) == "released", f"step {step}: state reversed"
        assert len(selector.rows) <= wave_size + ledger.live_count, (
            f"step {step}: retained {len(selector.rows)}"
        )

    def refund(entry):
        if selector.release(entry):
            released.append(entry)
            # A refund happened while every unsettled candidate waited.
            for other in ordered:
                if not selector._is_settled(other):
                    record(other, "refund")
            return True
        return False

    for step in range(200):
        action = rng.random()
        if action < 0.32:
            for entry in selector.take(ledger.live_count + rng.randint(1, 6)):
                store_id = entry["hit"]["store_id"]
                # Admission must rest on the row as it stands NOW.
                assert store_id in selector.rows, (
                    f"step {step}: admitted {store_id} without holding its row"
                )
                assert store_id not in deleted, (
                    f"step {step}: admitted {store_id} on a stale proof"
                )
                admitted.append(entry)
        elif action < 0.42:
            # Prefer deleting a row whose candidate is waiting with its row
            # already released -- the blocked -> evicted -> deleted path.
            waiting = [
                sid
                for sid, entry in by_store.items()
                if sid not in deleted
                and not ledger.holds_store(sid)
                and status_of(entry) in ("blocked", "rejected-post")
                and sid not in selector.rows
            ]
            pool = waiting or [
                sid
                for sid in by_store
                if sid not in deleted and not ledger.holds_store(sid)
            ]
            if pool:
                victim = rng.choice(pool)
                engine._store.drop(victim)
                selector.rows.pop(victim, None)
                deleted.add(victim)
                record(by_store[victim], "row-deleted")
        elif action < 0.54 and admitted:
            entry = rng.choice(admitted)
            if ledger.deliver(entry):
                delivered.append(entry)
        elif action < 0.80 and admitted:
            refund(rng.choice(admitted))
        elif action < 0.90 and admitted:
            # DRAIN: hand back every live slot at once. Without this the live
            # count rarely falls under the hydration budget, so a post-budget
            # rejection is only ever re-judged post-budget and the
            # rejected-then-revisited-in-budget transition stays unreachable.
            for entry in list(admitted):
                if ledger.state(entry) == "admitted":
                    refund(entry)
        elif released:
            before = ledger.live_count
            assert selector.release(rng.choice(released)) is False
            assert ledger.live_count == before, f"step {step}: double release refunded"
        observe()
        check(step)

    assert delivered, "the sequence must actually deliver something"
    assert set(map(id, ledger.delivered_entries())) == set(map(id, delivered))
    assert ledger.double_releases > 0, "the sequence must exercise double release"

    # -- Reach, proved on HISTORIES: the family must be inside the tested space,
    #    and aggregate counters cannot show that these events ever MET. --
    logs = list(history.values())
    assert any(
        _has_subsequence(
            log,
            [_is("blocked"), _is("evicted"), _is("row-deleted"), _is("refund"),
             _startswith("rejected")],
        )
        for log in logs
    ), "no candidate lived blocked -> evicted -> deleted -> refund -> re-examined"
    assert any(
        _has_subsequence(log, [_is("rejected-post"), _is("refund"), _is("admitted")])
        for log in logs
    ), "no candidate was post-budget rejected then revisited in-budget"


def test_take_tops_up_to_a_target_rather_than_adding_a_count():
    """A wave asks for what the response is still MISSING.

    Expressed as a count, a wave after a partial refund would admit a full
    count on top of what is already live and overshoot the response; expressed
    as a target, a refunded slot is exactly what the next wave refills.
    """
    ordered = [
        _message_entry(index, session_id=f"session-{index}")
        for index in range(1, 21)
    ]
    selector = lcm_tools._LcmRecallStrictSelector(
        ordered,
        engine=_stub_engine(range(1, 21)),
        per_session_limit=5,
        expanded_limit=0,
    )

    taken = selector.take(5)
    assert len(taken) == 5 and selector.ledger.live_count == 5
    assert selector.take(5) == [], "already at the target -- nothing to add"
    assert selector.ledger.live_count == 5

    assert selector.release(taken[0]) is True
    assert selector.ledger.live_count == 4
    assert len(selector.take(5)) == 1, "a wave refills exactly the freed slot"
    assert selector.ledger.live_count == 5


def test_long_seen_run_does_not_discard_novel_rows_behind_the_cap(
    recall_engine, monkeypatch
):
    """DELTA-5: refunds are sequential, so no bounded buffer can hold the queue.

    A reserve sized to "no more slots than were admitted" assumed refunds
    happen against SIMULTANEOUS admissions. Delta processing admits and
    releases in sequence, so a long run of already-seen references can refund
    far more slots than were ever live at once. With 100 ranked rows in one
    session, a cap of 5 and the first 30 references seen, the reserve filled
    with rows 6-30 and rows 31-100 were discarded outright -- the response came
    back EMPTY even though rows 31-35 were novel, citable and still ranked.

    Candidates are already rank-ordered, so a blocked one is rediscoverable by
    POSITION; nothing needs to be stored, and nothing can be dropped.
    """
    _only_vector_arms(monkeypatch)
    store_ids = _seed_citable_messages(recall_engine, 100, sessions=("session-a",))

    seen: list[str] = []
    while len(seen) < 30:
        page = _recall(
            recall_engine,
            monkeypatch,
            detail="answer_ready",
            include="verbatim",
            scope_bias=0.0,
            limit=10,
            **({"seen_refs": list(seen)} if seen else {}),
        )
        assert page["hits"], f"went empty after {len(seen)} seen refs"
        seen.extend(
            f"lcm:{hit['store_id']}:{hit['content_offset']}-"
            f"{hit['content_offset'] + hit['content_returned_chars']}"
            for hit in page["hits"]
        )

    after = _recall(
        recall_engine,
        monkeypatch,
        detail="answer_ready",
        include="verbatim",
        scope_bias=0.0,
        limit=10,
        seen_refs=seen[:30],
    )

    assert after["hits"], "novel rows behind the cap must remain reachable"
    assert not ({hit["exact_ref"] for hit in after["hits"]} & set(seen[:30]))
    assert {hit["store_id"] for hit in after["hits"]} <= set(store_ids)
    assert all(hit["content_offset"] is not None for hit in after["hits"])


# -- Delta-6 review: cache coherence across a rewind ---------------------------


def test_post_budget_rejection_does_not_bar_a_later_in_budget_admission():
    """DELTA-6 FINDING 1: a rejection is only as durable as its rule.

    The two budgets prove different things. In-budget, the delivered text is a
    window cut from the row, so the row existing IS the proof. Post-budget, the
    hit ships its own excerpt, which must be found at its offset. A candidate
    carrying an excerpt that does not match its row therefore fails post-budget
    and passes in-budget -- but the rejection was remembered without its rule,
    so once a refund made the candidate in-budget again it stayed skipped and
    the response underfilled.
    """
    # Excerpt does not match the row: post-budget it cannot be cited, in-budget
    # it can, because hydration cuts the text from the row itself.
    ordered = [
        _message_entry(1, session_id="session-a"),
        _message_entry(2, session_id="session-b", snippet="not in the row"),
    ]
    engine = _stub_engine([1, 2])
    selector = lcm_tools._LcmRecallStrictSelector(
        ordered, engine=engine, per_session_limit=5, expanded_limit=1
    )

    first = selector.take(1)
    assert [e["hit"]["store_id"] for e in first] == [1]
    # Live == expanded_limit, so candidate 2 is judged post-budget and rejected.
    assert selector.take(2) == []
    assert selector.unreferenced_dropped == 1

    # The refund puts the budget back; candidate 2 is now admissible in-budget.
    assert selector.release(first[0]) is True
    regained = selector.take(1)
    assert [e["hit"]["store_id"] for e in regained] == [2], (
        "a post-budget rejection must not bar an in-budget admission"
    )
    assert selector.unreferenced_dropped == 0


def test_row_deleted_while_blocked_is_reverified_before_admission():
    """DELTA-6 FINDING 2: cached proof is void once its row is let go.

    A blocked candidate releases its row. If that row is then deleted, a refund
    rewinding to the candidate must NOT admit it on the strength of the earlier
    verification -- the bytes it was proved against are gone.
    """
    ordered = [
        _message_entry(1, session_id="session-a"),
        _message_entry(2, session_id="session-a"),
        _message_entry(3, session_id="session-b"),
    ]
    engine = _stub_engine([1, 2, 3])
    selector = lcm_tools._LcmRecallStrictSelector(
        ordered, engine=engine, per_session_limit=1, expanded_limit=8
    )

    taken = selector.take(2)
    assert [e["hit"]["store_id"] for e in taken] == [1, 3]
    # Candidate 2 was verified, then blocked on session-a's cap, so its row was
    # released. Delete it from the store while it waits.
    assert 2 not in selector.rows
    engine._store.drop(2)

    assert selector.release(taken[0]) is True
    regained = selector.take(2)

    assert [e["hit"]["store_id"] for e in regained] == [], (
        "a deleted row must be re-proved, not admitted from a stale proof"
    )
    assert not selector.ledger.holds_store(2)
    assert selector.unreferenced_dropped == 1


def test_reversible_rejection_stays_reachable_after_a_partial_refund():
    """DELTA-7: the resume point must be DERIVED, not remembered.

    A stored resume cursor is state that outlives its precondition. The first
    rewind consumes it, and a later walk that SKIPS an already-rejected
    post-budget candidate never re-arms it -- so once the cursor has moved past
    that candidate, no subsequent refund can reach it and it is permanently
    unreachable although it is admissible again.

    The interleaving that exposes it: a density block earlier in the ranking
    takes the one stored resume slot, the rewind spends it, the next walk steps
    over the rejected candidate, and only then do the remaining refunds land.
    """
    # 2 carries an excerpt that does not match its row: rejected post-budget,
    # admissible in-budget where the text is cut from the row instead.
    ordered = [
        _message_entry(1, session_id="session-a"),
        _message_entry(2, session_id="session-a"),
        _message_entry(3, session_id="session-b", snippet="not in the row"),
        _message_entry(4, session_id="session-c"),
        _message_entry(5, session_id="session-d"),
    ]
    engine = _stub_engine([1, 2, 3, 4, 5])
    selector = lcm_tools._LcmRecallStrictSelector(
        ordered, engine=engine, per_session_limit=1, expanded_limit=1
    )

    first = selector.take(5)
    assert [e["hit"]["store_id"] for e in first] == [1, 4, 5]
    assert id(ordered[1]) in selector._blocked, "2 is held by the density cap"
    assert selector._rejected.get(id(ordered[2])) is False, "3 is post-budget rejected"

    # The density block takes the single stored resume slot; this refund spends it.
    assert selector.release(first[0]) is True
    assert [e["hit"]["store_id"] for e in selector.take(3)] == [2]

    # This walk stepped OVER candidate 3 without re-arming any stored resume.
    assert selector.release(first[1]) is True
    assert selector.take(3) == []
    assert selector.release(first[2]) is True
    assert selector.release(ordered[1]) is True
    assert selector.ledger.live_count == 0

    regained = selector.take(1)
    assert [e["hit"]["store_id"] for e in regained] == [3], (
        "a revisitable candidate must never become unreachable"
    )
