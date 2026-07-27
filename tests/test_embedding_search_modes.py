from __future__ import annotations

import json
import math
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import hermes_lcm.embedding_provider as embedding_provider
import hermes_lcm.tools as lcm_tools
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.embedding_provider import VoyageError
from hermes_lcm.project_scope import ProjectMetadata
from hermes_lcm.store import MessageStore
from hermes_lcm.vector_store import KNNResult, VectorStore


class MockProvider:
    provider_id = "mock"
    model_id = "mock-model"
    dim = 2

    def __init__(self, vector=(1.0, 0.0)):
        self.vector = list(vector)
        self.queries: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return list(self.vector)


@pytest.fixture
def semantic_engine(tmp_path):
    config = LCMConfig(
        database_path=str(tmp_path / "semantic.db"),
        embeddings_enabled=True,
        embedding_provider="ollama",
        embedding_model="mock-model",
        embedding_query_timeout_s=2.0,
    )
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


def _add_summary(
    engine,
    summary: str,
    *,
    created_at: float,
    source_ids=None,
    session_id="session-a",
) -> int:
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
            latest_at=created_at,
            expand_hint=f"Expand {summary[:20]}",
        )
    )


def _seed_vectors(engine, rows):
    store = VectorStore(engine._store.db_path, config=engine._config)
    try:
        store.register_profile("mock-model", "mock", 2)
        identity = store.capture_identity("mock-model", provider="mock")
        for node_id, vector in rows:
            store.record_embedding(
                str(node_id),
                "summary",
                "mock-model",
                vector,
                identity=identity,
            )
    finally:
        store.close()


def test_semantic_happy_path_orders_by_cosine_and_surfaces_confidence_coverage(
    semantic_engine,
    monkeypatch,
):
    scores = [0.7, 0.55, 0.4, 0.2]
    node_ids = [
        _add_summary(
            semantic_engine,
            f"semantic result {index}",
            created_at=float(index + 1),
        )
        for index in range(len(scores))
    ]
    _seed_vectors(
        semantic_engine,
        [
            (node_id, [score, math.sqrt(1.0 - score * score)])
            for node_id, score in zip(node_ids, scores)
        ],
    )
    provider = MockProvider()
    monkeypatch.setattr(
        embedding_provider, "resolve_provider", lambda _config: provider
    )
    monkeypatch.setattr(lcm_tools, "resolve_provider", lambda _config: provider)

    payload = json.loads(
        lcm_tools.lcm_grep(
            {"query": "meaning-preserving query", "mode": "semantic", "limit": 4},
            engine=semantic_engine,
        )
    )

    assert [hit["node_id"] for hit in payload["results"]] == node_ids
    assert [hit["confidence"] for hit in payload["results"]] == [
        "high",
        "medium",
        "low",
        "noise",
    ]
    assert [hit["confidence_band"] for hit in payload["results"]] == [
        "high",
        "medium",
        "low",
        "noise",
    ]
    assert [hit["cosine_score"] for hit in payload["results"]] == pytest.approx(scores)
    assert payload["coverage"] in {"full", "bounded"}
    assert payload["degraded_to_fts"] is False
    assert provider.queries == ["meaning-preserving query"]


@pytest.mark.parametrize("mode", ["semantic", "hybrid"])
def test_semantic_modes_fail_closed_for_empty_project_scope(
    semantic_engine, monkeypatch, mode
):
    node_id = _add_summary(
        semantic_engine,
        "project scoped semantic result",
        created_at=1.0,
    )
    _seed_vectors(semantic_engine, [(node_id, [1.0, 0.0])])
    semantic_engine._store.set_session_project_metadata(
        "session-a",
        ProjectMetadata("project-a", "/projects/a", "/projects/a"),
    )
    provider = MockProvider()
    monkeypatch.setattr(lcm_tools, "resolve_provider", lambda _config: provider)

    payload = json.loads(
        lcm_tools.lcm_grep(
            {
                "query": "project scoped semantic result",
                "mode": mode,
                "project_id": "missing-project",
            },
            engine=semantic_engine,
        )
    )

    assert payload["project_id"] == "missing-project"
    assert payload["results"] == []


def test_semantic_timeout_returns_explicit_deadline_without_starting_fallback(
    semantic_engine, monkeypatch
):
    semantic_engine._config.embedding_query_timeout_s = 0.02
    semantic_engine._store.append(
        "session-a",
        {"role": "user", "content": "needle survives provider timeout"},
    )

    class SlowProvider(MockProvider):
        def embed_query(self, text):
            time.sleep(0.1)
            return super().embed_query(text)

    monkeypatch.setattr(lcm_tools, "resolve_provider", lambda _config: SlowProvider())
    started = time.monotonic()
    payload = json.loads(
        lcm_tools.lcm_grep(
            {"query": "needle", "mode": "semantic"},
            engine=semantic_engine,
        )
    )

    assert time.monotonic() - started < 0.09
    assert payload["timeout"] is True
    assert payload["mode"] == "semantic"
    assert payload["timeout_stage"] == "full_text"


def test_semantic_budget_bounds_knn_and_does_not_start_fallback_after_expiry(
    semantic_engine, monkeypatch
):
    """The one request deadline prevents a post-timeout fallback from starting."""
    semantic_engine._config.embedding_query_timeout_s = 0.02
    semantic_engine._store.append(
        "session-a", {"role": "user", "content": "needle survives a slow knn"}
    )

    class SlowKNNStore:
        def __init__(self, *_args, **_kwargs):
            pass

        def knn(self, *_args, **_kwargs):
            time.sleep(0.2)  # far exceeds the 0.02s budget
            return KNNResult(coverage="full")

        def close(self):
            pass

    fallback_calls = 0

    def slow_full_text(_args, **_kwargs):
        nonlocal fallback_calls
        fallback_calls += 1
        time.sleep(0.08)
        return json.dumps({"results": []})

    monkeypatch.setattr(lcm_tools, "VectorStore", SlowKNNStore)
    monkeypatch.setattr(lcm_tools, "resolve_provider", lambda _config: MockProvider())
    monkeypatch.setattr(lcm_tools, "_lcm_grep_full_text", slow_full_text)

    started = time.monotonic()
    payload = json.loads(
        lcm_tools.lcm_grep(
            {"query": "needle", "mode": "semantic"}, engine=semantic_engine
        )
    )
    elapsed = time.monotonic() - started

    assert elapsed < 0.08
    assert payload["timeout"] is True
    assert fallback_calls == 0


def test_timeout_worker_is_daemon_and_provider_call_is_bounded(
    semantic_engine,
    monkeypatch,
):
    semantic_engine._config.embedding_query_timeout_s = 0.02
    semantic_engine._store.append(
        "session-a",
        {"role": "user", "content": "daemon timeout fallback"},
    )
    release = threading.Event()
    observed_timeouts: list[float] = []

    class BlockingProvider(MockProvider):
        def embed_query_interactive(self, _text, *, timeout):
            observed_timeouts.append(timeout)
            release.wait(1.0)
            return list(self.vector)

    monkeypatch.setattr(
        lcm_tools, "resolve_provider", lambda _config: BlockingProvider()
    )
    try:
        payload = json.loads(
            lcm_tools.lcm_grep(
                {"query": "daemon", "mode": "semantic"},
                engine=semantic_engine,
            )
        )

        live_workers = [
            thread
            for thread in threading.enumerate()
            if thread.name == "lcm-query-embed" and thread.is_alive()
        ]
        assert payload["timeout"] is True
        # The interactive timeout is the remaining absolute budget (~0.02s),
        # computed from a monotonic clock so a few microseconds may have elapsed.
        assert observed_timeouts and observed_timeouts[0] == pytest.approx(
            0.02, abs=0.01
        )
        assert live_workers
        assert all(thread.daemon for thread in live_workers)
    finally:
        release.set()
        for thread in threading.enumerate():
            if thread.name == "lcm-query-embed":
                thread.join(timeout=0.2)


def test_semantic_missing_provider_degrades_to_fts(semantic_engine, monkeypatch):
    semantic_engine._store.append(
        "session-a",
        {"role": "user", "content": "local fallback marker"},
    )
    monkeypatch.setattr(lcm_tools, "resolve_provider", lambda _config: None)

    payload = json.loads(
        lcm_tools.lcm_grep(
            {"query": "marker", "mode": "semantic"},
            engine=semantic_engine,
        )
    )

    assert payload["degraded_to_fts"] is True
    assert payload["results"][0]["type"] == "message"
    assert "not configured" in payload["degraded_reason"]


def test_semantic_capacity_degrades_using_independent_full_text_workers(
    semantic_engine, monkeypatch
):
    semantic_engine._store.append(
        "session-a",
        {"role": "user", "content": "capacity fallback marker"},
    )
    semantic_slots = threading.BoundedSemaphore(1)
    assert semantic_slots.acquire(blocking=False)
    monkeypatch.setattr(lcm_tools, "_lcm_semantic_worker_slots", semantic_slots)
    monkeypatch.setattr(
        lcm_tools,
        "_lcm_full_text_worker_slots",
        threading.BoundedSemaphore(1),
    )
    try:
        payload = json.loads(
            lcm_tools.lcm_grep(
                {"query": "capacity", "mode": "semantic"},
                engine=semantic_engine,
            )
        )
    finally:
        semantic_slots.release()

    assert payload["degraded_to_fts"] is True
    assert "capacity exhausted" in payload["degraded_reason"]
    assert payload["results"][0]["type"] == "message"


@pytest.mark.parametrize("mode", ["semantic", "hybrid"])
def test_none_vector_coverage_degrades_to_fts(
    semantic_engine,
    monkeypatch,
    mode,
):
    semantic_engine._store.append(
        "session-a",
        {"role": "user", "content": "coverage fallback marker"},
    )
    monkeypatch.setattr(lcm_tools, "resolve_provider", lambda _config: MockProvider())

    payload = json.loads(
        lcm_tools.lcm_grep(
            {"query": "coverage", "mode": mode},
            engine=semantic_engine,
        )
    )

    assert payload["mode"] == mode
    assert payload["coverage"] == "none"
    assert payload["degraded_to_fts"] is True
    assert payload["results"][0]["type"] == "message"
    assert "coverage=none" in payload["degraded_reason"]


def test_provider_is_cached_until_provider_or_model_changes(
    semantic_engine,
    monkeypatch,
):
    resolved: list[MockProvider] = []

    def factory(config):
        provider = MockProvider()
        provider.model_id = config.embedding_model
        resolved.append(provider)
        return provider

    monkeypatch.setattr(lcm_tools, "resolve_provider", factory)

    for query in ("first", "second"):
        json.loads(
            lcm_tools.lcm_grep(
                {"query": query, "mode": "semantic"},
                engine=semantic_engine,
            )
        )

    assert len(resolved) == 1
    assert resolved[0].queries == ["first", "second"]

    semantic_engine._config.embedding_model = "changed-model"
    json.loads(
        lcm_tools.lcm_grep(
            {"query": "third", "mode": "semantic"},
            engine=semantic_engine,
        )
    )

    assert len(resolved) == 2
    assert resolved[1].model_id == "changed-model"
    assert resolved[1].queries == ["third"]


def test_semantic_auth_error_is_operator_readable_and_does_not_degrade(
    semantic_engine,
    monkeypatch,
):
    class AuthProvider(MockProvider):
        def embed_query(self, _text):
            raise VoyageError("auth", "bad credentials", status_code=401)

    monkeypatch.setattr(lcm_tools, "resolve_provider", lambda _config: AuthProvider())
    payload = json.loads(
        lcm_tools.lcm_grep(
            {"query": "anything", "mode": "semantic"},
            engine=semantic_engine,
        )
    )

    assert "authentication failed" in payload["error"].lower()
    assert "degraded_to_fts" not in payload


def test_hybrid_semantic_timeout_keeps_completed_full_text_results(
    semantic_engine, monkeypatch
):
    completed_fts = {
        "query": "needle",
        "mode": "hybrid",
        "results": [{"type": "message", "store_id": 7, "snippet": "fts hit"}],
        "total_results": 1,
    }
    monkeypatch.setattr(
        lcm_tools,
        "_lcm_grep_full_text_with_deadline",
        lambda *_args, **_kwargs: completed_fts,
    )
    monkeypatch.setattr(
        lcm_tools,
        "_lcm_grep_semantic",
        lambda *_args, **_kwargs: {
            "error": "lcm_grep request deadline exceeded",
            "mode": "hybrid",
            "timeout": True,
            "timeout_stage": "provider_resolution",
        },
    )

    payload = json.loads(
        lcm_tools.lcm_grep(
            {"query": "needle", "mode": "hybrid"},
            engine=semantic_engine,
        )
    )

    assert payload["degraded_to_fts"] is True
    assert payload["results"] == completed_fts["results"]
    assert "deadline" in payload["degraded_reason"]


def test_hybrid_semantic_auth_error_remains_operator_readable(
    semantic_engine, monkeypatch
):
    semantic_engine._store.append(
        "session-a", {"role": "user", "content": "authentication marker"}
    )

    class AuthProvider(MockProvider):
        def embed_query(self, _text):
            raise VoyageError("auth", "bad credentials", status_code=401)

    monkeypatch.setattr(lcm_tools, "resolve_provider", lambda _config: AuthProvider())
    payload = json.loads(
        lcm_tools.lcm_grep(
            {"query": "authentication", "mode": "hybrid"},
            engine=semantic_engine,
        )
    )

    assert "authentication failed" in payload["error"].lower()
    assert "degraded_to_fts" not in payload


def test_hybrid_rrf_deduplicates_nodes_and_rewards_both_arms(
    semantic_engine,
    monkeypatch,
):
    message_id = semantic_engine._store.append(
        "session-a",
        {"role": "user", "content": "fusion appears in a raw record"},
    )
    both_node = _add_summary(
        semantic_engine,
        "fusion appears in this embedded summary",
        created_at=2.0,
        source_ids=[message_id],
    )
    semantic_only = _add_summary(
        semantic_engine,
        "conceptual neighbor without the lexical term",
        created_at=1.0,
    )
    _seed_vectors(
        semantic_engine,
        [(both_node, [1.0, 0.0]), (semantic_only, [0.8, 0.6])],
    )
    monkeypatch.setattr(lcm_tools, "resolve_provider", lambda _config: MockProvider())

    payload = json.loads(
        lcm_tools.lcm_grep(
            {"query": "fusion", "mode": "hybrid", "sort": "relevance", "limit": 10},
            engine=semantic_engine,
        )
    )

    node_hits = [hit for hit in payload["results"] if hit.get("node_id") == both_node]
    assert len(node_hits) == 1
    both_hit = node_hits[0]
    raw_hit = next(
        hit for hit in payload["results"] if hit.get("store_id") == message_id
    )
    assert both_hit["fts_rank"] >= 1 and both_hit["semantic_rank"] == 1
    assert both_hit["rrf_score"] == pytest.approx(
        1 / (60 + both_hit["fts_rank"]) + 1 / 61
    )
    assert both_hit["rrf_score"] > raw_hit["rrf_score"]
    assert payload["fusion"] == "rrf"
    assert payload["rrf_k"] == 60


@pytest.mark.parametrize(
    ("limit", "expected_candidates"), [(1, 50), (40, 120), (200, 500)]
)
def test_hybrid_limit_controls_bounded_candidate_overfetch(
    semantic_engine,
    monkeypatch,
    limit,
    expected_candidates,
):
    observed: list[int] = []

    class FakeVectorStore:
        def __init__(self, *_args, **_kwargs):
            pass

        def knn(self, _query, *, k, **_kwargs):
            observed.append(k)
            return KNNResult(coverage="full")

        def close(self):
            pass

    monkeypatch.setattr(lcm_tools, "VectorStore", FakeVectorStore)
    monkeypatch.setattr(lcm_tools, "resolve_provider", lambda _config: MockProvider())

    payload = json.loads(
        lcm_tools.lcm_grep(
            {"query": "candidate cap", "mode": "hybrid", "limit": limit},
            engine=semantic_engine,
        )
    )

    assert "error" not in payload
    assert observed == [expected_candidates]
    assert payload["limit"] == limit


def test_semantic_snippets_are_bounded(semantic_engine, monkeypatch):
    node_id = _add_summary(semantic_engine, "x" * 1_000, created_at=1.0)
    _seed_vectors(semantic_engine, [(node_id, [1.0, 0.0])])
    monkeypatch.setattr(lcm_tools, "resolve_provider", lambda _config: MockProvider())

    payload = json.loads(
        lcm_tools.lcm_grep(
            {"query": "bounded", "mode": "semantic"},
            engine=semantic_engine,
        )
    )

    assert len(payload["results"][0]["snippet"]) == 300


def test_full_text_modes_remain_byte_identical_with_embeddings_on_or_off(
    semantic_engine,
):
    semantic_engine._store.append(
        "session-a",
        {"role": "user", "content": "byte stable history result"},
    )

    for sort in ("recency", "relevance", "hybrid"):
        args = {"query": "stable", "sort": sort}
        semantic_engine._config.embeddings_enabled = False
        disabled = lcm_tools.lcm_grep(args, engine=semantic_engine)
        explicit = lcm_tools.lcm_grep(
            {**args, "mode": "full_text"}, engine=semantic_engine
        )
        semantic_engine._config.embeddings_enabled = True
        enabled = lcm_tools.lcm_grep(args, engine=semantic_engine)
        assert disabled == explicit == enabled


def _add_summary_in(engine, summary, *, session_id, created_at, source_ids=None):
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
            latest_at=created_at,
            expand_hint="",
        )
    )


def test_semantic_role_filter_degrades_to_full_text(semantic_engine, monkeypatch):
    semantic_engine._store.append(
        "session-a", {"role": "user", "content": "role marker"}
    )
    node = _add_summary(semantic_engine, "an embedded summary", created_at=1.0)
    _seed_vectors(semantic_engine, [(node, [1.0, 0.0])])
    monkeypatch.setattr(lcm_tools, "resolve_provider", lambda _config: MockProvider())

    payload = json.loads(
        lcm_tools.lcm_grep(
            {"query": "marker", "mode": "semantic", "role": "user"},
            engine=semantic_engine,
        )
    )

    # role is not enforceable over role-less summaries, so it degrades to
    # full_text (which does enforce role) rather than silently ignoring it.
    assert payload["degraded_to_fts"] is True
    assert "role" in payload["degraded_reason"].lower()
    assert all(hit.get("type") == "message" for hit in payload["results"])


def test_semantic_time_scoped_query_degrades_to_raw_full_text(
    semantic_engine, monkeypatch
):
    newer = _add_summary(semantic_engine, "newer high score", created_at=100.0)
    older = _add_summary(semantic_engine, "older lower score", created_at=1.0)
    _seed_vectors(semantic_engine, [(newer, [1.0, 0.0]), (older, [0.0, 1.0])])
    monkeypatch.setattr(lcm_tools, "resolve_provider", lambda _config: MockProvider())

    # time_from/time_to advertise raw-message hits only (schemas.LCM_GREP), and
    # full_text omits summaries when a time filter is set. The semantic arm,
    # which produces only summary hits, must therefore degrade to the raw
    # full_text path instead of returning time-scoped summary hits.
    payload = json.loads(
        lcm_tools.lcm_grep(
            {"query": "q", "mode": "semantic", "time_to": 50, "limit": 1},
            engine=semantic_engine,
        )
    )

    assert payload["degraded_to_fts"] is True
    assert "time" in payload["degraded_reason"].lower()
    assert all(hit.get("type") != "summary" for hit in payload.get("results", []))


def test_semantic_source_filter_excludes_ineligible_before_top_k(
    semantic_engine, monkeypatch
):
    keep_msg = semantic_engine._store.append(
        "session-a", {"role": "user", "content": "k"}, source="keep-src"
    )
    drop_msg = semantic_engine._store.append(
        "session-a", {"role": "user", "content": "d"}, source="drop-src"
    )
    keep = _add_summary(
        semantic_engine, "keep summary", created_at=1.0, source_ids=[keep_msg]
    )
    drop = _add_summary(
        semantic_engine, "drop summary", created_at=2.0, source_ids=[drop_msg]
    )
    # drop scores highest but its source is excluded, so it must not take the slot.
    _seed_vectors(semantic_engine, [(keep, [0.0, 1.0]), (drop, [1.0, 0.0])])
    monkeypatch.setattr(lcm_tools, "resolve_provider", lambda _config: MockProvider())

    payload = json.loads(
        lcm_tools.lcm_grep(
            {"query": "q", "mode": "semantic", "source": "keep-src", "limit": 1},
            engine=semantic_engine,
        )
    )

    assert [hit["node_id"] for hit in payload["results"]] == [keep]


def test_semantic_broad_scope_degrades_to_raw_full_text(semantic_engine, monkeypatch):
    semantic_engine._store.append(
        "session-a", {"role": "user", "content": "c"}, conversation_id="conv-1"
    )
    in_conv = _add_summary(semantic_engine, "in conversation", created_at=1.0)
    other = _add_summary_in(
        semantic_engine, "other session", session_id="session-b", created_at=2.0
    )
    _seed_vectors(semantic_engine, [(in_conv, [0.0, 1.0]), (other, [1.0, 0.0])])
    monkeypatch.setattr(lcm_tools, "resolve_provider", lambda _config: MockProvider())

    # Broader scopes ('all'/'session') return raw-message hits only. A summary
    # is cross-session/unexpandable, so semantic degrades to full_text rather
    # than emit cross-session summary hits.
    payload = json.loads(
        lcm_tools.lcm_grep(
            {
                "query": "q",
                "mode": "semantic",
                "session_scope": "all",
                "conversation_id": "conv-1",
                "limit": 1,
            },
            engine=semantic_engine,
        )
    )

    assert payload["degraded_to_fts"] is True
    assert "broader scopes" in payload["degraded_reason"]
    assert all(hit.get("type") != "summary" for hit in payload.get("results", []))


def test_semantic_conversation_filter_degrades_to_raw_full_text(
    semantic_engine, monkeypatch
):
    semantic_engine._store.append(
        "session-a", {"role": "user", "content": "c"}, conversation_id="conv-1"
    )
    in_conv = _add_summary(semantic_engine, "in conversation", created_at=1.0)
    _seed_vectors(semantic_engine, [(in_conv, [0.0, 1.0])])
    monkeypatch.setattr(lcm_tools, "resolve_provider", lambda _config: MockProvider())

    # A conversation lane maps to raw messages; a summary can aggregate multiple
    # lanes within one session, so semantic degrades to the raw full_text path
    # (which filters messages by conversation_id at the row level) rather than
    # leak wrong-lane summary hits.
    payload = json.loads(
        lcm_tools.lcm_grep(
            {
                "query": "q",
                "mode": "semantic",
                "conversation_id": "conv-1",
                "limit": 1,
            },
            engine=semantic_engine,
        )
    )

    assert payload["degraded_to_fts"] is True
    assert "conversation" in payload["degraded_reason"].lower()
    assert all(hit.get("type") != "summary" for hit in payload.get("results", []))


def test_slow_knn_degrades_within_total_budget(semantic_engine, monkeypatch):
    semantic_engine._config.embedding_query_timeout_s = 0.05
    semantic_engine._store.append(
        "session-a", {"role": "user", "content": "needle for fallback"}
    )

    class SlowVectorStore:
        def __init__(self, *_args, **_kwargs):
            pass

        def knn(self, *_args, **_kwargs):
            time.sleep(0.3)
            return KNNResult(coverage="full")

        def close(self):
            pass

    monkeypatch.setattr(lcm_tools, "VectorStore", SlowVectorStore)
    monkeypatch.setattr(lcm_tools, "resolve_provider", lambda _config: MockProvider())

    started = time.monotonic()
    payload = json.loads(
        lcm_tools.lcm_grep(
            {"query": "needle", "mode": "semantic"}, engine=semantic_engine
        )
    )
    elapsed = time.monotonic() - started

    # The whole operation returns within the tiny budget and does not begin a
    # fallback after the KNN consumes the remaining time.
    assert elapsed < 0.2
    assert payload["timeout"] is True


def test_provider_resolution_is_bounded_and_does_not_start_query_or_fallback(
    semantic_engine, monkeypatch
):
    semantic_engine._config.embedding_query_timeout_s = 0.02
    query_calls = 0
    fallback_calls = 0

    class CountingProvider(MockProvider):
        def embed_query(self, text):
            nonlocal query_calls
            query_calls += 1
            return super().embed_query(text)

    def slow_resolve(_config):
        time.sleep(0.1)
        return CountingProvider()

    def counted_full_text(_args, **_kwargs):
        nonlocal fallback_calls
        fallback_calls += 1
        return json.dumps({"results": []})

    monkeypatch.setattr(lcm_tools, "resolve_provider", slow_resolve)
    monkeypatch.setattr(lcm_tools, "_lcm_grep_full_text", counted_full_text)
    started = time.monotonic()
    payload = json.loads(
        lcm_tools.lcm_grep(
            {"query": "deadline", "mode": "semantic"}, engine=semantic_engine
        )
    )

    assert time.monotonic() - started < 0.08
    assert payload["timeout"] is True
    assert payload["timeout_stage"] == "provider_resolution"
    assert query_calls == 0
    assert fallback_calls == 0


def test_hybrid_does_not_start_semantic_arm_after_fts_exhausts_deadline(
    semantic_engine, monkeypatch
):
    semantic_engine._config.embedding_query_timeout_s = 0.02
    provider_calls = 0

    def slow_full_text(_args, **_kwargs):
        time.sleep(0.1)
        return json.dumps({"results": []})

    def resolve(_config):
        nonlocal provider_calls
        provider_calls += 1
        return MockProvider()

    monkeypatch.setattr(lcm_tools, "_lcm_grep_full_text", slow_full_text)
    monkeypatch.setattr(lcm_tools, "resolve_provider", resolve)
    started = time.monotonic()
    payload = json.loads(
        lcm_tools.lcm_grep(
            {"query": "deadline", "mode": "hybrid"}, engine=semantic_engine
        )
    )

    assert time.monotonic() - started < 0.08
    assert payload["timeout"] is True
    assert payload["mode"] == "hybrid"
    assert provider_calls == 0


def test_full_text_setup_expiry_does_not_start_search(semantic_engine, monkeypatch):
    semantic_engine._config.embedding_query_timeout_s = 0.02
    search_calls = 0
    release = threading.Event()
    opened = []

    class LateConnection:
        closed = False

        def close(self):
            self.closed = True

    def blocking_connect(*args, **kwargs):
        release.wait(1.0)
        connection = LateConnection()
        opened.append(connection)
        return connection

    def counted_full_text(_args, **_kwargs):
        nonlocal search_calls
        search_calls += 1
        return json.dumps({"results": []})

    monkeypatch.setattr(lcm_tools.sqlite3, "connect", blocking_connect)
    monkeypatch.setattr(lcm_tools, "_lcm_grep_full_text", counted_full_text)
    semantic_engine._config.embeddings_enabled = False
    try:
        payload = json.loads(
            lcm_tools.lcm_grep(
                {"query": "deadline", "mode": "semantic"}, engine=semantic_engine
            )
        )
        assert payload["timeout"] is True
        assert search_calls == 0
    finally:
        release.set()
        for thread in threading.enumerate():
            if thread.name == "lcm-full-text":
                thread.join(timeout=0.2)
    assert len(opened) == 1
    assert opened[0].closed is True


def test_result_hydration_is_inside_request_deadline(semantic_engine, monkeypatch):
    semantic_engine._config.embedding_query_timeout_s = 0.02

    class OneResultStore:
        def __init__(self, *_args, **_kwargs):
            pass

        def knn(self, *_args, **_kwargs):
            return KNNResult([("1", 1.0, "summary")], coverage="full")

        def close(self):
            pass

    release = threading.Event()

    def blocking_get_node(_node_id):
        release.wait(1.0)
        return None

    monkeypatch.setattr(lcm_tools, "VectorStore", OneResultStore)
    monkeypatch.setattr(lcm_tools, "resolve_provider", lambda _config: MockProvider())
    monkeypatch.setattr(semantic_engine._dag, "get_node", blocking_get_node)
    try:
        started = time.monotonic()
        payload = json.loads(
            lcm_tools.lcm_grep(
                {"query": "deadline", "mode": "semantic"}, engine=semantic_engine
            )
        )
        assert time.monotonic() - started < 0.08
        assert payload["timeout"] is True
        assert payload["timeout_stage"] == "result_resolution"
    finally:
        release.set()
        for thread in threading.enumerate():
            if thread.name == "lcm-result-hydration":
                thread.join(timeout=0.2)


def test_result_hydration_path_expiry_never_starts_database_connection(
    semantic_engine, monkeypatch
):
    semantic_engine._config.embedding_query_timeout_s = 0.02

    class OneResultStore:
        def __init__(self, *_args, **_kwargs):
            pass

        def knn(self, *_args, **_kwargs):
            return KNNResult([("1", 1.0, "summary")], coverage="full")

        def close(self):
            pass

    real_resolve = lcm_tools.Path.resolve
    connect_calls = 0

    def slow_resolve(path, *args, **kwargs):
        time.sleep(0.05)
        return real_resolve(path, *args, **kwargs)

    def counted_connect(*_args, **_kwargs):
        nonlocal connect_calls
        connect_calls += 1
        raise AssertionError("database connection started after path deadline")

    monkeypatch.setattr(lcm_tools, "VectorStore", OneResultStore)
    monkeypatch.setattr(lcm_tools, "resolve_provider", lambda _config: MockProvider())
    monkeypatch.setattr(lcm_tools.Path, "resolve", slow_resolve)
    monkeypatch.setattr(lcm_tools.sqlite3, "connect", counted_connect)

    payload = json.loads(
        lcm_tools.lcm_grep(
            {"query": "deadline", "mode": "semantic"}, engine=semantic_engine
        )
    )
    assert payload["timeout"] is True
    assert payload["timeout_stage"] == "result_resolution"
    # The bounded worker may still be finishing Path.resolve. Wait for it to
    # run the post-resolve deadline gate and prove connect never starts later.
    time.sleep(0.06)
    assert connect_calls == 0


def test_recall_eval_is_deterministic_and_hybrid_beats_fts_on_paraphrases():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "eval_retrieval_recall.py"
    first = subprocess.run(
        [sys.executable, str(script), "--json"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    second = subprocess.run(
        [sys.executable, str(script), "--json"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    assert first == second
    metrics = json.loads(first)
    assert (
        metrics["hybrid"]["paraphrase"]["recall@5"]
        >= metrics["full_text"]["paraphrase"]["recall@5"]
    )
    assert (
        metrics["hybrid"]["paraphrase"]["recall@10"]
        >= metrics["full_text"]["paraphrase"]["recall@10"]
    )


def test_semantic_content_scope_degrades_to_full_text(
    semantic_engine, monkeypatch, tmp_path
):
    """content_scope beyond 'history' is a payload-search dimension owned by
    the full-text arm; payloads are never embedded, so the semantic arm must
    degrade rather than silently return history-only semantic hits (combined
    contract with the externalized-payload-search train)."""
    # On a combined head the degraded full-text arm actually runs the payload
    # scan, which requires externalization to be enabled (disabled -> honest
    # error instead of degrade markers). Enable it so the degrade contract is
    # what gets asserted on every head this test runs against.
    monkeypatch.setattr(
        semantic_engine._config, "large_output_externalization_enabled", True
    )
    # The payload scan resolves its directory from the engine home; the
    # lightweight fixture engine needs one on combined heads (real engines
    # always have it).
    monkeypatch.setattr(semantic_engine, "_hermes_home", tmp_path, raising=False)
    semantic_engine._store.append(
        "session-a", {"role": "user", "content": "payload marker"}
    )
    node = _add_summary(semantic_engine, "an embedded summary", created_at=1.0)
    _seed_vectors(semantic_engine, [(node, [1.0, 0.0])])
    monkeypatch.setattr(lcm_tools, "resolve_provider", lambda _config: MockProvider())

    for scope in ("externalized", "both"):
        payload = json.loads(
            lcm_tools.lcm_grep(
                {"query": "marker", "mode": "semantic", "content_scope": scope},
                engine=semantic_engine,
            )
        )
        assert payload["degraded_to_fts"] is True, scope
        assert "full_text" in payload["degraded_reason"], scope
        # No semantic summary hit may leak through the requested payload scope.
        assert all(hit.get("type") != "summary" for hit in payload["results"]), scope


def test_hybrid_content_scope_degrades_to_full_text_arm(
    semantic_engine, monkeypatch, tmp_path
):
    """In hybrid mode the semantic arm's content_scope degrade must surface as
    the full-text-arm result (which owns payload scanning) plus the explicit
    degraded marker — never fused history-only semantic hits."""
    monkeypatch.setattr(
        semantic_engine._config, "large_output_externalization_enabled", True
    )
    # The payload scan resolves its directory from the engine home; the
    # lightweight fixture engine needs one on combined heads (real engines
    # always have it).
    monkeypatch.setattr(semantic_engine, "_hermes_home", tmp_path, raising=False)
    semantic_engine._store.append(
        "session-a", {"role": "user", "content": "payload marker"}
    )
    node = _add_summary(semantic_engine, "an embedded summary", created_at=1.0)
    _seed_vectors(semantic_engine, [(node, [1.0, 0.0])])
    monkeypatch.setattr(lcm_tools, "resolve_provider", lambda _config: MockProvider())

    payload = json.loads(
        lcm_tools.lcm_grep(
            {"query": "marker", "mode": "hybrid", "content_scope": "externalized"},
            engine=semantic_engine,
        )
    )
    assert payload["mode"] == "hybrid"
    assert payload["degraded_to_fts"] is True
    assert "full_text" in payload["degraded_reason"]
    assert all(hit.get("type") != "summary" for hit in payload["results"])
