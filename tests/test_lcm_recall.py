"""Tests for lcm_recall — the cross-conversation forever-memory recall tool.

Seeds summaries + chunks + raw messages across three synthetic sessions and
asserts the fused pipeline recalls cross-session WITHOUT a session filter, that
scope_bias and recency are soft ranking boosts (never filters), that chunk hits
dedupe against FTS by store_id, that rerank failures skip silently, and that the
degrade matrix (embeddings-off) still returns the FTS arm.
"""
from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

import hermes_lcm.tools as lcm_tools
from hermes_lcm import store as store_module
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


def test_recall_fts_finds_oldest_matching_message_beyond_candidate_window(
    recall_engine, monkeypatch
):
    recall_engine._config.embeddings_enabled = False
    oldest = recall_engine._store.append(
        "session-old",
        {"role": "user", "content": "cobalt-orchid immutable recovery note"},
    )
    for index in range(75):
        recall_engine._store.append(
            f"session-new-{index}",
            {"role": "user", "content": f"ordinary recent note {index}"},
        )

    payload = _recall(
        recall_engine,
        monkeypatch,
        provider=MockProvider(),
        query="cobalt orchid",
        include="verbatim",
        limit=1,
    )

    assert payload["provenance"]["coverage"]["fts"] == "full"
    assert payload["hits"][0]["store_id"] == oldest


def test_answer_ready_backfills_past_malformed_and_missing_references(
    recall_engine, monkeypatch
):
    recall_engine._config.embeddings_enabled = False
    stale_span_id = recall_engine._store.append(
        "session-stale",
        {"role": "user", "content": "stale span must not cite this offset-zero text"},
    )
    fallback_id = recall_engine._store.append(
        "session-fallback",
        {
            "role": "user",
            "content": "kanban dashboard sprint exact citable fallback from durable storage",
        },
    )
    malformed_hits = [
        {
            "kind": "message_excerpt",
            "store_id": "not-an-integer",
            "session_id": "session-bad",
            "snippet": "must not escape",
            "timestamp": 3.0,
        },
        {
            "kind": "message_excerpt",
            "store_id": 999_999,
            "session_id": "session-missing",
            "snippet": "must not cite a missing row",
            "timestamp": 2.0,
        },
        {
            "kind": "message_excerpt",
            "store_id": stale_span_id,
            "session_id": "forged-session-id",
            "snippet": "untrusted preview",
            "timestamp": 1.0,
            "chunk_span": {"char_start": "broken", "char_end": -1},
        },
        {
            "kind": "message_excerpt",
            "store_id": fallback_id,
            "session_id": "forged-session-id",
            "snippet": "untrusted preview",
            "timestamp": 0.5,
        },
    ]
    monkeypatch.setattr(
        lcm_tools,
        "_lcm_recall_fts_arm",
        lambda *_args, **_kwargs: (malformed_hits, None),
    )

    payload = _recall(
        recall_engine,
        monkeypatch,
        include="verbatim",
        detail="answer_ready",
        limit=1,
    )

    assert payload["detail"] == "answer_ready"
    assert payload["total_results"] == 1
    assert payload["hits"] == [
        {
            "kind": "message_excerpt",
            "store_id": fallback_id,
            "session_id": "session-fallback",
            "timestamp": payload["hits"][0]["timestamp"],
            "snippet": "kanban dashboard sprint exact citable fallback from durable storage",
            "score": payload["hits"][0]["score"],
            "expand_hint": f"lcm_expand(store_id={fallback_id}, content_offset=0)",
            "from_current_session": False,
            "arms": ["fts"],
            "citation": {
                "tool": "lcm_expand",
                "store_id": fallback_id,
                "content_offset": 0,
            },
        }
    ]
    assert payload["provenance"]["reference_strict"] == {
        "enabled": True,
        "omitted_uncitable": 3,
    }


def test_answer_ready_hydrates_summary_to_exact_raw_source(recall_engine, monkeypatch):
    source_id = recall_engine._store.append(
        "session-source",
        {
            "role": "assistant",
            "content": "raw kanban dashboard sprint source that supports the memory",
        },
    )
    node_id = recall_engine._dag.add_node(
        SummaryNode(
            session_id="session-summary",
            depth=0,
            summary="summary prose must not be presented as a verbatim citation",
            token_count=10,
            source_token_count=20,
            source_ids=[source_id],
            source_type="messages",
            created_at=1.0,
        )
    )
    monkeypatch.setattr(lcm_tools, "_lcm_recall_fts_arm", lambda *_a, **_k: ([], None))
    monkeypatch.setattr(
        lcm_tools,
        "_lcm_recall_summary_arm",
        lambda *_a, **_k: (
            [
                {
                    "kind": "summary",
                    "node_id": node_id,
                    "session_id": "session-summary",
                    "snippet": "summary prose must not be cited directly",
                    "timestamp": 1.0,
                    "from_current_session": False,
                }
            ],
            "full",
            1,
            1,
        ),
    )

    payload = _recall(
        recall_engine,
        monkeypatch,
        include="summaries",
        detail="answer_ready",
        limit=1,
    )

    assert payload["hits"][0]["kind"] == "message_excerpt"
    assert payload["hits"][0]["store_id"] == source_id
    assert payload["hits"][0]["snippet"] == (
        "raw kanban dashboard sprint source that supports the memory"
    )
    assert payload["hits"][0]["source_node_id"] == node_id


def test_answer_ready_summary_selects_query_relevant_source(recall_engine, monkeypatch):
    unrelated_id = recall_engine._store.append(
        "session-source",
        {"role": "assistant", "content": "the weather forecast is clear tomorrow"},
    )
    relevant_id = recall_engine._store.append(
        "session-source",
        {"role": "assistant", "content": "the cobalt orchid launch code is seven"},
    )
    node_id = recall_engine._dag.add_node(
        SummaryNode(
            session_id="session-summary",
            depth=0,
            summary="The cobalt orchid launch code is seven.",
            token_count=10,
            source_token_count=20,
            source_ids=[unrelated_id, relevant_id],
            source_type="messages",
            created_at=1.0,
        )
    )
    monkeypatch.setattr(lcm_tools, "_lcm_recall_fts_arm", lambda *_a, **_k: ([], None))
    monkeypatch.setattr(
        lcm_tools,
        "_lcm_recall_summary_arm",
        lambda *_a, **_k: (
            [{
                "kind": "summary",
                "node_id": node_id,
                "session_id": "session-summary",
                "snippet": "The cobalt orchid launch code is seven.",
                "timestamp": 1.0,
                "from_current_session": False,
            }],
            "full",
            1,
            1,
        ),
    )

    payload = _recall(
        recall_engine,
        monkeypatch,
        query="cobalt orchid launch code",
        include="summaries",
        detail="answer_ready",
        limit=1,
    )

    assert payload["hits"][0]["store_id"] == relevant_id
    assert payload["hits"][0]["snippet"] == "the cobalt orchid launch code is seven"
    assert payload["hits"][0]["source_node_id"] == node_id


def test_answer_ready_summary_selects_source_relevant_to_matched_fact(
    recall_engine, monkeypatch
):
    unrelated_id = recall_engine._store.append(
        "session-source",
        {"role": "assistant", "content": "weather forecast clear tomorrow"},
    )
    relevant_id = recall_engine._store.append(
        "session-source",
        {"role": "assistant", "content": "the launch vehicle uses a methane engine"},
    )
    node_id = recall_engine._dag.add_node(
        SummaryNode(
            session_id="session-summary",
            depth=0,
            summary="The launch vehicle uses a methane engine.",
            token_count=10,
            source_token_count=20,
            source_ids=[unrelated_id, relevant_id],
            source_type="messages",
            created_at=1.0,
        )
    )
    monkeypatch.setattr(lcm_tools, "_lcm_recall_fts_arm", lambda *_a, **_k: ([], None))
    monkeypatch.setattr(
        lcm_tools,
        "_lcm_recall_summary_arm",
        lambda *_a, **_k: (
            [{
                "kind": "summary",
                "node_id": node_id,
                "session_id": "session-summary",
                "snippet": "The launch vehicle uses a methane engine.",
                "timestamp": 1.0,
                "from_current_session": False,
            }],
            "full",
            1,
            1,
        ),
    )

    payload = _recall(
        recall_engine,
        monkeypatch,
        query="rocket propulsion",
        include="summaries",
        detail="answer_ready",
        limit=1,
    )

    assert payload["hits"][0]["store_id"] == relevant_id


def test_answer_ready_summary_uses_matched_fact_beyond_display_preview(
    recall_engine, monkeypatch
):
    preview_source_id = recall_engine._store.append(
        "session-source",
        {"role": "assistant", "content": "archive"},
    )
    supporting_id = recall_engine._store.append(
        "session-source",
        {
            "role": "assistant",
            "content": "the launch vehicle uses a methane engine",
        },
    )
    prefix = "archive " * 40
    fact = "the launch vehicle uses a methane engine"
    summary = prefix + fact
    assert len(prefix) > 300
    node_id = recall_engine._dag.add_node(
        SummaryNode(
            session_id="session-summary",
            depth=0,
            summary=summary,
            token_count=50,
            source_token_count=100,
            source_ids=[preview_source_id, supporting_id],
            source_type="messages",
            created_at=1.0,
        )
    )
    _seed_summary_vectors(recall_engine, [(node_id, [1.0, 0.0])])

    payload = _recall(
        recall_engine,
        monkeypatch,
        query="rocket propulsion",
        include="summaries",
        detail="answer_ready",
        limit=1,
    )

    assert payload["hits"][0]["store_id"] == supporting_id
    assert payload["hits"][0]["snippet"] == fact
    assert payload["hits"][0]["source_node_id"] == node_id


def test_answer_ready_summary_support_outranks_incidental_query_overlap(
    recall_engine, monkeypatch
):
    incidental_id = recall_engine._store.append(
        "session-source",
        {"role": "assistant", "content": "What"},
    )
    supporting_id = recall_engine._store.append(
        "session-source",
        {
            "role": "assistant",
            "content": "The launch vehicle uses a methane engine.",
        },
    )
    node_id = recall_engine._dag.add_node(
        SummaryNode(
            session_id="session-summary",
            depth=0,
            summary="The launch vehicle uses a methane engine.",
            token_count=10,
            source_token_count=20,
            source_ids=[incidental_id, supporting_id],
            source_type="messages",
            created_at=1.0,
        )
    )
    monkeypatch.setattr(lcm_tools, "_lcm_recall_fts_arm", lambda *_a, **_k: ([], None))
    monkeypatch.setattr(
        lcm_tools,
        "_lcm_recall_summary_arm",
        lambda *_a, **_k: ([{
            "kind": "summary",
            "node_id": node_id,
            "session_id": "session-summary",
            "snippet": "The launch vehicle uses a methane engine.",
            "timestamp": 1.0,
            "from_current_session": False,
        }], "full", 1, 1),
    )

    payload = _recall(
        recall_engine,
        monkeypatch,
        query="What powers rocket",
        include="summaries",
        detail="answer_ready",
        limit=1,
    )

    assert payload["hits"][0]["store_id"] == supporting_id
    assert payload["hits"][0]["snippet"].startswith(
        "The launch vehicle uses a methane engine."
    )
    assert payload["hits"][0]["source_node_id"] == node_id


def test_answer_ready_fts_citation_preserves_late_query_match_offset(
    recall_engine, monkeypatch
):
    recall_engine._config.embeddings_enabled = False
    prefix = "unrelated archive padding " * 16
    fact = "cobalt orchid launch code is seven"
    content = prefix + fact + " afterword"
    store_id = recall_engine._store.append(
        "session-source",
        {"role": "assistant", "content": content},
    )
    monkeypatch.setattr(
        lcm_tools,
        "_lcm_recall_fts_arm",
        lambda *_a, **_k: ([{
            "kind": "message_excerpt",
            "store_id": store_id,
            "session_id": "session-source",
            "snippet": fact,
            "timestamp": 1.0,
        }], None),
    )

    payload = _recall(
        recall_engine,
        monkeypatch,
        query="cobalt orchid launch code",
        include="verbatim",
        detail="answer_ready",
        limit=1,
    )

    expected_offset = content.index("cobalt orchid launch code")
    assert expected_offset > 300
    assert payload["hits"][0]["snippet"].startswith(fact)
    assert payload["hits"][0]["citation"]["content_offset"] == expected_offset
    assert payload["hits"][0]["expand_hint"] == (
        f"lcm_expand(store_id={store_id}, content_offset={expected_offset})"
    )


def test_answer_ready_implicit_and_backfills_when_terms_do_not_fit_one_window(
    recall_engine, monkeypatch
):
    recall_engine._config.embeddings_enabled = False
    fallback_id = recall_engine._store.append(
        "session-fallback",
        {"role": "assistant", "content": "alpha omega verified fallback"},
    )
    incomplete_id = recall_engine._store.append(
        "session-incomplete",
        {"role": "assistant", "content": "alpha" + (" " * 400) + "omega"},
    )

    payload = _recall(
        recall_engine,
        monkeypatch,
        query="alpha omega",
        include="verbatim",
        detail="answer_ready",
        limit=1,
    )

    assert payload["hits"][0]["store_id"] == fallback_id
    assert payload["hits"][0]["store_id"] != incomplete_id
    assert "alpha" in payload["hits"][0]["snippet"]
    assert "omega" in payload["hits"][0]["snippet"]
    assert payload["provenance"]["reference_strict"]["omitted_uncitable"] == 1


def test_answer_ready_quoted_phrase_and_term_must_fit_one_window(
    recall_engine, monkeypatch
):
    recall_engine._config.embeddings_enabled = False
    fallback_id = recall_engine._store.append(
        "session-fallback",
        {"role": "assistant", "content": "alpha beta omega verified fallback"},
    )
    incomplete_id = recall_engine._store.append(
        "session-incomplete",
        {"role": "assistant", "content": "alpha beta" + (" " * 400) + "omega"},
    )

    payload = _recall(
        recall_engine,
        monkeypatch,
        query='"alpha beta" omega',
        include="verbatim",
        detail="answer_ready",
        limit=1,
    )

    assert payload["hits"][0]["store_id"] == fallback_id
    assert payload["hits"][0]["store_id"] != incomplete_id
    assert "alpha beta" in payload["hits"][0]["snippet"]
    assert "omega" in payload["hits"][0]["snippet"]
    assert payload["provenance"]["reference_strict"]["omitted_uncitable"] == 1


@pytest.mark.parametrize("match_start", [299, 294])
def test_answer_ready_boundary_match_is_complete_and_offsets_are_truthful(
    recall_engine, monkeypatch, match_start
):
    recall_engine._config.embeddings_enabled = False
    fact = "cobalt orchid launch code is seven"
    content = ("x" * match_start) + fact + " afterword"
    store_id = recall_engine._store.append(
        "session-source",
        {"role": "assistant", "content": content},
    )
    monkeypatch.setattr(
        lcm_tools,
        "_lcm_recall_fts_arm",
        lambda *_a, **_k: ([{
            "kind": "message_excerpt",
            "store_id": store_id,
            "session_id": "session-source",
            "snippet": fact,
            "timestamp": 1.0,
        }], None),
    )

    payload = _recall(
        recall_engine,
        monkeypatch,
        query="cobalt orchid launch code",
        include="verbatim",
        detail="answer_ready",
        limit=1,
    )

    hit = payload["hits"][0]
    offset = hit["citation"]["content_offset"]
    assert fact in hit["snippet"]
    assert content[offset:offset + len(hit["snippet"])] == hit["snippet"]
    assert offset <= match_start
    assert match_start + len("cobalt orchid launch code") <= offset + len(hit["snippet"])


def test_answer_ready_casefold_match_uses_original_character_offset(
    recall_engine, monkeypatch
):
    recall_engine._config.embeddings_enabled = False
    prefix = "Straße " + ("🙂" * 320)
    fact = "cobalt orchid launch code"
    content = prefix + fact
    store_id = recall_engine._store.append(
        "session-source",
        {"role": "assistant", "content": content},
    )
    monkeypatch.setattr(
        lcm_tools,
        "_lcm_recall_fts_arm",
        lambda *_a, **_k: ([{
            "kind": "message_excerpt",
            "store_id": store_id,
            "session_id": "session-source",
            "snippet": fact,
            "timestamp": 1.0,
        }], None),
    )

    payload = _recall(
        recall_engine,
        monkeypatch,
        query="COBALT ORCHID LAUNCH CODE",
        include="verbatim",
        detail="answer_ready",
        limit=1,
    )

    hit = payload["hits"][0]
    expected_offset = content.index(fact)
    assert hit["citation"]["content_offset"] == expected_offset
    assert hit["snippet"].startswith(fact)
    assert content[expected_offset:expected_offset + len(hit["snippet"])] == hit["snippet"]


def test_answer_ready_preserves_exact_emoji_like_match(recall_engine, monkeypatch):
    recall_engine._config.embeddings_enabled = False
    content = "durable 🧠 note"
    store_id = recall_engine._store.append(
        "session-emoji",
        {"role": "user", "content": content},
    )

    payload = _recall(
        recall_engine,
        monkeypatch,
        query="🧠",
        include="verbatim",
        detail="answer_ready",
        limit=1,
    )

    assert payload["hits"][0]["store_id"] == store_id
    assert payload["hits"][0]["snippet"] == content
    assert payload["hits"][0]["citation"]["content_offset"] == 0


def test_answer_ready_emoji_sequence_uses_original_late_character_offset(
    recall_engine, monkeypatch
):
    recall_engine._config.embeddings_enabled = False
    prefix = "Straße " + ("🙂" * 320)
    query = "👩‍💻"
    content = prefix + query + " durable note"
    store_id = recall_engine._store.append(
        "session-emoji",
        {"role": "user", "content": content},
    )

    payload = _recall(
        recall_engine,
        monkeypatch,
        query=query,
        include="verbatim",
        detail="answer_ready",
        limit=1,
    )

    hit = payload["hits"][0]
    expected_offset = content.index(query)
    assert expected_offset > 300
    assert hit["store_id"] == store_id
    assert hit["citation"]["content_offset"] == expected_offset
    assert hit["snippet"].startswith(query)
    assert content[expected_offset:expected_offset + len(hit["snippet"])] == hit["snippet"]


def test_answer_ready_emoji_sequence_requires_verified_exact_match(
    recall_engine, monkeypatch
):
    recall_engine._config.embeddings_enabled = False
    store_id = recall_engine._store.append(
        "session-emoji",
        {"role": "user", "content": "durable 👩 note without the joined sequence"},
    )
    monkeypatch.setattr(
        lcm_tools,
        "_lcm_recall_fts_arm",
        lambda *_a, **_k: ([{
            "kind": "message_excerpt",
            "store_id": store_id,
            "session_id": "session-emoji",
            "snippet": "forged 👩‍💻 preview",
            "timestamp": 1.0,
        }], None),
    )

    payload = _recall(
        recall_engine,
        monkeypatch,
        query="👩‍💻",
        include="verbatim",
        detail="answer_ready",
        limit=1,
    )

    assert payload["hits"] == []
    assert payload["provenance"]["reference_strict"]["omitted_uncitable"] == 1


def test_answer_ready_long_unrelated_chunk_span_is_omitted_and_backfilled(
    recall_engine, monkeypatch
):
    unrelated_id = recall_engine._store.append(
        "session-unrelated",
        {"role": "assistant", "content": "x" * 600},
    )
    fallback_id = recall_engine._store.append(
        "session-fallback",
        {"role": "assistant", "content": "cobalt orchid verified fallback"},
    )
    monkeypatch.setattr(
        lcm_tools,
        "_lcm_recall_fts_arm",
        lambda *_a, **_k: ([
            {
                "kind": "message_excerpt",
                "store_id": unrelated_id,
                "session_id": "session-unrelated",
                "snippet": "forged cobalt orchid preview",
                "timestamp": 2.0,
                "chunk_span": {"char_start": 0, "char_end": 600},
            },
            {
                "kind": "message_excerpt",
                "store_id": fallback_id,
                "session_id": "session-fallback",
                "snippet": "cobalt orchid verified fallback",
                "timestamp": 1.0,
            },
        ], None),
    )

    payload = _recall(
        recall_engine,
        monkeypatch,
        query="cobalt orchid",
        include="verbatim",
        detail="answer_ready",
        limit=1,
    )

    assert payload["hits"][0]["store_id"] == fallback_id
    assert payload["provenance"]["reference_strict"]["omitted_uncitable"] == 1


def test_answer_ready_summary_citation_preserves_late_matched_fact_offset(
    recall_engine, monkeypatch
):
    prefix = "unrelated archive padding " * 16
    fact = "the launch vehicle uses a methane engine"
    content = prefix + fact + " afterword"
    source_id = recall_engine._store.append(
        "session-source",
        {"role": "assistant", "content": content},
    )
    node_id = recall_engine._dag.add_node(
        SummaryNode(
            session_id="session-summary",
            depth=0,
            summary="The launch vehicle uses a methane engine.",
            token_count=10,
            source_token_count=20,
            source_ids=[source_id],
            source_type="messages",
            created_at=1.0,
        )
    )
    monkeypatch.setattr(lcm_tools, "_lcm_recall_fts_arm", lambda *_a, **_k: ([], None))
    monkeypatch.setattr(
        lcm_tools,
        "_lcm_recall_summary_arm",
        lambda *_a, **_k: ([{
            "kind": "summary",
            "node_id": node_id,
            "session_id": "session-summary",
            "snippet": "The launch vehicle uses a methane engine.",
            "timestamp": 1.0,
            "from_current_session": False,
        }], "full", 1, 1),
    )

    payload = _recall(
        recall_engine,
        monkeypatch,
        query="rocket propulsion",
        include="summaries",
        detail="answer_ready",
        limit=1,
    )

    expected_offset = content.index(fact)
    assert expected_offset > 300
    assert payload["hits"][0]["snippet"].startswith(fact)
    assert payload["hits"][0]["citation"]["content_offset"] == expected_offset
    assert payload["hits"][0]["source_node_id"] == node_id


def test_answer_ready_omits_unverified_fts_match_and_backfills(
    recall_engine, monkeypatch
):
    recall_engine._config.embeddings_enabled = False
    unrelated_id = recall_engine._store.append(
        "session-unrelated",
        {"role": "assistant", "content": "weather forecast only"},
    )
    fallback_id = recall_engine._store.append(
        "session-fallback",
        {"role": "assistant", "content": "cobalt orchid verified fallback"},
    )
    monkeypatch.setattr(
        lcm_tools,
        "_lcm_recall_fts_arm",
        lambda *_a, **_k: ([
            {
                "kind": "message_excerpt",
                "store_id": unrelated_id,
                "session_id": "session-unrelated",
                "snippet": "forged cobalt orchid preview",
                "timestamp": 2.0,
            },
            {
                "kind": "message_excerpt",
                "store_id": fallback_id,
                "session_id": "session-fallback",
                "snippet": "cobalt orchid verified fallback",
                "timestamp": 1.0,
            },
        ], None),
    )

    payload = _recall(
        recall_engine,
        monkeypatch,
        query="cobalt orchid",
        include="verbatim",
        detail="answer_ready",
        limit=1,
    )

    assert payload["hits"][0]["store_id"] == fallback_id
    assert payload["provenance"]["reference_strict"]["omitted_uncitable"] == 1


def test_answer_ready_summary_without_relevant_source_backfills(
    recall_engine, monkeypatch
):
    unrelated_id = recall_engine._store.append(
        "session-source",
        {"role": "assistant", "content": "the weather forecast is clear tomorrow"},
    )
    relevant_id = recall_engine._store.append(
        "session-source",
        {"role": "assistant", "content": "cobalt orchid evidence from the later summary"},
    )
    unrelated_node = recall_engine._dag.add_node(
        SummaryNode(
            session_id="session-summary",
            depth=0,
            summary="Cobalt orchid launch code.",
            token_count=10,
            source_token_count=20,
            source_ids=[unrelated_id],
            source_type="messages",
            created_at=2.0,
        )
    )
    relevant_node = recall_engine._dag.add_node(
        SummaryNode(
            session_id="session-summary",
            depth=0,
            summary="Cobalt orchid evidence.",
            token_count=10,
            source_token_count=20,
            source_ids=[relevant_id],
            source_type="messages",
            created_at=1.0,
        )
    )
    monkeypatch.setattr(lcm_tools, "_lcm_recall_fts_arm", lambda *_a, **_k: ([], None))
    monkeypatch.setattr(
        lcm_tools,
        "_lcm_recall_summary_arm",
        lambda *_a, **_k: (
            [
                {
                    "kind": "summary",
                    "node_id": unrelated_node,
                    "session_id": "session-summary",
                    "snippet": "Cobalt orchid launch code.",
                    "timestamp": 2.0,
                    "from_current_session": False,
                },
                {
                    "kind": "summary",
                    "node_id": relevant_node,
                    "session_id": "session-summary",
                    "snippet": "Cobalt orchid evidence.",
                    "timestamp": 1.0,
                    "from_current_session": False,
                },
            ],
            "full",
            2,
            2,
        ),
    )

    payload = _recall(
        recall_engine,
        monkeypatch,
        query="cobalt orchid",
        include="summaries",
        detail="answer_ready",
        limit=1,
    )

    assert payload["hits"][0]["store_id"] == relevant_id
    assert payload["provenance"]["reference_strict"]["omitted_uncitable"] == 1


@pytest.mark.parametrize(
    "span",
    [
        None,
        [],
        {"char_start": "0", "char_end": 8},
        {"char_start": True, "char_end": 8},
        {"char_start": 8, "char_end": 8},
        {"char_start": 0, "char_end": 999},
    ],
)
def test_answer_ready_rejects_malformed_or_stale_chunk_span(
    recall_engine, monkeypatch, span
):
    recall_engine._config.embeddings_enabled = False
    store_id = recall_engine._store.append(
        "session-source",
        {"role": "user", "content": "unrelated offset-zero text then actual evidence"},
    )
    monkeypatch.setattr(
        lcm_tools,
        "_lcm_recall_fts_arm",
        lambda *_a, **_k: ([{
            "kind": "message_excerpt",
            "store_id": store_id,
            "session_id": "session-source",
            "snippet": "forged preview",
            "timestamp": 1.0,
            "chunk_span": span,
        }], None),
    )

    payload = _recall(
        recall_engine,
        monkeypatch,
        include="verbatim",
        detail="answer_ready",
        limit=1,
    )

    assert payload["hits"] == []
    assert payload["provenance"]["reference_strict"]["omitted_uncitable"] == 1


def test_retrieval_flags_restore_bounded_and_snippet_behavior(
    recall_engine, monkeypatch
):
    monkeypatch.setenv("LCM_RECALL_FULL_CORPUS_SCAN_ENABLED", "false")
    monkeypatch.setenv("LCM_RECALL_REFERENCE_STRICT", "false")
    config = LCMConfig.from_env()
    assert config.recall_full_corpus_scan_enabled is False
    assert config.recall_reference_strict is False

    recall_engine._config.recall_reference_strict = False
    recall_engine._config.embeddings_enabled = False
    monkeypatch.setattr(
        lcm_tools,
        "_lcm_recall_fts_arm",
        lambda *_a, **_k: (
            [
                {
                    "kind": "message_excerpt",
                    "store_id": "legacy-unchecked",
                    "session_id": "legacy-session",
                    "snippet": "legacy snippet",
                    "timestamp": 1.0,
                }
            ],
            None,
        ),
    )

    payload = _recall(
        recall_engine,
        monkeypatch,
        include="verbatim",
        detail="answer_ready",
        limit=1,
    )

    assert payload["detail"] == "snippets"
    assert payload["hits"][0]["store_id"] == "legacy-unchecked"
    assert payload["provenance"]["reference_strict"]["enabled"] is False


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


@pytest.mark.parametrize("slow_stage", ["dag", "message"])
def test_answer_ready_hydration_stops_at_deadline_and_reports_timeout(
    recall_engine, monkeypatch, slow_stage
):
    source_id = recall_engine._store.append(
        "session-source",
        {"role": "assistant", "content": "cobalt orchid deadline evidence"},
    )
    node_id = recall_engine._dag.add_node(
        SummaryNode(
            session_id="session-summary",
            depth=0,
            summary="Cobalt orchid deadline evidence.",
            token_count=10,
            source_token_count=20,
            source_ids=[source_id],
            source_type="messages",
            created_at=1.0,
        )
    )
    monkeypatch.setattr(lcm_tools, "_lcm_recall_fts_arm", lambda *_a, **_k: ([], None))
    monkeypatch.setattr(
        lcm_tools,
        "_lcm_recall_summary_arm",
        lambda *_a, **_k: (
            [
                {
                    "kind": "summary",
                    "node_id": node_id,
                    "session_id": "session-summary",
                    "snippet": "Cobalt orchid deadline evidence.",
                    "timestamp": 1.0,
                    "from_current_session": False,
                }
            ],
            "full",
            1,
            1,
        ),
    )
    real_get_node = recall_engine._dag.get_node
    real_get_batch = recall_engine._store.get_batch
    calls = {"dag": 0, "message": 0}

    def slow_get_node(requested_node_id):
        calls["dag"] += 1
        if slow_stage == "dag":
            time.sleep(0.55)
        return real_get_node(requested_node_id)

    def slow_get_batch(store_ids):
        calls["message"] += 1
        if slow_stage == "message":
            time.sleep(0.55)
        return real_get_batch(store_ids)

    monkeypatch.setattr(recall_engine._dag, "get_node", slow_get_node)
    monkeypatch.setattr(recall_engine._store, "get_batch", slow_get_batch)
    recall_engine._config.recall_query_timeout_s = 0.5

    payload = _recall(
        recall_engine,
        monkeypatch,
        query="cobalt orchid",
        include="summaries",
        detail="answer_ready",
        limit=1,
    )

    assert calls[slow_stage] == 1
    if slow_stage == "dag":
        assert calls["message"] == 0
    assert payload["hits"] == []
    assert payload["timeout"] is True
    assert payload["degraded"] is True
    assert "citation hydration timed out" in payload["degraded_reason"]


@pytest.mark.parametrize(
    ("arm", "include", "scan_name"),
    [
        ("summary", "summaries", "run_knn"),
        ("chunk", "verbatim", "run_chunk_knn"),
    ],
)
def test_full_corpus_scan_timeout_is_degraded(
    recall_engine, monkeypatch, arm, include, scan_name
):
    recall_engine._config.recall_full_corpus_scan_enabled = True
    monkeypatch.setattr(lcm_tools, "_lcm_recall_fts_arm", lambda *_a, **_k: ([], None))
    calls = []

    def timeout_scan(*_args, **kwargs):
        calls.append(kwargs)
        raise TimeoutError(f"simulated {arm} full-corpus scan timeout")

    monkeypatch.setattr(lcm_tools, scan_name, timeout_scan)

    payload = _recall(
        recall_engine,
        monkeypatch,
        include=include,
        limit=1,
    )

    assert len(calls) == 1
    assert calls[0]["full_scan"] is True
    assert payload["timeout"] is True
    assert payload["degraded"] is True
    assert payload["provenance"]["coverage"][arm] == "none"
    assert f"{arm} arm timed out" in payload["degraded_reason"]


@pytest.mark.parametrize("failure_path", ["search", "like_fallback"])
def test_recall_reports_full_text_failures_instead_of_full_empty_coverage(
    recall_engine, monkeypatch, failure_path
):
    recall_engine._config.embeddings_enabled = False

    def fail(*_args, **_kwargs):
        raise RuntimeError(f"simulated {failure_path} database failure")

    if failure_path == "search":
        monkeypatch.setattr(recall_engine._store, "search", fail)
        query = "cobalt orchid"
    else:
        monkeypatch.setattr(type(recall_engine._store), "_search_like", fail)
        query = "cobalt/orchid"

    payload = _recall(
        recall_engine,
        monkeypatch,
        query=query,
        include="verbatim",
        limit=1,
    )

    assert payload["hits"] == []
    assert payload["provenance"]["coverage"]["fts"] == "none"
    assert payload["degraded"] is True
    assert "full-text arm unavailable" in payload["degraded_reason"]


def test_recall_like_fallback_reports_bounded_coverage_when_older_match_is_omitted(
    recall_engine, monkeypatch
):
    recall_engine._config.embeddings_enabled = False
    old_exact = recall_engine._store.append(
        "session-old",
        {"role": "assistant", "content": "cobalt orchid exact durable fact"},
    )
    recent_partial = recall_engine._store.append(
        "session-new",
        {"role": "assistant", "content": "cobalt only recent partial"},
    )
    monkeypatch.setattr(store_module, "compute_search_candidate_cap", lambda _limit: 1)

    payload = _recall(
        recall_engine,
        monkeypatch,
        query="cobalt/orchid",
        include="verbatim",
        limit=1,
    )

    assert payload["hits"][0]["store_id"] == recent_partial
    assert old_exact not in {hit["store_id"] for hit in payload["hits"]}
    assert payload["provenance"]["coverage"]["fts"] == "bounded"
    assert payload["degraded"] is True
    assert "LIKE fallback" in payload["degraded_reason"]


@pytest.mark.parametrize(
    ("query", "content"),
    [
        ("alpha/beta", "alpha beta punctuation route"),
        ("記憶", "古い記憶の記録"),
        ("🧠", "durable 🧠 note"),
    ],
)
def test_recall_like_routes_disclose_bounded_coverage(
    recall_engine, monkeypatch, query, content
):
    recall_engine._config.embeddings_enabled = False
    recall_engine._store.append(
        "session-old",
        {"role": "assistant", "content": content},
    )

    payload = _recall(
        recall_engine,
        monkeypatch,
        query=query,
        include="verbatim",
        limit=1,
    )

    assert payload["hits"]
    assert payload["provenance"]["coverage"]["fts"] == "bounded"
    assert payload["degraded"] is True
    assert "LIKE fallback" in payload["degraded_reason"]


def test_recall_like_fallback_timeout_is_never_reported_as_full(
    recall_engine, monkeypatch
):
    recall_engine._config.embeddings_enabled = False
    monkeypatch.setattr(
        lcm_tools,
        "_lcm_grep_full_text_with_deadline",
        lambda *_a, **_k: {
            "error": "lcm_grep request deadline exceeded",
            "mode": "recall",
            "timeout": True,
        },
    )

    payload = _recall(
        recall_engine,
        monkeypatch,
        query="cobalt/orchid",
        include="verbatim",
        limit=1,
    )

    assert payload["hits"] == []
    assert payload["provenance"]["coverage"]["fts"] == "none"
    assert payload["degraded"] is True
    assert payload["timeout"] is True


def test_bounded_chunk_coverage_surfaces_as_degraded(recall_engine, monkeypatch):
    """SCAN-1: a recency-bounded chunk arm reports a degraded_reasons entry naming
    the arm + scanned/total, instead of silently truncating."""
    recall_engine._config.recall_scan_rows = 1
    recall_engine._config.recall_full_corpus_scan_enabled = False
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


def test_two_stage_full_approx_coverage_surfaces_as_approximate(recall_engine, monkeypatch):
    """FIX 2: a two-stage (binary prescreen) summary arm reaches the whole corpus
    but ranks approximately, so it reports coverage='full_approx' and discloses
    the approximate prescreen in degraded_reason (like 'bounded' is disclosed),
    rather than passing as an exact 'full'."""
    recall_engine._config.embedding_binary_prescreen = True
    recall_engine._config.recall_full_corpus_scan_enabled = False
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
        recall_engine._config.recall_full_corpus_scan_enabled = False
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
