"""Evidence-provenance contract tests for lcm_expand_query."""

import json

import pytest

import hermes_lcm.tools as lcm_tools
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.schemas import LCM_EXPAND_QUERY


@pytest.fixture
def engine(tmp_path):
    config = LCMConfig(database_path=str(tmp_path / "expand-query-provenance.db"))
    instance = LCMEngine(config=config)
    instance._session_id = "test-session"
    instance.context_length = 200_000
    instance.threshold_tokens = int(instance.context_length * config.context_threshold)
    try:
        yield instance
    finally:
        instance.shutdown()


def _add_message_summary(engine, content="The rollout target is Tuesday at 09:00 UTC."):
    store_id = engine._store.append(
        "test-session",
        {"role": "user", "content": content},
        source="cli",
    )
    node_id = engine._dag.add_node(
        SummaryNode(
            session_id="test-session",
            depth=0,
            summary="Rollout timing was discussed.",
            token_count=8,
            source_token_count=12,
            source_ids=[store_id],
            source_type="messages",
            created_at=1,
        )
    )
    return store_id, node_id


def test_expand_query_returns_tool_extracted_evidence_without_claiming_entailment(engine, monkeypatch):
    store_id, node_id = _add_message_summary(engine)
    monkeypatch.setattr(
        lcm_tools,
        "_synthesize_expansion_answer",
        lambda **kwargs: "The rollout is Friday and the moon is cheese.",
    )

    result = json.loads(
        engine.handle_tool_call(
            "lcm_expand_query",
            {"prompt": "When is rollout?", "node_ids": [node_id], "context_max_tokens": 1000},
        )
    )

    assert result["answer"] == "The rollout is Friday and the moon is cheese."
    evidence = result["evidence_provenance"]
    assert evidence["synthesis_status"] == "completed"
    assert evidence["semantic_entailment"] == "not_verified"
    assert evidence["quote_origin"] == "tool_extracted_from_synthesis_context"
    assert evidence["retrieval_scope"] == {"kind": "current_session", "session_ids": ["test-session"]}
    assert evidence["identifiers_are_authority"] is False
    assert evidence["locator_replay_safety"] == "not_guaranteed"
    assert evidence["locator_coverage"] == "complete"

    summary_item = next(item for item in evidence["items"] if item["source_type"] == "summary")
    message_item = next(item for item in evidence["items"] if item["source_type"] == "raw_message")
    assert summary_item["node_id"] == node_id
    assert summary_item["session_id"] == "test-session"
    assert summary_item["quote"] == "Rollout timing was discussed."
    assert summary_item["locator_present"] is True
    assert summary_item["locator_replay_status"] == "unverified"
    assert summary_item["expand_args"] == {"node_id": node_id}
    assert message_item["store_id"] == store_id
    assert message_item["session_id"] == "test-session"
    assert message_item["quote"] == "The rollout target is Tuesday at 09:00 UTC."
    assert message_item["locator_present"] is True
    assert message_item["locator_replay_status"] == "unverified"
    assert message_item["expand_args"] == {"store_id": store_id, "content_offset": 0}

    expanded = json.loads(engine.handle_tool_call("lcm_expand", message_item["expand_args"]))
    assert expanded["content"].startswith(message_item["quote"])


def test_expand_query_session_snapshot_survives_rebind_during_synthesis(engine, monkeypatch):
    _, node_id = _add_message_summary(engine)

    def synthesize(**kwargs):
        engine._session_id = "rebound-session"
        return "snapshot answer"

    monkeypatch.setattr(lcm_tools, "_synthesize_expansion_answer", synthesize)

    result = json.loads(
        engine.handle_tool_call(
            "lcm_expand_query",
            {
                "prompt": "When is rollout?",
                "node_ids": [node_id],
                "context_max_tokens": 1000,
            },
        )
    )

    evidence = result["evidence_provenance"]
    assert engine.current_session_id == "rebound-session"
    assert evidence["retrieval_scope"] == {
        "kind": "current_session",
        "session_ids": ["test-session"],
    }
    assert all(item["session_id"] == "test-session" for item in evidence["items"])


def test_expand_query_preserves_evidence_when_synthesis_fails(engine, monkeypatch):
    store_id, node_id = _add_message_summary(engine)

    def timeout(**kwargs):
        raise TimeoutError("provider timeout")

    monkeypatch.setattr(lcm_tools, "_synthesize_expansion_answer", timeout)

    result = json.loads(
        engine.handle_tool_call(
            "lcm_expand_query",
            {"prompt": "When is rollout?", "node_ids": [node_id], "context_max_tokens": 1000},
        )
    )

    assert result["degraded"] is True
    assert "answer" not in result
    assert result["evidence_provenance"]["synthesis_status"] == "failed"
    assert result["evidence_provenance"]["semantic_entailment"] == "not_applicable"
    assert any(item.get("store_id") == store_id for item in result["evidence_provenance"]["items"])


def test_expand_query_represents_empty_evidence_without_running_synthesis(engine, monkeypatch):
    called = False

    def synthesize(**kwargs):
        nonlocal called
        called = True
        return "should not run"

    monkeypatch.setattr(lcm_tools, "_synthesize_expansion_answer", synthesize)

    result = json.loads(
        engine.handle_tool_call(
            "lcm_expand_query",
            {"prompt": "What mentions a missing token?", "query": "NEVER_PRESENT"},
        )
    )

    assert called is False
    assert result["answer"] == "No matching summaries or raw messages found in the current session."
    assert result["evidence_provenance"]["items"] == []
    assert result["evidence_provenance"]["locator_coverage"] == "none"
    assert result["evidence_provenance"]["synthesis_status"] == "not_run"
    assert result["evidence_provenance"]["semantic_entailment"] == "not_applicable"


def test_expand_query_evidence_reports_partial_context_separately_from_entailment(engine, monkeypatch):
    _, node_id = _add_message_summary(engine, content="abcdef")
    monkeypatch.setattr(lcm_tools, "_synthesize_expansion_answer", lambda **kwargs: "bounded answer")

    result = json.loads(
        engine.handle_tool_call(
            "lcm_expand_query",
            {
                "prompt": "What raw detail?",
                "node_ids": [node_id],
                "max_tokens": 5,
                "context_max_tokens": 1,
            },
        )
    )

    assert result["context_truncated"] is True
    assert result["evidence_provenance"]["context_truncated"] is True
    assert result["evidence_provenance"]["semantic_entailment"] == "not_verified"


def test_expand_query_evidence_represents_distinct_transcript_content():
    evidence = lcm_tools._build_expand_query_evidence(
        [
            {
                "type": "messages",
                "messages": [
                    {
                        "store_id": 17,
                        "session_id": "test-session",
                        "role": "tool",
                        "source": "cli",
                        "content": "hydrated external payload",
                        "transcript_content": "[externalized ref=payload-17]",
                        "content_source": "externalized_payload",
                        "content_offset": 0,
                        "externalized": {"ref": "payload-17"},
                    }
                ],
            }
        ],
        session_id="test-session",
        context_truncated=False,
        synthesis_status="completed",
    )

    assert evidence["locator_coverage"] == "complete"
    assert evidence["context_unique_item_count"] == 2
    assert evidence["context_occurrence_count"] == 2
    by_source = {item["content_source"]: item for item in evidence["items"]}
    assert by_source["externalized_payload"]["quote"] == "hydrated external payload"
    assert by_source["externalized_payload"]["expand_args"] == {
        "externalized_ref": "payload-17",
        "content_offset": 0,
    }
    assert by_source["transcript_content"]["quote"] == "[externalized ref=payload-17]"
    assert by_source["transcript_content"]["expand_args"] == {
        "store_id": 17,
        "content_offset": 0,
    }


def test_expand_query_provenance_covers_recursive_descendants(engine, monkeypatch):
    store_id, leaf_id = _add_message_summary(engine, content="recursive leaf evidence")
    middle_id = engine._dag.add_node(
        SummaryNode(
            session_id="test-session",
            depth=1,
            summary="middle summary",
            token_count=4,
            source_token_count=12,
            source_ids=[leaf_id],
            source_type="nodes",
            created_at=2,
        )
    )
    parent_id = engine._dag.add_node(
        SummaryNode(
            session_id="test-session",
            depth=2,
            summary="parent summary",
            token_count=4,
            source_token_count=16,
            source_ids=[middle_id],
            source_type="nodes",
            created_at=3,
        )
    )
    monkeypatch.setattr(lcm_tools, "_synthesize_expansion_answer", lambda **kwargs: "recursive answer")

    result = json.loads(
        engine.handle_tool_call(
            "lcm_expand_query",
            {"prompt": "What is the leaf evidence?", "node_ids": [parent_id], "context_max_tokens": 1000},
        )
    )

    items = result["evidence_provenance"]["items"]
    assert any(item.get("node_id") == parent_id for item in items)
    middle_item = next(item for item in items if item.get("node_id") == middle_id)
    leaf_item = next(
        item
        for item in items
        if item.get("node_id") == leaf_id and item["source_type"] == "summary"
    )
    message_item = next(item for item in items if item.get("store_id") == store_id)
    assert message_item["quote"] == "recursive leaf evidence"
    assert middle_item["source_paths"] == [
        {
            "path": [{"node_id": parent_id, "source_index": 0}],
            "depth": 1,
            "truncated": False,
        }
    ]
    assert leaf_item["source_paths"] == [
        {
            "path": [
                {"node_id": parent_id, "source_index": 0},
                {"node_id": middle_id, "source_index": 0},
            ],
            "depth": 2,
            "truncated": False,
        }
    ]
    assert message_item["source_paths"] == [
        {
            "path": [
                {"node_id": parent_id, "source_index": 0},
                {"node_id": middle_id, "source_index": 0},
                {"node_id": leaf_id, "source_index": 0},
            ],
            "depth": 3,
            "truncated": False,
        }
    ]


def test_expand_query_raw_window_provenance_expands_from_exact_offset(engine, monkeypatch):
    content = "prefix-noise " * 80 + "TAILMATCH exact evidence"
    store_id = engine._store.append("test-session", {"role": "user", "content": content}, source="cli")
    stored = engine._store.get(store_id)
    stored["search_rank"] = 1
    stored["snippet"] = "TAILMATCH exact evidence"
    monkeypatch.setattr(engine._store, "search", lambda *args, **kwargs: [stored])
    monkeypatch.setattr(engine._dag, "search", lambda *args, **kwargs: [])
    monkeypatch.setattr(lcm_tools, "_synthesize_expansion_answer", lambda **kwargs: "raw answer")

    result = json.loads(
        engine.handle_tool_call(
            "lcm_expand_query",
            {
                "prompt": "What does TAILMATCH say?",
                "query": "TAILMATCH",
                "max_tokens": 20,
                "context_max_tokens": 8,
            },
        )
    )

    item = next(item for item in result["evidence_provenance"]["items"] if item.get("store_id") == store_id)
    assert item["content_source"] == "raw_search_hit"
    assert item["content_offset"] == content.index("TAILMATCH")
    assert "TAILMATCH" in item["quote"]
    expanded = json.loads(engine.handle_tool_call("lcm_expand", item["expand_args"]))
    assert expanded["content"].startswith(item["quote"])


def test_expand_query_externalized_provenance_is_directly_expandable(tmp_path, monkeypatch):
    config = LCMConfig(
        database_path=str(tmp_path / "externalized-provenance.db"),
        large_output_externalization_enabled=True,
        large_output_externalization_threshold_chars=200,
    )
    instance = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
    instance._session_id = "test-session"
    content = "EXTERNALIZED RAW DETAIL " + ("abcdef" * 100)
    try:
        instance._serialize_messages([{"role": "tool", "tool_call_id": "call_ext", "content": content}])
        ref = next((tmp_path / "hermes" / "lcm-large-outputs").glob("*.json")).name
        placeholder = f"[GC'd externalized tool output: tool_call_id=call_ext; chars={len(content)}; ref={ref}]"
        store_id = instance._store.append(
            "test-session",
            {"role": "tool", "tool_call_id": "call_ext", "content": placeholder},
        )
        node_id = instance._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="externalized payload summary",
                token_count=10,
                source_token_count=200,
                source_ids=[store_id],
                source_type="messages",
                created_at=0,
            )
        )
        monkeypatch.setattr(lcm_tools, "_synthesize_expansion_answer", lambda **kwargs: "externalized answer")

        result = json.loads(
            instance.handle_tool_call(
                "lcm_expand_query",
                {"prompt": "What externalized detail exists?", "node_ids": [node_id], "context_max_tokens": 500},
            )
        )
        items = result["evidence_provenance"]["items"]
        payload_item = next(item for item in items if item.get("content_source") == "externalized_payload")
        transcript_item = next(item for item in items if item.get("content_source") == "transcript_content")
        assert payload_item["source_type"] == "externalized_payload"
        assert payload_item["externalized_ref"] == ref
        assert payload_item["expand_args"] == {"externalized_ref": ref, "content_offset": 0}
        assert payload_item["quote"].startswith("EXTERNALIZED RAW DETAIL")
        assert transcript_item["source_type"] == "raw_message"
        assert transcript_item["quote"] == placeholder
        assert transcript_item["expand_args"] == {"store_id": store_id, "content_offset": 0}
        expanded = json.loads(instance.handle_tool_call("lcm_expand", payload_item["expand_args"]))
        assert expanded["content"].startswith(payload_item["quote"])
    finally:
        instance.shutdown()


def test_expand_query_evidence_bounds_dedupes_and_preserves_conflicts():
    messages = []
    for store_id in range(1, 27):
        content = ("A" if store_id == 1 else "B" if store_id == 2 else f"evidence {store_id}")
        if store_id == 1:
            content = content * 700
        messages.append(
            {
                "store_id": store_id,
                "session_id": "test-session",
                "role": "user",
                "source": "cli",
                "content": content,
                "content_source": "message",
                "content_offset": 0,
            }
        )
    messages.append(dict(messages[0]))

    evidence = lcm_tools._build_expand_query_evidence(
        [{"type": "raw_messages", "messages": messages}],
        session_id="test-session",
        context_truncated=False,
        synthesis_status="completed",
    )

    assert evidence["context_unique_item_count"] == 26
    assert evidence["context_occurrence_count"] == 27
    assert 2 <= len(evidence["items"]) <= 24
    assert evidence["items_truncated"] is True
    assert evidence["locator_coverage"] == "partial"
    assert evidence["serialized_char_limit"] == 10_000
    assert len(json.dumps(evidence)) <= evidence["serialized_char_limit"]
    assert evidence["items"][0]["quote_chars_before_provenance_cap"] == 700
    assert evidence["items"][0]["quote_truncated_by_provenance_cap"] is True
    assert len(evidence["items"][0]["quote"]) == 500
    assert evidence["items"][1]["quote"] == "B"


def test_expand_query_blank_synthesis_retains_failed_provenance(engine, monkeypatch):
    _, node_id = _add_message_summary(engine)
    monkeypatch.setattr(lcm_tools, "_synthesize_expansion_answer", lambda **kwargs: "   ")

    result = json.loads(
        engine.handle_tool_call(
            "lcm_expand_query",
            {"prompt": "When is rollout?", "node_ids": [node_id]},
        )
    )

    assert result["degraded"] is True
    assert result["evidence_provenance"]["synthesis_status"] == "failed"
    assert result["evidence_provenance"]["semantic_entailment"] == "not_applicable"
    assert result["evidence_provenance"]["items"]


def test_expand_query_schema_describes_evidence_boundary():
    description = LCM_EXPAND_QUERY["description"]

    assert "locator coverage" in description.lower()
    assert "semantic entailment" in description.lower()


def test_expand_query_evidence_cap_uses_final_ascii_escaping():
    messages = [
        {
            "store_id": index,
            "session_id": "unicode-session",
            "role": "user",
            "source": "cli",
            "content": ("漢字🙂" * 300) + str(index),
            "content_source": "message",
            "content_offset": 0,
        }
        for index in range(1, 25)
    ]

    evidence = lcm_tools._build_expand_query_evidence(
        [{"type": "raw_messages", "messages": messages}],
        session_id="unicode-session",
        context_truncated=False,
        synthesis_status="completed",
    )

    assert evidence["serialization"] == {
        "scope": "evidence_provenance_only",
        "json_ensure_ascii": True,
    }
    assert len(json.dumps(evidence)) <= evidence["serialized_char_limit"]
    assert len(evidence["items"]) < len(messages)
    assert evidence["items_truncated"] is True


def test_expand_query_evidence_bounds_oversized_scope_and_item_metadata():
    long_session = "session-" + ("x" * 20_000)
    long_ref = "ref-" + ("界" * 10_000)
    evidence = lcm_tools._build_expand_query_evidence(
        [
            {
                "type": "raw_messages",
                "messages": [
                    {
                        "store_id": 7,
                        "session_id": long_session,
                        "role": "role-" + ("r" * 2_000),
                        "source": "source-" + ("s" * 2_000),
                        "content": "bounded payload",
                        "content_source": "externalized_payload",
                        "content_offset": 0,
                        "externalized": {"ref": long_ref},
                    }
                ],
            }
        ],
        session_id=long_session,
        context_truncated=False,
        synthesis_status="completed",
    )

    assert len(json.dumps(evidence)) <= evidence["serialized_char_limit"]
    assert evidence["metadata_truncated"] is True
    assert len(evidence["retrieval_scope"]["session_ids"][0]) == 256
    scope_meta = evidence["retrieval_scope"]["metadata_truncation"]["session_ids[0]"]
    assert scope_meta["original_chars"] == len(long_session)
    assert len(scope_meta["sha256"]) == 64
    item = evidence["items"][0]
    assert item["locator_present"] is False
    assert item["locator_replay_status"] == "not_available"
    assert "expand_args" not in item
    assert item["metadata_truncation"]["externalized_ref"]["original_chars"] == len(long_ref)


def test_expand_query_locator_presence_does_not_claim_verified_replay():
    evidence = lcm_tools._build_expand_query_evidence(
        [
            {
                "type": "raw_messages",
                "messages": [
                    {
                        "store_id": 999_999_999,
                        "session_id": "missing-session",
                        "content": "context can carry a stale locator",
                        "content_source": "message",
                        "content_offset": 0,
                    }
                ],
            }
        ],
        session_id="missing-session",
        context_truncated=False,
        synthesis_status="completed",
    )

    item = evidence["items"][0]
    assert item["locator_present"] is True
    assert item["locator_replay_status"] == "unverified"
    assert evidence["locator_coverage"] == "complete"
    assert evidence["locator_replay_safety"] == "not_guaranteed"


def test_expand_query_dedupe_preserves_production_paths_and_occurrences():
    truncated_path = [
        {"node_id": node_id, "source_index": node_id - 3}
        for node_id in range(3, 11)
    ]
    blocks = [
        {
            "type": "child_nodes",
            "node_id": 1,
            "children": [
                {
                    "node_id": 99,
                    "source_index": 0,
                    "summary": "shared descendant",
                }
            ],
        },
        {
            "type": "descendant_child_nodes",
            "node_id": 20,
            "source_path": truncated_path,
            "source_path_depth": 10,
            "source_path_truncated": True,
            "children": [
                {
                    "node_id": 99,
                    "source_index": 3,
                    "summary": "shared descendant",
                }
            ],
        },
    ]

    evidence = lcm_tools._build_expand_query_evidence(
        blocks,
        session_id="test-session",
        context_truncated=False,
        synthesis_status="completed",
    )

    assert evidence["context_unique_item_count"] == 1
    assert evidence["context_occurrence_count"] == 2
    item = evidence["items"][0]
    assert item["context_occurrence_count"] == 2
    assert item["source_paths"] == [
        {
            "path": [{"node_id": 1, "source_index": 0}],
            "depth": 1,
            "truncated": False,
        },
        {
            "path": [
                *truncated_path[1:],
                {"node_id": 20, "source_index": 3},
            ],
            "depth": 11,
            "truncated": True,
        },
    ]


def test_expand_query_child_message_paths_include_final_dag_edge():
    evidence = lcm_tools._build_expand_query_evidence(
        [
            {
                "type": "child_messages",
                "node_id": 7,
                "source_path": [{"node_id": 1, "source_index": 0}],
                "source_path_depth": 1,
                "messages": [
                    {
                        "store_id": 42,
                        "source_index": 2,
                        "session_id": "test-session",
                        "content": "hydrated child evidence",
                        "transcript_content": "stored child transcript",
                    }
                ],
            }
        ],
        session_id="test-session",
        context_truncated=False,
        synthesis_status="completed",
    )

    assert evidence["context_unique_item_count"] == 2
    for item in evidence["items"]:
        assert item["source_paths"] == [
            {
                "path": [
                    {"node_id": 1, "source_index": 0},
                    {"node_id": 7, "source_index": 2},
                ],
                "depth": 2,
                "truncated": False,
            }
        ]
