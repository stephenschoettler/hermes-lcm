"""Foreground summaries must publish against unchanged durable sources."""

import copy
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.rollup_store import RollupStore


@pytest.fixture
def engine(tmp_path):
    config = LCMConfig(
        database_path=str(tmp_path / "foreground.db"),
        fresh_tail_count=2,
        leaf_chunk_tokens=1,
        dynamic_leaf_chunk_enabled=False,
        threshold_full_sweep_enabled=False,
    )
    instance = LCMEngine(config, hermes_home=str(tmp_path / "profile"))
    instance.on_session_start("foreground-session", conversation_id="foreground-lane")
    try:
        yield instance
    finally:
        instance.shutdown()


def _messages():
    return [
        {"role": "user", "content": "The original source fact. " * 100},
        {"role": "assistant", "content": "The original source reply. " * 100},
        {"role": "user", "content": "The recent question."},
        {"role": "assistant", "content": "The recent reply."},
    ]


def _publication_state(engine):
    # A separate reader observes committed state, including trigger side effects.
    with sqlite3.connect(engine._store.db_path) as reader:
        return (
            reader.execute("SELECT COUNT(*) FROM summary_nodes").fetchone()[0],
            reader.execute(
                "SELECT current_frontier_store_id FROM lcm_lifecycle_state "
                "WHERE conversation_id='foreground-lane'"
            ).fetchone()[0],
            reader.execute(
                "SELECT COUNT(*) FROM nodes_fts WHERE nodes_fts MATCH 'faithful'"
            ).fetchone()[0],
            reader.execute("SELECT COUNT(*) FROM lcm_rollup_invalidations").fetchone()[0],
        )


@pytest.mark.parametrize("mutation", ["content", "tool_calls", "delete", "frontier", "session", "parent"])
def test_compress_rejects_sources_changed_during_model(engine, monkeypatch, mutation):
    messages = _messages()
    if mutation == "tool_calls":
        messages[1]["tool_calls"] = [{
            "id": "synthetic-lookup", "type": "function",
            "function": {"name": "lookup_fact", "arguments": '{"version":"original"}'},
        }]
        messages.insert(2, {
            "role": "tool", "tool_call_id": "synthetic-lookup",
            "content": "The original lookup result.",
        })
    original = copy.deepcopy(messages)
    selected_ids = []

    def summarize(chunk, **kwargs):
        selected_ids.extend(engine._get_store_ids_for_messages(chunk))
        with sqlite3.connect(engine._store.db_path) as writer:
            if mutation == "delete":
                writer.execute("DELETE FROM messages WHERE store_id=?", (selected_ids[0],))
            elif mutation == "content":
                writer.execute("UPDATE messages SET content='changed source' WHERE store_id=?", (selected_ids[0],))
            elif mutation == "tool_calls":
                changed_calls = copy.deepcopy(messages[1]["tool_calls"])
                changed_calls[0]["function"]["arguments"] = '{"version":"changed"}'
                writer.execute("UPDATE messages SET tool_calls=? WHERE store_id=?", (json.dumps(changed_calls), selected_ids[1]))
            elif mutation == "frontier":
                writer.execute("UPDATE lcm_lifecycle_state SET current_frontier_store_id=99 WHERE conversation_id='foreground-lane'")
            elif mutation == "session":
                writer.execute("UPDATE lcm_lifecycle_state SET current_session_id='new-session' WHERE conversation_id='foreground-lane'")
            else:
                engine._dag.add_node(SummaryNode(
                    session_id=engine._session_id, summary="another writer summary",
                    source_ids=selected_ids[:1],
                ))
        return chunk, 1000, "faithful old-source summary", 1, 0

    monkeypatch.setattr(engine, "_summarize_leaf_chunk_with_rescue", summarize)
    with pytest.raises(RuntimeError, match="changed|disappeared|ownership"):
        engine.compress(messages, current_tokens=100000)

    nodes = engine._dag.get_session_nodes("foreground-session")
    assert len(nodes) == (1 if mutation == "parent" else 0)
    assert all(node.summary != "faithful old-source summary" for node in nodes)
    state = engine._lifecycle.get_by_conversation("foreground-lane")
    assert state.current_frontier_store_id == (99 if mutation == "frontier" else 0)
    assert messages == original
    rows = engine._store.get_session_messages("foreground-session")
    assert len(rows) == len(original) - (1 if mutation == "delete" else 0)
    assert rows[-2]["content"] == original[-2]["content"]
    assert rows[-1]["content"] == original[-1]["content"]


@pytest.mark.parametrize("mutation", ["summary", "delete", "parent"])
def test_condensation_rejects_sources_changed_during_model(engine, monkeypatch, mutation):
    import hermes_lcm.engine as module

    node = SummaryNode(session_id=engine._session_id, summary="original leaf", token_count=100)
    engine._dag.add_node(node)
    original = copy.deepcopy(node)

    def summarize(**kwargs):
        assert kwargs["text"] == "original leaf"
        with sqlite3.connect(engine._store.db_path) as writer:
            if mutation == "summary":
                writer.execute("UPDATE summary_nodes SET summary='changed leaf' WHERE node_id=?", (node.node_id,))
            elif mutation == "delete":
                writer.execute("DELETE FROM summary_nodes WHERE node_id=?", (node.node_id,))
            else:
                engine._dag.add_node(SummaryNode(
                    session_id=engine._session_id, summary="another parent", depth=1,
                    source_type="nodes", source_ids=[node.node_id],
                ))
        return "stale condensed summary", 1

    monkeypatch.setattr(module, "summarize_with_escalation", summarize)
    with pytest.raises(RuntimeError, match="changed|disappeared"):
        engine._condense_summary_nodes([node])
    assert all(n.summary != "stale condensed summary" for n in engine._dag.get_session_nodes(engine._session_id))
    assert node == original


def test_compress_frontier_failure_rolls_back_summary_and_triggers(engine, monkeypatch):
    messages = _messages()
    original = copy.deepcopy(messages)
    RollupStore(engine._store.db_path).close()
    with sqlite3.connect(engine._store.db_path) as writer:
        writer.execute(
            "CREATE TRIGGER fail_frontier BEFORE UPDATE ON lcm_lifecycle_state "
            "BEGIN SELECT RAISE(ABORT, 'injected frontier failure'); END"
        )
    monkeypatch.setattr(engine, "_summarize_leaf_chunk_with_rescue", lambda chunk, **kwargs: (chunk, 1000, "faithful summary", 1, 0))
    with pytest.raises(sqlite3.IntegrityError, match="injected frontier failure"):
        engine.compress(messages, current_tokens=100000)
    assert _publication_state(engine) == (0, 0, 0, 0)
    assert engine._last_compacted_store_id == 0
    assert messages == original
    assert [r["content"] for r in engine._store.get_session_messages(engine._session_id)] == [m["content"] for m in original]


def test_compress_commits_summary_frontier_and_triggers_together(engine, monkeypatch):
    messages = _messages()
    original = copy.deepcopy(messages)
    RollupStore(engine._store.db_path).close()
    monkeypatch.setattr(engine, "_summarize_leaf_chunk_with_rescue", lambda chunk, **kwargs: (chunk, 1000, "faithful summary", 1, 0))
    result = engine.compress(messages, current_tokens=100000)
    nodes = engine._dag.get_session_nodes(engine._session_id)
    assert len(nodes) == 1
    frontier = max(nodes[0].source_ids)
    assert _publication_state(engine) == (1, frontier, 1, 1)
    assert engine._last_compacted_store_id == frontier
    assert messages == original
    assert [r["content"] for r in engine._store.get_session_messages(engine._session_id)] == [m["content"] for m in original]
    assert result[-2:] == messages[-2:]


def test_concurrent_condensations_publish_the_same_sources_only_once(engine, monkeypatch):
    import hermes_lcm.engine as module

    sibling = LCMEngine(copy.deepcopy(engine._config), hermes_home=engine._hermes_home)
    sibling.on_session_start(engine._session_id, conversation_id="foreground-lane")
    node = SummaryNode(session_id=engine._session_id, summary="original leaf", token_count=100)
    engine._dag.add_node(node)
    barrier = Barrier(2)

    def summarize(**kwargs):
        assert kwargs["text"] == "original leaf"
        barrier.wait(timeout=5)
        return "one condensed parent", 1

    def condense(instance):
        try:
            instance._condense_summary_nodes([copy.deepcopy(node)])
            return "committed"
        except RuntimeError as exc:
            assert "changed" in str(exc)
            return "rejected"

    monkeypatch.setattr(module, "summarize_with_escalation", summarize)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(condense, instance) for instance in [engine, sibling]]
            assert sorted(future.result(timeout=10) for future in futures) == ["committed", "rejected"]
        assert len(engine._dag.get_session_nodes(engine._session_id)) == 2
    finally:
        sibling.shutdown()


def test_compress_rejects_source_changed_after_its_initial_mapping(engine, monkeypatch):
    messages = _messages()
    original = copy.deepcopy(messages)
    map_messages = engine._get_store_id_map_for_messages
    changed = False

    def map_then_change(chunk):
        nonlocal changed
        mapped = map_messages(chunk)
        if len(chunk) == 2 and mapped and not changed:
            changed = True
            with sqlite3.connect(engine._store.db_path) as writer:
                writer.execute(
                    "UPDATE messages SET content='concurrent changed fact' WHERE store_id=?",
                    (min(mapped.values()),),
                )
        return mapped

    monkeypatch.setattr(engine, "_get_store_id_map_for_messages", map_then_change)
    monkeypatch.setattr(engine, "_summarize_leaf_chunk_with_rescue", lambda chunk, **kwargs: (chunk, 1000, "stale source summary", 1, 0))
    with pytest.raises(RuntimeError, match="changed"):
        engine.compress(messages, current_tokens=100000)
    assert changed
    assert engine._dag.get_session_nodes(engine._session_id) == []
    assert engine._lifecycle.get_by_conversation("foreground-lane").current_frontier_store_id == 0
    assert messages == original
