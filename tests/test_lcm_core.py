"""Tests for LCM core components: store, DAG, tokens, config, escalation."""

import json
import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.tokens import count_tokens, count_message_tokens, count_messages_tokens
from hermes_lcm.store import MessageStore
from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.escalation import _deterministic_truncate


class TestConfig:
    def test_defaults(self):
        c = LCMConfig()
        assert c.fresh_tail_count == 64
        assert c.leaf_chunk_tokens == 20_000
        assert c.context_threshold == 0.75
        assert c.condensation_fanin == 4

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("LCM_FRESH_TAIL_COUNT", "32")
        monkeypatch.setenv("LCM_CONTEXT_THRESHOLD", "0.80")
        c = LCMConfig.from_env()
        assert c.fresh_tail_count == 32
        assert c.context_threshold == 0.80


class TestTokens:
    def test_count_tokens_empty(self):
        assert count_tokens("") == 0

    def test_count_tokens_nonempty(self):
        assert count_tokens("hello world") > 0

    def test_count_message_tokens(self):
        msg = {"role": "user", "content": "hello world this is a test"}
        assert count_message_tokens(msg) > 0

    def test_count_messages_tokens(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
        assert count_messages_tokens(msgs) > 0


class TestMessageStore:
    @pytest.fixture
    def store(self, tmp_path):
        return MessageStore(tmp_path / "test.db")

    def test_append_and_get(self, store):
        sid = store.append("sess1", {"role": "user", "content": "hello"}, token_estimate=5)
        assert sid > 0
        retrieved = store.get(sid)
        assert retrieved["role"] == "user"
        assert retrieved["content"] == "hello"

    def test_append_batch(self, store):
        msgs = [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
            {"role": "user", "content": "three"},
        ]
        ids = store.append_batch("sess1", msgs, [1, 2, 3])
        assert len(ids) == 3
        assert ids[0] < ids[1] < ids[2]

    def test_get_range(self, store):
        msgs = [{"role": "user", "content": f"msg {i}"} for i in range(10)]
        ids = store.append_batch("sess1", msgs)
        result = store.get_range("sess1", start_id=ids[3], end_id=ids[7])
        assert len(result) == 5

    def test_session_count(self, store):
        store.append("sess1", {"role": "user", "content": "a"})
        store.append("sess1", {"role": "assistant", "content": "b"})
        assert store.get_session_count("sess1") == 2
        assert store.get_session_count("sess2") == 0

    def test_search(self, store):
        store.append("sess1", {"role": "user", "content": "deploy the docker container"})
        store.append("sess1", {"role": "assistant", "content": "running kubectl"})
        results = store.search("docker", session_id="sess1")
        assert len(results) >= 1

    def test_pin_unpin(self, store):
        sid = store.append("sess1", {"role": "user", "content": "important"})
        store.pin(sid)
        assert store.get(sid)["pinned"] == 1
        store.unpin(sid)
        assert store.get(sid)["pinned"] == 0

    def test_to_openai_msg(self, store):
        sid = store.append("sess1", {
            "role": "assistant", "content": "hello",
            "tool_calls": [{"id": "tc1", "function": {"name": "t", "arguments": "{}"}}],
        })
        msg = store.to_openai_msg(store.get(sid))
        assert msg["role"] == "assistant"
        assert len(msg["tool_calls"]) == 1


class TestSummaryDAG:
    @pytest.fixture
    def dag(self, tmp_path):
        return SummaryDAG(tmp_path / "test.db")

    def test_add_and_get(self, dag):
        node = SummaryNode(
            session_id="s1", depth=0,
            summary="FastAPI project setup",
            token_count=10, source_token_count=500,
            source_ids=[1, 2, 3], source_type="messages",
            expand_hint="FastAPI setup",
        )
        nid = dag.add_node(node)
        assert nid > 0
        r = dag.get_node(nid)
        assert r.summary == "FastAPI project setup"
        assert r.source_ids == [1, 2, 3]

    def test_session_nodes(self, dag):
        for i in range(3):
            dag.add_node(SummaryNode(
                session_id="s1", depth=0, summary=f"S{i}",
                token_count=10, source_ids=[i], source_type="messages",
            ))
        dag.add_node(SummaryNode(
            session_id="s2", depth=0, summary="Other",
            token_count=10, source_ids=[99], source_type="messages",
        ))
        assert len(dag.get_session_nodes("s1")) == 3

    def test_count_at_depth(self, dag):
        for i in range(4):
            dag.add_node(SummaryNode(
                session_id="s1", depth=0, summary=f"D0-{i}",
                token_count=10, source_ids=[i], source_type="messages",
            ))
        dag.add_node(SummaryNode(
            session_id="s1", depth=1, summary="D1",
            token_count=20, source_ids=[1, 2, 3, 4], source_type="nodes",
        ))
        assert dag.count_at_depth("s1", 0) == 4
        assert dag.count_at_depth("s1", 1) == 1

    def test_search(self, dag):
        dag.add_node(SummaryNode(
            session_id="s1", depth=0,
            summary="Docker containers for the API",
            token_count=10, source_ids=[1], source_type="messages",
        ))
        results = dag.search("Docker", session_id="s1")
        assert len(results) >= 1

    def test_describe_subtree(self, dag):
        c1 = dag.add_node(SummaryNode(
            session_id="s1", depth=0, summary="Child 1",
            token_count=10, source_ids=[1], source_type="messages",
        ))
        c2 = dag.add_node(SummaryNode(
            session_id="s1", depth=0, summary="Child 2",
            token_count=15, source_ids=[2], source_type="messages",
        ))
        parent = dag.add_node(SummaryNode(
            session_id="s1", depth=1, summary="Parent",
            token_count=20, source_ids=[c1, c2], source_type="nodes",
        ))
        info = dag.describe_subtree(parent)
        assert info["depth"] == 1
        assert len(info["children"]) == 2


class TestEscalation:
    def test_truncate_long(self):
        result = _deterministic_truncate("A" * 10000, 100)
        assert len(result) < 10000
        assert "deterministic truncation" in result

    def test_truncate_short(self):
        assert _deterministic_truncate("hello", 1000) == "hello"
