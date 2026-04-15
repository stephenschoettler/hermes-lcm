"""Tests for LCM core components: store, DAG, tokens, config, escalation."""

import json
import sqlite3

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.tokens import count_tokens, count_message_tokens, count_messages_tokens
from hermes_lcm.store import MessageStore
from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.escalation import _deterministic_truncate
from hermes_lcm.session_patterns import (
    build_session_match_keys,
    compile_session_pattern,
    compile_session_patterns,
    matches_session_pattern,
)


class TestConfig:
    def test_defaults(self):
        c = LCMConfig()
        assert c.fresh_tail_count == 64
        assert c.leaf_chunk_tokens == 20_000
        assert c.context_threshold == 0.75
        assert c.condensation_fanin == 4
        assert c.ignore_session_patterns == []
        assert c.stateless_session_patterns == []
        assert c.ignore_session_patterns_source == "default"
        assert c.stateless_session_patterns_source == "default"
        assert c.summary_model == ""
        assert c.expansion_model == ""
        assert c.summary_timeout_ms == 60_000
        assert c.expansion_timeout_ms == 120_000

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("LCM_FRESH_TAIL_COUNT", "32")
        monkeypatch.setenv("LCM_CONTEXT_THRESHOLD", "0.80")
        monkeypatch.setenv("LCM_IGNORE_SESSION_PATTERNS", "cron:*,subagent:**")
        monkeypatch.setenv("LCM_STATELESS_SESSION_PATTERNS", "telegram:*, cli:debug")
        monkeypatch.setenv("LCM_EXPANSION_MODEL", "openai/gpt-5.4-mini")
        monkeypatch.setenv("LCM_SUMMARY_TIMEOUT_MS", "45000")
        monkeypatch.setenv("LCM_EXPANSION_TIMEOUT_MS", "90000")
        c = LCMConfig.from_env()
        assert c.fresh_tail_count == 32
        assert c.context_threshold == 0.80
        assert c.ignore_session_patterns == ["cron:*", "subagent:**"]
        assert c.stateless_session_patterns == ["telegram:*", "cli:debug"]
        assert c.ignore_session_patterns_source == "env"
        assert c.stateless_session_patterns_source == "env"
        assert c.expansion_model == "openai/gpt-5.4-mini"
        assert c.summary_timeout_ms == 45_000
        assert c.expansion_timeout_ms == 90_000


class TestSessionPatterns:
    def test_compile_pattern_wildcards(self):
        base_cron = compile_session_pattern("cron:*")
        deep_cron = compile_session_pattern("cron:**")

        assert base_cron.match("cron:job-123")
        assert not base_cron.match("cron:nightly:run-1")
        assert deep_cron.match("cron:nightly:run-1")

    def test_build_session_match_keys(self):
        assert build_session_match_keys("sess-123", platform="cron") == [
            "sess-123",
            "cron",
            "cron:sess-123",
        ]

    def test_matches_any_compiled_pattern(self):
        patterns = compile_session_patterns(["cron:**", "telegram:*"])
        assert matches_session_pattern(
            build_session_match_keys("cron_123", platform="cron"),
            patterns,
        )
        assert matches_session_pattern(
            build_session_match_keys("debug", platform="telegram"),
            patterns,
        )
        assert not matches_session_pattern(
            build_session_match_keys("sess-123", platform="cli"),
            patterns,
        )


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

    def test_init_repairs_malformed_message_fts_and_sets_schema_version(self, tmp_path):
        db_path = tmp_path / "legacy-store.db"
        conn = sqlite3.connect(db_path)
        conn.executescript(
            """
            CREATE TABLE messages (
                store_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                tool_call_id TEXT,
                tool_calls TEXT,
                tool_name TEXT,
                timestamp REAL NOT NULL,
                token_estimate INTEGER DEFAULT 0,
                pinned INTEGER DEFAULT 0
            );
            CREATE TABLE messages_fts (
                rowid INTEGER PRIMARY KEY,
                content TEXT
            );
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            );
            INSERT INTO messages (session_id, role, content, timestamp, token_estimate, pinned)
            VALUES ('sess1', 'user', 'legacy docker migration note', 1.0, 7, 0);
            """
        )
        conn.commit()
        conn.close()

        store = MessageStore(db_path)

        version = store._conn.execute(
            "SELECT value FROM metadata WHERE key = 'schema_version'"
        ).fetchone()
        assert version == ("1",)

        results = store.search("docker", session_id="sess1")
        assert len(results) == 1
        assert results[0]["content"] == "legacy docker migration note"

        store.close()

    def test_init_recreates_missing_message_fts_trigger(self, tmp_path):
        db_path = tmp_path / "legacy-trigger.db"
        conn = sqlite3.connect(db_path)
        conn.executescript(
            """
            CREATE TABLE messages (
                store_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                tool_call_id TEXT,
                tool_calls TEXT,
                tool_name TEXT,
                timestamp REAL NOT NULL,
                token_estimate INTEGER DEFAULT 0,
                pinned INTEGER DEFAULT 0
            );
            CREATE VIRTUAL TABLE messages_fts USING fts5(
                content,
                content=messages,
                content_rowid=store_id
            );
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            );
            """
        )
        conn.commit()
        conn.close()

        store = MessageStore(db_path)
        store.append("sess1", {"role": "user", "content": "fresh searchable message"})

        results = store.search("searchable", session_id="sess1")
        assert len(results) == 1

        trigger_names = {
            row[0]
            for row in store._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger' AND name='msg_fts_insert'"
            ).fetchall()
        }
        assert trigger_names == {"msg_fts_insert"}

        store.close()

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

    def test_init_repairs_malformed_nodes_fts_and_sets_schema_version(self, tmp_path):
        db_path = tmp_path / "legacy-dag.db"
        conn = sqlite3.connect(db_path)
        conn.executescript(
            """
            CREATE TABLE summary_nodes (
                node_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                depth INTEGER NOT NULL DEFAULT 0,
                summary TEXT NOT NULL,
                token_count INTEGER DEFAULT 0,
                source_token_count INTEGER DEFAULT 0,
                source_ids TEXT NOT NULL DEFAULT '[]',
                source_type TEXT NOT NULL DEFAULT 'messages',
                created_at REAL NOT NULL,
                expand_hint TEXT DEFAULT ''
            );
            CREATE TABLE nodes_fts (
                rowid INTEGER PRIMARY KEY,
                summary TEXT
            );
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            );
            INSERT INTO summary_nodes (
                session_id, depth, summary, token_count, source_token_count,
                source_ids, source_type, created_at, expand_hint
            ) VALUES (
                's1', 0, 'legacy summary about docker recovery', 9, 18,
                '[1]', 'messages', 1.0, ''
            );
            """
        )
        conn.commit()
        conn.close()

        dag = SummaryDAG(db_path)

        version = dag._conn.execute(
            "SELECT value FROM metadata WHERE key = 'schema_version'"
        ).fetchone()
        assert version == ("1",)

        results = dag.search("docker", session_id="s1")
        assert len(results) == 1
        assert results[0].summary == "legacy summary about docker recovery"

        dag.close()

    def test_init_recreates_missing_nodes_fts_trigger(self, tmp_path):
        db_path = tmp_path / "legacy-nodes-trigger.db"
        conn = sqlite3.connect(db_path)
        conn.executescript(
            """
            CREATE TABLE summary_nodes (
                node_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                depth INTEGER NOT NULL DEFAULT 0,
                summary TEXT NOT NULL,
                token_count INTEGER DEFAULT 0,
                source_token_count INTEGER DEFAULT 0,
                source_ids TEXT NOT NULL DEFAULT '[]',
                source_type TEXT NOT NULL DEFAULT 'messages',
                created_at REAL NOT NULL,
                expand_hint TEXT DEFAULT ''
            );
            CREATE VIRTUAL TABLE nodes_fts USING fts5(
                summary,
                content=summary_nodes,
                content_rowid=node_id
            );
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            );
            """
        )
        conn.commit()
        conn.close()

        dag = SummaryDAG(db_path)
        dag.add_node(SummaryNode(
            session_id="s1", depth=0, summary="fresh dag search result",
            token_count=5, source_ids=[1], source_type="messages",
        ))

        results = dag.search("fresh", session_id="s1")
        assert len(results) == 1

        trigger_names = {
            row[0]
            for row in dag._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger' AND name='nodes_fts_insert'"
            ).fetchall()
        }
        assert trigger_names == {"nodes_fts_insert"}

        dag.close()

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
