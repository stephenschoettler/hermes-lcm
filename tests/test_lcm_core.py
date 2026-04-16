"""Tests for LCM core components: store, DAG, tokens, config, escalation."""

import json
import sqlite3
import threading
import time

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
        assert c.dynamic_leaf_chunk_enabled is False
        assert c.dynamic_leaf_chunk_max == 40_000
        assert c.cache_friendly_condensation_enabled is False
        assert c.cache_friendly_min_debt_groups == 2
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
        monkeypatch.setenv("LCM_DYNAMIC_LEAF_CHUNK_ENABLED", "1")
        monkeypatch.setenv("LCM_DYNAMIC_LEAF_CHUNK_MAX", "64000")
        monkeypatch.setenv("LCM_CACHE_FRIENDLY_CONDENSATION_ENABLED", "1")
        monkeypatch.setenv("LCM_CACHE_FRIENDLY_MIN_DEBT_GROUPS", "3")
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
        assert c.dynamic_leaf_chunk_enabled is True
        assert c.dynamic_leaf_chunk_max == 64_000
        assert c.cache_friendly_condensation_enabled is True
        assert c.cache_friendly_min_debt_groups == 3

    def test_from_env_invalid_numeric_values_fall_back_to_defaults(self, monkeypatch):
        monkeypatch.setenv("LCM_FRESH_TAIL_COUNT", "not-a-number")
        monkeypatch.setenv("LCM_LEAF_CHUNK_TOKENS", "")
        monkeypatch.setenv("LCM_CONTEXT_THRESHOLD", "bad-float")
        monkeypatch.setenv("LCM_MAX_ASSEMBLY_TOKENS", "nope")
        monkeypatch.setenv("LCM_RESERVE_TOKENS_FLOOR", "still-nope")

        c = LCMConfig.from_env()

        assert c.fresh_tail_count == 64
        assert c.leaf_chunk_tokens == 20_000
        assert c.context_threshold == 0.75
        assert c.max_assembly_tokens == 0
        assert c.reserve_tokens_floor == 0


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
        assert version == ("2",)

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
            INSERT INTO metadata(key, value) VALUES ('schema_version', '1');
            """
        )
        conn.commit()
        conn.close()

        store = MessageStore(db_path)
        store.append("sess1", {"role": "user", "content": "fresh searchable message"})

        results = store.search("searchable", session_id="sess1")
        assert len(results) == 1

        version = store._conn.execute(
            "SELECT value FROM metadata WHERE key = 'schema_version'"
        ).fetchone()
        assert version == ("2",)

        migration_state = store._conn.execute(
            "SELECT step_name FROM lcm_migration_state ORDER BY step_name"
        ).fetchall()
        assert ("v2_external_content_fts_triggers",) in migration_state

        trigger_names = {
            row[0]
            for row in store._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger' AND name IN ('msg_fts_insert', 'msg_fts_delete')"
            ).fetchall()
        }
        assert trigger_names == {"msg_fts_insert", "msg_fts_delete"}

        store.close()

    def test_search_falls_back_to_like_when_message_fts_breaks(self, store):
        store.append("sess1", {"role": "user", "content": "docker fallback search still works"})
        store._conn.execute("DROP TABLE messages_fts")
        store._conn.commit()

        results = store.search("fallback", session_id="sess1")

        assert len(results) == 1
        assert results[0]["content"] == "docker fallback search still works"
        assert "fallback" in results[0]["snippet"].lower()

    def test_init_repairs_message_fts_drifted_row_count(self, tmp_path):
        db_path = tmp_path / "message-fts-drift.db"
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
            CREATE TRIGGER msg_fts_insert
                AFTER INSERT ON messages BEGIN
                INSERT INTO messages_fts(rowid, content)
                    VALUES (new.store_id, new.content);
            END;
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            );
            INSERT INTO metadata(key, value) VALUES ('schema_version', '2');
            INSERT INTO messages(session_id, role, content, timestamp, token_estimate, pinned)
            VALUES ('sess1', 'user', 'drifted search row', 1.0, 3, 0);
            DELETE FROM messages_fts;
            """
        )
        conn.commit()
        conn.close()

        store = MessageStore(db_path)
        results = store.search("drifted", session_id="sess1")

        assert len(results) == 1
        assert results[0]["content"] == "drifted search row"

        fts_count = store._conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]
        assert fts_count == 1
        store.close()

    def test_store_waits_out_long_write_lock_with_extended_busy_timeout(self, tmp_path):
        db_path = tmp_path / "busy-timeout.db"
        store = MessageStore(db_path)

        lock_conn = sqlite3.connect(db_path, timeout=1.0, check_same_thread=False)
        lock_conn.execute("PRAGMA journal_mode=WAL")
        lock_conn.execute("BEGIN IMMEDIATE")
        lock_conn.execute(
            "INSERT INTO messages(session_id, role, content, timestamp, token_estimate, pinned) VALUES (?, ?, ?, ?, ?, ?)",
            ("hold", "user", "holding write lock", 1.0, 1, 0),
        )

        def release_lock():
            time.sleep(6.2)
            lock_conn.commit()
            lock_conn.close()

        releaser = threading.Thread(target=release_lock, daemon=True)
        releaser.start()

        start = time.monotonic()
        store.append("sess1", {"role": "user", "content": "writer survives lock"})
        elapsed = time.monotonic() - start
        releaser.join(timeout=1.0)

        assert elapsed >= 6.0
        assert store.get_session_count("sess1") == 1
        assert store._conn.execute("PRAGMA busy_timeout").fetchone()[0] >= 30000

        store.close()

    def test_search_sort_modes_apply_before_limit(self, store):
        older_strong = store.append(
            "sess1",
            {
                "role": "user",
                "content": "database migration plan database migration plan database migration plan with rollback notes",
            },
        )
        newer_weak = store.append(
            "sess1",
            {
                "role": "assistant",
                "content": "recent status note about the database migration plan",
            },
        )
        store._conn.execute(
            "UPDATE messages SET timestamp = ? WHERE store_id = ?",
            (1_700_000_000, older_strong),
        )
        store._conn.execute(
            "UPDATE messages SET timestamp = ? WHERE store_id = ?",
            (1_800_000_000, newer_weak),
        )
        store._conn.commit()

        recency_results = store.search(
            '"database migration plan"',
            session_id="sess1",
            limit=1,
            sort="recency",
        )
        relevance_results = store.search(
            '"database migration plan"',
            session_id="sess1",
            limit=1,
            sort="relevance",
        )

        assert recency_results[0]["store_id"] == newer_weak
        assert relevance_results[0]["store_id"] == older_strong

    def test_search_cjk_queries_fall_back_with_aligned_sort_modes(self, store):
        older_strong = store.append(
            "sess1",
            {"role": "user", "content": "部署 部署 数据库迁移清单"},
        )
        newer_weak = store.append(
            "sess1",
            {"role": "assistant", "content": "最新部署状态更新"},
        )
        store._conn.execute(
            "UPDATE messages SET timestamp = ? WHERE store_id = ?",
            (1_700_000_000, older_strong),
        )
        store._conn.execute(
            "UPDATE messages SET timestamp = ? WHERE store_id = ?",
            (1_800_000_000, newer_weak),
        )
        store._conn.commit()

        recency_results = store.search("部署", session_id="sess1", limit=1, sort="recency")
        relevance_results = store.search("部署", session_id="sess1", limit=1, sort="relevance")

        assert recency_results[0]["store_id"] == newer_weak
        assert relevance_results[0]["store_id"] == older_strong

    def test_search_emoji_queries_fall_back_with_aligned_sort_modes(self, store):
        older_strong = store.append(
            "sess1",
            {"role": "user", "content": "🚀 🚀 launch checklist"},
        )
        newer_weak = store.append(
            "sess1",
            {"role": "assistant", "content": "fresh 🚀 status"},
        )
        store._conn.execute(
            "UPDATE messages SET timestamp = ? WHERE store_id = ?",
            (1_700_000_000, older_strong),
        )
        store._conn.execute(
            "UPDATE messages SET timestamp = ? WHERE store_id = ?",
            (1_800_000_000, newer_weak),
        )
        store._conn.commit()

        recency_results = store.search("🚀", session_id="sess1", limit=1, sort="recency")
        relevance_results = store.search("🚀", session_id="sess1", limit=1, sort="relevance")

        assert recency_results[0]["store_id"] == newer_weak
        assert relevance_results[0]["store_id"] == older_strong

    def test_search_hyphenated_operator_queries_fall_back_cleanly(self, store):
        target = store.append(
            "sess1",
            {
                "role": "user",
                "content": "hermes-lcm plugin-only external context-engine generic host support no vendoring stays external",
            },
        )
        store.append(
            "sess1",
            {
                "role": "assistant",
                "content": "or or or filler words without the target concepts",
            },
        )

        query = "8416 OR vendored OR vendoring OR plugin-only OR external context-engine OR generic host support OR hermes-lcm stays external OR no vendoring"
        results = store.search(query, session_id="sess1", limit=5, sort="relevance")

        assert len(results) == 1
        assert results[0]["store_id"] == target
        assert results[0]["snippet"]

    def test_search_prefers_conversational_hits_over_tool_output_noise(self, store):
        user_id = store.append(
            "sess1",
            {
                "role": "user",
                "content": "vendoring external plugin support should stay generic host support only",
            },
        )
        tool_id = store.append(
            "sess1",
            {
                "role": "tool",
                "content": '{"vendoring":"vendoring vendoring vendoring","payload":"external plugin generic host support"}',
            },
        )

        relevance_results = store.search("vendoring", session_id="sess1", limit=2, sort="relevance")
        fallback_results = store.search("hermes-lcm", session_id="sess1", limit=2, sort="relevance")

        assert relevance_results[0]["store_id"] == user_id
        assert relevance_results[1]["store_id"] == tool_id

        fallback_user_id = store.append(
            "sess1",
            {
                "role": "assistant",
                "content": "hermes-lcm should stay external and plugin-only in practice",
            },
        )
        fallback_tool_id = store.append(
            "sess1",
            {
                "role": "tool",
                "content": '{"query":"hermes-lcm","matches":["hermes-lcm","hermes-lcm"]}',
            },
        )
        fallback_results = store.search("hermes-lcm", session_id="sess1", limit=2, sort="relevance")
        assert fallback_results[0]["store_id"] == fallback_user_id
        assert fallback_results[1]["store_id"] == fallback_tool_id

    def test_search_relevance_prefers_user_over_newer_assistant_on_similar_match(self, store):
        user_id = store.append(
            "sess1",
            {
                "role": "user",
                "content": "vendoring should stay external plugin host support only",
            },
        )
        assistant_id = store.append(
            "sess1",
            {
                "role": "assistant",
                "content": "vendoring should stay external plugin host support only",
            },
        )

        results = store.search("vendoring", session_id="sess1", limit=2, sort="relevance")

        assert results[0]["store_id"] == user_id
        assert results[1]["store_id"] == assistant_id

    def test_search_relevance_does_not_let_weaker_user_hit_beat_stronger_assistant_hit(self, store):
        weaker_user_id = store.append(
            "sess1",
            {
                "role": "user",
                "content": "vendoring blah blah external blah host",
            },
        )
        stronger_assistant_id = store.append(
            "sess1",
            {
                "role": "assistant",
                "content": "vendoring external host",
            },
        )

        results = store.search("vendoring external host", session_id="sess1", limit=2, sort="relevance")

        assert results[0]["store_id"] == stronger_assistant_id
        assert results[1]["store_id"] == weaker_user_id

    def test_search_relevance_still_surfaces_preferred_user_hit_from_large_same_rank_pool(self, store):
        preferred_user_id = store.append(
            "sess1",
            {
                "role": "user",
                "content": "vendoring",
            },
        )
        for _ in range(150):
            store.append(
                "sess1",
                {
                    "role": "assistant",
                    "content": "vendoring",
                },
            )

        results = store.search("vendoring", session_id="sess1", limit=5, sort="relevance")

        assert results[0]["store_id"] == preferred_user_id
        assert results[0]["role"] == "user"

    def test_search_relevance_top_results_do_not_change_when_limit_increases_on_large_single_term_pool(self, store):
        store.append(
            "sess1",
            {
                "role": "user",
                "content": "vendoring",
            },
        )
        for idx in range(250):
            content = (
                '{"vendoring":"vendoring vendoring vendoring"}'
                if idx % 5 == 0
                else "vendoring vendoring vendoring vendoring vendoring spam"
            )
            store.append(
                "sess1",
                {
                    "role": "tool" if idx % 5 == 0 else "assistant",
                    "content": content,
                },
            )
        top_5 = [result["store_id"] for result in store.search("vendoring", session_id="sess1", limit=5, sort="relevance")]
        top_50 = [result["store_id"] for result in store.search("vendoring", session_id="sess1", limit=50, sort="relevance")[:5]]

        assert top_5 == top_50

    def test_search_relevance_prefers_assistant_over_tool_on_similar_match(self, store):
        assistant_id = store.append(
            "sess1",
            {
                "role": "assistant",
                "content": "plugin-only support should stay external and generic",
            },
        )
        tool_id = store.append(
            "sess1",
            {
                "role": "tool",
                "content": "plugin-only support should stay external and generic",
            },
        )

        results = store.search("plugin-only", session_id="sess1", limit=2, sort="relevance")

        assert results[0]["store_id"] == assistant_id
        assert results[1]["store_id"] == tool_id

    def test_search_relevance_still_returns_tool_when_it_is_only_real_hit(self, store):
        tool_id = store.append(
            "sess1",
            {
                "role": "tool",
                "content": '{"verdict":"tool-only hit about vendoring boundaries"}',
            },
        )

        results = store.search("tool-only", session_id="sess1", limit=2, sort="relevance")

        assert len(results) == 1
        assert results[0]["store_id"] == tool_id

    def test_search_relevance_prefers_direct_hit_over_repetition_spam_for_single_term_query(self, store):
        spam_id = store.append(
            "sess1",
            {
                "role": "assistant",
                "content": "query audit notes: vendoring vendoring vendoring vendoring vendoring",
            },
        )
        direct_id = store.append(
            "sess1",
            {
                "role": "assistant",
                "content": "Keep vendoring out of hermes-agent.",
            },
        )

        results = store.search("vendoring", session_id="sess1", limit=2, sort="relevance")

        assert results[0]["store_id"] == direct_id
        assert results[1]["store_id"] == spam_id

    def test_search_relevance_still_surfaces_direct_phrase_hit_when_phrase_matches_many_spammy_candidates(self, store):
        for _ in range(150):
            store.append(
                "sess1",
                {
                    "role": "assistant",
                    "content": 'vendoring external vendoring external vendoring external spam note',
                },
            )
        direct_id = store.append(
            "sess1",
            {
                "role": "assistant",
                "content": "Keep vendoring external support plugin-only.",
            },
        )

        results = store.search('"vendoring external"', session_id="sess1", limit=5, sort="relevance")

        assert direct_id in [result["store_id"] for result in results]

    def test_search_relevance_prefers_direct_phrase_hit_over_repeated_phrase_with_varied_filler(self, store):
        spam_id = store.append(
            "sess1",
            {
                "role": "assistant",
                "content": "vendoring external rollout checklist vendoring external support matrix vendoring external adapter notes",
            },
        )
        direct_id = store.append(
            "sess1",
            {
                "role": "assistant",
                "content": "Keep vendoring external support plugin-only.",
            },
        )

        results = store.search('"vendoring external"', session_id="sess1", limit=2, sort="relevance")

        assert results[0]["store_id"] == direct_id
        assert results[1]["store_id"] == spam_id

    def test_search_relevance_prefers_direct_phrase_hit_over_repeated_phrase_with_richer_filler(self, store):
        spam_id = store.append(
            "sess1",
            {
                "role": "assistant",
                "content": "vendoring external rollout checklist vendoring external support matrix vendoring external adapter integration notes",
            },
        )
        direct_id = store.append(
            "sess1",
            {
                "role": "assistant",
                "content": "Keep vendoring external support plugin-only.",
            },
        )

        results = store.search('"vendoring external"', session_id="sess1", limit=2, sort="relevance")

        assert results[0]["store_id"] == direct_id
        assert results[1]["store_id"] == spam_id

    def test_search_relevance_still_surfaces_direct_phrase_hit_when_phrase_plus_extra_term_matches_many_spammy_candidates(self, store):
        for idx in range(25):
            store.append(
                "sess1",
                {
                    "role": "assistant",
                    "content": f"vendoring external plugin rollout {idx} vendoring external plugin support {idx}",
                },
            )
        direct_id = store.append(
            "sess1",
            {
                "role": "assistant",
                "content": "Keep vendoring external plugin support simple.",
            },
        )

        results = store.search('"vendoring external" plugin', session_id="sess1", limit=5, sort="relevance")

        assert direct_id in [result["store_id"] for result in results]
        assert results[0]["store_id"] == direct_id

    def test_search_relevance_prefers_direct_phrase_hit_over_repeated_non_phrase_term_spam(self, store):
        for idx in range(30):
            store.append(
                "sess1",
                {
                    "role": "assistant",
                    "content": f"vendoring external plugin plugin plugin plugin {idx}",
                },
            )
        direct_id = store.append(
            "sess1",
            {
                "role": "assistant",
                "content": "Keep vendoring external plugin support simple.",
            },
        )

        results = store.search('"vendoring external" plugin', session_id="sess1", limit=5, sort="relevance")

        assert direct_id in [result["store_id"] for result in results]
        assert results[0]["store_id"] == direct_id

    def test_search_like_fallback_strips_unmatched_quote_characters(self, store):
        direct_id = store.append(
            "sess1",
            {
                "role": "assistant",
                "content": "Keep vendoring out of hermes-agent.",
            },
        )

        results = store.search('"vendoring', session_id="sess1", limit=5, sort="relevance")

        assert [result["store_id"] for result in results] == [direct_id]

    def test_search_recency_same_timestamp_pool_is_limit_stable(self, store):
        ids = store.append_batch(
            "sess1",
            [
                {
                    "role": "assistant",
                    "content": f"alpha alpha alpha beta beta gamma gamma gamma spam {idx}",
                }
                for idx in range(120)
            ] + [
                {
                    "role": "assistant",
                    "content": "keep alpha beta gamma concise",
                }
            ],
        )
        timestamp = store.get(ids[0])["timestamp"]

        short_results = store.search("alpha beta gamma", session_id="sess1", limit=5, sort="recency")
        long_results = store.search("alpha beta gamma", session_id="sess1", limit=200, sort="recency")

        assert [result["timestamp"] for result in short_results] == [timestamp] * len(short_results)
        assert [result["store_id"] for result in short_results] == [result["store_id"] for result in long_results[:5]]

    def test_search_hybrid_clamps_future_timestamps_consistently(self, store):
        now = time.time()
        future = now + (60 * 24 * 3600)
        current_ids = [
            store.append(
                "sess1",
                {"role": "assistant", "content": "vendoring external"},
            )
            for _ in range(20)
        ]
        future_id = store.append(
            "sess1",
            {"role": "assistant", "content": "vendoring external"},
        )
        for current_id in current_ids:
            store._conn.execute("UPDATE messages SET timestamp = ? WHERE store_id = ?", (now, current_id))
        store._conn.execute("UPDATE messages SET timestamp = ? WHERE store_id = ?", (future, future_id))
        store._conn.commit()

        results = store.search("vendoring external", session_id="sess1", limit=1, sort="hybrid")

        assert [result["store_id"] for result in results] == [future_id]

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
        assert version == ("2",)

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
            INSERT INTO metadata(key, value) VALUES ('schema_version', '1');
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

        version = dag._conn.execute(
            "SELECT value FROM metadata WHERE key = 'schema_version'"
        ).fetchone()
        assert version == ("2",)

        migration_state = dag._conn.execute(
            "SELECT step_name FROM lcm_migration_state ORDER BY step_name"
        ).fetchall()
        assert ("v2_external_content_fts_triggers",) in migration_state

        trigger_names = {
            row[0]
            for row in dag._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger' AND name IN ('nodes_fts_insert', 'nodes_fts_delete')"
            ).fetchall()
        }
        assert trigger_names == {"nodes_fts_insert", "nodes_fts_delete"}

        dag.close()

    def test_search_falls_back_to_like_when_nodes_fts_breaks(self, dag):
        dag.add_node(SummaryNode(
            session_id="s1", depth=0,
            summary="dag fallback search still works",
            token_count=10, source_ids=[1], source_type="messages",
        ))
        dag._conn.execute("DROP TABLE nodes_fts")
        dag._conn.commit()

        results = dag.search("fallback", session_id="s1")

        assert len(results) == 1
        assert results[0].summary == "dag fallback search still works"

    def test_init_repairs_nodes_fts_drifted_row_count(self, tmp_path):
        db_path = tmp_path / "nodes-fts-drift.db"
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
            CREATE TRIGGER nodes_fts_insert
                AFTER INSERT ON summary_nodes BEGIN
                INSERT INTO nodes_fts(rowid, summary)
                    VALUES (new.node_id, new.summary);
            END;
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            );
            INSERT INTO metadata(key, value) VALUES ('schema_version', '2');
            INSERT INTO summary_nodes(
                session_id, depth, summary, token_count, source_token_count,
                source_ids, source_type, created_at, expand_hint
            ) VALUES (
                's1', 0, 'drifted dag row', 5, 10,
                '[1]', 'messages', 1.0, ''
            );
            DELETE FROM nodes_fts;
            """
        )
        conn.commit()
        conn.close()

        dag = SummaryDAG(db_path)
        results = dag.search("drifted", session_id="s1")

        assert len(results) == 1
        assert results[0].summary == "drifted dag row"

        fts_count = dag._conn.execute("SELECT COUNT(*) FROM nodes_fts").fetchone()[0]
        assert fts_count == 1
        dag.close()

    def test_add_and_get_preserves_source_window_timestamps(self, dag):
        node_id = dag.add_node(SummaryNode(
            session_id="s1", depth=0,
            summary="Planning notes",
            token_count=10, source_ids=[1], source_type="messages",
            created_at=1_900_000_000,
            earliest_at=1_700_000_000,
            latest_at=1_800_000_000,
        ))

        node = dag.get_node(node_id)
        assert node.earliest_at == 1_700_000_000
        assert node.latest_at == 1_800_000_000

    def test_existing_db_is_upgraded_with_summary_source_window_columns(self, tmp_path):
        db_path = tmp_path / "legacy_dag.db"
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
            CREATE TRIGGER nodes_fts_insert
                AFTER INSERT ON summary_nodes BEGIN
                INSERT INTO nodes_fts(rowid, summary)
                    VALUES (new.node_id, new.summary);
            END;
            """
        )
        conn.commit()
        conn.close()

        dag = SummaryDAG(db_path)
        columns = {
            row[1] for row in dag._conn.execute("PRAGMA table_info(summary_nodes)").fetchall()
        }

        assert "earliest_at" in columns
        assert "latest_at" in columns

        dag.close()

    def test_search_sort_modes_apply_before_limit(self, dag):
        older_strong = dag.add_node(SummaryNode(
            session_id="s1", depth=0,
            summary="error handling checklist error handling checklist error handling checklist with confirmed fixes",
            token_count=18, source_ids=[1], source_type="messages",
            created_at=1_700_000_000,
            earliest_at=1_700_000_000,
            latest_at=1_700_000_000,
        ))
        newer_weak = dag.add_node(SummaryNode(
            session_id="s1", depth=0,
            summary="recent note mentioning the error handling checklist",
            token_count=9, source_ids=[2], source_type="messages",
            created_at=1_700_086_400,
            earliest_at=1_700_086_400,
            latest_at=1_700_086_400,
        ))

        recency_results = dag.search(
            '"error handling checklist"',
            session_id="s1",
            limit=1,
            sort="recency",
        )
        hybrid_results = dag.search(
            '"error handling checklist"',
            session_id="s1",
            limit=1,
            sort="hybrid",
        )

        assert recency_results[0].node_id == newer_weak
        assert hybrid_results[0].node_id == older_strong

    def test_search_cjk_queries_fall_back_with_aligned_sort_modes(self, dag):
        older_strong = dag.add_node(SummaryNode(
            session_id="s1", depth=0,
            summary="部署 部署 数据库迁移清单",
            token_count=12, source_ids=[1], source_type="messages",
            created_at=1_700_000_000,
            earliest_at=1_700_000_000,
            latest_at=1_700_000_000,
        ))
        newer_weak = dag.add_node(SummaryNode(
            session_id="s1", depth=0,
            summary="最新部署状态更新",
            token_count=8, source_ids=[2], source_type="messages",
            created_at=1_800_000_000,
            earliest_at=1_800_000_000,
            latest_at=1_800_000_000,
        ))

        recency_results = dag.search("部署", session_id="s1", limit=1, sort="recency")
        relevance_results = dag.search("部署", session_id="s1", limit=1, sort="relevance")

        assert recency_results[0].node_id == newer_weak
        assert relevance_results[0].node_id == older_strong

    def test_search_emoji_queries_fall_back_with_aligned_sort_modes(self, dag):
        older_strong = dag.add_node(SummaryNode(
            session_id="s1", depth=0,
            summary="🚀 🚀 launch checklist",
            token_count=12, source_ids=[1], source_type="messages",
            created_at=1_700_000_000,
            earliest_at=1_700_000_000,
            latest_at=1_700_000_000,
        ))
        newer_weak = dag.add_node(SummaryNode(
            session_id="s1", depth=0,
            summary="fresh 🚀 status",
            token_count=8, source_ids=[2], source_type="messages",
            created_at=1_800_000_000,
            earliest_at=1_800_000_000,
            latest_at=1_800_000_000,
        ))

        recency_results = dag.search("🚀", session_id="s1", limit=1, sort="recency")
        relevance_results = dag.search("🚀", session_id="s1", limit=1, sort="relevance")

        assert recency_results[0].node_id == newer_weak
        assert relevance_results[0].node_id == older_strong

    def test_search_hyphenated_operator_queries_fall_back_cleanly(self, dag):
        target = dag.add_node(SummaryNode(
            session_id="s1", depth=0,
            summary="hermes-lcm plugin-only external context-engine generic host support no vendoring stays external",
            token_count=10, source_ids=[1], source_type="messages",
            created_at=1_700_000_000,
            earliest_at=1_700_000_000,
            latest_at=1_700_000_000,
        ))
        dag.add_node(SummaryNode(
            session_id="s1", depth=0,
            summary="or or or filler words without the target concepts",
            token_count=10, source_ids=[2], source_type="messages",
            created_at=1_800_000_000,
            earliest_at=1_800_000_000,
            latest_at=1_800_000_000,
        ))

        query = "8416 OR vendored OR vendoring OR plugin-only OR external context-engine OR generic host support OR hermes-lcm stays external OR no vendoring"
        results = dag.search(query, session_id="s1", limit=5, sort="relevance")

        assert len(results) == 1
        assert results[0].node_id == target

    def test_search_hybrid_clamps_future_timestamps_consistently(self, dag):
        now = time.time()
        future = now + (60 * 24 * 3600)
        future_node = dag.add_node(SummaryNode(
            session_id="s1", depth=0,
            summary="vendoring",
            token_count=10, source_ids=[3001], source_type="messages",
            created_at=future,
            earliest_at=future,
            latest_at=future,
        ))
        current_node = dag.add_node(SummaryNode(
            session_id="s1", depth=0,
            summary="vendoring",
            token_count=10, source_ids=[3002], source_type="messages",
            created_at=now,
            earliest_at=now,
            latest_at=now,
        ))

        results = dag.search("vendoring", session_id="s1", limit=2, sort="hybrid")

        assert [node.node_id for node in results] == [future_node, current_node]

    def test_search_relevance_prefers_direct_summary_hit_over_repetition_spam_for_single_term_query(self, dag):
        spammy = dag.add_node(SummaryNode(
            session_id="s1", depth=0,
            summary="Summary notes: vendoring vendoring vendoring vendoring vendoring",
            token_count=10, source_ids=[1], source_type="messages",
            created_at=1_700_000_000,
            earliest_at=1_700_000_000,
            latest_at=1_700_000_000,
        ))
        direct = dag.add_node(SummaryNode(
            session_id="s1", depth=0,
            summary="Keep vendoring out of hermes-agent.",
            token_count=10, source_ids=[2], source_type="messages",
            created_at=1_699_999_000,
            earliest_at=1_699_999_000,
            latest_at=1_699_999_000,
        ))

        results = dag.search("vendoring", session_id="s1", limit=2, sort="relevance")

        assert results[0].node_id == direct
        assert results[1].node_id == spammy

    def test_search_relevance_still_surfaces_direct_summary_when_single_term_matches_many_spammy_candidates(self, dag):
        for idx in range(150):
            dag.add_node(SummaryNode(
                session_id="s1", depth=0,
                summary=f"Summary spam {idx}: vendoring vendoring vendoring vendoring vendoring",
                token_count=10, source_ids=[idx + 1], source_type="messages",
                created_at=1_700_000_000 + idx,
                earliest_at=1_700_000_000 + idx,
                latest_at=1_700_000_000 + idx,
            ))
        direct = dag.add_node(SummaryNode(
            session_id="s1", depth=0,
            summary="Keep vendoring out of hermes-agent.",
            token_count=10, source_ids=[999], source_type="messages",
            created_at=1_699_999_000,
            earliest_at=1_699_999_000,
            latest_at=1_699_999_000,
        ))

        results = dag.search("vendoring", session_id="s1", limit=5, sort="relevance")

        assert [node.node_id for node in results[:1]] == [direct]
        assert direct in [node.node_id for node in results]

    def test_search_relevance_prefers_direct_summary_over_risky_ascii_repetition_spam_in_like_fallback(self, dag):
        spammy = dag.add_node(SummaryNode(
            session_id="s1", depth=0,
            summary="plugin-only plugin-only plugin-only plugin-only status dump",
            token_count=10, source_ids=[1001], source_type="messages",
            created_at=1_700_000_100,
            earliest_at=1_700_000_100,
            latest_at=1_700_000_100,
        ))
        direct = dag.add_node(SummaryNode(
            session_id="s1", depth=0,
            summary="Keep plugin-only support external.",
            token_count=10, source_ids=[1002], source_type="messages",
            created_at=1_700_000_000,
            earliest_at=1_700_000_000,
            latest_at=1_700_000_000,
        ))

        results = dag.search("plugin-only", session_id="s1", limit=2, sort="relevance")

        assert results[0].node_id == direct
        assert results[1].node_id == spammy

    def test_search_relevance_still_surfaces_direct_phrase_summary_when_phrase_matches_many_spammy_candidates(self, dag):
        for idx in range(150):
            dag.add_node(SummaryNode(
                session_id="s1", depth=0,
                summary=f"Summary spam {idx}: vendoring external vendoring external vendoring external status",
                token_count=10, source_ids=[2000 + idx], source_type="messages",
                created_at=1_700_000_000 + idx,
                earliest_at=1_700_000_000 + idx,
                latest_at=1_700_000_000 + idx,
            ))
        direct = dag.add_node(SummaryNode(
            session_id="s1", depth=0,
            summary="Keep vendoring external support plugin-only.",
            token_count=10, source_ids=[9999], source_type="messages",
            created_at=1_699_999_000,
            earliest_at=1_699_999_000,
            latest_at=1_699_999_000,
        ))

        results = dag.search('"vendoring external"', session_id="s1", limit=5, sort="relevance")

        assert direct in [node.node_id for node in results]

    def test_search_relevance_prefers_direct_phrase_summary_over_repeated_phrase_with_varied_filler(self, dag):
        spammy = dag.add_node(SummaryNode(
            session_id="s1", depth=0,
            summary="vendoring external rollout checklist vendoring external support matrix vendoring external adapter notes",
            token_count=10, source_ids=[4001], source_type="messages",
            created_at=1_700_000_100,
            earliest_at=1_700_000_100,
            latest_at=1_700_000_100,
        ))
        direct = dag.add_node(SummaryNode(
            session_id="s1", depth=0,
            summary="Keep vendoring external support plugin-only.",
            token_count=10, source_ids=[4002], source_type="messages",
            created_at=1_700_000_000,
            earliest_at=1_700_000_000,
            latest_at=1_700_000_000,
        ))

        results = dag.search('"vendoring external"', session_id="s1", limit=2, sort="relevance")

        assert results[0].node_id == direct
        assert results[1].node_id == spammy

    def test_search_relevance_prefers_direct_phrase_summary_over_repeated_phrase_with_richer_filler(self, dag):
        spammy = dag.add_node(SummaryNode(
            session_id="s1", depth=0,
            summary="vendoring external rollout checklist vendoring external support matrix vendoring external adapter integration notes",
            token_count=10, source_ids=[4011], source_type="messages",
            created_at=1_700_000_100,
            earliest_at=1_700_000_100,
            latest_at=1_700_000_100,
        ))
        direct = dag.add_node(SummaryNode(
            session_id="s1", depth=0,
            summary="Keep vendoring external support plugin-only.",
            token_count=10, source_ids=[4012], source_type="messages",
            created_at=1_700_000_000,
            earliest_at=1_700_000_000,
            latest_at=1_700_000_000,
        ))

        results = dag.search('"vendoring external"', session_id="s1", limit=2, sort="relevance")

        assert results[0].node_id == direct
        assert results[1].node_id == spammy

    def test_search_relevance_still_surfaces_direct_phrase_summary_when_phrase_plus_extra_term_matches_many_spammy_candidates(self, dag):
        for idx in range(25):
            dag.add_node(SummaryNode(
                session_id="s1", depth=0,
                summary=f"vendoring external plugin rollout {idx} vendoring external plugin support {idx}",
                token_count=10, source_ids=[5000 + idx], source_type="messages",
                created_at=1_700_000_000 + idx,
                earliest_at=1_700_000_000 + idx,
                latest_at=1_700_000_000 + idx,
            ))
        direct = dag.add_node(SummaryNode(
            session_id="s1", depth=0,
            summary="Keep vendoring external plugin support simple.",
            token_count=10, source_ids=[5999], source_type="messages",
            created_at=1_699_999_000,
            earliest_at=1_699_999_000,
            latest_at=1_699_999_000,
        ))

        results = dag.search('"vendoring external" plugin', session_id="s1", limit=5, sort="relevance")

        assert direct in [node.node_id for node in results]
        assert results[0].node_id == direct

    def test_search_relevance_prefers_direct_phrase_summary_over_repeated_non_phrase_term_spam(self, dag):
        for idx in range(30):
            dag.add_node(SummaryNode(
                session_id="s1", depth=0,
                summary=f"vendoring external plugin plugin plugin plugin {idx}",
                token_count=10, source_ids=[6100 + idx], source_type="messages",
                created_at=1_700_000_000 + idx,
                earliest_at=1_700_000_000 + idx,
                latest_at=1_700_000_000 + idx,
            ))
        direct = dag.add_node(SummaryNode(
            session_id="s1", depth=0,
            summary="Keep vendoring external plugin support simple.",
            token_count=10, source_ids=[6999], source_type="messages",
            created_at=1_699_999_000,
            earliest_at=1_699_999_000,
            latest_at=1_699_999_000,
        ))

        results = dag.search('"vendoring external" plugin', session_id="s1", limit=5, sort="relevance")

        assert direct in [node.node_id for node in results]
        assert results[0].node_id == direct

    def test_search_like_fallback_strips_unmatched_quote_characters_for_summaries(self, dag):
        direct = dag.add_node(SummaryNode(
            session_id="s1", depth=0,
            summary="Keep vendoring out of hermes-agent.",
            token_count=10, source_ids=[7100], source_type="messages",
            created_at=1_700_000_000,
            earliest_at=1_700_000_000,
            latest_at=1_700_000_000,
        ))

        results = dag.search('"vendoring', session_id="s1", limit=5, sort="relevance")

        assert [node.node_id for node in results] == [direct]

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
