"""Tests for LCM core components: store, DAG, tokens, config, escalation."""

import json
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.tokens import count_tokens, count_message_tokens, count_messages_tokens
from hermes_lcm.store import MessageStore
from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.escalation import _deterministic_truncate
from hermes_lcm.lifecycle_state import LifecycleStateStore
from hermes_lcm.db_bootstrap import ExternalContentFtsSpec, ensure_external_content_fts
from hermes_lcm.search_query import sanitize_fts5_query
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
        assert c.custom_instructions == ""
        assert c.extraction_enabled is False
        assert c.extraction_model == ""
        assert c.extraction_output_path == ""
        assert c.large_output_externalization_enabled is False
        assert c.large_output_externalization_threshold_chars == 12_000
        assert c.large_output_externalization_path == ""
        assert c.large_output_transcript_gc_enabled is False
        assert c.deferred_maintenance_enabled is False
        assert c.deferred_maintenance_max_passes == 4
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
        monkeypatch.setenv("LCM_CUSTOM_INSTRUCTIONS", "Write as a neutral documenter.")
        monkeypatch.setenv("LCM_EXTRACTION_ENABLED", "true")
        monkeypatch.setenv("LCM_EXTRACTION_MODEL", "openai/gpt-5.4-mini")
        monkeypatch.setenv("LCM_EXTRACTION_OUTPUT_PATH", "/tmp/extractions")
        monkeypatch.setenv("LCM_LARGE_OUTPUT_EXTERNALIZATION_ENABLED", "true")
        monkeypatch.setenv("LCM_LARGE_OUTPUT_EXTERNALIZATION_THRESHOLD_CHARS", "4096")
        monkeypatch.setenv("LCM_LARGE_OUTPUT_EXTERNALIZATION_PATH", "/tmp/lcm-large-outputs")
        monkeypatch.setenv("LCM_LARGE_OUTPUT_TRANSCRIPT_GC_ENABLED", "true")
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
        assert c.custom_instructions == "Write as a neutral documenter."
        assert c.extraction_enabled is True
        assert c.extraction_model == "openai/gpt-5.4-mini"
        assert c.extraction_output_path == "/tmp/extractions"
        assert c.large_output_externalization_enabled is True
        assert c.large_output_externalization_threshold_chars == 4096
        assert c.large_output_externalization_path == "/tmp/lcm-large-outputs"
        assert c.large_output_transcript_gc_enabled is True

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

    def test_source_stored_and_filterable(self, store):
        store.append("sess1", {"role": "user", "content": "docker in cli"}, source="cli")
        store.append("sess2", {"role": "user", "content": "docker in discord"}, source="discord")

        cli_results = store.search("docker", source="cli")
        discord_results = store.search("docker", source="discord")

        assert len(cli_results) == 1
        assert cli_results[0]["source"] == "cli"
        assert cli_results[0]["session_id"] == "sess1"

        assert len(discord_results) == 1
        assert discord_results[0]["source"] == "discord"
        assert discord_results[0]["session_id"] == "sess2"

    def test_missing_source_is_normalized_to_unknown_and_filterable(self, store):
        store_id = store.append("sess-unknown", {"role": "user", "content": "docker with unknown source"})

        stored = store.get(store_id)
        unknown_results = store.search("docker", source="unknown")

        assert stored["source"] == "unknown"
        assert len(unknown_results) == 1
        assert unknown_results[0]["store_id"] == store_id
        assert unknown_results[0]["source"] == "unknown"

    def test_source_unknown_filter_matches_legacy_blank_source_rows(self, tmp_path):
        db_path = tmp_path / "legacy-unknown-source.db"
        store = MessageStore(db_path)
        store._conn.execute(
            """INSERT INTO messages
               (session_id, source, role, content, tool_call_id, tool_calls, tool_name, timestamp, token_estimate, pinned)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("legacy-session", "", "user", "docker with blank source", None, None, None, 1.0, 5, 0),
        )
        store._conn.commit()

        result = store.search("docker", source="unknown")
        fetched = store.get(1)

        assert len(result) == 1
        assert result[0]["session_id"] == "legacy-session"
        assert result[0]["source"] == "unknown"
        assert fetched["source"] == "unknown"

        store.close()

    def test_get_source_stats_reports_attributed_unknown_and_legacy_blank_counts(self, tmp_path):
        db_path = tmp_path / "source-stats.db"
        store = MessageStore(db_path)
        store.append("sess-known", {"role": "user", "content": "cli message"}, source="cli")
        store.append("sess-unknown", {"role": "user", "content": "unknown message"})
        store._conn.execute(
            """INSERT INTO messages
               (session_id, source, role, content, tool_call_id, tool_calls, tool_name, timestamp, token_estimate, pinned)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("legacy-session", "", "user", "legacy blank source", None, None, None, 1.0, 5, 0),
        )
        store._conn.commit()

        stats = store.get_source_stats()

        assert stats["messages_total"] == 3
        assert stats["attributed_messages"] == 1
        assert stats["normalized_unknown_messages"] == 1
        assert stats["legacy_blank_source_messages"] == 1
        assert stats["effective_unknown_messages"] == 2

        store.close()

    def test_gc_externalized_tool_result_rewrites_content_and_updates_fts(self, store):
        placeholder = "[GC'd externalized tool output: tool_call_id=call_gc; ref=payload.json]"
        store_id = store.append(
            "sess1",
            {"role": "tool", "tool_call_id": "call_gc", "content": "raw payload blob should disappear"},
            token_estimate=50,
        )

        rewritten = store.gc_externalized_tool_result(store_id, placeholder)

        assert rewritten is True
        updated = store.get(store_id)
        assert updated["content"] == placeholder
        assert updated["token_estimate"] == count_message_tokens(
            {"role": "tool", "tool_call_id": "call_gc", "content": placeholder}
        )
        assert updated["token_estimate"] < 50
        assert store.search("payload", session_id="sess1")[0]["store_id"] == store_id
        assert store.search("blob", session_id="sess1") == []

    def test_gc_externalized_tool_result_skips_pinned_messages(self, store):
        store_id = store.append(
            "sess1",
            {"role": "tool", "tool_call_id": "call_gc", "content": "raw payload blob should stay"},
            token_estimate=50,
        )
        store.pin(store_id)

        rewritten = store.gc_externalized_tool_result(
            store_id,
            "[GC'd externalized tool output: tool_call_id=call_gc; ref=payload.json]",
        )

        assert rewritten is False
        assert store.get(store_id)["content"] == "raw payload blob should stay"

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
        assert version == ("4",)

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
        assert version == ("4",)

        migration_state = store._conn.execute(
            "SELECT step_name FROM lcm_migration_state ORDER BY step_name"
        ).fetchall()
        assert ("v2_external_content_fts_triggers",) in migration_state
        assert ("v4_lifecycle_debt_columns",) in migration_state

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

    def test_search_like_fallback_sanitizes_fts_syntax_chars(self, store):
        store.append("sess1", {"role": "user", "content": "vendoring external support stays plugin-only"})

        results = store.search('"vendoring*', session_id="sess1")

        assert len(results) == 1
        assert results[0]["content"] == "vendoring external support stays plugin-only"

    def test_search_like_fallback_splits_unbalanced_quote_terms(self, store):
        store.append("sess1", {"role": "user", "content": "foo bar baz"})

        results = store.search('foo"bar', session_id="sess1")

        assert len(results) == 1
        assert results[0]["content"] == "foo bar baz"

    def test_search_uses_sanitized_terms_for_directness_scoring(self, store):
        store.append("sess1", {"role": "user", "content": "vendoring external support stays plugin-only"})

        results = store.search("vendoring*", session_id="sess1")

        assert len(results) == 1
        assert results[0]["_directness_score"] > 0

    def test_search_sanitizes_fts_wildcards_without_prefix_matching(self, store):
        store.append("sess1", {"role": "user", "content": "dockerization notes"})

        results = store.search("docker*", session_id="sess1")

        assert results == []

    def test_init_low_disk_degrades_without_leaving_broken_message_fts_triggers(self, tmp_path, monkeypatch):
        db_path = tmp_path / "low-disk-broken-message-fts.db"
        conn = sqlite3.connect(db_path)
        conn.executescript(
            """
            CREATE TABLE messages (
                store_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                source TEXT DEFAULT 'unknown',
                role TEXT NOT NULL,
                content TEXT,
                tool_call_id TEXT,
                tool_calls TEXT,
                tool_name TEXT,
                timestamp REAL NOT NULL,
                token_estimate INTEGER DEFAULT 0,
                pinned INTEGER DEFAULT 0
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
            INSERT INTO metadata(key, value) VALUES ('schema_version', '4');
            """
        )
        conn.commit()
        conn.close()

        monkeypatch.setattr("hermes_lcm.db_bootstrap._check_disk_space", lambda _path: False)

        store = MessageStore(db_path)
        try:
            store.append("sess1", {"role": "user", "content": "fallback remains writable"})

            results = store.search("fallback", session_id="sess1")
            trigger_names = {
                row[0]
                for row in store._conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='trigger' AND name IN ('msg_fts_insert', 'msg_fts_delete')"
                ).fetchall()
            }

            assert len(results) == 1
            assert results[0]["content"] == "fallback remains writable"
            assert trigger_names == set()
        finally:
            store.close()

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

    def test_search_like_fallback_applies_sql_limit_for_messages(self, store):
        for index in range(40):
            store.append(
                "sess1",
                {"role": "user", "content": f"bulk message {index} 🚀"},
            )

        statements: list[str] = []
        store._conn.set_trace_callback(statements.append)
        try:
            results = store.search("🚀", session_id="sess1", limit=5, sort="relevance")
        finally:
            store._conn.set_trace_callback(None)

        assert len(results) == 5
        like_sql = next(
            statement
            for statement in statements
            if "FROM messages" in statement and "LIKE" in statement
        )
        assert "LIMIT " in like_sql

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

    def test_search_like_fallback_applies_sql_limit(self, store):
        for idx in range(80):
            store.append("sess1", {"role": "assistant", "content": f"plugin-only fallback load test {idx}"})

        traced: list[str] = []
        store._conn.set_trace_callback(traced.append)
        try:
            results = store.search("plugin-only", session_id="sess1", limit=2, sort="relevance")
        finally:
            store._conn.set_trace_callback(None)

        assert len(results) == 2
        assert any(
            "FROM messages" in statement and "content LIKE" in statement and "LIMIT 20" in statement
            for statement in traced
        )

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

    def test_search_relevance_caps_fts_batches_for_large_single_term_pool(self, store):
        for _ in range(5_000):
            store.append(
                "sess1",
                {
                    "role": "assistant",
                    "content": "vendoring",
                },
            )

        statements: list[str] = []
        store._conn.set_trace_callback(statements.append)
        try:
            _ = store.search("vendoring", session_id="sess1", limit=10, sort="relevance")
        finally:
            store._conn.set_trace_callback(None)

        fts_selects = [
            sql for sql in statements
            if "FROM messages_fts" in sql and "LIMIT" in sql and "OFFSET" in sql
        ]
        assert len(fts_selects) <= 6

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

    def test_get_batch_returns_multiple_messages_in_single_query(self, store):
        id1 = store.append("sess1", {"role": "user", "content": "first"})
        id2 = store.append("sess1", {"role": "assistant", "content": "second"})
        id3 = store.append("sess1", {"role": "user", "content": "third"})

        result = store.get_batch([id1, id3])

        assert len(result) == 2
        assert result[id1]["content"] == "first"
        assert result[id3]["content"] == "third"
        assert id2 not in result

    def test_get_batch_returns_empty_dict_for_empty_input(self, store):
        assert store.get_batch([]) == {}

    def test_get_batch_skips_missing_store_ids(self, store):
        id1 = store.append("sess1", {"role": "user", "content": "exists"})

        result = store.get_batch([id1, 99999])

        assert len(result) == 1
        assert result[id1]["content"] == "exists"

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


class TestLifecycleStateStore:
    def test_init_creates_lifecycle_state_table(self, tmp_path):
        state = LifecycleStateStore(tmp_path / "lifecycle.db")

        tables = {
            row[0]
            for row in state._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='lcm_lifecycle_state'"
            ).fetchall()
        }
        assert tables == {"lcm_lifecycle_state"}
        assert state.get_by_session("missing") is None

        state.close()

    def test_init_upgrades_legacy_db_and_keeps_missing_state_safe(self, tmp_path):
        db_path = tmp_path / "legacy-lifecycle.db"
        conn = sqlite3.connect(db_path)
        conn.executescript(
            """
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            );
            INSERT INTO metadata(key, value) VALUES ('schema_version', '2');
            """
        )
        conn.commit()
        conn.close()

        state = LifecycleStateStore(db_path)

        version = state._conn.execute(
            "SELECT value FROM metadata WHERE key = 'schema_version'"
        ).fetchone()[0]
        assert version == "4"

        tables = {
            row[0]
            for row in state._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='lcm_lifecycle_state'"
            ).fetchall()
        }
        assert tables == {"lcm_lifecycle_state"}
        columns = {
            row[1]
            for row in state._conn.execute("PRAGMA table_info(lcm_lifecycle_state)").fetchall()
        }
        assert {"debt_kind", "debt_size_estimate", "debt_updated_at", "last_maintenance_attempt_at"} <= columns
        assert state.get_by_session("unknown-session") is None

        state.close()

    def test_record_debt_and_clear_debt(self, tmp_path):
        state = LifecycleStateStore(tmp_path / "lifecycle-debt.db")
        bound = state.bind_session("sess-1")

        updated = state.record_debt(bound.conversation_id, kind="raw_backlog", size_estimate=321)
        assert updated is not None
        assert updated.debt_kind == "raw_backlog"
        assert updated.debt_size_estimate == 321
        assert updated.debt_updated_at is not None

        attempted = state.record_maintenance_attempt(bound.conversation_id)
        assert attempted is not None
        assert attempted.last_maintenance_attempt_at is not None

        cleared = state.clear_debt(bound.conversation_id)
        assert cleared is not None
        assert cleared.debt_kind is None
        assert cleared.debt_size_estimate == 0

        state.close()

    def test_record_reset_clears_pending_debt(self, tmp_path):
        state = LifecycleStateStore(tmp_path / "lifecycle-reset-debt.db")
        bound = state.bind_session("sess-1")

        state.record_debt(bound.conversation_id, kind="raw_backlog", size_estimate=500)
        with_debt = state.get_by_conversation(bound.conversation_id)
        assert with_debt is not None
        assert with_debt.debt_kind == "raw_backlog"
        assert with_debt.debt_size_estimate == 500

        after_reset = state.record_reset(bound.conversation_id)
        assert after_reset is not None
        assert after_reset.debt_kind is None
        assert after_reset.debt_size_estimate == 0
        assert after_reset.last_reset_at is not None

        state.close()


class TestDbBootstrapGuards:
    def test_sanitize_fts5_query_preserves_balanced_phrase_quotes(self):
        assert sanitize_fts5_query('"vendoring external" *') == '"vendoring external"'

    def test_sanitize_fts5_query_breaks_unbalanced_quotes_into_separate_terms(self):
        assert sanitize_fts5_query('foo"bar') == 'foo bar'

    def test_ensure_external_content_fts_skips_rebuild_when_disk_is_low(self, tmp_path, monkeypatch):
        conn = sqlite3.connect(tmp_path / "low-disk.db")
        conn.executescript(
            """
            CREATE TABLE messages (
                store_id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT
            );
            INSERT INTO messages(content) VALUES ('fresh searchable message');
            """
        )
        spec = ExternalContentFtsSpec(
            table_name="messages_fts",
            content_table="messages",
            content_rowid="store_id",
            indexed_column="content",
            trigger_sqls=(),
        )
        monkeypatch.setattr("hermes_lcm.db_bootstrap._check_disk_space", lambda _path: False)

        ensure_external_content_fts(conn, spec)

        existing = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts'"
        ).fetchone()
        assert existing is None
        conn.close()


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
        assert version == ("4",)

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
        assert version == ("4",)

        migration_state = dag._conn.execute(
            "SELECT step_name FROM lcm_migration_state ORDER BY step_name"
        ).fetchall()
        assert ("v2_external_content_fts_triggers",) in migration_state
        assert ("v4_lifecycle_debt_columns",) in migration_state

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

    def test_search_like_fallback_sanitizes_fts_syntax_chars(self, dag):
        dag.add_node(SummaryNode(
            session_id="s1", depth=0,
            summary="vendoring external support stays plugin-only",
            token_count=8, source_ids=[1], source_type="messages",
        ))

        results = dag.search('"vendoring*', session_id="s1")

        assert len(results) == 1
        assert results[0].summary == "vendoring external support stays plugin-only"

    def test_search_like_fallback_splits_unbalanced_quote_terms(self, dag):
        dag.add_node(SummaryNode(
            session_id="s1", depth=0,
            summary="foo bar baz",
            token_count=4, source_ids=[1], source_type="messages",
        ))

        results = dag.search('foo"bar', session_id="s1")

        assert len(results) == 1
        assert results[0].summary == "foo bar baz"

    def test_search_sanitizes_fts_wildcards_without_prefix_matching(self, dag):
        dag.add_node(SummaryNode(
            session_id="s1", depth=0,
            summary="dockerization notes",
            token_count=4, source_ids=[1], source_type="messages",
        ))

        results = dag.search("docker*", session_id="s1")

        assert results == []

    def test_search_uses_sanitized_terms_for_directness_scoring(self, dag):
        dag.add_node(SummaryNode(
            session_id="s1", depth=0,
            summary="vendoring external support stays plugin-only",
            token_count=8, source_ids=[1], source_type="messages",
        ))

        results = dag.search("vendoring*", session_id="s1")

        assert len(results) == 1
        assert results[0].search_directness > 0

    def test_init_low_disk_degrades_without_leaving_broken_nodes_fts_triggers(self, tmp_path, monkeypatch):
        db_path = tmp_path / "low-disk-broken-nodes-fts.db"
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
            CREATE TRIGGER nodes_fts_insert
                AFTER INSERT ON summary_nodes BEGIN
                INSERT INTO nodes_fts(rowid, summary)
                    VALUES (new.node_id, new.summary);
            END;
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            );
            INSERT INTO metadata(key, value) VALUES ('schema_version', '4');
            """
        )
        conn.commit()
        conn.close()

        monkeypatch.setattr("hermes_lcm.db_bootstrap._check_disk_space", lambda _path: False)

        dag = SummaryDAG(db_path)
        try:
            dag.add_node(SummaryNode(
                session_id="s1", depth=0,
                summary="fallback dag stays writable",
                token_count=5, source_ids=[1], source_type="messages",
            ))

            results = dag.search("fallback", session_id="s1")
            trigger_names = {
                row[0]
                for row in dag._conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='trigger' AND name IN ('nodes_fts_insert', 'nodes_fts_delete')"
                ).fetchall()
            }

            assert len(results) == 1
            assert results[0].summary == "fallback dag stays writable"
            assert trigger_names == set()
        finally:
            dag.close()

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

    def test_search_like_fallback_applies_sql_limit_for_summary_nodes(self, dag):
        for index in range(40):
            dag.add_node(SummaryNode(
                session_id="s1", depth=0,
                summary=f"bulk summary {index} 🚀",
                token_count=4, source_ids=[index + 1], source_type="messages",
                created_at=1_700_000_000 + index,
                earliest_at=1_700_000_000 + index,
                latest_at=1_700_000_000 + index,
            ))

        statements: list[str] = []
        dag._conn.set_trace_callback(statements.append)
        try:
            results = dag.search("🚀", session_id="s1", limit=5, sort="relevance")
        finally:
            dag._conn.set_trace_callback(None)

        assert len(results) == 5
        like_sql = next(
            statement
            for statement in statements
            if "FROM summary_nodes" in statement and "LIKE" in statement
        )
        assert "LIMIT " in like_sql

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

    def test_search_like_fallback_applies_sql_limit(self, dag):
        for idx in range(80):
            dag.add_node(SummaryNode(
                session_id="s1", depth=0,
                summary=f"plugin-only dag fallback load test {idx}",
                token_count=8, source_ids=[idx + 10_000], source_type="messages",
                created_at=1_700_000_000 + idx,
                earliest_at=1_700_000_000 + idx,
                latest_at=1_700_000_000 + idx,
            ))

        traced: list[str] = []
        dag._conn.set_trace_callback(traced.append)
        try:
            results = dag.search("plugin-only", session_id="s1", limit=2, sort="relevance")
        finally:
            dag._conn.set_trace_callback(None)

        assert len(results) == 2
        assert any(
            "FROM summary_nodes" in statement and "summary LIKE" in statement and "LIMIT 20" in statement
            for statement in traced
        )

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

    def test_search_relevance_caps_fts_batches_for_large_single_term_pool(self, dag):
        for idx in range(5_000):
            dag.add_node(SummaryNode(
                session_id="s1", depth=0,
                summary=f"Summary {idx}: vendoring",
                token_count=10, source_ids=[idx + 1], source_type="messages",
                created_at=1_700_000_000 + idx,
                earliest_at=1_700_000_000 + idx,
                latest_at=1_700_000_000 + idx,
            ))

        statements: list[str] = []
        dag._conn.set_trace_callback(statements.append)
        try:
            _ = dag.search("vendoring", session_id="s1", limit=10, sort="relevance")
        finally:
            dag._conn.set_trace_callback(None)

        fts_selects = [
            sql for sql in statements
            if "FROM nodes_fts" in sql and "LIMIT" in sql and "OFFSET" in sql
        ]
        assert len(fts_selects) <= 6

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

    def test_custom_instructions_injected_into_l1_prompt(self):
        from hermes_lcm.escalation import _build_l1_prompt
        prompt = _build_l1_prompt(
            "test content", 500, depth=0,
            custom_instructions="Write as a neutral documenter.",
        )
        assert "Additional instructions:" in prompt
        assert "Write as a neutral documenter." in prompt

    def test_custom_instructions_injected_into_l2_prompt(self):
        from hermes_lcm.escalation import _build_l2_prompt
        prompt = _build_l2_prompt(
            "test content", 500,
            custom_instructions="Use third person only.",
        )
        assert "Additional instructions:" in prompt
        assert "Use third person only." in prompt

    def test_custom_instructions_omitted_when_empty(self):
        from hermes_lcm.escalation import _build_l1_prompt, _build_l2_prompt
        l1 = _build_l1_prompt("test", 500, depth=0, custom_instructions="")
        l2 = _build_l2_prompt("test", 500, custom_instructions="")
        assert "Additional instructions:" not in l1
        assert "Additional instructions:" not in l2


class TestExtraction:
    def test_serialize_messages_replaces_pure_inline_media_with_attachment_marker(self, tmp_path):
        from hermes_lcm.config import LCMConfig
        from hermes_lcm.engine import LCMEngine

        engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db")))

        serialized = engine._serialize_messages([
            {
                "role": "user",
                "content": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA",
            }
        ])

        assert "[USER]: [Media attachment]" == serialized
        assert "data:image/png;base64" not in serialized

    def test_serialize_messages_preserves_text_but_replaces_inline_media_suffix(self, tmp_path):
        from hermes_lcm.config import LCMConfig
        from hermes_lcm.engine import LCMEngine

        engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db")))

        serialized = engine._serialize_messages([
            {
                "role": "assistant",
                "content": "Here is the chart you asked for.\n\ndata:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA",
            }
        ])

        assert "Here is the chart you asked for." in serialized
        assert "[with media attachment]" in serialized
        assert "data:image/png;base64" not in serialized

    def test_serialize_messages_handles_chat_completions_style_multimodal_blocks(self, tmp_path):
        from hermes_lcm.config import LCMConfig
        from hermes_lcm.engine import LCMEngine

        engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db")))

        serialized = engine._serialize_messages([
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Please remember this screenshot."},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA"},
                    },
                ],
            }
        ])

        assert "Please remember this screenshot." in serialized
        assert "[with media attachment]" in serialized
        assert "data:image/png;base64" not in serialized

    def test_serialize_messages_handles_responses_style_multimodal_blocks(self, tmp_path):
        from hermes_lcm.config import LCMConfig
        from hermes_lcm.engine import LCMEngine

        engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db")))

        serialized = engine._serialize_messages([
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Describe this."},
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA",
                    },
                ],
            }
        ])

        assert "Describe this." in serialized
        assert "[with media attachment]" in serialized
        assert "data:image/png;base64" not in serialized

    def test_serialize_messages_leaves_non_media_application_data_uri_alone(self, tmp_path):
        from hermes_lcm.config import LCMConfig
        from hermes_lcm.engine import LCMEngine

        engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db")))

        content = "data:application/json;base64,eyJmb28iOiAiYmFyIiwgImJheiI6IDEyfQ=="
        serialized = engine._serialize_messages([
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": content,
            }
        ])

        assert content in serialized
        assert "[Media attachment]" not in serialized
        assert "[with media attachment]" not in serialized

    def test_serialize_messages_sanitizes_tool_call_arguments_media_payloads(self, tmp_path):
        from hermes_lcm.config import LCMConfig
        from hermes_lcm.engine import LCMEngine

        engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db")))

        serialized = engine._serialize_messages([
            {
                "role": "assistant",
                "content": "Calling the image tool now.",
                "tool_calls": [
                    {
                        "function": {
                            "name": "vision_analyze",
                            "arguments": '{"image":"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA"}',
                        }
                    }
                ],
            }
        ])

        assert "vision_analyze" in serialized
        assert '"image": "[Media attachment]"' in serialized
        assert "data:image/png;base64" not in serialized

    def test_serialize_messages_sanitizes_parsed_tool_call_arguments_media_payloads(self, tmp_path):
        from hermes_lcm.config import LCMConfig
        from hermes_lcm.engine import LCMEngine

        engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db")))

        serialized = engine._serialize_messages([
            {
                "role": "assistant",
                "content": "Calling the image tool now.",
                "tool_calls": [
                    {
                        "function": {
                            "name": "vision_analyze",
                            "arguments": {
                                "prompt": "Describe the chart",
                                "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA",
                            },
                        }
                    }
                ],
            }
        ])

        assert '"prompt": "Describe the chart"' in serialized
        assert '"image": "[Media attachment]"' in serialized
        assert "data:image/png;base64" not in serialized

    def test_serialize_messages_preserves_structured_file_block_metadata(self, tmp_path):
        from hermes_lcm.config import LCMConfig
        from hermes_lcm.engine import LCMEngine

        engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db")))

        serialized = engine._serialize_messages([
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Use the uploaded document."},
                    {
                        "type": "input_file",
                        "file_id": "file_123",
                        "filename": "requirements.pdf",
                        "mime_type": "application/pdf",
                    },
                ],
            }
        ])

        assert "Use the uploaded document." in serialized
        assert "type=input_file" in serialized
        assert "file_123" in serialized
        assert "requirements.pdf" in serialized

    def test_serialize_messages_uses_profile_safe_default_externalization_path_for_large_tool_output(self, tmp_path):
        from hermes_lcm.config import LCMConfig
        from hermes_lcm.engine import LCMEngine

        hermes_home = tmp_path / "hermes-home"
        engine = LCMEngine(
            config=LCMConfig(
                database_path=str(tmp_path / "lcm.db"),
                large_output_externalization_enabled=True,
                large_output_externalization_threshold_chars=200,
            ),
            hermes_home=str(hermes_home),
        )

        content = "tool-output:" + ("x" * 5000)
        serialized = engine._serialize_messages([
            {
                "role": "tool",
                "tool_call_id": "call_big_default",
                "content": content,
            }
        ])

        assert "[Externalized tool output" in serialized
        assert "call_big_default" in serialized
        assert content[:500] not in serialized

        payload_dir = hermes_home / "lcm-large-outputs"
        payload_files = list(payload_dir.glob("*.json"))
        assert len(payload_files) == 1

        payload = json.loads(payload_files[0].read_text())
        assert payload["kind"] == "tool_result"
        assert payload["tool_call_id"] == "call_big_default"
        assert payload["content"] == content

    def test_serialize_messages_leaves_large_tool_output_inline_when_externalization_disabled(self, tmp_path):
        from hermes_lcm.config import LCMConfig
        from hermes_lcm.engine import LCMEngine

        hermes_home = tmp_path / "hermes-home"
        engine = LCMEngine(
            config=LCMConfig(
                database_path=str(tmp_path / "lcm.db"),
                large_output_externalization_enabled=False,
                large_output_externalization_threshold_chars=200,
            ),
            hermes_home=str(hermes_home),
        )

        content = "tool-output:" + ("x" * 5000)
        serialized = engine._serialize_messages([
            {
                "role": "tool",
                "tool_call_id": "call_disabled",
                "content": content,
            }
        ])

        assert "[Externalized tool output" not in serialized
        assert "...[truncated]..." in serialized
        assert not (hermes_home / "lcm-large-outputs").exists()

    def test_serialize_messages_falls_back_to_truncation_when_externalization_path_is_unwritable(self, tmp_path):
        from hermes_lcm.config import LCMConfig
        from hermes_lcm.engine import LCMEngine

        blocked_path = tmp_path / "not-a-dir"
        blocked_path.write_text("occupied")
        engine = LCMEngine(
            config=LCMConfig(
                database_path=str(tmp_path / "lcm.db"),
                large_output_externalization_enabled=True,
                large_output_externalization_threshold_chars=200,
                large_output_externalization_path=str(blocked_path),
            )
        )

        content = "tool-output:" + ("x" * 5000)
        serialized = engine._serialize_messages([
            {
                "role": "tool",
                "tool_call_id": "call_unwritable",
                "content": content,
            }
        ])

        assert "[Externalized tool output" not in serialized
        assert "...[truncated]..." in serialized

    def test_serialize_messages_externalized_payloads_do_not_collide_for_same_second_same_tool_id(self, tmp_path, monkeypatch):
        from hermes_lcm.config import LCMConfig
        from hermes_lcm.engine import LCMEngine
        import hermes_lcm.externalize as ext_module

        output_dir = tmp_path / "externalized"
        original_strftime = ext_module.time.strftime
        monkeypatch.setattr(ext_module.time, "strftime", lambda *args, **kwargs: "20260418_060000")
        try:
            first = LCMEngine(
                config=LCMConfig(
                    database_path=str(tmp_path / "first.db"),
                    large_output_externalization_enabled=True,
                    large_output_externalization_threshold_chars=200,
                    large_output_externalization_path=str(output_dir),
                )
            )
            first._session_id = "telegram:first"

            second = LCMEngine(
                config=LCMConfig(
                    database_path=str(tmp_path / "second.db"),
                    large_output_externalization_enabled=True,
                    large_output_externalization_threshold_chars=200,
                    large_output_externalization_path=str(output_dir),
                )
            )
            second._session_id = "telegram:second"

            content = "RESULT:\n" + ("abcdef" * 2000)
            first_serialized = first._serialize_messages([
                {"role": "tool", "tool_call_id": "call_same", "content": content}
            ])
            second_serialized = second._serialize_messages([
                {"role": "tool", "tool_call_id": "call_same", "content": content}
            ])
        finally:
            monkeypatch.setattr(ext_module.time, "strftime", original_strftime)

        payload_files = sorted(output_dir.glob("*.json"))
        assert len(payload_files) == 2
        assert first_serialized != second_serialized

        payloads = [json.loads(path.read_text()) for path in payload_files]
        assert sorted(payload["session_id"] for payload in payloads) == ["telegram:first", "telegram:second"]

    def test_serialize_messages_reuses_existing_externalized_payload_for_same_session_content_and_tool_id(self, tmp_path):
        from hermes_lcm.config import LCMConfig
        from hermes_lcm.engine import LCMEngine

        hermes_home = tmp_path / "hermes-home"
        engine = LCMEngine(
            config=LCMConfig(
                database_path=str(tmp_path / "lcm.db"),
                large_output_externalization_enabled=True,
                large_output_externalization_threshold_chars=200,
            ),
            hermes_home=str(hermes_home),
        )
        engine._session_id = "test-session"

        content = "RESULT:\n" + ("abcdef" * 2000)
        first_serialized = engine._serialize_messages([
            {"role": "tool", "tool_call_id": "call_reuse", "content": content}
        ])
        second_serialized = engine._serialize_messages([
            {"role": "tool", "tool_call_id": "call_reuse", "content": content}
        ])

        payload_dir = hermes_home / "lcm-large-outputs"
        payload_files = sorted(payload_dir.glob("*.json"))
        assert len(payload_files) == 1
        assert first_serialized == second_serialized

        payload = json.loads(payload_files[0].read_text())
        assert payload["session_id"] == "test-session"
        assert payload["tool_call_id"] == "call_reuse"
        assert payload["content"] == content

    def test_serialize_messages_externalizes_large_tool_output_to_configured_path(self, tmp_path):
        from hermes_lcm.config import LCMConfig
        from hermes_lcm.engine import LCMEngine

        output_dir = tmp_path / "externalized"
        engine = LCMEngine(
            config=LCMConfig(
                database_path=str(tmp_path / "lcm.db"),
                large_output_externalization_enabled=True,
                large_output_externalization_threshold_chars=200,
                large_output_externalization_path=str(output_dir),
            )
        )

        content = "RESULT:\n" + ("abcdef" * 2000)
        serialized = engine._serialize_messages([
            {
                "role": "tool",
                "tool_call_id": "call_big_custom",
                "content": content,
            }
        ])

        assert "[Externalized tool output" in serialized
        assert "call_big_custom" in serialized
        assert content[:500] not in serialized

        payload_files = list(output_dir.glob("*.json"))
        assert len(payload_files) == 1

        payload = json.loads(payload_files[0].read_text())
        assert payload["kind"] == "tool_result"
        assert payload["tool_call_id"] == "call_big_custom"
        assert payload["content"] == content

    def test_run_pre_compaction_extraction_uses_media_cleaned_text(self, tmp_path):
        from hermes_lcm.config import LCMConfig
        from hermes_lcm.engine import LCMEngine
        import hermes_lcm.extraction as ext_module

        config = LCMConfig(
            database_path=str(tmp_path / "lcm_extract.db"),
            extraction_enabled=True,
            extraction_output_path=str(tmp_path / "extractions"),
        )
        engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
        engine._session_id = "test-session"

        original = ext_module._call_extraction_llm
        seen_prompt = {}

        def mock_llm(prompt, model="", timeout=None):
            seen_prompt["prompt"] = prompt
            return "- Captured media cleanup"

        ext_module._call_extraction_llm = mock_llm
        try:
            engine._run_pre_compaction_extraction([
                {
                    "role": "user",
                    "content": "Please save this image for later\n\ndata:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA",
                },
            ])
        finally:
            ext_module._call_extraction_llm = original

        assert "[with media attachment]" in seen_prompt["prompt"]
        assert "data:image/png;base64" not in seen_prompt["prompt"]

    def test_extract_writes_daily_file(self, tmp_path):
        from hermes_lcm.extraction import extract_before_compaction

        # Mock the LLM call
        import hermes_lcm.extraction as ext_module
        original = ext_module._call_extraction_llm

        def mock_llm(prompt, model="", timeout=None):
            return "- Decided to use PostgreSQL for the user store\n- Stephen will handle the migration by Friday"

        ext_module._call_extraction_llm = mock_llm
        try:
            output_dir = str(tmp_path / "extractions")
            result = extract_before_compaction(
                serialized_messages="[USER]: Let's use PostgreSQL\n[ASSISTANT]: Done",
                output_path=output_dir,
                session_id="test-session",
            )
            assert result is True

            files = list(Path(tmp_path / "extractions").glob("*.md"))
            assert len(files) == 1
            content = files[0].read_text()
            assert "PostgreSQL" in content
            assert "test-session" in content
            assert "migration" in content
        finally:
            ext_module._call_extraction_llm = original

    def test_extract_skips_when_nothing_to_extract(self, tmp_path):
        from hermes_lcm.extraction import extract_before_compaction
        import hermes_lcm.extraction as ext_module
        original = ext_module._call_extraction_llm

        def mock_llm(prompt, model="", timeout=None):
            return "NOTHING_TO_EXTRACT"

        ext_module._call_extraction_llm = mock_llm
        try:
            output_dir = str(tmp_path / "extractions")
            result = extract_before_compaction(
                serialized_messages="[USER]: hello\n[ASSISTANT]: hi",
                output_path=output_dir,
            )
            assert result is True
            files = list(Path(tmp_path / "extractions").glob("*.md"))
            assert len(files) == 0
        finally:
            ext_module._call_extraction_llm = original

    def test_extract_never_blocks_on_failure(self, tmp_path):
        from hermes_lcm.extraction import extract_before_compaction
        import hermes_lcm.extraction as ext_module
        original = ext_module._call_extraction_llm

        def mock_llm(prompt, model="", timeout=None):
            return None

        ext_module._call_extraction_llm = mock_llm
        try:
            result = extract_before_compaction(
                serialized_messages="test",
                output_path=str(tmp_path / "extractions"),
            )
            # Should return True (nothing to write) not raise
            assert result is True
        finally:
            ext_module._call_extraction_llm = original

    def test_engine_extraction_uses_default_path_when_config_empty(self, tmp_path, monkeypatch):
        from hermes_lcm.config import LCMConfig
        from hermes_lcm.engine import LCMEngine
        import hermes_lcm.extraction as ext_module

        config = LCMConfig(
            database_path=str(tmp_path / "lcm_extract.db"),
            extraction_enabled=True,
            extraction_output_path="",
        )
        engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
        engine._session_id = "test-session"

        original = ext_module._call_extraction_llm

        def mock_llm(prompt, model="", timeout=None):
            return "- Decided to use Redis for caching"

        ext_module._call_extraction_llm = mock_llm
        try:
            engine._run_pre_compaction_extraction([
                {"role": "user", "content": "Let's use Redis"},
                {"role": "assistant", "content": "Done"},
            ])
            extraction_dir = tmp_path / "hermes" / "lcm-extractions"
            files = list(extraction_dir.glob("*.md"))
            assert len(files) == 1
            assert "Redis" in files[0].read_text()
        finally:
            ext_module._call_extraction_llm = original

    def test_extract_appends_to_existing_daily_file(self, tmp_path):
        from hermes_lcm.extraction import extract_before_compaction
        import hermes_lcm.extraction as ext_module
        original = ext_module._call_extraction_llm

        call_count = 0

        def mock_llm(prompt, model="", timeout=None):
            nonlocal call_count
            call_count += 1
            return f"- Decision {call_count}"

        ext_module._call_extraction_llm = mock_llm
        try:
            output_dir = str(tmp_path / "extractions")
            extract_before_compaction("first", output_path=output_dir, session_id="s1")
            extract_before_compaction("second", output_path=output_dir, session_id="s2")

            files = list(Path(tmp_path / "extractions").glob("*.md"))
            assert len(files) == 1
            content = files[0].read_text()
            assert "Decision 1" in content
            assert "Decision 2" in content
            assert "s1" in content
            assert "s2" in content
        finally:
            ext_module._call_extraction_llm = original
