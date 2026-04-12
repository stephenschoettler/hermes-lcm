"""Integration tests for the LCM engine."""

import json
import logging
import time
import pytest

from agent.context_engine import ContextEngine
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine


@pytest.fixture
def engine(tmp_path):
    config = LCMConfig()
    config.fresh_tail_count = 4  # small for testing
    config.leaf_chunk_tokens = 100  # low threshold for testing
    config.database_path = str(tmp_path / "lcm_test.db")
    e = LCMEngine(config=config)
    e._session_id = "test-session"
    e.context_length = 200000
    e.threshold_tokens = int(200000 * config.context_threshold)
    return e


class TestEngineABC:
    def test_is_context_engine(self, engine):
        assert isinstance(engine, ContextEngine)

    def test_name(self, engine):
        assert engine.name == "lcm"

    def test_tool_schemas(self, engine):
        schemas = engine.get_tool_schemas()
        names = [s["name"] for s in schemas]
        assert "lcm_grep" in names
        assert "lcm_describe" in names
        assert "lcm_expand" in names
        assert "lcm_expand_query" in names

    def test_should_compress(self, engine):
        assert not engine.should_compress(1000)
        assert engine.should_compress(engine.threshold_tokens)

    def test_update_from_response(self, engine):
        engine.update_from_response({
            "prompt_tokens": 5000,
            "completion_tokens": 200,
            "total_tokens": 5200,
        })
        assert engine.last_prompt_tokens == 5000

    def test_session_reset(self, engine):
        engine.compression_count = 5
        engine.last_prompt_tokens = 9999
        engine.on_session_reset()
        assert engine.compression_count == 0
        assert engine.last_prompt_tokens == 0

    def test_get_status(self, engine):
        status = engine.get_status()
        assert status["engine"] == "lcm"
        assert "store_messages" in status
        assert "dag_nodes" in status

    def test_compress_accepts_focus_topic(self, engine, monkeypatch):
        import importlib

        captured = {}

        def mock_summary(**kwargs):
            captured["focus_topic"] = kwargs.get("focus_topic")
            return "Focused summary.\nExpand for details about: database", 1

        lcm_engine_module = importlib.import_module("hermes_lcm.engine")
        monkeypatch.setattr(lcm_engine_module, "summarize_with_escalation", mock_summary)

        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        for i in range(20):
            messages.append({"role": "user", "content": f"Question {i}: " + "x" * 200})
            messages.append({"role": "assistant", "content": f"Answer {i}: " + "y" * 200})

        engine.compress(messages, focus_topic="database migrations")

        assert captured["focus_topic"] == "database migrations"


class TestSessionFiltering:
    def test_on_session_start_marks_ignored_session_and_reports_status(self, tmp_path, caplog):
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_ignore.db"),
            ignore_session_patterns=["cron:*"],
            ignore_session_patterns_source="env",
        )
        instance = LCMEngine(config=config)

        with caplog.at_level("INFO", logger="hermes_lcm.engine"):
            instance.on_session_start("cron_123", platform="cron", context_length=1000)

        status = instance.get_status()
        assert status["session_ignored"] is True
        assert status["session_stateless"] is False
        assert status["ignore_session_patterns"] == ["cron:*"]
        assert status["ignore_session_patterns_source"] == "env"
        assert "LCM ignore_session_patterns from env: cron:*" in caplog.text
        assert "matched ignore_session_patterns" in caplog.text

    def test_on_session_start_marks_stateless_session_and_reports_status(self, tmp_path, caplog):
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_stateless.db"),
            stateless_session_patterns=["telegram:*"],
            stateless_session_patterns_source="env",
        )
        instance = LCMEngine(config=config)

        with caplog.at_level("INFO", logger="hermes_lcm.engine"):
            instance.on_session_start("debug", platform="telegram", context_length=1000)

        status = instance.get_status()
        assert status["session_ignored"] is False
        assert status["session_stateless"] is True
        assert status["stateless_session_patterns"] == ["telegram:*"]
        assert status["stateless_session_patterns_source"] == "env"
        assert "LCM stateless_session_patterns from env: telegram:*" in caplog.text
        assert "matched stateless_session_patterns" in caplog.text

    def test_ignored_session_does_not_write_to_store_or_compact(self, tmp_path):
        config = LCMConfig(
            fresh_tail_count=2,
            leaf_chunk_tokens=1,
            database_path=str(tmp_path / "lcm_ignore_behavior.db"),
            ignore_session_patterns=["cron:*"],
        )
        instance = LCMEngine(config=config)
        instance.on_session_start("cron_123", platform="cron", context_length=1000)

        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
            {"role": "user", "content": "again"},
        ]

        result = instance.compress(messages)

        assert result == messages
        assert instance._store.get_session_count("cron_123") == 0
        assert instance._dag.get_session_nodes("cron_123") == []
        assert instance.compression_count == 0

    def test_stateless_session_does_not_write_to_store_or_compact(self, tmp_path):
        config = LCMConfig(
            fresh_tail_count=2,
            leaf_chunk_tokens=1,
            database_path=str(tmp_path / "lcm_stateless_behavior.db"),
            stateless_session_patterns=["telegram:*"],
        )
        instance = LCMEngine(config=config)
        instance.on_session_start("debug", platform="telegram", context_length=1000)

        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
            {"role": "user", "content": "again"},
        ]

        result = instance.compress(messages)

        assert result == messages
        assert instance._store.get_session_count("debug") == 0
        assert instance._dag.get_session_nodes("debug") == []
        assert instance.compression_count == 0


class TestEngineIngest:
    def test_ingest_stores_messages(self, engine):
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        engine._ingest_messages(messages)
        count = engine._store.get_session_count("test-session")
        assert count == 3

    def test_ingest_idempotent(self, engine):
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]
        engine._ingest_messages(messages)
        engine._ingest_messages(messages)  # same messages again
        count = engine._store.get_session_count("test-session")
        assert count == 2  # not duplicated


class TestEngineCompress:
    def _make_long_conversation(self, n_turns=20):
        """Build a conversation with enough messages to trigger compaction."""
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        for i in range(n_turns):
            messages.append({"role": "user", "content": f"Question {i}: " + "x" * 200})
            messages.append({"role": "assistant", "content": f"Answer {i}: " + "y" * 200})
        return messages

    def test_compress_short_conversation_noop(self, engine):
        """Short conversations should pass through unchanged."""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        result = engine.compress(messages)
        assert len(result) == len(messages)

    def test_compress_preserves_system_and_tail(self, engine):
        """Compression should always keep system prompt and fresh tail."""
        messages = self._make_long_conversation(20)
        # Mock the summarization to avoid LLM calls
        import hermes_lcm.escalation as esc
        original_fn = esc._call_llm_for_summary

        def mock_summarize(prompt, max_tokens, model=""):
            return "Mock summary of earlier conversation.\nExpand for details about: earlier turns"

        esc._call_llm_for_summary = mock_summarize
        try:
            result = engine.compress(messages)

            # System prompt should be first
            assert result[0]["role"] == "system"

            # Last messages should be the fresh tail
            assert result[-1] == messages[-1]

            # Should be shorter than original
            assert len(result) < len(messages)

            # Compression count should increment
            assert engine.compression_count == 1
        finally:
            esc._call_llm_for_summary = original_fn

    def test_compress_creates_dag_node(self, engine):
        """Compression should create a DAG node."""
        messages = self._make_long_conversation(20)
        import hermes_lcm.escalation as esc
        original_fn = esc._call_llm_for_summary

        def mock_summarize(prompt, max_tokens, model=""):
            return "Mock summary.\nExpand for details about: everything"

        esc._call_llm_for_summary = mock_summarize
        try:
            engine.compress(messages)
            nodes = engine._dag.get_session_nodes("test-session")
            assert len(nodes) >= 1
            assert nodes[0].depth == 0
            assert nodes[0].source_type == "messages"
        finally:
            esc._call_llm_for_summary = original_fn


class TestPostCompactionIngestion:
    """Regression tests for issue #1 — messages must be persisted after
    compaction even though the active context is shorter than the store."""

    def _make_long_conversation(self, n_turns=20):
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        for i in range(n_turns):
            messages.append({"role": "user", "content": f"Question {i}: " + "x" * 200})
            messages.append({"role": "assistant", "content": f"Answer {i}: " + "y" * 200})
        return messages

    def _mock_summarize(self, prompt, max_tokens, model=""):
        return "Mock summary of conversation.\nExpand for details about: earlier turns"

    def test_ingest_after_compaction(self, engine):
        """New messages after compress() must still be persisted."""
        import hermes_lcm.escalation as esc
        original_fn = esc._call_llm_for_summary
        esc._call_llm_for_summary = self._mock_summarize
        try:
            messages = self._make_long_conversation(20)
            compressed = engine.compress(messages)
            count_after_compress = engine._store.get_session_count("test-session")
            assert count_after_compress == len(messages)  # all originals stored

            # Simulate new turns appended to compressed context
            compressed.append({"role": "user", "content": "Brand new question"})
            compressed.append({"role": "assistant", "content": "Brand new answer"})

            engine._ingest_messages(compressed)
            count_after_new = engine._store.get_session_count("test-session")
            assert count_after_new == count_after_compress + 2
        finally:
            esc._call_llm_for_summary = original_fn

    def test_ingest_cursor_reset_on_session_reset(self, engine):
        """on_session_reset() must reset the ingest cursor."""
        engine._ingest_cursor = 42
        engine.on_session_reset()
        assert engine._ingest_cursor == 0

    def test_multiple_compactions(self, engine):
        """Messages stay persisted across multiple compress() cycles."""
        import hermes_lcm.escalation as esc
        original_fn = esc._call_llm_for_summary
        esc._call_llm_for_summary = self._mock_summarize
        try:
            # First compaction
            messages = self._make_long_conversation(20)
            compressed = engine.compress(messages)
            count1 = engine._store.get_session_count("test-session")

            # Add new turns and compact again
            for i in range(15):
                compressed.append({"role": "user", "content": f"Round2 Q{i}: " + "z" * 200})
                compressed.append({"role": "assistant", "content": f"Round2 A{i}: " + "w" * 200})

            compressed2 = engine.compress(compressed)
            count2 = engine._store.get_session_count("test-session")
            # Should have original messages + 30 new ones
            assert count2 == count1 + 30

            # Add more after second compaction
            compressed2.append({"role": "user", "content": "Final question"})
            engine._ingest_messages(compressed2)
            count3 = engine._store.get_session_count("test-session")
            assert count3 == count2 + 1
        finally:
            esc._call_llm_for_summary = original_fn


class TestStoreIdMapping:
    """Regression test — _get_store_ids_for_messages must use content
    matching, not position, so source_ids stay correct after compaction."""

    def _make_long_conversation(self, n_turns=20):
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        for i in range(n_turns):
            messages.append({"role": "user", "content": f"Question {i}: " + "x" * 200})
            messages.append({"role": "assistant", "content": f"Answer {i}: " + "y" * 200})
        return messages

    def _mock_summarize(self, prompt, max_tokens, model=""):
        return "Mock summary of conversation.\nExpand for details about: earlier turns"

    def test_source_ids_correct_after_second_compaction(self, engine):
        """DAG nodes from a second compress() must not reference the
        synthetic summary message or map to wrong store rows."""
        import hermes_lcm.escalation as esc
        original_fn = esc._call_llm_for_summary
        esc._call_llm_for_summary = self._mock_summarize
        try:
            # First compaction
            messages = self._make_long_conversation(20)
            compressed = engine.compress(messages)

            # Add new turns and compact again
            for i in range(15):
                compressed.append({"role": "user", "content": f"Round2 Q{i}: " + "z" * 200})
                compressed.append({"role": "assistant", "content": f"Round2 A{i}: " + "w" * 200})

            engine.compress(compressed)

            # Get all DAG nodes — the second node's source_ids should
            # only reference real stored messages, not the summary
            nodes = engine._dag.get_session_nodes("test-session")
            assert len(nodes) >= 2

            second_node = nodes[1]
            for sid in second_node.source_ids:
                stored = engine._store.get(sid)
                assert stored is not None, f"source_id {sid} not in store"
                # Must not be a synthetic summary
                assert "Mock summary" not in (stored.get("content") or ""), \
                    f"source_id {sid} points to synthetic summary"
        finally:
            esc._call_llm_for_summary = original_fn

    def test_repeated_content_maps_to_later_store_rows(self, engine):
        import hermes_lcm.escalation as esc
        original_fn = esc._call_llm_for_summary
        esc._call_llm_for_summary = self._mock_summarize
        try:
            messages = [{"role": "system", "content": "You are a helpful assistant."}]
            for _ in range(20):
                messages.append({"role": "user", "content": "repeat"})
                messages.append({"role": "assistant", "content": "same"})

            compressed = engine.compress(messages)
            first_node = engine._dag.get_session_nodes("test-session")[0]
            first_max = max(first_node.source_ids)

            for _ in range(15):
                compressed.append({"role": "user", "content": "repeat"})
                compressed.append({"role": "assistant", "content": "same"})

            engine.compress(compressed)
            nodes = engine._dag.get_session_nodes("test-session")
            second_node = nodes[1]
            assert second_node.source_ids
            assert all(store_id > first_max for store_id in second_node.source_ids)
        finally:
            esc._call_llm_for_summary = original_fn


class TestSessionRetainDepth:
    """Tests for issue #2a — new_session_retain_depth wiring."""

    def test_retain_depth_zero_deletes_all(self, engine):
        """retain_depth=0 should delete all DAG nodes on reset."""
        engine._config.new_session_retain_depth = 0
        from hermes_lcm.dag import SummaryNode
        import time
        for d in range(3):
            engine._dag.add_node(SummaryNode(
                session_id="test-session", depth=d,
                summary=f"d{d} summary", token_count=100,
                source_token_count=500, source_ids=[],
                source_type="messages", created_at=time.time(),
            ))
        assert len(engine._dag.get_session_nodes("test-session")) == 3
        engine.on_session_reset()
        assert len(engine._dag.get_session_nodes("test-session")) == 0

    def test_retain_depth_keeps_high_nodes(self, engine):
        """retain_depth=2 should keep d2+ and delete d0, d1."""
        engine._config.new_session_retain_depth = 2
        from hermes_lcm.dag import SummaryNode
        import time
        for d in range(4):
            engine._dag.add_node(SummaryNode(
                session_id="test-session", depth=d,
                summary=f"d{d} summary", token_count=100,
                source_token_count=500, source_ids=[],
                source_type="messages", created_at=time.time(),
            ))
        engine.on_session_reset()
        remaining = engine._dag.get_session_nodes("test-session")
        assert len(remaining) == 2
        assert all(n.depth >= 2 for n in remaining)

    def test_retain_depth_minus_one_keeps_all(self, engine):
        """retain_depth=-1 should keep all nodes."""
        engine._config.new_session_retain_depth = -1
        from hermes_lcm.dag import SummaryNode
        import time
        for d in range(3):
            engine._dag.add_node(SummaryNode(
                session_id="test-session", depth=d,
                summary=f"d{d} summary", token_count=100,
                source_token_count=500, source_ids=[],
                source_type="messages", created_at=time.time(),
            ))
        engine.on_session_reset()
        assert len(engine._dag.get_session_nodes("test-session")) == 3

    def test_carry_over_moves_retained_nodes_into_new_session(self, engine):
        engine._config.new_session_retain_depth = 2
        import time
        for depth in range(4):
            engine._dag.add_node(SummaryNode(
                session_id="old-session", depth=depth,
                summary=f"d{depth} summary", token_count=100,
                source_token_count=500, source_ids=[],
                source_type="messages", created_at=time.time(),
            ))

        engine._session_id = "old-session"
        engine.on_session_reset()
        moved = engine.carry_over_new_session_context("old-session", "new-session")

        assert moved == 2
        assert engine._dag.get_session_nodes("old-session") == []
        new_nodes = engine._dag.get_session_nodes("new-session")
        assert len(new_nodes) == 2
        assert all(node.depth >= 2 for node in new_nodes)


class TestUnlimitedCondensationDepth:
    """Tests for issue #2b — max_depth=-1 should be truly unlimited."""

    def test_unlimited_depth_condenses_beyond_ten(self, engine):
        """With max_depth=-1, condensation should not be capped at depth 10."""
        engine._config.incremental_max_depth = -1
        engine._config.condensation_fanin = 2
        from hermes_lcm.dag import SummaryNode
        import time

        # Create nodes at depth 11 — old code would skip these
        for i in range(3):
            engine._dag.add_node(SummaryNode(
                session_id="test-session", depth=11,
                summary=f"Deep node {i}", token_count=100,
                source_token_count=200, source_ids=[],
                source_type="nodes", created_at=time.time(),
            ))

        import hermes_lcm.escalation as esc
        original_fn = esc._call_llm_for_summary

        def mock_summarize(prompt, max_tokens, model=""):
            return "Condensed.\nExpand for details about: deep nodes"

        esc._call_llm_for_summary = mock_summarize
        try:
            engine._maybe_condense()
            # Should have created a d12 node
            d12 = engine._dag.get_session_nodes("test-session", depth=12)
            assert len(d12) >= 1
        finally:
            esc._call_llm_for_summary = original_fn


class TestConfigCleanup:
    """Tests for issue #2c follow-up — expansion path is now separate from summary-only config."""

    def test_has_expansion_model(self):
        config = LCMConfig()
        assert hasattr(config, "expansion_model")
        assert config.expansion_model == ""

    def test_has_summary_timeout_ms(self):
        config = LCMConfig()
        assert hasattr(config, "summary_timeout_ms")
        assert config.summary_timeout_ms == 60_000

    def test_has_expansion_timeout_ms(self):
        config = LCMConfig()
        assert hasattr(config, "expansion_timeout_ms")
        assert config.expansion_timeout_ms == 120_000


class TestAssemblyGuardrails:
    def test_max_assembly_tokens_caps_recent_tail(self, tmp_path, monkeypatch):
        import importlib

        config = LCMConfig(
            fresh_tail_count=10,
            database_path=str(tmp_path / "lcm_guardrail.db"),
            max_assembly_tokens=60,
        )
        instance = LCMEngine(config=config)
        instance._session_id = "guardrail-session"
        instance.compression_count = 1

        lcm_engine_module = importlib.import_module("hermes_lcm.engine")
        monkeypatch.setattr(
            lcm_engine_module,
            "count_message_tokens",
            lambda msg: len(msg.get("content", "")),
        )

        result = instance._assemble_context(
            {"role": "system", "content": "s" * 10},
            [
                {"role": "user", "content": "a" * 20},
                {"role": "assistant", "content": "b" * 20},
                {"role": "user", "content": "c" * 20},
            ],
        )

        assert [msg["content"] for msg in result[1:]] == ["b" * 20, "c" * 20]

    def test_reserve_tokens_floor_caps_recent_tail(self, tmp_path, monkeypatch):
        import importlib

        config = LCMConfig(
            fresh_tail_count=10,
            database_path=str(tmp_path / "lcm_headroom.db"),
            reserve_tokens_floor=40,
        )
        instance = LCMEngine(config=config)
        instance._session_id = "guardrail-session"
        instance.compression_count = 1
        instance.context_length = 100

        lcm_engine_module = importlib.import_module("hermes_lcm.engine")
        monkeypatch.setattr(
            lcm_engine_module,
            "count_message_tokens",
            lambda msg: len(msg.get("content", "")),
        )

        result = instance._assemble_context(
            {"role": "system", "content": "s" * 10},
            [
                {"role": "user", "content": "a" * 20},
                {"role": "assistant", "content": "b" * 20},
                {"role": "user", "content": "c" * 20},
            ],
        )

        assert [msg["content"] for msg in result[1:]] == ["b" * 20, "c" * 20]

    def test_max_assembly_tokens_keeps_tail_contiguous_with_varied_sizes(self, tmp_path, monkeypatch):
        import importlib

        config = LCMConfig(
            fresh_tail_count=10,
            database_path=str(tmp_path / "lcm_guardrail_varied.db"),
            max_assembly_tokens=70,
        )
        instance = LCMEngine(config=config)
        instance._session_id = "guardrail-session"
        instance.compression_count = 1

        lcm_engine_module = importlib.import_module("hermes_lcm.engine")
        monkeypatch.setattr(
            lcm_engine_module,
            "count_message_tokens",
            lambda msg: len(msg.get("content", "")),
        )

        result = instance._assemble_context(
            {"role": "system", "content": "s" * 10},
            [
                {"role": "user", "content": "a" * 10},
                {"role": "assistant", "content": "b" * 45},
                {"role": "user", "content": "c" * 20},
            ],
        )

        assert [msg["content"] for msg in result[1:]] == ["c" * 20]

    def test_summary_budget_keeps_summary_order_contiguous(self, tmp_path, monkeypatch):
        import importlib
        from hermes_lcm.dag import SummaryNode

        config = LCMConfig(
            fresh_tail_count=10,
            database_path=str(tmp_path / "lcm_guardrail_summary.db"),
            max_assembly_tokens=189,
        )
        instance = LCMEngine(config=config)
        instance._session_id = "guardrail-session"
        instance.compression_count = 1

        lcm_engine_module = importlib.import_module("hermes_lcm.engine")
        monkeypatch.setattr(
            lcm_engine_module,
            "count_message_tokens",
            lambda msg: len(msg.get("content", "")),
        )

        instance._dag.add_node(SummaryNode(
            session_id="guardrail-session", depth=2,
            summary="A" * 15, token_count=15,
            source_token_count=100, source_ids=[],
            source_type="messages", created_at=time.time(),
        ))
        instance._dag.add_node(SummaryNode(
            session_id="guardrail-session", depth=1,
            summary="B" * 120, token_count=120,
            source_token_count=200, source_ids=[],
            source_type="messages", created_at=time.time(),
        ))
        instance._dag.add_node(SummaryNode(
            session_id="guardrail-session", depth=0,
            summary="C" * 10, token_count=10,
            source_token_count=80, source_ids=[],
            source_type="messages", created_at=time.time(),
        ))

        result = instance._assemble_context(
            {"role": "system", "content": "s" * 10},
            [{"role": "user", "content": "tail" * 10}],
        )

        assert len(result) == 3
        summary_blob = result[1]["content"]
        assert "A" * 15 in summary_blob
        assert "B" * 120 not in summary_blob
        assert "C" * 10 not in summary_blob

    def test_max_assembly_tokens_keeps_newest_tail_message_even_if_it_alone_exceeds_cap(self, tmp_path, monkeypatch):
        import importlib

        config = LCMConfig(
            fresh_tail_count=10,
            database_path=str(tmp_path / "lcm_guardrail_newest.db"),
            max_assembly_tokens=50,
        )
        instance = LCMEngine(config=config)
        instance._session_id = "guardrail-session"
        instance.compression_count = 1

        lcm_engine_module = importlib.import_module("hermes_lcm.engine")
        monkeypatch.setattr(
            lcm_engine_module,
            "count_message_tokens",
            lambda msg: len(msg.get("content", "")),
        )

        result = instance._assemble_context(
            {"role": "system", "content": "s" * 10},
            [
                {"role": "user", "content": "a" * 20},
                {"role": "assistant", "content": "b" * 60},
            ],
        )

        assert [msg["content"] for msg in result[1:]] == ["b" * 60]

    def test_reserve_tokens_floor_warns_when_misconfigured(self, tmp_path, caplog):
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_guardrail_warn.db"),
            reserve_tokens_floor=100,
        )
        instance = LCMEngine(config=config)
        instance.context_length = 100

        with caplog.at_level(logging.WARNING, logger="hermes_lcm.engine"):
            assert instance._effective_assembly_token_cap() is None

        assert "reserve_tokens_floor=100 disables reserve-based assembly cap" in caplog.text


class TestEngineTools:
    def test_handle_grep(self, engine):
        # Add some data
        engine._store.append("test-session", {"role": "user", "content": "deploy docker containers"})
        result = json.loads(engine.handle_tool_call("lcm_grep", {"query": "docker"}))
        assert "results" in result

    def test_handle_describe_overview(self, engine):
        result = json.loads(engine.handle_tool_call("lcm_describe", {}))
        assert "session_id" in result
        assert "store_message_count" in result

    def test_handle_unknown_tool(self, engine):
        result = json.loads(engine.handle_tool_call("unknown_tool", {}))
        assert "error" in result

    def test_tool_dispatch_is_bound_to_engine_instance(self, tmp_path):
        config_a = LCMConfig(database_path=str(tmp_path / "a.db"))
        config_b = LCMConfig(database_path=str(tmp_path / "b.db"))

        engine_a = LCMEngine(config=config_a)
        engine_a._session_id = "session-a"
        engine_b = LCMEngine(config=config_b)
        engine_b._session_id = "session-b"

        engine_a._store.append("session-a", {"role": "user", "content": "alpha project"})
        engine_b._store.append("session-b", {"role": "user", "content": "beta project"})

        result_a = json.loads(engine_a.handle_tool_call("lcm_grep", {"query": "alpha"}))
        result_b = json.loads(engine_b.handle_tool_call("lcm_grep", {"query": "beta"}))

        assert result_a["total_results"] == 1
        assert result_b["total_results"] == 1
        assert "alpha" in result_a["results"][0]["snippet"]
        assert "beta" in result_b["results"][0]["snippet"]

    def test_handle_expand_query_requires_prompt(self, engine):
        result = json.loads(engine.handle_tool_call("lcm_expand_query", {"query": "docker"}))
        assert "error" in result
        assert "prompt" in result["error"]

    def test_handle_expand_query_uses_expansion_model(self, engine, monkeypatch):
        engine._config.expansion_model = "expansion-model-x"
        engine._store.append("test-session", {"role": "user", "content": "Discussed docker rollout plan"})
        node_id = engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="Docker rollout summary",
                token_count=10,
                source_token_count=20,
                source_ids=[1],
                source_type="messages",
                created_at=0,
            )
        )

        seen = {}

        def fake_synthesize(*, prompt, context_blocks, model, max_tokens, timeout):
            seen["prompt"] = prompt
            seen["context_blocks"] = context_blocks
            seen["model"] = model
            seen["max_tokens"] = max_tokens
            seen["timeout"] = timeout
            return "Expansion answer"

        monkeypatch.setattr("hermes_lcm.tools._synthesize_expansion_answer", fake_synthesize)

        result = json.loads(
            engine.handle_tool_call(
                "lcm_expand_query",
                {"query": "docker", "prompt": "What was the plan?", "max_tokens": 500},
            )
        )

        assert result["answer"] == "Expansion answer"
        assert result["model"] == "expansion-model-x"
        assert result["node_ids"] == [node_id]
        assert seen["model"] == "expansion-model-x"
        assert seen["timeout"] == engine._config.expansion_timeout_ms / 1000
        assert seen["max_tokens"] == 500
        assert seen["prompt"] == "What was the plan?"
        assert seen["context_blocks"]

    def test_describe_and_expand_are_session_scoped(self, engine):
        node_id = engine._dag.add_node(
            SummaryNode(
                session_id="session-a",
                depth=0,
                summary="secret summary",
                token_count=10,
                source_token_count=20,
                source_ids=[],
                source_type="messages",
                created_at=0,
            )
        )

        engine._session_id = "session-b"

        describe = json.loads(engine.handle_tool_call("lcm_describe", {"node_id": node_id}))
        expand = json.loads(engine.handle_tool_call("lcm_expand", {"node_id": node_id}))

        assert "error" in describe
        assert "error" in expand

    def test_describe_overview_includes_sparse_high_depth_nodes(self, engine):
        engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=2,
                summary="durable summary",
                token_count=100,
                source_token_count=500,
                source_ids=[],
                source_type="messages",
                created_at=0,
            )
        )

        overview = json.loads(engine.handle_tool_call("lcm_describe", {}))
        assert "d2" in overview["depths"]
