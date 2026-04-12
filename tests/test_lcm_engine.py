"""Integration tests for the LCM engine."""

import json
import pytest

from agent.context_engine import ContextEngine
from hermes_lcm.config import LCMConfig
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
    """Tests for issue #2c — removed config options."""

    def test_no_expansion_model(self):
        config = LCMConfig()
        assert not hasattr(config, "expansion_model")

    def test_no_summary_timeout(self):
        config = LCMConfig()
        assert not hasattr(config, "summary_timeout_ms")

    def test_no_delegation_timeout(self):
        config = LCMConfig()
        assert not hasattr(config, "delegation_timeout_ms")


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
