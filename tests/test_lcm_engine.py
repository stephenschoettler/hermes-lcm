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
