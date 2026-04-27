"""Integration tests for the LCM engine."""

import json
import logging
import time
import pytest

import hermes_lcm.engine as lcm_engine
import hermes_lcm.tools as lcm_tools

from agent.context_engine import ContextEngine
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_message_tokens, count_messages_tokens


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


class TestEscalationStripReasoning:
    """Regression tests for thinking-model reasoning-tag stripping in
    escalation._call_llm_for_summary. Some thinking models (MiniMax-M2.7,
    GLM-5.1, Qwen QwQ, DeepSeek R1) inline reasoning inside <think>...</think>
    blocks within message.content; without stripping, the reasoning text gets
    persisted as the summary node and confuses downstream lcm_expand_query."""

    def _install_fake_auxiliary_client(self, monkeypatch, fake_call_llm):
        """Install a minimal agent.auxiliary_client module for CI, where the
        hermes-agent package is only stubbed enough for ContextEngine tests."""
        import sys
        import types

        agent_mod = sys.modules.get("agent") or types.ModuleType("agent")
        aux_mod = types.ModuleType("agent.auxiliary_client")
        aux_mod.call_llm = fake_call_llm
        agent_mod.auxiliary_client = aux_mod
        monkeypatch.setitem(sys.modules, "agent", agent_mod)
        monkeypatch.setitem(sys.modules, "agent.auxiliary_client", aux_mod)
        return aux_mod

    def test_strip_reasoning_blocks_handles_each_supported_tag(self):
        from hermes_lcm.escalation import _strip_reasoning_blocks

        cases = [
            ("<think>internal reasoning</think>final summary", "final summary"),
            ("<thinking>plan</thinking>actual content", "actual content"),
            ("<reasoning>scratch</reasoning>output", "output"),
            ("<thought>idea</thought>summary text", "summary text"),
            ("<REASONING_SCRATCHPAD>foo</REASONING_SCRATCHPAD>bar", "bar"),
            ("multi\n<think>line\nblock</think>\nrest", "multi\n\nrest"),
            ("plain text without tags", "plain text without tags"),
            ("", ""),
        ]
        for raw, expected in cases:
            got = _strip_reasoning_blocks(raw)
            assert got == expected, f"input={raw!r} expected={expected!r} got={got!r}"

    def test_strip_reasoning_blocks_is_idempotent(self):
        from hermes_lcm.escalation import _strip_reasoning_blocks

        once = _strip_reasoning_blocks("<think>foo</think>bar")
        twice = _strip_reasoning_blocks(once)
        assert once == twice == "bar"

    def test_strip_reasoning_blocks_handles_multiple_blocks(self):
        from hermes_lcm.escalation import _strip_reasoning_blocks

        raw = "<think>a</think>visible1<think>b</think>visible2"
        assert _strip_reasoning_blocks(raw) == "visible1visible2"

    def test_strip_reasoning_blocks_preserves_content_with_unrelated_angle_brackets(self):
        from hermes_lcm.escalation import _strip_reasoning_blocks

        raw = "Decision: x < y, and config <foo> stays"
        assert _strip_reasoning_blocks(raw) == raw

    def test_call_llm_for_summary_strips_reasoning_from_response(self, monkeypatch):
        """Integration: when the auxiliary LLM returns reasoning-contaminated
        content, _call_llm_for_summary returns the stripped summary text."""
        import hermes_lcm.escalation as esc

        class _FakeMessage:
            def __init__(self, content):
                self.content = content

        class _FakeChoice:
            def __init__(self, content):
                self.message = _FakeMessage(content)

        class _FakeResponse:
            def __init__(self, content):
                self.choices = [_FakeChoice(content)]

        contaminated = (
            "<think>The user asks me to compress this into bullet points. "
            "I should focus on decisions, files, errors, current state...</think>"
            "Summary: docker rollout completed. Auth migration pending review."
        )

        def fake_call_llm(**kwargs):
            return _FakeResponse(contaminated)

        # Patch the import inside _call_llm_for_summary by providing the
        # module the function imports from at call time.
        self._install_fake_auxiliary_client(monkeypatch, fake_call_llm)

        result = esc._call_llm_for_summary(
            prompt="please summarize",
            max_tokens=200,
            model="any",
            timeout=10.0,
        )

        assert result is not None
        assert "<think>" not in result
        assert "</think>" not in result
        assert "Summary: docker rollout completed" in result


    def test_synthesize_expansion_answer_strips_reasoning_from_response(self, monkeypatch):
        """Integration: lcm_expand_query routes through
        tools._synthesize_expansion_answer, which is a separate LLM call path
        from _call_llm_for_summary. Both must strip reasoning blocks before
        returning, otherwise expand_query answers leak the model's internal
        reasoning back to the caller."""
        import hermes_lcm.tools as tools_mod

        class _FakeMessage:
            def __init__(self, content):
                self.content = content

        class _FakeChoice:
            def __init__(self, content):
                self.message = _FakeMessage(content)

        class _FakeResponse:
            def __init__(self, content):
                self.choices = [_FakeChoice(content)]

        contaminated = (
            "<think>The user is asking what was discussed. Let me look at the "
            "context blocks and synthesize an answer...</think>"
            "We discussed the docker rollout plan and auth migration."
        )

        def fake_call_llm(**kwargs):
            return _FakeResponse(contaminated)

        self._install_fake_auxiliary_client(monkeypatch, fake_call_llm)

        result = tools_mod._synthesize_expansion_answer(
            prompt="What was discussed?",
            context_blocks=[{"role": "user", "content": "ignored in fake"}],
            model="any",
            max_tokens=200,
            timeout=10.0,
        )

        assert result is not None
        assert "<think>" not in result
        assert "</think>" not in result
        assert "We discussed the docker rollout plan" in result


    def test_call_extraction_llm_strips_reasoning_from_response(self, monkeypatch):
        """Integration: pre-compaction extraction routes through
        extraction._call_extraction_llm, the third LLM call path on top of
        _call_llm_for_summary (escalation) and _synthesize_expansion_answer
        (tools). All three must strip reasoning blocks before returning,
        otherwise the daily extraction .md file ends up with the model's
        internal reasoning instead of clean bullet points."""
        import hermes_lcm.extraction as extr

        class _FakeMessage:
            def __init__(self, content):
                self.content = content

        class _FakeChoice:
            def __init__(self, content):
                self.message = _FakeMessage(content)

        class _FakeResponse:
            def __init__(self, content):
                self.choices = [_FakeChoice(content)]

        contaminated = (
            "<think>Let me extract the relevant information from this "
            "conversation segment. Decisions made: ... Let me format these "
            "as clean bullet points.</think>"
            "- Decision: ship docker rollout on 2026-07-22\n"
            "- Action: Yvonne files FCC paperwork by 2026-06-30"
        )

        def fake_call_llm(**kwargs):
            return _FakeResponse(contaminated)

        self._install_fake_auxiliary_client(monkeypatch, fake_call_llm)

        result = extr._call_extraction_llm(
            prompt="extract decisions",
            model="any",
            timeout=10.0,
        )

        assert result is not None
        assert "<think>" not in result
        assert "</think>" not in result
        assert "Decision: ship docker rollout" in result


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
        assert "lcm_status" in names
        assert "lcm_doctor" in names
        assert "lcm_expand_query" in names

        grep_schema = next(s for s in schemas if s["name"] == "lcm_grep")
        grep_props = grep_schema["parameters"]["properties"]
        assert "session_scope" in grep_props
        assert "source" in grep_props
        assert "descendant source lineage" in grep_props["source"]["description"]
        assert "unknown" in grep_props["source"]["description"]
        assert "current session" in grep_schema["description"].lower()
        assert "session_search" in grep_schema["description"]
        assert "session_search" in grep_props["session_scope"]["description"]

        describe_schema = next(s for s in schemas if s["name"] == "lcm_describe")
        expand_schema = next(s for s in schemas if s["name"] == "lcm_expand")
        expand_query_schema = next(s for s in schemas if s["name"] == "lcm_expand_query")

        assert "current session" in describe_schema["description"].lower()
        assert "session_search" in describe_schema["description"]
        assert "current session" in expand_schema["description"].lower()
        assert "session_search" in expand_schema["description"]
        assert "current session" in expand_query_schema["description"].lower()
        assert "session_search" in expand_query_schema["description"]

    def test_should_compress(self, engine):
        assert not engine.should_compress(1000)
        assert engine.should_compress(engine.threshold_tokens)

    def test_should_compress_when_explicit_assembly_cap_is_hit(self, tmp_path):
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_should_compress_cap.db"),
            max_assembly_tokens=90,
        )
        instance = LCMEngine(config=config)
        instance.context_length = 200000
        instance.threshold_tokens = int(200000 * config.context_threshold)

        assert instance.should_compress(90)

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

    def test_on_session_start_resets_session_scoped_runtime_when_binding_new_session(self, engine):
        engine.compression_count = 5
        engine.last_prompt_tokens = 9999
        engine.last_completion_tokens = 333
        engine.last_total_tokens = 10332
        engine._last_compacted_store_id = 42
        engine._ingest_cursor = 7
        engine._context_probed = True
        engine._context_probe_persistable = True
        engine.on_session_start("fresh-session", platform="telegram", context_length=200000)

        assert engine._session_id == "fresh-session"
        assert engine.compression_count == 0
        assert engine.last_prompt_tokens == 0
        assert engine.last_completion_tokens == 0
        assert engine.last_total_tokens == 0
        assert engine._last_compacted_store_id == 0
        assert engine._ingest_cursor == 0
        assert engine._context_probed is False
        assert engine._context_probe_persistable is False

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

    def test_filter_config_diagnostics_log_only_once_per_engine_instance(self, tmp_path, caplog):
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_ignore_once.db"),
            ignore_session_patterns=["cron:*"],
            ignore_session_patterns_source="env",
        )
        instance = LCMEngine(config=config)

        with caplog.at_level("INFO", logger="hermes_lcm.engine"):
            instance.on_session_start("cron_123", platform="cron", context_length=1000)
            instance.on_session_start("cron_456", platform="cron", context_length=1000)

        assert caplog.text.count("LCM ignore_session_patterns from env: cron:*") == 1
        assert caplog.text.count("matched ignore_session_patterns") == 2

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

    def test_compress_leaf_node_tracks_source_window_from_message_timestamps(self, engine):
        messages = self._make_long_conversation(20)
        engine._ingest_messages(messages)
        all_rows = engine._store.get_session_messages("test-session")
        expected_store_ids = [row["store_id"] for row in all_rows[1:-engine._config.fresh_tail_count]]
        for idx, store_id in enumerate(expected_store_ids):
            engine._store._conn.execute(
                "UPDATE messages SET timestamp = ? WHERE store_id = ?",
                (1_700_000_000 + idx, store_id),
            )
        engine._store._conn.commit()

        import hermes_lcm.engine as engine_module
        original_fn = engine_module.summarize_with_escalation

        def mock_summary(**kwargs):
            return "Leaf summary.\nExpand for details about: leaf window", 1

        engine_module.summarize_with_escalation = mock_summary
        try:
            engine.compress(messages)
            node = engine._dag.get_session_nodes("test-session")[0]
            assert node.source_ids == expected_store_ids
            assert node.earliest_at == 1_700_000_000
            assert node.latest_at == 1_700_000_000 + len(expected_store_ids) - 1
        finally:
            engine_module.summarize_with_escalation = original_fn

    def test_condensed_parent_node_tracks_child_source_window(self, engine, monkeypatch):
        child_windows = [
            (1_700_000_010, 1_700_000_020),
            (1_700_000_030, 1_700_000_040),
            (1_700_000_050, 1_700_000_060),
            (1_700_000_070, 1_700_000_080),
        ]
        for idx, (earliest_at, latest_at) in enumerate(child_windows, start=1):
            engine._dag.add_node(SummaryNode(
                session_id="test-session",
                depth=0,
                summary=f"child {idx}",
                token_count=10,
                source_ids=[idx],
                source_type="messages",
                created_at=1_900_000_000 + idx,
                earliest_at=earliest_at,
                latest_at=latest_at,
            ))

        import hermes_lcm.engine as engine_module

        def mock_summary(**kwargs):
            return "Parent summary.\nExpand for details about: parent window", 1

        monkeypatch.setattr(engine_module, "summarize_with_escalation", mock_summary)

        engine._maybe_condense()

        nodes = engine._dag.get_session_nodes("test-session")
        parent = next(node for node in nodes if node.depth == 1)
        assert parent.earliest_at == child_windows[0][0]
        assert parent.latest_at == child_windows[-1][1]

    def test_dynamic_leaf_chunk_sizing_compacts_only_oldest_bounded_raw_chunk(self, tmp_path, monkeypatch):
        config = LCMConfig(
            fresh_tail_count=2,
            leaf_chunk_tokens=50,
            dynamic_leaf_chunk_enabled=True,
            dynamic_leaf_chunk_max=120,
            database_path=str(tmp_path / "lcm_dynamic_leaf.db"),
        )
        engine = LCMEngine(config=config)
        engine._session_id = "test-session"
        engine.context_length = 200000
        engine.threshold_tokens = int(200000 * config.context_threshold)

        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        for i in range(6):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({
                "role": role,
                "content": f"Message {i}: " + ("chunk " * 35),
            })

        candidate_raw = messages[1:-config.fresh_tail_count]
        candidate_tokens = [count_message_tokens(msg) for msg in candidate_raw]
        assert len(candidate_raw) == 4
        assert sum(candidate_tokens) > config.dynamic_leaf_chunk_max
        assert sum(candidate_tokens[:2]) <= config.dynamic_leaf_chunk_max
        assert sum(candidate_tokens[:3]) > config.dynamic_leaf_chunk_max

        import hermes_lcm.engine as engine_module

        def mock_summary(**kwargs):
            return "Dynamic leaf summary.\nExpand for details about: oldest raw chunk", 1

        monkeypatch.setattr(engine_module, "summarize_with_escalation", mock_summary)

        compressed = engine.compress(messages)

        nodes = engine._dag.get_session_nodes("test-session")
        assert len(nodes) == 1
        node = nodes[0]

        stored = engine._store.get_session_messages("test-session")
        selected_contents = [
            engine._store.get(store_id)["content"]
            for store_id in node.source_ids
        ]

        assert len(node.source_ids) == 2
        assert selected_contents == [msg["content"] for msg in candidate_raw[:2]]

        compressed_contents = [msg.get("content") for msg in compressed]
        assert candidate_raw[2]["content"] in compressed_contents
        assert candidate_raw[3]["content"] in compressed_contents
        assert messages[-1]["content"] in compressed_contents
        assert len(stored) == len(messages)

    def test_adaptive_leaf_rescue_retries_with_smaller_oldest_chunk(self, tmp_path, monkeypatch):
        config = LCMConfig(
            fresh_tail_count=2,
            leaf_chunk_tokens=50,
            dynamic_leaf_chunk_enabled=True,
            dynamic_leaf_chunk_max=120,
            database_path=str(tmp_path / "lcm_dynamic_leaf_retry.db"),
        )
        engine = LCMEngine(config=config)
        engine._session_id = "test-session"
        engine.context_length = 200000
        engine.threshold_tokens = int(200000 * config.context_threshold)

        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        for i in range(6):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({
                "role": role,
                "content": f"Message {i}: " + ("chunk " * 35),
            })

        candidate_raw = messages[1:-config.fresh_tail_count]
        initial_chunk = engine._select_oldest_leaf_chunk(
            candidate_raw,
            engine._working_leaf_chunk_tokens(count_messages_tokens(candidate_raw)),
        )
        assert len(initial_chunk) == 2
        first_msg_tokens = count_message_tokens(candidate_raw[0])
        assert count_messages_tokens(initial_chunk) > first_msg_tokens

        import hermes_lcm.engine as engine_module

        attempts: list[int] = []

        def flaky_summary(**kwargs):
            attempts.append(kwargs["source_tokens"])
            if kwargs["source_tokens"] > first_msg_tokens:
                raise RuntimeError("context length exceeded")
            return "Recovered smaller leaf summary.\nExpand for details about: oldest raw chunk", 1

        monkeypatch.setattr(engine_module, "summarize_with_escalation", flaky_summary)

        compressed = engine.compress(messages)

        assert len(attempts) == 2
        assert attempts[0] > attempts[1]

        nodes = engine._dag.get_session_nodes("test-session")
        assert len(nodes) == 1
        node = nodes[0]
        selected_contents = [engine._store.get(store_id)["content"] for store_id in node.source_ids]
        assert len(node.source_ids) == 1
        assert selected_contents == [candidate_raw[0]["content"]]

        compressed_contents = [msg.get("content") for msg in compressed]
        assert candidate_raw[1]["content"] in compressed_contents
        assert candidate_raw[2]["content"] in compressed_contents
        assert candidate_raw[3]["content"] in compressed_contents

    def test_dynamic_leaf_chunk_sizing_runs_bounded_catchup_passes_when_pressure_remains_high(self, tmp_path, monkeypatch):
        config = LCMConfig(
            fresh_tail_count=4,
            leaf_chunk_tokens=180,
            dynamic_leaf_chunk_enabled=True,
            dynamic_leaf_chunk_max=360,
            database_path=str(tmp_path / "lcm_dynamic_leaf_catchup.db"),
        )
        engine = LCMEngine(config=config)
        engine._session_id = "test-session"
        engine.context_length = 1200
        # Set threshold so estimated_active_tokens stays above it across at
        # least two compaction passes (794 → ~463 after pass 1 → ~375 after
        # pass 2), forcing the loop to run bounded catch-up.
        engine.threshold_tokens = 450

        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        for i in range(16):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({
                "role": role,
                "content": f"Message {i}: " + ("dense " * 40),
            })

        import hermes_lcm.engine as engine_module

        def mock_summary(**kwargs):
            return "Catchup summary.\nExpand for details about: oldest raw chunk", 1

        monkeypatch.setattr(engine_module, "summarize_with_escalation", mock_summary)

        compressed = engine.compress(messages, current_tokens=count_messages_tokens(messages))

        nodes = engine._dag.get_session_nodes("test-session")
        assert len(nodes) >= 2
        # Verify compaction reduced token count (assembly adds an LCM note to
        # the system message, so compare against starting tokens, not threshold)
        assert count_messages_tokens(compressed) < count_messages_tokens(messages)
        compressed_contents = [msg.get("content") for msg in compressed]
        assert messages[-1]["content"] in compressed_contents

    def test_adaptive_leaf_rescue_stops_after_bounded_retry_worthy_failures(self, tmp_path, monkeypatch):
        config = LCMConfig(
            fresh_tail_count=2,
            leaf_chunk_tokens=50,
            dynamic_leaf_chunk_enabled=True,
            dynamic_leaf_chunk_max=120,
            database_path=str(tmp_path / "lcm_dynamic_leaf_retry_fail.db"),
        )
        engine = LCMEngine(config=config)
        engine._session_id = "test-session"
        engine.context_length = 200000
        engine.threshold_tokens = int(200000 * config.context_threshold)

        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        for i in range(6):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({
                "role": role,
                "content": f"Message {i}: " + ("chunk " * 35),
            })

        import hermes_lcm.engine as engine_module

        attempts: list[int] = []

        def always_fails(**kwargs):
            attempts.append(kwargs["source_tokens"])
            raise RuntimeError("context length exceeded")

        monkeypatch.setattr(engine_module, "summarize_with_escalation", always_fails)

        with pytest.raises(RuntimeError, match="context length exceeded"):
            engine.compress(messages)

        assert len(attempts) == 2
        assert attempts[0] > attempts[-1]
        assert engine._dag.get_session_nodes("test-session") == []

    def test_adaptive_leaf_rescue_does_not_retry_non_retry_worthy_errors(self, tmp_path, monkeypatch):
        config = LCMConfig(
            fresh_tail_count=2,
            leaf_chunk_tokens=50,
            dynamic_leaf_chunk_enabled=True,
            dynamic_leaf_chunk_max=120,
            database_path=str(tmp_path / "lcm_dynamic_leaf_retry_nonretry.db"),
        )
        engine = LCMEngine(config=config)
        engine._session_id = "test-session"
        engine.context_length = 200000
        engine.threshold_tokens = int(200000 * config.context_threshold)

        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        for i in range(6):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({
                "role": role,
                "content": f"Message {i}: " + ("chunk " * 35),
            })

        import hermes_lcm.engine as engine_module

        call_count = 0

        def bad_template(**kwargs):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("template exploded")

        monkeypatch.setattr(engine_module, "summarize_with_escalation", bad_template)

        with pytest.raises(RuntimeError, match="template exploded"):
            engine.compress(messages)

        assert call_count == 1
        assert engine._dag.get_session_nodes("test-session") == []

    def test_cache_friendly_gating_suppresses_follow_on_condensation_for_single_fanin_group(self, tmp_path, monkeypatch):
        config = LCMConfig(
            fresh_tail_count=2,
            leaf_chunk_tokens=50,
            dynamic_leaf_chunk_enabled=True,
            dynamic_leaf_chunk_max=120,
            condensation_fanin=2,
            database_path=str(tmp_path / "lcm_cache_friendly_suppress.db"),
        )
        config.cache_friendly_condensation_enabled = True
        config.cache_friendly_min_debt_groups = 2
        engine = LCMEngine(config=config)
        engine._session_id = "test-session"
        engine.context_length = 200000
        engine.threshold_tokens = int(200000 * config.context_threshold)

        engine._dag.add_node(SummaryNode(
            session_id="test-session",
            depth=0,
            summary="Earlier leaf",
            token_count=40,
            source_token_count=80,
            source_ids=[1],
            source_type="messages",
            created_at=time.time() - 10,
            expand_hint="earlier leaf",
        ))

        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        for i in range(6):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({
                "role": role,
                "content": f"Message {i}: " + ("chunk " * 35),
            })

        import hermes_lcm.engine as engine_module

        def mock_summary(**kwargs):
            if kwargs["depth"] == 0:
                return "Leaf summary.\nExpand for details about: oldest raw chunk", 1
            return "Condensed summary.\nExpand for details about: d0 summaries", 1

        monkeypatch.setattr(engine_module, "summarize_with_escalation", mock_summary)

        engine.compress(messages)

        depth0 = engine._dag.get_session_nodes("test-session", depth=0)
        depth1 = engine._dag.get_session_nodes("test-session", depth=1)
        assert len(depth0) == 2
        assert depth1 == []
        assert engine.get_status()["condensation_suppressed_reason"] == "cache_friendly_single_group"

    def test_cache_friendly_gating_allows_condensation_when_debt_reaches_two_groups(self, tmp_path, monkeypatch):
        config = LCMConfig(
            fresh_tail_count=2,
            leaf_chunk_tokens=50,
            dynamic_leaf_chunk_enabled=True,
            dynamic_leaf_chunk_max=120,
            condensation_fanin=2,
            database_path=str(tmp_path / "lcm_cache_friendly_debt.db"),
        )
        config.cache_friendly_condensation_enabled = True
        config.cache_friendly_min_debt_groups = 2
        engine = LCMEngine(config=config)
        engine._session_id = "test-session"
        engine.context_length = 200000
        engine.threshold_tokens = int(200000 * config.context_threshold)

        for i in range(3):
            engine._dag.add_node(SummaryNode(
                session_id="test-session",
                depth=0,
                summary=f"Earlier leaf {i}",
                token_count=40,
                source_token_count=80,
                source_ids=[i + 1],
                source_type="messages",
                created_at=time.time() - (10 + i),
                expand_hint=f"earlier leaf {i}",
            ))

        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        for i in range(6):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({
                "role": role,
                "content": f"Message {i}: " + ("chunk " * 35),
            })

        import hermes_lcm.engine as engine_module

        def mock_summary(**kwargs):
            if kwargs["depth"] == 0:
                return "Leaf summary.\nExpand for details about: oldest raw chunk", 1
            return "Condensed summary.\nExpand for details about: d0 summaries", 1

        monkeypatch.setattr(engine_module, "summarize_with_escalation", mock_summary)

        engine.compress(messages)

        depth1 = engine._dag.get_session_nodes("test-session", depth=1)
        assert len(depth1) == 1
        assert engine.get_status()["condensation_suppressed_reason"] == ""

    def test_cache_friendly_gating_does_not_block_forced_overflow_condensation(self, tmp_path, monkeypatch):
        config = LCMConfig(
            fresh_tail_count=1,
            leaf_chunk_tokens=50,
            dynamic_leaf_chunk_enabled=True,
            dynamic_leaf_chunk_max=120,
            condensation_fanin=2,
            max_assembly_tokens=90,
            database_path=str(tmp_path / "lcm_cache_friendly_overflow.db"),
        )
        config.cache_friendly_condensation_enabled = True
        config.cache_friendly_min_debt_groups = 2
        engine = LCMEngine(config=config)
        engine._session_id = "test-session"
        engine.context_length = 200000
        engine.threshold_tokens = int(200000 * config.context_threshold)

        engine._dag.add_node(SummaryNode(
            session_id="test-session",
            depth=0,
            summary="Earlier leaf",
            token_count=40,
            source_token_count=80,
            source_ids=[1],
            source_type="messages",
            created_at=time.time() - 10,
            expand_hint="earlier leaf",
        ))

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "A " * 40},
            {"role": "assistant", "content": "B " * 40},
            {"role": "user", "content": "Tail " * 60},
        ]

        import hermes_lcm.engine as engine_module

        def mock_summary(**kwargs):
            if kwargs["depth"] == 0:
                return "Leaf summary.\nExpand for details about: oldest raw chunk", 1
            return "Condensed summary.\nExpand for details about: d0 summaries", 1

        monkeypatch.setattr(engine_module, "summarize_with_escalation", mock_summary)

        engine.compress(messages, current_tokens=120)

        depth1 = engine._dag.get_session_nodes("test-session", depth=1)
        assert len(depth1) == 1
        assert engine.get_status()["condensation_suppressed_reason"] == ""


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

    def test_carry_over_preserves_source_lineage_for_reassigned_nodes(self, engine):
        discord_store_id = engine._store.append(
            "old-session",
            {"role": "user", "content": "docker logs from discord"},
            source="discord",
        )
        engine._dag.add_node(SummaryNode(
            session_id="old-session",
            depth=2,
            summary="retained discord docker summary",
            token_count=100,
            source_token_count=200,
            source_ids=[discord_store_id],
            source_type="messages",
            created_at=time.time(),
        ))

        engine._session_id = "old-session"
        engine.on_session_reset()
        moved = engine.carry_over_new_session_context("old-session", "new-session")
        engine._session_id = "new-session"

        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {"query": "docker", "session_scope": "current", "source": "discord", "limit": 10},
            )
        )

        assert moved == 1
        assert any(item["type"] == "summary" for item in result["results"])
        assert all(item.get("session_id") == "new-session" for item in result["results"])
        assert any(
            "retained discord docker summary" in item.get("snippet", "")
            for item in result["results"]
            if item["type"] == "summary"
        )


class TestSessionRollover:
    def test_rollover_session_rebinds_engine_and_carries_retained_nodes(self, engine):
        engine._config.new_session_retain_depth = 2
        from hermes_lcm.dag import SummaryNode
        import time

        engine.on_session_start("old-session", platform="cli", context_length=200000)
        for depth in range(4):
            engine._dag.add_node(SummaryNode(
                session_id="old-session", depth=depth,
                summary=f"old d{depth}", token_count=100,
                source_token_count=500, source_ids=[],
                source_type="messages", created_at=time.time(),
            ))

        moved = engine.rollover_session(
            "old-session",
            "new-session",
            previous_messages=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world"},
            ],
            platform="cli",
            context_length=200000,
        )

        assert moved == 2
        assert engine._session_id == "new-session"
        assert engine._session_platform == "cli"
        assert engine._store.get_session_count("old-session") == 3
        assert engine._dag.get_session_nodes("old-session") == []
        new_nodes = engine._dag.get_session_nodes("new-session")
        assert len(new_nodes) == 2
        assert all(node.depth >= 2 for node in new_nodes)

    def test_rollover_session_supports_repeated_new_session_boundaries_without_duplicate_nodes(self, engine):
        engine._config.new_session_retain_depth = 2
        from hermes_lcm.dag import SummaryNode
        import time

        engine.on_session_start("s1", platform="cli", context_length=200000)
        for depth in range(4):
            engine._dag.add_node(SummaryNode(
                session_id="s1", depth=depth,
                summary=f"seed d{depth}", token_count=100,
                source_token_count=500, source_ids=[],
                source_type="messages", created_at=time.time(),
            ))

        moved1 = engine.rollover_session("s1", "s2", previous_messages=[], platform="cli", context_length=200000)
        assert moved1 == 2

        engine._dag.add_node(SummaryNode(
            session_id="s2", depth=2,
            summary="fresh d2", token_count=100,
            source_token_count=500, source_ids=[],
            source_type="messages", created_at=time.time(),
        ))
        engine._dag.add_node(SummaryNode(
            session_id="s2", depth=0,
            summary="fresh d0", token_count=100,
            source_token_count=500, source_ids=[],
            source_type="messages", created_at=time.time(),
        ))

        moved2 = engine.rollover_session("s2", "s3", previous_messages=[], platform="cli", context_length=200000)

        assert moved2 == 3
        s3_nodes = engine._dag.get_session_nodes("s3")
        assert len(s3_nodes) == 3
        assert sorted(node.summary for node in s3_nodes) == ["fresh d2", "seed d2", "seed d3"]
        assert engine._dag.get_session_nodes("s2") == []
        assert engine._session_id == "s3"

    def test_rollover_session_skips_carry_over_when_old_session_is_not_bound(self, engine):
        engine._config.new_session_retain_depth = 2

        engine.on_session_start("attacker-current", platform="cli", context_length=200000)
        engine._dag.add_node(SummaryNode(
            session_id="victim-session",
            depth=2,
            summary="victim summary",
            token_count=100,
            source_token_count=200,
            source_ids=[],
            source_type="messages",
            created_at=time.time(),
        ))

        moved = engine.rollover_session(
            "victim-session",
            "attacker-new",
            previous_messages=[],
            platform="cli",
            context_length=200000,
        )

        assert moved == 0
        assert len(engine._dag.get_session_nodes("victim-session")) == 1
        assert engine._dag.get_session_nodes("attacker-new") == []
        assert engine._session_id == "attacker-new"

    def test_compression_boundary_continues_logical_session_without_resetting_state(self, engine):
        engine.on_session_start("old-session", platform="telegram", context_length=200000)
        store_id = engine._store.append(
            "old-session",
            {"role": "user", "content": "important pre-rollover context"},
            token_estimate=17,
            source="telegram",
        )
        engine._dag.add_node(SummaryNode(
            session_id="old-session",
            depth=0,
            summary="pre-rollover summary",
            token_count=5,
            source_token_count=17,
            source_ids=[store_id],
            source_type="messages",
            created_at=time.time(),
        ))
        engine.compression_count = 1
        engine.last_prompt_tokens = 1000
        engine.last_completion_tokens = 50
        engine.last_total_tokens = 1050
        engine._last_compacted_store_id = store_id
        engine._ingest_cursor = 2
        old_conversation_id = engine._conversation_id

        engine.on_session_start(
            "new-session",
            boundary_reason="compression",
            old_session_id="old-session",
            platform="telegram",
            context_length=200000,
        )

        assert engine._session_id == "new-session"
        assert engine._conversation_id == old_conversation_id
        assert engine.compression_count == 1
        assert engine.last_prompt_tokens == 1000
        assert engine.last_completion_tokens == 50
        assert engine.last_total_tokens == 1050
        assert engine._last_compacted_store_id == store_id
        assert engine._ingest_cursor == 2
        assert engine._store.get_session_count("old-session") == 0
        assert engine._store.get_session_count("new-session") == 1
        assert engine._dag.get_session_nodes("old-session") == []
        new_nodes = engine._dag.get_session_nodes("new-session")
        assert len(new_nodes) == 1
        assert new_nodes[0].summary == "pre-rollover summary"

        status = engine.get_status()
        assert status["store_messages"] == 1
        assert status["dag_nodes"] == 1
        assert status["compression_count"] == 1
        assert status["lifecycle"]["current_session_id"] == "new-session"
        assert status["lifecycle"]["last_finalized_session_id"] == "old-session"
        assert status["lifecycle"]["current_frontier_store_id"] == store_id
        assert status["lifecycle"]["last_finalized_frontier_store_id"] == store_id
        assert status["lifecycle"]["last_rollover_at"] is not None
        assert status["lifecycle"]["last_reset_at"] is None

    def test_compression_boundary_mismatch_resets_session_scoped_state(self, engine):
        engine.on_session_start("bound-session", platform="telegram", context_length=200000)
        engine.compression_count = 3
        engine.last_prompt_tokens = 900
        engine.last_completion_tokens = 12
        engine.last_total_tokens = 912
        engine._last_compacted_store_id = 42
        engine._ingest_cursor = 7
        old_conversation_id = engine._conversation_id

        engine.on_session_start(
            "new-session",
            boundary_reason="compression",
            old_session_id="different-old-session",
            platform="telegram",
            context_length=200000,
        )

        assert engine._session_id == "new-session"
        assert engine._conversation_id != old_conversation_id
        assert engine.compression_count == 0
        assert engine.last_prompt_tokens == 0
        assert engine.last_completion_tokens == 0
        assert engine.last_total_tokens == 0
        assert engine._last_compacted_store_id == 0
        assert engine._ingest_cursor == 0

    def test_compression_boundary_reassigns_externalized_payload_session_metadata(self, tmp_path):
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_compression_externalized.db"),
            large_output_externalization_enabled=True,
            large_output_externalization_threshold_chars=200,
        )
        engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
        engine.on_session_start("old-session", platform="telegram", context_length=200000)

        content = "RESULT:\n" + ("abcdef" * 2000)
        engine._serialize_messages([
            {"role": "tool", "tool_call_id": "call_big", "content": content}
        ])
        payload_file = next((tmp_path / "hermes" / "lcm-large-outputs").glob("*.json"))
        placeholder = (
            "[Externalized tool output: tool_call_id=call_big; "
            f"chars={len(content)}; bytes={len(content.encode('utf-8'))}; ref={payload_file.name}]"
        )
        store_id = engine._store.append(
            "old-session",
            {"role": "tool", "tool_call_id": "call_big", "content": placeholder},
            token_estimate=17,
            source="telegram",
        )
        node_id = engine._dag.add_node(SummaryNode(
            session_id="old-session",
            depth=0,
            summary="Externalized tool-output summary",
            token_count=10,
            source_token_count=17,
            source_ids=[store_id],
            source_type="messages",
            created_at=time.time(),
        ))
        assert json.loads(payload_file.read_text())["session_id"] == "old-session"

        engine.on_session_start(
            "new-session",
            boundary_reason="compression",
            old_session_id="old-session",
            platform="telegram",
            context_length=200000,
        )

        assert json.loads(payload_file.read_text())["session_id"] == "new-session"
        result = json.loads(engine.handle_tool_call("lcm_expand", {"node_id": node_id}))
        assert result["expanded"][0]["externalized"]["session_id"] == "new-session"
        assert result["expanded"][0]["externalized"]["tool_call_id"] == "call_big"

    def test_rollover_session_records_durable_lifecycle_state_idempotently(self, engine):
        engine._config.new_session_retain_depth = 2
        from hermes_lcm.dag import SummaryNode
        import time

        engine.on_session_start("s1", platform="cli", context_length=200000)
        for depth in range(3):
            engine._dag.add_node(SummaryNode(
                session_id="s1", depth=depth,
                summary=f"seed d{depth}", token_count=100,
                source_token_count=500, source_ids=[],
                source_type="messages", created_at=time.time(),
            ))

        moved = engine.rollover_session("s1", "s2", previous_messages=[], platform="cli", context_length=200000)
        assert moved == 1

        state = engine._lifecycle.get_by_conversation(engine._conversation_id)
        assert state is not None
        assert state.current_session_id == "s2"
        assert state.last_finalized_session_id == "s1"

        moved_repeat = engine.rollover_session("s1", "s2", previous_messages=[], platform="cli", context_length=200000)
        assert moved_repeat == 0

        state_repeat = engine._lifecycle.get_by_conversation(engine._conversation_id)
        assert state_repeat is not None
        assert state_repeat.current_session_id == "s2"
        assert state_repeat.last_finalized_session_id == "s1"
        assert engine._lifecycle.row_count() == 1

    def test_on_session_start_recovers_durable_lifecycle_state_after_restart(self, engine, monkeypatch):
        engine.on_session_start("active-session", platform="cli", context_length=200000)
        monkeypatch.setattr(
            lcm_engine,
            "summarize_with_escalation",
            lambda **kwargs: ("durable summary", 1),
        )

        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "alpha " * 80},
            {"role": "assistant", "content": "beta " * 80},
            {"role": "user", "content": "gamma " * 80},
            {"role": "assistant", "content": "delta " * 80},
            {"role": "user", "content": "epsilon " * 80},
            {"role": "assistant", "content": "zeta"},
        ]
        engine.compress(messages)
        old_conversation_id = engine._conversation_id
        old_frontier = engine._last_compacted_store_id
        assert old_frontier > 0

        restarted = LCMEngine(config=engine._config)
        restarted.on_session_start("active-session", platform="cli", context_length=200000)

        assert restarted._conversation_id == old_conversation_id
        assert restarted._last_compacted_store_id == old_frontier
        recovered = restarted._lifecycle.get_by_conversation(old_conversation_id)
        assert recovered is not None
        assert recovered.current_session_id == "active-session"

    def test_frontier_marker_only_advances_after_successful_leaf_compaction(self, engine, monkeypatch):
        engine.on_session_start("frontier-session", platform="cli", context_length=200000)
        monkeypatch.setattr(
            lcm_engine,
            "summarize_with_escalation",
            lambda **kwargs: ("frontier summary", 1),
        )

        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "alpha " * 80},
            {"role": "assistant", "content": "beta " * 80},
            {"role": "user", "content": "gamma " * 80},
            {"role": "assistant", "content": "delta " * 80},
            {"role": "user", "content": "epsilon " * 80},
            {"role": "assistant", "content": "zeta"},
        ]
        engine.compress(messages)

        state = engine._lifecycle.get_by_conversation(engine._conversation_id)
        assert state is not None
        assert state.current_frontier_store_id == engine._last_compacted_store_id
        frontier_before_failure = state.current_frontier_store_id

        monkeypatch.setattr(
            lcm_engine,
            "summarize_with_escalation",
            lambda **kwargs: (_ for _ in ()).throw(TimeoutError("summary timed out")),
        )
        failing_messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "epsilon " * 80},
            {"role": "assistant", "content": "zeta " * 80},
            {"role": "user", "content": "eta " * 80},
            {"role": "assistant", "content": "theta " * 80},
            {"role": "user", "content": "iota " * 80},
            {"role": "assistant", "content": "kappa"},
        ]
        with pytest.raises(TimeoutError):
            engine.compress(failing_messages)

        after_failure = engine._lifecycle.get_by_conversation(engine._conversation_id)
        assert after_failure is not None
        assert after_failure.current_frontier_store_id == frontier_before_failure

    def test_rollover_resets_active_frontier_but_preserves_last_finalized_frontier(self, engine, monkeypatch):
        engine.on_session_start("frontier-old", platform="cli", context_length=200000)
        monkeypatch.setattr(
            lcm_engine,
            "summarize_with_escalation",
            lambda **kwargs: ("rollover summary", 1),
        )
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "alpha " * 80},
            {"role": "assistant", "content": "beta " * 80},
            {"role": "user", "content": "gamma " * 80},
            {"role": "assistant", "content": "delta " * 80},
            {"role": "user", "content": "epsilon " * 80},
            {"role": "assistant", "content": "zeta"},
        ]
        engine.compress(messages)
        old_frontier = engine._last_compacted_store_id

        engine.rollover_session("frontier-old", "frontier-new", previous_messages=[], platform="cli", context_length=200000)

        state = engine._lifecycle.get_by_conversation(engine._conversation_id)
        assert state is not None
        assert state.current_session_id == "frontier-new"
        assert state.current_frontier_store_id == 0
        assert state.last_finalized_session_id == "frontier-old"
        assert state.last_finalized_frontier_store_id == old_frontier
        assert state.last_rollover_at is not None
        assert state.last_reset_at is not None


class TestDeferredMaintenanceDebt:
    @staticmethod
    def _make_backlog_messages(count: int = 12) -> list[dict]:
        messages = [{"role": "system", "content": "sys"}]
        for i in range(count):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": (f"chunk-{i} " * 220).strip()})
        return messages

    def test_debt_persists_when_bounded_leaf_passes_leave_raw_backlog(self, engine, monkeypatch):
        engine._config.dynamic_leaf_chunk_enabled = True
        engine._config.dynamic_leaf_chunk_max = 100
        engine._config.leaf_chunk_tokens = 100
        engine._config.fresh_tail_count = 2
        engine._config.deferred_maintenance_enabled = True
        engine._config.deferred_maintenance_max_passes = 1
        engine.on_session_start("debt-session", platform="cli", context_length=200000)

        monkeypatch.setattr(lcm_engine, "summarize_with_escalation", lambda **kwargs: ("debt summary", 1))
        monkeypatch.setattr(engine, "_working_leaf_chunk_tokens", lambda raw_tokens: 100)
        monkeypatch.setattr(
            engine,
            "_assemble_context",
            lambda system_msg, tail_messages, assembly_cap_override=None, include_lcm_note=True: [system_msg, *tail_messages],
        )

        compressed = engine.compress(self._make_backlog_messages())
        state = engine._lifecycle.get_by_conversation(engine._conversation_id)

        assert state is not None
        assert state.debt_kind == "raw_backlog"
        assert state.debt_size_estimate > 0
        assert engine.should_compress_preflight(compressed) is True
        refreshed = engine._lifecycle.get_by_conversation(engine._conversation_id)
        assert refreshed is not None
        assert refreshed.debt_kind == "raw_backlog"

    def test_bounded_catchup_reduces_then_clears_debt_only_after_backlog_shrinks(self, engine, monkeypatch):
        engine._config.dynamic_leaf_chunk_enabled = True
        engine._config.dynamic_leaf_chunk_max = 100
        engine._config.leaf_chunk_tokens = 100
        engine._config.fresh_tail_count = 2
        engine._config.deferred_maintenance_enabled = True
        engine.on_session_start("debt-session", platform="cli", context_length=200000)

        monkeypatch.setattr(lcm_engine, "summarize_with_escalation", lambda **kwargs: ("debt summary", 1))
        monkeypatch.setattr(engine, "_working_leaf_chunk_tokens", lambda raw_tokens: 100)
        monkeypatch.setattr(
            engine,
            "_assemble_context",
            lambda system_msg, tail_messages, assembly_cap_override=None, include_lcm_note=True: [system_msg, *tail_messages],
        )

        first = engine.compress(self._make_backlog_messages())
        debt1 = engine._lifecycle.get_by_conversation(engine._conversation_id)
        assert debt1 is not None and debt1.debt_kind == "raw_backlog"

        engine._config.deferred_maintenance_max_passes = 1
        second = engine.compress(first)
        debt2 = engine._lifecycle.get_by_conversation(engine._conversation_id)
        assert debt2 is not None and debt2.debt_kind == "raw_backlog"
        assert debt2.debt_size_estimate < debt1.debt_size_estimate
        assert debt2.last_maintenance_attempt_at is not None

        engine._config.deferred_maintenance_max_passes = 10
        third = engine.compress(second)
        debt3 = engine._lifecycle.get_by_conversation(engine._conversation_id)
        assert debt3 is not None
        assert debt3.debt_kind is None
        assert debt3.debt_size_estimate == 0
        assert third[0]["role"] == "system"

    def test_status_and_lcm_status_surface_debt_state(self, engine, monkeypatch):
        engine._config.dynamic_leaf_chunk_enabled = True
        engine._config.dynamic_leaf_chunk_max = 100
        engine._config.leaf_chunk_tokens = 100
        engine._config.fresh_tail_count = 2
        engine._config.deferred_maintenance_enabled = True
        engine.on_session_start("debt-session", platform="cli", context_length=200000)

        monkeypatch.setattr(lcm_engine, "summarize_with_escalation", lambda **kwargs: ("debt summary", 1))
        monkeypatch.setattr(engine, "_working_leaf_chunk_tokens", lambda raw_tokens: 100)
        monkeypatch.setattr(
            engine,
            "_assemble_context",
            lambda system_msg, tail_messages, assembly_cap_override=None, include_lcm_note=True: [system_msg, *tail_messages],
        )

        engine.compress(self._make_backlog_messages())
        status = engine.get_status()
        assert status["lifecycle"]["debt_kind"] == "raw_backlog"
        assert status["lifecycle"]["debt_size_estimate"] > 0

        tool_status = json.loads(engine.handle_tool_call("lcm_status", {}))
        assert tool_status["lifecycle"]["debt_kind"] == "raw_backlog"
        assert tool_status["config"]["deferred_maintenance_enabled"] is True


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

    def test_max_assembly_tokens_drops_oversized_newest_tail_message(self, tmp_path, monkeypatch):
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

        assert result[1:] == []

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

    def test_compress_forces_overflow_recovery_when_context_hits_assembly_cap(self, tmp_path, monkeypatch):
        import importlib

        config = LCMConfig(
            fresh_tail_count=2,
            leaf_chunk_tokens=100,
            database_path=str(tmp_path / "lcm_guardrail_forced.db"),
            max_assembly_tokens=90,
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
        monkeypatch.setattr(
            lcm_engine_module,
            "count_messages_tokens",
            lambda messages: sum(len(msg.get("content", "")) for msg in messages),
        )
        monkeypatch.setattr(lcm_engine_module, "count_tokens", lambda text: len(text))
        monkeypatch.setattr(
            lcm_engine_module,
            "summarize_with_escalation",
            lambda **kwargs: ("summary", 1),
        )

        messages = [
            {"role": "system", "content": "s" * 10},
            {"role": "user", "content": "a" * 20},
            {"role": "assistant", "content": "b" * 20},
            {"role": "user", "content": "c" * 20},
            {"role": "assistant", "content": "d" * 20},
        ]

        result = instance.compress(messages, current_tokens=90)

        assert len(result) < len(messages)
        assert result[-2:] == messages[-2:]
        assert lcm_engine_module.count_messages_tokens(result) < 90
        assert instance._dag.get_session_nodes("guardrail-session")

    def test_forced_overflow_tail_capping_updates_bookkeeping_without_middle_compaction(self, tmp_path, monkeypatch):
        import importlib

        config = LCMConfig(
            fresh_tail_count=10,
            database_path=str(tmp_path / "lcm_guardrail_tail_only.db"),
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
        monkeypatch.setattr(
            lcm_engine_module,
            "count_messages_tokens",
            lambda messages: sum(len(msg.get("content", "")) for msg in messages),
        )

        messages = [
            {"role": "system", "content": "s" * 10},
            {"role": "user", "content": "a" * 40},
            {"role": "assistant", "content": "b" * 40},
        ]

        result = instance.compress(messages, current_tokens=90)

        assert result == [messages[0], messages[-1]]
        assert instance.compression_count == 1
        assert instance._ingest_cursor == len(result)
        assert not instance.get_status()["overflow_recovery_failed"]

    def test_forced_overflow_recovery_reserves_provider_overhead(self, tmp_path, monkeypatch):
        import importlib

        config = LCMConfig(
            fresh_tail_count=10,
            database_path=str(tmp_path / "lcm_guardrail_overhead.db"),
            max_assembly_tokens=90,
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
        monkeypatch.setattr(
            lcm_engine_module,
            "count_messages_tokens",
            lambda messages: sum(len(msg.get("content", "")) for msg in messages),
        )

        messages = [
            {"role": "system", "content": "s" * 10},
            {"role": "user", "content": "a" * 30},
            {"role": "assistant", "content": "b" * 40},
        ]

        result = instance.compress(messages, current_tokens=100)

        assert result == [messages[0], messages[-1]]
        assert lcm_engine_module.count_messages_tokens(result) < 70

    def test_forced_overflow_recovery_does_not_duplicate_existing_summary_message(self, tmp_path, monkeypatch):
        import importlib
        from hermes_lcm.dag import SummaryNode

        config = LCMConfig(
            fresh_tail_count=10,
            database_path=str(tmp_path / "lcm_guardrail_summary_dup.db"),
            max_assembly_tokens=90,
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
        monkeypatch.setattr(
            lcm_engine_module,
            "count_messages_tokens",
            lambda messages: sum(len(msg.get("content", "")) for msg in messages),
        )

        node = SummaryNode(
            session_id="guardrail-session",
            depth=0,
            summary="sum",
            token_count=3,
            source_token_count=50,
            source_ids=[],
            source_type="messages",
            created_at=time.time(),
            expand_hint="x",
        )
        node_id = instance._dag.add_node(node)
        summary_blob = (
            f"[Recent Summary (d0, node {node_id})]\n"
            f"sum\n"
            f"[Expand for details: x]"
        )
        messages = [
            {"role": "system", "content": "s" * 10},
            {"role": "assistant", "content": summary_blob},
            {"role": "user", "content": "tail" * 2},
        ]

        result = instance.compress(messages, current_tokens=90)

        joined = "\n\n".join(msg.get("content", "") for msg in result)
        assert joined.count("[Expand for details:") == 1
        assert not instance.get_status()["overflow_recovery_failed"]

    def test_forced_overflow_recovery_flags_irreducible_single_tail_overflow(self, tmp_path, monkeypatch):
        import importlib

        config = LCMConfig(
            fresh_tail_count=10,
            database_path=str(tmp_path / "lcm_guardrail_irreducible.db"),
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
        monkeypatch.setattr(
            lcm_engine_module,
            "count_messages_tokens",
            lambda messages: sum(len(msg.get("content", "")) for msg in messages),
        )

        messages = [
            {"role": "system", "content": "s" * 10},
            {"role": "user", "content": "a" * 20},
            {"role": "assistant", "content": "b" * 80},
        ]

        result = instance.compress(messages, current_tokens=110)

        assert result == [messages[0], messages[-1]]
        assert instance.get_status()["overflow_recovery_failed"]

    def test_overflow_recovery_failure_flag_resets_after_successful_compression(self, tmp_path, monkeypatch):
        import importlib

        config = LCMConfig(
            fresh_tail_count=2,
            leaf_chunk_tokens=100,
            database_path=str(tmp_path / "lcm_guardrail_flag_reset.db"),
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
        monkeypatch.setattr(
            lcm_engine_module,
            "count_messages_tokens",
            lambda messages: sum(len(msg.get("content", "")) for msg in messages),
        )
        monkeypatch.setattr(lcm_engine_module, "count_tokens", lambda text: len(text))
        monkeypatch.setattr(
            lcm_engine_module,
            "summarize_with_escalation",
            lambda **kwargs: ("summary", 1),
        )

        failed_messages = [
            {"role": "system", "content": "s" * 10},
            {"role": "user", "content": "a" * 20},
            {"role": "assistant", "content": "b" * 80},
        ]
        instance.compress(failed_messages, current_tokens=110)
        assert instance.get_status()["overflow_recovery_failed"]

        success_messages = [
            {"role": "system", "content": "s" * 10},
            {"role": "user", "content": "a" * 20},
            {"role": "assistant", "content": "b" * 20},
            {"role": "user", "content": "c" * 20},
            {"role": "assistant", "content": "d" * 20},
        ]
        instance.compress(success_messages, current_tokens=90)

        assert not instance.get_status()["overflow_recovery_failed"]

    def test_compress_ignores_stale_last_prompt_tokens_for_overflow_recovery(self, tmp_path, monkeypatch):
        import importlib

        config = LCMConfig(
            fresh_tail_count=10,
            database_path=str(tmp_path / "lcm_guardrail_stale_prompt.db"),
            max_assembly_tokens=70,
        )
        instance = LCMEngine(config=config)
        instance._session_id = "guardrail-session"
        instance.compression_count = 1
        instance.last_prompt_tokens = 200

        lcm_engine_module = importlib.import_module("hermes_lcm.engine")
        monkeypatch.setattr(
            lcm_engine_module,
            "count_message_tokens",
            lambda msg: len(msg.get("content", "")),
        )
        monkeypatch.setattr(
            lcm_engine_module,
            "count_messages_tokens",
            lambda messages: sum(len(msg.get("content", "")) for msg in messages),
        )

        messages = [
            {"role": "system", "content": "s" * 10},
            {"role": "user", "content": "a" * 20},
            {"role": "assistant", "content": "b" * 20},
        ]

        result = instance.compress(messages)

        assert result == messages


class TestEngineTools:
    def test_handle_grep(self, engine):
        # Add some data
        engine._store.append("test-session", {"role": "user", "content": "deploy docker containers"})
        result = json.loads(engine.handle_tool_call("lcm_grep", {"query": "docker"}))
        assert "results" in result

    def test_handle_grep_reports_sort_mode(self, engine):
        engine._store.append(
            "test-session",
            {"role": "user", "content": "database migration plan database migration plan"},
        )
        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {"query": '"database migration plan"', "limit": 1, "sort": "relevance"},
            )
        )
        assert result["sort"] == "relevance"

    def test_handle_grep_source_filter_in_current_session_includes_only_matching_summaries(self, engine):
        engine._store.append("test-session", {"role": "user", "content": "docker logs from discord"}, source="discord")
        engine._store.append("s-telegram", {"role": "user", "content": "docker logs from telegram"}, source="telegram")
        engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="discord summary about docker logs",
                token_count=10,
                source_token_count=10,
                source_ids=[1],
                source_type="messages",
                created_at=time.time(),
            )
        )
        engine._dag.add_node(
            SummaryNode(
                session_id="s-telegram",
                depth=0,
                summary="telegram summary about docker logs",
                token_count=10,
                source_token_count=10,
                source_ids=[2],
                source_type="messages",
                created_at=time.time(),
            )
        )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {"query": "docker", "session_scope": "current", "source": "discord", "limit": 10},
            )
        )

        assert result["session_scope"] == "current"
        assert result["source"] == "discord"
        assert any(item["type"] == "message" for item in result["results"])
        assert any(item["type"] == "summary" for item in result["results"])
        assert all(item.get("session_id") == "test-session" for item in result["results"])
        assert all(item.get("source", "discord") == "discord" for item in result["results"] if item["type"] == "message")

    def test_handle_grep_source_filter_excludes_unrelated_summaries_in_mixed_source_session(self, engine):
        discord_store_id = engine._store.append(
            "test-session",
            {"role": "user", "content": "docker logs from discord"},
            source="discord",
        )
        telegram_store_id = engine._store.append(
            "test-session",
            {"role": "user", "content": "docker logs from telegram"},
            source="telegram",
        )
        engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="discord summary about docker logs",
                token_count=10,
                source_token_count=10,
                source_ids=[discord_store_id],
                source_type="messages",
                created_at=time.time(),
            )
        )
        engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="telegram summary about docker logs",
                token_count=10,
                source_token_count=10,
                source_ids=[telegram_store_id],
                source_type="messages",
                created_at=time.time(),
            )
        )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {"query": "docker", "session_scope": "current", "source": "discord", "limit": 10},
            )
        )

        assert any(item["type"] == "message" for item in result["results"])
        assert any(item["type"] == "summary" for item in result["results"])
        assert all(item.get("session_id") == "test-session" for item in result["results"])
        assert all(
            "telegram summary" not in item.get("snippet", "")
            for item in result["results"]
            if item["type"] == "summary"
        )
        assert any(
            "discord summary" in item.get("snippet", "")
            for item in result["results"]
            if item["type"] == "summary"
        )

    def test_handle_grep_unknown_source_filter_matches_unknown_messages_and_summaries(self, engine):
        unknown_store_id = engine._store.append(
            "test-session",
            {"role": "user", "content": "docker logs from unknown source"},
            source="unknown",
        )
        engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="unknown summary about docker logs",
                token_count=10,
                source_token_count=10,
                source_ids=[unknown_store_id],
                source_type="messages",
                created_at=time.time(),
            )
        )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {"query": "docker", "session_scope": "current", "source": "unknown", "limit": 10},
            )
        )

        assert result["source"] == "unknown"
        assert any(item["type"] == "message" for item in result["results"])
        assert any(item["type"] == "summary" for item in result["results"])
        assert all(item.get("session_id") == "test-session" for item in result["results"])
        assert all(item.get("source", "unknown") == "unknown" for item in result["results"] if item["type"] == "message")
        assert any(
            "unknown summary" in item.get("snippet", "")
            for item in result["results"]
            if item["type"] == "summary"
        )

    def test_handle_grep_prefers_conversational_hits_over_tool_output_noise(self, engine):
        engine._store.append(
            "test-session",
            {"role": "user", "content": "vendoring should stay generic host support only"},
        )
        engine._store.append(
            "test-session",
            {"role": "tool", "content": '{"vendoring":"vendoring vendoring vendoring","payload":"generic host support"}'},
        )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {"query": "vendoring", "limit": 2, "sort": "relevance"},
            )
        )

        assert result["results"][0]["role"] == "user"
        assert result["results"][1]["role"] == "tool"

    def test_handle_grep_relevance_prefers_user_over_newer_assistant_on_similar_match(self, engine):
        engine._store.append(
            "test-session",
            {"role": "user", "content": "external plugin host support should stay generic"},
        )
        engine._store.append(
            "test-session",
            {"role": "assistant", "content": "external plugin host support should stay generic"},
        )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {"query": "external plugin host support", "limit": 2, "sort": "relevance"},
            )
        )

        assert result["results"][0]["role"] == "user"
        assert result["results"][1]["role"] == "assistant"

    def test_handle_grep_relevance_does_not_let_weaker_user_hit_beat_stronger_assistant_hit(self, engine):
        engine._store.append(
            "test-session",
            {"role": "user", "content": "vendoring blah blah external blah host"},
        )
        engine._store.append(
            "test-session",
            {"role": "assistant", "content": "vendoring external host"},
        )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {"query": "vendoring external host", "limit": 2, "sort": "relevance"},
            )
        )

        assert result["results"][0]["type"] == "message"
        assert result["results"][0]["role"] == "assistant"
        assert result["results"][1]["type"] == "message"
        assert result["results"][1]["role"] == "user"

    def test_handle_grep_relevance_still_surfaces_preferred_user_hit_from_large_same_rank_pool(self, engine):
        engine._store.append(
            "test-session",
            {"role": "user", "content": "vendoring"},
        )
        for _ in range(150):
            engine._store.append(
                "test-session",
                {"role": "assistant", "content": "vendoring"},
            )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {"query": "vendoring", "limit": 5, "sort": "relevance"},
            )
        )

        assert result["results"][0]["type"] == "message"
        assert result["results"][0]["role"] == "user"

    def test_handle_grep_relevance_prefers_assistant_over_tool_on_similar_match(self, engine):
        engine._store.append(
            "test-session",
            {"role": "assistant", "content": "plugin-only support should stay external and generic"},
        )
        engine._store.append(
            "test-session",
            {"role": "tool", "content": "plugin-only support should stay external and generic"},
        )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {"query": "plugin-only", "limit": 2, "sort": "relevance"},
            )
        )

        assert result["results"][0]["role"] == "assistant"
        assert result["results"][1]["role"] == "tool"

    def test_handle_grep_relevance_prefers_direct_hit_over_repetition_spam_for_single_term_query(self, engine):
        engine._store.append(
            "test-session",
            {"role": "assistant", "content": "query audit notes: vendoring vendoring vendoring vendoring vendoring"},
        )
        engine._store.append(
            "test-session",
            {"role": "assistant", "content": "Keep vendoring out of hermes-agent."},
        )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {"query": "vendoring", "limit": 2, "sort": "relevance"},
            )
        )

        assert result["results"][0]["snippet"].startswith("Keep >>>vendoring<<< out")

    def test_handle_grep_relevance_prefers_direct_summary_hit_over_repetition_spam_summary(self, engine):
        engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="Summary notes: vendoring vendoring vendoring vendoring vendoring",
                token_count=10,
                source_token_count=20,
                source_ids=[1],
                source_type="messages",
                created_at=1_700_000_000,
                earliest_at=1_700_000_000,
                latest_at=1_700_000_000,
            )
        )
        engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="Keep vendoring out of hermes-agent.",
                token_count=10,
                source_token_count=20,
                source_ids=[2],
                source_type="messages",
                created_at=1_699_999_000,
                earliest_at=1_699_999_000,
                latest_at=1_699_999_000,
            )
        )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {"query": "vendoring", "limit": 2, "sort": "relevance"},
            )
        )

        assert result["results"][0]["type"] == "summary"
        assert result["results"][0]["snippet"].startswith("Keep vendoring out of hermes-agent")

    def test_handle_grep_relevance_still_surfaces_direct_summary_when_single_term_matches_many_spammy_candidates(self, engine):
        for idx in range(150):
            engine._dag.add_node(
                SummaryNode(
                    session_id="test-session",
                    depth=0,
                    summary=f"Summary spam {idx}: vendoring vendoring vendoring vendoring vendoring",
                    token_count=10,
                    source_token_count=20,
                    source_ids=[idx + 1],
                    source_type="messages",
                    created_at=1_700_000_000 + idx,
                    earliest_at=1_700_000_000 + idx,
                    latest_at=1_700_000_000 + idx,
                )
            )
        engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="Keep vendoring out of hermes-agent.",
                token_count=10,
                source_token_count=20,
                source_ids=[999],
                source_type="messages",
                created_at=1_699_999_000,
                earliest_at=1_699_999_000,
                latest_at=1_699_999_000,
            )
        )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {"query": "vendoring", "limit": 5, "sort": "relevance"},
            )
        )

        assert result["results"][0]["type"] == "summary"
        assert result["results"][0]["snippet"].startswith("Keep vendoring out of hermes-agent")

    def test_handle_grep_relevance_still_surfaces_direct_phrase_summary_when_phrase_matches_many_spammy_candidates(self, engine):
        for idx in range(150):
            engine._dag.add_node(
                SummaryNode(
                    session_id="test-session",
                    depth=0,
                    summary=f"Summary spam {idx}: vendoring external vendoring external vendoring external status",
                    token_count=10,
                    source_token_count=20,
                    source_ids=[3000 + idx],
                    source_type="messages",
                    created_at=1_700_000_000 + idx,
                    earliest_at=1_700_000_000 + idx,
                    latest_at=1_700_000_000 + idx,
                )
            )
        engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="Keep vendoring external support plugin-only.",
                token_count=10,
                source_token_count=20,
                source_ids=[9999],
                source_type="messages",
                created_at=1_699_999_000,
                earliest_at=1_699_999_000,
                latest_at=1_699_999_000,
            )
        )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {"query": '"vendoring external"', "limit": 5, "sort": "relevance"},
            )
        )

        assert any(
            item["type"] == "summary" and item["snippet"].startswith("Keep vendoring external support")
            for item in result["results"]
        )

    def test_handle_grep_relevance_prefers_direct_phrase_summary_over_repeated_phrase_with_varied_filler(self, engine):
        engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="vendoring external rollout checklist vendoring external support matrix vendoring external adapter notes",
                token_count=10,
                source_token_count=20,
                source_ids=[4100],
                source_type="messages",
                created_at=1_700_000_100,
                earliest_at=1_700_000_100,
                latest_at=1_700_000_100,
            )
        )
        engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="Keep vendoring external support plugin-only.",
                token_count=10,
                source_token_count=20,
                source_ids=[4101],
                source_type="messages",
                created_at=1_700_000_000,
                earliest_at=1_700_000_000,
                latest_at=1_700_000_000,
            )
        )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {"query": '"vendoring external"', "limit": 2, "sort": "relevance"},
            )
        )

        assert result["results"][0]["type"] == "summary"
        assert result["results"][0]["snippet"].startswith("Keep vendoring external support")

    def test_handle_grep_relevance_prefers_direct_phrase_summary_over_repeated_phrase_with_richer_filler(self, engine):
        engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="vendoring external rollout checklist vendoring external support matrix vendoring external adapter integration notes",
                token_count=10,
                source_token_count=20,
                source_ids=[4110],
                source_type="messages",
                created_at=1_700_000_100,
                earliest_at=1_700_000_100,
                latest_at=1_700_000_100,
            )
        )
        engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="Keep vendoring external support plugin-only.",
                token_count=10,
                source_token_count=20,
                source_ids=[4111],
                source_type="messages",
                created_at=1_700_000_000,
                earliest_at=1_700_000_000,
                latest_at=1_700_000_000,
            )
        )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {"query": '"vendoring external"', "limit": 2, "sort": "relevance"},
            )
        )

        assert result["results"][0]["type"] == "summary"
        assert result["results"][0]["snippet"].startswith("Keep vendoring external support")

    def test_handle_grep_relevance_unmatched_quote_still_finds_results(self, engine):
        engine._store.append(
            "test-session",
            {"role": "assistant", "content": "Keep vendoring out of hermes-agent."},
        )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {"query": '"vendoring', "limit": 5, "sort": "relevance"},
            )
        )

        assert result["total_results"] == 1
        assert result["results"][0]["type"] == "message"
        assert result["results"][0]["snippet"].startswith("Keep vendoring out")

    def test_handle_grep_recency_same_timestamp_pool_matches_store_ordering(self, engine):
        ids = engine._store.append_batch(
            "test-session",
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

        store_results = engine._store.search("alpha beta gamma", session_id="test-session", limit=5, sort="recency")
        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {"query": "alpha beta gamma", "limit": 5, "sort": "recency"},
            )
        )

        assert [item["type"] for item in result["results"]] == ["message"] * len(result["results"])
        assert [item["store_id"] for item in result["results"]] == [hit["store_id"] for hit in store_results]

    def test_handle_grep_hybrid_summary_only_matches_dag_order_for_future_timestamps(self, engine):
        now = time.time()
        future = now + (60 * 24 * 3600)
        future_node = engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="vendoring",
                token_count=10,
                source_token_count=20,
                source_ids=[8001],
                source_type="messages",
                created_at=future,
                earliest_at=future,
                latest_at=future,
            )
        )
        current_node = engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="vendoring",
                token_count=10,
                source_token_count=20,
                source_ids=[8002],
                source_type="messages",
                created_at=now,
                earliest_at=now,
                latest_at=now,
            )
        )

        dag_results = engine._dag.search("vendoring", session_id="test-session", limit=2, sort="hybrid")
        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {"query": "vendoring", "limit": 2, "sort": "hybrid"},
            )
        )

        assert [node.node_id for node in dag_results] == [future_node, current_node]
        assert [item["node_id"] for item in result["results"]] == [future_node, current_node]

    def test_handle_grep_hybrid_message_only_clamps_future_timestamps_consistently(self, engine):
        now = time.time()
        future = now + (60 * 24 * 3600)
        current_ids = [
            engine._store.append(
                "test-session",
                {"role": "assistant", "content": "vendoring external"},
            )
            for _ in range(20)
        ]
        future_id = engine._store.append(
            "test-session",
            {"role": "assistant", "content": "vendoring external"},
        )
        for current_id in current_ids:
            engine._store._conn.execute("UPDATE messages SET timestamp = ? WHERE store_id = ?", (now, current_id))
        engine._store._conn.execute("UPDATE messages SET timestamp = ? WHERE store_id = ?", (future, future_id))
        engine._store._conn.commit()

        store_results = engine._store.search("vendoring external", session_id="test-session", limit=1, sort="hybrid")
        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {"query": "vendoring external", "limit": 1, "sort": "hybrid"},
            )
        )

        assert [hit["store_id"] for hit in store_results] == [future_id]
        assert [item["store_id"] for item in result["results"]] == [future_id]

    def test_handle_grep_relevance_prefers_much_better_summary_over_vague_user_hit(self, engine):
        store_id = engine._store.append(
            "test-session",
            {"role": "user", "content": "vendoring? maybe?"},
        )
        engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=1,
                summary="Summary: keep hermes-lcm external and never vendor it into hermes-agent. generic host support only.",
                token_count=20,
                source_token_count=40,
                source_ids=[store_id],
                source_type="messages",
                created_at=1_700_000_000,
                earliest_at=1_700_000_000,
                latest_at=1_700_000_000,
            )
        )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {"query": "never vendor hermes-agent", "limit": 2, "sort": "relevance"},
            )
        )

        assert result["results"][0]["type"] == "summary"
        assert result["results"][0]["snippet"].startswith("Summary: keep hermes-lcm external")
        assert result["results"][1]["type"] == "message"
        assert result["results"][1]["role"] == "user"

    def test_handle_grep_hybrid_prefers_much_better_summary_over_vague_recent_user_hit(self, engine):
        store_id = engine._store.append(
            "test-session",
            {"role": "user", "content": "vendoring? maybe?"},
        )
        engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=1,
                summary="Summary: keep hermes-lcm external and never vendor it into hermes-agent. generic host support only.",
                token_count=20,
                source_token_count=40,
                source_ids=[store_id],
                source_type="messages",
                created_at=1_700_000_000,
                earliest_at=1_700_000_000,
                latest_at=1_700_000_000,
            )
        )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {"query": "never vendor hermes-agent", "limit": 2, "sort": "hybrid"},
            )
        )

        assert result["results"][0]["type"] == "summary"
        assert result["results"][0]["snippet"].startswith("Summary: keep hermes-lcm external")
        assert result["results"][1]["type"] == "message"
        assert result["results"][1]["role"] == "user"

    def test_handle_grep_hybrid_does_not_let_weak_summary_beat_stronger_message_hit(self, engine):
        engine._store.append(
            "test-session",
            {"role": "assistant", "content": "Keep vendoring out of hermes-agent."},
        )
        engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="vendoring vendoring vendoring vendoring vendoring",
                token_count=10,
                source_token_count=20,
                source_ids=[1],
                source_type="messages",
                created_at=1_700_000_000,
                earliest_at=1_700_000_000,
                latest_at=1_700_000_000,
            )
        )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {"query": "vendoring", "limit": 2, "sort": "hybrid"},
            )
        )

        assert result["results"][0]["type"] == "message"
        assert result["results"][0]["role"] == "assistant"
        assert result["results"][1]["type"] == "summary"

    def test_handle_grep_recency_preserves_message_ordering_for_same_timestamp_hits(self, engine):
        ids = engine._store.append_batch(
            "test-session",
            [
                {"role": "user", "content": "vendoring"},
                {"role": "assistant", "content": "vendoring vendoring vendoring vendoring vendoring"},
            ],
        )
        engine._store._conn.execute(
            "UPDATE messages SET timestamp = ? WHERE store_id IN (?, ?)",
            (1_700_000_000, ids[0], ids[1]),
        )
        engine._store._conn.commit()

        store_hits = engine._store.search("vendoring", session_id="test-session", limit=2, sort="recency")
        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {"query": "vendoring", "limit": 2, "sort": "recency"},
            )
        )

        assert [hit["role"] for hit in store_hits] == ["user", "assistant"]
        assert [item["role"] for item in result["results"]] == ["user", "assistant"]

    def test_handle_grep_recency_prefers_message_over_weaker_summary_at_same_timestamp(self, engine):
        store_id = engine._store.append(
            "test-session",
            {"role": "user", "content": "Keep vendoring external support clean."},
        )
        engine._store._conn.execute(
            "UPDATE messages SET timestamp = ? WHERE store_id = ?",
            (1_700_000_000, store_id),
        )
        engine._store._conn.commit()
        engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="vendoring vendoring vendoring",
                token_count=10,
                source_token_count=20,
                source_ids=[store_id],
                source_type="messages",
                created_at=1_700_000_000,
                earliest_at=1_700_000_000,
                latest_at=1_700_000_000,
            )
        )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {"query": "vendoring", "limit": 2, "sort": "recency"},
            )
        )

        assert result["results"][0]["type"] == "message"
        assert result["results"][0]["role"] == "user"
        assert result["results"][1]["type"] == "summary"

    def test_handle_describe_overview(self, engine):
        result = json.loads(engine.handle_tool_call("lcm_describe", {}))
        assert "session_id" in result
        assert "store_message_count" in result

    def test_handle_expand_includes_externalized_metadata_for_large_tool_result_sources(self, tmp_path):
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_externalized_expand.db"),
            large_output_externalization_enabled=True,
            large_output_externalization_threshold_chars=200,
        )
        engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
        engine._session_id = "test-session"

        content = "RESULT:\n" + ("abcdef" * 2000)
        store_id = engine._store.append(
            "test-session",
            {"role": "tool", "tool_call_id": "call_big", "content": content},
        )
        engine._serialize_messages([
            {"role": "tool", "tool_call_id": "call_big", "content": content}
        ])
        node_id = engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="Externalized tool-output summary",
                token_count=10,
                source_token_count=20,
                source_ids=[store_id],
                source_type="messages",
                created_at=0,
            )
        )

        result = json.loads(engine.handle_tool_call("lcm_expand", {"node_id": node_id}))

        assert result["source_type"] == "messages"
        assert result["expanded"][0]["store_id"] == store_id
        assert result["expanded"][0]["externalized"]["tool_call_id"] == "call_big"
        assert result["expanded"][0]["externalized"]["content_chars"] == len(content)
        assert result["expanded"][0]["externalized"]["ref"].endswith(".json")

    def test_handle_expand_does_not_attach_other_sessions_externalized_metadata(self, tmp_path):
        shared_home = tmp_path / "hermes"
        config_a = LCMConfig(
            database_path=str(tmp_path / "a.db"),
            large_output_externalization_enabled=True,
            large_output_externalization_threshold_chars=200,
        )
        engine_a = LCMEngine(config=config_a, hermes_home=str(shared_home))
        engine_a._session_id = "session-a"

        content = "RESULT:\n" + ("abcdef" * 2000)
        engine_a._serialize_messages([
            {"role": "tool", "tool_call_id": "call_shared", "content": content}
        ])

        config_b = LCMConfig(
            database_path=str(tmp_path / "b.db"),
            large_output_externalization_enabled=True,
            large_output_externalization_threshold_chars=200,
        )
        engine_b = LCMEngine(config=config_b, hermes_home=str(shared_home))
        engine_b._session_id = "session-b"
        store_id = engine_b._store.append(
            "session-b",
            {"role": "tool", "tool_call_id": "call_shared", "content": content},
        )
        node_id = engine_b._dag.add_node(
            SummaryNode(
                session_id="session-b",
                depth=0,
                summary="Other session summary",
                token_count=10,
                source_token_count=20,
                source_ids=[store_id],
                source_type="messages",
                created_at=0,
            )
        )

        result = json.loads(engine_b.handle_tool_call("lcm_expand", {"node_id": node_id}))

        assert "externalized" not in result["expanded"][0]

    def test_handle_expand_finds_externalized_metadata_for_sanitized_tool_output(self, tmp_path):
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_externalized_sanitized.db"),
            large_output_externalization_enabled=True,
            large_output_externalization_threshold_chars=200,
        )
        engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
        engine._session_id = "test-session"

        raw_content = (("Chart screenshot notes. " * 80) + "\n\n" + "data:image/png;base64," + ("A" * 5000))
        store_id = engine._store.append(
            "test-session",
            {"role": "tool", "tool_call_id": "call_media", "content": raw_content},
        )
        engine._serialize_messages([
            {"role": "tool", "tool_call_id": "call_media", "content": raw_content}
        ])
        node_id = engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="Sanitized externalized tool-output summary",
                token_count=10,
                source_token_count=20,
                source_ids=[store_id],
                source_type="messages",
                created_at=0,
            )
        )

        result = json.loads(engine.handle_tool_call("lcm_expand", {"node_id": node_id}))

        assert result["expanded"][0]["externalized"]["tool_call_id"] == "call_media"
        assert result["expanded"][0]["externalized"]["session_id"] == "test-session"

    def test_handle_describe_externalized_ref_returns_metadata(self, tmp_path):
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_externalized_describe.db"),
            large_output_externalization_enabled=True,
            large_output_externalization_threshold_chars=200,
        )
        engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
        engine._session_id = "test-session"

        content = "RESULT:\n" + ("abcdef" * 2000)
        engine._serialize_messages([
            {"role": "tool", "tool_call_id": "call_big", "content": content}
        ])
        ref = next((tmp_path / "hermes" / "lcm-large-outputs").glob("*.json")).name

        result = json.loads(engine.handle_tool_call("lcm_describe", {"externalized_ref": ref}))

        assert result["externalized_ref"] == ref
        assert result["kind"] == "tool_result"
        assert result["tool_call_id"] == "call_big"
        assert result["session_id"] == "test-session"
        assert result["content_chars"] == len(content)
        assert result["content_preview"].startswith("RESULT:")

    def test_handle_expand_externalized_ref_returns_payload_content(self, tmp_path):
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_externalized_payload.db"),
            large_output_externalization_enabled=True,
            large_output_externalization_threshold_chars=200,
        )
        engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
        engine._session_id = "test-session"

        content = "RESULT:\n" + ("abcdef" * 2000)
        engine._serialize_messages([
            {"role": "tool", "tool_call_id": "call_big", "content": content}
        ])
        ref = next((tmp_path / "hermes" / "lcm-large-outputs").glob("*.json")).name

        result = json.loads(engine.handle_tool_call("lcm_expand", {"externalized_ref": ref}))

        assert result["externalized_ref"] == ref
        assert result["source_type"] == "externalized_payload"
        assert result["content"] == content
        assert result["tool_call_id"] == "call_big"
        assert result["content_truncated"] is False

    def test_handle_expand_externalized_ref_respects_max_tokens(self, tmp_path):
        from hermes_lcm.tokens import count_tokens

        config = LCMConfig(
            database_path=str(tmp_path / "lcm_externalized_payload_budget.db"),
            large_output_externalization_enabled=True,
            large_output_externalization_threshold_chars=200,
        )
        engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
        engine._session_id = "test-session"

        content = "RESULT:\n" + ("abcdef" * 2000)
        engine._serialize_messages([
            {"role": "tool", "tool_call_id": "call_big", "content": content}
        ])
        ref = next((tmp_path / "hermes" / "lcm-large-outputs").glob("*.json")).name

        result = json.loads(engine.handle_tool_call("lcm_expand", {"externalized_ref": ref, "max_tokens": 10}))

        assert result["externalized_ref"] == ref
        assert result["source_type"] == "externalized_payload"
        assert result["content_truncated"] is True
        assert count_tokens(result["content"]) <= 10
        assert result["tool_call_id"] == "call_big"

    def test_compress_gc_rewrites_summarized_externalized_tool_results(self, tmp_path, monkeypatch):
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_gc.db"),
            fresh_tail_count=0,
            leaf_chunk_tokens=50,
            large_output_externalization_enabled=True,
            large_output_externalization_threshold_chars=200,
            large_output_transcript_gc_enabled=True,
        )
        engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
        engine._session_id = "test-session"
        engine.context_length = 200000
        engine.threshold_tokens = int(200000 * config.context_threshold)

        monkeypatch.setattr(
            lcm_engine,
            "summarize_with_escalation",
            lambda **kwargs: ("Summarized tool result.\nExpand for details about: tool result", 1),
        )

        content = "RESULT:\n" + ("abcdef" * 2000)
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "tool", "tool_call_id": "call_gc", "content": content},
        ]

        compressed = engine.compress(messages)

        assert compressed[0]["role"] == "system"
        assert compressed[1]["role"] == "assistant"
        assert "Recent Summary" in compressed[1]["content"]
        stored_tool = next(row for row in engine._store.get_range("test-session") if row["role"] == "tool")
        assert stored_tool["content"].startswith("[GC'd externalized tool output:")
        assert "ref=" in stored_tool["content"]
        assert content[:100] not in stored_tool["content"]
        assert stored_tool["token_estimate"] == count_message_tokens(
            {"role": "tool", "tool_call_id": "call_gc", "content": stored_tool["content"]}
        )
        assert stored_tool["token_estimate"] < count_message_tokens(
            {"role": "tool", "tool_call_id": "call_gc", "content": content}
        )
        payload_files = list((tmp_path / "hermes" / "lcm-large-outputs").glob("*.json"))
        assert len(payload_files) == 1
        assert json.loads(payload_files[0].read_text())["content"] == content

    def test_handle_expand_still_resolves_externalized_metadata_after_transcript_gc_rewrite(self, tmp_path, monkeypatch):
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_gc_expand.db"),
            fresh_tail_count=0,
            leaf_chunk_tokens=50,
            large_output_externalization_enabled=True,
            large_output_externalization_threshold_chars=200,
            large_output_transcript_gc_enabled=True,
        )
        engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
        engine._session_id = "test-session"
        engine.context_length = 200000
        engine.threshold_tokens = int(200000 * config.context_threshold)

        monkeypatch.setattr(
            lcm_engine,
            "summarize_with_escalation",
            lambda **kwargs: ("Summarized tool result.\nExpand for details about: tool result", 1),
        )

        content = "RESULT:\n" + ("abcdef" * 2000)
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "tool", "tool_call_id": "call_gc", "content": content},
        ]

        engine.compress(messages)
        node_id = engine._dag.get_session_nodes("test-session")[0].node_id
        result = json.loads(engine.handle_tool_call("lcm_expand", {"node_id": node_id}))

        assert result["expanded"][0]["content"].startswith("[GC'd externalized tool output:")
        assert result["expanded"][0]["externalized"]["tool_call_id"] == "call_gc"
        assert result["expanded"][0]["externalized"]["ref"].endswith(".json")

    def test_compress_gc_skips_pinned_tool_results(self, tmp_path, monkeypatch):
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_gc_pinned.db"),
            fresh_tail_count=0,
            leaf_chunk_tokens=50,
            large_output_externalization_enabled=True,
            large_output_externalization_threshold_chars=200,
            large_output_transcript_gc_enabled=True,
        )
        engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
        engine._session_id = "test-session"
        engine.context_length = 200000
        engine.threshold_tokens = int(200000 * config.context_threshold)

        monkeypatch.setattr(
            lcm_engine,
            "summarize_with_escalation",
            lambda **kwargs: ("Summarized tool result.\nExpand for details about: tool result", 1),
        )

        content = "RESULT:\n" + ("abcdef" * 2000)
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "tool", "tool_call_id": "call_gc", "content": content},
        ]
        engine._ingest_messages(messages)
        tool_store_id = next(row for row in engine._store.get_range("test-session") if row["role"] == "tool")["store_id"]
        engine._store.pin(tool_store_id)

        engine.compress(messages)

        assert engine._store.get(tool_store_id)["content"] == content

    def test_gc_helper_does_not_miss_tool_rows_when_chunk_contains_unmatched_synthetic_messages(self, tmp_path):
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_gc_helper.db"),
            large_output_externalization_enabled=True,
            large_output_externalization_threshold_chars=200,
            large_output_transcript_gc_enabled=True,
        )
        engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
        engine._session_id = "test-session"

        content = "RESULT:\n" + ("abcdef" * 2000)
        tool_store_id = engine._store.append(
            "test-session",
            {"role": "tool", "tool_call_id": "call_gc", "content": content},
        )
        engine._serialize_messages([
            {"role": "tool", "tool_call_id": "call_gc", "content": content}
        ])

        engine._maybe_gc_compacted_tool_results(
            [
                {"role": "assistant", "content": "[Recent Summary (d0, node 1)]"},
                {"role": "tool", "tool_call_id": "call_gc", "content": content},
            ],
            [tool_store_id],
        )

        assert engine._store.get(tool_store_id)["content"].startswith("[GC'd externalized tool output:")

    def test_handle_expand_does_not_inline_full_externalized_payload_for_gc_rows(self, tmp_path, monkeypatch):
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_gc_budget.db"),
            fresh_tail_count=0,
            leaf_chunk_tokens=50,
            large_output_externalization_enabled=True,
            large_output_externalization_threshold_chars=200,
            large_output_transcript_gc_enabled=True,
        )
        engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
        engine._session_id = "test-session"
        engine.context_length = 200000
        engine.threshold_tokens = int(200000 * config.context_threshold)

        monkeypatch.setattr(
            lcm_engine,
            "summarize_with_escalation",
            lambda **kwargs: ("Summarized tool result.\nExpand for details about: tool result", 1),
        )

        content = "RESULT:\n" + ("abcdef" * 2000)
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "tool", "tool_call_id": "call_gc", "content": content},
        ]
        engine.compress(messages)
        node_id = engine._dag.get_session_nodes("test-session")[0].node_id

        result = json.loads(engine.handle_tool_call("lcm_expand", {"node_id": node_id, "max_tokens": 1}))

        assert result["expanded"][0]["content"].startswith("[GC'd externalized tool output:")
        assert result["expanded"][0]["externalized"]["ref"].endswith(".json")
        assert "content" not in result["expanded"][0]["externalized"]

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

        monkeypatch.setattr(lcm_tools, "_synthesize_expansion_answer", fake_synthesize)

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

    def test_handle_expand_query_hyphenated_operator_query_falls_back_cleanly(self, engine, monkeypatch):
        engine._store.append(
            "test-session",
            {
                "role": "user",
                "content": "hermes-lcm plugin-only external context-engine generic host support no vendoring stays external",
            },
        )
        node_id = engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="hermes-lcm plugin-only external context-engine generic host support no vendoring stays external",
                token_count=10,
                source_token_count=20,
                source_ids=[1],
                source_type="messages",
                created_at=0,
            )
        )

        monkeypatch.setattr(
            lcm_tools, "_synthesize_expansion_answer",
            lambda **kwargs: "Recovered through normalized retrieval",
        )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_expand_query",
                {
                    "query": "8416 OR vendored OR vendoring OR plugin-only OR external context-engine OR generic host support OR hermes-lcm stays external OR no vendoring",
                    "prompt": "What were the agreements?",
                    "max_tokens": 500,
                },
            )
        )

        assert result["answer"] == "Recovered through normalized retrieval"
        assert result["node_ids"] == [node_id]
        assert result["matches"]

    def test_handle_expand_query_rejects_non_numeric_limits(self, engine):
        result = json.loads(
            engine.handle_tool_call(
                "lcm_expand_query",
                {"query": "docker", "prompt": "What was the plan?", "max_tokens": "invalid"},
            )
        )

        assert result["error"] == "max_tokens must be an integer"

        result = json.loads(
            engine.handle_tool_call(
                "lcm_expand_query",
                {"query": "docker", "prompt": "What was the plan?", "max_results": "invalid"},
            )
        )

        assert result["error"] == "max_results must be an integer"

    def test_handle_expand_query_rejects_non_numeric_node_ids(self, engine):
        result = json.loads(
            engine.handle_tool_call(
                "lcm_expand_query",
                {"node_ids": ["not-a-number"], "prompt": "What was the plan?"},
            )
        )

        assert result["error"] == "node_ids must contain only integers"

    def test_handle_expand_query_accepts_valid_integer_node_ids(self, engine, monkeypatch):
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

        def fake_synthesize(*, prompt, context_blocks, model, max_tokens, timeout):
            return "Expansion answer"

        monkeypatch.setattr(lcm_tools, "_synthesize_expansion_answer", fake_synthesize)

        result = json.loads(
            engine.handle_tool_call(
                "lcm_expand_query",
                {"node_ids": [node_id], "prompt": "What was the plan?"},
            )
        )

        assert "error" not in result, f"unexpected error: {result.get('error')}"
        assert result["answer"] == "Expansion answer"
        assert result["node_ids"] == [node_id]

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

    def test_handle_status_returns_session_overview(self, engine):
        engine._store.append("test-session", {"role": "user", "content": "hello world"})
        engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="greeting summary",
                token_count=10,
                source_token_count=50,
                source_ids=[1],
                source_type="messages",
                created_at=0,
            )
        )
        engine.compression_count = 3
        engine.context_length = 128000
        engine.threshold_tokens = 96000
        engine.last_prompt_tokens = 40000

        result = json.loads(engine.handle_tool_call("lcm_status", {}))

        assert result["session_id"] == "test-session"
        assert result["compression_count"] == 3
        assert result["context_length"] == 128000
        assert result["store"]["messages"] == 1
        assert result["dag"]["total_nodes"] == 1
        assert "d0" in result["dag"]["depths"]
        assert result["config"]["fresh_tail_count"] == engine._config.fresh_tail_count
        assert result["session_filters"]["ignored"] is False
        assert result["source_lineage"]["messages_total"] == 1
        assert result["source_lineage"]["normalized_unknown_messages"] == 1
        assert result["source_lineage"]["legacy_blank_source_messages"] == 0

    def test_handle_status_shows_compression_ratio(self, engine):
        engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="short",
                token_count=10,
                source_token_count=100,
                source_ids=[],
                source_type="messages",
                created_at=0,
            )
        )

        result = json.loads(engine.handle_tool_call("lcm_status", {}))
        assert result["dag"]["compression_ratio"] == "10.0:1"

    def test_handle_doctor_returns_healthy(self, engine):
        result = json.loads(engine.handle_tool_call("lcm_doctor", {}))

        assert result["overall"] == "healthy"
        check_names = [c["check"] for c in result["checks"]]
        assert "database_integrity" in check_names
        assert "fts_index_sync" in check_names
        assert "orphaned_dag_nodes" in check_names
        assert "config_validation" in check_names
        assert all(c["status"] == "pass" for c in result["checks"])

    def test_handle_doctor_treats_legacy_blank_source_rows_as_healthy(self, engine):
        engine._store._conn.execute(
            """INSERT INTO messages
               (session_id, source, role, content, tool_call_id, tool_calls, tool_name, timestamp, token_estimate, pinned)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("legacy-session", "", "user", "legacy blank source", None, None, None, 1.0, 5, 0),
        )
        engine._store._conn.commit()

        result = json.loads(engine.handle_tool_call("lcm_doctor", {}))

        assert result["overall"] == "healthy"
        lineage_check = next(c for c in result["checks"] if c["check"] == "source_lineage_hygiene")
        assert lineage_check["status"] == "pass"
        assert lineage_check["detail"]["legacy_blank_source_messages"] == 1

    def test_handle_doctor_warns_on_bad_config(self, tmp_path):
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_doctor.db"),
            fresh_tail_count=1,
            context_threshold=0.99,
            condensation_fanin=1,
        )
        engine = LCMEngine(config=config)
        engine._session_id = "test-session"

        result = json.loads(engine.handle_tool_call("lcm_doctor", {}))

        assert result["overall"] == "warnings"
        config_check = next(c for c in result["checks"] if c["check"] == "config_validation")
        assert config_check["status"] == "warn"
        assert len(config_check["detail"]) == 3  # three warnings

    def test_handle_doctor_detects_orphaned_nodes(self, engine):
        # Add a node referencing a store_id that doesn't exist
        engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="orphan",
                token_count=10,
                source_token_count=50,
                source_ids=[99999],
                source_type="messages",
                created_at=0,
            )
        )

        result = json.loads(engine.handle_tool_call("lcm_doctor", {}))

        orphan_check = next(c for c in result["checks"] if c["check"] == "orphaned_dag_nodes")
        assert orphan_check["status"] == "warn"

    def test_handle_doctor_fts_sync_is_session_scoped(self, engine):
        engine._store.append("test-session", {"role": "user", "content": "session A"})
        engine._store.append("other-session", {"role": "user", "content": "session B 1"})
        engine._store.append("other-session", {"role": "assistant", "content": "session B 2"})

        result = json.loads(engine.handle_tool_call("lcm_doctor", {}))

        fts_check = next(c for c in result["checks"] if c["check"] == "fts_index_sync")
        assert fts_check["status"] == "pass"
        assert fts_check["detail"] == "1 session FTS rows, 1 session messages"


class TestExtractionDuringCompress:
    """Integration test: extraction runs end-to-end through engine.compress()."""

    def test_compress_with_extraction_enabled_writes_daily_file(self, tmp_path, monkeypatch):
        from pathlib import Path
        import hermes_lcm.engine as engine_module
        import hermes_lcm.extraction as ext_module

        config = LCMConfig(
            database_path=str(tmp_path / "lcm_extract.db"),
            extraction_enabled=True,
            extraction_output_path=str(tmp_path / "extractions"),
            extraction_model="test-extract-model",
            fresh_tail_count=4,
            leaf_chunk_tokens=100,
        )
        eng = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
        eng._session_id = "extract-integration"
        eng.context_length = 200000
        eng.threshold_tokens = 500

        extraction_calls = []

        def mock_extraction_llm(prompt, model="", timeout=None):
            extraction_calls.append({"prompt": prompt, "model": model, "timeout": timeout})
            return "- Decided to use PostgreSQL\n- Will migrate by Friday"

        def mock_summary(**kwargs):
            return "Leaf summary.\nExpand for details about: test", 1

        monkeypatch.setattr(ext_module, "_call_extraction_llm", mock_extraction_llm)
        monkeypatch.setattr(engine_module, "summarize_with_escalation", mock_summary)

        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        for i in range(20):
            messages.append({"role": "user", "content": f"Q{i}: " + "x" * 200})
            messages.append({"role": "assistant", "content": f"A{i}: " + "y" * 200})

        result = eng.compress(messages)

        # Extraction was invoked with correct model
        assert len(extraction_calls) > 0
        assert extraction_calls[0]["model"] == "test-extract-model"

        # Extraction prompt contains serialized message roles
        assert "[USER]:" in extraction_calls[0]["prompt"]
        assert "[ASSISTANT]:" in extraction_calls[0]["prompt"]

        # Daily file created with extracted content
        files = list(Path(tmp_path / "extractions").glob("*.md"))
        assert len(files) >= 1
        content = files[0].read_text()
        assert "PostgreSQL" in content
        assert "extract-integration" in content

        # Compression still completed (extraction didn't block it)
        assert result[0]["role"] == "system"
        assert len(eng._dag.get_session_nodes("extract-integration")) > 0

    def test_compress_proceeds_when_extraction_fails(self, tmp_path, monkeypatch):
        import hermes_lcm.engine as engine_module
        import hermes_lcm.extraction as ext_module

        config = LCMConfig(
            database_path=str(tmp_path / "lcm_extract_fail.db"),
            extraction_enabled=True,
            extraction_output_path=str(tmp_path / "extractions"),
            fresh_tail_count=4,
            leaf_chunk_tokens=100,
        )
        eng = LCMEngine(config=config)
        eng._session_id = "extract-fail"
        eng.context_length = 200000
        eng.threshold_tokens = 500

        def failing_extraction_llm(prompt, model="", timeout=None):
            raise RuntimeError("LLM service down")

        def mock_summary(**kwargs):
            return "Leaf summary.\nExpand for details about: test", 1

        monkeypatch.setattr(ext_module, "_call_extraction_llm", failing_extraction_llm)
        monkeypatch.setattr(engine_module, "summarize_with_escalation", mock_summary)

        messages = [{"role": "system", "content": "You are helpful."}]
        for i in range(20):
            messages.append({"role": "user", "content": f"Q{i}: " + "x" * 200})
            messages.append({"role": "assistant", "content": f"A{i}: " + "y" * 200})

        # Should not raise — extraction failure is non-blocking
        result = eng.compress(messages)
        assert result[0]["role"] == "system"
        assert len(eng._dag.get_session_nodes("extract-fail")) > 0
