"""Focused tests for opt-in active-replay tool-result stubbing."""

import json
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

import hermes_lcm.engine as lcm_engine
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.externalize import extract_externalized_ref, load_externalized_payload
from hermes_lcm.message_content import normalize_content_value


@pytest.fixture
def make_engine(tmp_path):
    engines = []

    def build(**overrides):
        settings = dict(
            database_path=str(tmp_path / f"active-stub-{len(engines)}.db"),
            fresh_tail_count=2,
            large_output_externalization_enabled=True,
            large_output_externalization_threshold_chars=1_000_000,
            large_output_active_replay_stubbing_enabled=True,
            large_output_active_replay_stub_threshold_tokens=5,
        )
        settings.update(overrides)
        config = LCMConfig(**settings)
        engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
        engine_index = len(engines)
        engine.on_session_start(
            f"active-stub-test-{engine_index}",
            conversation_id=f"active-stub-conversation-{engine_index}",
            context_length=200_000,
        )
        engines.append(engine)
        return engine

    yield build

    for engine in engines:
        engine.shutdown()


def tool_pair(call_id, payload, tool_name="read_file"):
    return [
        {
            "role": "assistant",
            "content": "running tool",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": tool_name, "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "content": payload},
    ]


def assembled_tool(result, call_id):
    return next(
        message
        for message in result
        if message.get("role") == "tool" and message.get("tool_call_id") == call_id
    )


def externalized_raw_cleanup_messages():
    original = [{"role": "user", "content": "large raw payload"}]
    cleanup = [
        {
            "role": "user",
            "content": (
                "[Externalized payload: kind=raw_payload; role=user; "
                "chars=17; bytes=17; ref=raw-payload.json]"
            ),
        }
    ]
    return original, cleanup


def test_active_stubbing_is_default_off(tmp_path):
    config = LCMConfig(
        database_path=str(tmp_path / "default-off.db"),
        fresh_tail_count=2,
        large_output_externalization_enabled=True,
        large_output_externalization_threshold_chars=1,
    )
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
    engine._session_id = "default-off-test"
    payload = "old tool payload " * 100
    tail = tool_pair("old-call", payload) + tool_pair("fresh-call", "fresh result")
    try:
        result = engine._assemble_context({"role": "system", "content": "system"}, tail)
    finally:
        engine.shutdown()

    assert assembled_tool(result, "old-call")["content"] == payload


def test_stubs_only_token_heavy_evictable_results_and_preserves_pairing(make_engine):
    engine = make_engine()
    old_payload = "alpha beta gamma delta epsilon " * 30
    fresh_payload = "fresh result " * 30
    tail = tool_pair("old-call", old_payload) + tool_pair("fresh-call", fresh_payload)

    result = engine._assemble_context({"role": "system", "content": "system"}, tail)

    old_result = assembled_tool(result, "old-call")
    fresh_result = assembled_tool(result, "fresh-call")
    assert old_result["content"].startswith("[Externalized tool output:")
    assert "ref=" in old_result["content"]
    assert fresh_result["content"] == fresh_payload
    assistant_call_ids = {
        call["id"]
        for message in result
        for call in (message.get("tool_calls") or [])
    }
    assert {"old-call", "fresh-call"} <= assistant_call_ids
    assert {old_result["tool_call_id"], fresh_result["tool_call_id"]} <= assistant_call_ids
    first_ref = extract_externalized_ref(old_result["content"])

    second_result = engine._assemble_context(
        {"role": "system", "content": "system"},
        tail,
    )
    second_ref = extract_externalized_ref(assembled_tool(second_result, "old-call")["content"])
    assert second_ref == first_ref
    assert len(list((Path(engine._hermes_home) / "lcm-large-outputs").glob("*.json"))) == 1


def test_token_threshold_can_force_externalization_below_character_threshold(make_engine):
    engine = make_engine()
    payload = "one two three four five six seven eight nine ten"
    tail = tool_pair("token-call", payload) + tool_pair("fresh-call", "fresh result")

    result = engine._assemble_context({"role": "system", "content": "system"}, tail)

    stub = assembled_tool(result, "token-call")["content"]
    assert stub.startswith("[Externalized tool output:")
    ref = extract_externalized_ref(stub)
    assert ref
    recovered = load_externalized_payload(
        ref,
        config=engine._config,
        hermes_home=engine._hermes_home,
    )
    assert recovered is not None
    assert recovered["content"] == payload
    expanded = json.loads(
        engine.handle_tool_call(
            "lcm_expand",
            {"externalized_ref": ref, "max_tokens": 1_000},
        )
    )
    assert expanded["content"] == payload


def test_structured_text_tool_content_remains_array_shaped_and_recoverable(make_engine):
    engine = make_engine()
    payload = [
        {"type": "text", "text": "structured payload " * 40},
        {"type": "input_text", "text": "second textual block " * 20},
    ]
    tail = tool_pair("structured-call", payload) + tool_pair("fresh-call", "fresh result")

    result = engine._assemble_context({"role": "system", "content": "system"}, tail)

    stub_content = assembled_tool(result, "structured-call")["content"]
    assert isinstance(stub_content, list)
    assert stub_content == [{"type": "text", "text": stub_content[0]["text"]}]
    assert stub_content[0]["text"].startswith("[Externalized tool output:")
    ref = extract_externalized_ref(stub_content[0]["text"])
    recovered = load_externalized_payload(
        ref,
        config=engine._config,
        hermes_home=engine._hermes_home,
    )
    assert recovered is not None
    assert json.loads(recovered["content"]) == json.loads(normalize_content_value(payload))


@pytest.mark.parametrize(
    ("payload", "expected_shape"),
    [
        (
            [{"type": "input_text", "text": "input payload " * 40}],
            lambda placeholder: [{"type": "input_text", "text": placeholder}],
        ),
        (
            [{"type": "output_text", "text": {"value": "output payload " * 40}}],
            lambda placeholder: [
                {"type": "output_text", "text": {"value": placeholder}}
            ],
        ),
        (
            [{"type": "text", "content": {"content": "content payload " * 40}}],
            lambda placeholder: [
                {"type": "text", "content": {"content": placeholder}}
            ],
        ),
    ],
)
def test_structured_text_stub_preserves_compatible_block_type_and_key(
    make_engine,
    payload,
    expected_shape,
):
    engine = make_engine(fresh_tail_count=0)
    tail = tool_pair("shape-call", payload)

    result = engine._assemble_context({"role": "system", "content": "system"}, tail)

    stub_content = assembled_tool(result, "shape-call")["content"]
    placeholder = stub_content[0].get("text") or stub_content[0].get("content")
    if isinstance(placeholder, dict):
        placeholder = placeholder.get("value") or placeholder.get("content")
    assert isinstance(placeholder, str)
    assert placeholder.startswith("[Externalized tool output:")
    assert stub_content == expected_shape(placeholder)


def test_structured_media_tool_result_stays_provider_usable_inline(make_engine):
    engine = make_engine()
    payload = [
        {"type": "text", "text": "image follows " * 40},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64," + "A" * 1_000},
        },
    ]
    tail = tool_pair("media-call", payload) + tool_pair("fresh-call", "fresh result")

    result = engine._assemble_context({"role": "system", "content": "system"}, tail)

    assert assembled_tool(result, "media-call")["content"] == payload


def test_externalization_failure_leaves_original_payload_inline(make_engine, monkeypatch):
    engine = make_engine()
    payload = "must remain inline " * 40
    tail = tool_pair("fail-open-call", payload) + tool_pair("fresh-call", "fresh result")
    monkeypatch.setattr(lcm_engine, "maybe_externalize_tool_output", lambda *args, **kwargs: None)

    result = engine._assemble_context({"role": "system", "content": "system"}, tail)

    assert assembled_tool(result, "fail-open-call")["content"] == payload


@pytest.mark.parametrize("tool_name", ["lcm_describe", "lcm_expand"])
def test_recovery_tool_output_is_not_recursively_stubbed(make_engine, tool_name):
    engine = make_engine()
    payload = "recovered full payload " * 100
    tail = tool_pair("recovery-call", payload, tool_name=tool_name) + tool_pair(
        "fresh-call",
        "fresh result",
    )

    result = engine._assemble_context({"role": "system", "content": "system"}, tail)

    assert assembled_tool(result, "recovery-call")["content"] == payload


def test_overflow_recovery_uses_same_active_stub_policy(make_engine):
    engine = make_engine()
    payload = "overflow payload " * 100
    tail = tool_pair("overflow-call", payload) + tool_pair("fresh-call", "fresh result")

    result = engine._assemble_overflow_recovery_context(
        {"role": "system", "content": "system"},
        tail,
        assembly_cap_override=10_000,
    )

    assert assembled_tool(result, "overflow-call")["content"].startswith(
        "[Externalized tool output:"
    )


def test_stubbing_happens_before_assembly_budget_selection(make_engine):
    stubbed_engine = make_engine()
    baseline_engine = make_engine(large_output_active_replay_stubbing_enabled=False)
    payload = "budget pressure payload " * 2_000
    tail = tool_pair("budget-call", payload) + tool_pair("fresh-call", "fresh result")

    stubbed = stubbed_engine._assemble_context(
        {"role": "system", "content": "system"},
        tail,
        assembly_cap_override=250,
    )
    baseline = baseline_engine._assemble_context(
        {"role": "system", "content": "system"},
        tail,
        assembly_cap_override=250,
    )

    assert assembled_tool(stubbed, "budget-call")["content"].startswith(
        "[Externalized tool output:"
    )
    assert not any(
        message.get("tool_call_id") == "budget-call" for message in baseline
    )


def test_orphan_tool_result_is_not_externalized_as_a_phantom_reference(make_engine):
    engine = make_engine(fresh_tail_count=1)
    orphan_payload = "orphan payload " * 40
    messages = [
        {"role": "tool", "content": orphan_payload},
        {"role": "user", "content": "fresh request"},
    ]

    stubbed = engine._stub_large_tool_results_for_active_replay(messages)

    assert stubbed == messages


def test_live_interceptor_adopts_current_tool_stub_below_compaction_threshold(make_engine):
    engine = make_engine(fresh_tail_count=32, leaf_chunk_tokens=20_000)
    engine.threshold_tokens = 100_000
    payload = "live current payload " * 100
    messages = [
        {"role": "system", "content": "system"},
        *tool_pair("live-call", payload),
    ]

    assert engine.should_compress_preflight(messages) is True
    result = engine.compress(messages, current_tokens=1_000)

    assert assembled_tool(result, "live-call")["content"].startswith(
        "[Externalized tool output:"
    )
    assert engine._dag.get_session_node_count(engine._session_id) == 0
    assert engine.last_compression_status == "sanitized"


def test_live_stub_adoption_outranks_compression_boundary_cooldown(make_engine):
    engine = make_engine(fresh_tail_count=32, leaf_chunk_tokens=20_000)
    engine.threshold_tokens = 100_000
    engine._last_boundary_skip_time = time.time()
    payload = "durable live payload during cooldown " * 100
    messages = [
        {"role": "system", "content": "system"},
        *tool_pair("cooldown-live-call", payload),
    ]

    assert engine.should_compress_preflight(messages) is True
    result = engine.compress(messages, current_tokens=1_000)

    stub = assembled_tool(result, "cooldown-live-call")["content"]
    assert stub.startswith("[Externalized tool output:")
    ref = extract_externalized_ref(stub)
    recovered = load_externalized_payload(
        ref,
        config=engine._config,
        hermes_home=engine._hermes_home,
    )
    assert recovered is not None
    assert recovered["content"] == payload
    assert engine._dag.get_session_node_count(engine._session_id) == 0
    assert engine.last_compression_status == "sanitized"


def test_flag_off_plain_preflight_cooldown_blocks_threshold_and_overflow(make_engine):
    engine = make_engine(
        large_output_active_replay_stubbing_enabled=False,
        max_assembly_tokens=10,
    )
    engine.threshold_tokens = 1
    engine._last_boundary_skip_time = time.time()
    messages = [{"role": "user", "content": "plain branch pressure " * 100}]

    assert engine._should_force_overflow_recovery(messages=messages) is True
    assert engine.should_compress_preflight(messages) is False


def test_flag_off_replay_cleanup_preflight_outranks_boundary_cooldown(
    make_engine,
    monkeypatch,
):
    engine = make_engine(large_output_active_replay_stubbing_enabled=False)
    engine.threshold_tokens = 100_000
    engine._last_boundary_skip_time = time.time()
    messages, cleanup_messages = externalized_raw_cleanup_messages()
    monkeypatch.setattr(engine, "_ingest_messages", lambda _messages: cleanup_messages)

    assert engine.should_compress_preflight(messages) is True
    assert engine._preflight_cleanup_only is True


def test_flag_off_replay_cleanup_cooldown_publishes_without_summary_llm(
    make_engine,
    monkeypatch,
):
    engine = make_engine(large_output_active_replay_stubbing_enabled=False)
    engine.threshold_tokens = 100_000
    engine._last_boundary_skip_time = time.time()
    messages, cleanup_messages = externalized_raw_cleanup_messages()
    monkeypatch.setattr(engine, "_ingest_messages", lambda _messages: cleanup_messages)
    summary_spy = Mock(
        side_effect=AssertionError("cooldown-limited replay cleanup must not summarize")
    )
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summary_spy)

    assert engine.should_compress_preflight(messages) is True
    result = engine.compress(messages, current_tokens=1_000)

    assert result == cleanup_messages
    assert engine.last_compression_status == "sanitized"
    assert engine._dag.get_session_node_count(engine._session_id) == 0
    summary_spy.assert_not_called()


def test_flag_off_noncleanup_replay_diff_remains_blocked_during_cooldown(
    make_engine,
    monkeypatch,
):
    engine = make_engine(large_output_active_replay_stubbing_enabled=False)
    engine.threshold_tokens = 1
    engine._last_boundary_skip_time = time.time()
    messages = [{"role": "user", "content": "original visible payload"}]
    replay_messages = [{"role": "user", "content": "normalized visible payload"}]
    monkeypatch.setattr(engine, "_ingest_messages", lambda _messages: replay_messages)

    assert engine._replay_diff_requests_ingest_cleanup(messages, replay_messages) is False
    assert engine._should_force_overflow_recovery(messages=replay_messages) is False
    assert engine.should_compress_preflight(messages) is False


def test_live_stub_cooldown_adoption_skips_eligible_leaf_work(make_engine, monkeypatch):
    engine = make_engine(fresh_tail_count=2, leaf_chunk_tokens=1)
    engine.threshold_tokens = 100_000
    engine._last_boundary_skip_time = time.time()
    payload = "fresh durable payload with old eligible backlog " * 100
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "old request with eligible raw backlog"},
        {"role": "assistant", "content": "old answer with eligible raw backlog"},
        *tool_pair("cooldown-eligible-call", payload),
    ]

    def fail_if_summarized(**_kwargs):
        raise AssertionError("boundary cooldown cleanup must not summarize")

    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", fail_if_summarized)

    assert engine.should_compress_preflight(messages) is True
    result = engine.compress(messages, current_tokens=1_000)

    assert assembled_tool(result, "cooldown-eligible-call")["content"].startswith(
        "[Externalized tool output:"
    )
    assert any(message.get("content") == "old request with eligible raw backlog" for message in result)
    assert engine._dag.get_session_node_count(engine._session_id) == 0
    assert engine.last_compression_status == "sanitized"


def test_live_stub_subthreshold_adoption_skips_eligible_leaf_work(make_engine, monkeypatch):
    """Replay cleanup must not smuggle in an early summary compaction.

    Hermes calls ``should_compress_preflight`` only after its normal threshold
    path declined to run.  A large-output stub may still need immediate
    adoption, but that deterministic cleanup must not piggyback eligible leaf
    work and rewrite the conversation hundreds of thousands of tokens early.
    """
    engine = make_engine(fresh_tail_count=2, leaf_chunk_tokens=1)
    engine.threshold_tokens = 100_000
    payload = "fresh durable payload with old eligible backlog " * 100
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "old request with eligible raw backlog"},
        {"role": "assistant", "content": "old answer with eligible raw backlog"},
        *tool_pair("subthreshold-eligible-call", payload),
    ]

    def fail_if_summarized(**_kwargs):
        raise AssertionError("subthreshold replay cleanup must not summarize")

    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", fail_if_summarized)

    assert engine.should_compress_preflight(messages) is True
    result = engine.compress(messages, current_tokens=1_000)

    assert assembled_tool(result, "subthreshold-eligible-call")["content"].startswith(
        "[Externalized tool output:"
    )
    assert any(message.get("content") == "old request with eligible raw backlog" for message in result)
    assert engine._dag.get_session_node_count(engine._session_id) == 0
    assert engine.last_compression_status == "sanitized"


def test_live_interceptor_converges_when_no_leaf_is_eligible(make_engine):
    engine = make_engine(fresh_tail_count=32, leaf_chunk_tokens=20_000)
    engine.threshold_tokens = 1
    payload = "no eligible leaf payload " * 100
    messages = [
        {"role": "system", "content": "system"},
        *tool_pair("no-leaf-call", payload),
    ]

    assert engine.should_compress_preflight(messages) is True
    result = engine.compress(messages, current_tokens=10_000)

    assert assembled_tool(result, "no-leaf-call")["content"].startswith(
        "[Externalized tool output:"
    )
    assert engine._dag.get_session_node_count(engine._session_id) == 0


def test_live_interceptor_survives_fresh_tail_overflow_recovery(make_engine):
    engine = make_engine(
        fresh_tail_count=32,
        max_assembly_tokens=200,
        leaf_chunk_tokens=20_000,
    )
    payload = "fresh overflow payload " * 2_000
    messages = [
        {"role": "system", "content": "system"},
        *tool_pair("fresh-overflow-call", payload),
    ]

    assert engine.should_compress_preflight(messages) is True
    result = engine.compress(messages, current_tokens=None)

    assert assembled_tool(result, "fresh-overflow-call")["content"].startswith(
        "[Externalized tool output:"
    )
    assert engine._dag.get_session_node_count(engine._session_id) == 0
    assert engine.last_compression_status in {"sanitized", "overflow_recovery"}


def test_live_interceptor_is_preserved_after_normal_leaf_compaction(make_engine, monkeypatch):
    engine = make_engine(fresh_tail_count=2, leaf_chunk_tokens=1)
    engine.threshold_tokens = 1
    payload = "current result after compaction " * 100
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "old request"},
        {"role": "assistant", "content": "old answer"},
        *tool_pair("normal-call", payload),
    ]
    monkeypatch.setattr(
        lcm_engine,
        "summarize_with_escalation",
        lambda **_kwargs: ("old work summary\nExpand for details about: old work", 1),
    )

    assert engine.should_compress_preflight(messages) is True
    result = engine.compress(messages, current_tokens=20_000)

    assert assembled_tool(result, "normal-call")["content"].startswith(
        "[Externalized tool output:"
    )
    assert engine._dag.get_session_node_count(engine._session_id) == 1
    assert engine.last_compression_status == "compacted"


def test_live_interceptor_keeps_media_and_recovery_results_inline(make_engine):
    engine = make_engine(fresh_tail_count=32, leaf_chunk_tokens=20_000)
    engine.threshold_tokens = 100_000
    media_payload = [
        {"type": "text", "text": "image follows " * 40},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 1_000}},
    ]
    messages = [
        {"role": "system", "content": "system"},
        *tool_pair("media-live-call", media_payload),
        *tool_pair("recovery-live-call", "recovered payload " * 100, tool_name="lcm_expand"),
    ]

    assert engine.should_compress_preflight(messages) is False
    cached = engine._cached_active_replay_messages(messages)

    assert cached is not None
    assert assembled_tool(cached, "media-live-call")["content"] == media_payload
    assert assembled_tool(cached, "recovery-live-call")["content"] == "recovered payload " * 100
