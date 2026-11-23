"""Tests for count/token-bounded fresh-tail selection."""

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.fresh_tail import resolve_fresh_tail_boundary
from hermes_lcm.tokens import count_message_tokens


def _user(content):
    return {"role": "user", "content": content}


def _tool_group(*, payload_a="a", payload_b="b"):
    return [
        {
            "role": "assistant",
            "content": "calling tools",
            "tool_calls": [
                {"id": "call-a", "function": {"name": "a", "arguments": "{}"}},
                {"id": "call-b", "function": {"name": "b", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "call-a", "content": payload_a},
        {"role": "tool", "tool_call_id": "call-b", "content": payload_b},
    ]


def test_zero_token_cap_preserves_count_only_behavior():
    messages = [_user(str(index)) for index in range(6)]

    boundary = resolve_fresh_tail_boundary(
        messages,
        fresh_tail_count=3,
        fresh_tail_max_tokens=0,
    )

    assert boundary.start == 3
    assert boundary.count == 3
    assert boundary.token_limited is False


def test_token_cap_moves_boundary_toward_newest_message():
    messages = [_user("old " * 100), _user("middle " * 100), _user("new")]
    newest_tokens = count_message_tokens(messages[-1])

    boundary = resolve_fresh_tail_boundary(
        messages,
        fresh_tail_count=3,
        fresh_tail_max_tokens=newest_tokens,
    )

    assert boundary.start == 2
    assert boundary.count == 1
    assert boundary.token_limited is True


def test_newest_message_is_kept_even_when_it_exceeds_token_cap():
    messages = [_user("old"), _user("oversized " * 100)]

    boundary = resolve_fresh_tail_boundary(
        messages,
        fresh_tail_count=2,
        fresh_tail_max_tokens=1,
    )

    assert boundary.start == 1
    assert boundary.count == 1
    assert boundary.tokens > 1


def test_enabled_token_cap_keeps_newest_when_message_count_is_zero():
    messages = [_user("old"), _user("new")]

    boundary = resolve_fresh_tail_boundary(
        messages,
        fresh_tail_count=0,
        fresh_tail_max_tokens=100,
    )

    assert boundary.start == 1
    assert boundary.count == 1


def test_boundary_retreats_to_complete_assistant_tool_group():
    messages = [_user("old"), *_tool_group(payload_a="a " * 100, payload_b="b")]

    boundary = resolve_fresh_tail_boundary(
        messages,
        fresh_tail_count=1,
        fresh_tail_max_tokens=1,
    )

    assert boundary.start == 1
    assert boundary.count == 3
    assert boundary.tool_group_extended is True


def test_orphan_tool_result_does_not_create_a_phantom_group():
    messages = [_user("old"), {"role": "tool", "tool_call_id": "missing", "content": "result"}]

    boundary = resolve_fresh_tail_boundary(
        messages,
        fresh_tail_count=1,
        fresh_tail_max_tokens=1,
    )

    assert boundary.start == 1
    assert boundary.tool_group_extended is False


def test_stored_tail_expands_backward_until_tool_group_is_complete(tmp_path):
    config = LCMConfig(
        database_path=str(tmp_path / "fresh-tail.db"),
        fresh_tail_count=1,
    )
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
    engine.on_session_start(
        "fresh-tail-session",
        conversation_id="fresh-tail-conversation",
        context_length=200_000,
    )
    try:
        engine.ingest(_tool_group(payload_b="final result"))

        tail, boundary = engine._get_session_fresh_tail("fresh-tail-session")

        assert [message["role"] for message in tail] == ["assistant", "tool", "tool"]
        assert boundary.tool_group_extended is True
    finally:
        engine.shutdown()


def test_raw_backlog_uses_token_bounded_tail(tmp_path):
    config = LCMConfig(
        database_path=str(tmp_path / "backlog.db"),
        fresh_tail_count=10,
        fresh_tail_max_tokens=5,
    )
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
    messages = [
        {"role": "system", "content": "system"},
        _user("old " * 20),
        _user("middle " * 20),
        _user("new"),
    ]
    try:
        assert engine._raw_backlog_messages(messages) == messages[1:3]
    finally:
        engine.shutdown()


def test_rotate_uses_effective_token_bounded_tail(tmp_path):
    config = LCMConfig(
        database_path=str(tmp_path / "rotate.db"),
        fresh_tail_count=10,
        fresh_tail_max_tokens=5,
    )
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
    engine.on_session_start(
        "rotate-tail-session",
        conversation_id="rotate-tail-conversation",
        context_length=200_000,
    )
    try:
        engine.ingest([_user("old " * 20), _user("middle " * 20), _user("new")])

        preview = engine.rotate_active_session(apply=False)

        # With the ingest-advance fix (2026-08-29), the frontier advances WITH
        # ingest — the 3 ingested messages are all consumed/current, so the
        # rotate correctly no-ops (they're within the fresh tail; nothing
        # pre-tail to rotate). Under the OLD buggy frontier-lag behavior this
        # asserted noop=False (the frontier lag made rotate think there was
        # rotatable backlog) — that was the frozen-frontier bug, not behavior.
        assert preview["ok"] is True
        assert preview["noop"] is True
        assert preview["fresh_tail_count"] == 10
        assert preview["fresh_tail_max_tokens"] == 5
    finally:
        engine.shutdown()
