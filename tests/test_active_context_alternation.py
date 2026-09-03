"""Active-context assembly must not emit consecutive assistant rows.

Active-context assembly strips ``tool_calls`` off assistant turns and drops the
``tool`` results (to shed token-heavy scaffolding), then persists that stripped
set. On the next turn those bare rows reload and run back through
``_sanitize_active_context_messages`` — arriving as consecutive bare assistant
rows (no tool_calls). Each carries no tool_calls, so ``_sanitize_tool_pairs``
leaves the adjacency untouched; the adjacent-assistant merge is what closes it.

A persisted run of consecutive assistant rows is a strict role-alternation
violation that every downstream load has to repair, and it inflates the
persisted message count above what the model actually replays.

The raw store and DAG stay lossless independent of this — granular rows remain
recoverable; this only sanitizes the active replay context.
"""

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


@pytest.fixture
def engine(tmp_path):
    config = LCMConfig(database_path=str(tmp_path / "alternation.db"))
    instance = LCMEngine(config=config, hermes_home=str(tmp_path))
    instance._session_id = "test-session"
    try:
        yield instance
    finally:
        instance.shutdown()


def _consecutive_assistant_runs(messages):
    return sum(
        1
        for i in range(1, len(messages))
        if messages[i].get("role") == "assistant"
        and messages[i - 1].get("role") == "assistant"
    )


def _bare_assistant_run(n_steps):
    """The shape re-ingested every turn: already-stripped bare assistant rows."""
    msgs = [
        {"role": "system", "content": "You are testing LCM."},
        {"role": "user", "content": "Do the multi-step task."},
    ]
    for i in range(n_steps):
        msgs.append(
            {
                "role": "assistant",
                "content": f"Step {i}: let me run the next probe.",
                "finish_reason": None,
            }
        )
    return msgs


def test_sanitize_active_context_has_no_consecutive_assistants(engine):
    messages = _bare_assistant_run(6)
    sanitized = engine._sanitize_active_context_messages(
        messages, insert_missing_tool_stubs=False
    )
    assert _consecutive_assistant_runs(sanitized) == 0, (
        "active context emitted consecutive assistant rows: "
        f"{[m.get('role') for m in sanitized]}"
    )


def test_merged_assistant_preserves_all_narration_text(engine):
    messages = _bare_assistant_run(4)
    sanitized = engine._sanitize_active_context_messages(
        messages, insert_missing_tool_stubs=False
    )
    merged_text = "\n".join(
        m["content"]
        for m in sanitized
        if m.get("role") == "assistant" and isinstance(m.get("content"), str)
    )
    for i in range(4):
        assert f"Step {i}:" in merged_text, f"lost narration for step {i}"


def test_codex_interim_turns_are_not_merged(engine):
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "interim 1",
            "codex_reasoning_items": [{"id": "rs_1"}],
            "finish_reason": "incomplete",
        },
        {
            "role": "assistant",
            "content": "interim 2",
            "codex_reasoning_items": [{"id": "rs_2"}],
            "finish_reason": "incomplete",
        },
    ]
    sanitized = engine._sanitize_active_context_messages(
        messages, insert_missing_tool_stubs=False
    )
    interim = [
        m
        for m in sanitized
        if m.get("role") == "assistant" and m.get("codex_reasoning_items")
    ]
    assert len(interim) == 2, "codex interim turns must NOT be merged"
