"""Regression tests: provider-visible reasoning must be counted and preserved.

Hermes replays assistant ``reasoning_content`` (and carries internal
``reasoning``) back to the provider when ``model.reasoning_echo`` is on,
so those tokens are real context-window cost.  The raw store must also
round-trip the reasoning sidecar unchanged, while keeping it OUT of the
FTS evidence index (hidden reasoning is not conversational evidence).
"""

from __future__ import annotations

from pathlib import Path

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.fresh_tail import resolve_fresh_tail_boundary
from hermes_lcm.sanitize import (
    _assistant_message_has_visible_content,
    _clean_active_assistant_message,
)
from hermes_lcm.store import MessageStore
from hermes_lcm.tokens import count_message_tokens


REASONING = "thinking " * 50
UNSEARCHABLE_MARKER = "zulu-foxtrot-secret-trace"


# ---------------------------------------------------------------------------
# A. Token accounting
# ---------------------------------------------------------------------------

def test_reasoning_content_counts_toward_message_tokens():
    plain = {"role": "assistant", "content": "42"}
    reasoned = {"role": "assistant", "content": "42", "reasoning_content": REASONING}

    assert count_message_tokens(reasoned) > count_message_tokens(plain)


def test_internal_reasoning_key_counts_toward_message_tokens():
    # Host-side history dicts carry the internal ``reasoning`` key; with
    # reasoning_echo on, those tokens become provider-visible on replay.
    plain = {"role": "assistant", "content": "42"}
    reasoned = {"role": "assistant", "content": "42", "reasoning": REASONING}

    assert count_message_tokens(reasoned) > count_message_tokens(plain)


def test_identical_reasoning_keys_are_not_double_counted():
    # Host rows often carry the same text under both ``reasoning`` and
    # ``reasoning_content``; counting must reflect one copy of the cost.
    single = {"role": "assistant", "content": "42", "reasoning_content": REASONING}
    doubled = {
        "role": "assistant",
        "content": "42",
        "reasoning_content": REASONING,
        "reasoning": REASONING,
    }

    assert count_message_tokens(doubled) == count_message_tokens(single)


# ---------------------------------------------------------------------------
# B. Fresh-tail boundary reacts to reasoning size
# ---------------------------------------------------------------------------

def test_fresh_tail_token_cap_reacts_to_reasoning_size():
    # Discriminating case: the middle assistant's visible content is tiny
    # but its reasoning is huge.  A cap that covers the blind cost of the
    # two newest messages must still drop the reasoning-heavy assistant
    # once reasoning is counted (reasoning-blind counting keeps it).
    messages = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "ok", "reasoning_content": REASONING},
        {"role": "user", "content": "new"},
    ]
    # 30 comfortably covers the blind cost (newest ~5 + middle ~5) of the
    # two newest messages, but cannot cover the middle's ~50+ reasoning.
    boundary = resolve_fresh_tail_boundary(
        messages,
        fresh_tail_count=3,
        fresh_tail_max_tokens=30,
    )

    assert boundary.token_limited is True
    assert boundary.start == 2
    assert boundary.count == 1


# ---------------------------------------------------------------------------
# C. Active-context sanitization preserves the reasoning sidecar
# ---------------------------------------------------------------------------

def test_clean_active_assistant_preserves_reasoning_content_sidecar():
    msg = {
        "role": "assistant",
        "content": "Visible answer",
        "reasoning_content": "Historical reasoning",
    }

    cleaned = _clean_active_assistant_message(msg)

    assert cleaned is not None
    assert cleaned["content"] == "Visible answer"
    assert cleaned.get("reasoning_content") == "Historical reasoning"


def test_clean_active_assistant_preserves_internal_reasoning_sidecar():
    msg = {
        "role": "assistant",
        "content": "Visible answer",
        "reasoning": "Internal reasoning",
    }

    cleaned = _clean_active_assistant_message(msg)

    assert cleaned is not None
    assert cleaned.get("reasoning") == "Internal reasoning"


def test_reasoning_only_assistant_is_still_dropped():
    # No visible content: hidden reasoning alone must not count as a
    # provider-visible assistant turn.
    msg = {"role": "assistant", "content": "", "reasoning_content": "only thinking"}

    assert _assistant_message_has_visible_content(msg) is False
    assert _clean_active_assistant_message(msg) is None


# ---------------------------------------------------------------------------
# D. Raw store round-trip
# ---------------------------------------------------------------------------

def test_store_round_trip_preserves_reasoning_content(tmp_path: Path):
    store = MessageStore(tmp_path / "lcm.db")
    try:
        store_id = store.append(
            "s1",
            {"role": "assistant", "content": "Visible", "reasoning_content": REASONING},
            token_estimate=count_message_tokens(
                {"role": "assistant", "content": "Visible", "reasoning_content": REASONING}
            ),
        )

        rows = store.get_session_messages("s1")

        assert len(rows) == 1
        assert rows[0]["store_id"] == store_id
        assert rows[0]["content"] == "Visible"
        assert rows[0].get("reasoning_content") == REASONING
    finally:
        store.close()


def test_store_round_trip_promotes_internal_reasoning_key(tmp_path: Path):
    # Host history carries ``reasoning``; the raw store must persist that
    # text so reconstructed messages can replay it as provider-visible
    # ``reasoning_content``.
    store = MessageStore(tmp_path / "lcm.db")
    try:
        store.append("s1", {"role": "assistant", "content": "Visible", "reasoning": REASONING})

        rows = store.get_session_messages("s1")

        assert rows[0].get("reasoning_content") == REASONING
    finally:
        store.close()


def test_to_openai_msg_preserves_reasoning_content(tmp_path: Path):
    store = MessageStore(tmp_path / "lcm.db")
    try:
        store.append(
            "s1",
            {"role": "assistant", "content": "Visible", "reasoning_content": REASONING},
        )
        stored = store.get_session_messages("s1")[0]

        api_msg = store.to_openai_msg(stored)

        assert api_msg["role"] == "assistant"
        assert api_msg["content"] == "Visible"
        assert api_msg.get("reasoning_content") == REASONING
    finally:
        store.close()


# ---------------------------------------------------------------------------
# E. Evidence semantics: reasoning must NOT become searchable evidence
# ---------------------------------------------------------------------------

def test_reasoning_content_is_not_fts_indexed(tmp_path: Path):
    store = MessageStore(tmp_path / "lcm.db")
    try:
        store.append(
            "s1",
            {
                "role": "assistant",
                "content": "a short visible answer",
                "reasoning_content": f"scratch work {UNSEARCHABLE_MARKER} end",
            },
        )

        hits = store.search(UNSEARCHABLE_MARKER, session_id="s1")

        assert hits == [] or all(
            UNSEARCHABLE_MARKER not in (row.get("content") or "") for row in hits
        )
    finally:
        store.close()


# ---------------------------------------------------------------------------
# F. Engine-level: fresh tail from the store keeps reasoning, and the
#    protected-tail token accounting sees it.
# ---------------------------------------------------------------------------

def test_engine_fresh_tail_keeps_reasoning_and_counts_it(tmp_path: Path):
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        fresh_tail_count=4,
    )
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
    engine.on_session_start(
        "reasoning-session",
        conversation_id="reasoning-conversation",
        context_length=200_000,
    )
    try:
        engine.ingest(
            [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1", "reasoning_content": REASONING},
                {"role": "user", "content": "q2"},
            ]
        )

        tail, boundary = engine._get_session_fresh_tail("reasoning-session")

        # The reasoning-heavy assistant must survive the store round-trip
        # inside the protected tail, and the tail's counted cost must
        # include its reasoning, not just its visible content.
        assert len(tail) >= 2
        kept = {m.get("content"): m for m in tail}
        assert kept.get("a1", {}).get("reasoning_content") == REASONING
        assert boundary.tokens >= count_message_tokens(
            {"role": "assistant", "content": "a1", "reasoning_content": REASONING}
        )
    finally:
        engine.shutdown()
