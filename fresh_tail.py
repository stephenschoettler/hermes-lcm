"""Shared fresh-tail boundary calculation.

The protected tail is primarily message-count bounded.  An optional token cap
can move the boundary toward the newest message, while a tool-call integrity
check may move it back to the assistant that opened a tool-result group.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

from .escalation import _truncate_text_to_tokens
from .message_analysis import _tool_call_id
from .message_content import text_content_for_pattern_matching
from .tokens import count_message_tokens, count_messages_tokens


_ACTIVE_TRUNCATION_MARKER = (
    "\n\n[...active-context truncation; full content retained in LCM storage...]"
)


@dataclass(frozen=True)
class FreshTailBoundary:
    """Resolved protected-tail metadata for one ordered message sequence."""

    start: int
    count: int
    tokens: int
    count_limit: int
    token_limit: int
    token_limited: bool
    tool_group_extended: bool


def _assistant_group_start(messages: Sequence[Dict[str, Any]], start: int) -> int:
    """Retreat a tool-result boundary to its nearest opening assistant."""
    if start >= len(messages) or messages[start].get("role") != "tool":
        return start
    result_id = str(messages[start].get("tool_call_id") or "").strip()
    if not result_id:
        return start

    for index in range(start - 1, -1, -1):
        message = messages[index]
        role = str(message.get("role") or "")
        if role in {"user", "system"}:
            break
        if role != "assistant":
            continue
        call_ids = {
            _tool_call_id(tool_call)
            for tool_call in (message.get("tool_calls") or [])
        }
        return index if result_id in call_ids else start
    return start


def resolve_fresh_tail_boundary(
    messages: Sequence[Dict[str, Any]],
    *,
    fresh_tail_count: int,
    fresh_tail_max_tokens: int = 0,
) -> FreshTailBoundary:
    """Resolve the protected suffix without splitting assistant/tool groups.

    ``fresh_tail_max_tokens`` is disabled at zero.  When enabled, the newest
    message is always retained even when it alone exceeds the cap.  Tool-call
    integrity takes precedence over both limits, so the returned tail can
    exceed a configured bound when necessary to retain the opening assistant.
    """
    message_count = len(messages)
    count_limit = max(0, int(fresh_tail_count or 0))
    token_limit = max(0, int(fresh_tail_max_tokens or 0))
    if message_count == 0:
        return FreshTailBoundary(0, 0, 0, count_limit, token_limit, False, False)

    effective_count_limit = max(1, count_limit) if token_limit > 0 else count_limit
    count_start = max(0, message_count - effective_count_limit)
    start = count_start
    token_limited = False

    if token_limit > 0:
        used = 0
        token_start = message_count - 1
        for index in range(message_count - 1, count_start - 1, -1):
            message_tokens = count_message_tokens(messages[index])
            if index != message_count - 1 and used + message_tokens > token_limit:
                token_limited = True
                break
            token_start = index
            used += message_tokens
        start = token_start

    group_start = _assistant_group_start(messages, start)
    tool_group_extended = group_start < start
    start = group_start
    tail = list(messages[start:])
    return FreshTailBoundary(
        start=start,
        count=len(tail),
        tokens=count_messages_tokens(tail),
        count_limit=count_limit,
        token_limit=token_limit,
        token_limited=token_limited,
        tool_group_extended=tool_group_extended,
    )


def resolve_emergency_fresh_tail_boundary(
    messages: Sequence[Dict[str, Any]],
) -> FreshTailBoundary:
    """Protect only the newest indivisible conversation unit under hard pressure.

    Normal compaction protects a generously sized recent suffix. Once the
    provider's hard context limit is reached that policy must yield or a
    conversation shorter than ``fresh_tail_count`` can never converge. The
    newest message remains protected, and a trailing tool result retreats to
    the assistant that opened its tool-call group so active replay stays valid.
    """
    if not messages:
        return FreshTailBoundary(0, 0, 0, 0, 0, False, False)

    start = len(messages) - 1
    group_start = _assistant_group_start(messages, start)
    tool_group_extended = group_start < start
    start = group_start
    tail = list(messages[start:])
    return FreshTailBoundary(
        start=start,
        count=len(tail),
        tokens=count_messages_tokens(tail),
        count_limit=len(tail),
        token_limit=0,
        token_limited=True,
        tool_group_extended=tool_group_extended,
    )


def fit_active_context_to_token_cap(
    messages: Sequence[Dict[str, Any]],
    token_cap: int,
) -> list[Dict[str, Any]]:
    """Return a best-effort provider replay strictly below ``token_cap``.

    Raw messages are already durable before this active-view operation runs.
    Prefer the newest complete assistant/tool group (or newest single message),
    retain a leading system message only when it fits, and as a final resort
    deterministically shorten provider-visible content. Tool metadata and IDs
    are never rewritten.
    """
    cap = max(0, int(token_cap or 0))
    copied = [dict(message) for message in messages]
    if not copied or cap <= 0 or count_messages_tokens(copied) < cap:
        return copied

    boundary = resolve_emergency_fresh_tail_boundary(copied)
    newest_group = copied[boundary.start:]
    leading_system = copied[0] if copied[0].get("role") == "system" else None
    candidate = list(newest_group)
    if leading_system is not None and leading_system not in candidate:
        with_system = [leading_system, *candidate]
        if count_messages_tokens(with_system) < cap:
            candidate = with_system

    if count_messages_tokens(candidate) < cap:
        return candidate

    # Preserve envelopes and tool-call identities, assigning the remaining
    # content budget newest-first. ``cap - 1`` makes the hard-limit comparison
    # unambiguous and leaves one token of safety for approximate counters.
    target = max(0, cap - 1)
    fitted = []
    original_texts: list[str] = []
    for message in candidate:
        clone = dict(message)
        original_texts.append(
            text_content_for_pattern_matching(message.get("content")) or ""
        )
        clone["content"] = ""
        fitted.append(clone)

    for index in range(len(fitted) - 1, -1, -1):
        original = original_texts[index]
        if not original:
            continue
        used = count_messages_tokens(fitted)
        available = target - used
        if available <= 0:
            continue
        if count_message_tokens({**fitted[index], "content": original}) - count_message_tokens(
            fitted[index]
        ) <= available:
            fitted[index]["content"] = original
            continue

        marker_tokens = count_message_tokens(
            {**fitted[index], "content": _ACTIVE_TRUNCATION_MARKER}
        ) - count_message_tokens(fitted[index])
        body_budget = max(0, available - marker_tokens)
        body = _truncate_text_to_tokens(original, body_budget)
        fitted[index]["content"] = body + _ACTIVE_TRUNCATION_MARKER

        # Token estimates can be non-additive. Tighten the body until the final
        # message list is definitely under the cap.
        while body and count_messages_tokens(fitted) > target:
            body_tokens = count_message_tokens({**fitted[index], "content": body})
            body = _truncate_text_to_tokens(body, max(0, body_tokens - 2))
            fitted[index]["content"] = body + _ACTIVE_TRUNCATION_MARKER

    # Extremely large tool metadata can itself exceed the limit. Preserve the
    # newest message rather than corrupting its tool schema; callers surface
    # this exceptional best-effort failure through existing overflow telemetry.
    return fitted
