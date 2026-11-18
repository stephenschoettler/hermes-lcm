"""Pure message-list analysis helpers for the LCM engine.

Isolated from ``engine.py`` (WS5 seam): extracting and pairing assistant/tool
tool-call ids across a message list, and detecting synthetic assistant "noise"
turns (acks/heartbeats), are stateless inspection helpers with no engine state.
``engine.py`` imports the ones it calls; the synthetic-noise vocabulary stays
internal to this module.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

_SYNTHETIC_ASSISTANT_NOISE = {
    "ack",
    "acknowledged",
    "heartbeat",
    "heartbeat ack",
    "keepalive",
    "keep alive",
    "pong",
}


def _tool_call_id(tool_call: Any) -> str:
    if not isinstance(tool_call, dict):
        return ""
    value = tool_call.get("id") or tool_call.get("tool_call_id")
    return str(value).strip() if value else ""


def _assistant_tool_call_ids(messages: List[Dict[str, Any]]) -> set[str]:
    call_ids: set[str] = set()
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for tool_call in msg.get("tool_calls") or []:
            call_id = _tool_call_id(tool_call)
            if call_id:
                call_ids.add(call_id)
    return call_ids


def _matched_tool_call_ids(messages: List[Dict[str, Any]]) -> set[str]:
    assistant_call_ids = _assistant_tool_call_ids(messages)
    tool_result_ids: set[str] = set()
    for msg in messages:
        if msg.get("role") == "tool":
            tool_call_id = str(msg.get("tool_call_id") or "").strip()
            if tool_call_id:
                tool_result_ids.add(tool_call_id)
    return assistant_call_ids & tool_result_ids


def _is_synthetic_assistant_noise(content: str) -> bool:
    normalized = re.sub(r"\s+", " ", (content or "").strip()).lower()
    if not normalized:
        return True
    normalized = normalized.strip("`*_ ")
    bracketless = normalized.strip("[](){} ")
    return normalized in _SYNTHETIC_ASSISTANT_NOISE or bracketless in _SYNTHETIC_ASSISTANT_NOISE


def _is_codex_interim(msg: Dict[str, Any]) -> bool:
    """A Codex Responses interim assistant turn.

    These legitimately keep multiple consecutive incomplete assistant turns in
    history, each carrying its own encrypted continuation state
    (``codex_reasoning_items`` / ``codex_message_items``) that must replay
    verbatim. Merging them corrupts the Responses replay chain, so the
    adjacent-assistant merge exempts them — the same exemption the host
    gateway's ``repair_message_sequence`` applies.
    """
    return bool(
        msg.get("codex_reasoning_items")
        or msg.get("codex_message_items")
        or msg.get("finish_reason") == "incomplete"
    )


def _merge_adjacent_assistant_messages(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge consecutive assistant messages into one (alternation repair).

    Active-context assembly drops ``tool`` rows and strips ``tool_calls`` off
    assistant turns to shed token-heavy scaffolding. That leaves the narration
    rows that used to be separated by tool results sitting adjacent — a strict
    role-alternation violation (``assistant`` followed by ``assistant``) that
    every downstream load then has to repair, and that inflates the persisted
    message count above the count the model actually replays. Collapse them at
    the single active-context funnel so the emitted context is
    alternation-clean by construction.

    Union ``tool_calls`` (order-preserving), concatenate plain-text content,
    carry the first non-empty ``reasoning_content``, and exempt Codex Responses
    interim turns. Operates only on the active replay list — the raw store and
    DAG are untouched, so recovery granularity is preserved.
    """
    collapsed: List[Dict[str, Any]] = []
    for msg in messages:
        if (
            collapsed
            and isinstance(msg, dict)
            and msg.get("role") == "assistant"
            and isinstance(collapsed[-1], dict)
            and collapsed[-1].get("role") == "assistant"
            and not _is_codex_interim(msg)
            and not _is_codex_interim(collapsed[-1])
            # Only merge plain-string narration turns. Multimodal / structured
            # (list or dict) content stays its own turn — collapsing it risks
            # mangling attachment/thinking blocks, and the alternation problem
            # this repair exists for is the bare-narration case, which is
            # always plain text.
            and isinstance(collapsed[-1].get("content"), str)
            and isinstance(msg.get("content"), str)
        ):
            prev = dict(collapsed[-1])
            prev_calls = list(prev.get("tool_calls") or [])
            new_calls = list(msg.get("tool_calls") or [])
            if new_calls:
                prev["tool_calls"] = prev_calls + new_calls
            elif prev_calls:
                prev["tool_calls"] = prev_calls
            prev["content"] = "\n".join(
                part
                for part in (prev["content"].strip(), msg["content"].strip())
                if part
            )
            if not prev.get("reasoning_content") and msg.get("reasoning_content"):
                prev["reasoning_content"] = msg["reasoning_content"]
            collapsed[-1] = prev
            continue
        collapsed.append(msg)
    return collapsed
