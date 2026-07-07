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
