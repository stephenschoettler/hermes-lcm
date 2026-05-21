"""Metric helpers for deterministic LCM benchmark replays."""

from __future__ import annotations

from typing import Any, Iterable

from .types import Canary


def normalize_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                value = item.get("text") or item.get("content") or ""
                parts.append(str(value))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def messages_text(messages: Iterable[dict[str, Any]]) -> str:
    return "\n".join(normalize_message_content(message.get("content")) for message in messages)


def canary_present(text: str, canary: Canary) -> bool:
    return canary.id in text and canary.value in text


def count_canary_hits(text: str, canaries: Iterable[Canary]) -> int:
    return sum(1 for canary in canaries if canary_present(text, canary))


def count_active_canaries(messages: Iterable[dict[str, Any]], canaries: Iterable[Canary]) -> int:
    return count_canary_hits(messages_text(messages), canaries)
