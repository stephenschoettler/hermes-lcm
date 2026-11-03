"""Active-context message sanitization helpers.

Pure functions that shape the active replay context emitted back to providers:
strip internal/reasoning content from assistant messages, decide whether an
assistant message still has visible content, and detect sensitive-redaction
markers. Raw store and DAG history stay lossless -- these only sanitize the
active context, never stored rows.

Extracted verbatim from ``LCMEngine`` (WS5 seam 2). These depend only on
``escalation._strip_reasoning_blocks`` and each other, so this module never
imports the engine and introduces no import cycle.
"""

from __future__ import annotations

from typing import Any, Dict

from .escalation import _strip_reasoning_blocks


_VISIBLE_TEXT_PART_TYPES = {"text", "input_text", "output_text"}
_INTERNAL_ASSISTANT_PART_TYPES = {
    "analysis",
    "chain_of_thought",
    "internal",
    "reasoning",
    "redacted_thinking",
    "scratchpad",
    "thought",
    "thinking",
}


def _contains_sensitive_redaction(value: Any) -> bool:
    if isinstance(value, str):
        return "[LCM sensitive redaction:" in value
    if isinstance(value, dict):
        return any(
            _contains_sensitive_redaction(item)
            for pair in value.items()
            for item in pair
        )
    if isinstance(value, list):
        return any(_contains_sensitive_redaction(item) for item in value)
    return False


def _structured_part_text(part: Dict[str, Any]) -> str:
    for key in ("text", "content", "value"):
        value = part.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            nested = value.get("value")
            if isinstance(nested, str):
                return nested
            nested = value.get("content")
            if isinstance(nested, str):
                return nested
    return ""


def _structured_part_has_visible_assistant_content(part: Any) -> bool:
    if part is None:
        return False
    if isinstance(part, str):
        return bool(_strip_reasoning_blocks(part).strip())
    if not isinstance(part, dict):
        return bool(str(part).strip())

    part_type = str(part.get("type") or "").strip().lower()
    if part_type in _INTERNAL_ASSISTANT_PART_TYPES:
        return False
    if part_type in _VISIBLE_TEXT_PART_TYPES:
        return bool(_strip_reasoning_blocks(_structured_part_text(part)).strip())

    # Unknown non-internal content blocks may be visible (for example
    # images/audio/annotations in provider-specific formats).  Preserve
    # them rather than risk dropping a legitimate assistant turn.
    return True


def _assistant_message_has_visible_content(msg: Dict[str, Any]) -> bool:
    content = msg.get("content")
    if content is None:
        return False
    if isinstance(content, str):
        return bool(_strip_reasoning_blocks(content).strip())
    if isinstance(content, list):
        return any(_structured_part_has_visible_assistant_content(part) for part in content)
    if isinstance(content, dict):
        return _structured_part_has_visible_assistant_content(content)
    return bool(str(content).strip())


def _strip_structured_text_part(part: Dict[str, Any]) -> Dict[str, Any] | None:
    cleaned = dict(part)
    for key in ("text", "content", "value"):
        value = cleaned.get(key)
        if isinstance(value, str):
            stripped = _strip_reasoning_blocks(value)
            if not stripped.strip():
                return None
            cleaned[key] = stripped
            return cleaned
        if isinstance(value, dict):
            nested = dict(value)
            for nested_key in ("value", "content", "text"):
                nested_value = nested.get(nested_key)
                if isinstance(nested_value, str):
                    stripped = _strip_reasoning_blocks(nested_value)
                    if not stripped.strip():
                        return None
                    nested[nested_key] = stripped
                    cleaned[key] = nested
                    return cleaned
    return cleaned if _structured_part_has_visible_assistant_content(cleaned) else None


def _sanitize_active_assistant_content(content: Any) -> Any | None:
    if content is None:
        return None
    if isinstance(content, str):
        stripped = _strip_reasoning_blocks(content)
        return stripped if stripped.strip() else None
    if isinstance(content, list):
        cleaned_parts: list[Any] = []
        for part in content:
            if isinstance(part, str):
                stripped = _strip_reasoning_blocks(part)
                if stripped.strip():
                    cleaned_parts.append(stripped)
                continue
            if isinstance(part, dict):
                part_type = str(part.get("type") or "").strip().lower()
                if part_type in _INTERNAL_ASSISTANT_PART_TYPES:
                    continue
                if part_type in _VISIBLE_TEXT_PART_TYPES:
                    cleaned_part = _strip_structured_text_part(part)
                    if cleaned_part is not None:
                        cleaned_parts.append(cleaned_part)
                    continue
            if _structured_part_has_visible_assistant_content(part):
                cleaned_parts.append(part)
        return cleaned_parts or None
    if isinstance(content, dict):
        part_type = str(content.get("type") or "").strip().lower()
        if part_type in _INTERNAL_ASSISTANT_PART_TYPES:
            return None
        if part_type in _VISIBLE_TEXT_PART_TYPES:
            return _strip_structured_text_part(content)
        return content if _structured_part_has_visible_assistant_content(content) else None
    return content if str(content).strip() else None


def _clean_active_assistant_message(msg: Dict[str, Any]) -> Dict[str, Any] | None:
    if msg.get("role") != "assistant":
        return msg
    if "content" not in msg:
        return msg
    cleaned_content = _sanitize_active_assistant_content(msg.get("content"))
    if cleaned_content is None:
        if not msg.get("tool_calls") and not msg.get("codex_message_items"):
            return None
        cleaned_content = ""
    if cleaned_content == msg.get("content"):
        return msg
    cleaned = dict(msg)
    cleaned["content"] = cleaned_content
    return cleaned


def _should_drop_active_assistant_message(msg: Dict[str, Any]) -> bool:
    if msg.get("role") != "assistant":
        return False
    if msg.get("tool_calls"):
        return False
    return _clean_active_assistant_message(msg) is None
