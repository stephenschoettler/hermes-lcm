"""Message content normalization helpers.

Hermes/OpenAI-format messages may carry ``content`` as plain text or as
structured content parts (for example text + image blocks). LCM persists and
accounts for message content as text, so all write/matching/token paths should
use deliberate normalization.
"""

from __future__ import annotations

import json
from typing import Any

_TEXT_PART_TYPES = {"text", "input_text", "output_text"}


def normalize_content_value(content: Any) -> str | None:
    """Return a stable text representation for message content.

    ``None`` remains ``None`` so callers that distinguish SQL NULL from an empty
    string can preserve that behavior. Strings are returned unchanged. Structured
    content is serialized deterministically so storage, source-id matching, and
    token accounting all see the same value.
    """
    if content is None:
        return None
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(content)


def text_content_for_pattern_matching(content: Any) -> str | None:
    """Return the operator-visible text string used by message filters.

    Structured multimodal payloads often arrive as lists of content parts. For
    ignore-pattern matching, prefer concatenated text parts so anchored patterns
    bind to the text an operator sees. If no text parts are present, fall back to
    the stable normalized representation used for storage.
    """
    if content is None or isinstance(content, str):
        return normalize_content_value(content)
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                part_type = part.get("type")
                text = part.get("text")
                if part_type in _TEXT_PART_TYPES and isinstance(text, str):
                    parts.append(text)
        if parts:
            return "\n".join(parts)
    return normalize_content_value(content)
