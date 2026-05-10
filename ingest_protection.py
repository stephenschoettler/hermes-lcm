"""Ingest-time protection for oversized message payloads.

This module prepares copies of messages for durable storage only. It must not
mutate the active provider transcript: assistant ``tool_calls`` and adjacent tool
results remain intact in the live message list while SQLite receives compact
externalized placeholders for supported oversized payload classes.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from .externalize import maybe_externalize_payload
from .message_content import normalize_content_value

_MEDIA_DATA_URI_RE = re.compile(
    r"data:(?:image|audio|video)/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=\s]{16,}",
    re.IGNORECASE,
)
_MEDIA_TYPE_HINTS = ("image", "audio", "video")
_MEDIA_VALUE_KEYS = (
    "image_url",
    "input_image",
    "output_image",
    "audio_url",
    "video_url",
    "image",
    "audio",
    "video",
)


def _contains_media_payload(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(_MEDIA_DATA_URI_RE.search(value))
    if isinstance(value, list):
        return any(_contains_media_payload(item) for item in value)
    if isinstance(value, dict):
        block_type = str(value.get("type") or "").lower()
        if any(hint in block_type for hint in _MEDIA_TYPE_HINTS):
            return True
        for key, nested in value.items():
            key_text = str(key).lower()
            if key_text in _MEDIA_VALUE_KEYS:
                return True
            if _contains_media_payload(nested):
                return True
    return False


def _externalization_kind_for_message(message: Dict[str, Any]) -> str:
    role = str(message.get("role") or "unknown")
    if role == "tool":
        return "tool_result"
    if _contains_media_payload(message.get("content")):
        return "media_payload"
    return "raw_payload"


def protect_message_for_ingest(
    message: Dict[str, Any],
    *,
    session_id: str,
    config,
    hermes_home: str = "",
) -> Dict[str, Any]:
    """Return a storage-safe copy of ``message``.

    The original message is never modified. If externalization is disabled,
    below threshold, or storage cannot be written, the returned copy keeps the
    original content so LCM does not silently lose data.
    """
    protected = dict(message)
    normalized_content = normalize_content_value(message.get("content"))
    if not normalized_content:
        return protected

    role = str(message.get("role") or "unknown")
    kind = _externalization_kind_for_message(message)
    externalized = maybe_externalize_payload(
        normalized_content,
        kind=kind,
        tool_call_id=str(message.get("tool_call_id") or ""),
        session_id=session_id,
        role=role,
        config=config,
        hermes_home=hermes_home,
    )
    if externalized:
        protected["content"] = externalized["placeholder"]
    return protected


def protect_messages_for_ingest(
    messages: List[Dict[str, Any]],
    *,
    session_id: str,
    config,
    hermes_home: str = "",
) -> List[Dict[str, Any]]:
    """Return storage-safe copies of all messages."""
    return [
        protect_message_for_ingest(
            message,
            session_id=session_id,
            config=config,
            hermes_home=hermes_home,
        )
        for message in messages
    ]
