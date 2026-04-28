"""Message content normalization helpers.

Hermes/OpenAI-format messages may carry ``content`` as plain text or as
structured content parts (for example text + image blocks). LCM persists and
accounts for message content as text, so all write/matching/token paths should
use the same normalization.
"""

from __future__ import annotations

import json
from typing import Any


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
