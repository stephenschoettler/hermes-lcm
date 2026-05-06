"""Message-content pattern helpers for LCM ingest filtering.

Patterns are Python regex strings. Compilation is tolerant: an invalid
pattern emits a warning and is skipped, leaving valid patterns in the
same list still active. Matching uses ``re.search`` so user-supplied
anchors (``^``, ``\\b``) and inline flags (``(?is)``) work as written.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable, List

logger = logging.getLogger(__name__)


def compile_message_patterns(patterns: Iterable[str]) -> List[re.Pattern[str]]:
    """Compile configured message patterns once at startup.

    Each pattern is compiled with ``re.compile``. Patterns that fail to
    compile are logged at WARNING level and dropped. Returns only the
    patterns that compiled successfully.
    """
    compiled: List[re.Pattern[str]] = []
    for pattern in patterns:
        try:
            compiled.append(re.compile(pattern))
        except re.error as exc:
            logger.warning(
                "LCM ignore_message_patterns: skipping invalid regex %r: %s",
                pattern,
                exc,
            )
    return compiled


def matches_message_pattern(text: str, patterns: Iterable[re.Pattern[str]]) -> bool:
    """Return True when ``text`` matches any of the compiled patterns."""
    if not text:
        return False
    return any(pattern.search(text) for pattern in patterns)
