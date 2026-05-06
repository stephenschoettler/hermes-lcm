"""Message-content pattern helpers for LCM ingest filtering.

Patterns are Python regex strings. Compilation is tolerant: an invalid
pattern emits a warning and is skipped, leaving valid patterns in the
same list still active. Matching passes a per-pattern timeout when the
runtime regex engine supports it so catastrophic backtracking cannot
stall synchronous ingest indefinitely. User-supplied anchors (``^``,
``\b``) and inline flags (``(?is)``) work as written.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Iterable

try:  # pragma: no cover - exercised when the optional dependency is absent
    import regex as _regex_engine
except Exception:  # pragma: no cover - keep the plugin importable in minimal installs
    _regex_engine = re

logger = logging.getLogger(__name__)

MESSAGE_PATTERN_MATCH_TIMEOUT_SECONDS = 0.05
MESSAGE_PATTERN_FALLBACK_MAX_CHARS = 100_000
_TIMEOUT_WARNED_PATTERNS: set[str] = set()


def _pattern_label(pattern: Any) -> str:
    return str(getattr(pattern, "pattern", repr(pattern)))


def compile_message_patterns(patterns: Iterable[str]) -> list[Any]:
    """Compile configured message patterns once at startup.

    Each pattern is compiled with the optional ``regex`` engine when it is
    installed, otherwise with stdlib ``re``. Patterns that fail to compile are
    logged at WARNING level and dropped. Returns only compiled patterns.
    """
    compiled: list[Any] = []
    for pattern in patterns:
        try:
            compiled.append(_regex_engine.compile(pattern))
        except _regex_engine.error as exc:
            logger.warning(
                "LCM ignore_message_patterns: skipping invalid regex %r: %s",
                pattern,
                exc,
            )
    return compiled


def _search_with_timeout(pattern: Any, text: str) -> Any:
    try:
        return pattern.search(text, timeout=MESSAGE_PATTERN_MATCH_TIMEOUT_SECONDS)
    except TypeError as exc:
        # stdlib re.Pattern.search has no timeout parameter. Keep minimal installs
        # working, but cap very large inputs to reduce worst-case exposure.
        if "timeout" not in str(exc):
            raise
        return pattern.search(text[:MESSAGE_PATTERN_FALLBACK_MAX_CHARS])


def matches_message_pattern(text: str, patterns: Iterable[Any]) -> bool:
    """Return True when ``text`` matches any compiled pattern.

    A pattern timeout is treated as a non-match and logged once per process for
    that pattern. This preserves ingest availability even when an operator
    configures a pathological regex.
    """
    if not text:
        return False
    for pattern in patterns:
        try:
            if _search_with_timeout(pattern, text):
                return True
        except TimeoutError:
            label = _pattern_label(pattern)
            if label not in _TIMEOUT_WARNED_PATTERNS:
                _TIMEOUT_WARNED_PATTERNS.add(label)
                logger.warning(
                    "LCM ignore_message_patterns: regex %r timed out after %.3gs; treating as no match",
                    label,
                    MESSAGE_PATTERN_MATCH_TIMEOUT_SECONDS,
                )
            continue
    return False
