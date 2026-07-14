"""Token counting utilities for LCM.

Uses tiktoken when available, falls back to char-based estimate.
"""

import logging
import threading
from functools import lru_cache
from typing import Any, Dict, List

from .message_content import normalize_content_value

logger = logging.getLogger(__name__)

_CHARS_PER_TOKEN = 4
_encoder = None
_encoder_ready = False
_encoder_lock = threading.Lock()
_encoder_thread = None
# Cache-key generation, bumped (under _encoder_lock) when the real encoder is
# adopted. The memoized count is keyed on (text, generation), so an estimator
# result computed before adoption can only ever be inserted under the old
# generation: a cache_clear() alone cannot stop an in-flight LRU miss from
# repopulating the cache with a stale estimate *after* the clear, but a stale
# insert under a dead key is unreachable by post-adoption lookups.
_encoder_generation = 0

# Bound on how long a count_tokens() caller will wait for the FIRST encoder
# load. tiktoken.get_encoding() downloads the BPE file over the network when
# it is not already cached on disk; on restricted-egress hosts that request
# can hang for minutes before failing (measured: ~127s per fresh process in a
# production deployment), and _get_encoder() sits on the host's post-turn
# hook path -- so the hang blocked reply delivery. Callers that hit the bound
# use the char-based estimate; the real encoder is adopted (and the count
# cache cleared) whenever the background load completes.
_ENCODER_FIRST_WAIT_S = 2.0


def _load_encoder():
    """The (possibly network-backed) tiktoken load. Patchable in tests."""
    import tiktoken
    return tiktoken.get_encoding("cl100k_base")


def _encoder_loader() -> None:
    global _encoder, _encoder_ready, _encoder_generation
    enc = None
    try:
        enc = _load_encoder()
    except Exception:
        logger.debug("tiktoken not available, using char-based estimates")
    with _encoder_lock:
        _encoder = enc
        _encoder_ready = True
        if enc is not None:
            # Counts for identical text change estimator -> encoder on
            # adoption; bumping the generation retires every pre-adoption
            # cache key, including ones an in-flight miss has yet to insert.
            _encoder_generation += 1
    if enc is not None:
        # Memory hygiene only (correctness comes from the generation key):
        # evict the now-unreachable old-generation entries.
        _count_tokens_cached.cache_clear()


def _get_encoder():
    """Return the tiktoken encoder, never blocking on unbounded network I/O.

    The load runs in a daemon thread. The first caller waits briefly
    (_ENCODER_FIRST_WAIT_S) so the common already-cached-on-disk case still
    gets exact counts immediately; after that, callers never wait -- they use
    the estimator until the loader finishes.
    """
    global _encoder_thread
    if _encoder_ready:
        return _encoder
    first = False
    with _encoder_lock:
        if _encoder_ready:
            return _encoder
        if _encoder_thread is None:
            _encoder_thread = threading.Thread(
                target=_encoder_loader, name="lcm-tiktoken-load", daemon=True
            )
            _encoder_thread.start()
            first = True
    if first:
        _encoder_thread.join(timeout=_ENCODER_FIRST_WAIT_S)
    return _encoder if _encoder_ready else None


def _fallback_token_estimate(text: str) -> int:
    # Latin text is ~4 chars/token, but CJK and other non-Latin scripts
    # tokenize far denser (~1-2 tokens/char). A flat len//4 undercounts them
    # ~3-4x, so preflight under-triggers and assembly can overflow the real
    # budget. ASCII-only text is overwhelmingly common and can use the cheap
    # legacy estimate without scanning every character.
    length = len(text)
    if length == 0:
        return 0
    if text.isascii():
        return length // _CHARS_PER_TOKEN + 1
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    ratio = non_ascii / length
    if ratio >= 0.5:
        divisor = 1.5
    elif ratio >= 0.2:
        divisor = 2.5
    else:
        divisor = _CHARS_PER_TOKEN
    return int(length / divisor) + 1


def _count_tokens_core(text) -> int:
    enc = _get_encoder()
    if enc is not None:
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    if isinstance(text, str):
        return _fallback_token_estimate(text)
    return len(text) // _CHARS_PER_TOKEN + 1


@lru_cache(maxsize=2048)
def _count_tokens_cached(text: str, generation: int) -> int:
    # tiktoken encoding is the dominant per-turn cost: assembly and preflight
    # re-count the same content many times per turn. The counting function is
    # stable within one encoder generation (see _encoder_generation), so a
    # (content, generation) key is stable.
    return _count_tokens_core(text)


# Cap what the LRU may retain by reference. Very large strings are the ones
# least likely to recur identically, and caching them would still let the
# bounded LRU pin unnecessary memory; count those uncached (cost is
# proportional to size either way).
_MAX_CACHEABLE_TOKEN_TEXT_CHARS = 32_768


def count_tokens(text) -> int:
    """Count tokens in a string."""
    if not text:
        return 0
    # Only strings are memoized. Callers may pass non-string, unhashable values
    # (e.g. tool_call arguments as a dict); preserve the legacy tolerance by
    # counting those uncached rather than feeding them to the LRU.
    if isinstance(text, str) and len(text) <= _MAX_CACHEABLE_TOKEN_TEXT_CHARS:
        return _count_tokens_cached(text, _encoder_generation)
    return _count_tokens_core(text)


def count_message_tokens(msg: Dict[str, Any]) -> int:
    """Estimate tokens for a single OpenAI-format message."""
    total = 4  # role + overhead
    content = normalize_content_value(msg.get("content")) or ""
    total += count_tokens(content)
    for tc in msg.get("tool_calls") or []:
        if isinstance(tc, dict):
            fn = tc.get("function", {})
            total += count_tokens(fn.get("name", ""))
            total += count_tokens(fn.get("arguments", ""))
        total += 3  # per-call overhead
    return total


def count_messages_tokens(messages: List[Dict[str, Any]]) -> int:
    """Estimate total tokens for a message list."""
    return sum(count_message_tokens(m) for m in messages)
