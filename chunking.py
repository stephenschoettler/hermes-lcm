"""Content-aware, turn-aligned chunking of raw conversation history.

This is the substrate for the second embedded corpus (see the chunk-vector
train unit): summaries cover gist and FTS covers exact tokens, while chunks let
recall find VERBATIM detail by meaning. Chunking never splits a message below
the ~600-token target; larger messages split at ~600-token sentence boundaries
with a one-sentence overlap so a span never severs a sentence. Every emitted
chunk carries ``(store_id, chunk_index, char_start, char_end)`` so a KNN hit
maps straight back to ``lcm_expand(store_id=..., content_offset=char_start)``.

The chunk_index is the message's natural window position, so a given physical
span keeps a stable index regardless of which policy selected it — the head
chunk is always index 0, an error chunk keeps the window index it fell on.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Iterable, Iterator

from .message_content import normalize_content_value
from .tokens import count_tokens

# The turn-alignment target: a message at or below this token estimate is a
# single chunk; larger messages split at sentence boundaries near this size.
_CHUNK_TARGET_TOKENS = 600
# ``conversational`` skips pure-acknowledgment turns; keep it simple and skip
# anything below this token estimate unconditionally, embed the rest. Stores
# dominated by short chat turns can lower it (0 embeds everything) — at the
# default, substantive short turns never enter the semantic arm at all.
# INGEST-TIME ONLY, like every chunk-eligibility knob (LCM_EMBED_CONTENT_POLICY
# included): changing it neither backfills newly-eligible turns nor archives
# newly-ineligible chunks already in lcm_chunk_meta. A store built under one
# value is a different corpus from one built under another — rebuild to change it.
MIN_CONVERSATIONAL_TOKENS_ENV = "LCM_CHUNK_MIN_CONVERSATIONAL_TOKENS"
_DEFAULT_MIN_CONVERSATIONAL_TOKENS = 40


def _min_conversational_tokens() -> int:
    raw = os.environ.get(MIN_CONVERSATIONAL_TOKENS_ENV)
    if raw is None:
        return _DEFAULT_MIN_CONVERSATIONAL_TOKENS
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_MIN_CONVERSATIONAL_TOKENS
    if value < 0:
        return _DEFAULT_MIN_CONVERSATIONAL_TOKENS
    return value
# ``full`` bounds a single message at voyage-context-4's per-chunk cap; a message
# larger than this is reduced to head + error chunks rather than fully windowed.
_FULL_PER_MESSAGE_TOKEN_CAP = 32_000
# ``heads`` (and the ``full`` giant-message path) extract at most this many
# error-signature chunks per message, in addition to the head chunk.
_MAX_ERROR_CHUNKS = 3

# Error signatures worth surfacing verbatim from a tool result.
_ERROR_SIGNATURE = re.compile(
    r"traceback|error[:\s]|exception|failed|warning", re.IGNORECASE
)

_CONVERSATIONAL_ROLES = frozenset({"user", "assistant"})
_TOOL_ROLES = frozenset({"tool"})

VALID_CONTENT_POLICIES = ("conversational", "heads", "full")
_DEFAULT_CONTENT_POLICY = "conversational"


@dataclass(frozen=True)
class Chunk:
    """One embeddable span of a single message, aligned to its natural window."""

    store_id: int
    chunk_index: int
    char_start: int
    char_end: int
    text: str
    token_estimate: int

    @property
    def chunk_id(self) -> str:
        """The durable primary key ``store_id:chunk_index``."""
        return f"{self.store_id}:{self.chunk_index}"


def normalize_content_policy(value: Any) -> str:
    """Return a supported content policy, defaulting to ``conversational``.

    Unknown/empty values fall back to the default rather than raising so a
    misconfigured ``LCM_EMBED_CONTENT_POLICY`` degrades to the safe posture.
    """
    normalized = str(value or "").strip().lower()
    return normalized if normalized in VALID_CONTENT_POLICIES else _DEFAULT_CONTENT_POLICY


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """Split text into contiguous sentence spans covering ``[0, len(text))``.

    Boundaries fall after a run of ``.!?`` followed by whitespace/end, and after
    a run of newlines. Trailing whitespace attaches to the preceding sentence so
    the returned spans tile the whole string with no gaps (the char offsets stay
    exact for span-to-content mapping).
    """
    n = len(text)
    spans: list[tuple[int, int]] = []
    start = 0
    i = 0
    while i < n:
        ch = text[i]
        if ch in ".!?":
            j = i + 1
            while j < n and text[j] in ".!?":
                j += 1
            if j >= n or text[j].isspace():
                end = j
                while end < n and text[end].isspace():
                    end += 1
                spans.append((start, end))
                start = end
                i = end
                continue
            i = j
        elif ch == "\n":
            j = i + 1
            while j < n and text[j] == "\n":
                j += 1
            spans.append((start, j))
            start = j
            i = j
        else:
            i += 1
    if start < n:
        spans.append((start, n))
    return spans


def _window_spans(text: str) -> list[tuple[int, int]]:
    """Return turn-aligned ~600-token window spans with one-sentence overlap.

    A message at or below the target is a single window ``[0, len(text))``.
    Larger messages accumulate whole sentences up to the target (always at least
    one), then the next window re-opens on the last sentence of the previous one
    so a one-sentence overlap stitches adjacent chunks. A single sentence larger
    than the target is its own window with no overlap (so windowing terminates).
    """
    n = len(text)
    if n == 0:
        return []
    if count_tokens(text) <= _CHUNK_TARGET_TOKENS:
        return [(0, n)]
    sentences = _sentence_spans(text)
    if not sentences:
        return [(0, n)]
    windows: list[tuple[int, int]] = []
    i = 0
    total = len(sentences)
    while i < total:
        win_start = sentences[i][0]
        accumulated = 0
        j = i
        while j < total:
            s, e = sentences[j]
            sentence_tokens = count_tokens(text[s:e])
            if j > i and accumulated + sentence_tokens > _CHUNK_TARGET_TOKENS:
                break
            accumulated += sentence_tokens
            j += 1
        windows.append((win_start, sentences[j - 1][1]))
        if j >= total:
            break
        # One-sentence overlap when the window held >=2 sentences; a lone
        # oversized sentence advances with no overlap so we never loop forever.
        i = (j - 1) if (j - i) >= 2 else j
    return windows


def _make_chunk(store_id: int, chunk_index: int, text: str, start: int, end: int) -> Chunk:
    span = text[start:end]
    return Chunk(
        store_id=int(store_id),
        chunk_index=int(chunk_index),
        char_start=int(start),
        char_end=int(end),
        text=span,
        token_estimate=count_tokens(span),
    )


def _full_chunks(store_id: int, text: str) -> list[Chunk]:
    return [
        _make_chunk(store_id, index, text, start, end)
        for index, (start, end) in enumerate(_window_spans(text))
    ]


def _head_and_error_chunks(store_id: int, text: str) -> list[Chunk]:
    """Emit the first window plus up to 3 later error-signature windows.

    Window indices are preserved (the head is index 0; each error chunk keeps
    the window position it fell on) so spans map back deterministically.
    """
    windows = _window_spans(text)
    if not windows:
        return []
    selected: list[tuple[int, tuple[int, int]]] = [(0, windows[0])]
    errors = 0
    for index in range(1, len(windows)):
        start, end = windows[index]
        if _ERROR_SIGNATURE.search(text[start:end]):
            selected.append((index, (start, end)))
            errors += 1
            if errors >= _MAX_ERROR_CHUNKS:
                break
    return [
        _make_chunk(store_id, index, text, start, end)
        for index, (start, end) in selected
    ]


def chunk_message(
    store_id: int,
    role: Any,
    content: Any,
    *,
    policy: str = _DEFAULT_CONTENT_POLICY,
) -> list[Chunk]:
    """Chunk one message under the given content policy.

    ``conversational`` (default): user + assistant turns, skipping turns below
    the acknowledgment threshold, otherwise fully windowed.
    ``heads``: conversational PLUS every tool result's head chunk and up to 3
    error-signature chunks.
    ``full``: every role fully windowed, with giant messages (> the per-chunk
    cap) reduced to head + error chunks.
    """
    policy = normalize_content_policy(policy)
    text = content if isinstance(content, str) else (normalize_content_value(content) or "")
    if not text.strip():
        return []
    role_normalized = str(role or "").strip().lower()
    is_conversational_role = role_normalized in _CONVERSATIONAL_ROLES
    is_tool_role = role_normalized in _TOOL_ROLES
    total_tokens = count_tokens(text)

    if policy == "conversational":
        if not is_conversational_role or total_tokens < _min_conversational_tokens():
            return []
        return _full_chunks(store_id, text)

    if policy == "heads":
        if is_conversational_role:
            if total_tokens < _min_conversational_tokens():
                return []
            return _full_chunks(store_id, text)
        if is_tool_role:
            return _head_and_error_chunks(store_id, text)
        return []

    # full: everything, with giant messages reduced to head + error chunks.
    if total_tokens > _FULL_PER_MESSAGE_TOKEN_CAP:
        return _head_and_error_chunks(store_id, text)
    return _full_chunks(store_id, text)


def group_by_store_id(store_ids: Iterable[Any]) -> list[list[int]]:
    """Group positional indexes into per-message documents by ``store_id``.

    Returns a list of index groups, one per contiguous run of the same source
    ``store_id`` (a message = one contextualization document), preserving both
    the store order and the within-message chunk order. Chunk discovery emits a
    message's chunks contiguously, so consecutive-run grouping keeps every
    message's chunks in one group; this is the per-document grouping consumed by
    the contextualized (cross-chunk) embedding path.
    """
    groups: list[list[int]] = []
    last: Any = object()
    for position, store_id in enumerate(store_ids):
        if groups and store_id == last:
            groups[-1].append(position)
        else:
            groups.append([position])
            last = store_id
    return groups


def iter_message_chunks(
    messages: Iterable[Any],
    *,
    policy: str = _DEFAULT_CONTENT_POLICY,
) -> Iterator[Chunk]:
    """Yield chunks for a sequence of message rows/dicts under one policy.

    Each item may be a mapping (or sqlite Row) exposing ``store_id``/``role``/
    ``content``. Rows missing a store_id are skipped rather than raising.
    """
    policy = normalize_content_policy(policy)
    for message in messages:
        store_id = _row_value(message, "store_id")
        if store_id is None:
            continue
        yield from chunk_message(
            int(store_id),
            _row_value(message, "role"),
            _row_value(message, "content"),
            policy=policy,
        )


def _row_value(row: Any, key: str) -> Any:
    if isinstance(row, dict):
        return row.get(key)
    try:
        return row[key]
    except (KeyError, IndexError, TypeError):
        return getattr(row, key, None)
