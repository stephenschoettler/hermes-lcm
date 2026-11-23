"""Ingest-cursor reconciliation and replay-identity for the LCM engine (WS5 Seam 4).

The ``ReconcileMixin`` holds the machinery that reconciles the persisted store
tail against the active message list after a process restart, plus the stable
replay-identity primitives it relies on. These methods were lifted verbatim out
of ``LCMEngine`` and continue to run bound to the engine instance (``self`` is
the ``LCMEngine``), so they read the engine's runtime state (``_store``,
``_session_id``, ``_config``, ``_ingest_cursor`` is written by the engine from
the value these return) and call back into engine helpers through normal
attribute lookup. ``LCMEngine`` mixes this in, so no call site and no test
changes.

``_PRESERVED_OBJECTIVE_CONTEXT_PREFIX`` lives here (used by the reconciliation
scan) and is re-exported to ``engine.py``; the two tool-call-identity
staticmethods reference the mixin class directly rather than ``LCMEngine`` to
avoid an import cycle (staticmethod resolution is identical).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .externalize import (
    extract_externalized_ref,
    externalized_tool_result_has_persisted_output_marker,
    find_externalized_tool_result_content_for_call,
    load_externalized_payload,
)
from .ingest_protection import (
    _add_inline_persisted_output_generation_metadata,
    _add_inline_persisted_output_identity_metadata,
    _expected_persisted_output_chars,
    _has_inline_persisted_output_generation_metadata,
    _has_lossy_sensitive_redaction,
    _is_hermes_persisted_output_marker,
    _json_has_duplicate_object_keys,
    _persisted_output_marker_identity_digest,
    _persisted_output_saved_path,
    recover_hermes_persisted_output_with_file_stat,
    redact_sensitive_value,
)
from .message_content import normalize_content_value, text_content_for_pattern_matching
from .sanitize import _clean_active_assistant_message
from .store import _normalize_observed_at

import logging

logger = logging.getLogger(__name__)

# Content-presence encoding for replay identities: content=None (SQL NULL in
# durable rows / absent active fields) must not collide with content='' in the
# 4-field identity tuple, so absence carries an unambiguous sentinel prefix and
# live values that already start with it are escaped. Both constants are
# impossible as provider message content because '[' cannot start an escaped
# form and the sentinel is fixed-length.
_REPLAY_IDENTITY_ABSENT_CONTENT_PREFIX = "[LCM replay identity: content absent]"
_REPLAY_IDENTITY_ABSENT_CONTENT_ESCAPE_PREFIX = "[LCM replay identity: content escaped] "


def _count_leading_reserved_prefixes(content: str) -> int:
    """How many reserved prefixes the string starts with, consuming greedily.

    The escape prefix "wins" when the string starts with it (it is longer and
    its own reserved namespace), so an escaped string's marker is consumed as
    one prefix before its remainder is probed.
    """
    count = 0
    # Offset-based scan (round-3 finding 4041509636): each slice assignment
    # copies the remaining string, making N leading prefixes quadratic in the
    # payload size; startswith(prefix, pos) scans in place — O(total prefix
    # bytes) instead of O(N * payload).
    pos = 0
    escape_len = len(_REPLAY_IDENTITY_ABSENT_CONTENT_ESCAPE_PREFIX)
    absent_len = len(_REPLAY_IDENTITY_ABSENT_CONTENT_PREFIX)
    size = len(content)
    while True:
        if content.startswith(_REPLAY_IDENTITY_ABSENT_CONTENT_ESCAPE_PREFIX, pos):
            pos += escape_len
        elif content.startswith(_REPLAY_IDENTITY_ABSENT_CONTENT_PREFIX, pos):
            pos += absent_len
        else:
            return count
        count += 1
        if pos >= size:
            return count


def _escape_replay_identity_content(normalized_content: str) -> str:
    """Injective encoding for the identity's content component.

    Unprefixed content passes through unchanged. Content starting with either
    reserved prefix carries an explicit count of the consumed leading prefixes
    plus the original string verbatim; the counted marker namespace ("[LCM
    replay identity: content escaped] x<count> ") is unreachable by uncounted
    strings because the marker itself starts with a reserved prefix and would
    therefore have been counted. Injectivity proof sketch: two encodings equal
    implies both unprefixed (then the strings are equal) or both counted (the
    count field matches, then the verbatim originals match).
    """
    count = _count_leading_reserved_prefixes(normalized_content)
    if count == 0:
        return normalized_content
    return (
        f"{_REPLAY_IDENTITY_ABSENT_CONTENT_ESCAPE_PREFIX}x{count} "
        + normalized_content
    )


# One-character content-shape tags (round-8 finding 4029411030). The identity's
# content component is prefixed with the shape of the ORIGINAL value so
# structured list/dict content and its JSON-text serialization stop sharing an
# identity — the sanitation handoff digest (compaction.py) flows from this
# component and inherits the distinction. The tag composes with the escape
# machinery (tag first, then the injective escape encoding), and the pair
# (tag, escaped content) stays injective because the tag is one fixed-alphabet
# character.
_REPLAY_IDENTITY_SHAPE_TAG_STRING = "s"
_REPLAY_IDENTITY_SHAPE_TAG_LIST = "l"
_REPLAY_IDENTITY_SHAPE_TAG_DICT = "d"
_REPLAY_IDENTITY_SHAPE_TAG_NULL = "n"
_REPLAY_IDENTITY_SHAPE_TAG_OTHER = "o"
_REPLAY_IDENTITY_SHAPE_TAGS = frozenset("sldno")


def _replay_identity_shape_tag_for_value(content: Any) -> str:
    """Shape tag for a LIVE (raw) message content value."""
    if content is None:
        return _REPLAY_IDENTITY_SHAPE_TAG_NULL
    if isinstance(content, str):
        return _REPLAY_IDENTITY_SHAPE_TAG_STRING
    if isinstance(content, list):
        return _REPLAY_IDENTITY_SHAPE_TAG_LIST
    if isinstance(content, dict):
        return _REPLAY_IDENTITY_SHAPE_TAG_DICT
    return _REPLAY_IDENTITY_SHAPE_TAG_OTHER


def _replay_identity_shape_tag_for_stored_text(normalized_content: str) -> str:
    """Shape tag for a STORED row's canonical text content.

    Storage canonicalizes structured content to JSON text (``normalize_content_value``
    in the store write path), so a stored row cannot carry the original shape.
    The stored side derives the tag by EXACT round-trip decode: text that parses
    as list/dict and re-serializes byte-identically is tagged structured — the
    same convention as ``_identity_content_for_active_cleanup`` — so a row
    written from a structured live value keeps that value's identity across the
    round trip. RESIDUAL LIMITATION (unavoidable, information-theoretic): a live
    STRING whose text is exactly the canonical JSON of a list/dict is stored
    byte-identically to the structured form, so its stored row carries the
    structured tag while the live value carried the string tag; such rows are
    not exact-matchable against their live replay after a restart (they still
    match through the escaped/active-cleanup fallback views where applicable).
    """
    if not normalized_content:
        return _REPLAY_IDENTITY_SHAPE_TAG_STRING
    probe = normalized_content.lstrip()
    first = probe[:1]
    if first not in "[{":
        return _REPLAY_IDENTITY_SHAPE_TAG_STRING
    try:
        decoded = json.loads(probe)
    except (TypeError, ValueError, json.JSONDecodeError):
        return _REPLAY_IDENTITY_SHAPE_TAG_STRING
    if isinstance(decoded, list):
        expected_tag = _REPLAY_IDENTITY_SHAPE_TAG_LIST
    elif isinstance(decoded, dict):
        expected_tag = _REPLAY_IDENTITY_SHAPE_TAG_DICT
    else:
        return _REPLAY_IDENTITY_SHAPE_TAG_STRING
    if normalize_content_value(decoded) == normalized_content:
        return expected_tag
    return _REPLAY_IDENTITY_SHAPE_TAG_STRING


def _strip_replay_identity_shape_tag(content: str) -> str:
    """Remove the shape tag from an identity content component (if present)."""
    if content and content[:1] in _REPLAY_IDENTITY_SHAPE_TAGS:
        return content[1:]
    return content


def _tail_tagless(identities: list[tuple[str, str, str, str]]) -> list[tuple[str, str, str, str]]:
    """Shape-tag-stripped identity list for content-only comparisons.

    Reconcile paths compare LIVE identities (list/dict-tagged structured
    content) against STORED identities (string-tagged text); the tag is a
    live-vs-claim distinction, not part of the row content identity
    ."""
    return [
        (role, _strip_replay_identity_shape_tag(content), tool_call_id, tool_calls)
        for role, content, tool_call_id, tool_calls in identities
    ]

_PRESERVED_OBJECTIVE_CONTEXT_PREFIX = "[Current user objective preserved from compacted history]"


_POST_INGEST_MUTATED_TOOL_ROLE_PREFIXES = (
    # Post-ingest rewrites that DESTROY the original tool payload: retry
    # recovery is impossible for these, so a replayed original aligns on
    # role + tool_call_id. The threshold-externalizer forms are deliberately
    # ABSENT: the store's dedupe-replay guard exempts them (_dedupe_replay_applies)
    # because retry semantics live above the store — a replayed raw tool
    # result against an externalized row must re-append, never collapse.
    "[Old tool output cleared to save context space]",
    "[LCM active replay placeholder:",
    "[GC'd externalized payload:",
    "[GC'd externalized tool output:",
)

_POST_INGEST_MUTATED_ANY_ROLE_PREFIXES = (
    "[Old tool output cleared to save context space]",
    "[LCM sensitive redaction:",
    "[LCM active replay placeholder:",
)


def _stored_row_content_post_ingest_mutated(content: str, *, role: str = "") -> bool:
    """True when a stored row's content is a known post-ingest rewrite.

    Host-side pruning clears tool outputs to a fixed placeholder, sensitive
    redaction and active-replay placeholder rewrites replace payload-bearing
    text with placeholder forms. For those rows the replayed original can
    never byte-match the stored copy, so the alignment falls back to the
    surviving role + tool_call_id pair.

    Role-specific: for tool rows only the payload-DESTROYING rewrites count
    (cleared outputs, GC'd externalized payloads). The threshold-externalizer
    forms are deliberately excluded — the store's dedupe-replay guard exempts
    them (``_dedupe_replay_applies``) because retry semantics live above the
    store: a replayed raw tool result against an externalized row must
    re-append, never collapse.
    """
    if not content:
        return False
    if role == "tool":
        prefixes = _POST_INGEST_MUTATED_TOOL_ROLE_PREFIXES
    else:
        prefixes = _POST_INGEST_MUTATED_ANY_ROLE_PREFIXES
    for prefix in prefixes:
        if content.startswith(prefix):
            return True
    return False


class ReconcileMixin:
    @staticmethod
    def _canonicalize_tool_call_identity_value(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: ReconcileMixin._canonicalize_tool_call_identity_value(val)
                for key, val in value.items()
            }
        if isinstance(value, list):
            return [ReconcileMixin._canonicalize_tool_call_identity_value(item) for item in value]
        if isinstance(value, str):
            stripped = value.strip()
            if stripped and stripped[0] in "[{":
                if _json_has_duplicate_object_keys(value):
                    return value
                try:
                    parsed = json.loads(value)
                except (TypeError, ValueError, json.JSONDecodeError):
                    return value
                if isinstance(parsed, (dict, list)):
                    canonical = ReconcileMixin._canonicalize_tool_call_identity_value(parsed)
                    return json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            return value
        return value

    @staticmethod
    def _stable_tool_calls_identity(tool_calls: Any) -> str:
        if not tool_calls:
            return ""
        try:
            canonical = ReconcileMixin._canonicalize_tool_call_identity_value(tool_calls)
            return json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        except (TypeError, ValueError):
            return str(tool_calls)

    def _has_durable_persisted_output_replay_identity(self, msg: Dict[str, Any]) -> bool:
        role = str(msg.get("role") or "unknown")
        content = normalize_content_value(msg.get("content")) or ""
        if role != "tool" or not _is_hermes_persisted_output_marker(content):
            return False
        expected_chars = _expected_persisted_output_chars(content)
        persisted_output_source_path = _persisted_output_saved_path(content)
        persisted_output_preview_sha256, allow_redacted_preview_match = self._persisted_output_marker_replay_proof(content)
        if (
            expected_chars is None
            or not persisted_output_source_path
            or not persisted_output_preview_sha256
        ):
            return False
        recovered_with_stat = recover_hermes_persisted_output_with_file_stat(content)
        if recovered_with_stat is None:
            return False
        require_live_file_freshness = True
        durable_content = find_externalized_tool_result_content_for_call(
            tool_call_id=str(msg.get("tool_call_id") or ""),
            session_id=str(msg.get("session_id") or self._session_id or ""),
            expected_chars=expected_chars,
            persisted_output_source_path=persisted_output_source_path,
            persisted_output_preview_sha256=persisted_output_preview_sha256,
            require_persisted_output_file_not_newer=require_live_file_freshness,
            allow_redacted_preview_match=allow_redacted_preview_match,
            config=self._config,
            hermes_home=self._hermes_home,
        )
        if durable_content is None:
            return False
        if recovered_with_stat is not None:
            recovered_content, _file_stat = recovered_with_stat
            if not self._recovered_content_matches_durable_identity(recovered_content, durable_content):
                return False
        return True

    def _message_replay_identity(self, msg: Dict[str, Any], *, stored_row: bool = False) -> tuple[str, str, str, str]:
        role = str(msg.get("role") or "unknown")
        normalized_content = normalize_content_value(msg.get("content"))
        # Encode content presence (None vs '') inside the existing content
        # component: callers unpack exactly 4 fields, so absence is marked with
        # a sentinel prefix instead of widening the identity tuple. Raw
        # placeholder-marker content is matched through the separate
        # raw-placeholder identity path, which restores the unprefixed form.
        # Content-presence AND prefix-namespace encoding (round-8 findings
        # 4029411030/1037). Absence is marked with the sentinel prefix. A live
        # value starting with either reserved prefix is escaped with a COUNTED
        # marker — the number of leading reserved prefixes is embedded
        # ("[LCM replay identity: content escaped] xN " + the original string
        # verbatim) — instead of one blind re-escape prepend: a naive
        # while-startswith loop never terminates (its own output still starts
        # with the escape prefix) and a single prepend leaves the
        # escape-prefixed namespace collidable (a provider string that merely
        # begins with the escape prefix passed through unchanged and collided
        # with escaped absent-prefixed content). The counted form is injective:
        # the marker (with its explicit count) is impossible as an unescaped
        # provider string's identity because any string starting with the
        # marker's own prefix is itself counted, and equal encodings imply
        # equal count + equal original.
        # The persisted-output marker branch below matches and rewrites the
        # UNTAGGED encoded content (marker text never starts with a shape tag
        # character in a reserved namespace, but the marker matchers expect the
        # bare encoded form); the shape tag is applied to the FINAL component
        # value at the end of this method.
        if normalized_content is None:
            content = _REPLAY_IDENTITY_ABSENT_CONTENT_PREFIX + ""
        else:
            content = _escape_replay_identity_content(normalized_content)
        if (
            role == "tool"
            and _is_hermes_persisted_output_marker(content)
            and bool(getattr(self._config, "large_output_externalization_enabled", True))
        ):
            expected_chars = _expected_persisted_output_chars(content)
            persisted_output_source_path = _persisted_output_saved_path(content)
            persisted_output_preview_sha256, allow_redacted_preview_match = self._persisted_output_marker_replay_proof(content)
            durable_content = None
            recovered_with_stat = recover_hermes_persisted_output_with_file_stat(content) if not stored_row else None
            recovered_content = recovered_with_stat[0] if recovered_with_stat is not None else None
            recovered_identity_content = None
            if recovered_content is not None:
                recovered_identity_content = normalize_content_value(
                    redact_sensitive_value(
                        recovered_content,
                        self._config,
                        parse_json_strings=False,
                    )
                )
            require_live_file_freshness = recovered_with_stat is not None

            def live_file_generation_identity() -> str:
                try:
                    live_stat = Path(str(persisted_output_source_path)).stat()
                    return (
                        "[LCM persisted-output live file: "
                        f"path={persisted_output_source_path}; "
                        f"mtime_ns={live_stat.st_mtime_ns}; "
                        f"chars={expected_chars}]"
                    )
                except OSError:
                    return (
                        "[LCM persisted-output live file: "
                        f"path={persisted_output_source_path}; "
                        f"chars={expected_chars}]"
                    )

            if (
                not stored_row
                and expected_chars is not None
                and persisted_output_source_path
                and persisted_output_preview_sha256
                and recovered_with_stat is not None
            ):
                durable_content = find_externalized_tool_result_content_for_call(
                    tool_call_id=str(msg.get("tool_call_id") or ""),
                    session_id=str(msg.get("session_id") or self._session_id or ""),
                    expected_chars=expected_chars,
                    persisted_output_source_path=persisted_output_source_path,
                    persisted_output_preview_sha256=persisted_output_preview_sha256,
                    require_persisted_output_file_not_newer=require_live_file_freshness,
                    allow_redacted_preview_match=allow_redacted_preview_match,
                    config=self._config,
                    hermes_home=self._hermes_home,
                )
            if durable_content is not None and (
                recovered_content is None or self._recovered_content_matches_durable_identity(recovered_content, durable_content)
            ):
                content = durable_content
            elif recovered_content is not None:
                stale_durable_content = find_externalized_tool_result_content_for_call(
                    tool_call_id=str(msg.get("tool_call_id") or ""),
                    session_id=str(msg.get("session_id") or self._session_id or ""),
                    expected_chars=expected_chars,
                    persisted_output_source_path=persisted_output_source_path,
                    persisted_output_preview_sha256=persisted_output_preview_sha256,
                    allow_redacted_preview_match=allow_redacted_preview_match,
                    config=self._config,
                    hermes_home=self._hermes_home,
                )
                if (
                    stale_durable_content is not None
                    and self._recovered_content_matches_durable_identity(recovered_content, stale_durable_content)
                    and not _has_lossy_sensitive_redaction(stale_durable_content)
                    and not _has_lossy_sensitive_redaction(recovered_identity_content)
                ):
                    content = stale_durable_content
                elif stale_durable_content is not None:
                    content = live_file_generation_identity()
                elif recovered_with_stat is not None:
                    content = _add_inline_persisted_output_generation_metadata(
                        _add_inline_persisted_output_identity_metadata(
                            content,
                            _persisted_output_marker_identity_digest(content),
                        ),
                        recovered_with_stat[1],
                    )
                elif recovered_identity_content is not None:
                    content = recovered_identity_content
        tool_calls = msg.get("tool_calls")
        if stored_row:
            session_id = str(msg.get("session_id") or self._session_id or "")
            content = self._restore_ingest_payload_placeholders_in_content_identity(
                content,
                session_id=session_id,
            )
            tool_calls = self._restore_ingest_payload_placeholders_in_value(tool_calls, session_id=session_id)
        ref = extract_externalized_ref(content)
        if ref and "quarantined_assistant_output" not in content:
            payload = load_externalized_payload(
                ref,
                config=self._config,
                hermes_home=self._hermes_home,
            )
            if payload is not None and isinstance(payload.get("content"), str):
                content = payload["content"]
        # Reserved-prefix re-escape (round-3 finding 4041846916): content
        # restored from sidecars/durable payloads above replaced the initially
        # encoded form; if the restored text begins with a reserved identity
        # prefix it must carry the counted escape encoding like any other
        # identity content, or live and stored identities diverge and the row
        # duplicates on restart.
        if (
            isinstance(content, str)
            and content
            and not content.startswith(_REPLAY_IDENTITY_ABSENT_CONTENT_PREFIX)
        ):
            leading = _count_leading_reserved_prefixes(content)
            if leading > 0 and not content.startswith(
                _REPLAY_IDENTITY_ABSENT_CONTENT_ESCAPE_PREFIX
            ):
                content = _escape_replay_identity_content(content)
        tool_calls_identity = self._stable_tool_calls_identity(tool_calls)
        # Shape tag (round-8 finding 4029411030): only the LIVE side tags the
        # RAW value shape — that is where structured-vs-string distinction is
        # knowable, and the handoff digest (both endpoints live) needs it. The
        # stored side tags text as STRING unconditionally (NULL keeps the
        # absent tag): a stored row cannot know whether its canonical JSON text
        # was written from a structured value or from a live string whose text
        # happens to be canonical JSON, and guessing breaks the engine contract
        # that a re-ingested literal-JSON-string content dedupes against itself
        # (TestAssemblyToolPairGuardrail rebind tests). Residual asymmetry: a
        # live list's stored row replays with the string tag after restart —
        # the pre-existing restart-duplication behavior, unchanged by the tag.
        if stored_row:
            shape_tag = (
                _REPLAY_IDENTITY_SHAPE_TAG_NULL
                if normalized_content is None
                else _REPLAY_IDENTITY_SHAPE_TAG_STRING
            )
        else:
            shape_tag = _replay_identity_shape_tag_for_value(msg.get("content"))
        return (
            role,
            shape_tag + content,
            str(msg.get("tool_call_id") or ""),
            tool_calls_identity,
        )

    @staticmethod
    def _matches_store_tail_suffix(
        stored_tail: list[tuple[str, str, str, str]],
        candidate_prefix: list[tuple[str, str, str, str]],
    ) -> bool:
        if not candidate_prefix:
            return True
        if len(candidate_prefix) > len(stored_tail):
            return False
        # Shape-tag agnostic (round-8 4029411030 follow-up): a stored row's tag
        # (string for text) can differ from the live message's tag (list for
        # structured content); reconciliation matches by CONTENT, so compare
        # with the tag stripped on both sides.
        def _tagless(identities: list[tuple[str, str, str, str]]) -> list[tuple[str, str, str, str]]:
            return [
                (role, _strip_replay_identity_shape_tag(content), tool_call_id, tool_calls)
                for role, content, tool_call_id, tool_calls in identities
            ]
        return _tagless(stored_tail[-len(candidate_prefix) :]) == _tagless(candidate_prefix)

    @staticmethod
    def _strip_inline_persisted_output_generation_identity(
        identity: tuple[str, str, str, str],
    ) -> tuple[str, str, str, str]:
        role, content, tool_call_id, tool_calls = identity
        if role != "tool" or not isinstance(content, str):
            return identity
        stripped = re.sub(
            r"\n?\[LCM persisted-output file generation: "
            r"size=\d+; mtime_ns=\d+; ctime_ns=\d+\]\n?(?=</persisted-output>)",
            "\n",
            content,
        )
        return (role, stripped, tool_call_id, tool_calls)

    def _stored_row_has_durable_persisted_output_marker(self, row: Dict[str, Any]) -> bool:
        if str(row.get("role") or "") != "tool":
            return False
        content = normalize_content_value(row.get("content")) or ""
        ref = extract_externalized_ref(content)
        if not ref:
            return False
        return externalized_tool_result_has_persisted_output_marker(
            ref,
            config=self._config,
            hermes_home=self._hermes_home,
        )

    @staticmethod
    def _persisted_output_durable_wildcard_identity(
        identity: tuple[str, str, str, str],
    ) -> tuple[str, str, str, str]:
        role, _content, tool_call_id, tool_calls = identity
        return (role, "[LCM persisted-output durable replay]", tool_call_id, tool_calls)

    def _matches_persisted_output_durable_full_replay(
        self,
        candidate_messages: list[Dict[str, Any]],
        candidate_prefix: list[tuple[str, str, str, str]],
        stored_tail: list[tuple[str, str, str, str]],
        stored_tail_rows: list[Dict[str, Any]] | None,
    ) -> bool:
        if not stored_tail_rows or len(candidate_prefix) != len(stored_tail) or len(candidate_messages) != len(candidate_prefix):
            return False
        transformed_candidate: list[tuple[str, str, str, str]] = []
        transformed_stored: list[tuple[str, str, str, str]] = []
        saw_persisted_output = False
        for candidate_msg, candidate_identity, stored_identity, stored_row in zip(
            candidate_messages,
            candidate_prefix,
            stored_tail,
            stored_tail_rows,
        ):
            candidate_content = normalize_content_value(candidate_msg.get("content")) or ""
            candidate_is_persisted_marker = (
                str(candidate_msg.get("role") or "") == "tool"
                and _is_hermes_persisted_output_marker(candidate_content)
            )
            stored_is_persisted_output = self._stored_row_has_durable_persisted_output_marker(stored_row)
            if candidate_is_persisted_marker or stored_is_persisted_output:
                if (
                    not candidate_is_persisted_marker
                    or not stored_is_persisted_output
                    or not self._has_durable_persisted_output_replay_identity(candidate_msg)
                ):
                    return False
                saw_persisted_output = True
                transformed_candidate.append(self._persisted_output_durable_wildcard_identity(candidate_identity))
                transformed_stored.append(self._persisted_output_durable_wildcard_identity(stored_identity))
                continue
            # Shape-tag agnostic : content-identity comparison
            # across the live/stored boundary must ignore the shape tag.
            def _tagless1(identity: tuple[str, str, str, str]) -> tuple[str, str, str, str]:
                return (
                    identity[0],
                    _strip_replay_identity_shape_tag(identity[1]),
                    identity[2],
                    identity[3],
                )
            transformed_candidate.append(_tagless1(candidate_identity))
            transformed_stored.append(_tagless1(stored_identity))
        return saw_persisted_output and transformed_candidate == transformed_stored

    @classmethod
    def _identity_content_for_active_cleanup(
        cls, content: str, content_is_tagged: bool = True
    ) -> Any:
        """Decode canonical stored JSON content before active-cleanup checks.

        Structured assistant content is persisted as deterministic JSON. Active
        replay cleanup sees the original list/dict shape, so restart
        reconciliation has to decode the stored identity before deciding whether
        a durable assistant row could be absent from sanitized active context.
        The identity's shape tag (round-8 finding 4029411030) is stripped before
        decoding so the tagged content component does not defeat the JSON parse.
        ``content_is_tagged=False`` : the caller already
        supplies a TAGLESS content component (the store-id map strips tags on
        both sides); peeling again would eat the first character of assistant
        content that merely starts with s/l/d/n/o (e.g. "null hypothesis").
        """
        if not isinstance(content, str):
            return content
        if content_is_tagged:
            content = _strip_replay_identity_shape_tag(content)
        # Absent-content sentinel (round-3 finding 4041846913): a stored row
        # with SQL-NULL content (the common assistant tool-call shape) carries
        # the sentinel as its identity content; assistant cleanup must see
        # content=None, not a nonempty sentinel string, or the cleaned durable
        # tail stops matching the active replay after restart.
        if content == _REPLAY_IDENTITY_ABSENT_CONTENT_PREFIX:
            return None
        try:
            decoded = json.loads(content)
        except (TypeError, ValueError, json.JSONDecodeError):
            return content
        if isinstance(decoded, (list, dict)) and normalize_content_value(decoded) == content:
            return decoded
        return content

    @classmethod
    def _active_cleanup_replay_identity(
        cls,
        identity: tuple[str, str, str, str],
        content_is_tagged: bool = True,
    ) -> tuple[str, str, str, str] | None:
        role, content, tool_call_id, tool_calls = identity
        if role != "assistant":
            return identity
        msg: dict[str, Any] = {
            "role": role,
            "content": cls._identity_content_for_active_cleanup(
                content, content_is_tagged=content_is_tagged
            ),
        }
        if tool_calls:
            try:
                decoded_tool_calls = json.loads(tool_calls)
            except (TypeError, ValueError, json.JSONDecodeError):
                decoded_tool_calls = tool_calls
            msg["tool_calls"] = decoded_tool_calls
        cleaned = _clean_active_assistant_message(msg)
        if cleaned is None:
            return None
        # Re-derive the shape tag from the cleaned value's RAW shape (round-8
        # finding 4029411030): active cleanup preserves list/dict shapes, so
        # the cleaned variant must stay comparable with live identities.
        # Tagless callers  get tagless output back — the
        # store-id map's comparisons are content-only by design.
        cleaned_content = cleaned.get("content")
        cleaned_normalized = normalize_content_value(cleaned_content) or ""
        if not content_is_tagged:
            return (role, cleaned_normalized, tool_call_id, tool_calls)
        cleaned_tag = _replay_identity_shape_tag_for_value(cleaned_content)
        return (
            role,
            cleaned_tag + cleaned_normalized,
            tool_call_id,
            tool_calls,
        )

    @staticmethod
    def _is_quarantined_assistant_replay_identity(identity: tuple[str, str, str, str]) -> bool:
        role, content, _tool_call_id, _tool_calls = identity
        if role != "assistant":
            return False
        text = str(_strip_replay_identity_shape_tag(content) or "").strip()
        return bool(
            re.fullmatch(
                r"\[Externalized LCM ingest payload: assistant output quarantined; "
                r"kind=quarantined_assistant_output; "
                r"reason=[A-Za-z0-9_.:/-]+; "
                r"field=[A-Za-z0-9_.:/<>\[\]-]+; "
                r"chars=\d+; bytes=\d+; "
                r"ref=[^\]\s]+\]",
                text,
            )
            or re.fullmatch(
                r"\[LCM active replay placeholder: assistant output quarantined; "
                r"kind=quarantined_assistant_output; "
                r"reason=[A-Za-z0-9_.:/-]+; "
                r"scope=ignored_message_pattern; field=content; "
                r"chars=\d+; bytes=\d+; "
                r"sha256=[0-9a-f]{16}\]",
                text,
            )
        )

    def _stored_tail_for_sanitized_active_replay(
        self,
        stored_tail: list[tuple[str, str, str, str]],
    ) -> list[tuple[str, str, str, str]]:
        """Mirror active-context cleanup for restart replay reconciliation.

        Raw storage remains lossless. This view is used only to reconcile a
        restarted process when the host replays sanitized active context where
        assistant rows may be removed or have internal content stripped.
        """
        sanitized_tail: list[tuple[str, str, str, str]] = []
        for identity in stored_tail:
            cleaned_identity = self._active_cleanup_replay_identity(identity)
            if cleaned_identity is not None:
                sanitized_tail.append(cleaned_identity)
        return sanitized_tail

    def _find_reconciled_cursor_for_store_tail(
        self,
        messages: List[Dict[str, Any]],
        stored_tail: list[tuple[str, str, str, str]],
        *,
        stored_tail_rows: list[Dict[str, Any]] | None = None,
        allow_empty_prefix: bool,
        session_count: int,
        raw_session_count: int,
    ) -> int | None:
        sanitized_replay_tail = self._stored_tail_for_sanitized_active_replay(stored_tail)
        effective_session_count = len(sanitized_replay_tail)
        sanitized_tail_collapsed = len(sanitized_replay_tail) < len(stored_tail)
        boundary_messages = list(stored_tail_rows or [])
        if not boundary_messages:
            for role, content, tool_call_id, tool_calls in stored_tail:
                try:
                    decoded_tool_calls = json.loads(tool_calls) if tool_calls else []
                except (TypeError, ValueError, json.JSONDecodeError):
                    decoded_tool_calls = []
                boundary_messages.append({
                    "role": role,
                    "content": _strip_replay_identity_shape_tag(content),
                    "tool_call_id": tool_call_id,
                    "tool_calls": decoded_tool_calls,
                })
        effective_fresh_tail_count = self._fresh_tail_boundary(boundary_messages).count
        empty_prefix_cursor: int | None = None
        for cursor in range(len(messages), -1, -1):
            candidate_messages = messages[:cursor]
            candidate_visible_messages = [
                msg
                for msg in candidate_messages
                if not self._is_replayed_context_scaffold_message(msg)
                and not self._matches_ignore_message_patterns(msg)
            ]
            candidate_non_placeholder_messages = [
                msg
                for msg in candidate_visible_messages
                if not self._is_volatile_ignored_quarantine_placeholder(
                    msg,
                    text_content_for_pattern_matching(msg.get("content")) or "",
                )
                and not self._is_ignored_active_replay_placeholder(
                    msg,
                    text_content_for_pattern_matching(msg.get("content")) or "",
                )
                and not (
                    self._compiled_ignore_message_patterns
                    and self._is_quarantined_assistant_replay_identity(
                        self._message_replay_identity(msg)
                    )
                    and self._matches_ignore_message_patterns(msg, stored_row=True)
                )
            ]
            filtered_candidate_placeholders = len(candidate_non_placeholder_messages) < len(candidate_visible_messages)
            candidate_has_scaffold_evidence = any(
                self._is_replayed_context_scaffold_message(msg) for msg in candidate_messages
            )
            candidate_has_quarantined_replay_evidence = any(
                self._is_quarantined_assistant_replay_identity(self._message_replay_identity(msg))
                for msg in candidate_messages
            )
            candidate_identity_messages = (
                candidate_non_placeholder_messages
                if candidate_non_placeholder_messages or filtered_candidate_placeholders
                else candidate_visible_messages
            )
            candidate_visible_prefix = [
                self._message_replay_identity(msg)
                for msg in candidate_visible_messages
            ]
            candidate_prefix = [
                self._message_replay_identity(msg)
                for msg in candidate_identity_messages
            ]
            if not candidate_prefix:
                empty_prefix_cursor = cursor
                if allow_empty_prefix and (
                    not filtered_candidate_placeholders
                    or candidate_has_scaffold_evidence
                    or candidate_has_quarantined_replay_evidence
                ):
                    return cursor
                continue

            matches_sanitized_tail = (
                len(candidate_prefix) <= len(sanitized_replay_tail)
                and self._matches_store_tail_suffix(sanitized_replay_tail, candidate_prefix)
            )
            matches_raw_tail = self._matches_store_tail_suffix(stored_tail, candidate_prefix)
            matches_visible_sanitized_tail = (
                filtered_candidate_placeholders
                and bool(candidate_visible_prefix)
                and len(candidate_visible_prefix) <= len(sanitized_replay_tail)
                and self._matches_store_tail_suffix(sanitized_replay_tail, candidate_visible_prefix)
            )
            matches_visible_raw_tail = (
                filtered_candidate_placeholders
                and bool(candidate_visible_prefix)
                and self._matches_store_tail_suffix(stored_tail, candidate_visible_prefix)
            )
            early_candidate_has_unrecoverable_persisted_marker = any(
                str(msg.get("role") or "") == "tool"
                and _is_hermes_persisted_output_marker(normalize_content_value(msg.get("content")) or "")
                and recover_hermes_persisted_output_with_file_stat(
                    normalize_content_value(msg.get("content")) or ""
                )
                is None
                for msg in candidate_identity_messages
            )
            if (matches_visible_sanitized_tail or matches_visible_raw_tail) and not early_candidate_has_unrecoverable_persisted_marker:
                return cursor
            candidate_has_persisted_marker = any(
                str(msg.get("role") or "") == "tool"
                and _is_hermes_persisted_output_marker(normalize_content_value(msg.get("content")) or "")
                for msg in candidate_identity_messages
            )
            matches_durable_persisted_output_full_replay = self._matches_persisted_output_durable_full_replay(
                candidate_identity_messages,
                candidate_prefix,
                stored_tail,
                stored_tail_rows,
            )
            candidate_has_unrecoverable_persisted_marker = any(
                str(msg.get("role") or "") == "tool"
                and _is_hermes_persisted_output_marker(normalize_content_value(msg.get("content")) or "")
                and recover_hermes_persisted_output_with_file_stat(
                    normalize_content_value(msg.get("content")) or ""
                )
                is None
                for msg in candidate_identity_messages
            )
            matches_inline_generation_cleanup_tail = False
            if candidate_has_unrecoverable_persisted_marker:
                generationless_sanitized_tail = [
                    self._strip_inline_persisted_output_generation_identity(identity)
                    for identity in sanitized_replay_tail
                ]
                generationless_candidate_prefix = [
                    self._strip_inline_persisted_output_generation_identity(identity)
                    for identity in candidate_prefix
                ]
                matches_inline_generation_cleanup_tail = self._matches_store_tail_suffix(
                    generationless_sanitized_tail,
                    generationless_candidate_prefix,
                )
            raw_tail_suffix = stored_tail[-len(candidate_prefix) :] if matches_raw_tail else []
            raw_suffix_needs_cleanup_equivalence = any(
                self._active_cleanup_replay_identity(identity) != identity
                for identity in raw_tail_suffix
            )
            if (
                not matches_sanitized_tail
                and not matches_raw_tail
                and not matches_inline_generation_cleanup_tail
                and not matches_durable_persisted_output_full_replay
            ):
                continue

            # Matching a stored suffix is not enough evidence by itself.  A
            # gateway restart may provide only newly arrived delta messages; if
            # the first delta happens to repeat the durable tail, treating that
            # row as replay silently loses it.  Only advance the cursor when the
            # incoming prefix proves replay by covering the full durable session.
            # A system prompt is a strong anchor. Older/minimal transcripts can
            # start directly with user/assistant turns, so multi-row full replay
            # is accepted only when active cleanup did not collapse the durable
            # tail; otherwise a fresh delta can repeat the remaining visible
            # suffix and must be preserved.
            candidate_has_system = any(identity[0] == "system" for identity in candidate_prefix)
            candidate_dropped_quarantine_replay_placeholder = any(
                self._is_volatile_ignored_quarantine_placeholder(
                    msg,
                    text_content_for_pattern_matching(msg.get("content")) or "",
                )
                or self._is_ignored_active_replay_placeholder(
                    msg,
                    text_content_for_pattern_matching(msg.get("content")) or "",
                )
                or (
                    self._compiled_ignore_message_patterns
                    and self._is_quarantined_assistant_replay_identity(
                        self._message_replay_identity(msg)
                    )
                    and self._matches_ignore_message_patterns(msg, stored_row=True)
                )
                for msg in candidate_messages
            )
            has_quarantined_singleton_replay = (
                matches_sanitized_tail
                and len(candidate_prefix) == 1
                and effective_session_count == 1
                and self._is_quarantined_assistant_replay_identity(candidate_prefix[0])
                and self._is_quarantined_assistant_replay_identity(sanitized_replay_tail[0])
            )
            candidate_singleton_original_content = (
                normalize_content_value(candidate_identity_messages[0].get("content")) or ""
                if len(candidate_identity_messages) == 1
                else ""
            )
            has_externalized_singleton_replay = (
                matches_raw_tail
                and len(candidate_prefix) == 1
                and raw_session_count == 1
                and bool(extract_externalized_ref(candidate_singleton_original_content))
                and _tail_tagless(candidate_prefix) == _tail_tagless(stored_tail)
            )
            has_persisted_marker_singleton_replay = (
                matches_raw_tail
                and not candidate_has_unrecoverable_persisted_marker
                and len(candidate_prefix) == 1
                and raw_session_count == 1
                and _tail_tagless(candidate_prefix) == _tail_tagless(stored_tail)
                and candidate_prefix[0][0] == "tool"
                and _is_hermes_persisted_output_marker(candidate_singleton_original_content)
            )
            has_durable_persisted_marker_suffix_replay = (
                (matches_sanitized_tail or matches_raw_tail)
                and any(
                    str(msg.get("role") or "") == "tool"
                    and _is_hermes_persisted_output_marker(normalize_content_value(msg.get("content")) or "")
                    and self._has_durable_persisted_output_replay_identity(msg)
                    for msg in candidate_messages
                )
            )
            has_filtered_full_replay = (
                matches_sanitized_tail
                and candidate_dropped_quarantine_replay_placeholder
                and len(candidate_prefix) >= effective_session_count
                and effective_session_count > 0
            )
            has_inline_generation_cleanup_replay = (
                matches_inline_generation_cleanup_tail
                and candidate_has_unrecoverable_persisted_marker
                and len(candidate_prefix) >= effective_session_count
                and effective_session_count > 0
            )
            has_inline_persisted_generation_suffix_replay = (
                matches_sanitized_tail
                and any(
                    str(msg.get("role") or "") == "tool"
                    and _is_hermes_persisted_output_marker(normalize_content_value(msg.get("content")) or "")
                    and _has_inline_persisted_output_generation_metadata(normalize_content_value(msg.get("content")) or "")
                    for msg in candidate_identity_messages
                )
            )
            if candidate_has_unrecoverable_persisted_marker:
                continue
            has_raw_persisted_marker_exact_replay = (
                candidate_has_persisted_marker
                and not candidate_has_unrecoverable_persisted_marker
                and matches_raw_tail
                and _tail_tagless(candidate_prefix) == _tail_tagless(stored_tail[-len(candidate_prefix) :])
            )
            has_persisted_marker_specific_replay_evidence = (
                not candidate_has_persisted_marker
                or has_durable_persisted_marker_suffix_replay
                or matches_durable_persisted_output_full_replay
                or has_inline_generation_cleanup_replay
                or has_inline_persisted_generation_suffix_replay
                or has_persisted_marker_singleton_replay
                or has_raw_persisted_marker_exact_replay
            )
            has_effective_full_replay = (
                has_persisted_marker_specific_replay_evidence
                and matches_sanitized_tail
                and len(candidate_prefix) >= effective_session_count
                and (
                    candidate_has_system
                    or (effective_session_count > 1 and not sanitized_tail_collapsed)
                    or has_quarantined_singleton_replay
                    or has_filtered_full_replay
                )
            )

            has_scaffold_evidence = any(
                self._is_replayed_context_scaffold_message(msg) for msg in candidate_messages
            )
            has_raw_full_replay = (
                has_persisted_marker_specific_replay_evidence
                and matches_raw_tail
                and not has_scaffold_evidence
                and len(candidate_messages) >= raw_session_count
                and raw_session_count > 1
            )
            has_preserved_objective_scaffold = any(
                str(msg.get("role") or "") != "system"
                and (normalize_content_value(msg.get("content")) or "").lstrip().startswith(
                    _PRESERVED_OBJECTIVE_CONTEXT_PREFIX
                )
                for msg in candidate_messages
            )
            candidate_suffix_has_user_turn = any(identity[0] == "user" for identity in candidate_prefix)
            has_scaffold_suffix_replay = (
                has_persisted_marker_specific_replay_evidence
                and matches_sanitized_tail
                and has_preserved_objective_scaffold
                and not candidate_suffix_has_user_turn
            )
            has_raw_cleanup_replay = (
                has_persisted_marker_specific_replay_evidence
                and matches_raw_tail
                and has_scaffold_evidence
                and cursor < len(messages)
                and len(candidate_prefix) >= max(1, effective_fresh_tail_count)
                and raw_suffix_needs_cleanup_equivalence
            )
            if (
                has_effective_full_replay
                or has_externalized_singleton_replay
                or has_persisted_marker_singleton_replay
                or has_durable_persisted_marker_suffix_replay
                or matches_durable_persisted_output_full_replay
                or has_inline_generation_cleanup_replay
                or has_inline_persisted_generation_suffix_replay
                or has_raw_full_replay
                or has_scaffold_suffix_replay
                or has_raw_cleanup_replay
            ):
                return cursor
        return empty_prefix_cursor if allow_empty_prefix else None

    def _record_ingest_reconciliation(
        self,
        *,
        action: str,
        reason: str,
        cursor: int,
        incoming: int,
        session_count: int,
        stored_tail_count: int,
        effective_incoming: int | None = None,
    ) -> None:
        self._last_ingest_reconciliation = {
            "action": action,
            "reason": reason,
            "cursor": cursor,
            "incoming": incoming,
            "session_count": session_count,
            "stored_tail_count": stored_tail_count,
        }
        if effective_incoming is not None:
            self._last_ingest_reconciliation["effective_incoming"] = effective_incoming

    def _effective_replay_identities(
        self,
        messages: List[Dict[str, Any]],
    ) -> list[tuple[str, str, str, str]]:
        return [
            self._message_replay_identity(msg)
            for msg in messages
            if not self._is_replayed_context_scaffold_message(msg)
            and not self._matches_ignore_message_patterns(msg)
        ]

    def _is_suspicious_stale_no_overlap_snapshot(
        self,
        incoming_identities: list[tuple[str, str, str, str]],
        stored_tail: list[tuple[str, str, str, str]],
        stored_head: list[tuple[str, str, str, str]],
    ) -> bool:
        """Return true for short stale snapshots with no durable-tail overlap.

        A restarted gateway can hand LCM a stale, short in-memory snapshot from
        the beginning of a longer session.  When that snapshot has no overlap
        with the durable tail, appending it as a delta creates duplicate rows.
        Fail closed only when the short batch is proven stale by matching the
        contiguous durable-store prefix; singleton no-overlap deltas remain
        ambiguous and are preserved.
        """
        if len(incoming_identities) <= 1:
            return False
        if incoming_identities[0][0] != "system":
            return False
        if not stored_tail or len(incoming_identities) >= len(stored_tail):
            return False
        # Shape-tag agnostic : stored rows are string-tagged
        # while live structured content is list/dict-tagged; staleness matching
        # is a CONTENT question, so compare tagless.
        def _tagless_identities(identities: list[tuple[str, str, str, str]]) -> list[tuple[str, str, str, str]]:
            return [
                (role, _strip_replay_identity_shape_tag(content), tool_call_id, tool_calls)
                for role, content, tool_call_id, tool_calls in identities
            ]
        tagless_incoming = _tagless_identities(incoming_identities)
        if set(tagless_incoming).intersection(_tagless_identities(stored_tail)):
            return False
        if len(tagless_incoming) > len(stored_head):
            return False
        return _tagless_identities(stored_head)[: len(tagless_incoming)] == tagless_incoming

    def _align_replayed_batch_against_stored_tail(
        self,
        messages: List[Dict[str, Any]],
        stored_rows: List[Dict[str, Any]],
    ) -> tuple[int, int, list[bool], bool]:
        """Mutation-tolerant replay alignment for the ambiguous-delta fallback.

        Reconcile's exact-match machinery needs byte identity between the
        replayed transcript and the CURRENT stored rows (the reconcile-duplication defect). When the
        stored tail was legitimately mutated post-ingest - host-side
        ``[Old tool output cleared to save context space]`` pruning,
        sensitive-redaction placeholders, externalization rewrites - that
        match fails and reconcile lands at cursor 0, and the per-turn
        whole-transcript replay re-persists the ENTIRE transcript every
        inbound message (the whole-transcript re-ingest defect's duplication, re-created through the
        reconcile-lands-at-0 path).

        This pass asks a weaker, mutation-stable question: is the incoming
        batch a REPLAY of (a superset of) the stored tail, or does it carry
        new durable turns? It walks TURN-level subsequence coverage: stored
        rows are grouped into turns — ``(role, tool_call_id)`` when the row
        carries a tool_call_id, ``(role, content)`` otherwise — and each
        turn, in first-occurrence order, must be carried by the incoming
        batch at-or-after the running floor: by the exact identity of ANY of
        its stored copies, or — when a copy's content was DESTROYED by a
        known post-ingest rewrite (cleared outputs, redaction /
        active-replay placeholders, GC'd externalized payloads) — by its
        surviving ``(role, tool_call_id)`` pair alone. Duplicate copies of
        an already-covered turn neither consume incoming occurrences nor
        count as coverage slots, so damage from earlier buggy bursts cannot
        starve the match (review finding).

        Limitation, by design: rows WITHOUT a tool_call_id whose content was
        rewritten (content IS those rows' only stable handle) cannot be
        aligned across the rewrite and fall back to the ambiguous append.

        Returns ``(covered, total, replay_mask, order_consistent)`` where
        ``covered`` counts the stored TURNS the batch carries, ``total`` is
        the turn count within the batch's replay SPAN (the window region
        from the oldest covered turn's copies onward - turns older than the
        batch's replay span are outside the verdict), ``replay_mask`` is a
        per-incoming-message boolean list marking the messages that replay
        a stored turn (source-time consistent), and ``order_consistent``
        reports whether the covered turns' first-occurrence positions are
        non-decreasing in stored-turn order (a reordered batch carrying the
        same identities is a fresh reordered batch, not a replay).
        """
        turn_keys: list[tuple[str, str]] = []
        turn_copy_positions: dict[tuple[str, str], list[int]] = {}
        turn_mutated: set[tuple[str, str]] = set()
        # Identity = (role, tool_call_id, tool_calls, content-presence,
        # content) — the store dedupe contract's own dimensions .
        stored_row_identities: list[tuple[str, str, str, str, str, str]] = []
        for row_idx, stored_row in enumerate(stored_rows):
            role = str(stored_row.get("role") or "unknown")
            tool_call_id = str(stored_row.get("tool_call_id") or "")
            content = normalize_content_value(stored_row.get("content")) or ""
            tool_calls_identity = self._stable_tool_calls_identity(
                stored_row.get("tool_calls")
            )
            tool_name = str(stored_row.get("tool_name") or "")
            content_presence = "" if stored_row.get("content") is None else "-"
            stored_row_identities.append(
                (role, tool_call_id, tool_calls_identity, tool_name, content_presence, content)
            )
            # Turn key = the FULL identity: empty-content assistant
            # tool-call turns are DISTINCT turns that share (role, '',
            # content) — collapsing them into one turn key masked only one
            # of the 967 live empty tool-call turns and left the rest
            # unmasked (the 51% masked fraction that failed the gate).
            # Byte-identical duplicate copies (the burst damage from earlier buggy passes
            # targets) still group correctly under the full identity; the
            # (role, tool_call_id) mutated fallback below handles copies
            # whose content was destroyed post-ingest.
            turn_key = stored_row_identities[-1]
            if turn_key not in turn_copy_positions:
                turn_keys.append(turn_key)
            turn_copy_positions.setdefault(turn_key, []).append(row_idx)
            if tool_call_id and _stored_row_content_post_ingest_mutated(content, role=role):
                turn_mutated.add(turn_key)

        incoming_identities = [
            (
                str(msg.get("role") or "unknown"),
                str(msg.get("tool_call_id") or ""),
                self._stable_tool_calls_identity(msg.get("tool_calls")),
                str(msg.get("tool_name") or ""),
                "" if msg.get("content") is None else "-",
                normalize_content_value(msg.get("content")) or "",
            )
            for msg in messages
        ]
        incoming_position_buckets: dict[tuple[str, str, str, str, str, str], list[int]] = {}
        incoming_call_position_buckets: dict[tuple[str, str], list[int]] = {}
        for msg_idx, identity in enumerate(incoming_identities):
            incoming_position_buckets.setdefault(identity, []).append(msg_idx)
            incoming_call_position_buckets.setdefault(
                (identity[0], identity[1]), []
            ).append(msg_idx)

        covered = 0
        covered_turn_keys: set = set()
        mutated_covered_calls: set[tuple[str, str]] = set()
        replay_mask = [False] * len(incoming_identities)

        # SURPLUS occurrences of a covered turn (the batch carries more
        # copies of a turn than the store holds — a damaged in-memory
        # transcript repeats turns) are masked ONLY when they are
        # source-time consistent with replay: their observed_at is None
        # (no source-time evidence at all) or not LATER than the stored
        # copies' source time. A surplus occurrence carrying a FRESH source
        # timestamp is a genuinely new turn (or a deliberate repeat) that
        # happens to reuse an identity — it must never be masked by
        # identity coincidence alone. The mutated fallback masks ONLY the
        # occurrence it consumed, never every payload sharing the call id
        # (a fresh retry with the same tool_call_id but distinct content is
        # new work).
        # Source-time lookup keyed by the SAME full identity the turn keys
        # use (the fresh-timestamp discipline looks turns up by identity;
        # a stale 2-tuple key would miss every lookup and mask fresh
        # identity-reuse rows).
        stored_turn_source_time: dict[tuple[str, str, str, str, str, str], float | None] = {}
        for row_idx, stored_row in enumerate(stored_rows):
            source_time = _normalize_observed_at(
                stored_row.get("observed_at")
                if stored_row.get("observed_at") is not None
                else stored_row.get("timestamp")
            )
            identity = stored_row_identities[row_idx]
            existing = stored_turn_source_time.get(identity)
            if existing is None or (source_time is not None and source_time > existing):
                stored_turn_source_time[identity] = source_time
        # Coverage = PURE EXISTENCE per turn key (the reconcile-duplication defect live follow-up):
        # a turn is covered when the batch carries it at all — by any
        # stored copy's exact identity, or (content-destroyed copies) by
        # the surviving (role, tool_call_id) pair. No monotone floor: the
        # store order of interleaved duplicate blocks does not match the
        # batch order, and a floor discipline starves legitimate coverage
        # (validated against the real 2068-row live burst: floor walks
        # covered 766/1507; existence covers 1507/1507). The replay-vs-
        # fresh distinction lives in the MASK (source-time consistency)
        # and the defer bars (coverage ratio, masked fraction, small-batch
        # and retry-signature stand-downs), not in the walk order.
        for turn_key in turn_keys:
            copy_positions = turn_copy_positions.get(turn_key, [])
            copy_identities = [stored_row_identities[row_idx] for row_idx in copy_positions]
            turn_source_time = stored_turn_source_time.get(turn_key)
            covered_here = False
            # Coverage requires at least one MASKED occurrence: a batch
            # occurrence whose only source-time evidence says FRESH (later
            # than the stored copy's) is new work - it neither proves the
            # turn replayed nor may be masked . Occurrences with no timestamp are
            # indistinguishable from replay (the incident's majority
            # class) and mask.
            # Exact-identity occurrences: any stored copy's identity.
            for identity in set(copy_identities):
                for msg_idx in incoming_position_buckets.get(identity, []):
                    observed = messages[msg_idx].get("timestamp")
                    candidate_source = _normalize_observed_at(observed)
                    if (
                        candidate_source is not None
                        and turn_source_time is not None
                        and candidate_source > turn_source_time
                    ):
                        continue  # fresh-timestamp repeat: new work
                    replay_mask[msg_idx] = True
                    covered_here = True
            if turn_key in turn_mutated:
                # Mutated copies: content destroyed post-ingest, so the
                # surviving (role, tool_call_id) pair is the only handle.
                # Mask ONE batch occurrence of this call id that is
                # source-time consistent with replay (the replayed original
                # whose stored copy was mutated away) - exactly one: the
                # batch can also carry a genuinely new retry sharing the
                # call id with NO timestamp (indistinguishable from replay
                # by time), and masking the whole bucket would silently
                # drop it (review finding). The retry then satisfies the
                # store guard's own retry semantics (it stores as new work).
                call_key = (turn_key[0], turn_key[1])
                for msg_idx in incoming_call_position_buckets.get(call_key, []):
                    observed = messages[msg_idx].get("timestamp")
                    candidate_source = _normalize_observed_at(observed)
                    if (
                        candidate_source is not None
                        and turn_source_time is not None
                        and candidate_source > turn_source_time
                    ):
                        continue
                    replay_mask[msg_idx] = True
                    covered_here = True
                    break
            if covered_here:
                covered += 1
                covered_turn_keys.add(turn_key)
                if turn_key in turn_mutated:
                    mutated_covered_calls.add((turn_key[0], turn_key[1]))
        # Span-relative coverage (review finding): the gate must measure
        # the turns the batch actually reaches, not the whole window - the
        # window (4x) includes rows older than the batch's replay span on
        # long sessions, and counting those turns sinks the coverage ratio.
        # The span starts at the oldest covered turn's earliest stored copy;
        # turns whose copies end before that span start are outside it.
        # The span is the BATCH-SPAN of the window : interleaved duplicate blocks mean every old turn's
        # copies reach the window's end, so "copies end after span_start"
        # includes turns the batch never carries. The batch can only replay
        # turns whose most recent stored copy lies within its own span —
        # the last len(messages) rows of the window. Those are the turns
        # the verdict measures.
        span_start = max(
            0,
            len(stored_rows) - len(messages),
        )
        # Numerator and denominator must measure the SAME set: turns whose
        # most recent stored copy lies within the batch-span. `covered`
        # counts only batch-span turns that the batch carries; turns older
        # than the span are outside the verdict entirely .
        span_turns = 0
        covered = 0
        for turn_key, copy_positions in turn_copy_positions.items():
            if not copy_positions or copy_positions[-1] < span_start:
                continue
            span_turns += 1
            if turn_key in covered_turn_keys:
                covered += 1
        # Order-consistency evidence (review finding): a batch that
        # carries the SAME identities as the stored tail but in a DIFFERENT
        # order is a reordered fresh batch, not a replay - pure existence
        # would pass the gates and silently discard it. The covered turns'
        # earliest consistent incoming positions must be non-decreasing in
        # stored-turn order. Mutated-covered turns contribute no position
        # (their order is unrecoverable - content destroyed).
        # Order consistency with a bounded tolerance: a damaged store's key
        # order diverges from transcript order (compaction assemblies,
        # ignore-filtered rows shift positions), so an occasional backward
        # step is historical disorder, not evidence of a reordered batch.
        # A genuinely REORDERED fresh batch violates pervasively (every
        # adjacent pair), so the verdict tolerates only a small fraction of
        # violations relative to the covered turns.
        violations = 0
        checks = 0
        prev_position = -1
        for tk in turn_keys:
            # Order evidence only from batch-span turns : covered turns whose last copy predates
            # span_start are outside the verdict; their stale positions
            # would distort the check.
            copy_positions = turn_copy_positions.get(tk, [])
            if not copy_positions or copy_positions[-1] < span_start:
                continue
            if tk not in covered_turn_keys:
                continue
            positions = [
                msg_idx
                for identity in {
                    stored_row_identities[ri] for ri in turn_copy_positions[tk]
                }
                for msg_idx in incoming_position_buckets.get(identity, [])
                if replay_mask[msg_idx]
            ]
            if not positions:
                continue
            checks += 1
            first_pos = min(positions)
            if first_pos < prev_position:
                violations += 1
            prev_position = first_pos
        # Tolerance is strictly proportional (review finding): a small
        # batch's complete reversal (2 violations in 2 checks) must fail,
        # so the floor is 1 violation, not 2.
        # Small batches tolerate ZERO inversions: a 3-row store replayed as
        # A,C,B produces 1 violation in 3 checks, and every reordered row is
        # untimestamped fresh content the lossless contract must keep
        # (review finding). Large replays keep the proportional
        # tolerance: a damaged store's key order diverges from transcript
        # order (compaction assemblies, ignore-filter shifts), so isolated
        # backward steps are historical disorder, not reordering.
        if checks <= 10:
            order_consistent = violations == 0
        else:
            order_consistent = violations <= max(1, checks // 50)
        # Expose the mutated-covered call keys: turns whose ONLY alignment
        # handle was the surviving (role, tool_call_id) pair because their
        # stored content was destroyed post-ingest. The store guard can
        # NEVER dedupe a replayed original against those rows (the stored
        # copy's content was rewritten), so the engine must mask-drop them
        # even when they carry a usable source timestamp.
        self._deferred_replay_alignment_mutated_calls = mutated_covered_calls
        return covered, span_turns, replay_mask, order_consistent

    def _replay_alignment_gate(
        self,
        messages: List[Dict[str, Any]],
        covered: int,
        span_turns: int,
        replay_mask: list[bool],
        order_consistent: bool,
    ) -> bool:
        """Shared defer-verdict math for the replay alignment.

        Defer (mask authoritative) requires: a multi-row batch (>= 3),
        near-full coverage of the replay span's turns (>= 90% - a live
        transcript replays its history plus a small fresh tail), a dominant
        masked fraction over the effective (non-ignored) incoming rows
        (>= 75%), order consistency, and the retry-batch stand-down (an
        unmasked tool persisted-output marker / externalized ref means the
        batch is deliberate retry traffic - it must append).
        """
        # The minimum batch size applies to EFFECTIVE (non-ignored) rows:
        # an ignored row is discarded before storage and must not inflate a
        # 2-row deliberate-repeat batch into "multi-row" (review finding).
        effective_rows = sum(
            1
            for msg in messages
            if not self._matches_ignore_message_patterns(msg)
        )
        if effective_rows < 3 or span_turns <= 0:
            return False
        if covered < (span_turns * 9 + 9) // 10:
            return False
        effective_incoming = sum(
            1
            for msg in messages
            if not self._matches_ignore_message_patterns(msg)
        )
        masked_effective = sum(
            1
            for msg_idx in range(len(messages))
            if replay_mask[msg_idx]
            and not self._matches_ignore_message_patterns(messages[msg_idx])
        )
        masked_fraction = (
            masked_effective / effective_incoming if effective_incoming else 0.0
        )
        if masked_fraction < 0.75:
            return False
        if not order_consistent:
            return False
        for msg_idx in range(len(messages)):
            if replay_mask[msg_idx]:
                continue
            msg = messages[msg_idx]
            if str(msg.get("role") or "") != "tool":
                continue
            content = normalize_content_value(msg.get("content")) or ""
            if _is_hermes_persisted_output_marker(content) or (
                extract_externalized_ref(content) is not None
            ):
                return False
        return True

    def _reconcile_ingest_cursor_from_store(self, messages: List[Dict[str, Any]]) -> int:
        """Infer the in-memory cursor for an existing session after process restart."""
        # A previous pass's alignment mask must never leak into this one:
        # early returns below (empty session, count failure, cursor
        # advance) do not all recompute it, and a stale mask from another
        # session's batch would silently drop matching rows here.
        self._deferred_replay_alignment_mask = []
        if not self._session_id or not messages:
            return 0

        try:
            session_count = self._store.get_session_count(self._session_id)
        except Exception as exc:  # pragma: no cover - defensive only
            logger.debug("LCM ingest cursor reconciliation count failed: %s", exc)
            return 0
        if session_count <= 0:
            placeholder_budget = self._load_generated_ignored_placeholder_hash_counts()
            placeholder_ordinals = self._load_generated_ignored_placeholder_hash_ordinals()
            if placeholder_budget and placeholder_ordinals:
                consumed: dict[str, int] = {}
                cursor = 0
                for msg in messages:
                    text = text_content_for_pattern_matching(msg.get("content")) or ""
                    digest = self._active_replay_placeholder_digest(text)
                    if not digest:
                        break
                    consumed[digest] = consumed.get(digest, 0) + 1
                    ordinal = consumed[digest]
                    remaining = int(placeholder_budget.get(digest, 0) or 0)
                    if remaining <= 0 or ordinal not in placeholder_ordinals.get(digest, set()):
                        break
                    cursor += 1
                if cursor > 0:
                    self._record_ingest_reconciliation(
                        action="advanced cursor",
                        reason="replayed generated placeholders in empty session",
                        cursor=cursor,
                        incoming=len(messages),
                        session_count=session_count,
                        stored_tail_count=0,
                        effective_incoming=cursor,
                    )
                    return cursor
            return 0

        # The alignment window must match the batch's replay SPAN (the reconcile-duplication defect
        # The alignment window keeps the 4x factor: it must include ALL
        # stored copies of the turns the batch replays (a damaged store
        # holds whole duplicate blocks), while span-relative coverage
        # measures the gate over the turns the batch actually reaches
        # (from the oldest covered turn's copies onward), so turns older
        # than the batch's replay span don't sink the verdict
        # (review finding).
        tail_limit = min(max(len(messages) * 4, 64), session_count)
        stored_rows = self._store.get_session_tail(self._session_id, limit=tail_limit)
        if not stored_rows:
            return 0
        stored_tail_rows = [
            row
            for row in stored_rows
            if not self._matches_ignore_message_patterns(row, stored_row=True)
        ]
        stored_tail = [
            self._message_replay_identity(row, stored_row=True)
            for row in stored_tail_rows
        ]
        cursor = self._find_reconciled_cursor_for_store_tail(
            messages,
            stored_tail,
            stored_tail_rows=stored_tail_rows,
            allow_empty_prefix=True,
            session_count=len(stored_tail),
            raw_session_count=session_count,
        )
        if cursor is not None and cursor > 0:
            reason = (
                "skipped scaffold-only prefix"
                if not self._effective_replay_identities(messages[:cursor])
                else "replayed durable tail"
            )
            # The cursor advance covers the FIRST copy of each replayed
            # turn, but a damaged host transcript can repeat turns WITHIN
            # one batch (live: the affected live session burst carried ~3
            # copies of every turn + fresh suffix). The copies past the
            # cursor ride messages[cursor:] into the store with the guard
            # OFF (reconcile decided), and the guard could not dedupe them
            # anyway (no observed_at). Compute the alignment mask on EVERY
            # reconcile pass so the engine can drop those aligned repeats
            # past the cursor; genuinely new turns stay unmasked.
            _cov, _turn_total, replay_mask, _order_ok = (
                self._align_replayed_batch_against_stored_tail(
                    messages,
                    stored_tail_rows,
                )
            )
            # The mask is AUTHORITATIVE for drops past the cursor only when
            # the advance was a full-replay proof ("replayed durable tail"):
            # then the batch provably replayed the durable session and its
            # aligned repeats past the cursor are replay copies. A
            # scaffold-only-prefix skip says nothing about the delta rows —
            # a delta matching the tail tip is a DELIBERATE append (the
            # lossless-first contract) and must store, so the mask stands
            # down there. It ALSO stands down when the past-cursor portion
            # carries a tool persisted-output marker or externalized ref:
            # those are the retry signature (the store guard exempts them,
            # _dedupe_replay_applies — retry semantics live above the
            # store), so the batch is deliberate retry traffic and
            # everything past the cursor re-appends (TestIngestExternalization).
            # The mask is authoritative past the cursor when the advance
            # was a full-replay proof ("replayed durable tail") OR when the
            # scaffold-only-prefix advance carries a batch whose POST-CURSOR
            # region provably replays the durable tail (live 21:56 burst:
            # reconcile advanced cursor=1 past one scaffold row on a
            # 2147-row whole-transcript replay, the mask stood down, and
            # the guard-off append re-persisted everything). Run the shared
            # gate on the alignment result for the scaffold path too: when
            # it passes, the mask governs the post-cursor region; when it
            # fails, the small delta keeps the deliberate-append behavior.
            # The retry-signature stand-down (markers/externalized refs in
            # the post-cursor region) applies on both paths.
            post_cursor = messages[cursor:]
            retry_signature = any(
                str(msg.get("role") or "") == "tool"
                and (
                    _is_hermes_persisted_output_marker(
                        normalize_content_value(msg.get("content")) or ""
                    )
                    or extract_externalized_ref(
                        normalize_content_value(msg.get("content")) or ""
                    )
                    is not None
                )
                for msg in post_cursor
            )
            if reason == "replayed durable tail":
                self._deferred_replay_alignment_mask = (
                    [] if retry_signature else replay_mask
                )
            else:
                # The gate reads the POST-CURSOR region only: slice the mask
                # at the cursor so its fraction/retry checks index
                # post-cursor bits, not the scaffold prefix's .
                post_cursor_mask = replay_mask[cursor:]
                if not retry_signature and self._replay_alignment_gate(
                    post_cursor,
                    _cov,
                    _turn_total,
                    post_cursor_mask,
                    _order_ok,
                ):
                    self._deferred_replay_alignment_mask = replay_mask
                else:
                    self._deferred_replay_alignment_mask = []
            self._record_ingest_reconciliation(
                action="advanced cursor",
                reason=reason,
                cursor=cursor,
                incoming=len(messages),
                session_count=session_count,
                stored_tail_count=len(stored_tail),
                effective_incoming=len(self._effective_replay_identities(messages)),
            )
            logger.debug(
                "LCM reconciled ingest cursor after existing-session bind: session=%s cursor=%d incoming=%d stored_tail=%d session_count=%d reason=%s",
                self._session_id,
                cursor,
                len(messages),
                len(stored_tail),
                session_count,
                reason,
            )
            return cursor

        incoming_identities = self._effective_replay_identities(messages)
        # The stale-snapshot proof keeps the WIDE window (4x): it compares a
        # short system-leading prefix snapshot against the session HEAD, and
        # the snapshot-no-overlap bail-out depends on the length guard —
        # shrinking it with the alignment window broke that proof . Only the replay ALIGNMENT uses the batch-span window.
        head_window_limit = min(max(len(messages) * 4, 64), session_count)
        stored_head_rows = self._store.get_session_messages(
            self._session_id,
            limit=head_window_limit,
        )
        stored_head = [self._message_replay_identity(row, stored_row=True) for row in stored_head_rows]
        # Stale-snapshot proof uses the raw durable prefix.  Ignore-message
        # filters may suppress noisy rows for tail reconciliation, but filtered
        # history alone must not create replay evidence for skipping a batch.
        incoming_has_unproofed_raw_persisted_marker = any(
            str(msg.get("role") or "") == "tool"
            and _is_hermes_persisted_output_marker(normalize_content_value(msg.get("content")) or "")
            and recover_hermes_persisted_output_with_file_stat(
                normalize_content_value(msg.get("content")) or ""
            )
            is None
            for msg in messages
        )
        if (
            not incoming_has_unproofed_raw_persisted_marker
            and self._is_suspicious_stale_no_overlap_snapshot(
                incoming_identities,
                stored_tail,
                stored_head,
            )
        ):
            self._record_ingest_reconciliation(
                action="skipped batch",
                reason="skipped stale no-overlap snapshot",
                cursor=len(messages),
                incoming=len(messages),
                session_count=session_count,
                stored_tail_count=len(stored_tail),
                effective_incoming=len(incoming_identities),
            )
            logger.warning(
                "LCM skipped stale no-overlap snapshot after existing-session bind: session=%s incoming=%d effective_incoming=%d stored_tail=%d session_count=%d",
                self._session_id,
                len(messages),
                len(incoming_identities),
                len(stored_tail),
                session_count,
            )
            return len(messages)

        # The reconcile-duplication defect trust flip: cursor=0 here means reconcile found NO match -
        # not even a partial prefix - which is exactly the mutated-tail
        # signature. The deliberate-append cases that motivated reconcile's
        # authority all matched a PREFIX and advanced >0; a zero cursor over a
        # non-empty store is un-decided, so hand the batch to the store-level
        # dedupe-replay guard instead of re-persisting blindly. Before
        # standing down, run a mutation-tolerant alignment as a fail-closed
        # second gate (the host replays whole transcripts whose stored copy
        # was pruned/redacted post-ingest, which byte-identity matching can
        # never see): a batch that is a replay of (a superset of) the stored
        # tail must not be re-persisted even if the guard below were
        # unavailable. New durable turns (alignment coverage below the
        # threshold) still append - ambiguity is preserved for genuinely
        # ambiguous small batches.
        #
        # The deliberate-append encodings are the small-batch boundary: a
        # 1-2 row batch whose content coincides with the tail tip is
        # indistinguishable from a deliberate repeat of the last turn(s) and
        # MUST append (reconcile's lossless-first contract). A whole-
        # transcript replay is a different shape entirely: many incoming
        # rows covering the stored tail. Defer therefore requires a
        # multi-row batch (>= 3 incoming rows), NEAR-FULL alignment coverage
        # of the stored turns (a live transcript replays its whole history
        # plus a small tail of genuinely new turns, so exact 100% coverage
        # is the wrong bar — >= 90% of stored turns covered is the replay
        # shape; the uncovered turns are precisely the new turns, which the
        # mask leaves alone), AND a dominant masked fraction (>= 75% of the
        # incoming rows masked — a replay-shaped batch is almost entirely
        # durable-turn copies; a fresh delta like [A,B,C] over a 1-row tail
        # covers its one turn but masks only 1/3 and stays ambiguous).
        # Partial coverage (a retry replay whose exempt tool rows
        # deliberately do not align, a sanitized-tail delta) fails these
        # bars and stays with the ambiguous-delta append: the uncovered rows
        # are exactly the content the lossless contract must keep.
        ambiguous_unaligned = True
        if len(stored_tail) > 0 and len(messages) >= 3:
            covered, stored_total, replay_mask, order_consistent = self._align_replayed_batch_against_stored_tail(
                messages,
                stored_tail_rows,
            )
            ambiguous_unaligned = not self._replay_alignment_gate(
                messages,
                covered,
                stored_total,
                replay_mask,
                order_consistent,
            )
            # Retry-batch stand-down (same rule as the cursor-advance
            # branch): a batch whose UNMASKED rows include a tool
            # persisted-output marker or externalized ref is deliberate
            # RETRY traffic — the guard exempts those classes (retry
            # semantics live above the store), so the whole batch must
            # append rather than defer-with-a-few-rows-left (e.g. nine
            # aligned rows + one raw marker whose backing file vanished is
            # a retry batch, not a replay with leftovers).
            if not ambiguous_unaligned and any(
                not replay_mask[msg_idx]
                and str(messages[msg_idx].get("role") or "") == "tool"
                and (
                    _is_hermes_persisted_output_marker(
                        normalize_content_value(messages[msg_idx].get("content")) or ""
                    )
                    or extract_externalized_ref(
                        normalize_content_value(messages[msg_idx].get("content")) or ""
                    )
                    is not None
                )
                for msg_idx in range(len(messages))
            ):
                ambiguous_unaligned = True
            # The mask is authoritative ONLY under the defer verdict. An
            # ambiguous-append decision (below the bars) deliberately keeps
            # every row — including repeats — so the mask must stand down,
            # or the engine would drop aligned rows the lossless contract
            # says to append (TestIngestExternalization retry replays).
            self._deferred_replay_alignment_mask = (
                [] if ambiguous_unaligned else replay_mask
            )
        else:
            self._deferred_replay_alignment_mask = []
        if ambiguous_unaligned:
            self._record_ingest_reconciliation(
                action="persisted batch",
                reason="persisted ambiguous delta",
                cursor=0,
                incoming=len(messages),
                session_count=session_count,
                stored_tail_count=len(stored_tail),
                effective_incoming=len(incoming_identities),
            )
            return 0
        self._record_ingest_reconciliation(
            action="deferred to replay guard",
            reason="ambiguous zero cursor over non-empty store",
            cursor=0,
            incoming=len(messages),
            session_count=session_count,
            stored_tail_count=len(stored_tail),
            effective_incoming=len(incoming_identities),
        )
        return None

    def _raw_externalized_placeholder_replay_identity(self, msg: Dict[str, Any]) -> tuple[str, str, str, str]:
        return (
            str(msg.get("role") or "unknown"),
            normalize_content_value(msg.get("content")) or "",
            self._stable_tool_calls_identity(msg.get("tool_calls")),
            str(msg.get("tool_call_id") or ""),
        )

    def _get_store_id_map_for_messages(self, messages: List[Dict[str, Any]]) -> dict[int, int]:
        """Map current raw message objects back to store_ids in stable order.

        Matching starts strictly after ``_last_compacted_store_id`` so repeated
        content from older already-compacted history cannot hijack the mapping.
        Synthetic summary messages simply fail to match and are skipped.  When
        active context has more occurrences of an identical replay identity than
        the store has, the surplus earliest active occurrences are treated as
        synthetic/carry-over and left unmapped so they cannot steal later stored
        literal copies with the same content.
        """
        candidates: list[Dict[str, Any]] = []
        next_candidate_after = self._last_compacted_store_id
        while True:
            page = self._store.get_session_messages_after(
                self._session_id,
                after_store_id=next_candidate_after,
            )
            if not page:
                break
            candidates.extend(page)
            next_candidate_after = page[-1]["store_id"]
        def _tagless(identity: tuple[Any, ...]) -> tuple[Any, ...]:
            return (identity[0], _strip_replay_identity_shape_tag(identity[1]), identity[2], identity[3])

        active_identity_counts: dict[tuple[Any, ...], int] = {}
        for msg in messages:
            identity = _tagless(self._message_replay_identity(msg))
            active_identity_counts[identity] = active_identity_counts.get(identity, 0) + 1
        stored_identity_counts: dict[tuple[Any, ...], int] = {}
        stored_cleanup_identity_counts: dict[tuple[Any, ...], int] = {}
        # Capture each candidate's identity (and its cleanup variant) here - both
        # are already computed for the counts below, so this adds no work. The
        # match-probe loops reuse them instead of recomputing
        # _message_replay_identity(stored_row=True) for every (message, probe)
        # pair. That call is expensive when a stored row carries an externalized
        # payload (JSON canonicalization + a payload-file read), so eliminating
        # the O(candidates^2) recomputes removes repeated disk reads on
        # tool-output-heavy histories. Raw-placeholder identities stay lazy (see
        # the memo below) since most rows never need them.
        # The store-id map matches rows to messages by CONTENT identity; the
        # shape tag is a live-vs-claim distinction (handoff digest), not a row
        # identity, and a stored row's tag (string for text) can differ from
        # the live message's tag (list for structured content). Strip the tag
        # on both sides WITHIN THIS MAP ONLY so structured content still maps
        # to its row; claim/digest paths keep the tagged identity.
        stored_identities: list[tuple[Any, ...]] = []
        stored_cleanup_identities: list[Optional[tuple[Any, ...]]] = []
        for stored in candidates:
            identity = _tagless(self._message_replay_identity(stored, stored_row=True))
            stored_identities.append(identity)
            cleanup_identity = self._active_cleanup_replay_identity(identity, content_is_tagged=False)
            stored_cleanup_identities.append(cleanup_identity)
            stored_identity_counts[identity] = stored_identity_counts.get(identity, 0) + 1
            if cleanup_identity is not None:
                stored_cleanup_identity_counts[cleanup_identity] = (
                    stored_cleanup_identity_counts.get(cleanup_identity, 0) + 1
                )

        # Lazily memoize raw-placeholder identities: only the placeholder-ref
        # paths need them, and most histories have few (or none), so computing
        # them on demand keeps the common case free.
        _raw_placeholder_identity_cache: dict[int, tuple[str, str, str, str]] = {}

        def stored_raw_placeholder_identity(probe_idx: int) -> tuple[str, str, str, str]:
            cached = _raw_placeholder_identity_cache.get(probe_idx)
            if cached is None:
                cached = self._raw_externalized_placeholder_replay_identity(candidates[probe_idx])
                _raw_placeholder_identity_cache[probe_idx] = cached
            return cached
        active_surplus_skips: dict[tuple[Any, ...], int] = {}
        generated_surplus_skip_message_ids: set[int] = set()
        generated_placeholder_message_ids = getattr(
            self,
            "_generated_ignored_active_replay_placeholder_message_ids",
            set(),
        )
        for identity, active_count in active_identity_counts.items():
            wanted_cleanup_identity = self._active_cleanup_replay_identity(identity, content_is_tagged=False)
            stored_exact = stored_identity_counts.get(identity, 0)
            stored_cleanup = 0
            if wanted_cleanup_identity is not None:
                stored_cleanup = stored_cleanup_identity_counts.get(wanted_cleanup_identity, 0)
            stored_available = max(stored_exact, stored_cleanup)
            if active_count > stored_available:
                surplus_count = active_count - stored_available
                for msg in messages:
                    if surplus_count <= 0:
                        break
                    if id(msg) not in generated_placeholder_message_ids:
                        continue
                    if _tagless(self._message_replay_identity(msg)) != identity:
                        continue
                    generated_surplus_skip_message_ids.add(id(msg))
                    surplus_count -= 1
                if surplus_count > 0:
                    active_surplus_skips[identity] = surplus_count

        placeholder_identity_counts: dict[tuple[str, str, str, str], int] = {}
        for msg in messages:
            msg_content = normalize_content_value(msg.get("content")) or ""
            if msg.get("store_id") is None and self._content_has_externalized_placeholder_ref(msg_content):
                raw_identity = self._raw_externalized_placeholder_replay_identity(msg)
                placeholder_identity_counts[raw_identity] = placeholder_identity_counts.get(raw_identity, 0) + 1
        self._current_compress_placeholder_identity_counts = placeholder_identity_counts

        def find_raw_placeholder_match_index(
            raw_identity: tuple[str, str, str, str],
            start_idx: int,
        ) -> int | None:
            probe_idx = start_idx
            while probe_idx < len(candidates):
                if stored_raw_placeholder_identity(probe_idx) == raw_identity:
                    return probe_idx
                probe_idx += 1
            return None

        def find_message_match_index(msg: Dict[str, Any], start_idx: int) -> int | None:
            msg_content = normalize_content_value(msg.get("content")) or ""
            if msg.get("store_id") is None and self._content_has_externalized_placeholder_ref(msg_content):
                raw_identity = self._raw_externalized_placeholder_replay_identity(msg)
                raw_match_idx = find_raw_placeholder_match_index(raw_identity, start_idx)
                if raw_match_idx is not None:
                    return raw_match_idx

            message_identity = _tagless(self._message_replay_identity(msg))
            wanted_cleanup_identity = self._active_cleanup_replay_identity(
                message_identity, content_is_tagged=False
            )
            probe_idx = start_idx
            while probe_idx < len(candidates):
                stored_identity = stored_identities[probe_idx]
                if stored_identity == message_identity:
                    return probe_idx
                if (
                    wanted_cleanup_identity is not None
                    and stored_cleanup_identities[probe_idx] == wanted_cleanup_identity
                ):
                    return probe_idx
                probe_idx += 1
            return None

        def matched_remaining_message_ids(
            message_start_idx: int,
            start_store_idx: int,
            surplus_skips: dict[tuple[Any, ...], int],
        ) -> set[int]:
            matched_message_ids: set[int] = set()
            local_surplus_skips = dict(surplus_skips)
            probe_idx = start_store_idx
            for remaining_msg in messages[message_start_idx:]:
                msg_content = normalize_content_value(remaining_msg.get("content")) or ""
                if (
                    remaining_msg.get("store_id") is None
                    and self._content_has_externalized_placeholder_ref(msg_content)
                ):
                    raw_identity = self._raw_externalized_placeholder_replay_identity(remaining_msg)
                    raw_match_idx = find_raw_placeholder_match_index(raw_identity, probe_idx)
                    if raw_match_idx is not None:
                        matched_message_ids.add(id(remaining_msg))
                        probe_idx = raw_match_idx + 1
                        continue
                message_identity = _tagless(self._message_replay_identity(remaining_msg))
                if id(remaining_msg) in generated_surplus_skip_message_ids:
                    continue
                surplus = local_surplus_skips.get(message_identity, 0)
                if surplus > 0:
                    local_surplus_skips[message_identity] = surplus - 1
                    continue
                match_idx = find_message_match_index(remaining_msg, probe_idx)
                if match_idx is None:
                    continue
                matched_message_ids.add(id(remaining_msg))
                probe_idx = match_idx + 1
            return matched_message_ids

        ids_by_message_id: dict[int, int] = {}
        store_idx = 0
        for msg_idx, msg in enumerate(messages):
            msg_content = normalize_content_value(msg.get("content")) or ""
            if msg.get("store_id") is None and self._content_has_externalized_placeholder_ref(msg_content):
                raw_identity = self._raw_externalized_placeholder_replay_identity(msg)
                if placeholder_identity_counts.get(raw_identity, 0) > 1:
                    match_idx = find_raw_placeholder_match_index(raw_identity, store_idx)
                    if match_idx is not None:
                        ids_by_message_id[id(msg)] = candidates[match_idx]["store_id"]
                        store_idx = match_idx + 1
                else:
                    # Prefer a later duplicate only when it does not orphan
                    # later active messages that still need monotonic mapping.
                    first_match_idx = find_raw_placeholder_match_index(raw_identity, store_idx)
                    if first_match_idx is not None:
                        baseline_suffix_ids = matched_remaining_message_ids(
                            msg_idx + 1,
                            first_match_idx + 1,
                            active_surplus_skips,
                        )
                    else:
                        baseline_suffix_ids = set()
                    probe_idx = len(candidates) - 1
                    while first_match_idx is not None and probe_idx >= first_match_idx:
                        stored = candidates[probe_idx]
                        if stored_raw_placeholder_identity(probe_idx) == raw_identity:
                            candidate_suffix_ids = matched_remaining_message_ids(
                                msg_idx + 1,
                                probe_idx + 1,
                                active_surplus_skips,
                            )
                            if not baseline_suffix_ids.issubset(candidate_suffix_ids):
                                probe_idx -= 1
                                continue
                            ids_by_message_id[id(msg)] = stored["store_id"]
                            store_idx = probe_idx + 1
                            break
                        probe_idx -= 1
                if id(msg) in ids_by_message_id:
                    continue
            message_identity = _tagless(self._message_replay_identity(msg))
            if id(msg) in generated_surplus_skip_message_ids:
                continue
            surplus = active_surplus_skips.get(message_identity, 0)
            if surplus > 0:
                active_surplus_skips[message_identity] = surplus - 1
                continue
            match_idx = find_message_match_index(msg, store_idx)
            if match_idx is not None:
                ids_by_message_id[id(msg)] = candidates[match_idx]["store_id"]
                store_idx = match_idx + 1

        return ids_by_message_id
