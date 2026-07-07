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

import logging

logger = logging.getLogger(__name__)

_PRESERVED_OBJECTIVE_CONTEXT_PREFIX = "[Current user objective preserved from compacted history]"


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
        content = normalize_content_value(msg.get("content")) or ""
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
        tool_calls_identity = self._stable_tool_calls_identity(tool_calls)
        return (
            role,
            content,
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
        return stored_tail[-len(candidate_prefix) :] == candidate_prefix

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
            transformed_candidate.append(candidate_identity)
            transformed_stored.append(stored_identity)
        return saw_persisted_output and transformed_candidate == transformed_stored

    @classmethod
    def _identity_content_for_active_cleanup(cls, content: str) -> Any:
        """Decode canonical stored JSON content before active-cleanup checks.

        Structured assistant content is persisted as deterministic JSON. Active
        replay cleanup sees the original list/dict shape, so restart
        reconciliation has to decode the stored identity before deciding whether
        a durable assistant row could be absent from sanitized active context.
        """
        if not isinstance(content, str):
            return content
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
    ) -> tuple[str, str, str, str] | None:
        role, content, tool_call_id, tool_calls = identity
        if role != "assistant":
            return identity
        msg: dict[str, Any] = {
            "role": role,
            "content": cls._identity_content_for_active_cleanup(content),
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
        return (
            role,
            normalize_content_value(cleaned.get("content")) or "",
            tool_call_id,
            tool_calls,
        )

    @staticmethod
    def _is_quarantined_assistant_replay_identity(identity: tuple[str, str, str, str]) -> bool:
        role, content, _tool_call_id, _tool_calls = identity
        if role != "assistant":
            return False
        text = str(content or "").strip()
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
                and candidate_prefix == stored_tail
            )
            has_persisted_marker_singleton_replay = (
                matches_raw_tail
                and not candidate_has_unrecoverable_persisted_marker
                and len(candidate_prefix) == 1
                and raw_session_count == 1
                and candidate_prefix == stored_tail
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
                and candidate_prefix == stored_tail[-len(candidate_prefix) :]
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
                and len(candidate_prefix) >= max(1, self._config.fresh_tail_count)
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
        if set(incoming_identities).intersection(stored_tail):
            return False
        if len(incoming_identities) > len(stored_head):
            return False
        return stored_head[: len(incoming_identities)] == incoming_identities

    def _reconcile_ingest_cursor_from_store(self, messages: List[Dict[str, Any]]) -> int:
        """Infer the in-memory cursor for an existing session after process restart."""
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
        stored_head_rows = self._store.get_session_messages(
            self._session_id,
            limit=tail_limit,
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
        active_identity_counts: dict[tuple[Any, ...], int] = {}
        for msg in messages:
            identity = self._message_replay_identity(msg)
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
        stored_identities: list[tuple[Any, ...]] = []
        stored_cleanup_identities: list[Optional[tuple[Any, ...]]] = []
        for stored in candidates:
            identity = self._message_replay_identity(stored, stored_row=True)
            stored_identities.append(identity)
            cleanup_identity = self._active_cleanup_replay_identity(identity)
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
            wanted_cleanup_identity = self._active_cleanup_replay_identity(identity)
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
                    if self._message_replay_identity(msg) != identity:
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

            message_identity = self._message_replay_identity(msg)
            wanted_cleanup_identity = self._active_cleanup_replay_identity(message_identity)
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
                message_identity = self._message_replay_identity(remaining_msg)
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
            message_identity = self._message_replay_identity(msg)
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
