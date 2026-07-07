"""Ignored-active-replay placeholder ledger for the LCM engine (WS5 seam).

The ``PlaceholderLedgerMixin`` holds the bookkeeping for ignored-active-replay
placeholders and dependent-reply records: session-scoped metadata-key builders,
generated-placeholder hash/count/ordinal load+persist, active-replay digest
budgets, dependent-reply fingerprints/records, and the placeholder application
pass. These methods were lifted verbatim out of ``LCMEngine`` and continue to
run bound to the engine instance (``self`` is the ``LCMEngine``), so they read
and write the engine's shared runtime state (``_store``, ``_session_id``, the
two ``_generated_ignored_active_replay_placeholder_*`` sets, the per-turn
``_current_compress_store_ids_by_message_id`` cache) and call back into engine
helpers (``_get_store_id_map_for_messages``, ``_matches_ignore_message_patterns``,
``_stable_tool_calls_identity``, ``_message_replay_identity``,
``_copy_active_replay_messages_preserving_generated_ids``) through normal
attribute lookup. ``LCMEngine`` mixes this in, so no call site changes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any, Dict, List, Optional

from .message_content import text_content_for_pattern_matching

logger = logging.getLogger(__name__)


class PlaceholderLedgerMixin:
    @staticmethod
    def _is_volatile_ignored_quarantine_placeholder(msg: Dict[str, Any], text: str) -> bool:
        if str(msg.get("role") or "") != "assistant":
            return False
        return bool(
            re.fullmatch(
                r"\[LCM active replay placeholder: assistant output quarantined; "
                r"kind=quarantined_assistant_output; "
                r"reason=[A-Za-z0-9_.:/-]+; "
                r"scope=ignored_message_pattern; field=content; "
                r"chars=\d+; bytes=\d+; "
                r"sha256=[0-9a-f]{16}\]",
                text.strip(),
            )
        )

    @staticmethod
    def _active_replay_placeholder_digest(text: str) -> Optional[str]:
        match = re.search(r"sha256=([0-9a-f]{16})\]$", text.strip())
        return match.group(1) if match else None

    @staticmethod
    def _ignored_active_replay_placeholder(content: str) -> str:
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        return (
            "[LCM active replay placeholder: message ignored; "
            "kind=ignored_message; "
            "scope=ignored_message_pattern; field=content; "
            f"chars={len(content)}; bytes={len(content.encode('utf-8'))}; "
            f"sha256={digest}]"
        )

    def _is_ignored_active_replay_placeholder(self, msg: Dict[str, Any], text: str) -> bool:
        match = re.fullmatch(
                r"\[LCM active replay placeholder: message ignored; "
                r"kind=ignored_message; "
                r"scope=ignored_message_pattern; field=content; "
                r"chars=\d+; bytes=\d+; "
                r"sha256=([0-9a-f]{16})\]",
                text.strip(),
            )
        if not match:
            return False
        if self._current_compress_store_ids_by_message_id.get(id(msg)) is not None:
            return False
        digest = match.group(1)
        hashes = getattr(self, "_generated_ignored_active_replay_placeholder_hashes", set())
        if digest in hashes:
            return True
        if digest in self._load_generated_ignored_placeholder_hashes():
            self._generated_ignored_active_replay_placeholder_hashes = set(hashes) | {digest}
            return True
        return False

    def _ignored_placeholder_metadata_key(self) -> str:
        return f"ignored_active_replay_placeholder_hashes:{self._session_id}"

    def _ignored_placeholder_metadata_keys(self) -> list[str]:
        return self._session_scoped_hash_metadata_keys("ignored_active_replay_placeholder_hashes")

    def _ignored_placeholder_count_metadata_keys(self) -> list[str]:
        return self._session_scoped_hash_metadata_keys("ignored_active_replay_placeholder_hash_counts")

    def _ignored_placeholder_ordinal_metadata_keys(self) -> list[str]:
        return self._session_scoped_hash_metadata_keys("ignored_active_replay_placeholder_hash_ordinals")

    def _ignored_dependent_reply_metadata_keys(self) -> list[str]:
        return self._session_scoped_hash_metadata_keys("ignored_dependent_reply_hashes")

    def _session_scoped_hash_metadata_keys(self, prefix: str, session_id: str | None = None) -> list[str]:
        scoped_session_id = self._session_id if session_id is None else session_id
        keys: list[str] = []
        if scoped_session_id:
            keys.append(f"{prefix}:{scoped_session_id}")
        return list(dict.fromkeys(keys))

    def _copy_generated_ignore_hashes_to_session(
        self,
        source_session_id: str,
        target_session_id: str,
        *,
        copy_dependent_content: bool = False,
        source_frontier_store_id: int = 0,
    ) -> None:
        if not source_session_id or not target_session_id or source_session_id == target_session_id:
            return
        source_keys = self._session_scoped_hash_metadata_keys(
            "ignored_active_replay_placeholder_hashes",
            source_session_id,
        )
        target_keys = self._session_scoped_hash_metadata_keys(
            "ignored_active_replay_placeholder_hashes",
            target_session_id,
        )
        for digest in self._load_hash_list_for_metadata_keys(source_keys):
            self._remember_hash_for_metadata_keys(digest, target_keys)

        if not copy_dependent_content:
            return

        dependent_target_keys = self._session_scoped_hash_metadata_keys(
            "ignored_dependent_reply_hashes",
            target_session_id,
        )
        dependent_records = self._load_generated_ignored_dependent_reply_records(
            self._session_scoped_hash_metadata_keys(
                "ignored_dependent_reply_hashes",
                source_session_id,
            )
        )
        active_dependent_store_digests: set[str] = set()
        try:
            active_rows = self._store.get_session_messages_after(
                source_session_id,
                after_store_id=max(0, int(source_frontier_store_id or 0)),
            )
            for row in active_rows:
                role = str(row.get("role") or "")
                if role not in {"assistant", "tool"}:
                    continue
                store_id = row.get("store_id")
                if store_id is None:
                    continue
                identity = f"{source_session_id}\0{int(store_id)}"
                active_dependent_store_digests.add(hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16])
        except Exception:
            logger.debug("LCM active dependent marker scan failed", exc_info=True)

        pending_records = [
            {"content": record["content"]}
            for record in dependent_records
            if record.get("content")
            and (not record.get("store") or record.get("store") in active_dependent_store_digests)
        ]
        if pending_records:
            target_records = self._load_generated_ignored_dependent_reply_records(dependent_target_keys)
            self._write_generated_ignored_dependent_reply_records(
                target_records + pending_records,
                dependent_target_keys,
            )

    def _load_hash_list_for_metadata_keys(self, keys: list[str]) -> list[str]:
        if not keys:
            return []
        try:
            ordered: list[str] = []
            seen: set[str] = set()
            for key in keys:
                data = self._store.read_metadata_json(key)
                if isinstance(data, list):
                    for item in data:
                        digest = str(item)
                        if re.fullmatch(r"[0-9a-f]{16}", digest) and digest not in seen:
                            ordered.append(digest)
                            seen.add(digest)
            return ordered
        except Exception:
            logger.debug("LCM scoped hash metadata load failed", exc_info=True)
        return []

    def _remember_hash_for_metadata_keys(self, digest: str, keys: list[str]) -> list[str]:
        if not re.fullmatch(r"[0-9a-f]{16}", digest):
            return []
        ordered_hashes = self._load_hash_list_for_metadata_keys(keys)
        ordered_hashes = [item for item in ordered_hashes if item != digest]
        ordered_hashes.append(digest)
        ordered_hashes = ordered_hashes[-512:]
        if not keys:
            return ordered_hashes
        try:
            payload = json.dumps(ordered_hashes)
            self._store.write_metadata_json(keys, payload)
        except Exception:
            logger.debug("LCM scoped hash metadata write failed", exc_info=True)
        return ordered_hashes

    def _load_generated_ignored_placeholder_hashes(self) -> set[str]:
        return set(self._load_generated_ignored_placeholder_hash_list())

    def _load_generated_ignored_placeholder_hash_list(self) -> list[str]:
        return self._load_hash_list_for_metadata_keys(self._ignored_placeholder_metadata_keys())

    def _load_generated_ignored_placeholder_hash_counts(
        self,
        keys: Optional[list[str]] = None,
    ) -> dict[str, int]:
        count_keys = self._ignored_placeholder_count_metadata_keys() if keys is None else keys
        counts: dict[str, int] = {}
        if not count_keys:
            return counts
        try:
            for key in count_keys:
                data = self._store.read_metadata_json(key)
                if not isinstance(data, dict):
                    continue
                for digest, count in data.items():
                    digest = str(digest)
                    if not re.fullmatch(r"[0-9a-f]{16}", digest):
                        continue
                    try:
                        parsed_count = max(0, int(count))
                    except (TypeError, ValueError):
                        continue
                    counts[digest] = max(counts.get(digest, 0), parsed_count)
        except Exception:
            logger.debug("LCM ignored placeholder count metadata load failed", exc_info=True)
        return counts

    def _write_generated_ignored_placeholder_hash_counts(
        self,
        counts: dict[str, int],
        keys: Optional[list[str]] = None,
    ) -> None:
        count_keys = self._ignored_placeholder_count_metadata_keys() if keys is None else keys
        if not count_keys:
            return
        payload: dict[str, int] = {}
        for digest, count in counts.items():
            digest = str(digest)
            if not re.fullmatch(r"[0-9a-f]{16}", digest):
                continue
            try:
                parsed_count = int(count)
            except (TypeError, ValueError):
                continue
            if parsed_count > 0:
                payload[digest] = parsed_count
        try:
            serialized = json.dumps(payload, sort_keys=True)
            # skip_unchanged avoids the fsync commit (under synchronous=FULL) when
            # the stored value already matches; this runs on every ingest.
            self._store.write_metadata_json(count_keys, serialized, skip_unchanged=True)
        except Exception:
            logger.debug("LCM ignored placeholder count metadata write failed", exc_info=True)

    def _load_generated_ignored_placeholder_hash_ordinals(
        self,
        keys: Optional[list[str]] = None,
    ) -> dict[str, set[int]]:
        ordinal_keys = self._ignored_placeholder_ordinal_metadata_keys() if keys is None else keys
        ordinals: dict[str, set[int]] = {}
        if not ordinal_keys:
            return ordinals
        try:
            for key in ordinal_keys:
                data = self._store.read_metadata_json(key)
                if not isinstance(data, dict):
                    continue
                for digest, values in data.items():
                    digest = str(digest)
                    if not re.fullmatch(r"[0-9a-f]{16}", digest) or not isinstance(values, list):
                        continue
                    bucket = ordinals.setdefault(digest, set())
                    for value in values:
                        try:
                            parsed = int(value)
                        except (TypeError, ValueError):
                            continue
                        if parsed > 0:
                            bucket.add(parsed)
        except Exception:
            logger.debug("LCM ignored placeholder ordinal metadata load failed", exc_info=True)
        return ordinals

    def _write_generated_ignored_placeholder_hash_ordinals(
        self,
        ordinals: dict[str, Any],
        keys: Optional[list[str]] = None,
    ) -> None:
        ordinal_keys = self._ignored_placeholder_ordinal_metadata_keys() if keys is None else keys
        if not ordinal_keys:
            return
        payload: dict[str, list[int]] = {}
        for digest, values in ordinals.items():
            digest = str(digest)
            if not re.fullmatch(r"[0-9a-f]{16}", digest):
                continue
            clean_values: set[int] = set()
            for value in values:
                try:
                    parsed = int(value)
                except (TypeError, ValueError):
                    continue
                if parsed > 0:
                    clean_values.add(parsed)
            clean = sorted(clean_values)
            if clean:
                payload[digest] = clean
        try:
            serialized = json.dumps(payload, sort_keys=True)
            # Skip the write (and its fsync commit) when unchanged; see the counts
            # writer above for rationale.
            self._store.write_metadata_json(ordinal_keys, serialized, skip_unchanged=True)
        except Exception:
            logger.debug("LCM ignored placeholder ordinal metadata write failed", exc_info=True)

    def _active_replay_generated_placeholder_digest_budget(self) -> dict[str, int]:
        return self._generated_placeholder_digest_budget_for_active_replay(
            self._last_active_replay_messages
        )

    def _generated_placeholder_digest_ordinals_for_active_replay(
        self,
        active_replay_messages: List[Dict[str, Any]],
    ) -> dict[str, set[int]]:
        generated_hashes = self._load_generated_ignored_placeholder_hashes()
        if not generated_hashes or not active_replay_messages:
            return {}
        stored_message_ids = set(self._get_store_id_map_for_messages(active_replay_messages))
        occurrence_by_digest: dict[str, int] = {}
        ordinals: dict[str, set[int]] = {}
        for msg in active_replay_messages:
            text = text_content_for_pattern_matching(msg.get("content")) or ""
            digest = self._active_replay_placeholder_digest(text)
            if not digest or digest not in generated_hashes:
                continue
            occurrence_by_digest[digest] = occurrence_by_digest.get(digest, 0) + 1
            if id(msg) in stored_message_ids:
                continue
            ordinals.setdefault(digest, set()).add(occurrence_by_digest[digest])
        return ordinals

    def _generated_placeholder_digest_budget_for_active_replay(
        self,
        active_replay_messages: List[Dict[str, Any]],
    ) -> dict[str, int]:
        generated_hashes = self._load_generated_ignored_placeholder_hashes()
        if not generated_hashes or not active_replay_messages:
            return {}
        stored_message_ids = set(self._get_store_id_map_for_messages(active_replay_messages))
        budget: dict[str, int] = {}
        for msg in active_replay_messages:
            if id(msg) in stored_message_ids:
                continue
            text = text_content_for_pattern_matching(msg.get("content")) or ""
            digest = self._active_replay_placeholder_digest(text)
            if digest and digest in generated_hashes:
                budget[digest] = budget.get(digest, 0) + 1
        return budget

    def _stored_active_replay_placeholder_digest_counts(
        self,
        session_id: str,
        *,
        after_store_id: int = 0,
    ) -> dict[str, int]:
        if not session_id:
            return {}
        counts: dict[str, int] = {}
        next_candidate_after = max(0, int(after_store_id or 0))
        while True:
            rows = self._store.get_session_messages_after(
                session_id,
                after_store_id=next_candidate_after,
            )
            if not rows:
                break
            for row in rows:
                text = text_content_for_pattern_matching(row.get("content")) or ""
                digest = self._active_replay_placeholder_digest(text)
                if digest:
                    counts[digest] = counts.get(digest, 0) + 1
            next_candidate_after = rows[-1]["store_id"]
        return counts

    @staticmethod
    def _subtract_placeholder_digest_counts(
        budget: dict[str, int],
        stored_counts: dict[str, int],
    ) -> dict[str, int]:
        adjusted: dict[str, int] = {}
        for digest, count in budget.items():
            parsed_count = max(0, int(count or 0))
            stored_count = max(0, int(stored_counts.get(digest, 0) or 0))
            remaining = max(0, parsed_count - stored_count)
            if remaining > 0:
                adjusted[digest] = remaining
        return adjusted

    def _remember_generated_ignored_placeholder_hash(self, digest: str) -> None:
        ordered_hashes = self._remember_hash_for_metadata_keys(
            digest,
            self._ignored_placeholder_metadata_keys(),
        )
        hashes = set(ordered_hashes)
        self._generated_ignored_active_replay_placeholder_hashes = hashes

    def _ignored_dependent_reply_store_fingerprint(self, msg: Dict[str, Any]) -> Optional[str]:
        role = str(msg.get("role") or "")
        if role not in {"assistant", "tool"}:
            return None
        # Store-scoped dependent markers must be tied to provenance the caller
        # already has; a singleton content lookup can bind repeated replies to
        # an older ignored-dependent row.
        store_id = msg.get("store_id")
        if store_id is None:
            store_id = self._current_compress_store_ids_by_message_id.get(id(msg))
        if store_id is None:
            return None
        identity = f"{self._session_id}\0{store_id}"
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]

    def _ignored_dependent_reply_content_fingerprint(self, msg: Dict[str, Any], text: str) -> Optional[str]:
        role = str(msg.get("role") or "")
        if role not in {"assistant", "tool"}:
            return None
        identity = "\0".join(
            (
                role,
                str(msg.get("tool_call_id") or ""),
                self._stable_tool_calls_identity(msg.get("tool_calls")),
                text,
            )
        )
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]

    def _load_generated_ignored_dependent_reply_records(
        self,
        keys: Optional[list[str]] = None,
    ) -> list[dict[str, str]]:
        keys = self._ignored_dependent_reply_metadata_keys() if keys is None else keys
        if not keys:
            return []
        try:
            records: list[dict[str, str]] = []
            seen: set[tuple[str, str]] = set()
            for key in keys:
                data = self._store.read_metadata_json(key)
                if not isinstance(data, list):
                    continue
                for item in data:
                    record: dict[str, str] = {}
                    if isinstance(item, dict):
                        store = str(item.get("store") or "")
                        content = str(item.get("content") or "")
                        if re.fullmatch(r"[0-9a-f]{16}", store):
                            record["store"] = store
                        if re.fullmatch(r"[0-9a-f]{16}", content):
                            record["content"] = content
                    elif re.fullmatch(r"[0-9a-f]{16}", str(item)):
                        record["store"] = str(item)
                    if not record:
                        continue
                    marker = (record.get("store", ""), record.get("content", ""))
                    if record.get("store"):
                        if marker in seen:
                            continue
                        seen.add(marker)
                    records.append(record)
            return records[-512:]
        except Exception:
            logger.debug("LCM ignored-dependent reply metadata load failed", exc_info=True)
        return []

    def _write_generated_ignored_dependent_reply_records(
        self,
        records: list[dict[str, str]],
        keys: Optional[list[str]] = None,
    ) -> None:
        keys = self._ignored_dependent_reply_metadata_keys() if keys is None else keys
        if not keys:
            return
        normalized: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for record in records:
            clean: dict[str, str] = {}
            store = str(record.get("store") or "")
            content = str(record.get("content") or "")
            if re.fullmatch(r"[0-9a-f]{16}", store):
                clean["store"] = store
            if re.fullmatch(r"[0-9a-f]{16}", content):
                clean["content"] = content
            if not clean:
                continue
            marker = (clean.get("store", ""), clean.get("content", ""))
            if clean.get("store"):
                if marker in seen:
                    continue
                seen.add(marker)
            normalized.append(clean)
        normalized = normalized[-512:]
        try:
            payload = json.dumps(normalized)
            self._store.write_metadata_json(keys, payload)
        except Exception:
            logger.debug("LCM ignored-dependent reply metadata write failed", exc_info=True)

    def _load_generated_ignored_dependent_reply_hashes(self) -> set[str]:
        return {
            value
            for record in self._load_generated_ignored_dependent_reply_records()
            for value in (record.get("store"), record.get("content"))
            if value
        }

    def _is_generated_ignored_dependent_reply(self, msg: Dict[str, Any], text: str) -> bool:
        store_digest = self._ignored_dependent_reply_store_fingerprint(msg)
        content_digest = self._ignored_dependent_reply_content_fingerprint(msg, text)
        records = self._load_generated_ignored_dependent_reply_records()
        if store_digest and any(record.get("store") == store_digest for record in records):
            return True
        if not content_digest:
            return False
        pending_index = next(
            (
                idx
                for idx, record in enumerate(records)
                if record.get("content") == content_digest and not record.get("store")
            ),
            None,
        )
        if pending_index is None:
            return False
        records.pop(pending_index)
        if store_digest:
            records.append({"store": store_digest, "content": content_digest})
        self._write_generated_ignored_dependent_reply_records(records)
        return True

    def _matches_preexisting_generated_ignored_dependent_reply(
        self,
        msg: Dict[str, Any],
        text: str,
        records: list[dict[str, str]],
    ) -> bool:
        store_digest = self._ignored_dependent_reply_store_fingerprint(msg)
        content_digest = self._ignored_dependent_reply_content_fingerprint(msg, text)
        if store_digest and any(record.get("store") == store_digest for record in records):
            return True
        if not content_digest:
            return False
        pending_index = next(
            (
                idx
                for idx, record in enumerate(records)
                if record.get("content") == content_digest and not record.get("store")
            ),
            None,
        )
        if pending_index is None:
            return False
        records.pop(pending_index)
        if store_digest:
            records.append({"store": store_digest, "content": content_digest})
        live_records = self._load_generated_ignored_dependent_reply_records()
        live_pending_index = next(
            (
                idx
                for idx, record in enumerate(live_records)
                if record.get("content") == content_digest and not record.get("store")
            ),
            None,
        )
        if live_pending_index is not None:
            live_records.pop(live_pending_index)
            if store_digest:
                live_records.append({"store": store_digest, "content": content_digest})
            self._write_generated_ignored_dependent_reply_records(live_records)
        return True

    def _drop_preexisting_generated_ignored_dependent_eof_replies(
        self,
        messages: List[Dict[str, Any]],
        records: list[dict[str, str]],
    ) -> List[Dict[str, Any]]:
        if not records or not messages:
            return messages
        previous_store_id_map = self._current_compress_store_ids_by_message_id
        self._current_compress_store_ids_by_message_id = self._get_store_id_map_for_messages(messages)
        try:
            drop_from = len(messages)
            idx = len(messages) - 1
            while idx >= 0:
                msg = messages[idx]
                role = str(msg.get("role") or "")
                if role not in {"assistant", "tool"}:
                    break
                text = text_content_for_pattern_matching(msg.get("content")) or ""
                if not self._matches_preexisting_generated_ignored_dependent_reply(
                    msg,
                    text,
                    records,
                ):
                    break
                drop_from = idx
                idx -= 1
            if drop_from == len(messages):
                return messages
            return messages[:drop_from]
        finally:
            self._current_compress_store_ids_by_message_id = previous_store_id_map

    def _remember_generated_ignored_dependent_reply(self, msg: Dict[str, Any], text: str) -> None:
        store_digest = self._ignored_dependent_reply_store_fingerprint(msg)
        content_digest = self._ignored_dependent_reply_content_fingerprint(msg, text)
        if not store_digest:
            return
        records = self._load_generated_ignored_dependent_reply_records()
        records.append({"store": store_digest, "content": content_digest or ""})
        self._write_generated_ignored_dependent_reply_records(records)

    def _apply_ignored_active_replay_placeholders(
        self,
        original_messages: List[Dict[str, Any]],
        replay_messages: List[Dict[str, Any]],
        *,
        scan_start: int = 0,
        ignored_messages: Optional[List[bool]] = None,
    ) -> List[Dict[str, Any]]:
        if not self._compiled_ignore_message_patterns:
            return replay_messages
        active_replay_messages = replay_messages
        for idx in range(max(0, scan_start), min(len(original_messages), len(replay_messages))):
            original_msg = original_messages[idx]
            replay_msg = replay_messages[idx]
            ignored = (
                ignored_messages[idx]
                if ignored_messages is not None and idx < len(ignored_messages)
                else self._matches_ignore_message_patterns(original_msg)
            )
            if not ignored:
                continue
            replay_text = text_content_for_pattern_matching(replay_msg.get("content")) or ""
            replay_preserves_ignore_decision = (
                self._is_volatile_ignored_quarantine_placeholder(replay_msg, replay_text)
                or self._is_ignored_active_replay_placeholder(replay_msg, replay_text)
            )
            if replay_preserves_ignore_decision:
                continue
            if active_replay_messages is replay_messages:
                active_replay_messages = self._copy_active_replay_messages_preserving_generated_ids(
                    replay_messages
                )
            original_text = text_content_for_pattern_matching(original_msg.get("content")) or ""
            placeholder = self._ignored_active_replay_placeholder(original_text)
            original_role = str(original_msg.get("role") or "")
            if original_role == "tool":
                active_message = {
                    "role": "tool",
                    "content": placeholder,
                    "tool_call_id": original_msg.get("tool_call_id") or replay_msg.get("tool_call_id") or "ignored_tool_call",
                }
            elif original_role == "assistant":
                active_message = {
                    "role": "assistant",
                    "content": placeholder,
                }
            elif original_role == "system":
                active_message = {"role": "system", "content": placeholder}
            else:
                active_message = {"role": "user", "content": placeholder}
            digest = hashlib.sha256(original_text.encode("utf-8")).hexdigest()[:16]
            self._remember_generated_ignored_placeholder_hash(digest)
            self._generated_ignored_active_replay_placeholder_message_ids.add(id(active_message))
            active_replay_messages[idx] = active_message
        return active_replay_messages
