"""Message-row adapter for the V1 opaque retrieval-reference foundation.

This module is intentionally one adapter vertical: it binds a durable
``messages.store_id`` to a caller-supplied scope and a semantic message
revision.  It does not add public tools, pagination, cursors, summary-node or
sidecar references, principals, tenants, or schema migrations.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Mapping

from .message_content import normalize_content_value
from .retrieval_references import (
    ReferenceEnvelope,
    ReferenceError,
    ReferenceErrorCode,
    ReferenceRecord,
    ReferenceResult,
    canonical_json,
    issue_reference,
    resolve_reference,
)

MESSAGE_REFERENCE_KIND = "message"
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_UNSET = object()
_MESSAGE_SELECT = (
    "store_id, session_id, source, conversation_id, role, content, "
    "tool_call_id, tool_calls, tool_name, timestamp, token_estimate, pinned"
)


MessageReferenceError = ReferenceError


@dataclass(frozen=True)
class MessageReferenceResult:
    """Safe message resolution result with the foundation's typed error code."""

    ok: bool
    message: dict[str, Any] | None = None
    record: ReferenceRecord | None = None
    error_code: str | None = None
    error: str | None = None

    def __bool__(self) -> bool:
        return self.ok

    def to_dict(self) -> dict[str, object]:
        if not self.ok:
            return {
                "error_code": self.error_code or ReferenceErrorCode.REFERENCE_INVALID.value,
                "error": self.error or "message reference resolution failed",
            }
        return {"ok": True, "message": self.message}

    def get(self, key: str, default: object = None) -> object:
        return self.to_dict().get(key, default)

    def __getitem__(self, key: str) -> object:
        return self.to_dict()[key]


class MessageReferenceAdapter:
    """Connection-bound facade for the message reference adapter."""

    def __init__(self, store_or_connection: object):
        self._store_or_connection = store_or_connection

    @property
    def connection(self) -> sqlite3.Connection:
        return _connection_for(self._store_or_connection)

    def issue(self, store_id: object, scope: object, **kwargs: object) -> ReferenceEnvelope:
        return issue_message_reference(self._store_or_connection, store_id, scope, **kwargs)

    def resolve(self, envelope: object, **kwargs: object) -> MessageReferenceResult:
        return resolve_message_reference(self._store_or_connection, envelope, **kwargs)


def _connection_for(store_or_connection: object) -> sqlite3.Connection:
    if isinstance(store_or_connection, sqlite3.Connection):
        return store_or_connection
    connection = getattr(store_or_connection, "connection", None)
    if isinstance(connection, sqlite3.Connection):
        return connection
    raise ReferenceError(
        ReferenceErrorCode.INVALID_REQUEST,
        "message reference adapter requires a live SQLite connection",
    )


def _store_lock(store_or_connection: object, *, write: bool):
    # MessageStore serializes writes to its shared connection with this lock. Use
    # it when available without making the adapter depend on MessageStore.
    lock = getattr(store_or_connection, "_write_lock", None) if write else None
    return lock if lock is not None else nullcontext()


@contextmanager
def _write_transaction(conn: sqlite3.Connection):
    """Own a writer transaction only when the caller does not already own one."""
    owns_transaction = not conn.in_transaction
    if owns_transaction:
        conn.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        if owns_transaction and conn.in_transaction:
            conn.rollback()
        raise
    else:
        if owns_transaction and conn.in_transaction:
            conn.commit()


@contextmanager
def _read_snapshot(conn: sqlite3.Connection):
    """Pin a read snapshot while preserving any caller-owned transaction."""
    owns_transaction = not conn.in_transaction
    if owns_transaction:
        # An explicit deferred BEGIN is required: sqlite3 does not begin a
        # transaction for a SELECT under its default isolation behavior.
        conn.execute("BEGIN")
    try:
        yield
    except BaseException:
        if owns_transaction and conn.in_transaction:
            conn.rollback()
        raise
    else:
        if owns_transaction and conn.in_transaction:
            conn.commit()


def _positive_store_id(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ReferenceError(
            ReferenceErrorCode.INVALID_REQUEST,
            "message reference store_id must be a positive integer",
        )
    return value


def _normalize_source(value: object) -> str:
    normalized = "" if value is None else str(value).strip()
    return normalized or "unknown"


def _normalize_conversation_id(value: object) -> str:
    return ("" if value is None else str(value)).strip()


def _reject_nonfinite_json(value: str) -> object:
    raise ValueError(f"non-finite JSON constant: {value}")


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _semantic_tool_calls(value: object) -> object:
    """Parse every valid JSON value, retaining invalid/ambiguous raw strings."""
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    try:
        return json.loads(
            value,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        return value


def _finite_timestamp(value: object) -> float:
    try:
        timestamp = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ReferenceError(
            ReferenceErrorCode.REFERENCE_STALE,
            "message row has an invalid timestamp",
        ) from exc
    if not math.isfinite(timestamp):
        raise ReferenceError(
            ReferenceErrorCode.REFERENCE_STALE,
            "message row has an invalid timestamp",
        )
    return timestamp


def _row_mapping(row: sqlite3.Row | tuple[object, ...]) -> dict[str, Any]:
    values = dict(zip(
        (
            "store_id",
            "session_id",
            "source",
            "conversation_id",
            "role",
            "content",
            "tool_call_id",
            "tool_calls",
            "tool_name",
            "timestamp",
            "token_estimate",
            "pinned",
        ),
        tuple(row),
    ))
    values["source"] = _normalize_source(values.get("source"))
    values["conversation_id"] = _normalize_conversation_id(values.get("conversation_id"))
    values["content"] = normalize_content_value(values.get("content"))
    values["tool_calls"] = _semantic_tool_calls(values.get("tool_calls"))
    return values


def _load_message(conn: sqlite3.Connection, store_id: int) -> dict[str, Any] | None:
    row = conn.execute(
        f"SELECT {_MESSAGE_SELECT} FROM messages WHERE store_id = ?",
        (store_id,),
    ).fetchone()
    return _row_mapping(row) if row is not None else None


def message_revision_preimage(message: Mapping[str, object]) -> dict[str, object]:
    """Return the exact normalized object hashed by :func:`message_revision`."""
    try:
        timestamp = _finite_timestamp(message.get("timestamp"))
    except AttributeError as exc:
        raise ReferenceError(
            ReferenceErrorCode.REFERENCE_STALE,
            "message row is not a mapping",
        ) from exc
    normalized = {
        "session_id": message.get("session_id"),
        "source": _normalize_source(message.get("source")),
        "conversation_id": _normalize_conversation_id(message.get("conversation_id")),
        "role": message.get("role"),
        "content": normalize_content_value(message.get("content")),
        "tool_call_id": message.get("tool_call_id"),
        "tool_calls": _semantic_tool_calls(message.get("tool_calls")),
        "tool_name": message.get("tool_name"),
        "timestamp": timestamp,
    }
    try:
        canonical_json(normalized)
    except ReferenceError as exc:
        raise ReferenceError(
            ReferenceErrorCode.REFERENCE_STALE,
            "message row contains values that are not canonical JSON",
        ) from exc
    return normalized


def message_revision(message: Mapping[str, object]) -> dict[str, str]:
    """Compute the frozen SHA-256 semantic revision for one message row."""
    preimage = message_revision_preimage(message)
    digest = hashlib.sha256(canonical_json(preimage).encode("utf-8")).hexdigest()
    return {"algorithm": "sha256", "semantic_digest": digest}


# Explicit aliases make the adapter's semantic operation discoverable without
# creating a second revision algorithm or public integration surface.
compute_message_revision = message_revision
message_semantic_revision = message_revision
semantic_message_revision = message_revision


def _message_target(record_target: object) -> int:
    if (
        not isinstance(record_target, Mapping)
        or set(record_target) != {"store_id"}
        or isinstance(record_target.get("store_id"), bool)
        or not isinstance(record_target.get("store_id"), int)
        or record_target.get("store_id") <= 0
    ):
        raise ReferenceError(
            ReferenceErrorCode.REFERENCE_STALE,
            "message reference target is malformed",
        )
    return int(record_target["store_id"])


def _message_revision(record_revision: object) -> dict[str, str]:
    if not isinstance(record_revision, Mapping) or set(record_revision) != {
        "algorithm",
        "semantic_digest",
    }:
        raise ReferenceError(
            ReferenceErrorCode.REFERENCE_STALE,
            "message reference revision is malformed",
        )
    if record_revision.get("algorithm") != "sha256":
        raise ReferenceError(
            ReferenceErrorCode.REFERENCE_STALE,
            "message reference revision is malformed",
        )
    digest = record_revision.get("semantic_digest")
    if not isinstance(digest, str) or _DIGEST_RE.fullmatch(digest) is None:
        raise ReferenceError(
            ReferenceErrorCode.REFERENCE_STALE,
            "message reference revision is malformed",
        )
    return {"algorithm": "sha256", "semantic_digest": digest}


def _result_failure(code: ReferenceErrorCode | str, error: str) -> MessageReferenceResult:
    return MessageReferenceResult(
        ok=False,
        error_code=ReferenceErrorCode(code).value,
        error=str(error),
    )


def _foundation_failure(result: ReferenceResult) -> MessageReferenceResult:
    return _result_failure(
        result.error_code or ReferenceErrorCode.REFERENCE_INVALID.value,
        result.error or "reference resolution failed",
    )


def _authorize_message_target(
    authorization_context: object,
    store_id: int,
    scope: object,
    authorize_target,
) -> None:
    try:
        allowed = (
            authorization_context is not None
            and authorize_target is not None
            and bool(
                authorize_target(authorization_context, store_id, scope)
            )
        )
    except Exception:
        allowed = False
    if not allowed:
        raise ReferenceError(
            ReferenceErrorCode.REFERENCE_FORBIDDEN,
            "message reference issuance is not authorized by the host",
        )


def issue_message_reference(
    store_or_connection: object,
    store_id: object,
    scope: object,
    *,
    authorization_context: object = None,
    authorize_target=None,
    authorize_issue=None,
    consistency: str = "live",
    issued_at: float | None = None,
    expires_at: float | None = None,
) -> ReferenceEnvelope:
    """Issue a ``kind=message`` reference for an existing message row.

    Target preauthorization runs before any transaction or message lookup. The
    row read, semantic digest, foundation authorization, and registry savepoint
    then run inside one adapter-owned ``BEGIN IMMEDIATE`` or the caller's
    already-active transaction.
    """
    store_id = _positive_store_id(store_id)
    _authorize_message_target(
        authorization_context,
        store_id,
        scope,
        authorize_target,
    )
    conn = _connection_for(store_or_connection)
    with _store_lock(store_or_connection, write=True):
        with _write_transaction(conn):
            message = _load_message(conn, store_id)
            if message is None:
                raise ReferenceError(
                    ReferenceErrorCode.REFERENCE_NOT_FOUND,
                    "message was not found",
                )
            revision = message_revision(message)
            return issue_reference(
                conn,
                MESSAGE_REFERENCE_KIND,
                {"store_id": store_id},
                scope,
                revision,
                authorization_context=authorization_context,
                authorize_issue=authorize_issue,
                consistency=consistency,
                issued_at=issued_at,
                expires_at=expires_at,
            )


def resolve_message_reference(
    store_or_connection: object,
    envelope: object,
    *,
    authorization_context: object = None,
    authorize=None,
    authorize_scope=None,
    expected_scope: object = _UNSET,
    now: float | None = None,
) -> MessageReferenceResult:
    """Resolve and hydrate one message after both foundation authorization gates."""
    conn = _connection_for(store_or_connection)
    with _store_lock(store_or_connection, write=True):
        with _read_snapshot(conn):
            try:
                kwargs: dict[str, object] = {
                    "authorization_context": authorization_context,
                    "authorize": authorize,
                    "authorize_scope": authorize_scope,
                    "expected_kind": MESSAGE_REFERENCE_KIND,
                    "now": now,
                }
                if expected_scope is not _UNSET:
                    kwargs["expected_scope"] = expected_scope
                foundation = resolve_reference(conn, envelope, **kwargs)
            except sqlite3.DatabaseError:
                return _result_failure(
                    ReferenceErrorCode.REFERENCE_INVALID,
                    "reference resolution failed",
                )
            if not foundation.ok:
                return _foundation_failure(foundation)
            if foundation.record is None:
                return _result_failure(
                    ReferenceErrorCode.REFERENCE_INVALID,
                    "reference resolution returned no binding",
                )
            try:
                store_id = _message_target(foundation.record.target)
                stored_revision = _message_revision(foundation.record.revision)
            except ReferenceError as exc:
                return _result_failure(exc.code, exc.error)

            try:
                message = _load_message(conn, store_id)
            except sqlite3.DatabaseError:
                return _result_failure(
                    ReferenceErrorCode.REFERENCE_INVALID,
                    "message lookup failed",
                )
            if message is None:
                return _result_failure(
                    ReferenceErrorCode.REFERENCE_NOT_FOUND,
                    "message was not found",
                )
            try:
                current_revision = message_revision(message)
            except ReferenceError as exc:
                return _result_failure(exc.code, exc.error)
            if current_revision != stored_revision:
                return _result_failure(
                    ReferenceErrorCode.REFERENCE_STALE,
                    "message reference is stale",
                )
            return MessageReferenceResult(
                ok=True,
                message=message,
                record=foundation.record,
            )


def resolve_message_reference_or_raise(
    store_or_connection: object,
    envelope: object,
    **kwargs: object,
) -> dict[str, Any]:
    result = resolve_message_reference(store_or_connection, envelope, **kwargs)
    if not result.ok:
        raise ReferenceError(
            result.error_code or ReferenceErrorCode.REFERENCE_INVALID.value,
            result.error or "message reference resolution failed",
        )
    assert result.message is not None
    return result.message


# Noun-order aliases for callers that use the retrieval-reference terminology.
issue_message_retrieval_reference = issue_message_reference
resolve_message_retrieval_reference = resolve_message_reference
resolve_message_retrieval_reference_or_raise = resolve_message_reference_or_raise


__all__ = [
    "MESSAGE_REFERENCE_KIND",
    "MessageReferenceAdapter",
    "MessageReferenceError",
    "MessageReferenceResult",
    "compute_message_revision",
    "issue_message_reference",
    "issue_message_retrieval_reference",
    "message_revision",
    "message_revision_preimage",
    "message_semantic_revision",
    "resolve_message_reference",
    "resolve_message_reference_or_raise",
    "resolve_message_retrieval_reference",
    "resolve_message_retrieval_reference_or_raise",
    "semantic_message_revision",
]
