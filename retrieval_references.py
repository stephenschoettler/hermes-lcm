"""V1 opaque retrieval-reference foundation.

This module deliberately stops at the durable reference registry.  It does not
load messages, DAG nodes, sidecars, cursors, principals, or tenants.  A caller
supplies a trusted host authorization context and, after successful resolution,
may hand the server-side binding to a later target adapter.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import secrets
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Mapping

from .db_bootstrap import (
    DATABASE_UUID_MIGRATION,
    DATABASE_UUID_METADATA_KEY,
    RETRIEVAL_REFERENCE_CONSISTENCIES,
    RETRIEVAL_REFERENCE_KINDS,
    RETRIEVAL_REFERENCES_TABLE,
    DatabaseIdentityError,
    RetrievalReferenceSchemaError,
    get_retrieval_reference_authority,
    rotate_database_uuid,
)

TOKEN_PREFIX = "rh1_"
_TOKEN_RE = re.compile(r"^rh1_[0-9a-f]{64}$")
_UNSET = object()


class ReferenceErrorCode(str, Enum):
    """Closed V1 error vocabulary shared by strong-reference paths."""

    INVALID_REQUEST = "invalid_request"
    REFERENCE_INVALID = "reference_invalid"
    REFERENCE_UNSUPPORTED_VERSION = "reference_unsupported_version"
    REFERENCE_DATABASE_MISMATCH = "reference_database_mismatch"
    REFERENCE_KIND_MISMATCH = "reference_kind_mismatch"
    REFERENCE_SCOPE_MISMATCH = "reference_scope_mismatch"
    REFERENCE_FORBIDDEN = "reference_forbidden"
    REFERENCE_NOT_FOUND = "reference_not_found"
    REFERENCE_STALE = "reference_stale"


class ReferenceKind(str, Enum):
    MESSAGE = "message"
    SUMMARY_NODE = "summary_node"
    EXTERNALIZED_PAYLOAD = "externalized_payload"
    SOURCE_CURSOR = "source_cursor"
    CONTENT_CURSOR = "content_cursor"
    SESSION_CURSOR = "session_cursor"


class ReferenceConsistency(str, Enum):
    LIVE = "live"
    HIGH_WATERMARK = "high_watermark"
    SNAPSHOT = "snapshot"


class ReferenceError(ValueError):
    """Typed strong-reference failure with a stable code and human message."""

    def __init__(self, code: ReferenceErrorCode | str, error: str):
        self.code = ReferenceErrorCode(code)
        self.error = str(error)
        super().__init__(self.error)

    @property
    def error_code(self) -> str:
        return self.code.value

    def as_dict(self) -> dict[str, object]:
        return {"error_code": self.error_code, "error": self.error}


@dataclass(frozen=True)
class ReferenceEnvelope:
    """The complete V1 wire value; target, scope, and revision stay server-side."""

    kind: str
    database_uuid: str
    token: str
    v: int = 1

    def to_dict(self) -> dict[str, object]:
        return {
            "v": self.v,
            "kind": self.kind,
            "database_uuid": self.database_uuid,
            "token": self.token,
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    def __getitem__(self, key: str) -> object:
        return self.to_dict()[key]


@dataclass(frozen=True)
class ReferenceRecord:
    """Validated server-side registry binding returned after authorization."""

    token_digest: str
    kind: str
    target: object
    scope: object
    revision: object
    consistency: str
    issued_at: float
    expires_at: float | None
    revoked_at: float | None

    @property
    def target_json(self) -> str:
        return canonical_json(self.target)

    @property
    def scope_json(self) -> str:
        return canonical_json(self.scope)

    @property
    def revision_json(self) -> str:
        return canonical_json(self.revision)


@dataclass(frozen=True)
class ReferenceResult:
    """Structured resolution/mutation result retaining a human-readable error."""

    ok: bool
    record: ReferenceRecord | None = None
    error_code: str | None = None
    error: str | None = None

    def __bool__(self) -> bool:
        return self.ok

    def to_dict(self) -> dict[str, object]:
        if not self.ok:
            return {
                "error_code": self.error_code or ReferenceErrorCode.REFERENCE_INVALID.value,
                "error": self.error or "reference operation failed",
            }
        if self.record is None:
            return {"ok": True}
        return {
            "ok": True,
            "kind": self.record.kind,
            "target": self.record.target,
            "scope": self.record.scope,
            "revision": self.record.revision,
            "consistency": self.record.consistency,
            "issued_at": self.record.issued_at,
            "expires_at": self.record.expires_at,
        }

    def get(self, key: str, default: object = None) -> object:
        return self.to_dict().get(key, default)

    def __getitem__(self, key: str) -> object:
        return self.to_dict()[key]


@dataclass(frozen=True)
class AuthorizationRequest:
    """Callback input containing only trusted context and the opaque envelope."""

    trusted_context: object
    envelope: ReferenceEnvelope


AuthorizationCallback = Callable[[object, ReferenceEnvelope], bool]
ScopeAuthorizationCallback = Callable[[object, ReferenceEnvelope, object], bool]
IssueAuthorizationCallback = Callable[[object, str, object, object, object], bool]


def canonical_json(value: object) -> str:
    """Serialize JSON with one deterministic V1 representation."""
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ReferenceError(
            ReferenceErrorCode.INVALID_REQUEST,
            "reference binding contains values that are not canonical JSON",
        ) from exc


def _json_object_without_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ReferenceError(
                ReferenceErrorCode.REFERENCE_INVALID,
                "reference envelope contains a duplicate JSON key",
            )
        result[key] = value
    return result


def _canonical_uuid4(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, TypeError, ValueError):
        return False
    return parsed.version == 4 and str(parsed) == value


def _coerce_kind(value: object) -> str:
    kind = value.value if isinstance(value, Enum) else value
    if not isinstance(kind, str) or kind not in RETRIEVAL_REFERENCE_KINDS:
        raise ReferenceError(
            ReferenceErrorCode.INVALID_REQUEST,
            "reference kind is not supported by V1",
        )
    return kind


def _coerce_consistency(value: object) -> str:
    consistency = value.value if isinstance(value, Enum) else value
    if not isinstance(consistency, str) or consistency not in RETRIEVAL_REFERENCE_CONSISTENCIES:
        raise ReferenceError(
            ReferenceErrorCode.INVALID_REQUEST,
            "reference consistency is not supported by V1",
        )
    return consistency


def generate_reference_token() -> str:
    """Return a random 256-bit V1 wire token; only its digest is persisted."""
    return TOKEN_PREFIX + secrets.token_bytes(32).hex()


def reference_token_digest(token: str) -> str:
    if not isinstance(token, str) or _TOKEN_RE.fullmatch(token) is None:
        raise ReferenceError(
            ReferenceErrorCode.REFERENCE_INVALID,
            "reference token is malformed",
        )
    return hashlib.sha256(token.encode("ascii")).hexdigest()


def encode_reference_envelope(
    *, kind: str, database_uuid: str, token: str
) -> ReferenceEnvelope:
    kind = _coerce_kind(kind)
    if not _canonical_uuid4(database_uuid):
        raise ReferenceError(
            ReferenceErrorCode.REFERENCE_INVALID,
            "reference database_uuid is malformed",
        )
    reference_token_digest(token)
    return ReferenceEnvelope(kind=kind, database_uuid=database_uuid, token=token)


def encode_envelope(*, kind: str, database_uuid: str, token: str) -> dict[str, object]:
    """Return the JSON-object form used by additive strong-reference fields."""
    return encode_reference_envelope(
        kind=kind, database_uuid=database_uuid, token=token
    ).to_dict()


def serialize_reference_envelope(envelope: ReferenceEnvelope | Mapping[str, object]) -> str:
    return decode_reference_envelope(envelope).to_json()


def decode_reference_envelope(value: object) -> ReferenceEnvelope:
    if isinstance(value, ReferenceEnvelope):
        if not isinstance(value.v, int) or isinstance(value.v, bool) or value.v != 1:
            raise ReferenceError(
                ReferenceErrorCode.REFERENCE_UNSUPPORTED_VERSION,
                "reference envelope version is not supported",
            )
        try:
            envelope = encode_reference_envelope(
                kind=value.kind,
                database_uuid=value.database_uuid,
                token=value.token,
            )
        except ReferenceError as exc:
            raise ReferenceError(
                ReferenceErrorCode.REFERENCE_INVALID,
                "reference envelope contains an invalid binding",
            ) from exc
    else:
        parsed: object = value
        if isinstance(value, (str, bytes, bytearray)):
            try:
                parsed = json.loads(
                    value,
                    object_pairs_hook=_json_object_without_duplicate_keys,
                )
            except ReferenceError:
                raise
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ReferenceError(
                    ReferenceErrorCode.REFERENCE_INVALID,
                    "reference envelope is not valid JSON",
                ) from exc
        if not isinstance(parsed, Mapping):
            raise ReferenceError(
                ReferenceErrorCode.REFERENCE_INVALID,
                "reference envelope must be a JSON object",
            )
        required = {"v", "kind", "database_uuid", "token"}
        if set(parsed) != required:
            raise ReferenceError(
                ReferenceErrorCode.REFERENCE_INVALID,
                "reference envelope has an invalid shape",
            )
        version = parsed.get("v")
        if not isinstance(version, int) or isinstance(version, bool) or version != 1:
            raise ReferenceError(
                ReferenceErrorCode.REFERENCE_UNSUPPORTED_VERSION,
                "reference envelope version is not supported",
            )
        try:
            envelope = encode_reference_envelope(
                kind=parsed["kind"],
                database_uuid=parsed["database_uuid"],
                token=parsed["token"],
            )
        except KeyError as exc:
            raise ReferenceError(
                ReferenceErrorCode.REFERENCE_INVALID,
                "reference envelope is missing a required field",
            ) from exc
        except ReferenceError as exc:
            raise ReferenceError(
                ReferenceErrorCode.REFERENCE_INVALID,
                "reference envelope contains an invalid binding",
            ) from exc
    if envelope.v != 1:
        raise ReferenceError(
            ReferenceErrorCode.REFERENCE_UNSUPPORTED_VERSION,
            "reference envelope version is not supported",
        )
    return envelope


# Concise aliases for callers that use the contract's noun rather than the
# implementation's longer function names.
decode_envelope = decode_reference_envelope
serialize_envelope = serialize_reference_envelope


@contextmanager
def _registry_write_transaction(conn):
    savepoint = "lcm_retrieval_reference_write"
    owns_transaction = not conn.in_transaction
    if owns_transaction:
        conn.execute("BEGIN IMMEDIATE")
    else:
        conn.execute(f'SAVEPOINT "{savepoint}"')
    try:
        yield
    except BaseException:
        if owns_transaction:
            conn.rollback()
        else:
            try:
                conn.execute(f'ROLLBACK TO "{savepoint}"')
            finally:
                conn.execute(f'RELEASE "{savepoint}"')
        raise
    else:
        if owns_transaction:
            conn.commit()
        else:
            conn.execute(f'RELEASE "{savepoint}"')


def _finite_time(value: object, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ReferenceError(
            ReferenceErrorCode.INVALID_REQUEST,
            f"{field} must be a finite timestamp",
        ) from exc
    if not math.isfinite(result):
        raise ReferenceError(
            ReferenceErrorCode.INVALID_REQUEST,
            f"{field} must be a finite timestamp",
        )
    return result


def issue_reference(
    conn,
    kind: str,
    target: object,
    scope: object,
    revision: object,
    *,
    authorization_context: object = None,
    authorize_issue: IssueAuthorizationCallback | None = None,
    consistency: str = "live",
    issued_at: float | None = None,
    expires_at: float | None = None,
) -> ReferenceEnvelope:
    """Persist one server-side binding and return its opaque V1 envelope."""
    kind = _coerce_kind(kind)
    consistency = _coerce_consistency(consistency)
    target_json = canonical_json(target)
    scope_json = canonical_json(scope)
    revision_json = canonical_json(revision)
    if authorize_issue is None or authorization_context is None:
        raise ReferenceError(
            ReferenceErrorCode.REFERENCE_FORBIDDEN,
            "reference issuance is not authorized by the host",
        )
    try:
        issue_allowed = bool(
            authorize_issue(
                authorization_context,
                kind,
                json.loads(target_json),
                json.loads(scope_json),
                json.loads(revision_json),
            )
        )
    except Exception:
        issue_allowed = False
    if not issue_allowed:
        raise ReferenceError(
            ReferenceErrorCode.REFERENCE_FORBIDDEN,
            "reference issuance is not authorized by the host",
        )
    issued = _finite_time(time.time() if issued_at is None else issued_at, "issued_at")
    expires = None if expires_at is None else _finite_time(expires_at, "expires_at")
    if expires is not None and expires < issued:
        raise ReferenceError(
            ReferenceErrorCode.INVALID_REQUEST,
            "expires_at cannot precede issued_at",
        )

    token = generate_reference_token()
    digest = reference_token_digest(token)
    with _registry_write_transaction(conn):
        # Identity, migration provenance, and registry publication share one
        # writer snapshot so clone rotation cannot leave a live row paired with
        # an envelope for the old database authority.
        database_uuid = get_retrieval_reference_authority(conn)
        conn.execute(
            f"""
            INSERT INTO {RETRIEVAL_REFERENCES_TABLE}(
                token_digest, kind, target_json, scope_json, revision_json,
                consistency, issued_at, expires_at, revoked_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)
            """,
            (
                digest,
                kind,
                target_json,
                scope_json,
                revision_json,
                consistency,
                issued,
                expires,
            ),
        )
    return encode_reference_envelope(
        kind=kind, database_uuid=database_uuid, token=token
    )


def _failure(code: ReferenceErrorCode | str, message: str) -> ReferenceResult:
    return ReferenceResult(False, error_code=ReferenceErrorCode(code).value, error=message)


def _safe_decode(value: object) -> tuple[ReferenceEnvelope | None, ReferenceResult | None]:
    try:
        return decode_reference_envelope(value), None
    except ReferenceError as exc:
        return None, _failure(exc.code, exc.error)


def _authorize(
    envelope: ReferenceEnvelope,
    *,
    authorization_context: object,
    authorize: AuthorizationCallback | None,
) -> ReferenceResult | None:
    """Run host authorization before any registry SELECT or binding disclosure."""
    if authorize is None or authorization_context is None:
        return _failure(
            ReferenceErrorCode.REFERENCE_FORBIDDEN,
            "reference use is not authorized by the host",
        )
    request = AuthorizationRequest(authorization_context, envelope)
    try:
        allowed = bool(authorize(request.trusted_context, request.envelope))
    except Exception:
        allowed = False
    if not allowed:
        return _failure(
            ReferenceErrorCode.REFERENCE_FORBIDDEN,
            "reference use is not authorized by the host",
        )
    return None


def _authorize_scope(
    envelope: ReferenceEnvelope,
    scope: object,
    *,
    authorization_context: object,
    authorize_scope: ScopeAuthorizationCallback | None,
) -> ReferenceResult | None:
    """Authorize the stored scope before target or revision data is decoded."""
    if authorize_scope is None:
        return _failure(
            ReferenceErrorCode.REFERENCE_FORBIDDEN,
            "reference scope is not authorized by the host",
        )
    try:
        allowed = bool(authorize_scope(authorization_context, envelope, scope))
    except Exception:
        allowed = False
    if not allowed:
        return _failure(
            ReferenceErrorCode.REFERENCE_FORBIDDEN,
            "reference scope is not authorized by the host",
        )
    return None


def resolve_reference(
    conn,
    envelope: object,
    *,
    authorization_context: object = None,
    authorize: AuthorizationCallback | None = None,
    authorize_scope: ScopeAuthorizationCallback | None = None,
    expected_kind: str | None = None,
    expected_scope: object = _UNSET,
    now: float | None = None,
) -> ReferenceResult:
    """Validate a reference in contract order and return its server-side binding.

    The preauthorization callback receives only trusted context and the parsed
    opaque envelope before any registry SELECT. After lookup, the scope callback
    receives only the canonical stored scope; it must authorize that binding
    before target or revision data is decoded or returned.
    """
    parsed, failure = _safe_decode(envelope)
    if failure is not None:
        return failure
    assert parsed is not None

    denied = _authorize(
        parsed,
        authorization_context=authorization_context,
        authorize=authorize,
    )
    if denied is not None:
        return denied

    try:
        database_uuid = get_retrieval_reference_authority(conn)
    except (DatabaseIdentityError, RetrievalReferenceSchemaError):
        return _failure(
            ReferenceErrorCode.REFERENCE_INVALID,
            "reference authority identity is unavailable",
        )
    if parsed.database_uuid != database_uuid:
        return _failure(
            ReferenceErrorCode.REFERENCE_DATABASE_MISMATCH,
            "reference belongs to a different database authority",
        )

    digest = reference_token_digest(parsed.token)
    row = conn.execute(
        f"""
        SELECT token_digest, kind, scope_json, consistency,
               issued_at, expires_at, revoked_at
        FROM {RETRIEVAL_REFERENCES_TABLE}
        WHERE token_digest = ?
        """,
        (digest,),
    ).fetchone()
    if row is None:
        return _failure(
            ReferenceErrorCode.REFERENCE_FORBIDDEN,
            "reference scope is not authorized by the host",
        )

    try:
        scope_json = str(row[2])
        authorization_scope = json.loads(scope_json)
        if canonical_json(authorization_scope) != scope_json:
            raise ValueError("non-canonical registry JSON")
    except (TypeError, ValueError, json.JSONDecodeError, ReferenceError):
        return _failure(
            ReferenceErrorCode.REFERENCE_FORBIDDEN,
            "reference scope is not authorized by the host",
        )

    scope_denied = _authorize_scope(
        parsed,
        authorization_scope,
        authorization_context=authorization_context,
        authorize_scope=authorize_scope,
    )
    if scope_denied is not None:
        return scope_denied

    if str(row[1]) != parsed.kind:
        return _failure(
            ReferenceErrorCode.REFERENCE_KIND_MISMATCH,
            "reference kind does not match its registry binding",
        )

    try:
        current = _finite_time(time.time() if now is None else now, "now")
    except ReferenceError as exc:
        return _failure(exc.code, exc.error)
    try:
        issued_at = _finite_time(row[4], "issued_at")
        expires_at = (
            None if row[5] is None else _finite_time(row[5], "expires_at")
        )
        if expires_at is not None and expires_at < issued_at:
            raise ReferenceError(
                ReferenceErrorCode.REFERENCE_STALE,
                "reference binding is no longer valid",
            )
    except ReferenceError:
        return _failure(
            ReferenceErrorCode.REFERENCE_STALE,
            "reference binding is no longer valid",
        )
    if row[6] is not None:
        return _failure(
            ReferenceErrorCode.REFERENCE_NOT_FOUND,
            "reference was not found",
        )
    if expires_at is not None and current >= expires_at:
        return _failure(
            ReferenceErrorCode.REFERENCE_STALE,
            "reference has expired",
        )

    if expected_kind is not None:
        try:
            expected = _coerce_kind(expected_kind)
        except ReferenceError as exc:
            return _failure(exc.code, exc.error)
        if str(row[1]) != expected:
            return _failure(
                ReferenceErrorCode.REFERENCE_KIND_MISMATCH,
                "reference kind does not match the requested adapter kind",
            )

    if expected_scope is not _UNSET:
        try:
            expected_scope_json = canonical_json(expected_scope)
        except ReferenceError as exc:
            return _failure(exc.code, exc.error)
        if scope_json != expected_scope_json:
            return _failure(
                ReferenceErrorCode.REFERENCE_SCOPE_MISMATCH,
                "reference scope does not match the requested scope",
            )

    try:
        binding = conn.execute(
            f"""
            SELECT registry.target_json, registry.revision_json
            FROM {RETRIEVAL_REFERENCES_TABLE} AS registry
            JOIN metadata AS authority
              ON authority.key = ? AND authority.value = ?
            WHERE registry.token_digest = ?
              AND registry.revoked_at IS NULL
              AND (registry.expires_at IS NULL OR registry.expires_at > ?)
            """,
            (DATABASE_UUID_METADATA_KEY, parsed.database_uuid, digest, current),
        ).fetchone()
        if binding is None:
            return _failure(
                ReferenceErrorCode.REFERENCE_NOT_FOUND,
                "reference was not found",
            )
        target = json.loads(str(binding[0]))
        scope = json.loads(scope_json)
        revision = json.loads(str(binding[1]))
        if (
            canonical_json(target) != str(binding[0])
            or canonical_json(revision) != str(binding[1])
        ):
            raise ValueError("non-canonical registry JSON")
    except (TypeError, ValueError, json.JSONDecodeError, ReferenceError):
        return _failure(
            ReferenceErrorCode.REFERENCE_STALE,
            "reference binding is no longer valid",
        )

    return ReferenceResult(
        True,
        record=ReferenceRecord(
            token_digest=str(row[0]),
            kind=str(row[1]),
            target=target,
            scope=scope,
            revision=revision,
            consistency=str(row[3]),
            issued_at=issued_at,
            expires_at=expires_at,
            revoked_at=None,
        ),
    )


def resolve_reference_or_raise(conn, envelope: object, **kwargs: object) -> ReferenceRecord:
    result = resolve_reference(conn, envelope, **kwargs)
    if not result.ok:
        raise ReferenceError(
            result.error_code or ReferenceErrorCode.REFERENCE_INVALID.value,
            result.error or "reference resolution failed",
        )
    assert result.record is not None
    return result.record


def revoke_reference(
    conn,
    envelope: object,
    *,
    authorization_context: object = None,
    authorize: AuthorizationCallback | None = None,
    authorize_scope: ScopeAuthorizationCallback | None = None,
    revoked_at: float | None = None,
) -> ReferenceResult:
    """Authorize and revoke a V1 row without exposing its target or revision."""
    parsed, failure = _safe_decode(envelope)
    if failure is not None:
        return failure
    assert parsed is not None

    denied = _authorize(
        parsed,
        authorization_context=authorization_context,
        authorize=authorize,
    )
    if denied is not None:
        return denied

    try:
        timestamp = _finite_time(
            time.time() if revoked_at is None else revoked_at, "revoked_at"
        )
    except ReferenceError as exc:
        return _failure(exc.code, exc.error)
    digest = reference_token_digest(parsed.token)
    with _registry_write_transaction(conn):
        try:
            database_uuid = get_retrieval_reference_authority(conn)
        except (DatabaseIdentityError, RetrievalReferenceSchemaError):
            return _failure(
                ReferenceErrorCode.REFERENCE_INVALID,
                "reference authority identity is unavailable",
            )
        if parsed.database_uuid != database_uuid:
            return _failure(
                ReferenceErrorCode.REFERENCE_DATABASE_MISMATCH,
                "reference belongs to a different database authority",
            )
        row = conn.execute(
            f"""
            SELECT kind, scope_json, revoked_at
            FROM {RETRIEVAL_REFERENCES_TABLE}
            WHERE token_digest = ?
            """,
            (digest,),
        ).fetchone()
        if row is None:
            return _failure(
                ReferenceErrorCode.REFERENCE_FORBIDDEN,
                "reference scope is not authorized by the host",
            )
        try:
            scope = json.loads(str(row[1]))
            if canonical_json(scope) != str(row[1]):
                raise ValueError("non-canonical registry JSON")
        except (TypeError, ValueError, json.JSONDecodeError, ReferenceError):
            return _failure(
                ReferenceErrorCode.REFERENCE_FORBIDDEN,
                "reference scope is not authorized by the host",
            )
        scope_denied = _authorize_scope(
            parsed,
            scope,
            authorization_context=authorization_context,
            authorize_scope=authorize_scope,
        )
        if scope_denied is not None:
            return scope_denied
        if row[2] is not None:
            return _failure(
                ReferenceErrorCode.REFERENCE_NOT_FOUND,
                "reference was not found",
            )
        if str(row[0]) != parsed.kind:
            return _failure(
                ReferenceErrorCode.REFERENCE_KIND_MISMATCH,
                "reference kind does not match its registry binding",
            )
        cursor = conn.execute(
            f"""
            UPDATE {RETRIEVAL_REFERENCES_TABLE}
            SET revoked_at = ?
            WHERE token_digest = ? AND revoked_at IS NULL
            """,
            (timestamp, digest),
        )
        if cursor.rowcount != 1:
            return _failure(
                ReferenceErrorCode.REFERENCE_NOT_FOUND,
                "reference was not found",
            )
    return ReferenceResult(True)


def rotate_explicit_clone(conn, **kwargs: object) -> str:
    """Explicit clone boundary: rotate UUID and revoke copied references atomically."""
    return rotate_database_uuid(conn, **kwargs)


# Public aliases used by callers that name the operation as a clone rather than
# as a database-UUID rotation.
rotate_clone = rotate_explicit_clone
rotate_database_identity = rotate_explicit_clone


# Public aliases used by callers that name the operation as a retrieval rather
# than as a registry implementation detail.
issue_retrieval_reference = issue_reference
resolve_retrieval_reference = resolve_reference
revoke_retrieval_reference = revoke_reference


class ReferenceRegistry:
    """Small connection-bound facade for the V1 primitives."""

    def __init__(self, conn):
        self.conn = conn

    def issue(self, kind: str, target: object, scope: object, revision: object, **kwargs: object) -> ReferenceEnvelope:
        return issue_reference(self.conn, kind, target, scope, revision, **kwargs)

    def resolve(self, envelope: object, **kwargs: object) -> ReferenceResult:
        return resolve_reference(self.conn, envelope, **kwargs)

    def revoke(self, envelope: object, **kwargs: object) -> ReferenceResult:
        return revoke_reference(self.conn, envelope, **kwargs)

    def rotate_clone(self, **kwargs: object) -> str:
        return rotate_explicit_clone(self.conn, **kwargs)


__all__ = [
    "AuthorizationCallback",
    "AuthorizationRequest",
    "DatabaseIdentityError",
    "DATABASE_UUID_MIGRATION",
    "IssueAuthorizationCallback",
    "ReferenceEnvelope",
    "ReferenceError",
    "ReferenceErrorCode",
    "ReferenceKind",
    "ReferenceConsistency",
    "ReferenceRecord",
    "ReferenceRegistry",
    "ReferenceResult",
    "ScopeAuthorizationCallback",
    "canonical_json",
    "decode_envelope",
    "decode_reference_envelope",
    "encode_envelope",
    "encode_reference_envelope",
    "generate_reference_token",
    "issue_reference",
    "issue_retrieval_reference",
    "reference_token_digest",
    "resolve_reference",
    "resolve_retrieval_reference",
    "resolve_reference_or_raise",
    "revoke_reference",
    "revoke_retrieval_reference",
    "rotate_clone",
    "rotate_database_identity",
    "rotate_explicit_clone",
    "serialize_envelope",
    "serialize_reference_envelope",
]
