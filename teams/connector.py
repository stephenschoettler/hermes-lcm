"""The provider-neutral management connector seam (#497).

An external control plane -- ElectricSheep's People & Access reconciler is the
first, but nothing here knows that -- must be able to drive Teams membership
WITHOUT going through the model or a slash command. A plugin slash command
receives only the argument string; the event carrying sender identity is never
passed, so there is no caller identity to authorize against. That is why this is
a typed Python API taking the acting principal as an explicit argument.

## What this module is, and deliberately is not

It is the SEAM. It is not a credential scheme.

The desired-state contract ratifies "authenticated HTTPS pull, no detached
signature", and records that signature verification "is a new auth layer, so it
is FLAGGED, NOT BUILT". Rebuilding key custody is an explicit charter non-goal.
So the connector consults a host-supplied check and has no opinion about what a
credential looks like.

**The check defaults to FAIL-CLOSED.** With nothing wired, every capability is
refused. This is the single most important property in the file: the failure
mode worth designing against is not a connector that rejects a valid caller --
that is loud, and someone fixes it in minutes -- but a connector that accepts
everyone precisely because nobody configured it, which is silent and survives to
production. An unconfigured connector must be useless, not permissive.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping

from ..access_context.denials import DenialReason
from . import catalog


class Capability(str, Enum):
    """The management capability families #497 requires.

    Named as ``family.verb`` so an operator reading an audit row sees the same
    string the control plane sent.
    """

    TEAMS_STATUS = "teams.status"
    TEAMS_ENABLE = "teams.enable"
    TEAMS_DISABLE = "teams.disable"
    PRINCIPALS_PROVISION = "principals.provision"
    PRINCIPALS_SUSPEND = "principals.suspend"
    PRINCIPALS_ARCHIVE = "principals.archive"
    MEMBERSHIPS_GRANT = "memberships.grant"
    MEMBERSHIPS_REVOKE = "memberships.revoke"
    COLLECTIONS_CREATE = "collections.create"
    COLLECTIONS_ARCHIVE = "collections.archive"
    SHARED_MEMORY_SEED_IMPORT = "shared_memory.seed_import"
    REVIEW_SCOPE_OPEN = "review_scope.open"
    REVIEW_SCOPE_CLOSE = "review_scope.close"
    AUDIT_READ = "audit.read"
    MIGRATION_STATUS = "migration.status"
    BACKUP_STATUS = "backup.status"


# There is deliberately no `principals.delete`. The ratified contract says
# removal is "disable-then-archive, never destructive delete" -- an archived
# principal's rows keep their access_scope stamps, which is what a later
# re-provision and every audit answer depend on.
_FORBIDDEN_VERBS = frozenset({"delete", "purge", "destroy"})


class FailureClass(str, Enum):
    """Typed failure taxonomy.

    Distinct classes so a control plane can branch, but note that
    UNAUTHORIZED and NOT_FOUND_OR_FORBIDDEN are deliberately NOT
    distinguishable for a target the caller may not see -- that collapse is the
    non-enumerating projection, and telling them apart is exactly the
    disclosure it exists to prevent.
    """

    UNAUTHENTICATED = "unauthenticated"
    UNAUTHORIZED = "unauthorized"
    CONFLICT = "conflict"
    NOT_FOUND_OR_FORBIDDEN = "not_found_or_forbidden"
    INVALID_REQUEST = "invalid_request"
    UNAVAILABLE = "unavailable"
    NOT_IMPLEMENTED = "not_implemented"


class ConnectorError(Exception):
    """A typed connector failure carrying only its public projection."""

    def __init__(self, failure: FailureClass, message: str = "") -> None:
        super().__init__(message or failure.value)
        self.failure = failure


@dataclass(frozen=True)
class ConnectorRequest:
    """One management request.

    ``request_id`` is the control plane's idempotency key. ``payload`` is the
    request body; its digest is what makes a replay distinguishable from a
    changed request reusing an id.
    """

    request_id: str
    capability: Capability
    acting_principal_id: str
    tenant_id: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    credential: str | None = None

    def digest(self) -> str:
        """A stable digest of the payload.

        ``sort_keys`` matters: two semantically identical bodies that differ
        only in key order must not read as a conflicting reuse of the id.
        """

        body = json.dumps(dict(self.payload), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(body.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ConnectorResult:
    status: str
    capability: str
    request_id: str
    data: Mapping[str, Any] = field(default_factory=dict)
    replayed: bool = False


# A host wires this to whatever it already uses to authenticate the control
# plane. It returns True for an acceptable credential. It is NEVER given the
# payload -- authentication decides who is calling, not what they may do.
CredentialCheck = Callable[[str | None], bool]

CapabilityHandler = Callable[[sqlite3.Connection, ConnectorRequest], Mapping[str, Any]]


class TeamsConnector:
    """The typed management API. One implementation; Phase 6's CLI is a shell.

    Construct with ``credential_check=None`` -- the default -- and every
    capability is refused. A host that wants a working connector must supply
    one deliberately.
    """

    def __init__(
        self,
        connection: sqlite3.Connection,
        *,
        credential_check: CredentialCheck | None = None,
        handlers: Mapping[Capability, CapabilityHandler] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._conn = connection
        self._credential_check = credential_check
        self._handlers = dict(handlers or {})
        self._clock = clock

    # -- authentication ---------------------------------------------------

    @property
    def is_wired(self) -> bool:
        """Whether a host supplied a credential check at all."""

        return self._credential_check is not None

    def _authenticate(self, request: ConnectorRequest) -> None:
        if self._credential_check is None:
            # THE fail-closed branch. Not an error path -- the designed
            # behaviour of an unconfigured connector.
            raise ConnectorError(
                FailureClass.UNAUTHENTICATED,
                "no connector credential check is wired; every capability is refused",
            )
        try:
            accepted = bool(self._credential_check(request.credential))
        except Exception as exc:  # noqa: BLE001
            # A check that raises is a check that did not say yes. Treating an
            # exception as anything but a refusal would make a broken host
            # config fail OPEN.
            raise ConnectorError(
                FailureClass.UNAUTHENTICATED, "credential check failed"
            ) from exc
        if not accepted:
            raise ConnectorError(FailureClass.UNAUTHENTICATED, "credential rejected")

    # -- idempotency ------------------------------------------------------

    def _prior(self, request: ConnectorRequest) -> tuple[str, str | None] | None:
        try:
            row = self._conn.execute(
                "SELECT payload_digest, result_json FROM lcm_teams_requests"
                " WHERE request_id = ?",
                (str(request.request_id),),
            ).fetchone()
        except sqlite3.Error:
            return None
        return (str(row[0]), row[1]) if row is not None else None

    def _remember(self, request: ConnectorRequest, data: Mapping[str, Any]) -> None:
        try:
            self._conn.execute(
                "INSERT OR REPLACE INTO lcm_teams_requests(request_id, payload_digest,"
                " capability, tenant_id, principal_id, recorded_at, result_json)"
                " VALUES(?,?,?,?,?,?,?)",
                (
                    str(request.request_id),
                    request.digest(),
                    request.capability.value,
                    str(request.tenant_id or ""),
                    str(request.acting_principal_id or ""),
                    float(self._clock()),
                    json.dumps(dict(data), sort_keys=True),
                ),
            )
            self._conn.commit()
        except sqlite3.Error:
            # The effect already happened. Losing the ledger row costs a replay
            # its cached answer; it must not cost the caller the operation.
            return

    # -- audit ------------------------------------------------------------

    def _audit(
        self, request: ConnectorRequest, *, allowed: bool, failure: FailureClass | None
    ) -> None:
        catalog.record_audit_event(
            self._conn,
            occurred_at=float(self._clock()),
            tenant_id=str(request.tenant_id or ""),
            principal_id=str(request.acting_principal_id or ""),
            operation=request.capability.value,
            allowed=allowed,
            # The PUBLIC class, never an internal reason, and never any part of
            # the payload -- audit rows reach tenant admins through audit.read.
            denial_reason=failure.value if failure is not None else None,
            detail=f"request_id={request.request_id}",
        )

    # -- the single entry point -------------------------------------------

    def execute(self, request: ConnectorRequest) -> ConnectorResult:
        """Authenticate, de-duplicate, dispatch, audit.

        Ordering is the property: authentication precedes every observable
        effect, including the idempotency lookup. Answering a replay before
        authenticating would let an unauthenticated caller confirm that a
        request id exists.
        """

        if not isinstance(request.capability, Capability):
            raise ConnectorError(FailureClass.INVALID_REQUEST, "unknown capability")
        if not str(request.request_id or "").strip():
            raise ConnectorError(FailureClass.INVALID_REQUEST, "request_id is required")
        if not str(request.acting_principal_id or "").strip():
            raise ConnectorError(
                FailureClass.INVALID_REQUEST, "acting_principal_id is required"
            )
        if request.capability.value.rsplit(".", 1)[-1] in _FORBIDDEN_VERBS:
            raise ConnectorError(FailureClass.INVALID_REQUEST, "destructive verb")

        try:
            self._authenticate(request)
        except ConnectorError as exc:
            self._audit(request, allowed=False, failure=exc.failure)
            raise

        prior = self._prior(request)
        if prior is not None:
            stored_digest, stored_result = prior
            if stored_digest != request.digest():
                # Same id, different body. Rejecting is the point: applying it
                # would silently overwrite the first effect, and applying it as
                # a second effect would break the idempotency guarantee.
                self._audit(request, allowed=False, failure=FailureClass.CONFLICT)
                raise ConnectorError(
                    FailureClass.CONFLICT,
                    "request_id was already used with a different payload",
                )
            self._audit(request, allowed=True, failure=None)
            return ConnectorResult(
                status="ok",
                capability=request.capability.value,
                request_id=request.request_id,
                data=json.loads(stored_result) if stored_result else {},
                replayed=True,
            )

        handler = self._handlers.get(request.capability)
        if handler is None:
            # Honest: the seam is built, this family is not. NOT_IMPLEMENTED is
            # a refusal, so an unbuilt family cannot be mistaken for a granted
            # one by a control plane that only checks for an exception.
            self._audit(request, allowed=False, failure=FailureClass.NOT_IMPLEMENTED)
            raise ConnectorError(
                FailureClass.NOT_IMPLEMENTED,
                f"no handler registered for {request.capability.value}",
            )

        try:
            data = handler(self._conn, request)
        except ConnectorError as exc:
            self._audit(request, allowed=False, failure=exc.failure)
            raise
        except sqlite3.Error as exc:
            self._audit(request, allowed=False, failure=FailureClass.UNAVAILABLE)
            raise ConnectorError(FailureClass.UNAVAILABLE, "store unavailable") from exc

        self._remember(request, data)
        self._audit(request, allowed=True, failure=None)
        return ConnectorResult(
            status="ok",
            capability=request.capability.value,
            request_id=request.request_id,
            data=dict(data),
        )


# The public denial projection a caller sees. Mapped rather than exposed
# directly so the connector cannot leak an internal DenialReason that
# distinguishes "forbidden" from "absent".
_PUBLIC_DENIAL = {
    FailureClass.UNAUTHORIZED: DenialReason.TARGET_NOT_FOUND_OR_FORBIDDEN,
    FailureClass.NOT_FOUND_OR_FORBIDDEN: DenialReason.TARGET_NOT_FOUND_OR_FORBIDDEN,
}


def public_denial_reason(failure: FailureClass) -> DenialReason | None:
    return _PUBLIC_DENIAL.get(failure)
