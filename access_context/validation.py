"""Deterministic validation and delegation derivation for ``AccessContextV1``."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from enum import Enum
from typing import Iterable

from .denials import Decision, DenialReason
from .model import ACCESS_CONTEXT_CONTRACT_REVISION, AccessContextV1, ScopeMismatchError


class ValidationStage(str, Enum):
    CONTEXT_PRESENT = "context_present"
    CONTEXT_WELL_FORMED = "context_well_formed"
    REVISION_SUPPORTED = "revision_supported"
    NOT_EXPIRED = "not_expired"
    NOT_REVOKED = "not_revoked"
    OWNERSHIP_CURRENT = "ownership_current"
    LEASE_CURRENT = "lease_current"
    SCOPE_PERMITTED = "scope_permitted"
    TARGET_RESOLUTION = "target_resolution"


VALIDATION_ORDER: tuple[ValidationStage, ...] = (
    ValidationStage.CONTEXT_PRESENT,
    ValidationStage.CONTEXT_WELL_FORMED,
    ValidationStage.REVISION_SUPPORTED,
    ValidationStage.NOT_EXPIRED,
    ValidationStage.NOT_REVOKED,
    ValidationStage.OWNERSHIP_CURRENT,
    ValidationStage.LEASE_CURRENT,
    ValidationStage.SCOPE_PERMITTED,
    ValidationStage.TARGET_RESOLUTION,
)


class ResolutionMode(str, Enum):
    STANDARD_UNMANAGED = "standard_unmanaged"
    FAIL_CLOSED = "fail_closed"
    ENFORCING = "enforcing"


def resolve_mode(carrier_context: AccessContextV1 | None, teams_enabled: bool) -> ResolutionMode:
    """Resolve the four-state absent/present-carrier compatibility matrix."""

    if not teams_enabled:
        return ResolutionMode.STANDARD_UNMANAGED
    return ResolutionMode.ENFORCING if carrier_context is not None else ResolutionMode.FAIL_CLOSED


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _scope_values(required_scope: str | Iterable[str] | None) -> frozenset[str]:
    if required_scope is None:
        return frozenset()
    if isinstance(required_scope, str):
        return frozenset({required_scope})
    return frozenset(str(item) for item in required_scope)


def _invalid_fields(context: AccessContextV1) -> tuple[str, ...]:
    required = (
        "context_id",
        "request_id",
        "source_kind",
        "authenticated_transport",
        "deployment_id",
        "tenant_id",
        "principal_id",
        "profile_id",
        "profile_incarnation",
        "session_id",
        "session_owner_principal_id",
        "conversation_id",
        "conversation_lane",
        "read_policy_ref",
        "lease_id",
    )
    return tuple(name for name in required if not getattr(context, name))


def validate(
    context: AccessContextV1 | None,
    *,
    required_scope: str | Iterable[str] | None = None,
    now: datetime,
    revoked: bool = False,
    revoked_context_ids: Iterable[str] | None = None,
    current_policy_revision: int | None = None,
    current_membership_revision: int | None = None,
    current_revocation_epoch: int | None = None,
    current_ownership_generation: int | None = None,
    current_lease_id: str | None = None,
    current_lease_generation: int | None = None,
    expected_principal_id: str | None = None,
    expected_request_id: str | None = None,
    expected_context_id: str | None = None,
    expected_lease_id: str | None = None,
    requested_collections: Iterable[str] | str | None = None,
    target_allowed: bool | None = None,
) -> Decision:
    """Walk :data:`VALIDATION_ORDER` and return the first failed stage.

    All state and time are injected.  No module, thread, or task global stores
    an active context, which keeps concurrent profile validations independent.
    """

    now = _utc(now)
    # Normalised through the same string-or-iterable helper as required_scope.
    # A bare "ctx-human" satisfies the declared Iterable[str], and frozenset()
    # would shred it into single characters -- so the revoked context would
    # match nothing and be ALLOWED. The failure is silent and inverts the stage.
    revoked_ids = _scope_values(revoked_context_ids)
    required = _scope_values(required_scope)

    for stage in VALIDATION_ORDER:
        if stage is ValidationStage.CONTEXT_PRESENT:
            if context is None:
                return Decision.deny(DenialReason.CONTEXT_MISSING)
        elif stage is ValidationStage.CONTEXT_WELL_FORMED:
            assert context is not None
            missing = _invalid_fields(context)
            if missing or context.issued_at >= context.expires_at:
                return Decision.deny(
                    DenialReason.CONTEXT_INVALID,
                    context_id=context.context_id,
                    request_id=context.request_id,
                )
        elif stage is ValidationStage.REVISION_SUPPORTED:
            assert context is not None
            if context.contract_revision != ACCESS_CONTEXT_CONTRACT_REVISION:
                return Decision.deny(
                    DenialReason.CONTEXT_UNSUPPORTED_VERSION,
                    context_id=context.context_id,
                )
        elif stage is ValidationStage.NOT_EXPIRED:
            assert context is not None
            # Both ends of the validity window. A future-dated envelope is as
            # unusable as a lapsed one -- issued_at bounds delegation lifetime,
            # so accepting now < issued_at would honour authority that has not
            # begun. CONTEXT_EXPIRED covers both ends rather than adding a
            # reason: the taxonomy is frozen by #482, and the public projection
            # deliberately does not distinguish before-window from after-window.
            if now >= context.expires_at or now < context.issued_at:
                return Decision.deny(
                    DenialReason.CONTEXT_EXPIRED,
                    context_id=context.context_id,
                    request_id=context.request_id,
                )
        elif stage is ValidationStage.NOT_REVOKED:
            assert context is not None
            if (
                revoked
                or context.context_id in revoked_ids
                or (
                    current_revocation_epoch is not None
                    and current_revocation_epoch != context.revocation_epoch
                )
                or (
                    current_membership_revision is not None
                    and current_membership_revision != context.membership_revision
                )
                or (
                    current_policy_revision is not None
                    and current_policy_revision != context.policy_revision
                )
            ):
                return Decision.deny(
                    DenialReason.CONTEXT_REVOKED,
                    context_id=context.context_id,
                    policy_revision=context.policy_revision,
                    membership_revision=context.membership_revision,
                    revocation_epoch=context.revocation_epoch,
                )
        elif stage is ValidationStage.OWNERSHIP_CURRENT:
            assert context is not None
            if (
                current_ownership_generation is not None
                and current_ownership_generation != context.ownership_generation
            ):
                return Decision.deny(
                    DenialReason.OWNERSHIP_CHANGED,
                    context_id=context.context_id,
                    ownership_generation=context.ownership_generation,
                    expected_ownership_generation=current_ownership_generation,
                )
            if expected_principal_id is not None and expected_principal_id != context.principal_id:
                return Decision.deny(
                    DenialReason.OWNERSHIP_CHANGED,
                    context_id=context.context_id,
                    principal_id=context.principal_id,
                )
            if expected_context_id is not None and expected_context_id != context.context_id:
                return Decision.deny(DenialReason.CONTEXT_INVALID, request_id=context.request_id)
            if expected_request_id is not None and expected_request_id != context.request_id:
                return Decision.deny(DenialReason.CONTEXT_INVALID, context_id=context.context_id)
            if required and ("owner_only" in required or "owner" in required):
                if context.session_owner_principal_id != context.principal_id:
                    return Decision.deny(
                        DenialReason.OWNERSHIP_CHANGED,
                        context_id=context.context_id,
                        principal_id=context.principal_id,
                    )
        elif stage is ValidationStage.LEASE_CURRENT:
            assert context is not None
            if (
                expected_lease_id is not None
                and expected_lease_id != context.lease_id
            ) or (
                current_lease_id is not None
                and current_lease_id != context.lease_id
            ) or (
                current_lease_generation is not None
                and current_lease_generation != context.lease_generation
            ):
                return Decision.deny(
                    DenialReason.LEASE_STALE,
                    context_id=context.context_id,
                    lease_id=context.lease_id,
                    lease_generation=context.lease_generation,
                )
        elif stage is ValidationStage.SCOPE_PERMITTED:
            assert context is not None
            if requested_collections is not None:
                requested_collections_set = _scope_values(requested_collections)
                parent_collections = context.collection_allowlist
                if parent_collections and not requested_collections_set <= parent_collections:
                    return Decision.deny(
                        DenialReason.SCOPE_MISMATCH,
                        context_id=context.context_id,
                        policy_revision=context.policy_revision,
                    )
            if not required <= context.operation_allowlist:
                return Decision.deny(
                    DenialReason.SCOPE_FORBIDDEN,
                    context_id=context.context_id,
                    policy_revision=context.policy_revision,
                )
        elif stage is ValidationStage.TARGET_RESOLUTION:
            if target_allowed is False:
                return Decision.deny(DenialReason.TARGET_NOT_FOUND_OR_FORBIDDEN)

    assert context is not None
    return Decision.allow(context_id=context.context_id, request_id=context.request_id)


def derive_child(
    parent: AccessContextV1,
    *,
    operations: Iterable[str] | str,
    collections: Iterable[str] | str | None = None,
    audience: Iterable[str] | str | None = None,
    expires_at: datetime | str,
    profile_id: str | None = None,
    session_id: str | None = None,
    conversation_id: str | None = None,
    policy_revision: int | None = None,
    membership_revision: int | None = None,
    revocation_epoch: int | None = None,
    current_policy_revision: int | None = None,
    current_membership_revision: int | None = None,
    current_revocation_epoch: int | None = None,
    child_context_id: str | None = None,
    child_request_id: str | None = None,
) -> AccessContextV1:
    """Derive delegated authority by intersecting every authority boundary."""

    if not parent.context_id:
        raise ScopeMismatchError("delegation requires a host-derived parent context")
    for alias, explicit, current in (
        ("policy_revision", policy_revision, current_policy_revision),
        ("membership_revision", membership_revision, current_membership_revision),
        ("revocation_epoch", revocation_epoch, current_revocation_epoch),
    ):
        if explicit is not None and current is not None and explicit != current:
            raise ScopeMismatchError(f"{alias} aliases disagree")
    policy_revision = policy_revision if policy_revision is not None else current_policy_revision
    membership_revision = membership_revision if membership_revision is not None else current_membership_revision
    revocation_epoch = revocation_epoch if revocation_epoch is not None else current_revocation_epoch
    narrowed = parent.narrow(
        operations=operations,
        collections=collections,
        audience=audience,
        expires_at=expires_at,
        profile_id=profile_id,
        session_id=session_id,
        conversation_id=conversation_id,
        policy_revision=policy_revision,
        membership_revision=membership_revision,
        revocation_epoch=revocation_epoch,
    )
    chain = parent.delegation_chain
    if not chain or chain[-1] != parent.context_id:
        chain = chain + (parent.context_id,)
    child_number = len(chain) + 1
    return replace(
        narrowed,
        context_id=child_context_id or f"{parent.context_id}:child:{child_number}",
        request_id=child_request_id or f"{parent.request_id}:child:{child_number}",
        delegation_chain=chain,
        delegated_by=parent.context_id,
    )
