"""Immutable, host-derived authorization context for the Teams seam.

This module deliberately contains no policy lookup, token handling, or runtime
integration.  It freezes the values that a future host carrier and LCM policy
adapter will exchange without changing the existing single-user engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable, Mapping


ACCESS_CONTEXT_CONTRACT_REVISION = "v1"


class ActorType(str, Enum):
    """Principal kind recorded by the authenticated host."""

    HUMAN = "human"
    AGENT = "agent"
    SERVICE = "service"


class ScopeMismatchError(ValueError):
    """Raised when a requested narrowing would widen an existing context."""

    denial_reason = "scope_mismatch"


def _freeze_strings(value: Iterable[str] | str | None) -> frozenset[str]:
    if value is None:
        return frozenset()
    if isinstance(value, str):
        return frozenset({value})
    return frozenset(str(item) for item in value)


def _freeze_chain(value: Iterable[str] | str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _as_datetime(value: datetime | str) -> datetime:
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, str):
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        raise TypeError("timestamps must be datetime values or ISO-8601 strings")
    if result.tzinfo is None:
        return result.replace(tzinfo=timezone.utc)
    return result


@dataclass(frozen=True)
class AccessContextV1:
    """A complete authority envelope derived from authenticated host state.

    IDs are lineage and routing values, never credentials.  A context made
    from IDs alone is not an authorized context: callers must use
    :meth:`from_host`, :meth:`narrow`, or fixture loading, and validation
    requires the authenticated transport and all host-derived bindings.
    """

    # Identity and envelope
    contract_revision: str = ACCESS_CONTEXT_CONTRACT_REVISION
    context_id: str = ""
    request_id: str = ""
    source_kind: str = ""
    authenticated_transport: str = ""

    # Tenancy and principal
    deployment_id: str = ""
    tenant_id: str = ""
    principal_id: str = ""
    actor_type: ActorType = ActorType.HUMAN
    profile_id: str = ""
    profile_incarnation: str = ""

    # Session and conversation
    session_id: str = ""
    session_owner_principal_id: str = ""
    conversation_id: str = ""
    conversation_lane: str = ""

    # Policy
    default_write_collection_id: str = ""
    read_policy_ref: str = ""
    grants: frozenset[str] = field(default_factory=frozenset)
    policy_revision: int = 0
    membership_revision: int = 0
    revocation_epoch: int = 0

    # Delegation and lifetime
    delegation_chain: tuple[str, ...] = ()
    delegated_by: str = ""
    audience: frozenset[str] = field(default_factory=frozenset)
    issued_at: datetime = field(default_factory=lambda: datetime.min.replace(tzinfo=timezone.utc))
    expires_at: datetime = field(default_factory=lambda: datetime.max.replace(tzinfo=timezone.utc))
    narrowing: frozenset[str] = field(default_factory=frozenset)

    # Ownership and lease
    ownership_generation: int = 0
    lease_id: str = ""
    lease_generation: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "actor_type", ActorType(self.actor_type))
        object.__setattr__(self, "grants", _freeze_strings(self.grants))
        object.__setattr__(self, "delegation_chain", _freeze_chain(self.delegation_chain))
        object.__setattr__(self, "audience", _freeze_strings(self.audience))
        object.__setattr__(self, "narrowing", _freeze_strings(self.narrowing))
        object.__setattr__(self, "issued_at", _as_datetime(self.issued_at))
        object.__setattr__(self, "expires_at", _as_datetime(self.expires_at))

    @classmethod
    def from_host(cls, *, authenticated_transport: str, **fields: Any) -> "AccessContextV1":
        """Construct a context from host-derived fields.

        The explicit transport argument makes the intended construction path
        visible.  Fixture loaders use :meth:`from_payload` instead.
        """

        if not authenticated_transport:
            raise ValueError("authenticated_transport is required for host-derived contexts")
        fields["authenticated_transport"] = authenticated_transport
        fields.setdefault("contract_revision", ACCESS_CONTEXT_CONTRACT_REVISION)
        return cls(**fields)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "AccessContextV1":
        """Build a test context from a JSON-compatible payload."""

        values = dict(payload)
        if "actor_type" in values:
            values["actor_type"] = ActorType(values["actor_type"])
        return cls(**values)

    @property
    def collection_allowlist(self) -> frozenset[str]:
        """Explicit collection restrictions encoded in ``narrowing``."""

        return frozenset(
            token.partition(":")[2]
            for token in self.narrowing
            if token.startswith("collection:") and token.partition(":")[2]
        )

    @property
    def operation_allowlist(self) -> frozenset[str]:
        """Explicit operation restrictions, or the grants when absent."""

        explicit = frozenset(
            token.partition(":")[2]
            for token in self.narrowing
            if token.startswith("operation:") and token.partition(":")[2]
        )
        return explicit or self.grants

    def narrow(
        self,
        *,
        grants: Iterable[str] | str | None = None,
        operations: Iterable[str] | str | None = None,
        collections: Iterable[str] | str | None = None,
        collection_allowlist: Iterable[str] | str | None = None,
        audience: Iterable[str] | str | None = None,
        expires_at: datetime | str | None = None,
        profile_id: str | None = None,
        session_id: str | None = None,
        conversation_id: str | None = None,
        default_write_collection_id: str | None = None,
        narrowing: Iterable[str] | str | None = None,
        policy_revision: int | None = None,
        membership_revision: int | None = None,
        revocation_epoch: int | None = None,
    ) -> "AccessContextV1":
        """Return a new context whose authority is a strict subset of this one."""

        if grants is not None and operations is not None and _freeze_strings(grants) != _freeze_strings(operations):
            raise ScopeMismatchError("grants and operations disagree")
        requested_operations = grants if grants is not None else operations
        parent_operations = self.operation_allowlist
        if requested_operations is None:
            child_grants = self.grants
        else:
            child_grants = _freeze_strings(requested_operations)
            if not child_grants <= parent_operations:
                raise ScopeMismatchError("operations would widen the parent allowlist")

        requested_collections = collections if collections is not None else collection_allowlist
        parent_collections = self.collection_allowlist
        if requested_collections is None:
            child_collections = parent_collections
        else:
            child_collections = _freeze_strings(requested_collections)
            if parent_collections and not child_collections <= parent_collections:
                raise ScopeMismatchError("collections would widen the parent allowlist")

        child_audience = self.audience if audience is None else _freeze_strings(audience)
        if self.audience and not child_audience <= self.audience:
            raise ScopeMismatchError("audience would widen the parent audience")

        child_expiry = self.expires_at if expires_at is None else _as_datetime(expires_at)
        if child_expiry > self.expires_at or child_expiry < self.issued_at:
            raise ScopeMismatchError("expiry would outlive the parent")

        for name, requested, current in (
            ("profile_id", profile_id, self.profile_id),
            ("session_id", session_id, self.session_id),
            ("conversation_id", conversation_id, self.conversation_id),
        ):
            if requested is not None and requested != current:
                raise ScopeMismatchError(f"{name} would change the binding")
        for name, requested, current in (
            ("policy_revision", policy_revision, self.policy_revision),
            ("membership_revision", membership_revision, self.membership_revision),
            ("revocation_epoch", revocation_epoch, self.revocation_epoch),
        ):
            if requested is not None and requested != current:
                raise ScopeMismatchError(f"{name} would change current state")

        requested_narrowing = _freeze_strings(narrowing)
        requested_collection_tokens = frozenset(
            token.partition(":")[2]
            for token in requested_narrowing
            if token.startswith("collection:") and token.partition(":")[2]
        )
        child_collection = self.default_write_collection_id
        if default_write_collection_id is not None:
            if parent_collections and default_write_collection_id not in parent_collections:
                raise ScopeMismatchError("default collection would widen the parent allowlist")
            if (
                not parent_collections
                and requested_collections is None
                and default_write_collection_id != child_collection
            ):
                raise ScopeMismatchError("default collection would change the parent binding")
            effective_collections = child_collections or requested_collection_tokens
            if effective_collections and default_write_collection_id not in effective_collections:
                raise ScopeMismatchError("default collection would fall outside the child allowlist")
            child_collection = default_write_collection_id
        elif (
            (child_collections or requested_collection_tokens)
            and child_collection not in (child_collections or requested_collection_tokens)
        ):
            raise ScopeMismatchError("default collection would fall outside the child allowlist")

        # Narrowing REPLACES a dimension; it does not union with it. Seeding the
        # child with every parent token and adding the child's on top keeps the
        # superseded ones: narrowing operation:{read,write} down to read left
        # operation:write in place, so the "read-only" child's
        # operation_allowlist still passed a write check. Each dimension the
        # caller actually narrows is cleared first -- whether it was named
        # through the typed argument or as an explicit token.
        narrowed_dimensions = {
            token.partition(":")[0]
            for token in requested_narrowing
            if token.partition(":")[2]
        }
        if requested_operations is not None:
            narrowed_dimensions.add("operation")
        if requested_collections is not None:
            narrowed_dimensions.add("collection")
        if audience is not None:
            narrowed_dimensions.add("audience")

        child_narrowing = {
            token
            for token in self.narrowing
            if token.partition(":")[0] not in narrowed_dimensions
        }
        if requested_operations is not None:
            child_narrowing.update(f"operation:{item}" for item in child_grants)
        if requested_collections is not None:
            child_narrowing.update(f"collection:{item}" for item in child_collections)
        if audience is not None:
            child_narrowing.update(f"audience:{item}" for item in child_audience)
        for token in requested_narrowing:
            prefix, _, value = token.partition(":")
            if prefix == "operation" and value not in (
                child_grants if requested_operations is not None else parent_operations
            ):
                raise ScopeMismatchError("narrowing names an operation outside the child allowlist")
            if prefix == "collection" and (
                requested_collections is not None or parent_collections
            ) and value not in child_collections:
                raise ScopeMismatchError("narrowing names a collection outside the child allowlist")
            if prefix == "audience" and self.audience and value not in child_audience:
                raise ScopeMismatchError("narrowing names an audience outside the child audience")
        child_narrowing.update(requested_narrowing)

        return AccessContextV1(
            contract_revision=self.contract_revision,
            context_id=self.context_id,
            request_id=self.request_id,
            source_kind=self.source_kind,
            authenticated_transport=self.authenticated_transport,
            deployment_id=self.deployment_id,
            tenant_id=self.tenant_id,
            principal_id=self.principal_id,
            actor_type=self.actor_type,
            profile_id=self.profile_id,
            profile_incarnation=self.profile_incarnation,
            session_id=self.session_id,
            session_owner_principal_id=self.session_owner_principal_id,
            conversation_id=self.conversation_id,
            conversation_lane=self.conversation_lane,
            default_write_collection_id=child_collection,
            read_policy_ref=self.read_policy_ref,
            grants=child_grants,
            policy_revision=self.policy_revision,
            membership_revision=self.membership_revision,
            revocation_epoch=self.revocation_epoch,
            delegation_chain=self.delegation_chain,
            delegated_by=self.delegated_by,
            audience=child_audience,
            issued_at=self.issued_at,
            expires_at=child_expiry,
            narrowing=frozenset(child_narrowing),
            ownership_generation=self.ownership_generation,
            lease_id=self.lease_id,
            lease_generation=self.lease_generation,
        )

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-compatible representation for shared fixtures."""

        values: dict[str, Any] = {}
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, Enum):
                value = value.value
            elif isinstance(value, (frozenset, tuple)):
                value = sorted(value) if isinstance(value, frozenset) else list(value)
            elif isinstance(value, datetime):
                value = value.isoformat().replace("+00:00", "Z")
            values[name] = value
        return values


def _same_or_unrestricted(child: frozenset[str], parent: frozenset[str]) -> bool:
    return not parent or child <= parent


def is_subset_of(child: AccessContextV1, parent: AccessContextV1) -> bool:
    """Prove that every authority-bearing bound in ``child`` is within parent."""

    if child.contract_revision != parent.contract_revision:
        return False
    for name in (
        "deployment_id",
        "tenant_id",
        "principal_id",
        "actor_type",
        "profile_id",
        "profile_incarnation",
        "session_id",
        "session_owner_principal_id",
        "conversation_id",
        "conversation_lane",
        "read_policy_ref",
        "policy_revision",
        "membership_revision",
        "revocation_epoch",
        "ownership_generation",
        "lease_id",
        "lease_generation",
        "source_kind",
        "authenticated_transport",
    ):
        if getattr(child, name) != getattr(parent, name):
            return False
    if not child.grants <= parent.grants:
        return False
    if not child.operation_allowlist <= parent.operation_allowlist:
        return False
    if not _same_or_unrestricted(child.audience, parent.audience):
        return False
    if child.expires_at > parent.expires_at or child.issued_at < parent.issued_at:
        return False
    if not parent.delegation_chain == child.delegation_chain[: len(parent.delegation_chain)]:
        return False
    if child.context_id != parent.context_id:
        if parent.context_id not in child.delegation_chain:
            return False
    if not parent.narrowing <= child.narrowing:
        return False
    parent_collections = parent.collection_allowlist
    child_collections = child.collection_allowlist
    if parent_collections and not child_collections <= parent_collections:
        return False
    if child_collections and child.default_write_collection_id not in child_collections:
        return False
    if child.default_write_collection_id != parent.default_write_collection_id:
        if not child_collections or child.default_write_collection_id not in child_collections:
            return False
    return True
