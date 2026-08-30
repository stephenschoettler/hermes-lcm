from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timezone

import pytest

from hermes_lcm.access_context import (
    ACCESS_CONTEXT_CONTRACT_REVISION,
    AccessContextV1,
    ActorType,
    ScopeMismatchError,
    is_subset_of,
    validate,
)
from hermes_lcm.access_context.fixtures import load_context

NOW = datetime(2026, 1, 2, tzinfo=timezone.utc)


def _human() -> AccessContextV1:
    context = load_context("tests/fixtures/access_context_v1/positive/human.json")
    assert context is not None
    return context


def test_model_is_frozen_and_collection_fields_are_immutable() -> None:
    context = _human()
    assert context.contract_revision == ACCESS_CONTEXT_CONTRACT_REVISION
    assert context.actor_type is ActorType.HUMAN
    with pytest.raises(FrozenInstanceError):
        context.principal_id = "other"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        context.grants.add("admin")  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        context.delegation_chain.append("other")  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        context.audience.add("other")  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        context.narrowing.add("operation:admin")  # type: ignore[attr-defined]
    assert "admin" not in context.grants


def test_from_host_is_explicit_and_ids_alone_do_not_validate() -> None:
    context = AccessContextV1.from_host(
        authenticated_transport="host-session",
        context_id="ctx-host",
        request_id="req-host",
        source_kind="host",
        deployment_id="dep-1",
        tenant_id="tenant-1",
        principal_id="principal-a",
        actor_type="service",
        profile_id="profile-a",
        profile_incarnation="profile-a-1",
        session_id="session-a",
        session_owner_principal_id="principal-a",
        conversation_id="conversation-a",
        conversation_lane="test",
        read_policy_ref="policy-1",
        grants=["read"],
        lease_id="lease-a",
        issued_at="2026-01-01T00:00:00Z",
        expires_at="2026-12-31T00:00:00Z",
    )
    assert context.actor_type is ActorType.SERVICE
    assert validate(context, required_scope="read", now=NOW).allowed
    ids_only = AccessContextV1(session_id="session-only", profile_id="profile-only")
    assert validate(ids_only, now=NOW).denial_reason.value == "context_invalid"
    with pytest.raises(ValueError):
        AccessContextV1.from_host(authenticated_transport="", session_id="session-only")


def test_narrow_returns_new_context_and_rejects_every_widening_boundary() -> None:
    parent = _human()
    child = parent.narrow(
        operations=["read"],
        audience=["profile-human"],
        expires_at="2026-06-01T00:00:00Z",
        narrowing=["operation:read"],
    )
    assert child is not parent
    assert child.grants == frozenset({"read"})
    assert child.expires_at < parent.expires_at
    assert is_subset_of(child, parent)

    attempts = (
        {"operations": ["admin"]},
        {"audience": ["profile-other"]},
        {"expires_at": "2027-01-01T00:00:00Z"},
        {"profile_id": "profile-other"},
        {"session_id": "session-other"},
        {"conversation_id": "conversation-other"},
        {"default_write_collection_id": "collection-other"},
        {"policy_revision": 2},
        {"membership_revision": 2},
        {"revocation_epoch": 1},
    )
    for attempt in attempts:
        with pytest.raises(ScopeMismatchError):
            parent.narrow(**attempt)


def test_narrowing_uses_effective_operation_and_collection_bounds() -> None:
    parent = _human().narrow(operations=["read"])
    assert parent.operation_allowlist == frozenset({"read"})
    with pytest.raises(ScopeMismatchError):
        parent.narrow(operations=["write"])
    with pytest.raises(ScopeMismatchError):
        parent.narrow(narrowing=["operation:write"])
    effective_parent = replace(_human(), narrowing=frozenset({"operation:read"}))
    with pytest.raises(ScopeMismatchError):
        effective_parent.narrow(operations=["write"])

    with pytest.raises(ScopeMismatchError):
        _human().narrow(collections=["collection-child"])
    child = _human().narrow(
        collections=["collection-child"],
        default_write_collection_id="collection-child",
    )
    assert child.default_write_collection_id == "collection-child"
    with pytest.raises(ScopeMismatchError):
        _human().narrow(
            collections=["collection-main"],
            narrowing=["collection:collection-other"],
        )


def test_subset_uses_effective_operations_and_transitive_delegation() -> None:
    root = _human()
    one = root.narrow(operations=["read"])
    one = replace(
        one,
        context_id="ctx-one",
        delegation_chain=(root.context_id,),
        delegated_by=root.context_id,
    )
    two = one.narrow(operations=["read"])
    two = replace(
        two,
        context_id="ctx-two",
        delegation_chain=(root.context_id, one.context_id),
        delegated_by=one.context_id,
    )
    assert is_subset_of(two, root)

    widened = replace(
        two,
        context_id="ctx-widened",
        narrowing=two.narrowing | {"operation:write"},
    )
    assert not is_subset_of(widened, one)


def test_from_payload_normalizes_json_collections_and_timestamps() -> None:
    context = _human()
    payload = context.to_payload()
    restored = AccessContextV1.from_payload(payload)
    assert restored == context
    assert isinstance(restored.grants, frozenset)
    assert isinstance(restored.delegation_chain, tuple)
    assert restored.issued_at.tzinfo is not None
