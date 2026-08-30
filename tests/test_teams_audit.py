"""The audit trail is written, and writes the PUBLIC reason.

The catalog created `lcm_teams_audit` and nothing ever wrote to it:
`audit_decision` was a no-op on both policies while 39 call sites dutifully
invoked it. An audit table that is never written is worse than no audit table,
because the schema implies a guarantee the code does not keep.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pytest

from hermes_lcm import db_bootstrap
from hermes_lcm.access_context import DenialReason
from hermes_lcm.access_context.denials import Decision
from hermes_lcm.access_context.model import AccessContextV1
from hermes_lcm.access_policy import TeamsPolicy
from hermes_lcm.teams import catalog


def _context(**overrides) -> AccessContextV1:
    fields = dict(
        authenticated_transport="host-session",
        context_id="ctx-1",
        request_id="req-1",
        source_kind="human",
        deployment_id="dep-1",
        tenant_id="tenant-1",
        principal_id="principal-a",
        profile_id="profile-a",
        profile_incarnation="incarnation-1",
        session_id="session-a",
        session_owner_principal_id="principal-a",
        conversation_id="conversation-a",
        conversation_lane="lane-a",
        read_policy_ref="policy-a",
        lease_id="lease-a",
        issued_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        expires_at=datetime(2027, 1, 1, tzinfo=timezone.utc),
    )
    fields.update(overrides)
    return AccessContextV1.from_host(**fields)


@pytest.fixture()
def store(tmp_path) -> sqlite3.Connection:
    conn = sqlite3.connect(tmp_path / "lcm.db")
    db_bootstrap.configure_connection(conn)
    catalog.ensure_teams_catalog(conn)
    try:
        yield conn
    finally:
        conn.close()


def _sink(conn: sqlite3.Connection):
    def sink(**fields):
        catalog.record_audit_event(conn, occurred_at=1.0, **fields)

    return sink


def test_a_denial_is_recorded(store: sqlite3.Connection) -> None:
    policy = TeamsPolicy(_context(), audit_sink=_sink(store))
    denied = Decision.deny(DenialReason.SCOPE_FORBIDDEN)

    policy.audit_decision(None, "write", denied.denial_reason, denied.public())

    events = catalog.read_audit_events(store)
    assert len(events) == 1
    assert events[0]["allowed"] is False
    assert events[0]["principal_id"] == "principal-a"
    assert events[0]["operation"] == "write"


def test_the_recorded_reason_is_the_public_projection(
    store: sqlite3.Connection,
) -> None:
    """#497 exposes an `audit.*` family, so these rows can leave the store.

    The internal reason distinguishes "forbidden" from "does not exist", which
    is exactly the distinction the public projection exists to collapse. Two
    different internal reasons must be indistinguishable here.
    """
    policy = TeamsPolicy(_context(), audit_sink=_sink(store))
    for reason in (DenialReason.SCOPE_FORBIDDEN, DenialReason.LEASE_STALE):
        denied = Decision.deny(reason)
        policy.audit_decision(None, "write", denied.denial_reason, denied.public())

    recorded = {event["denial_reason"] for event in catalog.read_audit_events(store)}
    assert recorded == {DenialReason.TARGET_NOT_FOUND_OR_FORBIDDEN.value}
    assert DenialReason.SCOPE_FORBIDDEN.value not in recorded
    assert DenialReason.LEASE_STALE.value not in recorded


def test_an_allowed_admin_operation_is_recorded(store: sqlite3.Connection) -> None:
    policy = TeamsPolicy(_context(), audit_sink=_sink(store))

    policy.audit_decision(None, "owner_only", None, Decision.allow().public())

    events = catalog.read_audit_events(store)
    assert len(events) == 1
    assert events[0]["allowed"] is True
    assert events[0]["operation"] == "owner_only"


def test_allowed_reads_are_not_recorded(store: sqlite3.Connection) -> None:
    """Bounded on purpose: reads dominate 39 call sites.

    A row per authorization puts an INSERT in the hot path of every retrieval,
    and this branch has already paid for that mistake once with a 48s->174s
    regression from less. Denials are always kept; allowed reads are not.
    """
    policy = TeamsPolicy(_context(), audit_sink=_sink(store))

    for _ in range(50):
        policy.audit_decision(None, "read", None, Decision.allow().public())

    assert catalog.read_audit_events(store) == []


def test_a_denied_read_is_still_recorded(store: sqlite3.Connection) -> None:
    """The volume bound must not swallow the events that matter."""
    policy = TeamsPolicy(_context(), audit_sink=_sink(store))
    denied = Decision.deny(DenialReason.SCOPE_FORBIDDEN)

    policy.audit_decision(None, "read", denied.denial_reason, denied.public())

    assert len(catalog.read_audit_events(store)) == 1


def test_auditing_never_breaks_the_operation_it_audits(
    store: sqlite3.Connection,
) -> None:
    """A store whose audit table is gone still serves its principals."""
    store.execute("DROP TABLE lcm_teams_audit")
    store.commit()
    policy = TeamsPolicy(_context(), audit_sink=_sink(store))
    denied = Decision.deny(DenialReason.SCOPE_FORBIDDEN)

    policy.audit_decision(None, "write", denied.denial_reason, denied.public())  # no raise


def test_no_sink_is_not_an_error(store: sqlite3.Connection) -> None:
    """A context carrier with no store resolves a policy with no sink."""
    TeamsPolicy(_context()).audit_decision(
        None, "write", DenialReason.SCOPE_FORBIDDEN, Decision.deny(
            DenialReason.SCOPE_FORBIDDEN
        ).public()
    )


def test_events_are_scoped_by_tenant(store: sqlite3.Connection) -> None:
    for tenant in ("tenant-1", "tenant-2"):
        policy = TeamsPolicy(_context(tenant_id=tenant), audit_sink=_sink(store))
        denied = Decision.deny(DenialReason.SCOPE_FORBIDDEN)
        policy.audit_decision(None, "write", denied.denial_reason, denied.public())

    assert len(catalog.read_audit_events(store, tenant_id="tenant-1")) == 1
    assert len(catalog.read_audit_events(store)) == 2
