"""Catalog accessors for principals, collections and memberships.

These are what the catalog promised and did not have, and their absence is the
reason `TeamsPolicy` decides from the CONTEXT rather than the catalog: with no
way to ask "which collections may this principal read", a shared collection
could not be modelled at all, so the policy could only ever answer the private
case correctly.

The test that matters most is the suspension one. Suspension is deliberately
NON-destructive -- membership rows survive, exactly as `disable_teams` keeps
access_scope stamps -- so an `authorized_collections` that went straight to the
memberships table would let a suspended principal keep every grant it had. The
status check is the whole point.
"""

from __future__ import annotations

import sqlite3

import pytest

from hermes_lcm.teams import catalog


NOW = 1_700_000_000.0


@pytest.fixture()
def conn():
    connection = sqlite3.connect(":memory:")
    catalog.ensure_teams_catalog(connection)
    connection.execute(
        "INSERT INTO lcm_teams_revisions(tenant_id) VALUES('t1')"
    )
    connection.commit()
    yield connection
    connection.close()


def _seed(conn) -> None:
    catalog.provision_principal(conn, principal_id="alice", tenant_id="t1", now=NOW)
    catalog.provision_principal(conn, principal_id="bob", tenant_id="t1", now=NOW)
    catalog.create_collection(conn, collection_id="alice-own", tenant_id="t1", kind="own", now=NOW)
    catalog.create_collection(conn, collection_id="company", tenant_id="t1", kind="shared", now=NOW)
    catalog.grant_membership(conn, principal_id="alice", collection_id="alice-own",
                             grants=["read", "write"], tenant_id="t1", now=NOW)
    catalog.grant_membership(conn, principal_id="alice", collection_id="company",
                             grants=["read"], tenant_id="t1", now=NOW)
    catalog.grant_membership(conn, principal_id="bob", collection_id="company",
                             grants=["read"], tenant_id="t1", now=NOW)


def test_a_principal_reads_its_own_and_the_shared_collection(conn) -> None:
    _seed(conn)
    assert catalog.authorized_collections(conn, "alice") == ("alice-own", "company")
    # bob is in the shared collection and NOT in alice's own -- the property the
    # whole feature exists for, now expressible from the catalog.
    assert catalog.authorized_collections(conn, "bob") == ("company",)


def test_write_grants_are_distinguished_from_read(conn) -> None:
    _seed(conn)
    assert catalog.authorized_collections(conn, "alice", grant="write") == ("alice-own",)
    assert catalog.authorized_collections(conn, "bob", grant="write") == ()


def test_a_suspended_principal_loses_access_without_losing_attribution(conn) -> None:
    """The test this file exists for."""
    _seed(conn)
    assert catalog.authorized_collections(conn, "alice")

    catalog.suspend_principal(conn, principal_id="alice", tenant_id="t1", now=NOW + 1)

    assert catalog.authorized_collections(conn, "alice") == (), (
        "a suspended principal kept its grants"
    )
    # Non-destructive: the rows are still there, which is what a later
    # re-provision and every audit answer depend on.
    assert len(catalog.read_memberships(conn, "alice")) == 2
    assert catalog.read_principal(conn, "alice").status == "suspended"


def test_suspension_bumps_the_revocation_epoch(conn) -> None:
    """#498 wants revocation to block the NEXT operation, not eventually."""
    _seed(conn)
    before = catalog.read_revisions(conn, "t1").revocation_epoch
    catalog.suspend_principal(conn, principal_id="alice", tenant_id="t1", now=NOW + 1)
    assert catalog.read_revisions(conn, "t1").revocation_epoch > before


def test_revoking_a_membership_bumps_the_epoch_too(conn) -> None:
    _seed(conn)
    before = catalog.read_revisions(conn, "t1").revocation_epoch
    assert catalog.revoke_membership(
        conn, principal_id="alice", collection_id="company", tenant_id="t1"
    )
    assert catalog.authorized_collections(conn, "alice") == ("alice-own",)
    assert catalog.read_revisions(conn, "t1").revocation_epoch > before


def test_re_provisioning_reactivates_without_duplicating(conn) -> None:
    _seed(conn)
    catalog.suspend_principal(conn, principal_id="alice", tenant_id="t1", now=NOW + 1)
    catalog.provision_principal(conn, principal_id="alice", tenant_id="t1", now=NOW + 2)

    assert catalog.read_principal(conn, "alice").status == "active"
    # The surviving grants are exactly why suspension is not a delete.
    assert catalog.authorized_collections(conn, "alice") == ("alice-own", "company")
    count = conn.execute(
        "SELECT COUNT(*) FROM lcm_teams_principals WHERE principal_id='alice'"
    ).fetchone()[0]
    assert count == 1


def test_granting_is_idempotent_and_replaces_rather_than_appends(conn) -> None:
    _seed(conn)
    catalog.grant_membership(conn, principal_id="bob", collection_id="company",
                             grants=["read", "write"], tenant_id="t1", now=NOW)
    memberships = catalog.read_memberships(conn, "bob")
    assert len(memberships) == 1
    assert memberships[0].grants == ("read", "write")


def test_an_unknown_principal_is_authorized_for_nothing(conn) -> None:
    assert catalog.authorized_collections(conn, "nobody") == ()
    assert catalog.read_principal(conn, "nobody") is None


def test_archive_has_nowhere_to_live_yet() -> None:
    """Recorded, not worked around.

    The ratified contract is "disable-then-archive, never destructive delete",
    but `status` is CHECK-constrained to active/suspended. Archive therefore
    needs a schema migration, not an accessor, and `principals.archive` cannot
    be honoured until it lands. This pins the gap so it is not quietly forgotten
    or silently mapped onto suspend.
    """
    assert catalog.PRINCIPAL_STATUSES == ("active", "suspended")
    assert "archived" not in catalog.PRINCIPAL_STATUSES
