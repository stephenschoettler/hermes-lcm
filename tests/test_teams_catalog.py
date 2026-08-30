"""The Teams catalog: created only on an explicit enable, and recognised.

Two properties carry this phase. A store that never enables Teams must end up
with no Teams tables at all -- that is what keeps a single-user install
untouched by any of this. And a store that DOES enable must still be repairable,
which it is not unless the family prefix is on db_bootstrap's allowlist.
"""

from __future__ import annotations

import sqlite3

import pytest

from hermes_lcm import db_bootstrap
from hermes_lcm.teams import catalog


@pytest.fixture()
def store(tmp_path) -> sqlite3.Connection:
    conn = sqlite3.connect(tmp_path / "lcm.db")
    db_bootstrap.configure_connection(conn)
    try:
        yield conn
    finally:
        conn.close()


def test_a_store_that_never_enables_teams_has_no_catalog(
    store: sqlite3.Connection,
) -> None:
    assert catalog.teams_catalog_exists(store) is False
    tables = {
        row[0]
        for row in store.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert not any(name.startswith("lcm_teams") for name in tables)


def test_checking_for_the_catalog_creates_nothing(store: sqlite3.Connection) -> None:
    """Runs on inspection paths; must not materialise what it reports on."""
    before = {
        row[0]
        for row in store.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    catalog.teams_catalog_exists(store)
    catalog.verify_teams_catalog(store)
    after = {
        row[0]
        for row in store.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert before == after


def test_ensure_creates_the_whole_family(store: sqlite3.Connection) -> None:
    created = catalog.ensure_teams_catalog(store)

    assert set(created) == set(catalog.TEAMS_TABLES)
    assert catalog.teams_catalog_exists(store) is True
    assert catalog.verify_teams_catalog(store) == []


def test_ensure_is_idempotent(store: sqlite3.Connection) -> None:
    catalog.ensure_teams_catalog(store)
    catalog.ensure_teams_catalog(store)
    assert catalog.verify_teams_catalog(store) == []


def test_a_missing_table_is_reported_as_a_defect(store: sqlite3.Connection) -> None:
    catalog.ensure_teams_catalog(store)
    store.execute("DROP TABLE lcm_teams_memberships")
    store.commit()

    assert catalog.verify_teams_catalog(store) == [
        "missing table:lcm_teams_memberships"
    ]


def test_every_table_carries_the_allowlisted_family_prefix() -> None:
    """Repair tooling recognises families by prefix, not by table name."""
    assert all(name.startswith("lcm_teams") for name in catalog.TEAMS_TABLES)
    assert "lcm_teams" in db_bootstrap._KNOWN_FEATURE_TABLE_PREFIXES


def test_an_unknown_tenant_reads_as_zero_rather_than_raising(
    store: sqlite3.Connection,
) -> None:
    """A context belonging to no tenant fails on its principal, more precisely."""
    catalog.ensure_teams_catalog(store)

    revisions = catalog.read_revisions(store, "tenant-nobody")

    assert revisions == catalog.CatalogRevisions(0, 0, 0)


def test_bumping_a_revision_makes_a_previously_issued_context_stale(
    store: sqlite3.Connection,
) -> None:
    """Revocation IS a revision bump; this is the operation that expires."""
    catalog.ensure_teams_catalog(store)
    before = catalog.read_revisions(store, "tenant-1")

    catalog.bump_revision(store, "tenant-1", "revocation_epoch")
    after = catalog.read_revisions(store, "tenant-1")

    assert before.revocation_epoch == 0
    assert after.revocation_epoch == 1
    # The other counters are untouched -- revoking is not a policy change.
    assert after.policy_revision == before.policy_revision
    assert after.membership_revision == before.membership_revision


def test_each_revision_counter_moves_independently(store: sqlite3.Connection) -> None:
    catalog.ensure_teams_catalog(store)
    for field in ("policy_revision", "membership_revision", "revocation_epoch"):
        assert catalog.bump_revision(store, "tenant-1", field) == 1

    assert catalog.read_revisions(store, "tenant-1") == catalog.CatalogRevisions(
        1, 1, 1
    )


def test_an_unknown_revision_field_is_refused(store: sqlite3.Connection) -> None:
    """The field name reaches an f-string, so it is checked rather than trusted."""
    catalog.ensure_teams_catalog(store)

    with pytest.raises(ValueError, match="unknown revision field"):
        catalog.bump_revision(store, "tenant-1", "revocation_epoch = 0 --")


def test_a_teams_store_is_still_classified_as_repairable(
    store: sqlite3.Connection,
) -> None:
    """Without the prefix allowlist entry, repair refuses a Teams store.

    classify_version_mismatch reads extra tables by family prefix; an
    unrecognised family means "a newer build owns this database", which is the
    one answer that stops the repair path running.
    """
    db_bootstrap.ensure_metadata_table(store)
    catalog.ensure_teams_catalog(store)

    extras = [
        row[0]
        for row in store.execute("SELECT name FROM sqlite_master WHERE type='table'")
        if str(row[0]).startswith("lcm_teams")
    ]
    assert extras, "the catalog should have created tables to classify"
    assert all(
        any(name.startswith(prefix) for prefix in db_bootstrap._KNOWN_FEATURE_TABLE_PREFIXES)
        for name in extras
    )
