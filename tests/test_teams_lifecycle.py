"""Durable Teams enable/disable lifecycle.

The defect these pin is not "the flag is lost on restart" -- it is worse than
that. ``enable_teams`` stamps rows in committed batches and records the flag
only after the whole backfill succeeds, so an enable that dies partway leaves
per-owner stamps with no recorded decision, no restart required. Reading that
state as "Teams is off" hands a permissive policy real scoped data.
"""

from __future__ import annotations

import sqlite3

import pytest

from hermes_lcm import scope_storage
from hermes_lcm.access_policy import FailClosedPolicy, TrustedOwnerPolicy, policy_for_engine
from hermes_lcm.scope_storage import (
    TEAMS_ENABLED_METADATA_KEY,
    access_scope_stamps_exist,
    persist_teams_enabled,
    read_persisted_teams_enabled,
    resolve_startup_teams_state,
)


@pytest.fixture()
def store(tmp_path) -> sqlite3.Connection:
    conn = sqlite3.connect(tmp_path / "lcm.db")
    conn.executescript(
        """
        CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE messages(
            store_id INTEGER PRIMARY KEY,
            session_id TEXT,
            content TEXT,
            access_scope TEXT
        );
        """
    )
    conn.commit()
    try:
        yield conn
    finally:
        conn.close()


def _stamp(conn: sqlite3.Connection, scope: str = "principal-a") -> None:
    conn.execute(
        "INSERT INTO messages(session_id, content, access_scope) VALUES('s', 'x', ?)",
        (scope,),
    )
    conn.commit()


class _Engine:
    """Stands in for a freshly bound engine with no context accessor wired."""


def test_a_store_that_never_enabled_teams_is_off(store: sqlite3.Connection) -> None:
    assert read_persisted_teams_enabled(store) is None
    assert access_scope_stamps_exist(store) is False
    assert resolve_startup_teams_state(store) == (False, "never-enabled")


def test_the_decision_survives_a_restart(store: sqlite3.Connection) -> None:
    persist_teams_enabled(store, True)
    assert read_persisted_teams_enabled(store) is True
    assert resolve_startup_teams_state(store) == (True, "enabled")


def test_an_aborted_enable_is_not_read_as_teams_off(store: sqlite3.Connection) -> None:
    """Stamps with no recorded decision must not resolve to permissive."""
    _stamp(store)
    assert read_persisted_teams_enabled(store) is None

    enabled, reason = resolve_startup_teams_state(store)
    assert (enabled, reason) == (True, "stamped-without-marker")


def test_an_aborted_enable_resolves_fail_closed_not_permissive(
    store: sqlite3.Connection,
) -> None:
    """The property the whole phase exists for.

    Reported as enabled, and with no context accessor wired the seam resolves
    enabled-but-unwired to FailClosedPolicy. The store refuses work rather than
    serving one principal's memory to another.
    """
    _stamp(store)
    engine = _Engine()
    enabled, _ = resolve_startup_teams_state(store)
    if enabled:
        scope_storage.mark_teams_enabled(engine)

    assert isinstance(policy_for_engine(engine), FailClosedPolicy)


def test_a_store_with_no_stamps_stays_permissive(store: sqlite3.Connection) -> None:
    """The negative control: default-off must not be dragged into fail-closed."""
    engine = _Engine()
    enabled, _ = resolve_startup_teams_state(store)
    if enabled:
        scope_storage.mark_teams_enabled(engine)

    assert isinstance(policy_for_engine(engine), TrustedOwnerPolicy)


def test_disable_is_distinguishable_from_an_aborted_enable(
    store: sqlite3.Connection,
) -> None:
    """Both leave stamps behind; only one of them means "an operator chose"."""
    _stamp(store)
    persist_teams_enabled(store, False)

    assert resolve_startup_teams_state(store) == (False, "disabled")
    # Same stamps on disk, opposite answer -- which is why the durable false
    # has to be recorded rather than simply clearing the marker.
    assert access_scope_stamps_exist(store) is True


def test_disable_retains_every_stamp(store: sqlite3.Connection) -> None:
    """Additive-only: unstamping would destroy attribution a re-enable needs."""
    _stamp(store)
    persist_teams_enabled(store, False)

    remaining = store.execute(
        "SELECT COUNT(*) FROM messages WHERE access_scope IS NOT NULL"
    ).fetchone()[0]
    assert remaining == 1


def test_enable_disable_re_enable_round_trips(store: sqlite3.Connection) -> None:
    persist_teams_enabled(store, True)
    persist_teams_enabled(store, False)
    persist_teams_enabled(store, True)
    assert resolve_startup_teams_state(store) == (True, "enabled")


def test_the_marker_is_keyed_apart_from_scope_v1(store: sqlite3.Connection) -> None:
    """scope_v1 means "the columns exist", which ordinary bootstrap writes.

    Sharing that key would make every store with the columns look enabled.
    """
    assert TEAMS_ENABLED_METADATA_KEY != "scope_v1"
    persist_teams_enabled(store, True)
    keys = {row[0] for row in store.execute("SELECT key FROM metadata").fetchall()}
    assert keys == {TEAMS_ENABLED_METADATA_KEY}


def test_stamp_detection_tolerates_absent_tables(store: sqlite3.Connection) -> None:
    """Optional scope tables are absent on most stores; that is not an error."""
    store.execute("DROP TABLE messages")
    store.commit()
    assert access_scope_stamps_exist(store) is False


def _doctor_status(result: dict) -> str:
    """Mirror of the mapping in tools.py's lcm_doctor scope_storage check."""
    status = str(result.get("status"))
    if status in {"fail", "stamped-without-marker"}:
        return "fail"
    if status == "nothing-to-verify":
        return "warn"
    return "pass"


def test_doctor_is_not_green_on_an_aborted_enable(store: sqlite3.Connection) -> None:
    """The check you would run to detect the bypass used to report pass.

    verify_scope_storage takes teams_enabled from the CALLER, and after an
    aborted enable that belief is wrong in the dangerous direction, so the
    status fell through to not-enabled with a reassuring "legacy-compatible"
    message on a store full of real per-owner stamps.
    """
    _stamp(store)

    result = scope_storage.verify_scope_storage(store, teams_enabled=False)

    assert result["status"] == "stamped-without-marker"
    assert _doctor_status(result) == "fail"
    assert "aborted partway" in str(result["message"])


def test_doctor_still_passes_a_store_that_never_enabled_teams(
    store: sqlite3.Connection,
) -> None:
    result = scope_storage.verify_scope_storage(store, teams_enabled=False)
    assert result["status"] == "not-enabled"
    assert _doctor_status(result) == "pass"


def test_doctor_passes_a_deliberately_disabled_store_and_says_stamps_are_kept(
    store: sqlite3.Connection,
) -> None:
    """Same stamps as the aborted case, opposite verdict."""
    _stamp(store)
    persist_teams_enabled(store, False)

    result = scope_storage.verify_scope_storage(store, teams_enabled=False)

    assert result["status"] == "not-enabled"
    assert _doctor_status(result) == "pass"
    assert "retained" in str(result["message"])


def test_reading_the_decision_never_creates_schema(tmp_path) -> None:
    """The read path runs on the doctor and on every bind.

    An earlier version called ensure_metadata_table here, so simply asking
    whether Teams was enabled materialised `metadata` on a header-only
    database -- a verification reporting a shape it had just created itself.
    """
    conn = sqlite3.connect(tmp_path / "header-only.db")
    try:
        assert read_persisted_teams_enabled(conn) is None
        assert resolve_startup_teams_state(conn) == (False, "never-enabled")

        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
        assert tables == []
    finally:
        conn.close()
