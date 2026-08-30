"""Enable preflight, operator overrides, and per-table failure isolation.

Enable stamps in committed batches, so an unattributable session discovered
halfway through leaves the store partly stamped -- the state Phase 1 has to
fail closed on. These pin the two things that stop that happening: answering
the attribution question before any write, and keeping one table's failure
from silently cancelling every table after it.
"""

from __future__ import annotations

import sqlite3

import pytest

from hermes_lcm.scope_storage import (
    ACCESS_SCOPE_COLUMN,
    ScopeBackfillIncompleteError,
    backfill_scopes,
    compose_scope_resolver,
    preflight_teams_scope,
)


KNOWN = {"known-a": "principal-a", "known-b": "principal-b"}


def _resolver(session_id: str) -> str | None:
    return KNOWN.get(session_id)


@pytest.fixture()
def store(tmp_path) -> sqlite3.Connection:
    conn = sqlite3.connect(tmp_path / "lcm.db")
    conn.executescript(
        """
        CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE messages(
            store_id INTEGER PRIMARY KEY, session_id TEXT, content TEXT
        );
        CREATE TABLE summary_nodes(node_id INTEGER PRIMARY KEY, session_id TEXT);
        """
    )
    conn.commit()
    try:
        yield conn
    finally:
        conn.close()


def _message(conn: sqlite3.Connection, session_id: str) -> None:
    conn.execute(
        "INSERT INTO messages(session_id, content) VALUES(?, 'x')", (session_id,)
    )
    conn.commit()


def test_preflight_names_every_session_it_cannot_attribute(
    store: sqlite3.Connection,
) -> None:
    for session_id in ("known-a", "orphan-1", "orphan-2"):
        _message(store, session_id)

    report = preflight_teams_scope(store, _resolver)

    assert report["ready"] is False
    assert report["unresolvable"] == ["orphan-1", "orphan-2"]


def test_preflight_writes_nothing_and_creates_nothing(
    store: sqlite3.Connection,
) -> None:
    """A preflight that migrates the store it inspects is not a preflight."""
    _message(store, "orphan-1")

    preflight_teams_scope(store, _resolver)

    columns = [row[1] for row in store.execute("PRAGMA table_info(messages)")]
    assert ACCESS_SCOPE_COLUMN not in columns
    stamped = store.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id IS NOT NULL"
    ).fetchone()[0]
    assert stamped == 1  # rows untouched, nothing added or removed


def test_preflight_is_ready_once_overrides_cover_the_gap(
    store: sqlite3.Connection,
) -> None:
    _message(store, "known-a")
    _message(store, "orphan-1")

    report = preflight_teams_scope(
        store, _resolver, overrides={"orphan-1": "principal-c"}
    )

    assert report["ready"] is True
    assert report["unresolvable"] == []


def test_a_fallback_owner_answers_whatever_the_overrides_did_not(
    store: sqlite3.Connection,
) -> None:
    _message(store, "orphan-1")
    _message(store, "orphan-2")

    report = preflight_teams_scope(
        store,
        _resolver,
        overrides={"orphan-1": "principal-c"},
        fallback_owner="principal-shared",
    )

    assert report["ready"] is True


def test_overrides_win_over_the_host_resolver(store: sqlite3.Connection) -> None:
    """An operator correcting a bad attribution must not be overruled."""
    resolve = compose_scope_resolver(
        _resolver, overrides={"known-a": "principal-corrected"}
    )
    assert resolve("known-a") == "principal-corrected"


def test_the_fallback_is_only_used_when_nothing_else_answers(
    store: sqlite3.Connection,
) -> None:
    resolve = compose_scope_resolver(_resolver, fallback_owner="principal-shared")
    assert resolve("known-a") == "principal-a"
    assert resolve("orphan-1") == "principal-shared"


def test_backfill_applies_overrides_and_fallback(store: sqlite3.Connection) -> None:
    for session_id in ("known-a", "orphan-1", "orphan-2"):
        _message(store, session_id)

    result = backfill_scopes(
        store,
        _resolver,
        overrides={"orphan-1": "principal-c"},
        fallback_owner="principal-shared",
    )

    assert result["complete"] is True
    stamps = dict(
        store.execute("SELECT session_id, access_scope FROM messages").fetchall()
    )
    assert stamps == {
        "known-a": "principal-a",
        "orphan-1": "principal-c",
        "orphan-2": "principal-shared",
    }


def test_one_table_failing_does_not_cancel_the_tables_after_it(
    store: sqlite3.Connection,
) -> None:
    """The defect: no try/except meant later tables were never STARTED.

    Not truncated -- never started. The operator saw a single error with no way
    to tell which tables had been attempted at all.
    """
    _message(store, "orphan")  # messages cannot resolve
    store.execute("INSERT INTO summary_nodes(session_id) VALUES('known-a')")
    store.commit()

    with pytest.raises(ScopeBackfillIncompleteError) as excinfo:
        backfill_scopes(store, _resolver)

    report = excinfo.value.report
    assert "messages" in report["failures"]
    assert "summary_nodes" in report["attempted"]
    assert report["updated"].get("summary_nodes") == 1
    assert report["complete"] is False


def test_a_failed_table_leaves_its_own_rows_unstamped(
    store: sqlite3.Connection,
) -> None:
    """Isolation must not mean the failure is papered over."""
    _message(store, "orphan")
    store.execute("INSERT INTO summary_nodes(session_id) VALUES('known-a')")
    store.commit()

    with pytest.raises(ScopeBackfillIncompleteError):
        backfill_scopes(store, _resolver)

    assert store.execute("SELECT access_scope FROM messages").fetchone()[0] is None
    assert (
        store.execute("SELECT access_scope FROM summary_nodes").fetchone()[0]
        == "principal-a"
    )


def test_rerunning_after_supplying_the_missing_owner_completes(
    store: sqlite3.Connection,
) -> None:
    """The backfill is idempotent, so a fixed run resumes rather than redoes."""
    _message(store, "known-a")
    _message(store, "orphan")

    with pytest.raises(ScopeBackfillIncompleteError) as excinfo:
        backfill_scopes(store, _resolver)
    assert excinfo.value.report["complete"] is False

    second = backfill_scopes(store, _resolver, overrides={"orphan": "principal-c"})
    assert second["complete"] is True

    stamps = dict(
        store.execute("SELECT session_id, access_scope FROM messages").fetchall()
    )
    assert stamps == {"known-a": "principal-a", "orphan": "principal-c"}
