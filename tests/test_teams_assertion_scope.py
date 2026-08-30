"""Assertions inherit the owner of the message they were extracted from.

An adversarial audit reproduced this on real customer data: `lcm_query_state`
returned assertions derived from EVERY principal's messages, each carrying the
source quote verbatim plus the foreign session_id and store_id.

Two things combined to allow it:

- `lcm_query_state` addresses assertions by `subject_key`, which names no row,
  so the tool-boundary gate had no owner to attach and allowed unconditionally.
- The assertion tables carry no `access_scope` column, so `authorize_stored_scope`
  had nothing to compare either.

They do not need a column of their own. An assertion is DERIVED from a source
message and quotes it, so it inherits that message's owner -- and `messages` is
already joined on `source_store_id` in both queries.
"""

from __future__ import annotations

import sqlite3

import pytest

from hermes_lcm.assertion_store import AssertionStore


def _seed(conn: sqlite3.Connection) -> None:
    """Two principals, one subject_key, one assertion each.

    The source hash and the quote must be genuine: `query_assertions` validates
    provenance on every row it returns, so a fabricated digest fails before the
    scoping can be observed at all.
    """
    import hashlib

    for store_id, owner, text in (
        (1, "acorn", "Acorn's private note about the project."),
        (76, "carus", "Carus's private note about the project."),
    ):
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        conn.execute(
            "INSERT INTO messages (store_id, session_id, source, role, content,"
            " timestamp, access_scope) VALUES (?,?,?,?,?,?,?)",
            (store_id, f"openclaw-lcm:agent:{owner}:s1", "", "user", text, 1.0, owner),
        )
        conn.execute(
            "INSERT INTO lcm_assertion_sources (source_store_id, extraction_version,"
            " source_content_sha256, source_session_id, source_role, source_timestamp,"
            " candidate_digest, processed_at) VALUES (?,?,?,?,?,?,?,?)",
            (store_id, "v1", digest, f"openclaw-lcm:agent:{owner}:s1", "user",
             1.0, "b" * 64, 1.0),
        )
        conn.execute(
            "INSERT INTO lcm_assertions (assertion_id, source_store_id,"
            " extraction_version, source_content_sha256, subject_key, predicate_key,"
            " object_json, kind, polarity, observed_at, source_span_start,"
            " source_span_end, source_quote, confidence)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (f"{store_id:064d}", store_id, "v1", digest, "project", "status",
             '"open"', "fact", "positive", 1.0, 0, len(text), text, 1.0),
        )
    conn.commit()


@pytest.fixture()
def store(tmp_path):
    from hermes_lcm.store import MessageStore

    db = tmp_path / "lcm.db"
    messages = MessageStore(db)          # creates the message schema
    assertions = AssertionStore(db)      # creates the assertion schema
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("ALTER TABLE messages ADD COLUMN access_scope TEXT")
    except sqlite3.OperationalError:
        pass  # Teams already enabled on this schema
    _seed(conn)
    conn.close()
    try:
        yield assertions
    finally:
        assertions.close()
        messages.close()


def test_a_scoped_query_returns_only_the_principals_own_assertions(store) -> None:
    rows = store.query_assertions(
        extraction_version="v1", subject_key="project", access_scope="acorn"
    )
    assert rows, "POSITIVE CONTROL: acorn must still see its own assertion"
    owners = {int(row["source_store_id"]) for row in rows}
    assert owners == {1}, f"carus's assertion leaked into acorn's result: {owners}"

    quotes = " ".join(str(row["source_quote"]) for row in rows)
    assert "Carus's private note" not in quotes


def test_the_symmetric_guarantee(store) -> None:
    rows = store.query_assertions(
        extraction_version="v1", subject_key="project", access_scope="carus"
    )
    assert rows, "POSITIVE CONTROL: carus must still see its own assertion"
    assert {int(row["source_store_id"]) for row in rows} == {76}


def test_an_unscoped_query_is_unchanged(store) -> None:
    """Default-off must stay byte-identical: no predicate, every row."""
    rows = store.query_assertions(extraction_version="v1", subject_key="project")
    assert {int(row["source_store_id"]) for row in rows} == {1, 76}


def test_both_query_paths_accept_the_predicate() -> None:
    """Relations disclose BOTH endpoints' quotes, so they need it too."""
    import inspect

    for method in (AssertionStore.query_assertions, AssertionStore.query_relations):
        assert "access_scope" in inspect.signature(method).parameters, (
            f"{method.__name__} cannot be scoped"
        )
        assert (
            inspect.signature(method).parameters["access_scope"].default is None
        ), "the predicate must default to None so default-off is unchanged"


def test_relations_require_every_endpoint_to_belong_to_the_principal() -> None:
    """Filtering only the relation's own source still leaks a foreign quote."""
    import inspect

    source = inspect.getsource(AssertionStore.query_relations)
    for alias in ("relation_message", "from_message", "to_message"):
        assert f"{alias}.access_scope = ?" in source, (
            f"{alias} is unfiltered; a relation across the principal boundary "
            f"would disclose its quote"
        )
