"""AMR Level-1 slice tests: quote anchors, epistemic marking, migration, verifier."""

from __future__ import annotations

import sqlite3

import pytest

from hermes_lcm.assertion_amr import (
    OUTCOME_ANCHOR_TAMPERED,
    OUTCOME_OK,
    OUTCOME_SOURCE_DRIFTED,
    OUTCOME_SOURCE_MISSING,
    verify_assertion_citations,
    verify_relation_citations,
)
from hermes_lcm.assertion_store import (
    AssertionCandidate,
    AssertionRelationCandidate,
    AssertionStore,
)
from hermes_lcm.db_bootstrap import (
    ASSERTION_EPISTEMIC_VALUES,
    _normalize_quote_for_hash,
    _quote_hash,
)


def _candidate(
    content: str,
    quote: str,
    *,
    subject: str = "user",
    predicate: str = "likes",
    value: object = "tea",
    kind: str = "fact",
    epistemic: str | None = None,
) -> AssertionCandidate:
    start = content.index(quote)
    return AssertionCandidate(
        source_span_start=start,
        source_span_end=start + len(quote),
        subject_key=subject,
        predicate_key=predicate,
        object_value=value,
        value_text=str(value),
        kind=kind,
        epistemic=epistemic,
    )


@pytest.fixture
def amr_db(tmp_path):
    db_path = tmp_path / "lcm.db"
    from hermes_lcm.store import MessageStore

    messages = MessageStore(db_path)
    assertions = AssertionStore(db_path)
    try:
        yield db_path, messages, assertions
    finally:
        assertions.close()
        messages.close()


def test_amr_declaration_vocabulary_is_closed():
    assert ASSERTION_EPISTEMIC_VALUES == frozenset({
        "fact", "inference", "open_question", "unverified",
    })


def test_quote_hash_is_algorithm_prefixed_and_normalized():
    hashed = _quote_hash("curly \u201cquote\u201d — dash \u2026 done")
    assert hashed.startswith("sha256:")
    assert hashed == _quote_hash("curly \"quote\" - dash ... done")
    # Normalization is idempotent.
    once = _normalize_quote_for_hash("a\u00a0b")
    assert _normalize_quote_for_hash(once) == once
    # Whitespace runs collapse.
    assert _normalize_quote_for_hash("a \n\t b") == "a b"


def test_publish_stores_prefix_hash_and_epistemic(amr_db):
    _db_path, messages, assertions = amr_db
    content = "The reviewer confirmed the release at noon."
    store_id = messages.append(
        "session-a", {"role": "user", "content": content}, source="cli"
    )
    snapshot = assertions.snapshot_source(store_id)
    result = assertions.publish_source(
        snapshot,
        [_candidate(content, "confirmed the release", epistemic="inference")],
    )
    assert result.already_current is False
    rows = assertions.query_assertions(source_store_id=store_id)
    assert len(rows) == 1
    row = rows[0]
    assert row["epistemic"] == "inference"
    expected = _quote_hash("confirmed the release")
    assert row["source_quote_hash"] == expected


def test_unmarked_is_null_not_fact(amr_db):
    _db_path, messages, assertions = amr_db
    content = "Plain observation without marking."
    store_id = messages.append(
        "session-a", {"role": "user", "content": content}, source="cli"
    )
    snapshot = assertions.snapshot_source(store_id)
    assertions.publish_source(snapshot, [_candidate(content, "Plain observation")])
    row = assertions.query_assertions(source_store_id=store_id)[0]
    assert row["epistemic"] is None


def test_epistemic_outside_vocabulary_is_rejected(amr_db):
    _db_path, messages, assertions = amr_db
    content = "A record with a made-up epistemic value."
    store_id = messages.append(
        "session-a", {"role": "user", "content": content}, source="cli"
    )
    snapshot = assertions.snapshot_source(store_id)
    with pytest.raises(ValueError, match="unsupported epistemic"):
        assertions.publish_source(
            snapshot,
            [_candidate(content, "made-up epistemic", epistemic="probably_true")],
        )
    assert assertions.query_assertions(source_store_id=store_id) == []


def test_epistemic_is_case_sensitive(amr_db):
    _db_path, messages, assertions = amr_db
    content = "Uppercase FACT must not be coerced to fact."
    store_id = messages.append(
        "session-a", {"role": "user", "content": content}, source="cli"
    )
    snapshot = assertions.snapshot_source(store_id)
    with pytest.raises(ValueError, match="unsupported epistemic"):
        assertions.publish_source(
            snapshot,
            [_candidate(content, "Uppercase FACT", epistemic="FACT")],
        )
    assert assertions.query_assertions(source_store_id=store_id) == []


def test_verifier_ok_against_unchanged_source(amr_db):
    _db_path, messages, assertions = amr_db
    content = "The deploy finished at 12:00."
    store_id = messages.append(
        "session-a", {"role": "user", "content": content}, source="cli"
    )
    snapshot = assertions.snapshot_source(store_id)
    assertions.publish_source(snapshot, [_candidate(content, "finished at 12:00")])
    report = verify_assertion_citations(assertions, source_store_id=store_id)
    assert len(report) == 1
    assert report[0]["outcome"] == OUTCOME_OK
    assert report[0]["partial"] is False


def test_verifier_source_drifted_after_content_change(amr_db):
    _db_path, messages, assertions = amr_db
    content = "The deploy finished at 12:00."
    store_id = messages.append(
        "session-a", {"role": "user", "content": content}, source="cli"
    )
    snapshot = assertions.snapshot_source(store_id)
    assertions.publish_source(snapshot, [_candidate(content, "finished at 12:00")])
    with sqlite3.connect(_db_path) as raw:
        raw.execute("UPDATE messages SET content = 'Moved to 13:00.' WHERE store_id = ?", (store_id,))
    report = verify_assertion_citations(
        assertions, source_store_id=store_id, include_invalidated=True
    )
    assert report[0]["outcome"] == OUTCOME_SOURCE_DRIFTED


def test_verifier_anchor_tampered_when_hash_disagrees(amr_db):
    _db_path, messages, assertions = amr_db
    content = "Honest record, untouched source."
    store_id = messages.append(
        "session-a", {"role": "user", "content": content}, source="cli"
    )
    snapshot = assertions.snapshot_source(store_id)
    assertions.publish_source(snapshot, [_candidate(content, "Honest record")])
    with sqlite3.connect(_db_path) as raw:
        raw.execute(
            "UPDATE lcm_assertions SET source_quote_hash = ? WHERE source_store_id = ?",
            ("sha256:" + "0" * 64, store_id),
        )
    report = verify_assertion_citations(assertions, source_store_id=store_id)
    assert report[0]["outcome"] == OUTCOME_ANCHOR_TAMPERED


def test_schema_rejects_bare_digest_and_wrong_length(amr_db):
    """The column CHECK enforces the AMR hash format even against raw SQL."""
    _db_path, messages, assertions = amr_db
    content = "Bare digest must not be guessed."
    store_id = messages.append(
        "session-a", {"role": "user", "content": content}, source="cli"
    )
    snapshot = assertions.snapshot_source(store_id)
    assertions.publish_source(snapshot, [_candidate(content, "Bare digest")])
    with sqlite3.connect(_db_path) as raw:
        with pytest.raises(sqlite3.IntegrityError):
            raw.execute(
                "UPDATE lcm_assertions SET source_quote_hash = ? WHERE source_store_id = ?",
                ("a" * 64, store_id),
            )
        with pytest.raises(sqlite3.IntegrityError):
            raw.execute(
                "UPDATE lcm_assertions SET source_quote_hash = ? WHERE source_store_id = ?",
                ("md5:" + "a" * 32, store_id),
            )
        # Case enforcement: same length, same prefix, uppercase hex must be
        # rejected (GLOB '[...]*' alone does NOT do this — SQLite GLOB
        # character classes match a single char followed by any chars).
        with pytest.raises(sqlite3.IntegrityError):
            raw.execute(
                "UPDATE lcm_assertions SET source_quote_hash = ? WHERE source_store_id = ?",
                ("sha256:" + "a" * 63 + "A", store_id),
            )
        with pytest.raises(sqlite3.IntegrityError):
            raw.execute(
                "UPDATE lcm_assertions SET source_quote_hash = ? WHERE source_store_id = ?",
                ("sha256:" + "a" * 30 + "F" + "a" * 33, store_id),
            )


def test_verifier_source_missing_when_message_gone(amr_db):
    _db_path, messages, assertions = amr_db
    content = "Deleted underneath the citation."
    store_id = messages.append(
        "session-a", {"role": "user", "content": content}, source="cli"
    )
    snapshot = assertions.snapshot_source(store_id)
    assertions.publish_source(snapshot, [_candidate(content, "Deleted underneath")])
    # Delete only the message row: the assertion survives with its source
    # generation, but the cited content can no longer be resolved.
    with sqlite3.connect(_db_path) as raw:
        raw.execute("DELETE FROM messages WHERE store_id = ?", (store_id,))
    report = verify_assertion_citations(
        assertions, source_store_id=store_id, include_invalidated=True
    )
    assert len(report) == 1
    assert report[0]["outcome"] == OUTCOME_SOURCE_MISSING


def test_migration_preserves_rows_and_invalidations(tmp_path):
    from hermes_lcm.store import MessageStore

    db_path = tmp_path / "legacy.db"
    messages = MessageStore(db_path)
    assertions = AssertionStore(db_path)
    content = "Legacy claim survives the migration."
    store_id = messages.append(
        "session-a", {"role": "user", "content": content}, source="cli"
    )
    snapshot = assertions.snapshot_source(store_id)
    assertions.publish_source(snapshot, [_candidate(content, "Legacy claim")])
    # Simulate a source update invalidating the published generation.
    content_v2 = "Legacy claim, now updated."
    with sqlite3.connect(db_path) as raw:
        raw.execute("UPDATE messages SET content = ? WHERE store_id = ?", (content_v2, store_id))
        raw.execute(
            "UPDATE lcm_assertion_sources"
            " SET invalidated_at = 1000.0, invalidation_reason = 'source_updated'"
            " WHERE source_store_id = ?",
            (store_id,),
        )
    # Second, still-valid generation under a different extraction version.
    snapshot2 = assertions.snapshot_source(store_id)
    assertions.publish_source(
        snapshot2, [_candidate(content_v2, "now updated")], extraction_version="assertions-v2"
    )
    legacy_assertion = assertions.connection.execute(
        "SELECT assertion_id, source_quote FROM lcm_assertions"
        " WHERE extraction_version = 'assertions-v1'"
    ).fetchone()
    legacy_relations_count = 0
    assertions.close()
    del legacy_relations_count

    # Reopen with the new build: the migration must backfill hashes without
    # losing the invalidated row.
    reopened = AssertionStore(db_path)
    try:
        assert reopened.connection.execute(
            "SELECT COUNT(*) FROM lcm_assertions"
        ).fetchone()[0] == 2
        row = reopened.connection.execute(
            "SELECT source_quote_hash, epistemic FROM lcm_assertions"
            " WHERE assertion_id = ?",
            (legacy_assertion[0],),
        ).fetchone()
        assert row["source_quote_hash"] == _quote_hash(legacy_assertion[1])
        assert row["epistemic"] is None
        findings = reopened.connection.execute(
            "SELECT COUNT(*) FROM lcm_assertions WHERE source_quote_hash IS NULL"
        ).fetchone()[0]
        assert findings == 0
        # Sources untouched, including the invalidation history.
        invalidated = reopened.connection.execute(
            "SELECT invalidation_reason FROM lcm_assertion_sources"
            " WHERE invalidated_at IS NOT NULL"
        ).fetchone()
        assert invalidated is not None
    finally:
        reopened.close()
        messages.close()


def test_relations_carry_quote_hash_and_verify(amr_db):
    _db_path, messages, assertions = amr_db
    content = "I liked tea before, but I prefer coffee now."
    store_id = messages.append("session-a", {"role": "user", "content": content})
    snapshot = assertions.snapshot_source(store_id)
    tea = _candidate(content, "tea", predicate="preferred_drink", value="tea")
    coffee = _candidate(content, "coffee", predicate="preferred_drink", value="coffee")
    tea_id = assertions.assertion_id_for(snapshot, tea)
    coffee_id = assertions.assertion_id_for(snapshot, coffee)
    relation = AssertionRelationCandidate(
        source_span_start=content.index("coffee"),
        source_span_end=content.index("coffee") + len("coffee"),
        from_assertion_id=coffee_id,
        relation_type="supersedes",
        to_assertion_id=tea_id,
    )
    assertions.publish_source(snapshot, [tea, coffee], relations=[relation])
    relations = assertions.query_relations()
    assert len(relations) == 1
    assert relations[0]["source_quote_hash"] == _quote_hash("coffee")
    report = verify_relation_citations(assertions, source_store_id=store_id)
    assert len(report) == 1
    assert report[0]["outcome"] == OUTCOME_OK


def test_idempotent_republication_is_still_zero_write(amr_db):
    _db_path, messages, assertions = amr_db
    content = "Repeat publishes must remain no-ops."
    store_id = messages.append(
        "session-a", {"role": "user", "content": content}, source="cli"
    )
    snapshot = assertions.snapshot_source(store_id)
    first = assertions.publish_source(snapshot, [_candidate(content, "Repeat publishes")])
    second = assertions.publish_source(snapshot, [_candidate(content, "Repeat publishes")])
    assert first.already_current is False
    assert second.already_current is True
    rows = assertions.query_assertions(source_store_id=store_id)
    assert len(rows) == 1
    assert rows[0]["source_quote_hash"] == _quote_hash("Repeat publishes")