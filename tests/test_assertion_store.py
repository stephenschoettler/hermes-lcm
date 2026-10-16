"""Focused tests for the opt-in, same-database V4 assertion foundation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import sqlite3
import threading
from types import SimpleNamespace

import pytest

from hermes_lcm.assertion_store import (
    AssertionCandidate,
    AssertionPublicationConflictError,
    AssertionRelationCandidate,
    AssertionSchemaUnavailableError,
    AssertionStoreError,
    AssertionSourceStaleError,
    AssertionStore,
)
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG
from hermes_lcm.db_bootstrap import (
    ASSERTION_MIGRATION_STEP,
    SCHEMA_VERSION,
    VERSION_MISMATCH_GENUINELY_NEWER,
    VERSION_MISMATCH_INTERIM_STAMP,
    classify_version_mismatch,
    remediate_interim_schema_stamp,
    verify_assertion_schema,
)
from hermes_lcm.maintenance import (
    backup_database,
    flush_engine_connections,
    rotate_backup_database,
)
from hermes_lcm.lifecycle_state import LifecycleStateStore
from hermes_lcm.store import MessageStore


def _candidate(
    content: str,
    quote: str,
    *,
    subject: str = "user",
    predicate: str = "likes",
    value: object = "tea",
    kind: str = "fact",
    event_at: float | None = None,
    valid_from: float | None = None,
    valid_to: float | None = None,
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
        event_at=event_at,
        valid_from=valid_from,
        valid_to=valid_to,
    )


@pytest.fixture
def assertion_db(tmp_path):
    db_path = tmp_path / "lcm.db"
    messages = MessageStore(db_path)
    assertions = AssertionStore(db_path)
    try:
        yield db_path, messages, assertions
    finally:
        assertions.close()
        messages.close()


def _assertion_objects(conn: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE name LIKE 'lcm_assertion%' ORDER BY name"
        )
    }


def test_feature_defaults_off_and_does_not_materialize_schema(tmp_path, monkeypatch):
    monkeypatch.delenv("LCM_ASSERTIONS_ENABLED", raising=False)
    db_path = tmp_path / "disabled.db"
    messages = MessageStore(db_path)
    try:
        assert LCMConfig().assertions_enabled is False
        assert LCMConfig.from_env().assertions_enabled is False
        assert _assertion_objects(messages._conn) == set()
        assert int(
            messages._conn.execute(
                "SELECT value FROM metadata WHERE key = 'schema_version'"
            ).fetchone()[0]
        ) == SCHEMA_VERSION
    finally:
        messages.close()

    monkeypatch.setenv("LCM_ASSERTIONS_ENABLED", "true")
    assert LCMConfig.from_env().assertions_enabled is True


def test_read_only_open_without_schema_fails_without_mutation(tmp_path):
    db_path = tmp_path / "legacy.db"
    messages = MessageStore(db_path)
    before = tuple(
        tuple(row)
        for row in messages._conn.execute(
            "SELECT type, name, sql FROM sqlite_master ORDER BY type, name"
        ).fetchall()
    )
    messages.close()

    with pytest.raises(AssertionSchemaUnavailableError):
        AssertionStore(db_path, read_only=True)

    with sqlite3.connect(db_path) as conn:
        after = tuple(
            conn.execute(
                "SELECT type, name, sql FROM sqlite_master ORDER BY type, name"
            ).fetchall()
        )
    assert after == before


def test_lazy_schema_marker_and_legacy_migration_stay_on_core_v5(tmp_path):
    db_path = tmp_path / "legacy.db"
    messages = MessageStore(db_path)
    assertions = AssertionStore(db_path)
    try:
        assert verify_assertion_schema(assertions.connection) == []
        assert int(
            assertions.connection.execute(
                "SELECT value FROM metadata WHERE key = 'schema_version'"
            ).fetchone()[0]
        ) == SCHEMA_VERSION
        assert assertions.connection.execute(
            "SELECT COUNT(*) FROM lcm_migration_state WHERE step_name = ?",
            (ASSERTION_MIGRATION_STEP,),
        ).fetchone()[0] == 1

        second = AssertionStore(db_path)
        second.close()
        assert assertions.connection.execute(
            "SELECT COUNT(*) FROM lcm_migration_state WHERE step_name = ?",
            (ASSERTION_MIGRATION_STEP,),
        ).fetchone()[0] == 1
    finally:
        assertions.close()
        messages.close()


def test_schema_verifier_rejects_constraint_and_trigger_drift(tmp_path):
    db_path = tmp_path / "malformed.db"
    messages = MessageStore(db_path)
    assertions = AssertionStore(db_path)
    try:
        table_info = assertions.connection.execute(
            "PRAGMA table_info(lcm_assertions)"
        ).fetchall()
        assertions.connection.execute("DROP TABLE lcm_assertions")
        columns = ", ".join(
            f'"{row[1]}" {row[2] or "TEXT"}' for row in table_info
        )
        assertions.connection.execute(f"CREATE TABLE lcm_assertions({columns})")
        assertions.connection.execute("DROP TRIGGER lcm_assertion_message_update")
        assertions.connection.execute(
            """
            CREATE TRIGGER lcm_assertion_message_update
            AFTER UPDATE OF content ON messages BEGIN SELECT 1; END
            """
        )
        assertions.connection.execute("CREATE TABLE lcm_assertion_future_state(value TEXT)")

        findings = verify_assertion_schema(assertions.connection)

        assert "malformed table:lcm_assertions" in findings
        assert "malformed trigger:lcm_assertion_message_update" in findings
        assert "unexpected-table:lcm_assertion_future_state" in findings
    finally:
        assertions.close()
        messages.close()


def test_schema_stamp_remediation_keeps_valid_family_and_drops_malformed_family(
    tmp_path,
):
    valid_path = tmp_path / "valid-interim.db"
    valid_messages = MessageStore(valid_path)
    valid_dag = SummaryDAG(valid_path)
    valid_lifecycle = LifecycleStateStore(valid_path)
    valid_dag.close()
    valid_lifecycle.close()
    valid_assertions = AssertionStore(valid_path)
    valid_assertions.connection.execute(
        "UPDATE metadata SET value = ? WHERE key = 'schema_version'",
        (str(SCHEMA_VERSION + 1),),
    )
    valid_assertions.connection.commit()
    assert classify_version_mismatch(valid_assertions.connection) == VERSION_MISMATCH_INTERIM_STAMP
    result = remediate_interim_schema_stamp(valid_assertions.connection, apply=True)
    assert result["applied"] is True
    assert "lcm_assertions" in _assertion_objects(valid_assertions.connection)
    valid_assertions.close()
    valid_messages.close()

    malformed_path = tmp_path / "malformed-interim.db"
    malformed_messages = MessageStore(malformed_path)
    malformed_dag = SummaryDAG(malformed_path)
    malformed_lifecycle = LifecycleStateStore(malformed_path)
    malformed_dag.close()
    malformed_lifecycle.close()
    malformed_assertions = AssertionStore(malformed_path)
    malformed_assertions.connection.execute("DROP TRIGGER lcm_assertion_message_update")
    malformed_assertions.connection.execute(
        "UPDATE metadata SET value = ? WHERE key = 'schema_version'",
        (str(SCHEMA_VERSION + 1),),
    )
    malformed_assertions.connection.commit()
    assert classify_version_mismatch(malformed_assertions.connection) == VERSION_MISMATCH_INTERIM_STAMP
    result = remediate_interim_schema_stamp(malformed_assertions.connection, apply=True)
    assert result["applied"] is True
    assert _assertion_objects(malformed_assertions.connection) == set()
    assert malformed_assertions.connection.execute(
        "SELECT COUNT(*) FROM messages"
    ).fetchone()[0] == 0
    malformed_assertions.close()
    malformed_messages.close()


def test_schema_stamp_remediation_refuses_same_name_future_assertion_shape(tmp_path):
    db_path = tmp_path / "future-assertion-shape.db"
    messages = MessageStore(db_path)
    dag = SummaryDAG(db_path)
    lifecycle = LifecycleStateStore(db_path)
    dag.close()
    lifecycle.close()
    assertions = AssertionStore(db_path)
    try:
        content = "I prefer tea."
        store_id = messages.append(
            "session-a", {"role": "user", "content": content}
        )
        assertions.publish_source(
            assertions.snapshot_source(store_id), [_candidate(content, "tea")]
        )
        assertions.connection.execute("DROP TRIGGER lcm_assertion_message_update")
        assertions.connection.execute(
            """
            CREATE TRIGGER lcm_assertion_message_update
            AFTER UPDATE OF content ON messages
            BEGIN
                SELECT 1;
            END
            """
        )
        assertions.connection.execute(
            "UPDATE metadata SET value = ? WHERE key = 'schema_version'",
            (str(SCHEMA_VERSION + 1),),
        )
        assertions.connection.commit()

        findings = verify_assertion_schema(assertions.connection)
        assert "malformed trigger:lcm_assertion_message_update" in findings
        assert (
            classify_version_mismatch(assertions.connection)
            == VERSION_MISMATCH_GENUINELY_NEWER
        )
        objects_before = _assertion_objects(assertions.connection)
        result = remediate_interim_schema_stamp(
            assertions.connection, apply=True
        )

        assert result["status"] == "refused"
        assert result["applied"] is False
        assert _assertion_objects(assertions.connection) == objects_before
        assert assertions.connection.execute(
            "SELECT COUNT(*) FROM lcm_assertions"
        ).fetchone()[0] == 1
    finally:
        assertions.close()
        messages.close()


def test_concurrent_lazy_initialization_is_idempotent(tmp_path):
    db_path = tmp_path / "concurrent.db"
    messages = MessageStore(db_path)
    messages.close()

    def initialize() -> list[str]:
        store = AssertionStore(db_path)
        try:
            return verify_assertion_schema(store.connection)
        finally:
            store.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _index: initialize(), range(2)))

    assert results == [[], []]
    with sqlite3.connect(db_path) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM lcm_migration_state WHERE step_name = ?",
            (ASSERTION_MIGRATION_STEP,),
        ).fetchone()[0] == 1


def test_unicode_exact_span_publish_preserves_messages_and_fts(assertion_db):
    _db_path, messages, assertions = assertion_db
    content = "I prefer café ☕ after yoga."
    store_id = messages.append(
        "session-a", {"role": "user", "content": content}, source="cli"
    )
    snapshot = assertions.snapshot_source(store_id)
    candidate = _candidate(
        content,
        "café ☕",
        predicate="prefers_drink",
        value="café ☕",
        kind="preference",
    )
    before_messages = tuple(messages._conn.execute("SELECT * FROM messages").fetchall())
    before_fts = tuple(
        messages._conn.execute(
            "SELECT rowid, content FROM messages_fts ORDER BY rowid"
        ).fetchall()
    )

    result = assertions.publish_source(snapshot, [candidate])
    rows = assertions.query_assertions(source_store_id=store_id)

    assert result.already_current is False
    assert len(rows) == 1
    assert rows[0]["source_quote"] == "café ☕"
    assert content[rows[0]["source_span_start"] : rows[0]["source_span_end"]] == "café ☕"
    assert rows[0]["object_value"] == "café ☕"
    assert tuple(messages._conn.execute("SELECT * FROM messages").fetchall()) == before_messages
    assert tuple(
        messages._conn.execute(
            "SELECT rowid, content FROM messages_fts ORDER BY rowid"
        ).fetchall()
    ) == before_fts
    assert assertions.plan_rebuild(limit=10).pending_count == 0


def test_zero_assertion_receipt_prevents_repeated_work(assertion_db):
    _db_path, messages, assertions = assertion_db
    store_id = messages.append("session-a", {"role": "user", "content": "No durable claim."})
    snapshot = assertions.snapshot_source(store_id)

    first = assertions.publish_source(snapshot, [])
    changes = assertions.connection.total_changes
    second = assertions.publish_source(snapshot, [])

    assert first.already_current is False
    assert first.assertions_written == 0
    assert second.already_current is True
    assert assertions.connection.total_changes == changes
    assert assertions.plan_rebuild(limit=10).selected_sources == ()


def test_invalid_candidate_batch_is_atomic(assertion_db):
    _db_path, messages, assertions = assertion_db
    content = "Tea is preferred."
    store_id = messages.append("session-a", {"role": "user", "content": content})
    snapshot = assertions.snapshot_source(store_id)
    valid = _candidate(content, "Tea", value="tea")
    invalid = AssertionCandidate(
        source_span_start=0,
        source_span_end=len(content) + 1,
        subject_key="user",
        predicate_key="likes",
    )

    with pytest.raises(ValueError, match="invalid source span"):
        assertions.publish_source(snapshot, [valid, invalid])

    assert assertions.connection.execute(
        "SELECT COUNT(*) FROM lcm_assertion_sources"
    ).fetchone()[0] == 0
    assert assertions.connection.execute(
        "SELECT COUNT(*) FROM lcm_assertions"
    ).fetchone()[0] == 0

    with pytest.raises(ValueError, match="duplicate assertion candidates"):
        assertions.publish_source(snapshot, [valid, valid])


def test_idempotency_conflict_versioning_and_content_reversion(assertion_db):
    _db_path, messages, assertions = assertion_db
    original = "The preferred drink is tea."
    store_id = messages.append("session-a", {"role": "user", "content": original})
    original_snapshot = assertions.snapshot_source(store_id)
    tea = _candidate(original, "tea", predicate="preferred_drink", value="tea")

    first = assertions.publish_source(original_snapshot, [tea])
    before_repeat = assertions.connection.total_changes
    repeated = assertions.publish_source(original_snapshot, [tea])
    assert repeated.already_current is True
    assert assertions.connection.total_changes == before_repeat

    changed_output = _candidate(
        original, "tea", predicate="preferred_drink", value="green tea"
    )
    with pytest.raises(AssertionPublicationConflictError):
        assertions.publish_source(original_snapshot, [changed_output])

    newer = assertions.publish_source(
        original_snapshot, [changed_output], extraction_version="assertions-v2"
    )
    assert newer.already_current is False
    assert len(assertions.query_assertions(extraction_version="assertions-v2")) == 1

    messages._conn.execute(
        "UPDATE messages SET content = ? WHERE store_id = ?",
        ("The preferred drink is coffee.", store_id),
    )
    messages._conn.commit()
    coffee_snapshot = assertions.snapshot_source(store_id)
    coffee = _candidate(coffee_snapshot.content, "coffee", value="coffee")
    assertions.publish_source(coffee_snapshot, [coffee])

    messages._conn.execute(
        "UPDATE messages SET content = ? WHERE store_id = ?", (original, store_id)
    )
    messages._conn.commit()
    reverted = assertions.publish_source(assertions.snapshot_source(store_id), [tea])

    assert first.assertion_ids == reverted.assertion_ids
    assert reverted.already_current is True
    assert len(assertions.query_assertions()) == 1


def test_relations_are_explicit_and_hidden_when_an_endpoint_is_invalidated(assertion_db):
    _db_path, messages, assertions = assertion_db
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
    assert len(assertions.query_assertions()) == 2
    assert [row["relation_type"] for row in assertions.query_relations()] == [
        "supersedes"
    ]

    messages._conn.execute(
        "UPDATE messages SET content = ? WHERE store_id = ?",
        ("I prefer water now.", store_id),
    )
    messages._conn.commit()

    assert assertions.query_assertions() == []
    assert assertions.query_relations() == []
    assert len(assertions.query_relations(include_invalidated=True)) == 1


def test_relation_query_fails_closed_on_untriggered_source_tamper(assertion_db):
    _db_path, messages, assertions = assertion_db
    content = "Tea was preferred before coffee."
    store_id = messages.append("session-a", {"role": "user", "content": content})
    snapshot = assertions.snapshot_source(store_id)
    tea = _candidate(content, "Tea", predicate="preferred_drink", value="tea")
    coffee = _candidate(content, "coffee", predicate="preferred_drink", value="coffee")
    relation = AssertionRelationCandidate(
        source_span_start=content.index("coffee"),
        source_span_end=content.index("coffee") + len("coffee"),
        from_assertion_id=assertions.assertion_id_for(snapshot, coffee),
        relation_type="supersedes",
        to_assertion_id=assertions.assertion_id_for(snapshot, tea),
    )
    assertions.publish_source(snapshot, [tea, coffee], relations=[relation])
    messages._conn.execute("DROP TRIGGER lcm_assertion_message_update")
    messages._conn.execute(
        "UPDATE messages SET content = 'Water is preferred.' WHERE store_id = ?",
        (store_id,),
    )
    messages._conn.commit()

    with pytest.raises(AssertionSourceStaleError, match="relation source.*hash changed"):
        assertions.query_relations()


def test_observed_time_not_event_time_controls_historical_visibility(assertion_db):
    _db_path, messages, assertions = assertion_db
    content = "The launch happened earlier."
    store_id = messages.append("session-a", {"role": "user", "content": content})
    messages._conn.execute(
        "UPDATE messages SET timestamp = 200.0 WHERE store_id = ?", (store_id,)
    )
    messages._conn.commit()
    snapshot = assertions.snapshot_source(store_id)
    candidate = _candidate(
        content,
        "launch",
        predicate="launched",
        value=True,
        kind="event",
        event_at=50.0,
        valid_from=100.0,
        valid_to=300.0,
    )
    assertions.publish_source(snapshot, [candidate])

    assert assertions.query_assertions(as_of=150.0) == []
    assert len(assertions.query_assertions(as_of=250.0)) == 1
    assert assertions.query_assertions(as_of=350.0) == []


def test_source_invalidation_is_transactional_and_metadata_updates_do_not_invalidate(
    assertion_db,
):
    _db_path, messages, assertions = assertion_db
    content = "I prefer tea."
    store_id = messages.append("session-a", {"role": "user", "content": content})
    assertions.publish_source(
        assertions.snapshot_source(store_id), [_candidate(content, "tea")]
    )

    messages.pin(store_id)
    messages.reassign_session_messages("session-a", "session-b")
    messages._conn.execute(
        "UPDATE messages SET source = 'cli', timestamp = timestamp + 1 WHERE store_id = ?",
        (store_id,),
    )
    messages._conn.commit()
    assert len(assertions.query_assertions()) == 1

    messages._conn.execute("BEGIN IMMEDIATE")
    messages._conn.execute(
        "UPDATE messages SET content = 'changed' WHERE store_id = ?", (store_id,)
    )
    assert messages._conn.execute(
        "SELECT invalidation_reason FROM lcm_assertion_sources "
        "WHERE source_store_id = ? AND invalidated_at IS NOT NULL",
        (store_id,),
    ).fetchone()[0] == "source_updated"
    messages._conn.rollback()
    assert len(assertions.query_assertions()) == 1

    assert messages.delete_session_messages("session-b") == 1
    assert assertions.query_assertions() == []
    historical = assertions.query_assertions(include_invalidated=True)
    assert historical[0]["invalidation_reason"] == "source_deleted"


def test_late_publish_cas_rejects_changed_source(assertion_db):
    _db_path, messages, assertions = assertion_db
    content = "I prefer tea."
    store_id = messages.append("session-a", {"role": "user", "content": content})
    stale_snapshot = assertions.snapshot_source(store_id)
    candidate = _candidate(content, "tea")
    messages._conn.execute(
        "UPDATE messages SET content = 'I prefer coffee.' WHERE store_id = ?", (store_id,)
    )
    messages._conn.commit()

    with pytest.raises(AssertionSourceStaleError):
        assertions.publish_source(stale_snapshot, [candidate])


def test_query_fails_closed_if_trigger_is_removed_and_source_is_tampered(assertion_db):
    _db_path, messages, assertions = assertion_db
    content = "I prefer tea."
    store_id = messages.append("session-a", {"role": "user", "content": content})
    assertions.publish_source(
        assertions.snapshot_source(store_id), [_candidate(content, "tea")]
    )
    messages._conn.execute("DROP TRIGGER lcm_assertion_message_update")
    messages._conn.execute(
        "UPDATE messages SET content = 'I prefer coffee.' WHERE store_id = ?", (store_id,)
    )
    messages._conn.commit()

    with pytest.raises(AssertionSourceStaleError, match="hash changed"):
        assertions.query_assertions()


def test_read_only_planning_has_zero_writes(assertion_db):
    db_path, messages, assertions = assertion_db
    messages.append("session-a", {"role": "user", "content": "Plan this source."})
    assertions.commit()
    reader = AssertionStore(db_path, read_only=True)
    try:
        before = reader.connection.total_changes
        plan = reader.plan_rebuild(limit=10)
        assert len(plan.selected_sources) == 1
        assert reader.connection.total_changes == before == 0
        with pytest.raises(AssertionStoreError, match="read-only"):
            reader.publish_source(plan.selected_sources[0], [])
    finally:
        reader.close()


def test_backup_restore_and_rotation_include_same_db_assertions(assertion_db, tmp_path):
    db_path, messages, assertions = assertion_db
    content = "I prefer tea."
    store_id = messages.append("session-a", {"role": "user", "content": content})
    assertions.publish_source(
        assertions.snapshot_source(store_id), [_candidate(content, "tea")]
    )
    backup_dir = tmp_path / "backups"
    engine = SimpleNamespace(
        _store=messages,
        _dag=SimpleNamespace(_conn=messages._conn),
        _lifecycle=None,
        _assertions=assertions,
        backup_dir=lambda: backup_dir,
        rotate_backup_path=lambda: backup_dir / "rotate-latest.sqlite3",
    )

    timestamped = backup_database(engine)
    rotated = rotate_backup_database(engine)

    assert timestamped["ok"] is True
    assert rotated["ok"] is True
    for backup_path in (timestamped["backup_path"], rotated["backup_path"]):
        restored = AssertionStore(backup_path, read_only=True)
        try:
            rows = restored.query_assertions()
            assert len(rows) == 1
            assert rows[0]["source_store_id"] == store_id
            assert rows[0]["source_quote"] == "tea"
        finally:
            restored.close()
    assert db_path.exists()


def test_maintenance_flush_waits_for_atomic_assertion_publication(
    assertion_db, monkeypatch
):
    _db_path, messages, assertions = assertion_db
    content = "I prefer tea."
    store_id = messages.append(
        "session-a", {"role": "user", "content": content}
    )
    snapshot = assertions.snapshot_source(store_id)
    candidate = _candidate(content, "tea")
    candidate_id = assertions.assertion_id_for(snapshot, candidate)
    invalid_relation = AssertionRelationCandidate(
        source_span_start=content.index("tea"),
        source_span_end=content.index("tea") + len("tea"),
        from_assertion_id=candidate_id,
        relation_type="confirms",
        to_assertion_id="f" * 64,
    )
    transaction_started = threading.Event()
    release_publication = threading.Event()
    original_begin_write = assertions._begin_write

    def paused_begin_write():
        original_begin_write()
        transaction_started.set()
        assert release_publication.wait(timeout=5)

    monkeypatch.setattr(assertions, "_begin_write", paused_begin_write)
    engine = SimpleNamespace(
        _store=messages,
        _dag=SimpleNamespace(_conn=messages._conn),
        _lifecycle=None,
        _assertions=assertions,
    )
    publish_errors: list[BaseException] = []
    flush_errors: list[BaseException] = []
    flush_finished = threading.Event()

    def publish():
        try:
            assertions.publish_source(
                snapshot, [candidate], relations=[invalid_relation]
            )
        except BaseException as exc:  # noqa: BLE001 - thread result is asserted below
            publish_errors.append(exc)

    def flush():
        try:
            flush_engine_connections(engine)
        except BaseException as exc:  # noqa: BLE001 - thread result is asserted below
            flush_errors.append(exc)
        finally:
            flush_finished.set()

    publish_thread = threading.Thread(target=publish)
    publish_thread.start()
    assert transaction_started.wait(timeout=5)
    flush_thread = threading.Thread(target=flush)
    flush_thread.start()
    assert not flush_finished.wait(timeout=0.1)

    release_publication.set()
    publish_thread.join(timeout=5)
    flush_thread.join(timeout=5)

    assert not publish_thread.is_alive()
    assert not flush_thread.is_alive()
    assert len(publish_errors) == 1
    assert isinstance(publish_errors[0], ValueError)
    assert "missing or invalidated" in str(publish_errors[0])
    assert flush_errors == []
    assert assertions.connection.execute(
        "SELECT COUNT(*) FROM lcm_assertion_sources"
    ).fetchone()[0] == 0
    assert assertions.connection.execute(
        "SELECT COUNT(*) FROM lcm_assertions"
    ).fetchone()[0] == 0
    assert assertions.connection.execute(
        "SELECT COUNT(*) FROM lcm_assertion_relations"
    ).fetchone()[0] == 0
